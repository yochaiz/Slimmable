from os import makedirs
from time import time
from scipy.stats import entropy
from argparse import Namespace

from torch import tensor, zeros
from torch import save as saveCheckpoint
from torch.optim.sgd import SGD
from torch.optim.lr_scheduler import ReduceLROnPlateau

from .regime import TrainRegime
from .PreTrainedRegime import PreTrainedTrainWeights, EpochData
from models.BaseNet import BaseNet
from utils.flopsLoss import FlopsLoss
from utils.HtmlLogger import HtmlLogger
from utils.trainWeights import TrainWeights
from utils.replicator import ModelReplicator
from utils.checkpoint import save_checkpoint
from utils.training import AlphaTrainingStats


class EpochTrainWeights(PreTrainedTrainWeights):
    _nRoundDigits = AlphaTrainingStats.nRoundDigits

    def __init__(self, regime, maxEpoch, currEpoch, logger):
        self.logger = logger
        self.tableTitle = 'Train model weights - Epoch:[{}]'.format(currEpoch)

        super(EpochTrainWeights, self).__init__(regime, maxEpoch)

        # init average dictionary
        self._avgDict = None
        # init (key, data) mapping
        self._map = {self.trainLossKey: lambda trainData, validData: trainData.lossDict(),
                     self.trainAccKey: lambda trainData, validData: trainData.accDict(),
                     self.validLossKey: lambda trainData, validData: validData.lossDict(),
                     self.validAccKey: lambda trainData, validData: validData.accDict()
                     }
        # init train sums ((train, validation) x (loss, acc)) dictionary
        self._sumDict = {k: {} for k in self._map.keys()}
        # init epoch average update function
        self._epochAvgUpdateFunc = self._initEpochAvgUpdate

    def getLogger(self):
        return self.logger

    def _initEpochAvgUpdate(self, trainData: EpochData, validData: EpochData):
        for k, mapFunc in self._map.items():
            self._sumDict[k] = mapFunc(trainData, validData)

    def _standardEpochAvgUpdate(self, trainData: EpochData, validData: EpochData):
        for k, mapFunc in self._map.items():
            data = mapFunc(trainData, validData)
            for n, value in data.items():
                self._sumDict[k][n] += value

    def postEpoch(self, epoch, optimizer, trainData: EpochData, validData: EpochData):
        # perform base class postEpoch()
        super(EpochTrainWeights, self).postEpoch(epoch, optimizer, trainData, validData)

        # update averages
        self._epochAvgUpdateFunc(trainData, validData)
        # set epoch average update function to standard function
        self._epochAvgUpdateFunc = self._standardEpochAvgUpdate

    def postTrain(self):
        avgDict = {}
        # average sums
        for k, data in self._sumDict.items():
            avgDict[k] = {}
            for n, sum in data.items():
                avgDict[k][n] = round(sum / self.maxEpoch, self._nRoundDigits)

        # add epoch title
        avgDict[self.epochNumKey] = 'Average'
        # sort values formats
        self._applyFormats(avgDict)
        # add summary row
        self.logger.addSummaryDataRow(avgDict)
        # set average dict
        self._avgDict = avgDict

        # perform base class postTrain()
        super(EpochTrainWeights, self).postTrain()

    def avgDictDataRow(self):
        return self._avgDict


class SearchRegime(TrainRegime):
    # init train logger key
    trainLoggerKey = TrainWeights.trainLoggerKey
    summaryKey = TrainWeights.summaryKey
    # init table columns names
    archLossKey = 'Arch Loss'
    pathsListKey = 'Paths list'
    # get keys from TrainWeights
    batchNumKey = TrainWeights.batchNumKey
    epochNumKey = TrainWeights.epochNumKey
    forwardCountersKey = TrainWeights.forwardCountersKey
    timeKey = TrainWeights.timeKey
    trainLossKey = TrainWeights.trainLossKey
    trainAccKey = TrainWeights.trainAccKey
    validLossKey = TrainWeights.validLossKey
    validAccKey = TrainWeights.validAccKey
    widthKey = TrainWeights.widthKey
    lrKey = TrainWeights.lrKey
    validFlopsRatioKey = TrainWeights.validFlopsRatioKey

    # init table columns
    k = 2
    alphasTableTitle = 'Alphas (top [{}])'.format(k)
    # init table columns names
    colsTrainAlphas = [batchNumKey, archLossKey, alphasTableTitle, pathsListKey, timeKey]
    colsMainLogger = [epochNumKey, archLossKey, trainLossKey, trainAccKey, validLossKey, validAccKey, validFlopsRatioKey, widthKey, lrKey]

    # init statistics (plots) keys template
    lossAvgTemplate = '{}_Loss_Avg'
    lossVarianceTemplate = '{}_Loss_Variance'
    # init statistics (plots) keys
    entropyKey = 'Alphas_Entropy'
    alphaDistributionKey = 'Alphas_Distribution'

    # init formats for keys
    formats = {
        archLossKey: lambda x: HtmlLogger.dictToRows(x, nElementPerRow=1)
    }

    def __init__(self, args, logger):
        self.lossClass = FlopsLoss
        super(SearchRegime, self).__init__(args, logger)

        # init number of epochs
        self.nEpochs = 100
        # init main table
        logger.createDataTable('Search summary', self.colsMainLogger)
        # update max table cell length
        logger.setMaxTableCellLength(30)

        # add TrainWeights formats to self.formats
        self.formats.update(TrainWeights.getFormats())
        # update epoch key format
        self.formats[self.epochNumKey] = lambda x: '{}/{}'.format(x, self.nEpochs)

        # init flops loss
        self.flopsLoss = FlopsLoss(args, getattr(args, self.model.baselineFlopsKey()))
        self.flopsLoss = self.flopsLoss.cuda()

        # load model pre-trained weights
        TrainWeights.loadPreTrained(self.model, args.pre_trained, self.logger)
        # reset args.pre_trained, we don't want to load these weights anymore
        args.pre_trained = None
        # init model replications
        self.replicator = ModelReplicator(self)

        # create folder for jobs checkpoints
        self.jobsPath = '{}/jobs'.format(args.save)
        makedirs(self.jobsPath)
        # init data table row keys to replace
        self.rowKeysToReplace = [self.validLossKey, self.validAccKey]

        # init email time
        self.lastMailTime = time()
        self.secondsBetweenMails = 1 * 3600

    def buildStatsContainers(self):
        model = self.model
        lossClass = self.lossClass

        container = {self.alphaDistributionKey: self._containerPerAlpha(model),
                     self.entropyKey: [{layerIdx: [] for layerIdx in range(len(model.layersList()))}]}
        # add loss average keys
        for k in lossClass.lossKeys():
            container[self.lossAvgTemplate.format(k)] = self._containerPerAlpha(model)
        # add loss variance keys
        container[self.lossVarianceTemplate.format(lossClass.totalKey())] = self._containerPerAlpha(model)

        return container

    def buildStatsRules(self):
        return {self.alphaDistributionKey: 1.1}

    # apply defined format functions on dict values by keys
    def _applyFormats(self, dict):
        for k in dict.keys():
            if k in self.formats:
                dict[k] = self.formats[k](dict[k])

    def trainAlphas(self, search_queue, optimizer, epoch, loggers):
        print('*** trainAlphas() ***')
        model = self.model
        replicator = self.replicator
        nSamples = self.args.nSamplesPerAlpha
        # init trainingStats instance
        trainStats = AlphaTrainingStats(self.flopsLoss.lossKeys(), useAvg=False)
        # init new epoch in replications
        replicator.initNewEpoch()

        trainLogger = loggers.get(self.trainLoggerKey)
        if trainLogger:
            trainLogger.createDataTable('Epoch:[{}] - Alphas'.format(epoch), self.colsTrainAlphas)

        def createInfoTable(dict, key, logger, rows):
            dict[key] = logger.createInfoTable('Show', rows)

        def createAlphasTable(k, rows):
            createInfoTable(dataRow, self.alphasTableTitle, trainLogger, rows)

        nBatches = len(search_queue)
        # update batch num key format
        self.formats[self.batchNumKey] = lambda x: '{}/{}'.format(batchNum, nBatches)

        for batchNum, (input, target) in enumerate(search_queue):
            startTime = time()

            input = input.cuda().clone().detach().requires_grad_(False)
            target = target.cuda(async=True).clone().detach().requires_grad_(False)

            # reset optimizer gradients
            optimizer.zero_grad()
            # evaluate on samples and calc alphas gradients
            lossAvgDict, pathsList = self._loss(input, target, nSamples)
            # perform optimizer step
            optimizer.step()

            endTime = time()

            # update training stats
            for lossName, loss in lossAvgDict.items():
                trainStats.update(lossName, input, loss)
            # save alphas to csv
            model.saveAlphasCsv(data=[epoch, batchNum])
            # update alphas distribution statistics (after optimizer step)
            self._calcAlphasDistribStats(model)
            # update statistics plots
            self.statistics.plotData()
            # save checkpoint
            save_checkpoint(self.trainFolderPath, model, optimizer, lossAvgDict)

            if trainLogger:
                # add numbering to paths list
                pathsList = [['#', 'Path']] + [[idx + 1, v] for idx, v in enumerate(pathsList)]
                # init data row
                dataRow = {self.batchNumKey: batchNum, self.archLossKey: trainStats.batchLoss(),
                           self.pathsListKey: trainLogger.createInfoTable('Show', pathsList), self.timeKey: endTime - startTime}
                # add alphas distribution table
                model.logTopAlphas(self.k, [createAlphasTable])
                # apply formats
                self._applyFormats(dataRow)
                # add row to data table
                trainLogger.addDataRow(dataRow)

        epochLossDict = trainStats.epochLoss()
        # log summary row
        summaryDataRow = {self.batchNumKey: self.summaryKey, self.archLossKey: epochLossDict}
        # delete batch num key format
        del self.formats[self.batchNumKey]
        # apply formats
        self._applyFormats(summaryDataRow)
        # add row to data table
        trainLogger.addSummaryDataRow(summaryDataRow)

        return epochLossDict, summaryDataRow

    def _calcAlphasDistribStats(self, model):
        stats = self.statistics
        for layerIdx, layer in enumerate(model.layersList()):
            # calc layer alphas distribution
            probs = layer.probs().cpu()
            # add entropy to statistics
            stats.addValue(lambda containers: containers[self.entropyKey][0][layerIdx], entropy(probs))
            # add alphas distribution
            for alphaIdx, p in enumerate(probs):
                alphaTitle = self._alphaPlotTitle(layer, alphaIdx)
                stats.addValue(lambda containers: containers[self.alphaDistributionKey][layerIdx][alphaTitle], p.item())

    def _loss(self, input: tensor, target: tensor, nSamples: int) -> (dict, list):
        # calc paths loss for each alpha using replicator
        lossDictsList, pathsList = self.replicator.loss(input, target, nSamples)
        # update statistics and alphas gradients based on loss
        lossAvgDict = self._updateAlphasGradients(lossDictsList)

        return lossAvgDict, pathsList

    # updates alphas gradients
    # updates statistics
    # def _updateAlphasGradients(self, lossValues, ceLossValues, flopsLossValues):
    def _updateAlphasGradients(self, lossDictsList: list) -> dict:
        model = self.model
        totalKey = self.flopsLoss.totalKey()

        # init total loss
        totalLoss = 0.0
        # init losses averages
        lossAvgDict = {k: 0.0 for k in self.flopsLoss.lossKeys()}
        # count how many alphas we have sum their loss average
        nAlphas = 0
        # get statistics element with a shorter name
        stats = self.statistics
        # init model probs list for gradient calcs
        probsList = []
        # after we finished iterating over samples, we can calculate loss average & variance for each alpha
        for layerIdx, layer in enumerate(model.layersList()):
            layerLossDicts = lossDictsList[layerIdx]
            # get layer alphas probabilities
            layerProbs = layer.probs()
            # add to model probs list
            probsList.append(layerProbs)
            # init layer alphas gradient vector
            layerAlphasGrad = zeros(layer.nWidths(), requires_grad=True).cuda()
            # iterate over alphas
            for idx, alphaLossDict in enumerate(layerLossDicts):
                alphaLossAvgDict = {}
                for k, lossList in alphaLossDict.items():
                    # calc loss list average
                    alphaLossAvgDict[k] = sum(lossList) / len(lossList)
                    # add loss average to total loss average
                    lossAvgDict[k] += alphaLossAvgDict[k]
                # update number of alphas summed into lossAvgDict
                nAlphas += 1
                # set total loss average
                alphaLossAvg = alphaLossAvgDict[totalKey]
                # update alpha gradient
                layerAlphasGrad[idx] = alphaLossAvg
                # update total loss
                totalLoss += (alphaLossAvg * layerProbs[idx])
                # calc alpha loss variance
                alphaLossVariance = [((x - alphaLossAvg) ** 2) for x in alphaLossDict[totalKey]]
                alphaLossVariance = sum(alphaLossVariance) / (len(alphaLossVariance) - 1)
                # add values to statistics
                alphaTitle = self._alphaPlotTitle(layer, idx)
                # init template for get list function based on container key
                getListFunc = lambda key: lambda containers: containers[key][layerIdx][alphaTitle]
                # add loss average values to statistics
                for lossKey, lossAvg in alphaLossAvgDict.items():
                    stats.addValue(getListFunc(self.lossAvgTemplate.format(lossKey)), lossAvg)
                # add loss variance values to statistics
                stats.addValue(getListFunc(self.lossVarianceTemplate.format(totalKey)), alphaLossVariance)

            # update layer alphas gradient
            layerAlphas = layer.alphas()
            layerAlphas.grad = layerAlphasGrad

        # average total loss
        totalLoss /= len(model.layersList())
        print('totalLoss:[{:.5f}]'.format(totalLoss))
        # subtract average total loss from every alpha gradient
        for layerIdx, (layer, layerProbs) in enumerate(zip(model.layersList(), probsList)):
            layerAlphas = layer.alphas()
            layerAlphas.grad -= totalLoss
            # multiply each grad by its probability
            layerAlphas.grad *= layerProbs
            print('Layer:[{}] - alphas gradient:{}'.format(layerIdx, layerAlphas.grad))

        # average (total loss average) by number of alphas
        for k in lossAvgDict.keys():
            lossAvgDict[k] /= nAlphas

        return lossAvgDict

    @staticmethod
    def _getEpochRange(nEpochs: int) -> range:
        return range(1, nEpochs + 1)

    @staticmethod
    def _generateTableValue(jobName, key) -> str:
        return {BaseNet.partitionKey(): '{}_{}'.format(jobName, key)}

    def _createPartitionInfoTable(self, partition):
        rows = [['Layer #', self.widthKey]] + [[layerIdx, w] for layerIdx, w in enumerate(partition)]
        table = self.logger.createInfoTable('Show', rows)
        return table

    def _createJob(self, epoch: int, id: int, choosePathFunc: callable) -> dict:
        model = self.model
        args = self.args
        # clone args
        job = Namespace(**vars(args))
        # init job name
        jobName = '[{}]-[{}]-[{}]'.format(args.time, epoch, id)
        # create job data row
        dataRow = {k: self._generateTableValue(jobName, k) for k in self.rowKeysToReplace}
        # sample path from alphas distribution
        choosePathFunc()
        # set attributes
        job.partition = model.currWidthRatio()
        job.epoch = epoch
        job.id = id
        job.jobName = jobName
        job.tableKeys = dataRow
        # save job
        jobPath = '{}/{}.pth.tar'.format(self.jobsPath, job.jobName)
        saveCheckpoint(job, jobPath)

        # add flops ratio to data row
        dataRow[self.validFlopsRatioKey] = model.flopsRatio()
        # add path width ratio to data row
        dataRow[self.widthKey] = self._createPartitionInfoTable(job.partition)
        # add epoch number to data row
        dataRow[self.epochNumKey] = epoch
        # apply formats
        self._applyFormats(dataRow)

        return dataRow

    def _createEpochJobs(self, epoch: int) -> list:
        # init epoch data rows list
        epochDataRows = []
        # init model path chooser function
        choosePathFunc = self.model.chooseAlphaMax
        for id in self._getEpochRange(self.args.nJobs):
            jobDataRow = self._createJob(epoch, id, choosePathFunc)
            epochDataRows.append(jobDataRow)
            # only 1st job should be based on alphas max, the rest should sample from alphas distribution
            choosePathFunc = self.model.choosePathByAlphas

        return epochDataRows

    def train(self):
        args = self.args
        model = self.model
        logger = self.logger
        epochRange = self._getEpochRange(self.nEpochs)

        # init optimizer
        optimizer = SGD(model.alphas(), args.search_learning_rate, momentum=args.search_momentum, weight_decay=args.search_weight_decay)
        # init scheduler
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.95, patience=2, min_lr=args.search_learning_rate_min)

        for epoch in epochRange:
            print('========== Epoch:[{}] =============='.format(epoch))
            # # calc alpha trainset loss on baselines
            # self.calcAlphaTrainsetLossOnBaselines(self.trainFolderPath, '{}_{}'.format(epoch, self.archLossKey), logger)

            # init epoch train logger
            trainLogger = HtmlLogger(self.trainFolderPath, epoch)
            # set loggers dictionary
            loggersDict = {self.trainLoggerKey: trainLogger}

            # train alphas
            epochLossDict, alphasDataRow = self.trainAlphas(self.search_queue[epoch % args.alphas_data_parts], optimizer, epoch, loggersDict)
            # update scheduler
            scheduler.step(epochLossDict.get(self.flopsLoss.totalKey()))

            # add values to alphas data row
            additionalData = {self.epochNumKey: epoch, self.lrKey: optimizer.param_groups[0]['lr']}
            self._applyFormats(additionalData)
            alphasDataRow.update(additionalData)
            logger.addDataRow(alphasDataRow)

            # create epoch jobs
            epochDataRows = self._createEpochJobs(epoch)

            # train weights
            wEpochName = '{}_w'.format(epoch)
            weightsLogger = HtmlLogger(self.trainFolderPath, wEpochName)
            trainWeights = EpochTrainWeights(self, 100, epoch, weightsLogger)
            trainWeights.train(wEpochName)
            # add data row
            trainDataRow = trainWeights.avgDictDataRow()
            trainDataRow[self.epochNumKey] = self.formats[self.epochNumKey](epoch)
            logger.addDataRow(trainDataRow)

            # add epoch data rows
            for jobDataRow in epochDataRows:
                logger.addDataRow(jobDataRow, trType='<tr bgcolor="#2CBDD6">')

            # save checkpoint
            save_checkpoint(self.trainFolderPath, model, optimizer, {})

# class TrainPathWeights(TrainWeights):
#     def __init__(self, regime):
#         super(TrainPathWeights, self).__init__(regime)
#
#         # disable args.pre_trained to avoid loading model weights
#         self.getArgs().pre_trained = None
#         # disable optimizer state_dict to avoid loading optimizer state
#         self.optimizerStateDict = None
#         # save model weights
#         self.modelOrgWeights = self.getModel().state_dict()
#         # init layer paths dictionary
#         self.layerPaths = {}
#
#     def stopCondition(self, epoch):
#         return epoch >= 5
#
#     def widthList(self):
#         return self.layerPaths.items()
#
#     def restoreModelOriginalWeights(self):
#         self.getModel().load_state_dict(self.modelOrgWeights)
#
#     def train(self, layer):
#         model = self.getModel()
#         layerPaths = self.layerPaths
#         # restore model original weights
#         self.restoreModelOriginalWeights()
#         # reset layer paths for training
#         layerPaths.clear()
#         # collect layer paths for training
#         for idx in range(layer.nWidths()):
#             # set path to go through width[idx] in current layer
#             layer.setCurrWidthIdx(idx)
#             # add path to dictionary
#             layerPaths[layer.widthRatioByIdx(idx)] = model.currWidthIdx()
#
#         # init optimizer
#         optimizer = self._initOptimizer()
#         # train
#         epoch = 0
#         while not self.stopCondition(epoch):
#             epoch += 1
#             self.weightsEpoch(optimizer, epoch, {})

# ======= compare replicator vs. standard model, who is FASTER ===========
# init model constructor func
# modelConstructor = lambda: self.buildModel(args)
# replicator = ModelReplicator(modelConstructor, args.gpu, self.logger)

# # calc replicator time
# print('Running replicator')
# startTime = time()
# replicator.run()
# endTime = time()
# print('Replicator time:[{}]'.format(endTime - startTime))
#
# # calc standard model time
# data = randn(250, 3, 32, 32).cuda()
# print('Running standard model')
# startTime = time()
# for _ in range(5000):
#     self.model(data)
#     data[0, 0, 0, 0] += 0.001
# endTime = time()
# print('Standard model time:[{}]'.format(endTime - startTime))
