from time import time
from scipy.stats import entropy

from torch import tensor, zeros
from torch.optim.sgd import SGD
from torch.optim.lr_scheduler import ReduceLROnPlateau

from .regime import TrainRegime

from utils.flopsLoss import FlopsLoss
from utils.HtmlLogger import HtmlLogger
from utils.trainWeights import TrainWeights
from utils.training import AlphaTrainingStats


class TrainPathWeights(TrainWeights):
    def __init__(self, regime):
        super(TrainPathWeights, self).__init__(regime)

        # disable args.pre_trained to avoid loading model weights
        self.getArgs().pre_trained = None
        # disable optimizer state_dict to avoid loading optimizer state
        self.optimizerStateDict = None
        # save model weights
        self.modelOrgWeights = self.getModel().state_dict()
        # init layer paths dictionary
        self.layerPaths = {}

    def stopCondition(self, epoch):
        return epoch >= 5

    def widthList(self):
        return self.layerPaths.items()

    def restoreModelOriginalWeights(self):
        self.getModel().load_state_dict(self.modelOrgWeights)

    def train(self, layer):
        model = self.getModel()
        layerPaths = self.layerPaths
        # restore model original weights
        self.restoreModelOriginalWeights()
        # reset layer paths for training
        layerPaths.clear()
        # collect layer paths for training
        for idx in range(layer.nWidths()):
            # set path to go through width[idx] in current layer
            layer.setCurrWidthIdx(idx)
            # add path to dictionary
            layerPaths[layer.widthRatioByIdx(idx)] = model.currWidthIdx()

        # init optimizer
        optimizer = self._initOptimizer()
        # train
        epoch = 0
        while not self.stopCondition(epoch):
            epoch += 1
            self.weightsEpoch(optimizer, epoch, {})


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

        # add TrainWeights formats to self.formats
        self.formats.update(TrainWeights.getFormats())

        # init flops loss
        self.flopsLoss = FlopsLoss(args, getattr(args, self.model.baselineFlopsKey()))
        self.flopsLoss = self.flopsLoss.cuda()

        # init main table
        logger.createDataTable('Search summary', self.colsMainLogger)

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

    # apply defined format functions on dict values by keys
    def _applyFormats(self, dict):
        for k in dict.keys():
            if k in self.formats:
                dict[k] = self.formats[k](dict[k])

    def trainAlphas(self, search_queue, optimizer, epoch, loggers):
        print('*** trainAlphas() ***')
        model = self.model
        # init trainingStats instance
        trainStats = AlphaTrainingStats(self.flopsLoss.lossKeys(), useAvg=False)
        # init TrainWeights instance
        trainWeights = TrainPathWeights(self)

        trainLogger = loggers.get(self.trainLoggerKey)
        if trainLogger:
            trainLogger.createDataTable('Epoch:[{}] - Alphas'.format(epoch), self.colsTrainAlphas)

        def createInfoTable(dict, key, logger, rows):
            dict[key] = logger.createInfoTable('Show', rows)

        def createAlphasTable(k, rows):
            createInfoTable(dataRow, self.alphasTableTitle, trainLogger, rows)

        def createForwardCountersTable(rows):
            createInfoTable(dataRow, self.forwardCountersKey, trainLogger, rows)

        nBatches = len(search_queue)
        for batchNum, (input, target) in enumerate(search_queue):
            startTime = time()

            input = tensor(input, requires_grad=False).cuda()
            target = tensor(target, requires_grad=False).cuda(async=True)

            # reset optimizer gradients
            optimizer.zero_grad()
            # evaluate on samples and calc alphas gradients
            lossAvgDict, pathsList = self._loss(input, target, self._samplesSamePath, trainWeights)
            # perform optimizer step
            optimizer.step()

            endTime = time()

            # update training stats
            for lossName, loss in lossAvgDict.items():
                trainStats.update(lossName, input, loss)
            # update alphas distribution statistics (after optimizer step)
            self._calcAlphasDistribStats(model)
            # update statistics plots
            self.statistics.plotData()

            if trainLogger:
                dataRow = {self.batchNumKey: '{}/{}'.format(batchNum, nBatches), self.archLossKey: trainStats.batchLoss(),
                           self.pathsListKey: trainLogger.createInfoTable('Show', pathsList), self.timeKey: endTime - startTime}
                model.logTopAlphas(self.k, [createAlphasTable])
                model.logForwardCounters([])
                # apply formats
                self._applyFormats(dataRow)
                # add row to data table
                trainLogger.addDataRow(dataRow)

            break

        epochLossDict = trainStats.epochLoss()
        # log summary row
        summaryDataRow = {self.batchNumKey: self.summaryKey, self.archLossKey: epochLossDict}
        # apply formats
        self._applyFormats(summaryDataRow)
        # add row to data table
        trainLogger.addDataRow(summaryDataRow)

        # restore model original weights
        trainWeights.restoreModelOriginalWeights()

        return epochLossDict

    def _calcAlphasDistribStats(self, model):
        stats = self.statistics
        for layerIdx, layer in enumerate(model.layersList()):
            # calc layer alphas distribution
            probs = layer.probs().detach()
            # add entropy to statistics
            stats.addValue(lambda containers: containers[self.entropyKey][0][layerIdx], entropy(probs))
            # add alphas distribution
            for alphaIdx, p in enumerate(probs):
                alphaTitle = self._alphaPlotTitle(layer, alphaIdx)
                stats.addValue(lambda containers: containers[self.alphaDistributionKey][layerIdx][alphaTitle], p.item())

    # given paths history dictionary and current path, checks if current path exists in history dictionary
    def _doesPathExist(self, pathsHistoryDict, currPath):
        currDict = pathsHistoryDict
        # init boolean flag
        pathExists = True
        for v in currPath:
            if v not in currDict:
                pathExists = False
                currDict[v] = {}
            # update current dict, we move to next layer
            currDict = currDict[v]

        return pathExists

    # select paths for each alpha and calc loss
    # add statistics
    # update alphas gradients
    def _loss(self, input, target, pathSelectionFunc, trainWeights):
        model = self.model

        # switch to inference mode
        model.eval()
        # calc paths loss for each alpha
        lossDictsList = pathSelectionFunc(input, target, trainWeights)
        # switch to train mode
        model.train()
        # update statistics and alphas gradients based on loss
        lossAvgDict = self._updateAlphasGradients(lossDictsList)

        return lossAvgDict

    # evaluate alphas on same paths
    def _samplesSamePath(self, input, target, trainWeights):
        model = self.model
        modelParallel = self.modelParallel
        nSamples = self.args.nSamplesPerAlpha
        # init samples (paths) history, to make sure we don't select the same sample twice
        pathsHistoryDict = {}
        # init samples (paths) history list for logging purposes
        pathsList = []

        # init containers to save loss values and variance
        lossDictsList = [[{k: [] for k in self.flopsLoss.lossKeys()} for _ in range(layer.nWidths())] for layer in model.layersList()]

        # iterate over samples. generate a sample (path), train it and evaluate alphas on sample
        for sampleIdx in range(nSamples):
            print('===== Sample idx:[{}] ====='.format(sampleIdx))
            # select new path based on alphas distribution.
            # check that selected path hasn't been selected before
            pathExists = True
            while pathExists:
                # select path based on alphas distribution
                model.choosePathByAlphas()
                # get selected path indices
                pathWidthIdx = model.currWidthIdx()
                # check that selected path hasn't been selected before
                pathExists = self._doesPathExist(pathsHistoryDict, pathWidthIdx)
            # add path to paths list
            pathsList.append([layer.widthByIdx(p) for p, layer in zip(pathWidthIdx, model.layersList())])

            # iterate over layers. in each layer iterate over alphas
            for layerIdx, layer in enumerate(model.layersList()):
                print('=== Layer idx:[{}] ==='.format(layerIdx))
                # init containers to save loss values and variance
                layerLossDicts = lossDictsList[layerIdx]
                # save layer current width idx
                layerCurrWidthIdx = layer.currWidthIdx()
                # train model on layer path
                trainWeights.train(layer)
                # iterate over alphas and calc loss
                for idx in range(layer.nWidths()):
                    # set path to go through width[idx] in current layer
                    layer.setCurrWidthIdx(idx)
                    print(model.currWidthIdx())
                    # forward input in model selected path
                    logits = modelParallel(input)
                    # calc loss
                    lossDict = self.flopsLoss(logits, target, model.countFlops())
                    # add loss to container
                    alphaLossDict = layerLossDicts[idx]
                    for k, loss in lossDict.items():
                        alphaLossDict[k].append(loss.item())

                # restore layer current width idx
                layer.setCurrWidthIdx(layerCurrWidthIdx)

        return lossDictsList, pathsList

    # updates alphas gradients
    # update statistics
    # def _updateAlphasGradients(self, lossValues, ceLossValues, flopsLossValues):
    def _updateAlphasGradients(self, lossDictsList):
        model = self.model
        nSamples = self.args.nSamplesPerAlpha
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
                    alphaLossAvgDict[k] = sum(lossList) / nSamples
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
                alphaLossVariance = sum(alphaLossVariance) / (nSamples - 1)
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
        # subtract average total loss from every alpha gradient
        for layer, layerProbs in zip(model.layersList(), probsList):
            layerAlphas = layer.alphas()
            layerAlphas.grad -= totalLoss
            # multiply each grad by its probability
            layerAlphas.grad *= layerProbs

        # average (total loss average) by number of alphas
        for k in lossAvgDict.keys():
            lossAvgDict[k] /= nAlphas

        return lossAvgDict

    @staticmethod
    def __getEpochRange(nEpochs):
        return range(1, nEpochs + 1)

    def train(self):
        args = self.args
        model = self.model
        logger = self.logger
        # init number of epochs
        nEpochs = 40
        epochRange = self.__getEpochRange(nEpochs)

        # init optimizer
        optimizer = SGD(model.alphas(), args.search_learning_rate, momentum=args.search_momentum, weight_decay=args.search_weight_decay)
        # init scheduler
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.95, patience=2, min_lr=args.search_learning_rate_min)

        for epoch in epochRange:
            print('========== Epoch:[{}] =============='.format(epoch))
            # # calc alpha trainset loss on baselines
            # self.calcAlphaTrainsetLossOnBaselines(self.trainFolderPath, '{}_{}'.format(epoch, self.archLossKey), logger)

            # init main logger data row
            dataRow = {self.epochNumKey: '{}/{}'.format(epoch, nEpochs), self.lrKey: optimizer.param_groups[0]['lr']}
            # init epoch train logger
            trainLogger = HtmlLogger(self.trainFolderPath, epoch)
            # set loggers dictionary
            loggersDict = {self.trainLoggerKey: trainLogger}

            # train alphas
            epochLossDict = self.trainAlphas(self.search_queue[epoch % args.alphas_data_parts], optimizer, epoch, loggersDict)
            # update scheduler
            scheduler.step(epochLossDict.get(self.flopsLoss.totalKey()))

            # train weights

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
