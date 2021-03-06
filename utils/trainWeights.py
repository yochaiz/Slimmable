from abc import abstractmethod
from time import time
from os import makedirs
from os.path import exists

from torch import no_grad
from torch import load as loadModel
from torch.nn import CrossEntropyLoss
from torch.optim.sgd import SGD
from torch.optim.lr_scheduler import ReduceLROnPlateau

from models.BaseNet.BaseNet import BaseNet
from utils.training import TrainingStats
from utils.HtmlLogger import HtmlLogger


class EpochData:
    def __init__(self, lossDict, accDict, summaryDataRow):
        self._lossDict = lossDict
        self._accDict = accDict
        self._summaryDataRow = summaryDataRow

    def lossDict(self):
        return self._lossDict

    def accDict(self):
        return self._accDict

    def summaryDataRow(self):
        return self._summaryDataRow


class TrainWeights:
    # init train logger key
    trainLoggerKey = 'train'
    summaryKey = 'Summary'
    # init tables keys
    trainLossKey = 'Training loss'
    trainAccKey = 'Training acc'
    validLossKey = 'Validation loss'
    validAccKey = 'Validation acc'
    flopsRatioKey = 'Flops ratio'
    epochNumKey = 'Epoch #'
    batchNumKey = 'Batch #'
    timeKey = 'Time'
    lrKey = 'Optimizer lr'
    widthKey = 'Width'
    forwardCountersKey = 'Forward counters'

    # init formats for keys
    formats = {
        timeKey: lambda x: '{:.3f}'.format(x),
        lrKey: lambda x: '{:.8f}'.format(x),
        trainLossKey: lambda x: HtmlLogger.dictToRows(x, nElementPerRow=1),
        trainAccKey: lambda x: HtmlLogger.dictToRows(x, nElementPerRow=1),
        validLossKey: lambda x: HtmlLogger.dictToRows(x, nElementPerRow=1),
        validAccKey: lambda x: HtmlLogger.dictToRows(x, nElementPerRow=1),
        flopsRatioKey: lambda x: '{:.3f}'.format(x)
    }

    # init tables columns
    colsTrainWeights = [batchNumKey, trainLossKey, trainAccKey, timeKey]
    colsValidation = [batchNumKey, validLossKey, validAccKey, timeKey]

    # def __init__(self, regime):
    def __init__(self, getModel, getModelParallel, getArgs, getLogger, getTrainQueue, getValidQueue, getTrainFolderPath):
        # init functions
        self.getModel = getModel
        self.getModelParallel = getModelParallel
        self.getArgs = getArgs
        self.getLogger = getLogger
        self.getTrainQueue = getTrainQueue
        self.getValidQueue = getValidQueue
        self.getTrainFolderPath = getTrainFolderPath

        # self.regime = regime
        # init cross entropy loss
        self.cross_entropy = CrossEntropyLoss().cuda()

        # load pre-trained model & optimizer
        self.optimizerStateDict = self.loadPreTrained(self.getModel(), self.getArgs().pre_trained, self.getLogger())

    # apply defined format functions on dict values by keys
    def _applyFormats(self, dict):
        for k in dict.keys():
            if k in self.formats:
                dict[k] = self.formats[k](dict[k])

    @staticmethod
    def getFormats():
        return TrainWeights.formats

    @abstractmethod
    def stopCondition(self, epoch):
        raise NotImplementedError('subclasses must override stopCondition()!')

    @abstractmethod
    # returns (widthRatio, idxList) list or generator
    def widthList(self):
        raise NotImplementedError('subclasses must override widthList()!')

    @abstractmethod
    def schedulerMetric(self, validLoss):
        raise NotImplementedError('subclasses must override schedulerMetric()!')

    @abstractmethod
    def postEpoch(self, epoch, optimizer, trainData: EpochData, validData: EpochData):
        raise NotImplementedError('subclasses must override postEpoch()!')

    @abstractmethod
    def postTrain(self):
        raise NotImplementedError('subclasses must override postTrain()!')

    # generic epoch flow
    def _genericEpoch(self, forwardFunc, data_queue, loggers, lossKey, accKey, tableTitle, tableCols, forwardCountersTitle) -> EpochData:
        trainStats = TrainingStats([k for k, v in self.widthList()])

        trainLogger = loggers.get(self.trainLoggerKey)
        if trainLogger:
            trainLogger.createDataTable(tableTitle, tableCols)

        nBatches = len(data_queue)

        for batchNum, (input, target) in enumerate(data_queue):
            startTime = time()

            input = input.cuda().clone().detach().requires_grad_(False)
            target = target.cuda(async=True).clone().detach().requires_grad_(False)

            # do forward
            forwardFunc(input, target, trainStats)

            endTime = time()

            if trainLogger:
                dataRow = {
                    self.batchNumKey: '{}/{}'.format(batchNum, nBatches), self.timeKey: (endTime - startTime),
                    lossKey: trainStats.batchLoss(), accKey: trainStats.prec1()
                }
                # apply formats
                self._applyFormats(dataRow)
                # add row to data table
                trainLogger.addDataRow(dataRow)

        epochLossDict = trainStats.epochLoss()
        epochAccDict = trainStats.top1()
        # # add epoch data to statistics plots
        # self.statistics.addBatchData(epochLossDict, epochAccDict)
        # log accuracy, loss, etc.
        summaryData = {lossKey: epochLossDict, accKey: epochAccDict, self.batchNumKey: self.summaryKey}
        # apply formats
        self._applyFormats(summaryData)

        for logger in loggers.values():
            if logger:
                logger.addSummaryDataRow(summaryData)

        # log forward counters. if loggerFuncs==[] then it is just resets counters
        func = [lambda rows: trainLogger.addInfoTable(title=forwardCountersTitle, rows=rows)] if trainLogger else []
        self.getModel().logForwardCounters(loggerFuncs=func)

        return EpochData(epochLossDict, epochAccDict, summaryData)

    def _slimForward(self, input, target, trainStats):
        model = self.getModel()
        modelParallel = self.getModelParallel()
        crit = self.cross_entropy
        # init loss list
        lossList = []
        # iterate & forward widths
        for widthRatio, idxList in self.widthList():
            # set model layers current width index
            model.setCurrWidthIdx(idxList)
            # forward
            logits = modelParallel(input)
            # calc loss
            loss = crit(logits, target)
            # add to loss list
            lossList.append(loss)
            # update training stats
            trainStats.update(widthRatio, logits, target, loss)

        return lossList

    # performs single epoch of model weights training
    def weightsEpoch(self, optimizer, epoch, loggers) -> EpochData:
        # print('*** weightsEpoch() ***')
        model = self.getModel()
        modelParallel = self.getModelParallel()

        modelParallel.train()
        assert (model.training is True)

        def forwardFunc(input, target, trainStats):
            # optimize model weights
            optimizer.zero_grad()
            # forward
            lossList = self._slimForward(input, target, trainStats)
            # back propagate
            for loss in lossList:
                loss.backward()
            # update weights
            optimizer.step()

        tableTitle = 'Epoch:[{}] - Training weights'.format(epoch)
        forwardCountersTitle = '{} - Training'.format(self.forwardCountersKey)
        return self._genericEpoch(forwardFunc, self.getTrainQueue(), loggers, self.trainLossKey, self.trainAccKey, tableTitle, self.colsTrainWeights,
                                  forwardCountersTitle)

    # performs single epoch of model inference
    def inferEpoch(self, nEpoch, loggers) -> EpochData:
        print('*** inferEpoch() ***')
        model = self.getModel()
        modelParallel = self.getModelParallel()

        modelParallel.eval()
        assert (model.training is False)

        def forwardFunc(input, target, trainStats):
            with no_grad():
                self._slimForward(input, target, trainStats)

        tableTitle = 'Epoch:[{}] - Validation'.format(nEpoch)
        forwardCountersTitle = '{} - Validation'.format(self.forwardCountersKey)
        return self._genericEpoch(forwardFunc, self.getValidQueue(), loggers, self.validLossKey, self.validAccKey, tableTitle, self.colsValidation,
                                  forwardCountersTitle)

    def _initOptimizer(self):
        modelParallel = self.getModelParallel()
        args = self.getArgs()

        optimizer = SGD(modelParallel.parameters(), args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)
        # load optimizer pre-trained state dict if exists
        if self.optimizerStateDict:
            optimizer.load_state_dict(self.optimizerStateDict)

        return optimizer

    def train(self, trainFolderName='init_weights_train'):
        args = self.getArgs()

        # create train folder
        folderPath = '{}/{}'.format(self.getTrainFolderPath(), trainFolderName)
        if not exists(folderPath):
            makedirs(folderPath)

        # init optimizer
        optimizer = self._initOptimizer()
        # init scheduler
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.95, patience=args.weights_patience, min_lr=args.learning_rate_min)

        epoch = 0
        trainLoggerFlag = True

        while not self.stopCondition(epoch):
            # update epoch number
            epoch += 1
            # init train logger
            trainLogger = None
            if trainLoggerFlag:
                trainLogger = HtmlLogger(folderPath, epoch)
                trainLogger.addInfoTable('Learning rates', [['optimizer_lr', self.formats[self.lrKey](optimizer.param_groups[0]['lr'])]])

            # update train logger condition for next epoch
            trainLoggerFlag = ((epoch + 1) % args.logInterval) == 0

            # set loggers dictionary
            loggersDict = {self.trainLoggerKey: trainLogger}

            print('========== Epoch:[{}] =============='.format(epoch))
            # train
            trainData = self.weightsEpoch(optimizer, epoch, loggersDict)
            # validation
            validData = self.inferEpoch(epoch, loggersDict)

            # update scheduler
            scheduler.step(self.schedulerMetric(validData.lossDict()))

            self.postEpoch(epoch, optimizer, trainData, validData)

        self.postTrain()

    @staticmethod
    def loadPreTrained(model: BaseNet, path: str, logger: HtmlLogger) -> dict:
        optimizerStateDict = None

        if path is not None:
            if exists(path):
                # load checkpoint
                checkpoint = loadModel(path, map_location=lambda storage, loc: storage.cuda())
                # load weights
                model.loadPreTrained(checkpoint['state_dict'])
                # # load optimizer state dict
                # optimizerStateDict = checkpoint['optimizer']
                # add info rows about checkpoint
                loggerRows = []
                loggerRows.append(['Path', '{}'.format(path)])
                validationAccRows = [['Ratio', 'Accuracy']] + HtmlLogger.dictToRows(checkpoint['best_prec1'], nElementPerRow=1)
                loggerRows.append(['Validation accuracy', validationAccRows])
                # set optimizer table row
                optimizerRow = HtmlLogger.dictToRows(optimizerStateDict, nElementPerRow=3) if optimizerStateDict else optimizerStateDict
                loggerRows.append(['Optimizer', optimizerRow])
                logger.addInfoTable('Pre-trained model', loggerRows)
            else:
                raise ValueError('Failed to load pre-trained from [{}], path does not exists'.format(path))

        return optimizerStateDict

# def getModel(self):
#     return self.regime.model
#
# def getModelParallel(self):
#     return self.regime.modelParallel
#
# def getArgs(self):
#     return self.regime.args
#
# def getLogger(self):
#     return self.regime.logger
#
# def getTrainQueue(self):
#     return self.regime.train_queue
#
# def getValidQueue(self):
#     return self.regime.valid_queue
#
# def getTrainFolderPath(self):
#     return self.regime.trainFolderPath
