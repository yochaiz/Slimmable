from abc import abstractmethod
from time import time
from os import makedirs
from os.path import exists

from torch import tensor, no_grad
from torch.nn import CrossEntropyLoss
from torch.optim.sgd import SGD
from torch.optim.lr_scheduler import ReduceLROnPlateau

from utils.training import TrainingStats
from utils.HtmlLogger import HtmlLogger


class TrainWeights:
    # init tables keys
    trainLossKey = 'Training loss'
    trainAccKey = 'Training acc'
    validLossKey = 'Validation loss'
    validAccKey = 'Validation acc'
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
    }

    # init tables columns
    colsTrainWeights = [batchNumKey, trainLossKey, trainAccKey, timeKey]
    colsValidation = [batchNumKey, validLossKey, validAccKey, timeKey]

    def __init__(self, args, model, modelParallel, train_queue, valid_queue):
        self.args = args
        # save models
        self.model = model
        self.modelParallel = modelParallel

        # save datasets
        self.train_queue = train_queue
        self.valid_queue = valid_queue

        # init cross entropy loss
        self.cross_entropy = CrossEntropyLoss().cuda()

        self.trainFolderPath = '{}/{}'.format(args.save, args.trainFolder)

    # apply defined format functions on dict values by keys
    def _applyFormats(self, dict):
        for k in dict.keys():
            if k in self.formats:
                dict[k] = self.formats[k](dict[k])

    @abstractmethod
    def stopCondition(self):
        raise NotImplementedError('subclasses must override stopCondition()!')

    @abstractmethod
    def schedulerMetric(self, validLoss):
        raise NotImplementedError('subclasses must override schedulerMetric()!')

    @abstractmethod
    def postEpoch(self, epoch, optimizer, trainData, validData, validAcc, validLoss):
        raise NotImplementedError('subclasses must override postEpoch()!')

    @abstractmethod
    def postTrain(self):
        raise NotImplementedError('subclasses must override postTrain()!')

    # performs single epoch of model weights training
    def weightsEpoch(self, optimizer, epoch, loggers):
        print('*** weightsEpoch() ***')
        model = self.model
        modelParallel = self.modelParallel
        crit = self.cross_entropy
        train_queue = self.train_queue
        trainStats = TrainingStats(model.baselineWidthKeys())

        trainLogger = loggers.get('train')
        if trainLogger:
            trainLogger.createDataTable('Epoch:[{}] - Training weights'.format(epoch), self.colsTrainWeights)

        nBatches = len(train_queue)

        modelParallel.train()
        assert (model.training is True)

        for step, (input, target) in enumerate(train_queue):
            startTime = time()

            input = tensor(input, requires_grad=False).cuda()
            target = tensor(target, requires_grad=False).cuda(async=True)

            # optimize model weights
            optimizer.zero_grad()
            # iterate & forward widths
            for widthRatio, idxList in model.baselineWidth():
                # set model layers current width index
                model.setCurrWidthIdx(idxList)
                # forward
                logits = modelParallel(input)
                # calc loss
                loss = crit(logits, target)
                # back propagate
                loss.backward()
                # update training stats
                trainStats.update(widthRatio, logits, target, loss)
            # update weights
            optimizer.step()

            endTime = time()

            if trainLogger:
                dataRow = {
                    self.batchNumKey: '{}/{}'.format(step, nBatches), self.timeKey: (endTime - startTime),
                    self.trainLossKey: trainStats.batchLoss(), self.trainAccKey: trainStats.prec1()
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
        summaryData = {self.trainLossKey: epochLossDict, self.trainAccKey: epochAccDict, self.batchNumKey: 'Summary'}
        # apply formats
        self._applyFormats(summaryData)

        for logger in loggers.values():
            if logger:
                logger.addSummaryDataRow(summaryData)

        # log forward counters. if loggerFuncs==[] then it is just resets counters
        func = [lambda rows: trainLogger.addInfoTable(title='{} - Training'.format(self.forwardCountersKey), rows=rows)] if trainLogger else []
        model.logForwardCounters(loggerFuncs=func)

        return summaryData

    # performs single epoch of model inference
    def inferEpoch(self, nEpoch, loggers):
        print('*** inferEpoch() ***')
        model = self.model
        modelParallel = self.modelParallel
        valid_queue = self.valid_queue
        crit = self.cross_entropy
        trainStats = TrainingStats(model.baselineWidthKeys())

        trainLogger = loggers.get('train')
        if trainLogger:
            trainLogger.createDataTable('Epoch:[{}] - Validation'.format(nEpoch), self.colsValidation)

        nBatches = len(valid_queue)

        modelParallel.eval()
        assert (model.training is False)

        with no_grad():
            for step, (input, target) in enumerate(valid_queue):
                startTime = time()

                input = tensor(input).cuda()
                target = tensor(target).cuda(async=True)

                # iterate & forward widths
                for widthRatio, idxList in model.baselineWidth():
                    # set model layers current width index
                    model.setCurrWidthIdx(idxList)
                    # forward
                    logits = modelParallel(input)
                    # calc loss
                    loss = crit(logits, target)
                    # update training stats
                    trainStats.update(widthRatio, logits, target, loss)

                endTime = time()

                if trainLogger:
                    dataRow = {
                        self.batchNumKey: '{}/{}'.format(step, nBatches), self.validLossKey: trainStats.batchLoss(),
                        self.validAccKey: trainStats.prec1(), self.timeKey: endTime - startTime
                    }
                    # apply formats
                    self._applyFormats(dataRow)
                    # add row to data table
                    trainLogger.addDataRow(dataRow)

        # create summary row
        validAcc = trainStats.top1()
        validLoss = trainStats.epochLoss()
        summaryRow = {self.batchNumKey: 'Summary', self.validLossKey: validLoss, self.validAccKey: validAcc}
        # apply formats
        self._applyFormats(summaryRow)

        for logger in loggers.values():
            if logger:
                logger.addSummaryDataRow(summaryRow)

        # log forward counters. if loggerFuncs==[] then it is just resets counters
        func = [lambda rows: trainLogger.addInfoTable(title='{} - Validation'.format(self.forwardCountersKey), rows=rows)] if trainLogger else []
        model.logForwardCounters(loggerFuncs=func)

        return validAcc, validLoss, summaryRow

    def train(self, trainFolderName):
        modelParallel = self.modelParallel
        args = self.args

        # create train folder
        folderPath = '{}/{}'.format(self.trainFolderPath, trainFolderName)
        if not exists(folderPath):
            makedirs(folderPath)

        # init optimizer
        optimizer = SGD(modelParallel.parameters(), args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)
        # init scheduler
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.95, patience=2, min_lr=args.learning_rate_min)

        epoch = 0
        trainLoggerFlag = True

        while not self.stopCondition():
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
            loggersDict = dict(train=trainLogger)

            print('========== Epoch:[{}] =============='.format(epoch))
            # train
            trainData = self.weightsEpoch(optimizer, epoch, loggersDict)
            # validation
            validAcc, validLoss, validData = self.inferEpoch(epoch, loggersDict)

            # update scheduler
            scheduler.step(self.schedulerMetric(validLoss))

            self.postEpoch(epoch, optimizer, trainData, validData, validAcc, validLoss)

        self.postTrain()
