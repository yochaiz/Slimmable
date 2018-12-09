from abc import abstractmethod
from time import time
from os import makedirs
from os.path import exists

from torch import tensor, no_grad
from torch.nn import CrossEntropyLoss
from torch.optim.sgd import SGD
from torch.optim.lr_scheduler import ReduceLROnPlateau

from utils.training import AvgrageMeter, accuracy

import models
from utils.data import load_data
from utils.args import logParameters
from utils.checkpoint import save_checkpoint
from utils.HtmlLogger import HtmlLogger


class TrainRegime:
    trainLossKey = 'Training loss'
    trainAccKey = 'Training acc'
    validLossKey = 'Validation loss'
    validAccKey = 'Validation acc'
    archLossKey = 'Arch loss'
    crossEntropyKey = 'CrossEntropy loss'
    flopsLossKey = 'Flops loss'
    epochNumKey = 'Epoch #'
    batchNumKey = 'Batch #'
    pathFlopsRatioKey = 'Path flops ratio'
    optFlopsRatioKey = 'Optimal flops ratio'
    validFlopsRatioKey = 'Validation flops ratio'
    timeKey = 'Time'
    lrKey = 'Optimizer lr'
    widthKey = 'Width'
    forwardCountersKey = 'Forward counters'

    # init formats for keys
    formats = {validLossKey: '{:.5f}', validAccKey: '{:.3f}', optFlopsRatioKey: '{:.3f}', timeKey: '{:.3f}',
               archLossKey: '{:.5f}', lrKey: '{:.5f}', flopsLossKey: '{:.5f}', crossEntropyKey: '{:.5f}',
               trainLossKey: '{:.5f}', trainAccKey: '{:.3f}', pathFlopsRatioKey: '{:.3f}', validFlopsRatioKey: '{:.3f}'
               }

    initWeightsTrainTableTitle = 'Initial weights training'
    k = 2
    alphasTableTitle = 'Alphas (top [{}])'.format(k)

    colsMainLogger = [epochNumKey, archLossKey, trainLossKey, trainAccKey, validLossKey, validAccKey, validFlopsRatioKey, widthKey, lrKey]
    colsMainInitWeightsTrain = [epochNumKey, trainLossKey, trainAccKey, validLossKey, validAccKey, validFlopsRatioKey, lrKey]
    colsTrainWeights = [batchNumKey, trainLossKey, trainAccKey, widthKey, pathFlopsRatioKey, timeKey]
    colsValidation = [batchNumKey, validLossKey, validAccKey, timeKey]
    colsValidationStatistics = [forwardCountersKey, widthKey, validFlopsRatioKey]

    def __init__(self, args, logger):
        # build model for uniform distribution of bits
        modelClass = models.__dict__[args.model]
        # init model
        model = modelClass(args)
        model = model.cuda()
        # create DataParallel model instance
        self.modelParallel = model
        # self.modelParallel = DataParallel(model, args.gpu)
        # assert (id(model) == id(self.modelParallel.module))

        # load data
        self.train_queue, self.search_queue, self.valid_queue = load_data(args)

        # log parameters
        logParameters(logger, args, model)

        self.args = args
        self.model = model
        self.modelClass = modelClass
        self.logger = logger

        # init email time
        self.lastMailTime = time()
        self.secondsBetweenMails = 1 * 3600

        self.trainFolderPath = '{}/{}'.format(args.save, args.trainFolder)

        # init cross entropy loss
        self.cross_entropy = CrossEntropyLoss().cuda()

        self.initialWeightsTraining(trainFolderName='init_weights_train')

        # init logger data table
        logger.createDataTable('Summary', self.colsMainLogger)

    @abstractmethod
    def train(self):
        raise NotImplementedError('subclasses must override train()!')

    # apply defined formats on dict values by keys
    def _applyFormats(self, dict):
        for k in dict.keys():
            if k in self.formats:
                dict[k] = self.formats[k].format(dict[k])

    def trainWeights(self, optimizer, epoch, loggers):
        print('*** trainWeights() ***')
        loss_container = AvgrageMeter()
        top1 = AvgrageMeter()

        model = self.model
        modelParallel = self.modelParallel
        crit = self.cross_entropy
        train_queue = self.train_queue

        trainLogger = loggers.get('train')
        if trainLogger:
            trainLogger.createDataTable('Epoch:[{}] - Training weights'.format(epoch), self.colsTrainWeights)

        nBatches = len(train_queue)

        modelParallel.train()
        assert (model.training is True)

        for step, (input, target) in enumerate(train_queue):
            startTime = time()
            n = input.size(0)

            input = tensor(input, requires_grad=False).cuda()
            target = tensor(target, requires_grad=False).cuda(async=True)

            # optimize model weights
            optimizer.zero_grad()
            logits = modelParallel(input)
            # calc loss
            loss = crit(logits, target)
            # back propagate
            loss.backward()
            # update weights
            optimizer.step()

            prec1 = accuracy(logits, target)[0]
            loss_container.update(loss.item(), n)
            top1.update(prec1.item(), n)

            endTime = time()

            if trainLogger:
                dataRow = {
                    self.batchNumKey: '{}/{}'.format(step, nBatches),
                    self.timeKey: (endTime - startTime), self.trainLossKey: loss, self.trainAccKey: prec1
                }
                # apply formats
                self._applyFormats(dataRow)
                # add row to data table
                trainLogger.addDataRow(dataRow)

        # log accuracy, loss, etc.
        summaryData = {self.trainLossKey: loss_container.avg, self.trainAccKey: top1.avg, self.batchNumKey: 'Summary'}
        # apply formats
        self._applyFormats(summaryData)

        for _, logger in loggers.items():
            logger.addSummaryDataRow(summaryData)

        # log forward counters. if loggerFuncs==[] then it is just resets counters
        func = [lambda rows: trainLogger.addInfoTable(title=self.forwardCountersKey, rows=rows)] if trainLogger else []
        model.logForwardCounters(loggerFuncs=func)

        return summaryData

    def infer(self, nEpoch, loggers):
        print('*** infer() ***')
        objs = AvgrageMeter()
        top1 = AvgrageMeter()

        model = self.model
        modelParallel = self.modelParallel
        valid_queue = self.valid_queue
        crit = self.cross_entropy

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

                logits = modelParallel(input)
                loss = crit(logits, target)

                prec1 = accuracy(logits, target)[0]
                n = input.size(0)
                objs.update(loss.item(), n)
                top1.update(prec1.item(), n)

                endTime = time()

                if trainLogger:
                    dataRow = {
                        self.batchNumKey: '{}/{}'.format(step, nBatches), self.validLossKey: loss, self.validAccKey: prec1,
                        self.timeKey: endTime - startTime
                    }
                    # apply formats
                    self._applyFormats(dataRow)
                    # add row to data table
                    trainLogger.addDataRow(dataRow)

        # create summary row
        summaryRow = {self.batchNumKey: 'Summary', self.validLossKey: objs.avg, self.validAccKey: top1.avg}
        # apply formats
        self._applyFormats(summaryRow)

        for _, logger in loggers.items():
            logger.addSummaryDataRow(summaryRow)

        # log forward counters. if loggerFuncs==[] then it is just resets counters
        func = []
        forwardCountersData = [[]]
        if trainLogger:
            func = [lambda rows: forwardCountersData.append(trainLogger.createInfoTable('Show', rows))]

        model.logForwardCounters(loggerFuncs=func)

        if trainLogger:
            # create new data table for validation statistics
            trainLogger.createDataTable('Validation statistics', self.colsValidationStatistics)
            # add bitwidth & forward counters statistics
            dataRow = {self.forwardCountersKey: forwardCountersData[-1]}
            # apply formats
            self._applyFormats(dataRow)
            # add row to table
            trainLogger.addDataRow(dataRow)

        return top1.avg, objs.avg, summaryRow

    def initialWeightsTraining(self, trainFolderName, filename=None):
        model = self.model
        modelParallel = self.modelParallel
        args = self.args
        logger = self.logger

        # create train folder
        folderPath = '{}/{}'.format(self.trainFolderPath, trainFolderName)
        if not exists(folderPath):
            makedirs(folderPath)

        # init optimizer
        optimizer = SGD(modelParallel.parameters(), args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)

        epoch = 0
        # init table in main logger
        logger.createDataTable(self.initWeightsTrainTableTitle, self.colsMainInitWeightsTrain)

        # # calc alpha trainset loss on baselines
        # self.calcAlphaTrainsetLossOnBaselines(folderPath, self.archLossKey, logger)

        # init validation best precision value
        best_prec1 = 0.0
        best_valid_loss = 0.0

        # count how many epochs current optimum hasn't changed
        nEpochsOptimum = 0
        logger.addInfoTable('Optimum', [['Epochs as optimum', nEpochsOptimum], ['Update time', logger.getTimeStr()]])

        # init scheduler
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.95, patience=2, min_lr=args.learning_rate_min)

        while nEpochsOptimum <= args.optimal_epochs:
            epoch += 1
            trainLogger = HtmlLogger(folderPath, str(epoch))
            trainLogger.addInfoTable('Learning rates', [['optimizer_lr', self.formats[self.lrKey].format(optimizer.param_groups[0]['lr'])]])

            # set loggers dictionary
            loggersDict = dict(train=trainLogger)
            # training
            print('========== Epoch:[{}] =============='.format(epoch))
            trainData = self.trainWeights(optimizer, epoch, loggersDict)

            # add epoch number
            trainData[self.epochNumKey] = epoch
            # add learning rate
            trainData[self.lrKey] = self.formats[self.lrKey].format(optimizer.param_groups[0]['lr'])

            # validation
            valid_acc, valid_loss, validData = self.infer(epoch, loggersDict)
            # merge trainData with validData
            for k, v in validData.items():
                trainData[k] = v

            # update scheduler
            scheduler.step(valid_loss)

            # update best precision only after switching stage is complete
            is_best = valid_acc > best_prec1
            if is_best:
                best_prec1 = valid_acc
                best_valid_loss = valid_loss
                # found new optimum, reset nEpochsOptimum
                nEpochsOptimum = 0
            else:
                # optimum hasn't changed
                nEpochsOptimum += 1

            # update nEpochsOptimum table
            logger.addInfoTable('Optimum', [[self.validAccKey, self.formats[self.validAccKey].format(best_prec1)], ['Epoch#', epoch - nEpochsOptimum],
                                            ['Epochs as optimum', nEpochsOptimum], ['Update time', logger.getTimeStr()]])

            # save model checkpoint
            save_checkpoint(self.trainFolderPath, model, args, best_prec1, is_best, filename)

            # add data to main logger table
            logger.addDataRow(trainData)

        # add optimal accuracy
        summaryRow = {self.epochNumKey: 'Optimal', self.validAccKey: best_prec1, self.validLossKey: best_valid_loss}
        self._applyFormats(summaryRow)
        logger.addSummaryDataRow(summaryRow)

        # # save pre-trained checkpoint
        # save_checkpoint(self.trainFolderPath, model, args, epoch, best_prec1, is_best=False, filename='pre_trained')

        # save optimal validation values
        setattr(args, self.validAccKey, best_prec1)
        setattr(args, self.validLossKey, best_valid_loss)
