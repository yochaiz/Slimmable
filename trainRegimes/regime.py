from abc import abstractmethod

from torch.nn.parallel.data_parallel import DataParallel

import models
from utils.data import load_data
from utils.args import logParameters
from utils.statistics import Statistics


# from os import makedirs
# from os.path import exists
# from torch.optim.sgd import SGD
# from torch.optim.lr_scheduler import ReduceLROnPlateau
# from utils.checkpoint import save_checkpoint
# from utils.HtmlLogger import HtmlLogger
# from utils.training import TrainingOptimum


class TrainRegime:
    def __init__(self, args, logger):
        # init model
        model = self.buildModel(args)
        model = model.cuda()
        # create DataParallel model instance
        self.modelParallel = model
        # self.modelParallel = DataParallel(model, args.gpu)
        # assert (id(model) == id(self.modelParallel.module))

        self.args = args
        self.model = model
        self.logger = logger

        # load data
        self.train_queue, self.search_queue, self.valid_queue = load_data(args)
        # init train folder path, where to save loggers, checkpoints, etc.
        self.trainFolderPath = '{}/{}'.format(args.save, args.trainFolder)

        # build statistics containers
        containers = self.buildStatsContainers()
        # containers = {
        #     self.lossAvgKey: self._containerPerAlpha(model),
        #     self.crossEntropyLossAvgKey: self._containerPerAlpha(model),
        #     self.flopsLossAvgKey: self._containerPerAlpha(model),
        #     self.lossVarianceKey: self._containerPerAlpha(model),
        #     self.alphaDistributionKey: self._containerPerAlpha(model),
        #     self.entropyKey: [{layerIdx: [] for layerIdx in range(len(model.layersList()))}]
        # }
        # init statistics instance
        self.statistics = Statistics(containers, args.save)

        # log parameters
        logParameters(logger, args, model)

    @abstractmethod
    def train(self):
        raise NotImplementedError('subclasses must override train()!')

    @abstractmethod
    def buildStatsContainers(self):
        raise NotImplementedError('subclasses must override buildStatsContainers()!')

    def buildModel(self, args):
        # get model constructor
        modelKey = '{}_{}'.format(args.model, args.dataset)
        modelClass = models.__dict__[modelKey]

        return modelClass(args)

    def _containerPerAlpha(self, model):
        return [{self._alphaPlotTitle(layer, idx): [] for idx in range(layer.nWidths())} for layer in model.layersList()]

    def _alphaPlotTitle(self, layer, alphaIdx):
        return '{} ({})'.format(layer.widthRatioByIdx(alphaIdx), layer.widthByIdx(alphaIdx))

    # def initialWeightsTraining(self, trainFolderName, filename=None):
    #     model = self.model
    #     modelParallel = self.modelParallel
    #     args = self.args
    #     logger = self.logger
    #
    #     # create train folder
    #     folderPath = '{}/{}'.format(self.trainFolderPath, trainFolderName)
    #     if not exists(folderPath):
    #         makedirs(folderPath)
    #
    #     # init optimizer
    #     optimizer = SGD(modelParallel.parameters(), args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)
    #
    #     # init table in main logger
    #     logger.createDataTable(self.initWeightsTrainTableTitle, self.colsMainInitWeightsTrain)
    #
    #     # # calc alpha trainset loss on baselines
    #     # self.calcAlphaTrainsetLossOnBaselines(folderPath, self.archLossKey, logger)
    #
    #     # init optimum info table headers
    #     optimumTableHeaders = [self.widthKey, self.validAccKey, self.epochNumKey, 'Epochs as optimum']
    #     # init TrainingOptimum instance
    #     trainOptimum = TrainingOptimum(model.baselineWidthKeys(), optimumTableHeaders, lambda value, optValue: value > optValue)
    #     # init optimal epoch data, we will display it in summary row
    #     optimalEpochData = None
    #
    #     # count how many epochs current optimum hasn't changed
    #     epoch = 0
    #     nEpochsOptimum = 0
    #     trainLoggerFlag = True
    #
    #     # init scheduler
    #     scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.95, patience=2, min_lr=args.learning_rate_min)
    #
    #     while nEpochsOptimum <= args.optimal_epochs:
    #         # update epoch number
    #         epoch += 1
    #         # init train logger
    #         trainLogger = None
    #         if trainLoggerFlag:
    #             trainLogger = HtmlLogger(folderPath, str(epoch))
    #             trainLogger.addInfoTable('Learning rates', [['optimizer_lr', self.formats[self.lrKey](optimizer.param_groups[0]['lr'])]])
    #
    #         # update train logger condition for next epoch
    #         trainLoggerFlag = ((epoch + 1) % args.logInterval) == 0
    #
    #         # set loggers dictionary
    #         loggersDict = dict(train=trainLogger)
    #         # training
    #         print('========== Epoch:[{}] =============='.format(epoch))
    #         trainData = self.trainWeightsEpoch(optimizer, epoch, loggersDict)
    #
    #         # add epoch number
    #         trainData[self.epochNumKey] = epoch
    #         # add learning rate
    #         trainData[self.lrKey] = self.formats[self.lrKey](optimizer.param_groups[0]['lr'])
    #
    #         # validation
    #         validAcc, validLoss, validData = self.inferEpoch(epoch, loggersDict)
    #         # merge trainData with validData
    #         for k, v in validData.items():
    #             trainData[k] = v
    #
    #         # update scheduler
    #         scheduler.step(trainOptimum.dictAvg(validLoss))
    #
    #         # update optimum values according to current epoch values and get optimum table for logger
    #         optimumTable = trainOptimum.update(validAcc, epoch)
    #         # add update time to optimum table
    #         optimumTable.append(['Update time', logger.getTimeStr()])
    #         # update nEpochsOptimum table
    #         logger.addInfoTable('Optimum', optimumTable)
    #
    #         # update best precision only after switching stage is complete
    #         is_best = trainOptimum.is_best(epoch)
    #         if is_best:
    #             # update optimal epoch data
    #             optimalEpochData = (validAcc, validLoss)
    #             # found new optimum, reset nEpochsOptimum
    #             nEpochsOptimum = 0
    #         else:
    #             # optimum hasn't changed
    #             nEpochsOptimum += 1
    #
    #         # save model checkpoint
    #         save_checkpoint(self.trainFolderPath, model, optimizer, validAcc, is_best, filename)
    #
    #         # add data to main logger table
    #         logger.addDataRow(trainData)
    #
    #     # add optimal accuracy
    #     optAcc, optLoss = optimalEpochData
    #     summaryRow = {self.epochNumKey: 'Optimal', self.validAccKey: optAcc, self.validLossKey: optLoss}
    #     self._applyFormats(summaryRow)
    #     logger.addSummaryDataRow(summaryRow)
    #
    #     # # save pre-trained checkpoint
    #     # save_checkpoint(self.trainFolderPath, model, args, epoch, best_prec1, is_best=False, filename='pre_trained')
    #
    #     # save optimal validation values
    #     setattr(args, self.validAccKey, optAcc)
    #     setattr(args, self.validLossKey, optLoss)
