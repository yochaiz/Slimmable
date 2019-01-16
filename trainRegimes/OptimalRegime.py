from .regime import TrainRegime

from torch import save as saveCheckpoint

from utils.trainWeights import TrainWeights, EpochData
from utils.training import TrainingOptimum
from utils.checkpoint import save_checkpoint


class OptimalTrainWeights(TrainWeights):
    initWeightsTrainTableTitle = 'Initial weights training'
    colsMainInitWeightsTrain = [TrainWeights.epochNumKey, TrainWeights.trainLossKey, TrainWeights.trainAccKey,
                                TrainWeights.validLossKey, TrainWeights.validAccKey, TrainWeights.lrKey]

    def __init__(self, regime):
        super(OptimalTrainWeights, self).__init__(regime)

        # init table in main logger
        self.getLogger().createDataTable(self.initWeightsTrainTableTitle, self.colsMainInitWeightsTrain)

        # # calc alpha trainset loss on baselines
        # self.calcAlphaTrainsetLossOnBaselines(folderPath, self.archLossKey, logger)

        # init optimum info table headers
        optimumTableHeaders = [self.widthKey, self.validAccKey, self.epochNumKey, 'Epochs as optimum']
        # init TrainingOptimum instance
        self.trainOptimum = TrainingOptimum(self.getModel().baselineWidthKeys(), optimumTableHeaders, lambda value, optValue: value > optValue)
        # init optimal epoch data, we will display it in summary row
        self.optimalEpochData = None

        # count how many epochs current optimum hasn't changed
        self.nEpochsOptimum = 0

    def stopCondition(self, epoch):
        return self.nEpochsOptimum > self.getArgs().optimal_epochs

    def widthList(self):
        return self.getModel().baselineWidth()

    def schedulerMetric(self, validLoss):
        return self.trainOptimum.dictAvg(validLoss)

    def postEpoch(self, epoch, optimizer, trainData:EpochData, validData:EpochData):
        logger = self.getLogger()
        model = self.getModel()
        # add epoch number
        trainData[self.epochNumKey] = epoch
        # add learning rate
        trainData[self.lrKey] = self.formats[self.lrKey](optimizer.param_groups[0]['lr'])

        # merge trainData with validData
        for k, v in validData.items():
            trainData[k] = v

        # get valid acc dict & loss dict
        validAccDict = validData.accDict()
        validLossDict = validData.lossDict()
        # update optimum values according to current epoch values and get optimum table for logger
        optimumTable = self.trainOptimum.update(validAccDict, epoch)
        # add update time to optimum table
        optimumTable.append(['Update time', logger.getTimeStr()])
        # update nEpochsOptimum table
        logger.addInfoTable('Optimum', optimumTable)

        # update best precision only after switching stage is complete
        is_best = self.trainOptimum.is_best(epoch)
        if is_best:
            # update optimal epoch data
            self.optimalEpochData = (validAccDict, validLossDict)
            # found new optimum, reset nEpochsOptimum
            self.nEpochsOptimum = 0
        else:
            # optimum hasn't changed
            self.nEpochsOptimum += 1

        # save model checkpoint
        save_checkpoint(self.getTrainFolderPath(), model, optimizer, validAccDict, is_best)

        # add data to main logger table
        logger.addDataRow(trainData)

    def postTrain(self):
        args = self.getArgs()
        # add optimal accuracy
        optAcc, optLoss = self.optimalEpochData
        summaryRow = {self.epochNumKey: 'Optimal', self.validAccKey: optAcc, self.validLossKey: optLoss}
        self._applyFormats(summaryRow)
        self.getLogger().addSummaryDataRow(summaryRow)

        # # save pre-trained checkpoint
        # save_checkpoint(self.getTrainFolderPath(), model, args, epoch, best_prec1, is_best=False, filename='pre_trained')

        # save optimal validation values
        setattr(args, self.validAccKey, optAcc)
        setattr(args, self.validLossKey, optLoss)


class OptimalRegime(TrainRegime):
    def __init__(self, args, logger):
        # update values
        args.train_portion = 1.0
        # args.batch_size = 250

        super(OptimalRegime, self).__init__(args, logger)

        self.trainWeights = OptimalTrainWeights(self)

    def buildStatsContainers(self):
        pass

    def buildStatsRules(self):
        return {}

    def train(self):
        args = self.args
        logger = self.logger
        # train model weights
        self.trainWeights.train()

        # init logger data table
        self.logger.createDataTable(self.trainWeights.summaryKey, self.colsMainLogger)
        # set save target path
        targetPath = args.json
        # best_prec1, best_valid_loss are now part of args, therefore we have to save args again
        saveCheckpoint(args, targetPath)
        # log finish
        print(args)
        logger.addInfoToDataTable('Saved args to [{}]'.format(targetPath))
        logger.addInfoToDataTable('Done !')
