from .regime import TrainRegime

from utils.trainWeights import TrainWeights, EpochData
from utils.checkpoint import save_checkpoint


class PreTrainedTrainWeights(TrainWeights):
    pathKey = 'Path'
    tableTitle = 'Train pre-trained model'
    tableCols = [TrainWeights.epochNumKey, TrainWeights.trainLossKey, TrainWeights.trainAccKey,
                 TrainWeights.validLossKey, TrainWeights.validAccKey, TrainWeights.validFlopsRatioKey, TrainWeights.lrKey]

    def __init__(self, getModel, getModelParallel, getArgs, getLogger, getTrainQueue, getValidQueue, getTrainFolderPath, maxEpoch):
        super(PreTrainedTrainWeights, self).__init__(getModel, getModelParallel, getArgs, getLogger, getTrainQueue, getValidQueue, getTrainFolderPath)

        # init table in main logger
        self.getLogger().createDataTable(self.tableTitle, self.tableCols)
        # select new path
        self._selectNewPath()
        # init max epoch
        self.maxEpoch = maxEpoch

    def _selectNewPath(self):
        model = self.getModel()
        # select new path
        model.choosePathByAlphas()
        # get new path indices
        self.widthIdxList = model.currWidthIdx()
        print(self.widthIdxList)

    def stopCondition(self, epoch):
        return epoch >= self.maxEpoch

    def widthList(self):
        return {self.pathKey: self.widthIdxList}.items()

    def schedulerMetric(self, validLoss):
        return validLoss[self.pathKey]

    def postEpoch(self, epoch, optimizer, trainData: EpochData, validData: EpochData):
        logger = self.getLogger()
        model = self.getModel()

        trainDataRow = trainData.summaryDataRow()
        validDataRow = validData.summaryDataRow()

        # add epoch number
        trainDataRow[self.epochNumKey] = epoch
        # add learning rate
        trainDataRow[self.lrKey] = self.formats[self.lrKey](optimizer.param_groups[0]['lr'])
        # add flops ratio
        trainDataRow[self.validFlopsRatioKey] = self.formats[self.validFlopsRatioKey](model.flopsRatio())

        # merge trainDataRow with validDataRow
        for k, v in validDataRow.items():
            trainDataRow[k] = v

        # save model checkpoint
        save_checkpoint(self.getTrainFolderPath(), model, optimizer, validData.accDict())

        # add data to main logger table
        logger.addDataRow(trainDataRow)

        # select new path for next epoch
        self._selectNewPath()

    def postTrain(self):
        self.getLogger().addInfoToDataTable('Done !')


class PreTrainedRegime(TrainRegime):
    def __init__(self, args, logger):
        super(PreTrainedRegime, self).__init__(args, logger)

        self.trainWeights = PreTrainedTrainWeights(self.getModel, self.getModelParallel, self.getArgs, self.getLogger, self.getTrainQueue,
                                                   self.getValidQueue, self.getTrainFolderPath, args.weights_epochs)

    def buildStatsContainers(self):
        pass

    def buildStatsRules(self):
        return {}

    def train(self):
        self.trainWeights.train()
