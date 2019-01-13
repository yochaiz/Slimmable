from .regime import TrainRegime

from utils.trainWeights import TrainWeights
from utils.checkpoint import save_checkpoint


class PreTrainedTrainWeights(TrainWeights):
    pathKey = 'Path'
    tableTitle = 'Train pre-trained model'
    tableCols = [TrainWeights.epochNumKey, TrainWeights.trainLossKey, TrainWeights.trainAccKey,
                 TrainWeights.validLossKey, TrainWeights.validAccKey, TrainWeights.validFlopsRatioKey, TrainWeights.lrKey]

    def __init__(self, regime, maxEpoch):
        super(PreTrainedTrainWeights, self).__init__(regime)

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

    def postEpoch(self, epoch, optimizer, trainData, validData, validAcc, validLoss):
        logger = self.getLogger()
        model = self.getModel()
        # add epoch number
        trainData[self.epochNumKey] = epoch
        # add learning rate
        trainData[self.lrKey] = self.formats[self.lrKey](optimizer.param_groups[0]['lr'])
        # add flops ratio
        trainData[self.validFlopsRatioKey] = self.formats[self.validFlopsRatioKey](model.flopsRatio())

        # merge trainData with validData
        for k, v in validData.items():
            trainData[k] = v

        # save model checkpoint
        save_checkpoint(self.getTrainFolderPath(), model, optimizer, validAcc)

        # add data to main logger table
        logger.addDataRow(trainData)

        # select new path for next epoch
        self._selectNewPath()

    def postTrain(self):
        self.getLogger().addInfoToDataTable('Done !')


class PreTrainedRegime(TrainRegime):
    def __init__(self, args, logger):
        super(PreTrainedRegime, self).__init__(args, logger)

        self.trainWeights = PreTrainedTrainWeights(self, 2000)

    def buildStatsContainers(self):
        pass

    def train(self):
        self.trainWeights.train('init_weights_train')
