from .regime import TrainRegime

from utils.trainWeights import TrainWeights
from utils.checkpoint import save_checkpoint


class PreTrainedTrainWeights(TrainWeights):
    pathKey = 'Path'
    tableTitle = 'Train pre-trained model'
    tableCols = [TrainWeights.epochNumKey, TrainWeights.trainLossKey, TrainWeights.trainAccKey,
                 TrainWeights.validLossKey, TrainWeights.validAccKey, TrainWeights.validFlopsRatioKey, TrainWeights.lrKey]

    def __init__(self, args, model, modelParallel, logger, train_queue, valid_queue, trainFolderPath):
        super(PreTrainedTrainWeights, self).__init__(args, model, modelParallel, logger, train_queue, valid_queue)

        self.trainFolderPath = trainFolderPath
        # init table in main logger
        self.logger.createDataTable(self.tableTitle, self.tableCols)
        # select new path
        self._selectNewPath()

    def _selectNewPath(self):
        # select new path
        self.model.choosePathByAlphas()
        # get new path indices
        self.widthIdxList = self.model.currWidthIdx()
        print(self.widthIdxList)

    def stopCondition(self, epoch):
        return epoch >= 2000

    def widthList(self):
        return {self.pathKey: self.widthIdxList}.items()

    def schedulerMetric(self, validLoss):
        return validLoss[self.pathKey]

    def postEpoch(self, epoch, optimizer, trainData, validData, validAcc, validLoss):
        logger = self.logger
        model = self.model
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
        save_checkpoint(self.trainFolderPath, model, optimizer, validAcc)

        # add data to main logger table
        logger.addDataRow(trainData)

        # select new path for next epoch
        self._selectNewPath()

    def postTrain(self):
        self.logger.addInfoToDataTable('Done !')


class PreTrainedRegime(TrainRegime):
    def __init__(self, args, logger):
        super(PreTrainedRegime, self).__init__(args, logger)

        self.trainWeights = PreTrainedTrainWeights(self.args, self.model, self.modelParallel, self.logger, self.train_queue, self.valid_queue,
                                                   self.trainFolderPath)

    def buildStatsContainers(self):
        pass

    def train(self):
        self.trainWeights.train(self.trainFolderPath, 'init_weights_train')
