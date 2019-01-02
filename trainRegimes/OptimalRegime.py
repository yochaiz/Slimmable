from .regime import TrainRegime


class OptimalRegime(TrainRegime):
    def __init__(self, args, logger):
        # update values
        args.train_portion = 1.0
        # args.batch_size = 250

        super(OptimalRegime, self).__init__(args, logger)

    def train(self):
        # train model weights
        self.initialWeightsTraining(trainFolderName='init_weights_train')

        # init logger data table
        self.logger.createDataTable('Summary', self.colsMainLogger)
