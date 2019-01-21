from time import time
from utils.trainWeights import TrainWeights


class TrainPathWeights(TrainWeights):
    def __init__(self, getModel, getModelParallel, getArgs, getLogger, getTrainQueue, getValidQueue, getTrainFolderPath, gpu):
        super(TrainPathWeights, self).__init__(getModel, getModelParallel, getArgs, getLogger, getTrainQueue, getValidQueue, getTrainFolderPath)

        # init layer paths dictionary
        self.layerPaths = {}
        # save GPU device number
        self._gpu = gpu

    def stopCondition(self, epoch):
        return epoch >= 5

    def widthList(self):
        return self.layerPaths.items()

    def train(self, paths: dict):
        # update training paths
        self.layerPaths = paths
        # init optimizer
        optimizer = self._initOptimizer()
        # train
        epoch = 0
        startTime = time()
        while not self.stopCondition(epoch):
            epoch += 1
            self.weightsEpoch(optimizer, epoch, {})

        # count training time
        endTime = time()
        print('Train time:[{}] - GPU:[{}]'.format(self.formats[self.timeKey](endTime - startTime), self._gpu))

# def getModel(self):
#     return self._replica.cModel()
#
# def getModelParallel(self):
#     return self.getModel()
#
# def getTrainQueue(self):
#     return self._replica.trainQueue()
