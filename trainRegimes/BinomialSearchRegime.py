from .SearchRegime import EpochTrainWeights
from .MultinomialSearchRegime import MultinomialSearchRegime


class BinomialTrainWeights(EpochTrainWeights):
    def __init__(self, getModel, getModelParallel, getArgs, getLogger, getTrainQueue, getValidQueue, getTrainFolderPath, maxEpoch, currEpoch):
        super(BinomialTrainWeights, self).__init__(getModel, getModelParallel, getArgs, getLogger, getTrainQueue, getValidQueue, getTrainFolderPath,
                                                   maxEpoch, currEpoch)

        model = getModel()
        # init training width list with baseline widths
        self._widthList = {k: v for k, v in model.baselineWidth()}
        # set new path to alpha mean
        # we train only on this path, we don't select path in each epoch
        model.choosePathAlphasAsPartition()
        # add new path indices to training width list
        self._widthList[self.pathKey] = model.currWidthIdx()

    def widthList(self):
        return self._widthList.items()

    def _selectNewPath(self):
        pass


class BinomialSearchRegime(MultinomialSearchRegime):
    def __init__(self, args, logger):
        super(BinomialSearchRegime, self).__init__(args, logger)

    def TrainWeightsClass(self):
        return BinomialTrainWeights
