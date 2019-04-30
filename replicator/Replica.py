from abc import abstractmethod

from .TrainPathWeights import TrainPathWeights


class Replica:
    def __init__(self, buildModelFunc: callable, modelStateDict: dict, modelAlphas: list, gpu: int, trainWeightsElements: tuple):
        # can't pass regime class functions, therefore passing functions return value
        args, logger, trainQueue, trainFolderPath = trainWeightsElements

        # replicate regime model
        self._replicateModel(buildModelFunc, args, modelStateDict, modelAlphas)
        # init trainWeights instance
        self._trainWeights = TrainPathWeights(getModel=self.getModel, getModelParallel=self.getModel, getArgs=lambda: args, getLogger=lambda: logger,
                                              getTrainQueue=lambda: trainQueue, getValidQueue=lambda: None,
                                              getTrainFolderPath=lambda: trainFolderPath, gpu=gpu)

    @abstractmethod
    # init paths for training
    def initTrainPaths(self, params: tuple) -> dict:
        raise NotImplementedError('subclasses must override initTrainPaths()!')

    @abstractmethod
    # init paths for loss evaluation
    def initLossEvaluationPaths(self, params: tuple, trainPaths: dict) -> dict:
        raise NotImplementedError('subclasses must override initLossEvaluationPaths()!')

    @abstractmethod
    # restore cModel weights before training paths
    def restoreModelOriginalWeights(self):
        raise NotImplementedError('subclasses must override restoreModelOriginalWeights()!')

    def getModel(self):
        return self._cModel

    def _updateWeights(self, srcModelStateDict: dict):
        model = self._cModel
        # copy weights
        model.load_state_dict(srcModelStateDict)
        # save current cModel weights
        self._originalWeightsDict = model.state_dict()

    def _updateAlphas(self, srcModelAlphas: list):
        model = self._cModel
        # copy alphas
        model.updateAlphas(srcModelAlphas)

    def _replicateModel(self, buildModelFunc, args, modelStateDict: dict, modelAlphas: list):
        # create model new instance
        cModel = buildModelFunc(args)
        # set model to cuda on specific GPU
        cModel = cModel.cuda()
        # set mode to eval mode
        cModel.eval()
        # set as class member
        self._cModel = cModel
        # copy weights from regime model
        self._updateWeights(modelStateDict)
        # copy alphas
        self._updateAlphas(modelAlphas)

    # assumes cModel has original weights + original BNs, i.e. restoreModelOriginalWeights() has been applied
    def train(self, params: tuple):
        # init training paths
        trainPaths = self.initTrainPaths(params)
        # train
        self._trainWeights.train(trainPaths)

        return self.initLossEvaluationPaths(params, trainPaths)


class MultinomialReplica(Replica):
    def __init__(self, buildModelFunc: callable, modelStateDict: dict, modelAlphas: list, gpu: int, trainWeightsElements: tuple):
        super(MultinomialReplica, self).__init__(buildModelFunc, modelStateDict, modelAlphas, gpu, trainWeightsElements)

    # restore cModel weights before training paths
    def restoreModelOriginalWeights(self):
        model = self._cModel
        # restore model state_dict structure
        model.restoreOriginalStateDictStructure()
        # load original weights
        model.load_state_dict(self._originalWeightsDict)

    def initTrainPaths(self, params: tuple) -> dict:
        srcPath = params
        # init trained paths with homogeneous paths
        trainPaths = {width: path for width, path in self._cModel.baselineWidth()}
        # add path sampled from distribution
        trainPaths[self._cModel.partitionKey()] = srcPath
        print('nTrainedConfigs:[{}]'.format(len(trainPaths)))

        # return {self._cModel.partitionKey(): srcPath}
        return trainPaths

    def initLossEvaluationPaths(self, params: tuple, trainPaths: dict) -> dict:
        srcPath = params
        return {self._cModel.partitionKey(): srcPath}


class CategoricalReplica(Replica):
    def __init__(self, buildModelFunc: callable, modelStateDict: dict, modelAlphas: list, gpu: int, trainWeightsElements: tuple):
        super(CategoricalReplica, self).__init__(buildModelFunc, modelStateDict, modelAlphas, gpu, trainWeightsElements)

    # restore cModel weights before training paths
    def restoreModelOriginalWeights(self):
        model = self._cModel
        # restore cModel original BNs
        model.restoreOriginalBNs()
        # load weights
        model.load_state_dict(self._originalWeightsDict)

    def initTrainPaths(self, params: tuple) -> dict:
        model = self._cModel
        # extract params
        layer, srcPath = params
        # generate independent BNs in each layer for each path
        model.generatePathBNs(layer)
        # init layer paths for training
        layerPaths = {}
        # set layer paths for training
        for idx in range(layer.nWidths()):
            # we generated new BNs in other layers, therefore paths have no overlap
            layerPaths[layer.widthRatioByIdx(idx)] = [idx] * len(srcPath)

        return layerPaths

    def initLossEvaluationPaths(self, params: tuple, trainPaths: dict) -> dict:
        return trainPaths

# def gpu(self) -> int:
#     return self._gpu

# def cModel(self) -> BaseNet:
#     return self._cModel

# def regime(self) -> TrainRegime:
#     return self._regime

# def trainQueue(self) -> DataLoader:
#     return self._regime.train_queue

# def initNewEpoch(self, modelStateDict: dict):
#     # update replica weights
#     self._updateWeights(modelStateDict)
#     # init new TrainPathWeights instance
#     self._trainWeights = TrainPathWeights(self)
