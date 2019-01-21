from abc import abstractmethod

from .TrainPathWeights import TrainPathWeights


class Replica:
    def __init__(self, buildModelFunc: callable, modelStateDict: dict, modelAlphas: list, gpu: int, trainWeightsElements: tuple):
        # # set device to required gpu
        # set_device(gpu)
        # self._regime = regime

        # can't pass regime class functions, therefore passing functions return value
        args, logger, trainQueue, validQueue, trainFolderPath = trainWeightsElements

        # self._gpu = gpu
        # replicate regime model
        self._replicateModel(buildModelFunc, args, modelStateDict, modelAlphas)
        # init trainWeights instance
        self._trainWeights = TrainPathWeights(getModel=self.getModel, getModelParallel=self.getModel, getArgs=lambda: args, getLogger=lambda: logger,
                                              getTrainQueue=lambda: trainQueue, getValidQueue=lambda: validQueue,
                                              getTrainFolderPath=lambda: trainFolderPath, gpu=gpu)

    @abstractmethod
    # init paths for training
    def initTrainPaths(self, params: tuple):
        raise NotImplementedError('subclasses must override initTrainPaths()!')

    @abstractmethod
    # restore cModel weights before training paths
    def restoreModelOriginalWeights(self):
        raise NotImplementedError('subclasses must override restoreModelOriginalWeights()!')

    def getModel(self):
        return self._cModel

    # def gpu(self) -> int:
    #     return self._gpu

    # def cModel(self) -> BaseNet:
    #     return self._cModel

    # def regime(self) -> TrainRegime:
    #     return self._regime

    # def trainQueue(self) -> DataLoader:
    #     return self._regime.train_queue

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

    # def initNewEpoch(self, modelStateDict: dict):
    #     # update replica weights
    #     self._updateWeights(modelStateDict)
    #     # init new TrainPathWeights instance
    #     self._trainWeights = TrainPathWeights(self)

    # assumes cModel has original weights + original BNs, i.e. restoreModelOriginalWeights() has been applied
    def train(self, params: tuple):
        # init training paths
        layerPaths = self.initTrainPaths(params)
        # # train
        # self._trainWeights.train(layerPaths)

        return layerPaths


class MultinomialReplica(Replica):
    def __init__(self, buildModelFunc: callable, modelStateDict: dict, modelAlphas: list, gpu: int, trainWeightsElements: tuple):
        super(MultinomialReplica, self).__init__(buildModelFunc, modelStateDict, modelAlphas, gpu, trainWeightsElements)

    # restore cModel weights before training paths
    def restoreModelOriginalWeights(self):
        model = self._cModel
        # load original weights
        model.load_state_dict(self._originalWeightsDict)

    def initTrainPaths(self, params: tuple):
        srcPath = params
        return {0: srcPath}


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

    def initTrainPaths(self, params: tuple):
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
