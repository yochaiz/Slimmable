from math import floor
from time import time
from multiprocessing.pool import Pool

from torch import tensor, zeros
from torch.cuda import set_device, current_device
from torch.utils.data.dataloader import DataLoader

from trainRegimes.regime import TrainRegime
from models.BaseNet import BaseNet, SlimLayer
from utils.trainWeights import TrainWeights


# from multiprocessing import Process
# class NoDaemonProcess(Process):
#     # make 'daemon' attribute always return False
#     def _get_daemon(self):
#         return False
#
#     def _set_daemon(self, value):
#         pass
#
#     daemon = property(_get_daemon, _set_daemon)
#
#
# # We sub-class multiprocessing.pool.Pool instead of multiprocessing.Pool
# # because the latter is only a wrapper function, not a proper class.
# class MyPool(Pool):
#     Process = NoDaemonProcess

# # usage example
# with MyPool(processes=nCopies, maxtasksperchild=1) as p:


class Replica:
    def __init__(self, regime: TrainRegime, gpu: int):
        # set device to required gpu
        set_device(gpu)

        self._regime = regime
        self._gpu = gpu
        # init empty trainWeights instance
        self._trainWeights = None
        # replicate regime model
        self._replicateModel(regime)

    def gpu(self) -> int:
        return self._gpu

    def cModel(self) -> BaseNet:
        return self._cModel

    def regime(self) -> TrainRegime:
        return self._regime

    def trainQueue(self) -> DataLoader:
        return self._regime.train_queue

    def _updateWeights(self, srcModel: BaseNet):
        # copy weights
        self._cModel.load_state_dict(srcModel.state_dict())
        # save current cModel weights
        self._originalWeightsDict = self._cModel.state_dict()

    # restore cModel weights before training paths
    def restoreModelOriginalWeights(self):
        model = self._cModel
        # restore cModel original BNs
        model.restoreOriginalBNs()
        # load weights
        model.load_state_dict(self._originalWeightsDict)

    def _replicateModel(self, regime: TrainRegime) -> BaseNet:
        # create model new instance
        cModel = regime.buildModel(regime.args)
        # set model to cuda on specific GPU
        cModel = cModel.cuda()
        # set mode to eval mode
        cModel.eval()
        # set as class member
        self._cModel = cModel
        # copy weights from regime model
        self._updateWeights(regime.model)

    def initNewEpoch(self, model: BaseNet):
        # update replica weights
        self._updateWeights(model)
        # init new TrainPathWeights instance
        self._trainWeights = TrainPathWeights(self)

    # assumes cModel has original weights + original BNs, i.e. restoreModelOriginalWeights() has been applied
    def train(self, layer: SlimLayer, srcPath: list):
        model = self._cModel
        # generate independent BNs in each layer for each path
        model.generatePathBNs(layer)
        # init layer paths for training
        layerPaths = {}
        # set layer paths for training
        for idx in range(layer.nWidths()):
            # we generated new BNs in other layers, therefore paths have no overlap
            layerPaths[layer.widthRatioByIdx(idx)] = [idx] * len(srcPath)

        # train
        self._trainWeights.train(layerPaths)

        return layerPaths


class TrainPathWeights(TrainWeights):
    def __init__(self, replica: Replica):
        self._replica = replica

        super(TrainPathWeights, self).__init__(replica.regime())

        # init layer paths dictionary
        self.layerPaths = {}

    def getModel(self):
        return self._replica.cModel()

    def getModelParallel(self):
        return self.getModel()

    def getTrainQueue(self):
        return self._replica.trainQueue()

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
        print('Train time:[{}] - GPU:[{}]'.format(self.formats[self.timeKey](endTime - startTime), self._replica.gpu()))


class ModelReplicator:
    title = 'Replications'

    def __init__(self, regime: TrainRegime):
        self.gpuIDs = regime.args.gpu
        self._model = regime.model
        self._flopsLoss = regime.flopsLoss

        # save current device
        currDevice = current_device()
        # create replications
        self.replications = [Replica(regime, gpu) for gpu in self.gpuIDs]
        # reset device back to current device
        set_device(currDevice)

        # create info table
        regime.logger.addInfoTable(self.title, [['#', len(self.replications)]])

    def initNewEpoch(self):
        for replica in self.replications:
            replica.initNewEpoch(self._model)

    def buildArgs(self, dataPerGPU: dict, nSamplesPerCopy: int):
        args = ((replica, dataPerGPU[replica.gpu()], nSamplesPerCopy[replica.gpu()]) for replica in self.replications)
        return args

    # given paths history dictionary and current path, checks if current path exists in history dictionary
    @staticmethod
    def _doesPathExist(pathsHistoryDict: dict, currPath: list) -> bool:
        currDict = pathsHistoryDict
        # init boolean flag
        pathExists = True
        for v in currPath:
            if v not in currDict:
                pathExists = False
                currDict[v] = {}
            # update current dict, we move to next layer
            currDict = currDict[v]

        return pathExists

    # select new path based on alphas distribution.
    # check that selected path hasn't been selected before
    def _generateNewPath(self, replica: Replica, cModel: BaseNet, pathsHistoryDict: dict) -> list:
        # restore model original weights & BNs
        replica.restoreModelOriginalWeights()
        # init does path exist flag
        pathExists = True
        while pathExists:
            # select path based on alphas distribution
            cModel.choosePathByAlphas()
            # get selected path indices
            pathWidthIdx = cModel.currWidthIdx()
            # check that selected path hasn't been selected before
            pathExists = self._doesPathExist(pathsHistoryDict, pathWidthIdx)

        print('pathWidthIdx:{}'.format(pathWidthIdx))
        return pathWidthIdx

    def _splitSamples(self, nSamples: int) -> dict:
        nCopies = len(self.gpuIDs)
        # split number of samples between model replications
        nSamplesPerCopy = {gpu: floor(nSamples / nCopies) for gpu in self.gpuIDs}
        # calc difference between nSamples to samples assigned to copies
        diff = nSamples - sum(nSamplesPerCopy.values())
        # split difference evenly between copies
        for idx in range(diff):
            nSamplesPerCopy[self.gpuIDs[idx]] += 1

        assert (sum(nSamplesPerCopy.values()) == nSamples)
        return nSamplesPerCopy

    def _updateReplicationsAlphas(self):
        for replica in self.replications:
            cModel = replica.cModel()
            for cLayer, mLayer in zip(cModel.layersList(), self._model.layersList()):
                cLayer.alphas().data.copy_(mLayer.alphas().data)

    def _cloneTensors(self, tensorsList: list) -> dict:
        dataPerGPU = {}
        for gpu in self.gpuIDs:
            gpuTensorsList = [t if (gpu == t.device.index) else t.clone().cuda(gpu) for t in tensorsList]
            dataPerGPU[gpu] = gpuTensorsList

        return dataPerGPU

    def loss(self, input: tensor, target: tensor, nSamples: int):
        # clone input & target to all GPUs
        dataPerGPU = self._cloneTensors([input, target])
        # split samples between model copies
        nSamplesPerModel = self._splitSamples(nSamples)
        # update replications alphas values
        self._updateReplicationsAlphas()
        # init loss replication function arguments
        args = self.buildArgs(dataPerGPU, nSamplesPerModel)

        nCopies = len(self.replications)
        with Pool(processes=nCopies, maxtasksperchild=1) as p:
            results = p.map(self.lossPerReplication, args)

        return self.processResults(results)

    def lossPerReplication(self, args: tuple) -> (list, list):
        replica, data, nSamples = args
        input, target = data
        cModel = replica.cModel()
        gpu = replica.gpu()
        # switch to process GPU
        set_device(gpu)

        # init samples (paths) history, to make sure we don't select the same sample twice
        pathsHistoryDict = {}
        # init samples (paths) history list for logging purposes
        pathsList = [[] for _ in cModel.layersList()]

        # init containers to save loss values
        lossDictsList = [[{k: [] for k in self._flopsLoss.lossKeys()} for _ in range(layer.nWidths())] for layer in cModel.layersList()]

        # iterate over samples. generate a sample (path), train it and evaluate alphas on sample
        for sampleIdx in range(nSamples):
            print('===== Sample idx:[{}/{}] - GPU:[{}] ====='.format(sampleIdx, nSamples, gpu))
            # iterate over layers. in each layer iterate over alphas
            for layerIdx, layer in enumerate(cModel.layersList()):
                print('=== Layer idx:[{}] - GPU:[{}] ==='.format(layerIdx, gpu))
                # select new path based on alphas distribution.
                # check that selected path hasn't been selected before
                pathWidthIdx = self._generateNewPath(replica, cModel, pathsHistoryDict)
                # add path to paths list
                pathsList[layerIdx].append([layer.widthByIdx(p) for p, layer in zip(pathWidthIdx, cModel.layersList())])
                # init containers to save loss values and variance
                layerLossDicts = lossDictsList[layerIdx]
                # train model on layer paths
                trainedPaths = replica.train(layer, pathWidthIdx)
                # switch to eval mode
                cModel.eval()
                # evaluate batch over trained paths
                for widthRatio, trainedPathIdx in trainedPaths.items():
                    # set cModel path to trained path
                    cModel.setCurrWidthIdx(trainedPathIdx)
                    # forward input in model selected path
                    logits = cModel(input)
                    # calc loss
                    lossDict = self._flopsLoss(logits, target, cModel.countFlops())
                    # get alpha Idx based on widthRatio
                    alphaIdx = layer.widthRatioIdx(widthRatio)
                    # add loss to container
                    alphaLossDict = layerLossDicts[alphaIdx]
                    for k, loss in lossDict.items():
                        alphaLossDict[k].append(loss.item())

        return lossDictsList, pathsList

    def processResults(self, results: list) -> (list, list):
        lossDictsList, pathsList = results[0]

        # append lists from GPUs
        for gpuLossDictsList, gpuPathsList in results[1:]:
            for layerIdx, layerLossDicts in enumerate(gpuLossDictsList):
                # add paths to list by layer index
                pathsList[layerIdx].extend(gpuPathsList[layerIdx])
                for alphaIdx, alphaLossDict in enumerate(layerLossDicts):
                    for lossName, lossList in alphaLossDict.items():
                        # add loss values to dict
                        lossDictsList[layerIdx][alphaIdx][lossName].extend(lossList)

        return lossDictsList, pathsList

# # calc loss distributed, i.e. for each model replication
# @abstractmethod
# def lossPerReplication(self, args):
#     raise NotImplementedError('subclasses must override lossPerReplication()!')

# # build args for pool.map
# @abstractmethod
# def buildArgs(self, inputPerGPU, targetPerGPU, nSamplesPerModel):
#     raise NotImplementedError('subclasses must override buildArgs()!')

# def demo(self, args):
#     (cModel, gpu), nSamples = args
#     # switch to process GPU
#     set_device(gpu)
#     assert (cModel.training is False)
#
#     data = randn(250, 3, 32, 32).cuda()
#
#     print('gpu [{}] start'.format(gpu))
#
#     with no_grad():
#         for _ in range(nSamples):
#             cModel(data)
#             data[0, 0, 0, 0] += 0.001
#
#     print('gpu [{}] end'.format(gpu))
#
# def replicationFunc(self, args):
#     self.demo(args)
#
# def run(self):
#     nCopies = len(self.replications)
#
#     nSamples = int(5000 / nCopies)
#     print('nSamples:[{}]'.format(nSamples))
#     args = ((r, nSamples) for r in self.replications)
#
#     with Pool(processes=nCopies, maxtasksperchild=1) as pool:
#         results = pool.map(self.replicationFunc, args)
#
#     return results

# # restore cModel original BNs
# def _restoreOriginalBNs(self):
#     for layer, (bn, widthList) in zip(self._cModel.layersList(), self._originalBNs):
#         layer.bn = bn
#         layer._widthList = widthList

# def _generatePathBNs(self, layerIdx: int):
#     model = self._cModel
#     # iterate over layers (except layerIdx) layer and generate new BNs
#     for idx, layer in enumerate(model.layersList()):
#         if idx == layerIdx:
#             continue
#
#         # get layer src BN based on currPathIdx
#         currBN = layer.bn[layer.currWidthIdx()]
#         bnFeatures = currBN.num_features
#         # generate layer new BNs ModuleList
#         newBNs = ModuleList([BatchNorm2d(bnFeatures) for _ in range(layer.nWidths())]).cuda()
#         # copy weights to new BNs
#         for bn in newBNs:
#             bn.load_state_dict(currBN.state_dict())
#         # set layer BNs
#         layer.bn = newBNs
#         # set layer width list to the same width
#         layer._widthList = [layer.currWidth()] * layer.nWidths()

# def jjj(self, args: tuple):
#     replica, data, nSamples = args
#
#     gpu = replica.gpu()
#     # switch to process GPU
#     set_device(gpu)
#
#     regime = replica._regime
#     model = regime.buildModel(regime.args)
#     model = model.cuda()
#     model.train()
#
#     from trainRegimes.PreTrainedRegime import PreTrainedTrainWeights
#     class YY(PreTrainedTrainWeights):
#         def __init__(self, regime, maxEpoch, model):
#             self._model = model
#             super(YY, self).__init__(regime, maxEpoch)
#
#         def getModel(self):
#             return self._model
#
#         def getModelParallel(self):
#             return self.getModel()
#
#         def widthList(self):
#             return {k: self.widthIdxList for k in range(4)}.items()
#
#     trainWeights = YY(regime, 5, model)
#     trainWeights.train('fiuiu')
