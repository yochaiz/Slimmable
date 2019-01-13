from math import floor
from time import time
from multiprocessing import Pool

from torch import tensor, zeros
from torch.cuda import set_device, current_device
from torch.utils.data.dataloader import DataLoader

from trainRegimes.regime import TrainRegime
from models.BaseNet import BaseNet, SlimLayer
from utils.trainWeights import TrainWeights


class Replica:
    def __init__(self, regime: TrainRegime, gpu: int):
        # set device to required gpu
        set_device(gpu)

        self._regime = regime
        self._gpu = gpu
        self._cModel = self._replicateModel(regime)
        self._trainWeights = None

    def gpu(self) -> int:
        return self._gpu

    def cModel(self) -> BaseNet:
        return self._cModel

    def regime(self) -> TrainRegime:
        return self._regime

    def trainQueue(self) -> DataLoader:
        return self._regime.train_queue

    @staticmethod
    def _updateWeights(srcModel: BaseNet, dstModel: BaseNet):
        dstModel.load_state_dict(srcModel.state_dict())

    def _replicateModel(self, regime: TrainRegime) -> BaseNet:
        # create model new instance
        cModel = regime.buildModel(regime.args)
        # set model to cuda on specific GPU
        cModel = cModel.cuda()
        # copy weights from regime model
        self._updateWeights(regime.model, cModel)
        # set mode to eval mode
        cModel.eval()

        return cModel

    def initNewEpoch(self, model: BaseNet):
        # update replica weights
        self._updateWeights(model, self.cModel())
        # init new TrainPathWeights instance
        self._trainWeights = TrainPathWeights(self)

    def train(self, layer: SlimLayer):
        self._trainWeights.train(layer)


class TrainPathWeights(TrainWeights):
    def __init__(self, replica: Replica):
        self._replica = replica

        super(TrainPathWeights, self).__init__(replica.regime())

        # save model weights
        self.modelOrgWeights = self.getModel().state_dict()
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

    def restoreModelOriginalWeights(self):
        self.getModel().load_state_dict(self.modelOrgWeights)

    def train(self, layer: SlimLayer):
        model = self.getModel()
        layerPaths = self.layerPaths
        # restore model original weights
        self.restoreModelOriginalWeights()
        # reset layer paths for training
        layerPaths.clear()
        # collect layer paths for training
        for idx in range(layer.nWidths()):
            # set path to go through width[idx] in current layer
            layer.setCurrWidthIdx(idx)
            # add path to dictionary
            layerPaths[layer.widthRatioByIdx(idx)] = model.currWidthIdx()

        # init optimizer
        optimizer = self._initOptimizer()
        # train
        epoch = 0
        startTime = time()
        while not self.stopCondition(epoch):
            epoch += 1
            self.weightsEpoch(optimizer, epoch, {})
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
        with Pool(processes=nCopies, maxtasksperchild=1) as pool:
            results = pool.map(self.lossPerReplication, args)

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
        pathsList = []

        # init containers to save loss values
        lossDictsList = [[{k: [] for k in self._flopsLoss.lossKeys()} for _ in range(layer.nWidths())] for layer in cModel.layersList()]

        # iterate over samples. generate a sample (path), train it and evaluate alphas on sample
        for sampleIdx in range(nSamples):
            print('===== Sample idx:[{}] - GPU:[{}] ====='.format(sampleIdx, gpu))
            # select new path based on alphas distribution.
            # check that selected path hasn't been selected before
            pathExists = True
            while pathExists:
                # select path based on alphas distribution
                cModel.choosePathByAlphas()
                # get selected path indices
                pathWidthIdx = cModel.currWidthIdx()
                # check that selected path hasn't been selected before
                pathExists = self._doesPathExist(pathsHistoryDict, pathWidthIdx)
            # add path to paths list
            pathsList.append([layer.widthByIdx(p) for p, layer in zip(pathWidthIdx, cModel.layersList())])

            # iterate over layers. in each layer iterate over alphas
            for layerIdx, layer in enumerate(cModel.layersList()):
                print('=== Layer idx:[{}] - GPU:[{}] ==='.format(layerIdx, gpu))
                # init containers to save loss values and variance
                layerLossDicts = lossDictsList[layerIdx]
                # save layer current width idx
                layerCurrWidthIdx = layer.currWidthIdx()
                # train model on layer path
                replica.train(layer)
                # switch to eval mode
                cModel.eval()
                # iterate over alphas and calc loss
                for idx in range(layer.nWidths()):
                    # set path to go through width[idx] in current layer
                    layer.setCurrWidthIdx(idx)
                    print(cModel.currWidthIdx())
                    # forward input in model selected path
                    logits = cModel(input)
                    # calc loss
                    lossDict = self._flopsLoss(logits, target, cModel.countFlops())
                    # add loss to container
                    alphaLossDict = layerLossDicts[idx]
                    for k, loss in lossDict.items():
                        alphaLossDict[k].append(loss.item())

                # restore layer current width idx
                layer.setCurrWidthIdx(layerCurrWidthIdx)

        return lossDictsList, pathsList

    def processResults(self, results: list) -> (list, list):
        lossDictsList, pathsList = results[0]

        # append lists from GPUs
        for gpuLossDictsList, gpuPathsList in results[1:]:
            # add paths to list
            pathsList.extend(gpuPathsList)
            # add loss values to dict
            for layerIdx, layerLossDicts in enumerate(gpuLossDictsList):
                for alphaIdx, alphaLossDict in enumerate(layerLossDicts):
                    for lossName, lossList in alphaLossDict.items():
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
