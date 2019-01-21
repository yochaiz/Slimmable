from .Replicator import ModelReplicator
from .Replica import CategoricalReplica

from trainRegimes.SearchRegime import SearchRegime


class CategoricalReplicator(ModelReplicator):
    def __init__(self, regime: SearchRegime):
        super(CategoricalReplicator, self).__init__(regime)

    def initPathsList(self) -> list:
        return [[] for _ in self._model.layersList()]

    def initLossDictsList(self) -> list:
        flopsLoss = self._regime.flopsLoss
        return [[{k: [] for k in flopsLoss.lossKeys()} for _ in range(layer.nWidths())] for layer in self._model.layersList()]

    def replicaClass(self) -> CategoricalReplica:
        return CategoricalReplica

    @staticmethod
    def iterateOverSamples(replica: CategoricalReplica, lossFunc, data, pathsHistoryDict, pathsList, lossDictsList, gpu: int):
        cModel = replica.getModel()
        # iterate over layers. in each layer iterate over alphas
        for layerIdx, layer in enumerate(cModel.layersList()):
            print('=== Layer idx:[{}] - GPU:[{}] ==='.format(layerIdx, gpu))
            generateTrainParams = lambda pathWidthIdx: (layer, pathWidthIdx)

            def addLossDict(lossDict: dict, lossDictsList: list, widthRatio: float, trainedPathIdx: list):
                # get alpha index based on widthRatio
                alphaIdx = layer.widthRatioIdx(widthRatio)
                # add loss to container
                alphaLossDict = lossDictsList[layerIdx][alphaIdx]
                for k, loss in lossDict.items():
                    alphaLossDict[k].append(loss.item())

            ModelReplicator.evaluateSample(replica, lossFunc, data, pathsHistoryDict, pathsList[layerIdx], lossDictsList, generateTrainParams,
                                           addLossDict)

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
