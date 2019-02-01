from .Replicator import ModelReplicator
from .Replica import MultinomialReplica


class MultinomialReplicator(ModelReplicator):
    def __init__(self, regime):
        super(MultinomialReplicator, self).__init__(regime)

    def initPathsList(self) -> list:
        return []

    def initLossDictsList(self) -> list:
        return []

    @staticmethod
    def generateTrainParams(pathWidthIdx):
        return pathWidthIdx

    def replicaClass(self) -> MultinomialReplica:
        return MultinomialReplica

    @staticmethod
    def addLossDict(lossDict: dict, lossDictsList: list, widthRatio: float, trainedPathIdx: list):
        print('widthRatio:[{}] - trainedPathIdx:{}'.format(widthRatio, trainedPathIdx))
        lossDictsList.append((lossDict, trainedPathIdx))

    @staticmethod
    def iterateOverSamples(replica: MultinomialReplica, lossFunc, data, pathsHistoryDict, pathsList, lossDictsList, gpu: int):
        generateTrainParams = MultinomialReplicator.generateTrainParams
        addLossDict = MultinomialReplicator.addLossDict

        ModelReplicator.evaluateSample(replica, lossFunc, data, pathsHistoryDict, pathsList, lossDictsList,
                                       generateTrainParams, addLossDict)

    def processResults(self, results: list) -> (list, list):
        lossDictsList, pathsList = results[0]

        # append lists from GPUs
        for gpuLossDictsList, gpuPathsList in results[1:]:
            lossDictsList.extend(gpuLossDictsList)
            pathsList.extend(gpuPathsList)

        return lossDictsList, pathsList
