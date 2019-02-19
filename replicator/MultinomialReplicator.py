from .Replicator import ModelReplicator
from .Replica import MultinomialReplica


class MultinomialReplicator(ModelReplicator):
    def __init__(self, regime):
        super(MultinomialReplicator, self).__init__(regime)

    # def initPathsList(self) -> list:
    #     return []

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
    def iterateOverSamples(replica: MultinomialReplica, lossFunc, data, pathsHistoryDict, lossDictsList, gpu: int):
        generateTrainParams = MultinomialReplicator.generateTrainParams
        addLossDict = MultinomialReplicator.addLossDict

        ModelReplicator.evaluateSample(replica, lossFunc, data, pathsHistoryDict, lossDictsList, generateTrainParams, addLossDict)

    def processResults(self, results: list) -> list:
        lossDictsList = results[0]

        # sort loss dictionaries in lossDictsList by batch
        # each element in lossDictsList is a list of loss dictionaries of the same batch
        lossDictsList = [[lossDict] for lossDict in lossDictsList]

        # append lists from GPUs
        for gpuLossDictsList in results[1:]:
            # add loss dictionaries by batch
            assert (len(lossDictsList) == len(gpuLossDictsList))
            for batchLossDictsList, lossDict in zip(lossDictsList, gpuLossDictsList):
                batchLossDictsList.append(lossDict)

        return lossDictsList

# def processResults(self, results: list) -> (list, list):
#     lossDictsList, pathsList = results[0]
#
#     # append lists from GPUs
#     for gpuLossDictsList, gpuPathsList in results[1:]:
#         lossDictsList.extend(gpuLossDictsList)
#         pathsList.extend(gpuPathsList)
#
# return lossDictsList, pathsList
