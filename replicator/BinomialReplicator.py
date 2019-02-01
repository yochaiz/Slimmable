from replicator.MultinomialReplicator import MultinomialReplicator, MultinomialReplica


class BinomialReplicator(MultinomialReplicator):
    def __init__(self, regime):
        super(BinomialReplicator, self).__init__(regime)

    @staticmethod
    def iterateOverSamples(replica: MultinomialReplica, lossFunc, data, pathsHistoryDict, pathsList, lossDictsList, gpu: int):
        cModel = replica.getModel()

        def addLossDict(lossDict: dict, lossDictsList: list, widthRatio: float, trainedPathIdx: list):
            trainedPathWidth = [layer.widthByIdx(trainedPathIdx[layerIdx]) for layerIdx, layer in enumerate(cModel.layersList())]
            print('widthRatio:[{}] - trainedPathWidth:{}'.format(widthRatio, trainedPathWidth))
            lossDictsList.append((lossDict, trainedPathWidth))

        generateTrainParams = MultinomialReplicator.generateTrainParams

        BinomialReplicator.evaluateSample(replica, lossFunc, data, pathsHistoryDict, pathsList, lossDictsList,
                                          generateTrainParams, addLossDict)
