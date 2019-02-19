from replicator.MultinomialReplicator import MultinomialReplicator, MultinomialReplica


class BinomialReplicator(MultinomialReplicator):
    def __init__(self, regime):
        super(BinomialReplicator, self).__init__(regime)

    @staticmethod
    def iterateOverSamples(replica: MultinomialReplica, lossFunc, data, pathsHistoryDict, lossDictsList, gpu: int):
        cModel = replica.getModel()

        def addLossDict(lossDict: dict, lossDictsList: list, widthRatio: float, trainedPathIdx: list):
            trainedPathWidth = cModel.currWidth()
            trainPathWidthRatio = cModel.currWidthRatio()
            print('trainedPathWidth:{}'.format(trainedPathWidth))
            lossDictsList.append((lossDict, trainedPathWidth, trainPathWidthRatio))

        generateTrainParams = MultinomialReplicator.generateTrainParams

        BinomialReplicator.evaluateSample(replica, lossFunc, data, pathsHistoryDict, lossDictsList, generateTrainParams, addLossDict)
