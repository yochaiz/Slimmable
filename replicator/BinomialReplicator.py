from replicator.MultinomialReplicator import MultinomialReplicator, MultinomialReplica


class BinomialReplicator(MultinomialReplicator):
    def __init__(self, regime):
        super(BinomialReplicator, self).__init__(regime)

    @staticmethod
    def _addLossDictFunc(cModel):
        def addLossDict(lossDict: dict, lossDictsList: list, widthRatio: float, trainedPathIdx: list):
            trainedPathWidth = cModel.currWidth()
            trainPathWidthRatio = cModel.currWidthRatio()

            diffList = []
            for layerIdx, layer in enumerate(cModel.layersList()):
                diff = trainedPathWidth[layerIdx] - layer.alphaWidthMean().item()
                diffList.append(diff)

            lossDictsList.append((lossDict, diffList, trainPathWidthRatio))

        return addLossDict

    @staticmethod
    def iterateOverSamples(replica: MultinomialReplica, lossFunc, data, pathsHistoryDict, lossDictsList, gpu: int):
        cModel = replica.getModel()

        addLossDictFunc = BinomialReplicator._addLossDictFunc(cModel)
        generateTrainParams = MultinomialReplicator.generateTrainParams

        BinomialReplicator.evaluateSample(replica, lossFunc, data, pathsHistoryDict, lossDictsList, generateTrainParams, addLossDictFunc)


class BlockBinomialReplicator(MultinomialReplicator):
    def __init__(self, regime):
        super(BlockBinomialReplicator, self).__init__(regime)

    @staticmethod
    def _addLossDictFunc(cModel):
        def addLossDict(lossDict: dict, lossDictsList: list, widthRatio: float, trainedPathIdx: list):
            trainedPathWidth = cModel.currWidth()
            trainPathWidthRatio = cModel.currWidthRatio()

            alphasDict = cModel.alphasDict()
            widthDiffDict = {}
            for width, alphaWidth in alphasDict.items():
                # take one of alpha layers index, in order to get actual alpha width in current partition
                layerIdx = alphaWidth.layersIdxList()[0]
                widthDiffDict[width] = trainedPathWidth[layerIdx] - alphaWidth.mean(width).item()

            lossDictsList.append((lossDict, widthDiffDict, trainPathWidthRatio))

        return addLossDict

    @staticmethod
    def iterateOverSamples(replica: MultinomialReplica, lossFunc, data, pathsHistoryDict, lossDictsList, gpu: int):
        cModel = replica.getModel()

        addLossDictFunc = BlockBinomialReplicator._addLossDictFunc(cModel)
        generateTrainParams = MultinomialReplicator.generateTrainParams

        BlockBinomialReplicator.evaluateSample(replica, lossFunc, data, pathsHistoryDict, lossDictsList, generateTrainParams, addLossDictFunc)
