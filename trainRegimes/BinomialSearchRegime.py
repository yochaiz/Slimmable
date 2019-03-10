from .SearchRegime import SearchRegime, EpochTrainWeights, HtmlLogger
from models.BaseNet.BaseNet_binomial import BaseNet_Binomial
from replicator.BinomialReplicator import BinomialReplicator
from torch import zeros


class BinomialTrainWeights(EpochTrainWeights):
    def __init__(self, getModel, getModelParallel, getArgs, getLogger, getTrainQueue, getValidQueue, getTrainFolderPath, maxEpoch, currEpoch):
        super(BinomialTrainWeights, self).__init__(getModel, getModelParallel, getArgs, getLogger, getTrainQueue, getValidQueue, getTrainFolderPath,
                                                   maxEpoch, currEpoch)

        model = getModel()
        # init training width list with baseline widths
        self._widthList = {k: v for k, v in model.baselineWidth()}
        # set new path to alpha mean
        # we train only on this path, we don't select path in each epoch
        model.choosePathAlphasAsPartition()
        # add new path indices to training width list
        self._widthList[self.pathKey] = model.currWidthIdx()

    def widthList(self):
        return self._widthList.items()

    def _selectNewPath(self):
        pass


class BinomialSearchRegime(SearchRegime):
    def __init__(self, args, logger):
        super(BinomialSearchRegime, self).__init__(args, logger)

    def TrainWeightsClass(self):
        return BinomialTrainWeights

    def initReplicator(self) -> BinomialReplicator:
        return BinomialReplicator(self)

    def _alphaPlotTitle(self, layerIdx):
        layer = self.model.layersList()[layerIdx]
        return '{}'.format(layer.toStr())

    def _containerPerAlpha(self, model: BaseNet_Binomial) -> list:
        return [{self._alphaPlotTitle(layerIdx): []} for layerIdx in range(len(model.alphas()))]

    def _alphaGradTitle(self, layer, alphaIdx: int):
        return alphaIdx

    def buildStatsContainers(self) -> dict:
        model = self.model
        lossClass = self.lossClass

        container = {self.batchAlphaDistributionKey: self._containerPerAlpha(model),
                     self.epochAlphaDistributionKey: self._containerPerAlpha(model)}
        # add loss average keys
        for k in lossClass.lossKeys():
            container[self.batchLossAvgTemplate.format(k)] = [{0: []}]
            container[self.epochLossAvgTemplate.format(k)] = [{0: []}]
        # add loss variance keys
        container[self.batchLossVarianceTemplate.format(lossClass.totalKey())] = [{0: []}]

        return container

    def _calcAlphasDistribStats(self, model: BaseNet_Binomial, alphaDistributionKey: str):
        stats = self.statistics
        # add alphas distribution
        for layerIdx, layer in enumerate(model.layersList()):
            alphaTitle = self._alphaPlotTitle(layerIdx)
            stats.addValue(lambda containers: containers[alphaDistributionKey][layerIdx][alphaTitle], layer.probs().item())

    def _pathsListToRows(self, batchLossDictsList: list) -> list:
        pathsListRows = [['#', 'Paths']]
        for pathIdx, (lossDict, widthDiffDict, partitionRatio) in enumerate(batchLossDictsList):
            # build formatted loss dict
            formattedLossDict = {k: '{:.3f}'.format(v) for k, v in lossDict.items()}
            pathsListRows.append([pathIdx + 1, [['Path', partitionRatio], ['Loss', self.formats[self.trainLossKey](formattedLossDict)]]])

        return pathsListRows

    def _getListFunc(self):
        return lambda key: lambda containers: containers[key][0][0]

    def _updateEpochLossStats(self, epochLossDict: dict):
        self._addValuesToStatistics(self._getListFunc(), self.epochLossAvgTemplate, epochLossDict)

    # updates alphas gradients
    # updates statistics
    def _updateAlphasGradients(self, lossDictsPartitionList: list) -> dict:
        model = self.model
        nSamples = len(lossDictsPartitionList)
        totalKey = self.flopsLoss.totalKey()

        # get model alphas
        alphas = model.alphas()
        nAlphas = len(alphas)
        # init loss dicts list
        lossDictsList = []
        # init losses averages
        lossAvgDict = {k: 0.0 for k in self.flopsLoss.lossKeys()}
        # init model alphas gradient tensor
        alphasGrad = [zeros(1, requires_grad=True).cuda() for _ in range(nAlphas)]
        # iterate over losses
        for lossDict, diffList, partitionRatio in lossDictsPartitionList:
            # add lossDict to loss dicts list
            lossDictsList.append(lossDict)
            # sum loss by keys
            for k, v in lossDict.items():
                lossAvgDict[k] += v.item()
            # calc lossDict contribution to each layer alpha gradient
            assert (len(alphasGrad) == len(diffList))
            for layerIdx, diff in enumerate(diffList):
                alphasGrad[layerIdx] += (diff * lossDict[totalKey].item())

        # average gradient and put in layer.alpha.grad
        assert (len(alphas) == len(alphasGrad))
        for alpha, alphaGrad in zip(alphas, alphasGrad):
            alpha.grad = (alphaGrad / nSamples)
        # update gradient
        # average losses
        for k in lossAvgDict.keys():
            lossAvgDict[k] /= nSamples

        # init total loss average
        lossAvg = lossAvgDict[totalKey]
        # calc loss variance
        lossVariance = [((x[totalKey].item() - lossAvg) ** 2) for x in lossDictsList]
        lossVariance = sum(lossVariance) / (nSamples - 1)

        # add values to statistics
        # init template for get list function based on container key
        getListFunc = self._getListFunc()
        # add loss average values to statistics
        self._addValuesToStatistics(getListFunc, self.batchLossAvgTemplate, lossAvgDict)
        # add loss variance values to statistics
        self._addValuesToStatistics(getListFunc, self.batchLossVarianceTemplate, {totalKey: lossVariance})

        return lossAvgDict
