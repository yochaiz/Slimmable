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

        container = {self.alphaDistributionKey: self._containerPerAlpha(model)}
        # add loss average keys
        for k in lossClass.lossKeys():
            container[self.lossAvgTemplate.format(k)] = [{0: []}]
        # add loss variance keys
        container[self.lossVarianceTemplate.format(lossClass.totalKey())] = [{0: []}]

        return container

    def _calcAlphasDistribStats(self, model: BaseNet_Binomial):
        stats = self.statistics
        # add alphas distribution
        for layerIdx, layer in enumerate(model.layersList()):
            alphaTitle = self._alphaPlotTitle(layerIdx)
            stats.addValue(lambda containers: containers[self.alphaDistributionKey][layerIdx][alphaTitle], layer.probs().item())

    def _pathsListToRows(self, pathsList: list) -> list:
        pathsListRows = [['#', 'Paths']]
        for pathIdx, (path, lossDict) in enumerate(pathsList):
            pathsListRows.append([pathIdx + 1, [['Path', path], ['Loss', HtmlLogger.dictToRows(lossDict, nElementPerRow=2)]]])

        return pathsListRows

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
        for lossDict, partition in lossDictsPartitionList:
            # add lossDict to loss dicts list
            lossDictsList.append(lossDict)
            # sum loss by keys
            for k, v in lossDict.items():
                lossAvgDict[k] += v.item()
            # calc lossDict contribution to each layer alpha gradient
            for layerIdx, layer in enumerate(model.layersList()):
                alphasGrad[layerIdx] += ((partition[layerIdx] - layer.alphaWidthMean()) * lossDict[totalKey].item())

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

        # get statistics element with a shorter name
        stats = self.statistics
        # add values to statistics
        # init template for get list function based on container key
        getListFunc = lambda key: lambda containers: containers[key][0][0]
        # add loss average values to statistics
        for lossKey, lossAvg in lossAvgDict.items():
            stats.addValue(getListFunc(self.lossAvgTemplate.format(lossKey)), lossAvg)
        # add loss variance values to statistics
        stats.addValue(getListFunc(self.lossVarianceTemplate.format(totalKey)), lossVariance)

        return lossAvgDict
