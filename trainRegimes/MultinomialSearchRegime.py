from .SearchRegime import SearchRegime, HtmlLogger
from models.BaseNet.BaseNet_multinomial import BaseNet_Multinomial
from replicator.MultinomialReplicator import MultinomialReplicator
from scipy.stats import entropy
from itertools import groupby
from torch import zeros


class MultinomialSearchRegime(SearchRegime):
    def __init__(self, args, logger):
        super(MultinomialSearchRegime, self).__init__(args, logger)

    def initReplicator(self) -> MultinomialReplicator:
        return MultinomialReplicator(self)

    def _containerPerAlpha(self, model: BaseNet_Multinomial) -> list:
        layer = model.layersList()[0]
        return [{self._alphaPlotTitle(layer, idx): [] for idx in range(len(alphas))} for alphas in model.alphas()]

    def _alphaGradTitle(self, layer, alphaIdx: int):
        return layer.widthRatioByIdx(alphaIdx)

    def buildStatsContainers(self) -> dict:
        model = self.model
        lossClass = self.lossClass

        container = {self.batchAlphaDistributionKey: self._containerPerAlpha(model),
                     self.entropyKey: [{0: []}]}
        # add loss average keys
        for k in lossClass.lossKeys():
            container[self.batchLossAvgTemplate.format(k)] = [{0: []}]
        # add loss variance keys
        container[self.batchLossVarianceTemplate.format(lossClass.totalKey())] = [{0: []}]

        return container

    def _pathsListToRows(self, pathsList: list) -> list:
        raise ValueError('_pathsListToRows() is not compatible with latest changes')
        pathsListRows = [['#', 'Paths']]
        for pathIdx, (path, lossDict) in enumerate(pathsList):
            pathsListRows.append([pathIdx + 1, [['Path', path], ['Loss', HtmlLogger.dictToRows(lossDict, nElementPerRow=2)]]])

        return pathsListRows

    def _calcAlphasDistribStats(self, model: BaseNet_Multinomial, alphaDistributionKey: str):
        stats = self.statistics
        # get probs
        probs = model.probs().cpu()
        # add entropy to statistics
        stats.addValue(lambda containers: containers[self.entropyKey][0][0], entropy(probs))
        # add alphas distribution
        layer = model.layersList()[0]
        for alphaIdx, p in enumerate(probs):
            alphaTitle = self._alphaPlotTitle(layer, alphaIdx)
            stats.addValue(lambda containers: containers[alphaDistributionKey][0][alphaTitle], p.item())

    # updates alphas gradients
    # updates statistics
    def _updateAlphasGradients(self, lossDictsPartitionList: list) -> dict:
        model = self.model
        nSamples = len(lossDictsPartitionList)
        totalKey = self.flopsLoss.totalKey()

        alphas = model.alphas()[0]
        probs = model.probs()
        nAlphas = len(alphas)

        # init loss dicts list
        lossDictsList = []
        # init losses averages
        lossAvgDict = {k: 0.0 for k in self.flopsLoss.lossKeys()}
        # calc v2
        v2 = zeros(nAlphas, requires_grad=True).cuda()
        for lossDict, partition in lossDictsPartitionList:
            # add lossDict to loss dicts list
            lossDictsList.append(lossDict)
            # sum loss by keys
            for k, v in lossDict.items():
                lossAvgDict[k] += v.item()
            # group alphas indices from partition
            groups = groupby(partition, key=lambda x: x)
            # sort groups size in a tensor
            partitionGroupsSize = zeros(nAlphas).cuda()
            for _, group in groups:
                group = list(group)
                if len(group) > 0:
                    partitionGroupsSize[group[0]] = len(group)
            # add weighted loss sum to v2
            v2 += (lossDict[totalKey].item() * partitionGroupsSize)

        # average weighted loss sum
        v2 /= nSamples
        # average losses
        for k in lossAvgDict.keys():
            lossAvgDict[k] /= nSamples

        # init total loss average
        lossAvg = lossAvgDict[totalKey]
        # calc v1
        v1 = lossAvg * model.nLayers() * probs
        # update alphas grad = E[I_ni*Loss] - E[I_ni]*E[Loss] = v2 - v1
        alphas.grad = v2 - v1

        # calc loss variance
        lossVariance = [((x[totalKey].item() - lossAvg) ** 2) for x in lossDictsList]
        lossVariance = sum(lossVariance) / (nSamples - 1)

        # add values to statistics
        # init template for get list function based on container key
        getListFunc = lambda key: lambda containers: containers[key][0][0]
        # add loss average values to statistics
        self._addValuesToStatistics(getListFunc, self.batchLossAvgTemplate, lossAvgDict)
        # add loss variance values to statistics
        self._addValuesToStatistics(getListFunc, self.batchLossVarianceTemplate, {totalKey: lossVariance})

        return lossAvgDict
