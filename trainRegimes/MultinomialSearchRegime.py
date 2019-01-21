from .SearchRegime import SearchRegime
from models.BaseNet.BaseNet_multinomial import BaseNet_Multinomial
from replicator.MultinomialReplicator import MultinomialReplicator
from scipy.stats import entropy
from torch import zeros


class MultinomialSearchRegime(SearchRegime):
    def __init__(self, args, logger):
        super(MultinomialSearchRegime, self).__init__(args, logger)

    def initReplicator(self) -> MultinomialReplicator:
        return MultinomialReplicator(self)

    def buildStatsContainers(self) -> dict:
        model = self.model
        lossClass = self.lossClass

        container = {self.alphaDistributionKey: self._containerPerAlpha(model),
                     self.entropyKey: [{0: []}]}
        # add loss average keys
        for k in lossClass.lossKeys():
            container[self.lossAvgTemplate.format(k)] = [{0: []}]
        # add loss variance keys
        container[self.lossVarianceTemplate.format(lossClass.totalKey())] = [{0: []}]

        return container

    def _pathsListToRows(self, pathsList: list) -> list:
        # add numbering to paths list
        pathsListRows = [['#', 'Path']] + [[idx + 1, v] for idx, v in enumerate(pathsList)]

        return pathsListRows

    def _containerPerAlpha(self, model: BaseNet_Multinomial) -> list:
        layer = model.layersList()[0]
        return [{self._alphaPlotTitle(layer, idx): [] for idx in range(len(alphas))} for alphas in model.alphas()]

    def _calcAlphasDistribStats(self, model: BaseNet_Multinomial):
        stats = self.statistics
        # get probs
        probs = model.probs().cpu()
        # add entropy to statistics
        stats.addValue(lambda containers: containers[self.entropyKey][0][0], entropy(probs))
        # add alphas distribution
        layer = model.layersList()[0]
        for alphaIdx, p in enumerate(probs):
            alphaTitle = self._alphaPlotTitle(layer, alphaIdx)
            stats.addValue(lambda containers: containers[self.alphaDistributionKey][0][alphaTitle], p.item())

    # updates alphas gradients
    # updates statistics
    def _updateAlphasGradients(self, lossDictsList: list) -> dict:
        model = self.model
        nSamples = self.args.nSamples
        totalKey = self.flopsLoss.totalKey()
        assert (len(lossDictsList) == nSamples)

        # init losses averages
        lossAvgDict = {k: 0.0 for k in self.flopsLoss.lossKeys()}
        # get statistics element with a shorter name
        stats = self.statistics

        # sum losses
        for lossDict, _ in lossDictsList:
            for k, v in lossDict.items():
                lossAvgDict[k] += v.item()
        # average losses
        for k in lossAvgDict.keys():
            lossAvgDict[k] /= nSamples

        # init total loss average
        lossAvg = lossAvgDict[totalKey]

        alphas = model.alphas()[0]
        probs = model.probs()
        nAlphas = len(alphas)
        # calc v1
        v1 = lossAvg * model.nLayers() * probs
        # calc v2
        v2 = zeros(nAlphas, requires_grad=True).cuda()
        for alphaIdx in range(nAlphas):
            # calc weighted loss average
            weightedLossAvg = 0.0
            for l, p in lossDictsList:
                weightedLossAvg += l[totalKey].item() * sum(1 for layerWidthIdx in p if layerWidthIdx == alphaIdx)
            weightedLossAvg /= nSamples
            v2[alphaIdx] = weightedLossAvg

        # update alphas grad = E[I_ni*Loss] - E[I_ni]*E[Loss] = v2 - v1
        alphas.grad = v2 - v1
        print('alphas gradient:{}'.format(alphas.grad.data))

        # calc loss variance
        lossVariance = [((x[totalKey].item() - lossAvg) ** 2) for x, _ in lossDictsList]
        lossVariance = sum(lossVariance) / (nSamples - 1)

        # add values to statistics
        # init template for get list function based on container key
        getListFunc = lambda key: lambda containers: containers[key][0][0]
        # add loss average values to statistics
        for lossKey, lossAvg in lossAvgDict.items():
            stats.addValue(getListFunc(self.lossAvgTemplate.format(lossKey)), lossAvg)
        # add loss variance values to statistics
        stats.addValue(getListFunc(self.lossVarianceTemplate.format(totalKey)), lossVariance)

        return lossAvgDict
