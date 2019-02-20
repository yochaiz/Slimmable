from .SearchRegime import SearchRegime, HtmlLogger
from models.BaseNet.BaseNet_categorical import BaseNet_Categorical
from replicator.CategoricalReplicator import CategoricalReplicator
from scipy.stats import entropy
from torch import zeros


class CategoricalSearchRegime(SearchRegime):
    def __init__(self, args, logger):
        super(CategoricalSearchRegime, self).__init__(args, logger)

    def initReplicator(self) -> CategoricalReplicator:
        return CategoricalReplicator(self)

    def buildStatsContainers(self) -> dict:
        model = self.model
        lossClass = self.lossClass

        container = {self.batchAlphaDistributionKey: self._containerPerAlpha(model),
                     self.entropyKey: [{layerIdx: [] for layerIdx in range(len(model.layersList()))}]}
        # add loss average keys
        for k in lossClass.lossKeys():
            container[self.batchLossAvgTemplate.format(k)] = self._containerPerAlpha(model)
        # add loss variance keys
        container[self.batchLossVarianceTemplate.format(lossClass.totalKey())] = self._containerPerAlpha(model)

        return container

    def _pathsListToRows(self, pathsList: list) -> list:
        raise ValueError('_pathsListToRows() is not compatible with latest changes')
        # add numbering to paths list
        pathsListRows = [['Layer #', 'Paths']]
        for layerIdx, layerPaths in enumerate(pathsList):
            layerRows = []
            for pathIdx, (path, lossDict) in enumerate(layerPaths):
                layerRows.append([pathIdx + 1, [['Path', path], ['Loss', HtmlLogger.dictToRows(lossDict, nElementPerRow=2)]]])
            # add layer paths to table
            pathsListRows.append([layerIdx, layerRows])

        return pathsListRows

    def _containerPerAlpha(self, model: BaseNet_Categorical) -> list:
        return [{self._alphaPlotTitle(layer, idx): [] for idx in range(layer.nWidths())} for layer in model.layersList()]

    def _alphaGradTitle(self, layer, alphaIdx: int):
        return layer.widthRatioByIdx(alphaIdx)

    def _calcAlphasDistribStats(self, model: BaseNet_Categorical, alphaDistributionKey: str):
        stats = self.statistics
        for layerIdx, layer in enumerate(model.layersList()):
            # calc layer alphas distribution
            probs = layer.probs().cpu()
            # add entropy to statistics
            stats.addValue(lambda containers: containers[self.entropyKey][0][layerIdx], entropy(probs))
            # add alphas distribution
            for alphaIdx, p in enumerate(probs):
                alphaTitle = self._alphaPlotTitle(layer, alphaIdx)
                stats.addValue(lambda containers: containers[alphaDistributionKey][layerIdx][alphaTitle], p.item())

    # updates alphas gradients
    # updates statistics
    def _updateAlphasGradients(self, lossDictsList: list) -> dict:
        model = self.model
        totalKey = self.flopsLoss.totalKey()

        # init total loss
        totalLoss = 0.0
        # init losses averages
        lossAvgDict = {k: 0.0 for k in self.flopsLoss.lossKeys()}
        # count how many alphas we have sum their loss average
        nAlphas = 0
        # init model probs list for gradient calcs
        probsList = []
        # after we finished iterating over samples, we can calculate loss average & variance for each alpha
        for layerIdx, layer in enumerate(model.layersList()):
            layerLossDicts = lossDictsList[layerIdx]
            # get layer alphas probabilities
            layerProbs = layer.probs()
            # add to model probs list
            probsList.append(layerProbs)
            # init layer alphas gradient vector
            layerAlphasGrad = zeros(layer.nWidths(), requires_grad=True).cuda()
            # iterate over alphas
            for idx, alphaLossDict in enumerate(layerLossDicts):
                alphaLossAvgDict = {}
                for k, lossList in alphaLossDict.items():
                    # calc loss list average
                    alphaLossAvgDict[k] = sum(lossList) / len(lossList)
                    # add loss average to total loss average
                    lossAvgDict[k] += alphaLossAvgDict[k]
                # update number of alphas summed into lossAvgDict
                nAlphas += 1
                # set total loss average
                alphaLossAvg = alphaLossAvgDict[totalKey]
                # update alpha gradient
                layerAlphasGrad[idx] = alphaLossAvg
                # update total loss
                totalLoss += (alphaLossAvg * layerProbs[idx])
                # calc alpha loss variance
                alphaLossVariance = [((x - alphaLossAvg) ** 2) for x in alphaLossDict[totalKey]]
                alphaLossVariance = sum(alphaLossVariance) / (len(alphaLossVariance) - 1)
                # add values to statistics
                alphaTitle = self._alphaPlotTitle(layer, idx)
                # init template for get list function based on container key
                getListFunc = lambda key: lambda containers: containers[key][layerIdx][alphaTitle]
                # add loss average values to statistics
                self._addValuesToStatistics(getListFunc, self.batchLossAvgTemplate, alphaLossAvgDict)
                # add loss variance values to statistics
                self._addValuesToStatistics(getListFunc, self.batchLossVarianceTemplate, {totalKey: alphaLossVariance})

            # update layer alphas gradient
            layerAlphas = layer.alphas()
            layerAlphas.grad = layerAlphasGrad

        # average total loss
        totalLoss /= len(model.layersList())
        # subtract average total loss from every alpha gradient
        for layerIdx, (layer, layerProbs) in enumerate(zip(model.layersList(), probsList)):
            layerAlphas = layer.alphas()
            layerAlphas.grad -= totalLoss
            # multiply each grad by its probability
            layerAlphas.grad *= layerProbs

        # average (total loss average) by number of alphas
        for k in lossAvgDict.keys():
            lossAvgDict[k] /= nAlphas

        return lossAvgDict
