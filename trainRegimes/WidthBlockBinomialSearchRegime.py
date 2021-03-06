from .BinomialSearchRegime import BinomialSearchRegime, zeros
from replicator.BinomialReplicator import BlockBinomialReplicator
from models.BaseNet.BaseNet_widthblock_binomial import BaseNet_WidthBlock_Binomial


class WidthBlockBinomialSearchRegime(BinomialSearchRegime):
    def __init__(self, args, logger):
        super(WidthBlockBinomialSearchRegime, self).__init__(args, logger)

    def initReplicator(self) -> BlockBinomialReplicator:
        return BlockBinomialReplicator(self)

    def _alphaPlotTitle(self, width: int):
        return 'Width:[{}]'.format(width)

    def _containerPerAlpha(self, model: BaseNet_WidthBlock_Binomial, alphaDistributionKey: str) -> list:
        # containersList is a list of containers per block
        containersList = []
        for width, alphaWidth in model.alphasDict().items():
            # create block container
            container = {self._alphaPlotTitle(width): []}
            # add block container to containersList
            containersList.append(container)
            # attach container to alpha (block)
            alphaWidth.addContainer(alphaDistributionKey, container)

        return containersList

    def buildStatsContainers(self) -> dict:
        model = self.model
        lossClass = self.lossClass

        container = {self.batchAlphaDistributionKey: self._containerPerAlpha(model, self.batchAlphaDistributionKey),
                     self.epochAlphaDistributionKey: self._containerPerAlpha(model, self.epochAlphaDistributionKey)}
        # add loss average keys
        for k in lossClass.lossKeys():
            container[self.batchLossAvgTemplate.format(k)] = [{0: []}]
            container[self.epochLossAvgTemplate.format(k)] = [{0: []}]
        # add loss variance keys
        container[self.batchLossVarianceTemplate.format(lossClass.totalKey())] = [{0: []}]

        return container

    def _calcAlphasDistribStats(self, model: BaseNet_WidthBlock_Binomial, alphaDistributionKey: str):
        stats = self.statistics
        # add alphas distribution
        for width, alphaWidth in model.alphasDict().items():
            alphaTitle = self._alphaPlotTitle(width)
            stats.addValue(lambda containers: alphaWidth.container()[alphaDistributionKey][alphaTitle], alphaWidth.prob().item())

    # updates alphas gradients
    # updates statistics
    def _updateAlphasGradients(self, lossDictsPartitionList: list) -> dict:
        model = self.model
        totalKey = self.flopsLoss.totalKey()
        nSamples = len(lossDictsPartitionList)

        # get model alphas
        alphasDict = model.alphasDict()
        nAlphas = len(alphasDict.keys())
        # init loss dicts list
        lossDictsList = []
        # init losses averages
        lossAvgDict = {k: 0.0 for k in self.flopsLoss.lossKeys()}
        # init model alphas gradient tensor
        alphasGrad = {width: zeros(1, requires_grad=True).cuda() for width in alphasDict.keys()}
        # iterate over losses
        for lossDict, widthDiffDict, partitionRatio in lossDictsPartitionList:
            # add lossDict to loss dicts list
            lossDictsList.append(lossDict)
            # sum loss by keys
            for k, v in lossDict.items():
                lossAvgDict[k] += v.item()
            # calc lossDict contribution to each layer alpha gradient
            for width, widthDiff in widthDiffDict.items():
                # add element to alpha gradient
                alphasGrad[width] += (widthDiff * lossDict[totalKey].item())

        # average gradient and put in layer.alpha.grad
        assert (nAlphas == len(alphasGrad.keys()))
        for width, alphaWidth in alphasDict.items():
            alphaTensor = alphaWidth.tensor()
            alphaTensor.grad = (alphasGrad[width] / nSamples)

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
        getListFunc = lambda key: lambda containers: containers[key][0][0]
        # add loss average values to statistics
        self._addValuesToStatistics(getListFunc, self.batchLossAvgTemplate, lossAvgDict)
        # add loss variance values to statistics
        self._addValuesToStatistics(getListFunc, self.batchLossVarianceTemplate, {totalKey: lossVariance})

        return lossAvgDict
