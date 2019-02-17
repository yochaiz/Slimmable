from .BinomialSearchRegime import BinomialSearchRegime, zeros
from models.BaseNet.BaseNet_widthblock_binomial import BaseNet_WidthBlock_Binomial


class WidthBlockBinomialSearchRegime(BinomialSearchRegime):
    def __init__(self, args, logger):
        super(WidthBlockBinomialSearchRegime, self).__init__(args, logger)

    def _alphaPlotTitle(self, width: int):
        return 'Width:[{}]'.format(width)

    def _containerPerAlpha(self, model: BaseNet_WidthBlock_Binomial) -> list:
        containersList = []
        for width, alphaWidth in model.alphasDict().items():
            container = {self._alphaPlotTitle(width): []}
            containersList.append(container)
            alphaWidth.setContainer(container)

        return containersList

    def _calcAlphasDistribStats(self, model: BaseNet_WidthBlock_Binomial):
        stats = self.statistics
        # add alphas distribution
        for width, alphaWidth in model.alphasDict().items():
            alphaTitle = self._alphaPlotTitle(width)
            stats.addValue(lambda containers: alphaWidth.container[alphaTitle], alphaWidth.prob.item())

    # updates alphas gradients
    # updates statistics
    def _updateAlphasGradients(self, lossDictsPartitionList: list) -> dict:
        model: BaseNet_WidthBlock_Binomial = self.model
        nSamples = self.args.nSamples
        totalKey = self.flopsLoss.totalKey()
        assert (len(lossDictsPartitionList) == nSamples)

        # get model alphas
        # alphas = model.alphas()
        alphasDict = model.alphasDict()
        nAlphas = len(alphasDict.keys())
        # init loss dicts list
        lossDictsList = []
        # init losses averages
        lossAvgDict = {k: 0.0 for k in self.flopsLoss.lossKeys()}
        # init model alphas gradient tensor
        alphasGrad = {width: zeros(1, requires_grad=True).cuda() for width in alphasDict.keys()}
        # iterate over losses
        for lossDict, partition in lossDictsPartitionList:
            # add lossDict to loss dicts list
            lossDictsList.append(lossDict)
            # sum loss by keys
            for k, v in lossDict.items():
                lossAvgDict[k] += v.item()
            # calc lossDict contribution to each layer alpha gradient
            for width, alphaWidth in alphasDict.items():
                # take one of alpha layers index, in order to get actual alpha width in current partition
                layerIdx = alphaWidth.layersIdxList[0]
                # add element to alpha gradient
                alphasGrad[width] += ((partition[layerIdx] - alphaWidth.mean(width)) * lossDict[totalKey].item())

        # average gradient and put in layer.alpha.grad
        assert (nAlphas == len(alphasGrad.keys()))
        for width, alphaWidth in alphasDict.items():
            alphaTensor = alphaWidth.tensor
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
