from .BaseNet import BaseNet
from ..ResNet18 import BasicBlock
from models.modules.Alphas import Alphas
from models.modules.ConvSlimLayer import ConvSlimLayer, BatchNorm2d

from torch import tensor, zeros, sigmoid, int32
from torch import round as roundTensor
from torch.distributions.binomial import Binomial


class ConvSlimLayerWithAlpha(ConvSlimLayer):
    def __init__(self, widthRatioList, out_planes, kernel_size, stride, prevLayer, countFlopsFlag):
        super(ConvSlimLayerWithAlpha, self).__init__(widthRatioList, out_planes, kernel_size, stride, prevLayer, countFlopsFlag)

        # init alphas
        self._alphas = zeros(1).cuda().clone().detach().requires_grad_(True)
        # add additional BN for selected width
        self.bn.append(None)
        # add additional width & width ratio for selected width
        self._widthList.append(None)
        self._widthRatioList.append(None)

    def restoreOriginalStateDictStructure(self):
        self.bn[len(self.bn) - 1] = None
        self._widthList[-1] = None
        self._widthRatioList[-1] = None

    def flopsWidthList(self):
        return list(range(1, self.outputChannels() + 1))

    # returns alphas value
    def alphas(self) -> tensor:
        return self._alphas

    # return alphas probabilities
    def probs(self):
        return sigmoid(self._alphas).detach()

    # generate new BN for current width
    def generateWidthBN(self, width):
        self.bn[len(self.bn) - 1] = BatchNorm2d(width).cuda()
        self._widthList[-1] = width
        self._widthRatioList[-1] = width / self.outputChannels()

    # set newWidth as new current layer width
    def _setCurrWidth(self, newWidth):
        # set current width index
        self.setCurrWidthIdx(self.nWidths() - 1)
        # build last BN in self.bn according to newWidth
        self.generateWidthBN(newWidth)

    def _sampleWidthByAlphas(self):
        # define Binomial distribution on n-1 layer filters (because we have to choose at least one filter)
        dist = Binomial(self.outputChannels() - 1, logits=self._alphas)
        # draw from distribution
        width = 1 + dist.sample().type(int32).item()
        return width

    def alphaWidthMean(self):
        return 1 + (self.probs() * (self.outputChannels() - 1))

    # choose alpha based on alphas distribution
    def choosePathByAlphas(self):
        # sample width from the distribution
        newWidth = self._sampleWidthByAlphas()
        # set new width
        self._setCurrWidth(newWidth)

    # choose layer width based on alpha mean value
    def chooseAlphaMean(self):
        newWidth = roundTensor(self.alphaWidthMean()).type(int32).item()
        # set new width
        self._setCurrWidth(newWidth)


class BasicBlock_Binomial(BasicBlock):
    def __init__(self, widthRatioList, out_planes, kernel_size, stride, prevLayer, countFlopsFlag):
        super(BasicBlock_Binomial, self).__init__(widthRatioList, out_planes, kernel_size, stride, prevLayer, countFlopsFlag)

    @staticmethod
    def ConvSlimLayer() -> ConvSlimLayer:
        return ConvSlimLayerWithAlpha

    def updateCurrWidth(self):
        self.downsample.updateCurrWidth()
        # generate new BN for downsample layer
        _downsample = self.downsample.downsample()
        if _downsample is not None:
            _downsample.generateWidthBN(self.conv2.currWidth())


class BaseNet_Binomial(BaseNet):
    def __init__(self, args, initLayersParams):
        super(BaseNet_Binomial, self).__init__(args, initLayersParams)

    # generic method to choose layer path
    def _choosePath(self, chooseLayerPathFunc):
        for layer in self._layers.optimization():
            chooseLayerPathFunc(layer)

        # update curr width changes in each block
        for block in self.blocks:
            block.updateCurrWidth()

    # choose alpha based on alphas distribution
    def choosePathByAlphas(self):
        def chooseLayerPathFunc(layer: ConvSlimLayerWithAlpha):
            layer.choosePathByAlphas()

        self._choosePath(chooseLayerPathFunc)

    # select maximal alpha in each layer
    def choosePathAlphasAsPartition(self):
        def chooseLayerPathFunc(layer: ConvSlimLayerWithAlpha):
            layer.chooseAlphaMean()

        self._choosePath(chooseLayerPathFunc)

    def restoreOriginalStateDictStructure(self):
        for layer in self._layers.forwardCounters():
            layer.restoreOriginalStateDictStructure()

    def _alphasClass(self):
        return AlphaPerLayer


class AlphaPerLayer(Alphas):
    def __init__(self, model: BaseNet_Binomial, saveFolder: str):
        super(AlphaPerLayer, self).__init__(model, saveFolder)

    def buildAlphas(self, model: BaseNet_Binomial):
        return [layer.alphas() for layer in model.layersList()]

    def loadFromSource(self, model: BaseNet_Binomial, alphasSrc: list):
        for layer, layerAlphasSrc in zip(model.layersList(), alphasSrc):
            layerAlphas = layer.alphas()
            layerAlphas.data.copy_(layerAlphasSrc.data)

    def initColumns(self, model: BaseNet_Binomial):
        return ['Layer_{}'.format(i) for i in range(len(model.layersList()))]

    def alphasValues(self, model: BaseNet_Binomial):
        return [[round(e.item(), self._roundDigits) for e in layer.alphas()] for layer in model.layersList()]

    def alphasList(self, model: BaseNet_Binomial):
        return [(layer.alphas(), layer.probs(), ['{:.3f}'.format(layer.alphaWidthMean().item())], '{}'.format(layer)) for layer in model.layersList()]
