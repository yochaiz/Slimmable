from .BaseNet import BaseNet
from .BaseNet_binomial import ConvSlimLayer, BinomialConvSlimLayer, BasicBlock
from models.modules.Alphas import Alphas

from torch import zeros, sigmoid, int32, tensor
from torch.distributions.binomial import Binomial


# WidthBlock Binomial is a model where we sample from Binomial distribution for each WidthBlock
# WidthBlock is a group of consecutive blocks who have the same number of output channels, i.e. 16, 32, 64, etc.

class BasicBlock_WidthBlock_Binomial(BasicBlock):
    def __init__(self, widthRatioList, out_planes, kernel_size, stride, prevLayer, countFlopsFlag):
        super(BasicBlock_WidthBlock_Binomial, self).__init__(widthRatioList, out_planes, kernel_size, stride, prevLayer, countFlopsFlag)

    @staticmethod
    def ConvSlimLayer() -> ConvSlimLayer:
        return BinomialConvSlimLayer

    def updateCurrWidth(self):
        self.downsample.updateCurrWidth()
        # generate new BN for downsample layer
        _downsample = self.downsample.downsample()
        if _downsample is not None:
            _downsample.generateWidthBN(self.conv2.currWidth())


class BaseNet_WidthBlock_Binomial(BaseNet):
    def __init__(self, args, initLayersParams):
        super(BaseNet_WidthBlock_Binomial, self).__init__(args, initLayersParams)

    def alphasDict(self):
        return self._alphas.alphasDict

    def _choosePath(self, calcNewWidthFunc: callable):
        for width, alphaWidth in self._alphas.alphasDict.items():
            # calc new width
            newWidth = calcNewWidthFunc(width, alphaWidth)
            # apply new width to layers
            for layer in alphaWidth.layersList:
                layer.setCurrWidth(newWidth)

        # update curr width changes in each block
        for block in self.blocks:
            block.updateCurrWidth()

    # choose alpha based on alphas distribution
    def choosePathByAlphas(self):
        def calcNewWidthFunc(width: int, alphaWidth: AlphaPerWidthBlock.AlphaWidth):
            # define Binomial distribution on n-1 layer filters (because we have to choose at least one filter)
            dist = Binomial(width - 1, logits=alphaWidth.tensor)
            # draw from distribution
            return 1 + dist.sample().type(int32).item()

        self._choosePath(calcNewWidthFunc)

    # choose partition based on alphas probs as partition
    def choosePathAlphasAsPartition(self):
        def calcNewWidthFunc(width: int, alphaWidth: AlphaPerWidthBlock.AlphaWidth):
            return round(alphaWidth.mean(width).item())

        self._choosePath(calcNewWidthFunc)

    def restoreOriginalStateDictStructure(self):
        for layer in self._layers.forwardCounters():
            layer.restoreOriginalStateDictStructure()

    def _alphasClass(self):
        return AlphaPerWidthBlock


class AlphaPerWidthBlock(Alphas):
    class AlphaWidth:
        def __init__(self, _tensor: tensor):
            self._tensor = _tensor
            self._layers = []
            self._layersIdx = []
            self._container = None

        def addLayer(self, layer, layerIdx):
            self._layers.append(layer)
            self._layersIdx.append(layerIdx)

        def setContainer(self, _container):
            self._container = _container

        @property
        def tensor(self):
            return self._tensor

        @property
        def container(self):
            return self._container

        @property
        def prob(self):
            return sigmoid(self._tensor).detach()

        def mean(self, width: int):
            return 1 + (self.prob * (width - 1))

        @property
        def layersList(self):
            return self._layers

        @property
        def layersIdxList(self):
            return self._layersIdx

    def __init__(self, model: BaseNet_WidthBlock_Binomial, saveFolder: str):
        super(AlphaPerWidthBlock, self).__init__(model, saveFolder)

    @property
    def alphasDict(self):
        return self._alphasDict

    def buildAlphas(self, model: BaseNet_WidthBlock_Binomial):
        # init alphas list
        _alphas = []
        # init AlphaWidth dictionary. key is layer width, value is AlphaWidth instance
        _alphasDict = {}

        for layerIdx, layer in enumerate(model.layersList()):
            width = layer.outputChannels()
            if width not in _alphasDict:
                _tensor = zeros(1).cuda().clone().detach().requires_grad_(True)
                _alphas.append(_tensor)
                _alphasDict[width] = self.AlphaWidth(_tensor)

            # add layer to _alphasLayers
            _alphasDict[width].addLayer(layer, layerIdx)

        self._alphasDict = _alphasDict
        return _alphas

    def initColumns(self, model: BaseNet_WidthBlock_Binomial):
        return ['Block_[{}]'.format(width) for width in self._alphasDict.keys()]

    def alphasValues(self, model: BaseNet_WidthBlock_Binomial):
        return [round(e.item(), self._roundDigits) for e in self._alphas]

    def alphasList(self, model: BaseNet_WidthBlock_Binomial):
        _alphasList = []
        for width, alphaWidth in self._alphasDict.items():
            _data = (alphaWidth.tensor, alphaWidth.prob, ['{:.3f}'.format(alphaWidth.mean(width).item())], 'Block width:[{}]'.format(width))
            _alphasList.append(_data)

        return _alphasList
