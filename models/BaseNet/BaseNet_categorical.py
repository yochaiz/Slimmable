from .BaseNet import BaseNet
from ..ResNet18 import BasicBlock, ConvSlimLayer
from models.modules.Alphas import Alphas

from torch import tensor, zeros
from torch.nn.functional import softmax
from torch.distributions.categorical import Categorical


class ConvSlimLayerWithAlphas(ConvSlimLayer):
    def __init__(self, widthRatioList, out_planes, kernel_size, stride, prevLayer, countFlopsFlag):
        super(ConvSlimLayerWithAlphas, self).__init__(widthRatioList, out_planes, kernel_size, stride, prevLayer, countFlopsFlag)

        # init alphas
        self._alphas = zeros(self.nWidths()).cuda().clone().detach().requires_grad_(True)

    def flopsWidthList(self):
        return self.widthList()

    # returns alphas value
    def alphas(self) -> tensor:
        return self._alphas

    # return alphas probabilities
    def probs(self):
        return softmax(self._alphas, dim=-1).detach()

    # choose alpha based on alphas distribution
    def choosePathByAlphas(self):
        dist = Categorical(logits=self._alphas)
        chosen = dist.sample()
        self._currWidthIdx = chosen.item()

    # choose maximal alpha
    def chooseAlphaMax(self):
        self._currWidthIdx = self._alphas.argmax().item()


class BasicBlock_Categorical(BasicBlock):
    def __init__(self, widthRatioList, out_planes, kernel_size, stride, prevLayer, countFlopsFlag):
        super(BasicBlock_Categorical, self).__init__(widthRatioList, out_planes, kernel_size, stride, prevLayer, countFlopsFlag)

    @staticmethod
    def ConvSlimLayer() -> ConvSlimLayer:
        return ConvSlimLayerWithAlphas

    def updateCurrWidth(self):
        self.downsample.updateCurrWidth()


class BaseNet_Categorical(BaseNet):
    def __init__(self, args, initLayersParams):
        super(BaseNet_Categorical, self).__init__(args, initLayersParams)

    # choose alpha based on alphas distribution
    def choosePathByAlphas(self):
        for layer in self._layers.optimization():
            layer.choosePathByAlphas()

        # update curr width changes in each block
        for block in self.blocks:
            block.updateCurrWidth()

    # select maximal alpha in each layer
    def choosePathAlphasAsPartition(self):
        for layer in self._layers.optimization():
            layer.chooseAlphaMax()

        # update curr width changes in each block
        for block in self.blocks:
            block.updateCurrWidth()

    def restoreOriginalStateDictStructure(self):
        pass

    def initAlphas(self, saveFolder: str):
        return AlphasPerLayer(self, saveFolder)


class AlphasPerLayer(Alphas):
    def __init__(self, model: BaseNet_Categorical, saveFolder: str):
        super(AlphasPerLayer, self).__init__(model, saveFolder)

    def buildAlphas(self, model: BaseNet_Categorical):
        return [layer.alphas() for layer in model.layersList()]

    def initColumns(self, model: BaseNet_Categorical):
        return ['Layer_{}'.format(i) for i in range(len(model.layersList()))]

    def alphasValues(self, model: BaseNet_Categorical):
        return [[round(e.item(), self._roundDigits) for e in layer.alphas()] for layer in model.layersList()]

    def alphasList(self, model: BaseNet_Categorical):
        return [(layer.alphas(), layer.probs(), layer.widthList(), '{}'.format(layer)) for layer in model.layersList()]
