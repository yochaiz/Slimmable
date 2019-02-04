from .BaseNet import BaseNet
from ..ResNet18 import BasicBlock, ConvSlimLayer
from models.modules.Alphas import Alphas

from torch import tensor, zeros, int32
from torch.nn.functional import softmax
from torch.distributions.multinomial import Multinomial


class ConvSlimLayerNoAlphas(ConvSlimLayer):
    def __init__(self, widthRatioList, in_planes, out_planes, kernel_size, stride, prevLayer=None):
        super(ConvSlimLayerNoAlphas, self).__init__(widthRatioList, in_planes, out_planes, kernel_size, stride, prevLayer)

    def flopsWidthList(self):
        return self.widthList()


class BasicBlock_Multinomial(BasicBlock):
    def __init__(self, widthRatioList, in_planes, out_planes, kernel_size, stride, prevLayer=None):
        super(BasicBlock_Multinomial, self).__init__(widthRatioList, in_planes, out_planes, kernel_size, stride, prevLayer)

    @staticmethod
    def ConvSlimLayer() -> ConvSlimLayer:
        return ConvSlimLayerNoAlphas

    def updateCurrWidth(self):
        self.downsample.updateCurrWidth()


class BaseNet_Multinomial(BaseNet):
    def __init__(self, args, initLayersParams):
        super(BaseNet_Multinomial, self).__init__(args, initLayersParams)

    # return alphas probabilities
    def probs(self):
        return self._alphas.probs()

    def _setPartitionPath(self, partition: tensor):
        assert (partition.sum().item() == self.nLayers())
        # convert partition to idxList
        idxList = []
        for alphaIdx, nLayers in enumerate(partition):
            for _ in range(nLayers):
                idxList.append(alphaIdx)
        # set model path
        self.setCurrWidthIdx(idxList)

    # choose alpha based on alphas distribution
    def choosePathByAlphas(self):
        alphas = self.alphas()[0]
        # draw partition from multinomial distribution
        dist = Multinomial(total_count=self.nLayers(), logits=alphas)
        partition = dist.sample().type(int32)
        # set partition as model path
        self._setPartitionPath(partition)

    # choose partition based on alphas probs as partition
    def choosePathAlphasAsPartition(self):
        probs = self.probs()
        # calc partition from probs
        partition = (probs * self.nLayers()).type(int32)
        # add diff to lower alphas
        diff = self.nLayers() - partition.sum().item()
        for idx in range(diff):
            partition[idx] += 1
        # set partition as model path
        self._setPartitionPath(partition)

    def restoreOriginalStateDictStructure(self):
        pass

    def initAlphas(self, saveFolder: str):
        return AlphasPerModel(self, saveFolder)


class AlphasPerModel(Alphas):
    def __init__(self, model: BaseNet_Multinomial, saveFolder: str):
        super(AlphasPerModel, self).__init__(model, saveFolder)

    def buildAlphas(self, model: BaseNet_Multinomial):
        nWidths = len(model.layersList()[0].widthList())
        return [zeros(nWidths).cuda().clone().detach().requires_grad_(True)]

    def initColumns(self, model: BaseNet_Multinomial):
        return [self._alphasKey]

    def alphasValues(self, model: BaseNet_Multinomial):
        return [[round(e.item(), self._roundDigits) for e in self._alphas[0]]]

    def probs(self):
        return softmax(self._alphas[0], dim=-1).detach()

    def alphasList(self, model: BaseNet_Multinomial):
        widthList = model.layersList()[0].widthList()
        return [(self._alphas[0], self.probs(), widthList, None)]
