from abc import abstractmethod
from math import floor

from torch.nn import Module, ModuleList, Conv2d, BatchNorm2d
from torch.nn.functional import conv2d

from utils.HtmlLogger import HtmlLogger


# abstract class for model block
class Block(Module):
    @abstractmethod
    def getLayers(self):
        raise NotImplementedError('subclasses must override getLayers()!')


# abstract class for model layer
class Layer(Block):
    def __init__(self, nFiltersList):
        super(Layer, self).__init__()

        # save list of number of filters
        self.nFiltersList = nFiltersList
        # init current number of filters index
        self.nFiltersCurrIdx = 0

    def getCurrFilterIdx(self):
        return self.nFiltersCurrIdx

    @abstractmethod
    def getAllWidths(self):
        raise NotImplementedError('subclasses must override getAllWidths()!')

    @abstractmethod
    def nFilters(self):
        raise NotImplementedError('subclasses must override nFilters()!')


class ConvLayer(Layer):
    def __init__(self, widthRatioList, in_planes, out_planes, kernel_size, stride):
        super(ConvLayer, self).__init__([int(x * out_planes) for x in widthRatioList])

        # init conv2d module
        self.conv = Conv2d(in_planes, out_planes, kernel_size, stride=stride, padding=floor(kernel_size / 2), bias=False)
        # init independent batchnorm module for number of filters
        self.bn = ModuleList([BatchNorm2d(n) for n in self.nFiltersList])

    def forward(self, x):
        # narrow conv weights (i.e. filters) according to current nFilters
        out = conv2d(x, self.conv.weight.narrow(0, 0, self.nFiltersList[self.nFiltersCurrIdx]),
                     bias=self.conv.bias, stride=self.conv.stride, padding=self.conv.padding, dilation=self.conv.dilation, groups=self.conv.groups)
        out = self.bn[self.nFiltersCurrIdx](out)
        return out

    def getLayers(self):
        return [self]

    def getAllWidths(self):
        return self.nFiltersList

    def nFilters(self):
        return self.conv.out_channels


class BaseNet(Module):
    def buildLayersList(self):
        layersList = []
        for layer in self.layers:
            layersList.extend(layer.getLayers())

        return layersList

    def __init__(self, args, initLayersParams):
        super(BaseNet, self).__init__()
        # init save folder
        saveFolder = args.save
        # init layers
        self.layers = self.initLayers(initLayersParams)
        # build mixture layers list
        self.layersList = self.buildLayersList()

        self.printToFile(saveFolder)

    @abstractmethod
    def initLayers(self, params):
        raise NotImplementedError('subclasses must override initLayers()!')

    @abstractmethod
    def forward(self, x):
        raise NotImplementedError('subclasses must override forward()!')

    def printToFile(self, saveFolder):
        logger = HtmlLogger(saveFolder, 'model')
        logger.setMaxTableCellLength(1000)

        layerIdxKey = 'Layer#'
        nFiltersKey = 'Filters#'
        widthsKey = 'Width'
        layerArchKey = 'Layer Architecture'
        alphasKey = 'Alphas distribution'

        logger.createDataTable('Model architecture', [layerIdxKey, nFiltersKey, widthsKey, layerArchKey])
        for layerIdx, layer in enumerate(self.layersList):
            widths = layer.getAllWidths()

            dataRow = {layerIdxKey: layerIdx, nFiltersKey: layer.nFilters(), widthsKey: [widths], layerArchKey: layer}
            logger.addDataRow(dataRow)

        # # log layers alphas distribution
        # self.logDominantQuantizedOp(len(bitwidths), loggerFuncs=[lambda k, rows: logger.addInfoTable(alphasKey, rows)])
