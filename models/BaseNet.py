from abc import abstractmethod
from math import floor
from numpy import argsort
from functools import reduce

from torch.nn import Module, ModuleList, Conv2d, BatchNorm2d
from torch.nn.functional import conv2d

from utils.HtmlLogger import HtmlLogger


# abstract class for model block
class Block(Module):
    @abstractmethod
    def getLayers(self):
        raise NotImplementedError('subclasses must override getLayers()!')

    @abstractmethod
    def outputLayer(self):
        raise NotImplementedError('subclasses must override outputLayer()!')


# abstract class for model layer
class SlimLayer(Block):
    def __init__(self, nFiltersList, prevLayer):
        super(SlimLayer, self).__init__()

        # save previous layer
        self.prevLayer = [prevLayer]

        # save list of number of filters
        self.nFiltersList = nFiltersList
        # init current number of filters index
        self.nFiltersCurrIdx = 0

        # init forward counters
        self._forwardCounters = self._initForwardCounters()

    def getCurrFilterIdx(self):
        return self.nFiltersCurrIdx

    def getForwardCounters(self):
        return self._forwardCounters

    def getWidth(self, idx):
        return self.nFiltersList[idx]

    def nWidths(self):
        return len(self.nFiltersList)

    def _initForwardCounters(self):
        return [0] * self.nWidths()

    def resetForwardCounters(self):
        self._forwardCounters = self._initForwardCounters()

    def outputLayer(self):
        return self

    @abstractmethod
    def getAllWidths(self):
        raise NotImplementedError('subclasses must override getAllWidths()!')

    @abstractmethod
    # number of total filters in layer
    def nFilters(self):
        raise NotImplementedError('subclasses must override nFilters()!')

    @abstractmethod
    # current number of filters in layer
    def nCurrFilters(self):
        raise NotImplementedError('subclasses must override nFilters()!')


class ConvSlimLayer(SlimLayer):
    def __init__(self, widthRatioList, in_planes, out_planes, kernel_size, stride, prevLayer=None):
        super(ConvSlimLayer, self).__init__([int(x * out_planes) for x in widthRatioList], prevLayer)

        # init conv2d module
        self.conv = Conv2d(in_planes, out_planes, kernel_size, stride=stride, padding=floor(kernel_size / 2), bias=False)
        # init independent batchnorm module for number of filters
        self.bn = ModuleList([BatchNorm2d(n) for n in self.nFiltersList])

    def forward(self, x):
        # narrow conv weights (i.e. filters) according to current nFilters
        convWeights = self.conv.weight.narrow(0, 0, self.nFiltersList[self.nFiltersCurrIdx])
        # narrow conv weights (i.e. filters) according to previous layer nFilters
        if self.prevLayer[0] is not None:
            convWeights = convWeights.narrow(1, 0, self.prevLayer[0].nCurrFilters())

        # perform forward
        out = conv2d(x, convWeights, bias=self.conv.bias, stride=self.conv.stride, padding=self.conv.padding, dilation=self.conv.dilation,
                     groups=self.conv.groups)
        out = self.bn[self.nFiltersCurrIdx](out)

        # update forward counters
        self._forwardCounters[self.nFiltersCurrIdx] += 1

        return out

    def getLayers(self):
        return [self]

    def getAllWidths(self):
        return self.nFiltersList

    # current number of filters in layer
    def nCurrFilters(self):
        return self.nFiltersList[self.nFiltersCurrIdx]

    # number of total filters in layer
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
        self._layersList = self.buildLayersList()

        self.printToFile(saveFolder)
        # calc number of width permutations in model
        self.nPerms = reduce(lambda x, y: x * y, [layer.nWidths() for layer in self.layersList()])

    @abstractmethod
    def initLayers(self, params):
        raise NotImplementedError('subclasses must override initLayers()!')

    @abstractmethod
    def forward(self, x):
        raise NotImplementedError('subclasses must override forward()!')

    # iterate over model layers
    def layersList(self):
        for layer in self._layersList:
            yield layer

    def printToFile(self, saveFolder):
        logger = HtmlLogger(saveFolder, 'model')
        logger.setMaxTableCellLength(1000)

        layerIdxKey = 'Layer#'
        nFiltersKey = 'Filters#'
        widthsKey = 'Width'
        layerArchKey = 'Layer Architecture'
        alphasKey = 'Alphas distribution'

        logger.createDataTable('Model architecture', [layerIdxKey, nFiltersKey, widthsKey, layerArchKey])
        for layerIdx, layer in enumerate(self.layersList()):
            widths = layer.getAllWidths()

            dataRow = {layerIdxKey: layerIdx, nFiltersKey: layer.nFilters(), widthsKey: [widths], layerArchKey: layer}
            logger.addDataRow(dataRow)

        # # log layers alphas distribution
        # self.logDominantQuantizedOp(len(bitwidths), loggerFuncs=[lambda k, rows: logger.addInfoTable(alphasKey, rows)])

    def _resetForwardCounters(self):
        for layer in self.layersList():
            # reset layer forward counters
            layer.resetForwardCounters()

    def logForwardCounters(self, loggerFuncs):
        if isinstance(loggerFuncs, list) and len(loggerFuncs) > 0:
            rows = [['Layer #', 'Counters']]
            counterCols = ['Width', 'Counter']

            for layerIdx, layer in enumerate(self.layersList()):
                layerForwardCounters = layer.getForwardCounters()
                # sort layer forward counters indices in descending order, [::-1] changes to descending order
                indices = argsort(layerForwardCounters)[::-1]
                # build layer data row
                layerRows = [counterCols]
                for idx in indices:
                    layerRows.append([layer.getWidth(idx), layerForwardCounters[idx]])
                # add summary row
                layerRows.append(['Total', sum(layerForwardCounters)])

                # add layer row to model table
                rows.append([layerIdx, layerRows])

            # apply loggers functions
            for f in loggerFuncs:
                f(rows)

        # reset counters
        self._resetForwardCounters()
