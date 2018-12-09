from abc import abstractmethod
from math import floor
from numpy import argsort
from functools import reduce
from collections import OrderedDict
from os.path import exists

from torch import load as loadModel
from torch.nn import Module, ModuleList, Conv2d, BatchNorm2d
from torch.nn.functional import conv2d

from utils.HtmlLogger import HtmlLogger
from utils.flops_benchmark import count_flops


# abstract class for model block
class Block(Module):
    @abstractmethod
    def getLayers(self):
        raise NotImplementedError('subclasses must override getLayers()!')

    @abstractmethod
    def outputLayer(self):
        raise NotImplementedError('subclasses must override outputLayer()!')

    @abstractmethod
    def countFlops(self):
        raise NotImplementedError('subclasses must override countFlops()!')


# abstract class for model layer
class SlimLayer(Block):
    def __init__(self, buildParams, widthList, prevLayer):
        super(SlimLayer, self).__init__()

        # save previous layer
        self.prevLayer = [prevLayer]

        # save list of number of filters
        self.widthList = widthList
        # init current number of filters index
        self.currWidthIdx = 0

        # init forward counters
        self._forwardCounters = self._initForwardCounters()

        # build layer modules
        self.buildModules(buildParams)

        # count flops for each width
        self.flopsDict, self.output_size = self.countWidthFlops(self.prevLayer[0].outputSize())

    @abstractmethod
    def buildModules(self, buildParams):
        raise NotImplementedError('subclasses must override getAllWidths()!')

    @abstractmethod
    def getAllWidths(self):
        raise NotImplementedError('subclasses must override getAllWidths()!')

    @abstractmethod
    # number of output channels in layer
    def outputChannels(self):
        raise NotImplementedError('subclasses must override outputChannels()!')

    @abstractmethod
    # current number of filters in layer
    def currWidth(self):
        raise NotImplementedError('subclasses must override currWidth()!')

    @abstractmethod
    # count flops for each width
    def countWidthFlops(self, input_size):
        raise NotImplementedError('subclasses must override countWidthFlops()!')

    def countFlops(self):
        return self.flopsDict[(self.prevLayer[0].currWidth(), self.currWidth())]

    def getCurrWidthIdx(self):
        return self.currWidthIdx

    def getForwardCounters(self):
        return self._forwardCounters

    def getWidth(self, idx):
        return self.widthList[idx]

    def nWidths(self):
        return len(self.widthList)

    def _initForwardCounters(self):
        return [0] * self.nWidths()

    def resetForwardCounters(self):
        self._forwardCounters = self._initForwardCounters()

    def outputLayer(self):
        return self

    def outputSize(self):
        return self.output_size


class ConvSlimLayer(SlimLayer):
    def __init__(self, widthRatioList, in_planes, out_planes, kernel_size, stride, prevLayer=None):
        super(ConvSlimLayer, self).__init__((in_planes, out_planes, kernel_size, stride), [int(x * out_planes) for x in widthRatioList], prevLayer)

    def buildModules(self, params):
        in_planes, out_planes, kernel_size, stride = params
        # init conv2d module
        self.conv = Conv2d(in_planes, out_planes, kernel_size, stride=stride, padding=floor(kernel_size / 2), bias=False).cuda()
        # init independent batchnorm module for number of filters
        self.bn = ModuleList([BatchNorm2d(n) for n in self.widthList]).cuda()

    def forward(self, x):
        # narrow conv weights (i.e. filters) according to current nFilters
        convWeights = self.conv.weight.narrow(0, 0, self.widthList[self.currWidthIdx])
        # narrow conv weights (i.e. filters) according to previous layer nFilters
        convWeights = convWeights.narrow(1, 0, self.prevLayer[0].currWidth())

        # perform forward
        out = conv2d(x, convWeights, bias=self.conv.bias, stride=self.conv.stride, padding=self.conv.padding, dilation=self.conv.dilation,
                     groups=self.conv.groups)
        out = self.bn[self.currWidthIdx](out)

        # update forward counters
        self._forwardCounters[self.currWidthIdx] += 1

        return out

    def getLayers(self):
        return [self]

    def getAllWidths(self):
        return self.widthList

    # current number of filters in layer
    def currWidth(self):
        return self.widthList[self.currWidthIdx]

    # number of total output filters in layer
    def outputChannels(self):
        return self.conv.out_channels

    # count flops for each width
    def countWidthFlops(self, input_size):
        # init flops dictionary, each key is (in_channels, out_channels)
        # where in_channels is number of filters in previous layer
        # out_channels in number of filters in current layer
        flopsDict = {}

        # iterate over current layer widths & previous layer widths
        for width in self.getAllWidths():
            for prevWidth in self.prevLayer[0].getAllWidths():
                conv = Conv2d(prevWidth, width, self.conv.kernel_size, bias=self.conv.bias, stride=self.conv.stride,
                              padding=self.conv.padding, dilation=self.conv.dilation, groups=self.conv.groups)
                flops, output_size = count_flops(conv, input_size, prevWidth)
                flopsDict[(prevWidth, width)] = flops

        return flopsDict, output_size


class BaseNet(Module):
    def buildLayersList(self):
        layersList = []
        for layer in self.blocks:
            layersList.extend(layer.getLayers())

        return layersList

    def __init__(self, args, initLayersParams):
        super(BaseNet, self).__init__()
        # init save folder
        saveFolder = args.save
        # init layers
        self.blocks = self.initBlocks(initLayersParams)
        # build mixture layers list
        self._layersList = self.buildLayersList()

        self.printToFile(saveFolder)
        # calc number of width permutations in model
        self.nPerms = reduce(lambda x, y: x * y, [layer.nWidths() for layer in self.layersList()])

    @abstractmethod
    def initBlocks(self, params):
        raise NotImplementedError('subclasses must override initLayers()!')

    @abstractmethod
    def forward(self, x):
        raise NotImplementedError('subclasses must override forward()!')

    def countFlops(self):
        return sum([block.countFlops() for block in self.blocks])

    # iterate over model layers
    def layersList(self):
        for layer in self._layersList:
            yield layer

    def loadPreTrained(self, path, logger):
        loggerRows = []
        if path is not None:
            if exists(path):
                # load checkpoint
                checkpoint = loadModel(path, map_location=lambda storage, loc: storage.cuda())
                # update weights
                chckpntStateDict = checkpoint['state_dict']
                # replace old key (.layers.) with new key (.blocks.)
                chckpntUpdatedDict = OrderedDict()
                oldKey = 'layers.'
                newKey = 'blocks.'
                for dictKey in chckpntStateDict.keys():
                    newDictKey = dictKey
                    if oldKey in dictKey:
                        newDictKey = dictKey.replace(oldKey, newKey)

                    chckpntUpdatedDict[newDictKey] = chckpntStateDict[dictKey]
                # load model state dict keys
                modelStateDictKeys = set(self.state_dict().keys())
                # compare dictionaries
                dictDiff = modelStateDictKeys.symmetric_difference(set(chckpntUpdatedDict.keys()))
                # load weights
                self.load_state_dict(chckpntUpdatedDict)
                # add info rows about checkpoint
                loggerRows.append(['Path', '{}'.format(path)])
                loggerRows.append(['Validation accuracy', '{:.5f}'.format(checkpoint['best_prec1'])])
                loggerRows.append(['StateDict diff', list(dictDiff)])
            else:
                loggerRows.append(['Path', 'Failed to load pre-trained from [{}], path does not exists'.format(path)])

            # load pre-trained model if we tried to load pre-trained
            logger.addInfoTable('Pre-trained model', loggerRows)

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

            dataRow = {layerIdxKey: layerIdx, nFiltersKey: layer.outputChannels(), widthsKey: [widths], layerArchKey: layer}
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
