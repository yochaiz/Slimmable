from abc import abstractmethod
from math import floor
from numpy import argsort
from functools import reduce
from pandas import DataFrame

from torch import zeros
from torch.nn import Module, ModuleList, Conv2d, BatchNorm2d
from torch.nn.functional import conv2d, softmax
from torch.distributions.categorical import Categorical

from utils.HtmlLogger import HtmlLogger
from utils.flops_benchmark import count_flops


# abstract class for model block
class Block(Module):
    @abstractmethod
    def getOptimizationLayers(self):
        raise NotImplementedError('subclasses must override getOptimizationLayers()!')

    @abstractmethod
    def getFlopsLayers(self):
        raise NotImplementedError('subclasses must override getFlopsLayers()!')

    @abstractmethod
    def getCountersLayers(self):
        raise NotImplementedError('subclasses must override getCountersLayers()!')

    @abstractmethod
    def outputLayer(self):
        raise NotImplementedError('subclasses must override outputLayer()!')

    @abstractmethod
    def countFlops(self):
        raise NotImplementedError('subclasses must override countFlops()!')

    @abstractmethod
    # make some adjustments in model due to current width selected
    def updateCurrWidth(self):
        raise NotImplementedError('subclasses must override updateCurrWidth()!')

    @abstractmethod
    # generate new BNs for current model path, except for given srcLayer
    def generatePathBNs(self, srcLayer):
        raise NotImplementedError('subclasses must override generatePathBNs()!')


# abstract class for model layer
class SlimLayer(Block):
    def __init__(self, buildParams, widthRatioList, widthList, prevLayer):
        super(SlimLayer, self).__init__()

        assert (len(widthRatioList) == len(widthList))
        # save previous layer
        self.prevLayer = [prevLayer]

        # save width ratio list
        self._widthRatioList = widthRatioList
        # save list of number of filters
        self._widthList = widthList
        # init current number of filters index
        self._currWidthIdx = 0

        # init alphas
        self._alphas = zeros(self.nWidths()).cuda().clone().detach().requires_grad_(True)

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
    # number of output channels in layer
    def outputChannels(self):
        raise NotImplementedError('subclasses must override outputChannels()!')

    @abstractmethod
    # count flops for each width
    def countWidthFlops(self, input_size):
        raise NotImplementedError('subclasses must override countWidthFlops()!')

    def countFlops(self):
        return self.flopsDict[(self.prevLayer[0].currWidth(), self.currWidth())]

    def updateCurrWidth(self):
        pass

    # return alphas value
    def alphas(self):
        return self._alphas

    # return alphas probabilities
    def probs(self):
        return softmax(self._alphas, dim=-1).detach()

    def widthList(self):
        return self._widthList

    # current number of filters in layer
    def currWidth(self):
        return self._widthList[self._currWidthIdx]

    # current width ratio
    def currWidthRatio(self):
        return self._widthRatioList[self._currWidthIdx]

    def forwardCounters(self):
        return self._forwardCounters

    def widthByIdx(self, idx):
        return self._widthList[idx]

    def widthRatioByIdx(self, idx):
        return self._widthRatioList[idx]

    def currWidthIdx(self):
        return self._currWidthIdx

    def setCurrWidthIdx(self, idx):
        assert (0 <= idx <= len(self._widthList))
        self._currWidthIdx = idx

    # returns the index of given width ratio
    def widthRatioIdx(self, widthRatio):
        return self._widthRatioList.index(widthRatio)

    def nWidths(self):
        return len(self._widthList)

    def _initForwardCounters(self):
        return [0] * self.nWidths()

    def resetForwardCounters(self):
        self._forwardCounters = self._initForwardCounters()

    def outputLayer(self):
        return self

    # layer output tensor size, not number of output channels
    def outputSize(self):
        return self.output_size

    # select alpha based on alphas distribution
    def choosePathByAlphas(self):
        dist = Categorical(logits=self._alphas)
        chosen = dist.sample()
        self._currWidthIdx = chosen.item()

    # select maximal alpha
    def chooseAlphaMax(self):
        self._currWidthIdx = self._alphas.argmax().item()


class ConvSlimLayer(SlimLayer):
    def __init__(self, widthRatioList, in_planes, out_planes, kernel_size, stride, prevLayer=None):
        super(ConvSlimLayer, self).__init__((in_planes, out_planes, kernel_size, stride), widthRatioList,
                                            [int(x * out_planes) for x in widthRatioList], prevLayer)

        # update get layers functions
        self.getOptimizationLayers = self.getLayers
        self.getFlopsLayers = self.getLayers
        self.getCountersLayers = self.getLayers

        # init layer original BNs container
        self._orgBNs = [self.bn]
        # init layer original width list
        self._orgWidthList = self.widthList()
        # init layer original width ratio list
        self._orgWidthRatioList = self._widthRatioList

    def orgBNs(self):
        return self._orgBNs[0]

    def buildModules(self, params):
        in_planes, out_planes, kernel_size, stride = params
        # init conv2d module
        self.conv = Conv2d(in_planes, out_planes, kernel_size, stride=stride, padding=floor(kernel_size / 2), bias=False).cuda()
        # init independent batchnorm module for number of filters
        self.bn = ModuleList([BatchNorm2d(n) for n in self._widthList]).cuda()

    # generate new BNs based on current width
    def generatePathBNs(self, srcLayer):
        if self != srcLayer:
            # get current BN
            currBN = self.orgBNs()[self.currWidthIdx()]
            # get current BN num_features
            bnFeatures = currBN.num_features
            # generate new BNs ModuleList
            newBNs = ModuleList([BatchNorm2d(bnFeatures) for _ in range(self.nWidths())]).cuda()
            # copy weights to new BNs
            for bn in newBNs:
                bn.load_state_dict(currBN.state_dict())
            # set layer BNs
            self.bn = newBNs
            # update width List
            self._widthList = [self.currWidth()] * self.nWidths()
            # update width ratio list
            self._widthRatioList = [self.currWidthRatio()] * self.nWidths()

    def restoreOriginalBNs(self):
        self.bn = self.orgBNs()
        self._widthList = self._orgWidthList
        self._widthRatioList = self._orgWidthRatioList

    def forward(self, x):
        # narrow conv weights (i.e. filters) according to current nFilters
        convWeights = self.conv.weight.narrow(0, 0, self._widthList[self._currWidthIdx])
        # narrow conv weights (i.e. filters) according to previous layer nFilters
        convWeights = convWeights.narrow(1, 0, self.prevLayer[0].currWidth())

        # perform forward
        out = conv2d(x, convWeights, bias=self.conv.bias, stride=self.conv.stride, padding=self.conv.padding, dilation=self.conv.dilation,
                     groups=self.conv.groups)
        out = self.bn[self._currWidthIdx](out)

        # update forward counters
        self._forwardCounters[self._currWidthIdx] += 1

        return out

    def getLayers(self):
        return [self]

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
        for width in self.widthList():
            for prevWidth in self.prevLayer[0].widthList():
                conv = Conv2d(prevWidth, width, self.conv.kernel_size, bias=self.conv.bias, stride=self.conv.stride,
                              padding=self.conv.padding, dilation=self.conv.dilation, groups=self.conv.groups)
                flops, output_size = count_flops(conv, input_size, prevWidth)
                flopsDict[(prevWidth, width)] = flops

        return flopsDict, output_size


class BaseNet(Module):
    class Layers:
        def __init__(self, blocks):
            self._blocks = blocks
            # assuming these lists layers don't change
            self._optim = self._buildLayersList(blocks, lambda layer: layer.getOptimizationLayers())
            self._forwardCounters = self._buildLayersList(blocks, lambda layer: layer.getCountersLayers())

        def _buildLayersList(self, blocks, getLayersFunc):
            layersList = []
            for layer in blocks:
                layersList.extend(getLayersFunc(layer))

            return layersList

        def optimization(self):
            return self._optim

        def flops(self):
            return self._buildLayersList(self._blocks, lambda layer: layer.getFlopsLayers())

        def forwardCounters(self):
            return self._forwardCounters

    _partitionKey = 'Partition'
    _baselineFlopsKey = 'baselineFlops'
    _baselineFlopsRatioKey = 'baselineFlopsRatio'
    _alphasCsvFileName = 'alphas.csv'

    def __init__(self, args, initLayersParams):
        super(BaseNet, self).__init__()
        # init save folder
        saveFolder = args.save
        # init layers
        self.blocks = self.initBlocks(initLayersParams)
        # init Layers class instance
        self._layers = self.Layers(self.blocks)
        # build layers alphas list
        self._alphas = [layer.alphas() for layer in self.layersList()]

        # init dictionary of layer width indices list per width ratio
        self._baselineWidth = self.buildHomogeneousWidthIdx(args.width)
        # add partition to baseline width dictionary
        # partition batchnorm is the last one in each layer batchnorms list
        if args.partition:
            self._baselineWidth[self._partitionKey] = [len(self._baselineWidth)] * len(self._layers.optimization())
        # count baseline models widths flops
        baselineFlops = self.calcBaselineFlops()
        # save baseline flops, for calculating flops ratio
        self.baselineFlops = baselineFlops.get(args.baseline)
        # add baseline models widths flops to args
        setattr(args, self._baselineFlopsKey, baselineFlops)
        # add baseline models widths flops ratio to args
        setattr(args, self._baselineFlopsRatioKey, {k: (v / self.baselineFlops) for k, v in baselineFlops.items()})
        self.printToFile(saveFolder)
        # calc number of width permutations in model
        self.nPerms = reduce(lambda x, y: x * y, [layer.nWidths() for layer in self._layers.optimization()])

        # init alphas DataFrame
        self.alphas_df = None
        self.__initAlphasDataFrame(saveFolder)

    @abstractmethod
    def initBlocks(self, params):
        raise NotImplementedError('subclasses must override initLayers()!')

    @staticmethod
    @abstractmethod
    # number of model blocks for partition, in order to generate different width for each block
    # returns tuple (number of blocks as int, list of number of layer in each block)
    def nPartitionBlocks():
        raise NotImplementedError('subclasses must override nPartitionBlocks()!')

    @abstractmethod
    def forward(self, x):
        raise NotImplementedError('subclasses must override forward()!')

    @staticmethod
    def partitionKey():
        return BaseNet._partitionKey

    @staticmethod
    def baselineFlopsKey():
        return BaseNet._baselineFlopsKey

    def additionalLayersToLog(self):
        return []

    def countFlops(self):
        return sum([block.countFlops() for block in self.blocks])

    def flopsRatio(self):
        return self.countFlops() / self.baselineFlops

    def layersList(self):
        return self._layers.optimization()

    def alphas(self):
        return self._alphas

    def baselineWidthKeys(self):
        return list(self._baselineWidth.keys())

    def baselineWidth(self):
        for v in self._baselineWidth.items():
            yield v

    def currWidthIdx(self):
        return [layer.currWidthIdx() for layer in self._layers.optimization()]

    def currWidthRatio(self):
        return [layer.currWidthRatio() for layer in self._layers.optimization()]

    def setCurrWidthIdx(self, idxList):
        for layer, idx in zip(self._layers.optimization(), idxList):
            layer.setCurrWidthIdx(idx)

        # update curr width changes in each block
        for block in self.blocks:
            block.updateCurrWidth()

    # select alpha based on alphas distribution
    def choosePathByAlphas(self):
        for layer in self._layers.optimization():
            layer.choosePathByAlphas()

        # update curr width changes in each block
        for block in self.blocks:
            block.updateCurrWidth()

    # select maximal alpha in each layer
    def chooseAlphaMax(self):
        for layer in self._layers.optimization():
            layer.chooseAlphaMax()

        # update curr width changes in each block
        for block in self.blocks:
            block.updateCurrWidth()

    # build a dictionary where each key is width ratio and each value is the list of layer indices in order to set the key width ratio as current
    # width in each layer
    def buildHomogeneousWidthIdx(self, widthRatioList):
        homogeneousWidth = {}

        for widthRatio in widthRatioList:
            # check if width is not in baselineResults dictionary
            if widthRatio not in homogeneousWidth:
                # build layer indices for current width ratio
                homogeneousWidth[widthRatio] = [l.widthRatioIdx(widthRatio) for l in self._layers.optimization()]

        return homogeneousWidth

    def calcBaselineFlops(self):
        return self.applyOnBaseline(self.countFlops)

    # apply some function on baseline models
    # baseline models are per layer width
    # this function create a map from baseline width to func() result on baseline model
    # def applyOnBaseline(self, func, applyOnAlphasDistribution=False):
    def applyOnBaseline(self, func):
        baselineResults = {}
        # save current model width indices
        modelCurrWidthIdx = self.currWidthIdx()
        # iterate over width ratios
        for widthRatio, idxList in self.baselineWidth():
            # set model layers current width index
            self.setCurrWidthIdx(idxList)
            # update value in dictionary
            baselineResults[widthRatio] = func()

        # # apply on current alphas distribution
        # if applyOnAlphasDistribution:
        #     self.setFiltersByAlphas()
        #     # &#945; is greek alpha symbol in HTML
        #     baselineResults['&#945;'] = func()

        # restore model layers current width
        self.setCurrWidthIdx(modelCurrWidthIdx)

        return baselineResults

    def loadPreTrained(self, state_dict):
        self.load_state_dict(state_dict)

    def __initAlphasDataFrame(self, saveFolder):
        if saveFolder:
            # update save path if saveFolder exists
            self._alphasCsvFileName = '{}/{}'.format(saveFolder, self._alphasCsvFileName)
            # init DataFrame cols
            cols = ['Epoch', 'Batch']
            cols += ['Layer_{}'.format(i) for i in range(len(self.layersList()))]
            self.cols = cols
            # init DataFrame
            self.alphas_df = DataFrame([], columns=cols)
            # set init data
            data = ['init', 'init']
            # save alphas data
            self.saveAlphasCsv(data)

    # save alphas values to csv
    def saveAlphasCsv(self, data):
        if self.alphas_df is not None:
            data += [[round(e.item(), 5) for e in layer.alphas()] for layer in self.layersList()]
            # create new row
            d = DataFrame([data], columns=self.cols)
            # add row
            self.alphas_df = self.alphas_df.append(d)
            # save DataFrame
            self.alphas_df.to_csv(self._alphasCsvFileName)

    def _topAlphas(self, k):
        top = []
        for layer in self._layers.optimization():
            alphas = layer.alphas()
            # sort alphas probabilities
            wSorted, wIndices = layer.probs().sort(descending=True)
            # keep only top-k
            wSorted = wSorted[:k]
            wIndices = wIndices[:k]
            # get layer widths
            widths = layer.widthList()
            # add to top
            top.append([(i, w.item(), alphas[i], widths[i]) for w, i in zip(wSorted, wIndices)])

        return top

    def logTopAlphas(self, k, loggerFuncs=[]):
        if (not loggerFuncs) or (len(loggerFuncs) == 0):
            return

        rows = [['Layer #', 'Alphas']]
        alphaCols = ['Index', 'Ratio', 'Value', 'Width']

        top = self._topAlphas(k=k)
        for i, layerTop in enumerate(top):
            layerRow = [alphaCols]
            for idx, w, alpha, width in layerTop:
                alphaRow = [idx, '{:.5f}'.format(w), '{:.5f}'.format(alpha), width]
                # add alpha data row to layer data table
                layerRow.append(alphaRow)
            # add layer data table to model table as row
            rows.append([i, layerRow])

        # apply loggers functions
        for f in loggerFuncs:
            f(k, rows)

    def printToFile(self, saveFolder):
        logger = HtmlLogger(saveFolder, 'model')
        logger.setMaxTableCellLength(1000)

        layerIdxKey = 'Layer#'
        nFiltersKey = 'Filters#'
        widthsKey = 'Width'
        layerArchKey = 'Layer Architecture'
        alphasKey = 'Alphas distribution'

        logger.createDataTable('Model architecture', [layerIdxKey, nFiltersKey, widthsKey, layerArchKey])
        for layerIdx, layer in enumerate(self._layers.flops()):
            widths = layer.widthList()

            dataRow = {layerIdxKey: layerIdx, nFiltersKey: layer.outputChannels(), widthsKey: [widths], layerArchKey: layer}
            logger.addDataRow(dataRow)

        layerIdx += 1
        # log additional layers, like Linear, MaxPool2d, AvgPool2d
        for layer in self.additionalLayersToLog():
            dataRow = {layerIdxKey: layerIdx, layerArchKey: layer}
            logger.addDataRow(dataRow)
            layerIdx += 1

        # log layers alphas distribution
        self.logTopAlphas(len(widths), loggerFuncs=[lambda k, rows: logger.addInfoTable(alphasKey, rows)])

    def _resetForwardCounters(self):
        for layer in self._layers.forwardCounters():
            # reset layer forward counters
            layer.resetForwardCounters()

    def logForwardCounters(self, loggerFuncs):
        if isinstance(loggerFuncs, list) and len(loggerFuncs) > 0:
            rows = [['Layer #', 'Counters']]
            counterCols = ['Width', 'Counter']

            for layerIdx, layer in enumerate(self._layers.forwardCounters()):
                layerForwardCounters = layer.forwardCounters()
                # sort layer forward counters indices in descending order, [::-1] changes to descending order
                indices = argsort(layerForwardCounters)[::-1]
                # build layer data row
                layerRows = [[layer], counterCols]
                for idx in indices:
                    layerRows.append([layer.widthByIdx(idx), layerForwardCounters[idx]])
                # add summary row
                layerRows.append(['Total', sum(layerForwardCounters)])

                # add layer row to model table
                rows.append([layerIdx, layerRows])

            # apply loggers functions
            for f in loggerFuncs:
                f(rows)

        # reset counters
        self._resetForwardCounters()

# def loadPreTrained(self, state_dict):
#     from collections import OrderedDict
#     newDict = OrderedDict()
#
#     tokenOrg = '.downsample.'
#     tokenNew = '.downsample.downsampleSrc.'
#     for key in state_dict.keys():
#         if tokenOrg in key:
#             newDict[key.replace(tokenOrg, tokenNew)] = state_dict[key]
#         else:
#             newDict[key] = state_dict[key]
#
#     currDict = self.state_dict()
#     for key in currDict.keys():
#         if key not in newDict:
#             newDict[key] = currDict[key]
#
#     # load weights
#     self.load_state_dict(newDict)

# def loadPreTrained(self, path, logger):
#     loggerRows = []
#     if path is not None:
#         if exists(path):
#             # load checkpoint
#             checkpoint = loadModel(path, map_location=lambda storage, loc: storage.cuda())
#             # set checkpoint state dict
#             chckpntStateDict = checkpoint['state_dict']
#             # load model state dict keys
#             modelStateDict = self.state_dict()
#             modelStateDictKeys = set(modelStateDict.keys())
#             # compare dictionaries
#             dictDiff = modelStateDictKeys.symmetric_difference(set(chckpntStateDict.keys()))
#
#             # init batchnorm token
#             token = '.bn.'
#             # init how many batchnorms we have loaded from pre-trained
#             bnLoadedCounter = 0
#             # init how many batchnorms are in total
#             bnTotalCounter = sum(1 for key in modelStateDictKeys if token in key)
#
#             # init new dict, based on current dict
#             newDict = modelStateDict
#             # iterate over checkpoint state dict keys
#             for key in chckpntStateDict.keys():
#                 # duplicate values with their corresponding new keys
#                 if token in key:
#                     # filters out num_batches_tracked in cases it is not needed
#                     if key in modelStateDict:
#                         if modelStateDict[key].size() == chckpntStateDict[key].size():
#                             newDict[key] = chckpntStateDict[key]
#                             bnLoadedCounter += 1
#                         else:
#                             # add model state dict values to new dict
#                             newDict[key] = modelStateDict[key]
#                 else:
#                     # add checkpoint state dict values to new dict
#                     newDict[key] = chckpntStateDict[key]
#
#             # load weights
#             self.load_state_dict(newDict)
#             # add info rows about checkpoint
#             loggerRows.append(['Path', '{}'.format(path)])
#             validationAccRows = [['Ratio', 'Accuracy']] + HtmlLogger.dictToRows(checkpoint['best_prec1'], nElementPerRow=1)
#             loggerRows.append(['Validation accuracy', validationAccRows])
#             loggerRows.append(['StateDict diff', list(dictDiff)])
#             loggerRows.append(['Loaded Batchnorm #', '{}/{}'.format(bnLoadedCounter, bnTotalCounter)])
#         else:
#             raise ValueError('Failed to load pre-trained from [{}], path does not exists'.format(path))
#
#         # load pre-trained model if we tried to load pre-trained
#         logger.addInfoTable('Pre-trained model', loggerRows)


# def loadPreTrained(self, path, logger):
# # replace old key (.layers.) with new key (.blocks.)
# chckpntUpdatedDict = OrderedDict()
# oldKey = 'layers.'
# newKey = 'blocks.'
# for dictKey in chckpntStateDict.keys():
#     newDictKey = dictKey
#     if oldKey in dictKey:
#         newDictKey = dictKey.replace(oldKey, newKey)
#
#     chckpntUpdatedDict[newDictKey] = chckpntStateDict[dictKey]

# # duplicate BN weights
# if duplicateBNWeights:
#     # init new dict, since we have to add new keys
#     newDict = OrderedDict()
#     # init tokens
#     token = '.bn.0.'
#     template = '.bn.{}.'
#     # iterate over checkpoint state dict keys
#     for key in chckpntStateDict.keys():
#         # add checkpoint state dict values to new dict
#         newDict[key] = chckpntStateDict[key]
#         # duplicate values with their corresponding new keys
#         if token in key:
#             idx = 1
#             newKey = key.replace(token, template.format(idx))
#             while newKey in dictDiff:
#                 newDict[newKey] = chckpntStateDict[key]
#                 dictDiff.remove(newKey)
#                 idx += 1
#                 newKey = key.replace(token, template.format(idx))
#
#     # update the state dict we want to load to model
#     chckpntStateDict = newDict
