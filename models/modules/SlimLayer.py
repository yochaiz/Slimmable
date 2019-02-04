from abc import abstractmethod
from collections import defaultdict
from .block import Block


# abstract class for model layer
class SlimLayer(Block):
    def __init__(self, buildParams, widthRatioList, widthList, prevLayer):
        super(SlimLayer, self).__init__()

        assert (len(widthRatioList) == len(widthList))
        # save previous layer
        self._prevLayer = [prevLayer]

        # save width ratio list
        self._widthRatioList = widthRatioList.copy()
        # save list of number of filters
        self._widthList = widthList.copy()
        # init current number of filters index
        self._currWidthIdx = 0

        # init forward counters
        self._forwardCounters = self._initForwardCounters()

        # build layer modules
        self.buildModules(buildParams)

        # count flops for each width
        self.flopsDict, self.output_size = self.countWidthFlops(self._prevLayer[0].outputSize())

    @abstractmethod
    def buildModules(self, buildParams):
        raise NotImplementedError('subclasses must override getAllWidths()!')

    @abstractmethod
    # number of output channels in layer
    def outputChannels(self):
        raise NotImplementedError('subclasses must override outputChannels()!')

    @abstractmethod
    # number of output channels in layer
    def toStr(self):
        raise NotImplementedError('subclasses must override toStr()!')

    @abstractmethod
    # count flops for each width
    def countWidthFlops(self, input_size):
        raise NotImplementedError('subclasses must override countWidthFlops()!')

    def countFlops(self):
        return self.flopsDict[(self._prevLayer[0].currWidth(), self.currWidth())]

    @property
    def prevLayer(self):
        return self._prevLayer[0]

    def updateCurrWidth(self):
        pass

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
        assert (0 <= idx < len(self._widthList))
        self._currWidthIdx = idx

    # returns the index of given width ratio
    def widthRatioIdx(self, widthRatio):
        return self._widthRatioList.index(widthRatio)

    def nWidths(self):
        return len(self._widthList)

    def _initForwardCounters(self):
        return defaultdict(int)
        # return [0] * self.nWidths()

    def resetForwardCounters(self):
        self._forwardCounters = self._initForwardCounters()

    def outputLayer(self):
        return self

    # layer output tensor size, not number of output channels
    def outputSize(self):
        return self.output_size
