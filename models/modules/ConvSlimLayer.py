from .SlimLayer import SlimLayer, abstractmethod
from math import floor

from torch.nn import ModuleList, Conv2d, BatchNorm2d
from torch.nn.functional import conv2d

from utils.flops_benchmark import count_flops


class ConvSlimLayer(SlimLayer):
    def __init__(self, widthRatioList, out_planes, kernel_size, stride, prevLayer, countFlopsFlag):
        super(ConvSlimLayer, self).__init__((prevLayer.outputChannels(), out_planes, kernel_size, stride), widthRatioList,
                                            [int(x * out_planes) for x in widthRatioList], prevLayer, countFlopsFlag)

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
        convWeights = convWeights.narrow(1, 0, self.prevLayer.currWidth())

        # perform forward
        out = conv2d(x, convWeights, bias=self.conv.bias, stride=self.conv.stride, padding=self.conv.padding, dilation=self.conv.dilation,
                     groups=self.conv.groups)
        out = self.bn[self._currWidthIdx](out)

        # update forward counters
        _currWidth = self.currWidth()
        self._forwardCounters[_currWidth] += 1

        return out

    def getLayers(self):
        return [self]

    # number of total output filters in layer
    def outputChannels(self):
        return self.conv.out_channels

    def toStr(self):
        return 'Conv2d[{}, {}, kernel={}]'.format(self.conv.in_channels, self.conv.out_channels, self.conv.kernel_size)

    @abstractmethod
    # returns list of widths we want to calc flops for
    def flopsWidthList(self):
        raise NotImplementedError('subclasses must override flopsWidthList()!')

    # count flops for each width
    def countWidthFlops(self, input_size):
        # init flops dictionary, each key is (in_channels, out_channels)
        # where in_channels is number of filters in previous layer
        # out_channels in number of filters in current layer
        flopsDict = {}
        print('== Counting width flops ==')

        # iterate over current layer widths & previous layer widths
        for width in self.flopsWidthList():
            for prevWidth in self.prevLayer.flopsWidthList():
                conv = Conv2d(prevWidth, width, self.conv.kernel_size, bias=self.conv.bias, stride=self.conv.stride,
                              padding=self.conv.padding, dilation=self.conv.dilation, groups=self.conv.groups)
                flops, output_size = count_flops(conv, input_size, prevWidth)
                flopsDict[(prevWidth, width)] = flops

        print(flopsDict.keys())
        print('== Done counting ==')
        return flopsDict, output_size
