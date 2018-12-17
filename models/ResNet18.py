from torch.nn import ModuleList, ReLU, Linear, AvgPool2d
from torch.nn.functional import linear

from .BaseNet import BaseNet, Block, ConvSlimLayer


class Input:
    def __init__(self, channels, input_size):
        self.channels = channels
        self.input_size = input_size

    # number of channels in input
    def currWidth(self):
        return self.channels

    def outputChannels(self):
        return self.channels

    def widthList(self):
        return [self.channels]

    def outputSize(self):
        return self.input_size


class BasicBlock(Block):
    def __init__(self, widthRatioList, in_planes, out_planes, kernel_size, stride, prevLayer=None):
        super(BasicBlock, self).__init__()

        stride1 = stride if in_planes == out_planes else (stride + 1)

        # build 1st block
        self.conv1 = ConvSlimLayer(widthRatioList, in_planes, out_planes, kernel_size, stride1, prevLayer=prevLayer)
        self.relu1 = ReLU(inplace=True)

        # build 2nd block
        self.conv2 = ConvSlimLayer(widthRatioList, out_planes, out_planes, kernel_size, stride, prevLayer=self.conv1)
        self.relu2 = ReLU(inplace=True)

        # init downsample
        self.downsample = ConvSlimLayer(widthRatioList, in_planes, out_planes, 1, stride1, prevLayer=prevLayer) \
            if in_planes != out_planes else None

    def forward(self, x):
        residual = self.downsample(x) if self.downsample else x

        out = self.conv1(x)
        out = self.relu1(out)
        out = self.conv2(out)
        out += residual
        out = self.relu2(out)

        return out

    def getLayers(self):
        downsample = [] if self.downsample is None else [self.downsample]
        return [self.conv1] + [self.conv2] + downsample

    def outputLayer(self):
        return self.conv2

    def countFlops(self):
        return sum([layer.countFlops() for layer in self.getLayers()])


class ResNet18(BaseNet):
    def __init__(self, args):
        super(ResNet18, self).__init__(args, initLayersParams=(args.width, args.kernel, args.nClasses, args.input_size, args.partition))

    # init layers (type, out_planes)
    def initBlocksPlanes(self):
        return [(ConvSlimLayer, 16), (BasicBlock, 16), (BasicBlock, 16), (BasicBlock, 16),
                (BasicBlock, 32), (BasicBlock, 32), (BasicBlock, 32),
                (BasicBlock, 64), (BasicBlock, 64), (BasicBlock, 64)]

    @staticmethod
    def nPartitionBlocks():
        return 3, [4, 3, 3]

    def initBlocks(self, params):
        widthRatioList, kernel_size, nClasses, input_size, partition = params

        blocksPlanes = self.initBlocksPlanes()

        # create list of blocks from blocksPlanes
        blocks = ModuleList()
        prevLayer = Input(3, input_size)
        stride = 1
        for i, (blockType, out_planes) in enumerate(blocksPlanes):
            layerWidthRatioList = widthRatioList.copy()
            # add partition ratio if exists
            if partition:
                layerWidthRatioList += [partition[i]]
            # build layer
            l = blockType(layerWidthRatioList, prevLayer.outputChannels(), out_planes, kernel_size, stride, prevLayer)
            # add layer to blocks list
            blocks.append(l)
            # update previous layer
            prevLayer = l.outputLayer()

        self.avgpool = AvgPool2d(8)
        self.fc = Linear(64, nClasses).cuda()

        return blocks

    def forward(self, x):
        out = x
        for layer in self.blocks:
            out = layer(out)

        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        # narrow linear according to last conv2d layer
        out = linear(out, self.fc.weight.narrow(1, 0, layer.outputLayer().currWidth()), bias=self.fc.bias)

        return out
