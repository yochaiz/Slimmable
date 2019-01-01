from abc import abstractmethod

from torch.nn import ModuleList, ReLU, Linear, AvgPool2d, MaxPool2d
from torch.nn.functional import linear

from .BaseNet import BaseNet, Block, ConvSlimLayer


class Input:
    def __init__(self, channels, output_size):
        self.channels = channels
        self.output_size = output_size

    # number of channels in input
    def currWidth(self):
        return self.channels

    def outputChannels(self):
        return self.channels

    def widthList(self):
        return [self.channels]

    def outputSize(self):
        return self.output_size


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
            if ((in_planes != out_planes) or (prevLayer.widthList()[-1] != self.conv1.widthList()[-1])) else None

        # init function to calculate residual
        self.residual = self.standardResidual if self.downsample is None else self.downsampleResidual

    @staticmethod
    # calc residual without downsample
    def standardResidual(downsample, x):
        return x

    @staticmethod
    # calc residual with downsample
    def downsampleResidual(downsample, x):
        return downsample(x)

    def forward(self, x):
        out = self.conv1(x)
        out = self.relu1(out)
        out = self.conv2(out)
        out += self.residual(self.downsample, x)
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
        super(ResNet18, self).__init__(args, initLayersParams=(args.width, args.nClasses, args.input_size, args.partition))

    # init layers (type, out_planes)
    def initBlocksPlanes(self):
        return [(ConvSlimLayer, 16), (BasicBlock, 16), (BasicBlock, 16), (BasicBlock, 16),
                (BasicBlock, 32), (BasicBlock, 32), (BasicBlock, 32),
                (BasicBlock, 64), (BasicBlock, 64), (BasicBlock, 64)]

    @staticmethod
    def nPartitionBlocks():
        return 3, [4, 3, 3]

    @abstractmethod
    def initBlocks(self, params):
        raise NotImplementedError('subclasses must override initBlocks()!')

    @abstractmethod
    def forward(self, x):
        raise NotImplementedError('subclasses must override forward()!')


class ResNet18_Cifar(ResNet18):
    def __init__(self, args):
        super(ResNet18_Cifar, self).__init__(args)

    def initBlocks(self, params):
        widthRatioList, nClasses, input_size, partition = params

        blocksPlanes = self.initBlocksPlanes()

        # init parameters
        kernel_size = 3
        stride = 1

        # create list of blocks from blocksPlanes
        blocks = ModuleList()
        prevLayer = Input(3, input_size)

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
        for block in self.blocks:
            out = block(out)

        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        # narrow linear according to last conv2d layer
        out = linear(out, self.fc.weight.narrow(1, 0, block.outputLayer().currWidth()), bias=self.fc.bias)

        return out


class ResNet18_Imagenet(ResNet18):
    def __init__(self, args):
        super(ResNet18_Imagenet, self).__init__(args)

    def initBlocks(self, params):
        widthRatioList, nClasses, input_size, partition = params

        blocksPlanes = self.initBlocksPlanes()

        # init parameters
        kernel_size = 7
        stride = 2

        # create list of blocks from blocksPlanes
        blocks = ModuleList()
        # output size is divided by 2 due to maxpool after 1st conv layer
        prevLayer = Input(3, input_size / 2)

        for i, (blockType, out_planes) in enumerate(blocksPlanes):
            # increase number of out_planes
            out_planes *= 4
            # copy width ratio list
            layerWidthRatioList = widthRatioList.copy()
            # add partition ratio if exists
            if partition:
                layerWidthRatioList += [partition[i]]
            # build layer
            l = blockType(layerWidthRatioList, prevLayer.outputChannels(), out_planes, kernel_size, stride, prevLayer)
            # update kernel size
            kernel_size = 3
            # update stride
            stride = 1
            # add layer to blocks list
            blocks.append(l)
            # update previous layer
            prevLayer = l.outputLayer()

        self.maxpool = MaxPool2d(kernel_size=kernel_size, stride=2, padding=1)
        self.avgpool = AvgPool2d(7)
        self.fc = Linear(1024, nClasses).cuda()

        return blocks

    def forward(self, x):
        out = x

        block = self.blocks[0]
        out = block(out)
        out = self.maxpool(out)

        for block in self.blocks[1:]:
            out = block(out)

        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        # narrow linear according to last conv2d layer
        in_features = int(self.fc.in_features * block.outputLayer().currWidthRatio())
        out = linear(out, self.fc.weight.narrow(1, 0, in_features), bias=self.fc.bias)

        return out
