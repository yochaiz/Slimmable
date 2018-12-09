from torch.nn import ModuleList, ReLU, Linear, AvgPool2d
from torch.nn.functional import linear

from .BaseNet import BaseNet, Block, ConvSlimLayer


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
        self.downsample = ConvSlimLayer(widthRatioList, in_planes, out_planes, kernel_size=1, stride=stride1, prevLayer=prevLayer) \
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


class ResNet18(BaseNet):
    def __init__(self, args):
        super(ResNet18, self).__init__(args, initLayersParams=(args.width, args.kernel, args.nClasses))

    # init layers (type, in_planes, out_planes)
    def initLayersPlanes(self):
        return [(ConvSlimLayer, 3, 16), (BasicBlock, 16, 16), (BasicBlock, 16, 16), (BasicBlock, 16, 16),
                (BasicBlock, 16, 32), (BasicBlock, 32, 32), (BasicBlock, 32, 32),
                (BasicBlock, 32, 64), (BasicBlock, 64, 64), (BasicBlock, 64, 64)]

    def initLayers(self, params):
        widthRatioList, kernel_size, nClasses = params
        widthRatioList = widthRatioList.copy()

        layersPlanes = self.initLayersPlanes()
        # TODO: get input size from dataset and calculate input_size per block

        # create list of layers from layersPlanes
        # supports bitwidth as list of ints, i.e. same bitwidths to all layers
        layers = ModuleList()
        prevLayer = None
        for i, (layerType, in_planes, out_planes) in enumerate(layersPlanes):
            # build layer
            l = layerType(widthRatioList, in_planes, out_planes, kernel_size, stride=1, prevLayer=prevLayer)
            # add layer to layers list
            layers.append(l)
            # update previous layer
            prevLayer = l.outputLayer()

        self.avgpool = AvgPool2d(8)
        self.fc = Linear(64, nClasses).cuda()

        return layers

    def forward(self, x):
        out = x
        for layer in self.layers:
            out = layer(out)

        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        # narrow linear according to last conv2d layer
        out = linear(out, self.fc.weight.narrow(1, 0, layer.outputLayer().nCurrFilters()), bias=self.fc.bias)

        return out
