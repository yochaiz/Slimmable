from torch.nn import ModuleList, ReLU, Linear, AvgPool2d

from .BaseNet import BaseNet, Block, ConvLayer


class BasicBlock(Block):
    def __init__(self, widthRatioList, in_planes, out_planes, kernel_size, stride):
        super(BasicBlock, self).__init__()

        stride1 = stride if in_planes == out_planes else (stride + 1)

        # build 1st block
        self.conv1 = ConvLayer(widthRatioList, in_planes, out_planes, kernel_size, stride1)
        self.relu1 = ReLU(inplace=True)

        # build 2nd block
        self.conv2 = ConvLayer(widthRatioList, out_planes, out_planes, kernel_size, stride)
        self.relu2 = ReLU(inplace=True)

        # init downsample
        self.downsample = ConvLayer(widthRatioList, in_planes, out_planes, kernel_size=1, stride=stride1) \
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


class ResNet18(BaseNet):
    def __init__(self, args):
        super(ResNet18, self).__init__(args, initLayersParams=(args.width, args.kernel, args.nClasses))

    # init layers (type, in_planes, out_planes)
    def initLayersPlanes(self):
        return [(ConvLayer, 3, 16), (BasicBlock, 16, 16), (BasicBlock, 16, 16), (BasicBlock, 16, 16),
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
        for i, (layerType, in_planes, out_planes) in enumerate(layersPlanes):
            # build layer
            l = layerType(widthRatioList, in_planes, out_planes, kernel_size, stride=1)
            # add layer to layers list
            layers.append(l)

        self.avgpool = AvgPool2d(8)
        self.fc = Linear(64, nClasses).cuda()

        return layers

    def forward(self, x):
        out = x
        for layer in self.layers:
            out = layer(out)

        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)

        return out
