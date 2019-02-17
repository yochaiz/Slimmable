from .BaseNet.BaseNet_categorical import BaseNet_Categorical, BasicBlock_Categorical
from .BaseNet.BaseNet_multinomial import BaseNet_Multinomial, BasicBlock_Multinomial
from .BaseNet.BaseNet_binomial import BaseNet_Binomial, BasicBlock_Binomial
from .BaseNet.BaseNet_widthblock_binomial import BaseNet_WidthBlock_Binomial, BasicBlock_WidthBlock_Binomial
from utils.args import Switcher


class ResNetSwitcher:
    _categoricalKey = Switcher.categoricalKey()
    _multinomialKey = Switcher.multinomialKey()
    _binomialKey = Switcher.binomialKey()
    _blockBinomialKey = Switcher.blockBinomialKey()

    _BaseNetDict = {_categoricalKey: BaseNet_Categorical, _multinomialKey: BaseNet_Multinomial, _binomialKey: BaseNet_Binomial,
                    _blockBinomialKey: BaseNet_WidthBlock_Binomial}
    _BasicBlockDict = {_categoricalKey: BasicBlock_Categorical, _multinomialKey: BasicBlock_Multinomial, _binomialKey: BasicBlock_Binomial,
                       _blockBinomialKey: BasicBlock_WidthBlock_Binomial}
    _ResNetModels = ['resnet18_cifar10', 'resnet18_cifar100', 'resnet18_imagenet']

    @staticmethod
    def getModelNames():
        return ResNetSwitcher._ResNetModels

    @staticmethod
    def getModelDict(classKey):
        # get BaseNet class
        BaseNetClass = ResNetSwitcher._BaseNetDict[classKey]
        # get BasicBlock class
        BasicBlockClass = ResNetSwitcher._BasicBlockDict[classKey]
        # create ResNet18 class
        from .ResNet18 import ResNet18
        resnet18 = ResNet18(BaseNetClass, BasicBlockClass)

        # import models functions
        from .ResNet18 import ResNet18_Cifar as resnet18_cifar10
        from .ResNet18 import ResNet18_Cifar as resnet18_cifar100
        from .ResNet18 import ResNet18_Imagenet as resnet18_imagenet
        # group models functions
        modelFuncs = [resnet18_cifar10, resnet18_cifar100, resnet18_imagenet]
        # apply models functions
        models = [modelFunc(resnet18) for modelFunc in modelFuncs]

        return {k: v for k, v in zip(ResNetSwitcher._ResNetModels, models)}
