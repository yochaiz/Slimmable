from .BaseNet.BaseNet_categorical import BaseNet_Categorical
from .BaseNet.BaseNet_multinomial import BaseNet_Multinomial
from utils.args import Switcher


class ResNetSwitcher:
    _categoricalKey = Switcher.categoricalKey()
    _multinomialKey = Switcher.multinomialKey()
    _BaseNetDict = {_categoricalKey: BaseNet_Categorical, _multinomialKey: BaseNet_Multinomial}
    _ResNetModels = ['resnet18_cifar10', 'resnet18_cifar100', 'resnet18_imagenet']

    @staticmethod
    def getModelNames():
        return ResNetSwitcher._ResNetModels

    @staticmethod
    def getModelDict(classKey):
        # get BaseNet class
        BaseNetClass = ResNetSwitcher._BaseNetDict[classKey]
        # create ResNet18 class
        from .ResNet18 import ResNet18
        resnet18 = ResNet18(BaseNetClass)

        # import models functions
        from .ResNet18 import ResNet18_Cifar as resnet18_cifar10
        from .ResNet18 import ResNet18_Cifar as resnet18_cifar100
        from .ResNet18 import ResNet18_Imagenet as resnet18_imagenet
        # group models functions
        modelFuncs = [resnet18_cifar10, resnet18_cifar100, resnet18_imagenet]
        # apply models functions
        models = [modelFunc(resnet18) for modelFunc in modelFuncs]

        return {k: v for k, v in zip(ResNetSwitcher._ResNetModels, models)}
