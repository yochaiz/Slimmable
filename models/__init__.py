_modelsNames = ['resnet18_cifar10', 'resnet18_cifar100', 'resnet18_imagenet']


def getModelNames():
    return _modelsNames


def getModelDict():
    from .ResNet18 import ResNet18_Cifar as resnet18_cifar10
    from .ResNet18 import ResNet18_Cifar as resnet18_cifar100
    from .ResNet18 import ResNet18_Imagenet as resnet18_imagenet

    models = [resnet18_cifar10, resnet18_cifar100, resnet18_imagenet]
    return {k: v for k, v in zip(_modelsNames, models)}
