from collections import OrderedDict
from json import load as jsonLoad

from torch import load
from torch.optim.sgd import SGD
from torch.nn.modules.batchnorm import BatchNorm2d

from models.BaseNet.BaseNet import BaseNet
from trainRegimes.OptimalRegime import OptimalRegime
from utils.HtmlLogger import HtmlLogger


def loadWeights(model, checkpoint):
    # update downsample keys
    chkpntStateDict = checkpoint['state_dict']
    newStateDict = OrderedDict()

    tokenOrg = '.downsample.'
    tokenNew = '.downsample.downsampleSrc.'
    tokenOther = '.downsample.downsample.'
    for key, v in chkpntStateDict.items():
        if tokenOther in key:
            continue
        if (tokenNew not in key) and (tokenOrg in key):
            newStateDict[key.replace(tokenOrg, tokenNew)] = v
        else:
            newStateDict[key] = v

    # update unused downsample keys
    currStateDict = model.state_dict()
    for key, v in currStateDict.items():
        if tokenOrg in key:
            if key not in newStateDict:
                newStateDict[key] = v

    # load state dict
    model.load_state_dict(newStateDict)


# get the indices of partition path, but through the homogeneous BNs
def getPartitionHomogeneousPathIdx(model, partition):
    assert (len(partition) == len(model.blocks))
    partitionHomoPath = []
    for widthRatio, block in zip(partition, model.blocks):
        for layer in block.getOptimizationLayers():
            idx = layer.widthRatioIdx(widthRatio)
            partitionHomoPath.append(idx)

    return partitionHomoPath


# compare model partition accuracy with accuracy of model partition, but through the homogeneous BNs path
def partitionVsHomoPartition(regime, model, partition, logger):
    # update baseline widths
    _partitionKey = BaseNet.partitionKey()
    _partitionThroughHomogeneousKey = 'Partition (homogeneous)'
    newBaselineWidth = {}
    # add current partition
    if _partitionKey in model._baselineWidth:
        newBaselineWidth[_partitionKey] = model._baselineWidth[_partitionKey]
    # add partition through homogeneous
    newBaselineWidth[_partitionThroughHomogeneousKey] = getPartitionHomogeneousPathIdx(model, partition)
    # update model baseline widths
    model._baselineWidth = newBaselineWidth
    # infer
    return regime.trainWeights.inferEpoch(0, {regime.trainWeights.trainLoggerKey: logger})


# p = "../results/cifar10/['Avg']-[mix]-[without_pre_trained]/[resnet18]-[cifar10]-['Avg', 'Avg', 'Avg']-2.pth.tar"
p = "../results/[resnet18],[cifar10],[0.0],[0.25, 0.5, 0.75, 1.0],[20190108-190437]/args.txt"
# args = load(p)
with open(p, 'r') as f:
    argsDict = jsonLoad(f)
from argparse import Namespace

args = Namespace()
for k, v in argsDict.items():
    setattr(args, k, v)

nEpochs = 2

savePath = '../results/path_evaluation/pre_trained:[heterogeneous]/[resnet18],[cifar10],[0.0],[0.25, 0.5, 0.75, 1.0],[20190108-190440]'
folderName = '[set-2]_train_BNs_only_[{}]_epochs'.format(nEpochs)
folderPath = '{}/{}'.format(savePath, folderName)
# init main logger
logger = HtmlLogger(folderPath, 'log')
# init regime
regime = OptimalRegime(args, logger)
# update model and modelParallel
model = regime.trainWeights.model

# checkpointPath = '{}/train/model_opt.pth.tar'.format(args.save)
checkpointPath = '{}/train/model_checkpoint.pth.tar'.format(args.save)
checkpoint = load(checkpointPath)
loadWeights(model, checkpoint)

# partitionVsHomoPartition(regime, model, args.partition, logger)

# partitionCheckpointList = ["../results/mixed_training/[resnet18]-[cifar10]-[0.75, 0.75, 1.0]-1.pth.tar",
#                            "../results/cifar10/mixed_training/[0.75]/[resnet18]-[cifar10]-[0.75, 0.75, 0.75]-1.pth.tar",
#                            #                            "../results/cifar10/mixed_training/[1.0]/[resnet18]-[cifar10]-[1.0, 1.0, 1.0]-1.pth.tar",
#                            #                            "../results/mixed_training/[resnet18]-[cifar10]-[0.75, 0.5, 1.0]-1.pth.tar",
#                            "../results/mixed_training/[resnet18]-[cifar10]-[0.75, 1.0, 0.25]-1.pth.tar",
#                            #                            "../results/mixed_training/[resnet18]-[cifar10]-[1.0, 0.25, 1.0]-1.pth.tar"
#                            ]

partitionCheckpointList = ["../results/mixed_training/[resnet18]-[cifar10]-[0.25, 0.75, 0.5]-1.pth.tar",
                           "../results/cifar10/mixed_training/[0.50]/[resnet18]-[cifar10]-[0.5, 0.5, 0.5]-1.pth.tar",
                           "../results/mixed_training/[resnet18]-[cifar10]-[0.75, 0.25, 0.25]-1.pth.tar"
                           ]

# partitionCheckpointList = ["../results/individual_training/[resnet18]-[cifar10]-[0.25, 0.5, 0.5]-1.pth.tar",
# "../results/individual_training/[resnet18]-[cifar10]-[0.25, 0.5, 0.5]-[0.375, 0.25, 0.25, 0.25, 0.5, 0.375, 0.375, 0.375, 0.375, 0.625]-1.pth.tar",
#  "../results/individual_training/[resnet18]-[cifar10]-[0.25, 0.5, 0.5]-[0.375, 0.25, 0.25, 0.375, 0.5, 0.625, 0.625, 0.625, 0.375, 0.5]-1.pth.tar"
#                            ]

# init new baseline width dictionary
newBaselineWidth = {}
# modifiedBaselineWidth = {}
for checkpointPath in partitionCheckpointList:
    checkpoint = load(checkpointPath)
    partition = checkpoint.partition
    partitionHomoPath = getPartitionHomogeneousPathIdx(model, partition)
    newBaselineWidth[str(partition)] = partitionHomoPath
    # # modify partition by single layer (1st one)
    # partition = [checkpoint.width[randint(0, layer.nWidths() - 1)] for layer in model.layersList()]
    # idx = randint(0, len(partition) - 1)
    # for w in checkpoint.width:
    #     partition[idx] = w
    #     partitionHomoPath = [layer.widthRatioIdx(widthRatio) for widthRatio, layer in zip(partition, model.layersList())]
    #     # partitionHomoPath = getPartitionHomogeneousPathIdx(model, partition)
    #     newBaselineWidth[str(partition)] = partitionHomoPath
    #     modifiedBaselineWidth[str(partition)] = partitionHomoPath

# update model baseline widths
model._baselineWidth = newBaselineWidth
# infer
regime.trainWeights.inferEpoch(0, {regime.trainWeights.trainLoggerKey: logger})

args.optimal_epochs = 5
args.learning_rate = 0.1
args.logInterval = 1

# init optimizer
bnParams = []
for m in model.modules():
    if isinstance(m, BatchNorm2d):
        for p in m.parameters():
            bnParams.append(p)
optimizer = SGD(bnParams, args.learning_rate,
                momentum=args.momentum, weight_decay=args.weight_decay)

for epoch in range(1, nEpochs + 1):
    # train weights only on modified partition
    # model._baselineWidth = modifiedBaselineWidth
    trainLogger = HtmlLogger('{}/train_weights'.format(folderPath), epoch)
    trainLogger.addInfoTable('Learning rates', [['optimizer_lr', optimizer.param_groups[0]['lr']]])
    regime.trainWeights.weightsEpoch(optimizer, epoch, {regime.trainWeights.trainLoggerKey: trainLogger})
    # # update model baseline widths
    # model._baselineWidth = newBaselineWidth
    # infer
    regime.trainWeights.inferEpoch(epoch, {regime.trainWeights.trainLoggerKey: logger})
