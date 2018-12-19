from os import listdir, remove
from os.path import isfile, isdir, exists
from shutil import copy2

from torch import load, save

from models.BaseNet import BaseNet
from trainRegimes.regime import TrainRegime
from utils.statistics import Statistics
from utils.checkpoint import checkpointFileType, blocksPartitionKey
from utils.training import TrainingData

_flopsKey = Statistics.flopsKey()
_partitionKey = BaseNet.partitionKey()
_avgKey = TrainingData.avgKey()


def generateCSV(folderPath):
    # results dictionary, the keys are the partitions, the values are dictionary of (flops, list of results)
    data = {}
    # partition keys as lists in order to iterate them sorted by number rather by strings
    # sort is different if sorting [0.5, 0.25, 0.75] as float or as string
    partitionKeys = []
    # key for results inner dictionary, the key of results list
    _resultsKey = 'results'

    # iterate over files
    for file in sorted(listdir(folderPath)):
        fPath = '{}/{}'.format(folderPath, file)
        if isfile(fPath):
            checkpoint = load(fPath)
            # get attributes from checkpoint
            try:
                baselineFlops = getattr(checkpoint, BaseNet.baselineFlopsKey())
                flops = baselineFlops.get(BaseNet.partitionKey())
                validAcc = getattr(checkpoint, TrainRegime.validAccKey)
                repeatNum = int(file[file.rfind('-') + 1:file.rfind(checkpointFileType) - 1])
                partition = getattr(checkpoint, blocksPartitionKey)
            except Exception as e:
                print(file)
                # remove(fPath)
                continue

            partitionStr = str(partition)
            if partitionStr not in data:
                data[partitionStr] = {_flopsKey: flops, _resultsKey: [None] * 5}
                partitionKeys.append(partition)

            # update checkpoint result in results list
            results = data[partitionStr][_resultsKey]
            results[repeatNum - 1] = validAcc

    # iterate (sorted) over partitions results
    for partition in reversed(sorted(partitionKeys, key=lambda x: [10] * len(x) if isinstance(x[0], str) else x)):
        partitionStr = str(partition)
        partitionData = data[partitionStr]

        flops = partitionData[_flopsKey]
        resultsList = partitionData[_resultsKey]
        resultsStr = ''
        for r in resultsList:
            resultsStr += ',{:.3f}'.format(r.get(BaseNet.partitionKey())) if r else ','
        print('"{}",{}{}'.format(partition, flops, resultsStr))

    print('Partition,Flops,1,2,3,4,5')


def updateCheckpointAttribute(folderPath, attribute, flopsDict):
    for file in sorted(listdir(folderPath)):
        fPath = '{}/{}'.format(folderPath, file)
        if isfile(fPath):
            print(fPath)
            checkpoint = load(fPath)
            setattr(checkpoint, attribute, flopsDict)
            save(checkpoint, fPath)


def buildCheckpoint(folderPath, checkpointSrcPrefix, nBlocks):
    # init flops per width ratio dictionary
    flopsDict = {0.25: 2633728.0, 0.5: 10313728.0, 0.75: 23040000.0, 1.0: 40812544.0}
    # add average flops to dict
    flopsDict[_avgKey] = [v for k, v in flopsDict.items()]
    flopsDict[_avgKey] = sum(flopsDict[_avgKey]) / len(flopsDict[_avgKey])
    # init checkpoint index per key
    checkpointIdx = {k: 1 for k in flopsDict.keys()}
    # init checkpoint source path per key
    checkpointSrcPath = {width: '{}/{}-{}.{}'.format(folderPath, checkpointSrcPrefix, [width] * nBlocks, checkpointFileType)
                         for width in flopsDict.keys()}
    # iterate over folders in folderPath
    for file in listdir(folderPath):
        fPath = '{}/{}'.format(folderPath, file)
        if isdir(fPath):
            folderCheckpointPath = '{}/train/model_opt.pth.tar'.format(fPath)
            if exists(folderCheckpointPath):
                # load folder checkpoint
                checkpoint = load(folderCheckpointPath)
                # get accuracy dictionary from checkpoint
                accDict = checkpoint.get('best_prec1')
                # iterate over widths accuracy
                for width, acc in accDict.items():
                    # make copy of source checkpoint
                    checkpointDstName = '{}-{}-{}.{}'.format(checkpointSrcPrefix, [width] * nBlocks, checkpointIdx[width], checkpointFileType)
                    checkpointDstPath = '{}/{}'.format(folderPath, checkpointDstName)
                    copy2(checkpointSrcPath[width], checkpointDstPath)
                    # load new checkpoint
                    dstCheckpoint = load(checkpointDstPath)
                    # set attributes in destination checkpoint
                    attributes = [(TrainRegime.validAccKey, {BaseNet.partitionKey(): acc}), (blocksPartitionKey, [width] * nBlocks),
                                  (BaseNet.baselineFlopsKey(), {BaseNet.partitionKey(): flopsDict[width]})]
                    for attrKey, attrValue in attributes:
                        setattr(dstCheckpoint, attrKey, attrValue)
                    # save new checkpoint
                    save(dstCheckpoint, checkpointDstPath)
                    print('Updated values {} in checkpoint {}'.format(attributes, checkpointDstName))
                    # update checkpoint index
                    checkpointIdx[width] += 1

    # remove source checkpoints
    for width in flopsDict.keys():
        remove('{}/{}-{}.{}'.format(folderPath, checkpointSrcPrefix, [width] * nBlocks, checkpointFileType))


def buildWidthRatioMissingCheckpoints(widthRatio, nBlocks):
    checkpointSrcPrefix = '[resnet18]-[cifar10]'

    for dataset in ['cifar10', 'cifar100']:
        for pre_trained in ['with_pre_trained', 'without_pre_trained']:
            folderPath = '/home/vista/Desktop/Architecture_Search/results/{}/width:{}/{}'.format(dataset, widthRatio, pre_trained)

            buildCheckpoint(folderPath, checkpointSrcPrefix, nBlocks)
            generateCSV(folderPath)
            print(folderPath)
            print('=================================================')


# folderPath should be a path to folder which has folders inside
# each inner folder will be the title for the checkpoints in it
def plotFolders(folderPath):
    # init flops data with inner folders as keys, [] as values
    flopsData = {}
    # init labels to connect dictionary
    labelsToConnect = dict(with_pre_trained=[], without_pre_trained=[])
    # iterate over folders
    for folder in listdir(folderPath):
        fPath = '{}/{}'.format(folderPath, folder)
        if isdir(fPath):
            # add to labelsToConnect dict
            for key in labelsToConnect:
                if key in folder:
                    labelsToConnect[key].append(folder)
                    break
            # init empty list under folder key
            flopsData[folder] = []
            # iterate over checkpoints
            for file in listdir(fPath):
                filePath = '{}/{}'.format(fPath, file)
                if isfile(filePath):
                    checkpoint = load(filePath)
                    # get attributes from checkpoint
                    try:
                        baselineFlops = getattr(checkpoint, BaseNet.baselineFlopsKey())
                        flops = baselineFlops.get(BaseNet.partitionKey())
                        validAcc = getattr(checkpoint, TrainRegime.validAccKey)
                        validAcc = validAcc.get(BaseNet.partitionKey())
                        repeatNum = int(file[file.rfind('-') + 1:file.rfind(checkpointFileType) - 1])
                        # partition = getattr(checkpoint, blocksPartitionKey)
                    except Exception as e:
                        print('Missing values in {}'.format(file))
                        # remove(filePath)
                        continue

                    # add attributes to plot data
                    flopsData[folder].append((repeatNum, flops, validAcc))

    # plot
    Statistics.plotFlops(flopsData, list(labelsToConnect.values()), 'acc_vs_flops_summary', folderPath)


widthRatio = [0.25, 0.5, 0.75, 1.0]
dataset = 'cifar10'
# folderPath = '/home/vista/Desktop/Architecture_Search/results/{}/width:{}/'.format(dataset, widthRatio)
folderPath = '/home/vista/Desktop/Architecture_Search/results/{}/individual_training'.format(dataset)

# buildWidthRatioMissingCheckpoints(widthRatio, nBlocks=3)
plotFolders(folderPath)
# generateCSV(folderPath)
