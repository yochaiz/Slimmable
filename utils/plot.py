from os import listdir, remove
from os.path import isfile, isdir, exists
from shutil import copy2

from torch import load, save

from models.BaseNet import BaseNet
from trainRegimes.regime import TrainRegime
from utils.statistics import Statistics
from utils.checkpoint import checkpointFileType

_baselineFlopsKey = 'baselineFlops'
_blocksPartitionKey = 'blocksPartition'
_flopsKey = Statistics.flopsKey()
_partitionKey = BaseNet.partitionKey()


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
                baselineFlops = getattr(checkpoint, _baselineFlopsKey)
                flops = baselineFlops.get(BaseNet.partitionKey())
                validAcc = getattr(checkpoint, TrainRegime.validAccKey)
                repeatNum = int(file[file.rfind('-') + 1:file.rfind(checkpointFileType) - 1])
                partition = getattr(checkpoint, _blocksPartitionKey)
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
    for partition in reversed(sorted(partitionKeys)):
        partitionStr = str(partition)
        partitionData = data[partitionStr]

        flops = partitionData[_flopsKey]
        resultsList = partitionData[_resultsKey]
        resultsStr = ''
        for r in resultsList:
            resultsStr += ',{:.3f}'.format(r.get(BaseNet.partitionKey())) if r else ','
        print('"{}",{}{}'.format(partition, flops, resultsStr))

    print('Partition,Bops,1,2,3,4,5')


def updateCheckpointAttribute(folderPath, attribute, flopsDict):
    for file in sorted(listdir(folderPath)):
        fPath = '{}/{}'.format(folderPath, file)
        if isfile(fPath):
            print(fPath)
            checkpoint = load(fPath)
            setattr(checkpoint, attribute, flopsDict)
            save(checkpoint, fPath)


def buildCheckpoint(folderPath, checkpointFilename):
    checkpointSrcPath = '{}/{}'.format(folderPath, checkpointFilename)
    checkpointSrcPrefix = checkpointFilename[:-len(checkpointFileType) - 1]
    # init checkpoint index
    checkpointIdx = 1
    # check checkpoint source file exists
    if exists(checkpointSrcPath):
        # iterate over folders in folderPath
        for file in listdir(folderPath):
            fPath = '{}/{}'.format(folderPath, file)
            if isdir(fPath):
                folderCheckpointPath = '{}/train/model_opt.pth.tar'.format(fPath)
                if exists(folderCheckpointPath):
                    # load folder checkpoint
                    checkpoint = load(folderCheckpointPath)
                    # get values from checkpoint
                    acc = checkpoint.get('best_prec1')
                    if isinstance(acc, dict):
                        acc = acc.get(next(iter(acc)))
                    acc = {BaseNet.partitionKey(): acc}
                    # acc = getattr(checkpoint, TrainRegime.validAccKey, None)
                    # loss = getattr(checkpoint, TrainRegime.validLossKey, None)
                    # make copy of source checkpoint
                    checkpointDstName = '{}-{}.{}'.format(checkpointSrcPrefix, checkpointIdx, checkpointFileType)
                    checkpointDstPath = '{}/{}'.format(folderPath, checkpointDstName)
                    copy2(checkpointSrcPath, checkpointDstPath)
                    # load new checkpoint
                    dstCheckpoint = load(checkpointDstPath)
                    # set attributes in destination checkpoint
                    setattr(dstCheckpoint, TrainRegime.validAccKey, acc)
                    # save new checkpoint
                    save(dstCheckpoint, checkpointDstPath)
                    print('Updated values {} in checkpoint {}'.format([acc], checkpointDstName))
                    # update checkpoint index
                    checkpointIdx += 1


def buildWidthRatioMissingCheckpoints(widthRatio, flops):
    flopsDict = {_partitionKey: flops}
    checkpointFilename = '[resnet18]-[cifar10]-{}.pth.tar'.format([widthRatio] * 3)

    for dataset in ['cifar10', 'cifar100']:
        for pre_trained in ['with_pre_trained', 'without_pre_trained']:
            folderPath = '/home/vista/Desktop/Architecture_Search/results/{}/width:[{}]/{}/'.format(dataset, widthRatio, pre_trained)

            buildCheckpoint(folderPath, checkpointFilename)
            remove('{}/{}'.format(folderPath, checkpointFilename))
            updateCheckpointAttribute(folderPath, _baselineFlopsKey, flopsDict)
            updateCheckpointAttribute(folderPath, _blocksPartitionKey, [widthRatio] * 3)
            generateCSV(folderPath)


# folderPath should be a path to folder which has folders inside
# each inner folder will be the title for the checkpoints in it
def plotFolders(folderPath):
    # init flops data with inner folders as keys, [] as values
    flopsData = {}
    # init plot data
    plotData = {_flopsKey: flopsData}
    # iterate over folders
    for folder in listdir(folderPath):
        fPath = '{}/{}'.format(folderPath, folder)
        if isdir(fPath):
            # init empty list under folder key
            flopsData[folder] = []
            # iterate over checkpoints
            for file in listdir(fPath):
                filePath = '{}/{}'.format(fPath, file)
                if isfile(filePath):
                    checkpoint = load(filePath)
                    # get attributes from checkpoint
                    try:
                        baselineFlops = getattr(checkpoint, _baselineFlopsKey)
                        flops = baselineFlops.get(BaseNet.partitionKey())
                        validAcc = getattr(checkpoint, TrainRegime.validAccKey)
                        validAcc = validAcc.get(BaseNet.partitionKey())
                        repeatNum = int(file[file.rfind('-') + 1:file.rfind(checkpointFileType) - 1])
                        partition = getattr(checkpoint, _blocksPartitionKey)
                    except Exception as e:
                        print('Missing values in {}'.format(file))
                        # remove(filePath)
                        continue

                    # add attributes to plot data
                    flopsData[folder].append((repeatNum, flops, validAcc))

    # plot
    Statistics.plotFlops(plotData, _flopsKey, None, folderPath)


widthRatio = 1.0
dataset = 'cifar100'
folderPath = '/home/vista/Desktop/Architecture_Search/results/{}/width:[{}]'.format(dataset, widthRatio)

# buildWidthRatioMissingCheckpoints(widthRatio, flops=40812544.0)
# plotFolders(folderPath)
# generateCSV(folderPath)
