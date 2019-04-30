from os import listdir, remove
from os.path import isfile, isdir, exists
from shutil import copy2
from ast import literal_eval
from numpy import linspace

from torch import load, save

from models.BaseNet.BaseNet import BaseNet
from utils.trainWeights import TrainWeights
from utils.flopsPlot import plotFlopsData, FlopsPlot, PlotLabelData, plt
from utils.checkpoint import checkpointFileType, blocksPartitionKey
from utils.training import TrainingData

_flopsKey = FlopsPlot.flopsKey()
_partitionKey = BaseNet.partitionKey()
_avgKey = TrainingData.avgKey()
_titleKey = FlopsPlot.getTitleKey()


def getPartitionFlops(checkpoint):
    baselineFlops = getattr(checkpoint, BaseNet.baselineFlopsKey())
    flops = baselineFlops[BaseNet.partitionKey()]
    return flops


def getPartitionValidAcc(checkpoint):
    validAcc = getattr(checkpoint, TrainWeights.validAccKey)
    validAcc = validAcc[BaseNet.partitionKey()]
    return validAcc


def extractAttributesFromCheckpoint(file, checkpoint):
    # get attributes from checkpoint
    flops = getPartitionFlops(checkpoint)
    validAcc = getPartitionValidAcc(checkpoint)
    repeatNum = int(file[file.rfind('-') + 1:file.rfind(checkpointFileType) - 1])

    return flops, validAcc, repeatNum


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
                partition = getattr(checkpoint, blocksPartitionKey)
                flops, validAcc, repeatNum = extractAttributesFromCheckpoint(file, checkpoint)
            except Exception as e:
                print('Missing values in {}'.format(file))
                # remove(fPath)
                continue

            partitionStr = str(partition)
            if partitionStr not in data:
                data[partitionStr] = {_flopsKey: flops, _resultsKey: [','] * 7}
                partitionKeys.append(partition)

            # update checkpoint result in results list
            data[partitionStr][_flopsKey] = min(data[partitionStr][_flopsKey], flops)
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
            resultsStr += ',{:.3f}'.format(r) if isinstance(r, float) else r
        print('"{}",{}{}'.format(partition, flops, resultsStr))

    print('Partition,Flops,1,2,3,4,5')


def updateCheckpointAttribute(folderPath, attribute, value):
    for file in sorted(listdir(folderPath)):
        fPath = '{}/{}'.format(folderPath, file)
        if isfile(fPath):
            print(fPath)
            checkpoint = load(fPath)
            setattr(checkpoint, attribute, value)
            save(checkpoint, fPath)


def updateCheckpointBlocksPartition(folderPath):
    startTag = ']-['
    endTag = ']-'
    for file in sorted(listdir(folderPath)):
        fPath = '{}/{}'.format(folderPath, file)
        if isfile(fPath):
            print(fPath)
            checkpoint = load(fPath)
            # extract blocks partition from file name
            blocksPartition = file[file.rfind(startTag) + len(startTag) - 1: file.rfind(endTag) + 1]
            blocksPartition = literal_eval(blocksPartition)
            setattr(checkpoint, blocksPartitionKey, blocksPartition)
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
                    attributes = [(TrainWeights.validAccKey, {BaseNet.partitionKey(): acc}), (blocksPartitionKey, [width] * nBlocks),
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


def fixCheckpointFlops(folderPath):
    modelFileName = 'model.html'
    # init array of folders name we need to fix their checkpoint flops dictionary
    foldersToFix = []
    # iterate over folder and mark the folders checkpoint we have to fix their flops dictionary
    for folder in sorted(listdir(folderPath)):
        fPath = '{}/{}'.format(folderPath, folder)
        if isdir(fPath):
            modelFilePath = '{}/{}'.format(fPath, modelFileName)
            if exists(modelFilePath):
                with open(modelFilePath, 'r') as modelFile:
                    # read model HTML file
                    modelStr = modelFile.read()
                    # count how many conv with kernel_size=(1, 1) exist
                    nDownsample = modelStr.count('kernel_size=(1, 1)')
                    if nDownsample > 2:
                        foldersToFix.append(folder)

    # load the wrong checkpoints and fix their flops dictionary
    for folder in foldersToFix:
        checkpointPath = '{}/{}.{}'.format(folderPath, folder, checkpointFileType)
        checkpoint = load(checkpointPath)
        # make sure checkpoint is ready, i.e. we are not in the middle of training
        if hasattr(checkpoint, TrainWeights.validAccKey):
            # reset flops dictionary
            setattr(checkpoint, BaseNet.baselineFlopsKey(), None)
            save(checkpoint, checkpointPath)


def plotMethods(baselineFoldersPath, methodsFoldersPath):
    # init flops data with inner folders as keys, [] as values
    flopsData = {}
    # init labels to connect list
    labelsToConnect = []

    # init colors
    colormap = plt.cm.hot
    baselineColorVal = 0.7
    # init baseline color
    baselineColor = colormap(baselineColorVal)

    # iterate over baseline folders
    folderPath = baselineFoldersPath
    for folder in sorted(listdir(folderPath)):
        fPath = '{}/{}'.format(folderPath, folder)
        if isdir(fPath):
            # set label
            label = folder
            # create PlotLabelData instance
            labelData = PlotLabelData(label, label, baselineColor)
            # add PlotLabelData instance to folder FlopsData dictionary
            flopsData[label] = labelData
            # add to labelsToConnect list
            labelsToConnect.append(label)
            # iterate over checkpoints
            for file in listdir(fPath):
                filePath = '{}/{}'.format(fPath, file)
                # load checkpoint
                checkpoint = load(filePath)
                # add checkpoint
                labelData.addCheckpoint(checkpoint)

    # iterate over methods folders
    folderPath = methodsFoldersPath
    foldersList = listdir(folderPath)

    # init list of colors for checkpoints per epoch
    colors = [colormap(i) for i in linspace(baselineColorVal, 0.0, len(foldersList) + 1)]
    # init color index
    colorIdx = 0

    # init dictionary for methods checkpoints
    filesFlopsData = {}

    newList = [foldersList[0]] + [foldersList[3]] + [foldersList[2]] + [foldersList[1]]

    for folder in newList:
        print(folder)
        fPath = '{}/{}'.format(folderPath, folder)
        # update color index
        colorIdx += 1
        if isdir(fPath):
            # set label
            label = folder
            # init folder checkpoints dictionary
            filesFlopsData[label] = {}
            # iterate over method checkpoints
            for file in sorted(listdir(fPath)):
                checkpointPath = '{}/{}'.format(fPath, file)
                # load checkpoint
                checkpoint = load(checkpointPath)
                # set checkpoint label
                checkpointLabel = '[{}]-[{}]'.format(checkpoint.epoch, checkpoint.id)
                if checkpointLabel not in filesFlopsData[label]:
                    filesFlopsData[label][checkpointLabel] = PlotLabelData(label, '', colors[colorIdx])

                filesFlopsData[label][checkpointLabel].addCheckpoint(checkpoint)

    for folderLabel, folderDict in filesFlopsData.items():
        for checkpointLabel, checkpointData in folderDict.items():
            newLabel = '{}-{}'.format(folderLabel, checkpointLabel)
            flopsData[newLabel] = checkpointData

    # plot
    partitionAttrKey = 'partition'
    partitionFunc = lambda checkpoint: getattr(checkpoint, partitionAttrKey, None)
    plotFlopsData(flopsData.values(), (getPartitionFlops, getPartitionValidAcc, partitionFunc), [labelsToConnect],
                  't', folderPath)


# folderPath should be a path to folder which has folders inside
# each inner folder will be the title for the checkpoints in it
def plotFolders(folderPath):
    # init flops data with inner folders as keys, [] as values
    flopsData = {}
    filesFlopsData = {}
    # init labels to connect list
    labelsToConnect = []
    # Map each partition to index, for easier mapping in plot
    partitionIdxMap = {}
    # init colors
    colormap = plt.cm.hot
    folderColorVal = 0.7
    # init folders color
    # folderColor = colormap(folderColorVal)
    folderColor = plt.cm.cool(0.1)
    # iterate over folders
    for folder in sorted(listdir(folderPath)):
        fPath = '{}/{}'.format(folderPath, folder)
        try:
            if isdir(fPath):
                # set label
                label = folder
                # create PlotLabelData instance
                labelData = PlotLabelData(label, label, folderColor)
                # add PlotLabelData instance to folder FlopsData dictionary
                flopsData[label] = labelData
                # add to labelsToConnect list
                labelsToConnect.append(label)
                # iterate over checkpoints
                for file in listdir(fPath):
                    filePath = '{}/{}'.format(fPath, file)
                    if isfile(filePath):
                        # repeatNum = int(file[file.rfind('-') + 1:file.rfind(checkpointFileType) - 1])
                        # load checkpoint
                        checkpoint = load(filePath)
                        # check accuracy exists in checkpoint
                        if hasattr(checkpoint, TrainWeights.validAccKey):
                            # set title to checkpoint
                            setattr(checkpoint, _titleKey, label)
                            # add checkpoint
                            labelData.addCheckpoint(checkpoint)
            elif isfile(fPath):
                partitionAttrKey = 'partition'
                # partitionAttrKey = blocksPartitionKey
                # load checkpoint
                checkpoint = load(fPath)
                # check accuracy exists in checkpoint
                if hasattr(checkpoint, TrainWeights.validAccKey):
                    # get partition from checkpoint
                    partition = str(getattr(checkpoint, partitionAttrKey))
                    # create partition key in flopsData dict if does not exist
                    if partition not in partitionIdxMap:
                        idx = len(partitionIdxMap)
                        partitionIdxMap[partition] = idx
                    else:
                        idx = partitionIdxMap[partition]
                    # set partition key
                    # partitionKey = '{}-[{}]'.format(partition, idx)
                    partitionKey = '[{}]-[{}]'.format(checkpoint.epoch, checkpoint.id)
                    # set label
                    # label = '[{}]'.format(idx)
                    label = partitionKey
                    # label = ''
                    # set label to checkpoint
                    setattr(checkpoint, _titleKey, label)

                    # save checkpoints by epoch and ID in order to be able to sort them by epoch
                    epoch = checkpoint.epoch
                    id = checkpoint.id
                    if epoch not in filesFlopsData:
                        filesFlopsData[epoch] = {}
                    if id not in filesFlopsData[epoch]:
                        # add PlotLabelData instance to folder FlopsData dictionary
                        filesFlopsData[epoch][id] = PlotLabelData(partitionKey, label)

                    # add checkpoint
                    filesFlopsData[epoch][id].addCheckpoint(checkpoint)

                    # # check if partitionKey exits under folderName
                    # if partitionKey not in flopsData:
                    #     # add PlotLabelData instance to folder FlopsData dictionary
                    #     flopsData[partitionKey] = PlotLabelData(partitionKey, label)
                    # # add checkpoint
                    # flopsData[partitionKey].addCheckpoint(checkpoint)

        except Exception as e:
            print('Missing values in {}'.format(folder))
            # remove(fPath)
            continue

    # # init colors
    # colormap = plt.cm.hot
    # colors = [colormap(i) for i in linspace(folderColorVal, 0.0, len(flopsData.keys()))]
    # # set color to each key
    # for idx, v in enumerate(flopsData.values()):
    #     v.setColor(colors[idx])

    # init list of colors for checkpoints per epoch
    colors = [colormap(i) for i in linspace(folderColorVal, 0.0, len(filesFlopsData.keys()) + 1)]
    # set next color index
    nextColorIdx = 1
    #
    # iterate over checkpoints: set them color per epoch and add them sorted to flopsData
    for epoch in sorted(filesFlopsData.keys()):
        for id in sorted(filesFlopsData[epoch].keys()):
            # get checkpoint
            elem = filesFlopsData[epoch][id]
            # set color
            elem.setColor(colors[nextColorIdx])
            # add to flopsData
            flopsData[elem.legendString()] = elem
        # set new color to new epoch
        nextColorIdx += 1

    # homoFlopsData = {}
    # for label in labelsToConnect:
    #     v = flopsData[label]
    #     homoFlopsData[label] = PlotLabelData(v.legendString() + '_', v.annotateString(), v.color())
    #     for c in v.checkpoints():
    #         homoFlopsData[label].addCheckpoint(c)

    # plot
    partitionFunc = lambda checkpoint: getattr(checkpoint, partitionAttrKey, None)
    plotFlopsData(flopsData.values(), (getPartitionFlops, getPartitionValidAcc, partitionFunc), [labelsToConnect],
                  'acc_vs_flops_summary', folderPath)

    # 'acc_vs_flops_summary'


def plotCompareFolders(foldersList):
    # init colors
    colormap = plt.cm.hot
    colors = [colormap(i) for i in linspace(0.7, 0.0, len(foldersList))]
    # init flops data with inner folders as keys, [] as values
    flopsData = {}
    # init labels to connect list
    labelsToConnect = []
    # Map each partition to index, for easier mapping in plot
    partitionIdxMap = {}
    # iterate over folders in foldersList
    for folderIdx, (folderPath, folderName) in enumerate(foldersList):
        # add folderName as key to flopsData
        flopsData[folderName] = {}
        folderFlopsData = flopsData[folderName]
        folderLabelsToConnect = []
        labelsToConnect.append(folderLabelsToConnect)
        # iterate over checkpoints in folder
        for name in sorted(listdir(folderPath)):
            fPath = '{}/{}'.format(folderPath, name)
            if isdir(fPath):
                # set label
                label = name
                # create PlotLabelData instance
                legendString = '[{}]-{}'.format(folderName, label)
                labelData = PlotLabelData(legendString, label, colors[folderIdx])
                # add PlotLabelData instance to folder FlopsData dictionary
                folderFlopsData[label] = labelData
                # add to labelsToConnect list
                folderLabelsToConnect.append(legendString)
                # iterate over checkpoints
                for file in listdir(fPath):
                    filePath = '{}/{}'.format(fPath, file)
                    if isfile(filePath):
                        # repeatNum = int(file[file.rfind('-') + 1:file.rfind(checkpointFileType) - 1])
                        # load checkpoint
                        checkpoint = load(filePath)
                        # check accuracy exists in checkpoint
                        if hasattr(checkpoint, TrainWeights.validAccKey):
                            # set label to checkpoint
                            setattr(checkpoint, _titleKey, label)
                            # add checkpoint
                            labelData.addCheckpoint(checkpoint)
            elif isfile(fPath) and fPath.endswith(checkpointFileType):
                # partitionAttrKey = 'partition'
                partitionAttrKey = blocksPartitionKey
                # load checkpoint
                checkpoint = load(fPath)
                # check accuracy exists in checkpoint
                if hasattr(checkpoint, TrainWeights.validAccKey):
                    # get partition from checkpoint
                    partition = str(getattr(checkpoint, partitionAttrKey))
                    # create partition key in flopsData dict if does not exist
                    if partition not in partitionIdxMap:
                        idx = len(partitionIdxMap)
                        partitionIdxMap[partition] = idx
                    else:
                        idx = partitionIdxMap[partition]
                    # set partition key
                    partitionKey = '{}-[{}]'.format(partition, idx)
                    # set label
                    label = '[{}]'.format(idx)
                    # set label to checkpoint
                    setattr(checkpoint, _titleKey, label)
                    # check if partitionKey exits under folderName
                    if partitionKey not in folderFlopsData:
                        # add PlotLabelData instance to folder FlopsData dictionary
                        folderFlopsData[partitionKey] = PlotLabelData('[{}]-{}'.format(folderName, partitionKey), label, colors[folderIdx])
                    # add checkpoint
                    folderFlopsData[partitionKey].addCheckpoint(checkpoint)

    # put all PlotLabelData instances in list
    labelsList = []
    for v in flopsData.values():
        labelsList.extend(v.values())

    # group partition label from all folders, in order to connect them in plot
    # labelsDict = {}
    # for labelData in labelsList:
    #     key = labelData.annotateString()
    #     if key not in labelsDict:
    #         labelsDict[key] = []
    #     labelsDict[key].append(labelData.legendString())
    # labelsToConnect = list(labelsDict.values())

    # plot
    partitionFunc = lambda checkpoint: getattr(checkpoint, partitionAttrKey, None)
    plotFlopsData(labelsList, (getPartitionFlops, getPartitionValidAcc, partitionFunc), labelsToConnect,
                  'acc_vs_flops_folders_summary', folderPath)


# widthRatio = [0.25, 0.5, 0.75, 1.0]
# dataset = 'imagenet'
# basePath = '/home/vista/Desktop/Architecture_Search/results/{}/'.format(dataset)

folderPath = '/home/vista/Desktop/Architecture_Search/results/imagenet/checkpoints-imagenet'
# folderPath = '/home/vista/Desktop/Architecture_Search/results_block_binomial/ended/plot_compare_methods'
# folderPath = '/home/vista/Desktop/F-BANNAS_depracated/6.12/Updated'
baselinePath = '{}/baseline'.format(folderPath)
methodsPath = '{}/methods'.format(folderPath)

# buildWidthRatioMissingCheckpoints(widthRatio, nBlocks=3)
# updateCheckpointBlocksPartition(folderPath)
plotFolders(folderPath)
# plotMethods(baselinePath, methodsPath)
# generateCSV(folderPath)
# fixCheckpointFlops('/home/vista/Desktop/Architecture_Search/results/mixed_training')
# plotCompareFolders([('{}/{}'.format(basePath, x), x) for x in ['mixed_training', 'individual_training']])
