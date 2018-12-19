from shutil import copyfile

from torch import save as saveModel

checkpointFileType = 'pth.tar'
stateFilenameDefault = 'model'
stateCheckpointPattern = '{}/{}_checkpoint.' + checkpointFileType
stateOptModelPattern = '{}/{}_opt.' + checkpointFileType
blocksPartitionKey = 'blocksPartition'


def save_state(state, is_best, path, filename):
    default_filename = stateCheckpointPattern.format(path, filename)
    saveModel(state, default_filename)

    is_best_filename = None
    if is_best:
        is_best_filename = stateOptModelPattern.format(path, filename)
        copyfile(default_filename, is_best_filename)

    return default_filename, is_best_filename


def save_checkpoint(path, model, optimizer, best_prec1, is_best=False, filename=None):
    print('*** save_checkpoint ***')
    # set state dictionary
    state = dict(state_dict=model.state_dict(), best_prec1=best_prec1, optimizer=optimizer.state_dict())
    # set state filename
    filename = filename or stateFilenameDefault
    # save state to file
    filePaths = save_state(state, is_best, path=path, filename=filename)

    return state, filePaths


def generate_partitions(args, blocksPermutationList, modelBlocks):
    nBlocks, nLayersPerBlock = modelBlocks

    # add next block permutations
    def addBlockPerms(currPermTuple, nLayers):
        currPerm, currPermByBlock = currPermTuple
        newPerms = []
        for v in blocksPermutationList:
            newPerms.append((currPerm + ([v] * nLayers), currPermByBlock + [v]))

        return newPerms

    perms = addBlockPerms(([], []), nLayersPerBlock[0])

    for blockIdx in range(1, nBlocks):
        newPerms = []
        for i in range(len(perms)):
            perms[i] = addBlockPerms(perms[i], nLayersPerBlock[blockIdx])
            newPerms.extend(perms[i])
        perms = newPerms

    # generate checkpoints
    for partition, blocksPartition in perms:
        # update partition
        args.partition = partition
        setattr(args, blocksPartitionKey, blocksPartition)
        # save checkpoint
        saveModel(args, '[{}]-[{}]-{}.{}'.format(args.model, args.dataset, blocksPartition, checkpointFileType))
        # print(blocksPartition)
