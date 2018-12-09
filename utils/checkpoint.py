from shutil import copyfile

from torch import save as saveModel
from torch import load as loadModel

checkpointFileType = 'pth.tar'
stateFilenameDefault = 'model'
stateCheckpointPattern = '{}/{}_checkpoint.' + checkpointFileType
stateOptModelPattern = '{}/{}_opt.' + checkpointFileType


def save_state(state, is_best, path, filename):
    default_filename = stateCheckpointPattern.format(path, filename)
    saveModel(state, default_filename)

    is_best_filename = None
    if is_best:
        is_best_filename = stateOptModelPattern.format(path, filename)
        copyfile(default_filename, is_best_filename)

    return default_filename, is_best_filename


def save_checkpoint(path, model, args, best_prec1, is_best=False, filename=None):
    print('*** save_checkpoint ***')
    # set state dictionary
    state = dict(state_dict=model.state_dict(), best_prec1=best_prec1, learning_rate=args.learning_rate)
    # set state filename
    filename = filename or stateFilenameDefault
    # save state to file
    filePaths = save_state(state, is_best, path=path, filename=filename)

    return state, filePaths
