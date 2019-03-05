from argparse import ArgumentParser
from datetime import datetime
from numpy import random
from shutil import copy2
from os import listdir, remove
from os.path import dirname, basename, exists

from torch import load as loadCheckpoint
from torch import manual_seed as torch_manual_seed
from torch.cuda import manual_seed as cuda_manual_seed
from torch.cuda import is_available, set_device
import torch.backends.cudnn as cudnn

from models.BaseNet.BaseNet import BaseNet
from trainRegimes.OptimalRegime import OptimalRegime

from utils.trainWeights import TrainWeights
from utils.HtmlLogger import HtmlLogger
from utils.zip import create_exp_dir
from utils.checkpoint import checkpointFileType


def checkpointPrefix(fileName):
    return fileName[:-len(checkpointFileType) - 1]


def train(scriptArgs):
    # load args from file
    args = loadCheckpoint(scriptArgs.json, map_location=lambda storage, loc: storage.cuda())

    # terminate if validAcc exists
    _validAccKey = TrainWeights.validAccKey
    if hasattr(args, _validAccKey):
        print('[{}] exists'.format(_validAccKey))
        exit(0)

    # no need to save model random weights
    args.saveRandomWeights = False

    # update args parameters
    args.seed = datetime.now().microsecond
    # update cudnn parameters
    random.seed(args.seed)
    set_device(scriptArgs.gpu[0])
    cudnn.benchmark = True
    torch_manual_seed(args.seed)
    cudnn.enabled = True
    cuda_manual_seed(args.seed)
    # copy scriptArgs values to args
    for k, v in vars(scriptArgs).items():
        setattr(args, k, v)

    # load model flops
    _modelFlopsPathKey = BaseNet.modelFlopsPathKey()
    modelFlopsPath = getattr(args, _modelFlopsPathKey)
    if modelFlopsPath and exists(modelFlopsPath):
        setattr(args, BaseNet.modelFlopsKey(), loadCheckpoint(modelFlopsPath))

    folderNotExists = not exists(args.save)
    if folderNotExists:
        create_exp_dir(args.save)
        # init logger
        logger = HtmlLogger(args.save, 'log')

        if scriptArgs.individual:
            args.width = []

        alphasRegime = OptimalRegime(args, logger)
        # train according to chosen regime
        alphasRegime.train()

    return folderNotExists


def iterateFolder(scriptArgs):
    foundCheckpointToTrain = True
    while foundCheckpointToTrain:
        foundCheckpointToTrain = False
        checkpointsList = sorted(listdir(scriptArgs.folderPath), reverse=True)
        for file in checkpointsList:
            if file.endswith('].{}'.format(checkpointFileType)):
                # init folder name
                prefix = checkpointPrefix(file)
                folderName = '{}-{}'.format(prefix, scriptArgs.repeatNum)
                # init checkpoint folder path
                scriptArgs.save = '{}/{}'.format(scriptArgs.folderPath, folderName)
                if not exists(scriptArgs.save):
                    # init new checkpoint file name
                    scriptArgs.json = '{}.{}'.format(folderName, checkpointFileType)
                    # init original & new checkpoints paths
                    filePath = '{}/{}'.format(scriptArgs.folderPath, file)
                    scriptArgs.json = '{}/{}'.format(scriptArgs.folderPath, scriptArgs.json)
                    # copy checkpoint
                    copy2(filePath, scriptArgs.json)
                    # train checkpoint
                    trainedCheckpoint = train(scriptArgs)
                    foundCheckpointToTrain = foundCheckpointToTrain or trainedCheckpoint
                    # remove original checkpoint
                    if exists(filePath):
                        remove(filePath)


if not is_available():
    print('no gpu device available')
    exit(1)

parser = ArgumentParser()
# parser.add_argument('--json', type=str, required=True, help='JSON file path')
parser.add_argument('--folderPath', type=str, required=True, help='checkpoints folder path')
parser.add_argument('--data', type=str, required=True, help='datasets folder path')
parser.add_argument('--repeatNum', type=int, required=True, choices=range(1, 100), help='checkpoint training repeat number')
parser.add_argument('--gpu', type=str, default='0', help='gpu device id, e.g. 0,1,3')
parser.add_argument('--workers', type=int, default=0, choices=range(0, 32), help='num of workers')
parser.add_argument('--optimal_epochs', type=int, default=150, help='stop training weights if there is no new optimum in last optimal_epochs')
parser.add_argument('--pre_trained', type=str, default=None, help='pre-trained model to copy weights from')
parser.add_argument('--individual', action='store_true', default=False, help='Trains the partition individually in case value is True')

scriptArgs = parser.parse_args()
# update GPUs list
if type(scriptArgs.gpu) is str:
    scriptArgs.gpu = [int(i) for i in scriptArgs.gpu.split(',')]

iterateFolder(scriptArgs)
