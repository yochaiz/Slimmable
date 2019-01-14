from argparse import ArgumentParser
from datetime import datetime
from numpy import random
from os import path

from torch import load as loadCheckpoint
from torch import manual_seed as torch_manual_seed
from torch.cuda import manual_seed as cuda_manual_seed
from torch.cuda import is_available, set_device
import torch.backends.cudnn as cudnn

from trainRegimes.OptimalRegime import OptimalRegime

from utils.trainWeights import TrainWeights
from utils.HtmlLogger import HtmlLogger
from utils.zip import create_exp_dir
from utils.checkpoint import checkpointFileType


def train(scriptArgs):
    # load args from file
    args = loadCheckpoint(scriptArgs.json, map_location=lambda storage, loc: storage.cuda())

    # terminate if validAcc exists
    if hasattr(args, TrainWeights.validAccKey):
        exit(0)

    if not hasattr(args, 'logInterval'):
        args.logInterval = 50

    # update args parameters
    args.seed = datetime.now().microsecond
    # update cudnn parameters
    random.seed(args.seed)
    set_device(scriptArgs.gpu[0])
    cudnn.benchmark = True
    torch_manual_seed(args.seed)
    cudnn.enabled = True
    cuda_manual_seed(args.seed)
    # update values
    args.gpu = scriptArgs.gpu
    args.workers = scriptArgs.workers
    args.data = scriptArgs.data
    args.pre_trained = scriptArgs.pre_trained
    args.optimal_epochs = scriptArgs.optimal_epochs
    args.json = scriptArgs.json

    # extract args JSON folder path
    folderName = path.dirname(scriptArgs.json)
    # results folder is JSON filename
    jsonFileName = path.basename(scriptArgs.json)
    # set results folder path
    args.save = '{}/{}'.format(folderName, jsonFileName[:-len(checkpointFileType) - 1])
    if not path.exists(args.save):
        create_exp_dir(args.save)
        # init logger
        logger = HtmlLogger(args.save, 'log')

        if scriptArgs.individual:
            args.width = []

        alphasRegime = OptimalRegime(args, logger)
        # train according to chosen regime
        alphasRegime.train()



if not is_available():
    print('no gpu device available')
    exit(1)

parser = ArgumentParser()
parser.add_argument('--json', type=str, required=True, help='JSON file path')
parser.add_argument('--data', type=str, required=True, help='datasets folder path')
parser.add_argument('--gpu', type=str, default='0', help='gpu device id, e.g. 0,1,3')
parser.add_argument('--workers', type=int, default=0, choices=range(0, 32), help='num of workers')
parser.add_argument('--optimal_epochs', type=int, default=150, help='stop training weights if there is no new optimum in last optimal_epochs')
parser.add_argument('--pre_trained', type=str, default=None, help='pre-trained model to copy weights from')
parser.add_argument('--individual', action='store_true', default=False, help='Trains the partition individually in case value is True')

scriptArgs = parser.parse_args()
# update GPUs list
if type(scriptArgs.gpu) is str:
    scriptArgs.gpu = [int(i) for i in scriptArgs.gpu.split(',')]

train(scriptArgs)
