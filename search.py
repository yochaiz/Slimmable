from argparse import ArgumentParser
from inspect import isclass
from time import strftime
from datetime import datetime
from numpy import random as nprandom

from torch.multiprocessing import set_start_method
import torch.backends.cudnn as cudnn
from torch.cuda import is_available, set_device
from torch.cuda import manual_seed as cuda_manual_seed
from torch import manual_seed as torch_manual_seed

import models
from utils import create_exp_dir


def parseArgs():
    modelNames = [name for (name, obj) in models.__dict__.items() if isclass(obj) and name.islower()]
    datasets = dict(cifar10=10, cifar100=100, imagenet=1000)

    parser = ArgumentParser("Slimmable")
    parser.add_argument('--data', type=str, required=True, help='location of the data corpus')
    parser.add_argument('--dataset', metavar='DATASET', default='cifar10', choices=datasets.keys(), help='dataset name')
    parser.add_argument('--model', metavar='MODEL', default='resnet18', choices=modelNames)
    parser.add_argument('--batch_size', type=int, default=250, help='batch size')
    parser.add_argument('--learning_rate', type=float, default=0.01, help='init learning rate')
    parser.add_argument('--learning_rate_min', type=float, default=1E-8, help='min learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    parser.add_argument('--weight_decay', type=float, default=4e-5, help='weight decay')
    parser.add_argument('--gpu', type=str, default='0', help='gpu device id, e.g. 0,1,3')

    parser.add_argument('--train_portion', type=float, default=0.5, help='portion of training data')

    parser.add_argument('--kernel', type=int, default=3, help='conv kernel size, e.g. 1,3,5')

    parser.add_argument('--width', type=str, required=True, help='list of width values, e.g. 0.25,0.5,0.75,1.0')

    args = parser.parse_args()

    # update GPUs list
    if type(args.gpu) is str:
        args.gpu = [int(i) for i in args.gpu.split(',')]

    # convert width to list
    args.width = [float(x) for x in args.width.split(',')]
    assert (0 < max(args.width) <= 1)

    # set number of model output classes
    args.nClasses = datasets[args.dataset]

    # create folder
    args.time = strftime("%Y%m%d-%H%M%S")
    args.lmbda = 0.0  # TODO: replace with real lambda
    args.folderName = '{},[{}],[{}],[{}]'.format(args.width, args.lmbda, args.dataset, args.time)
    args.save = '../results/{}'.format(args.folderName)
    args.codeZip = create_exp_dir(args.save)

    return args


if __name__ == '__main__':
    args = parseArgs()

    if not is_available():
        print('no gpu device available')
        exit(1)

    args.seed = datetime.now().microsecond
    nprandom.seed(args.seed)
    set_device(args.gpu[0])
    cudnn.benchmark = True
    torch_manual_seed(args.seed)
    cudnn.enabled = True
    cuda_manual_seed(args.seed)

    try:
        set_start_method('spawn', force=True)
    except RuntimeError:
        raise ValueError('spawn failed')

    # build model for uniform distribution of bits
    modelClass = models.__dict__[args.model]
    # init model
    model = modelClass(args)
    model = model.cuda()
    print(model)

    a = 5
