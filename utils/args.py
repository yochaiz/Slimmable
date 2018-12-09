from json import dump
from argparse import ArgumentParser
from inspect import isclass
from time import strftime

import models
import trainRegimes
from utils.HtmlLogger import HtmlLogger
from utils.zip import create_exp_dir


def saveArgsToJSON(args):
    # save args to JSON
    args.jsonPath = '{}/args.txt'.format(args.save)
    with open(args.jsonPath, 'w') as f:
        dump(vars(args), f, indent=4, sort_keys=True)


def parseArgs():
    modelNames = [name for (name, obj) in models.__dict__.items() if isclass(obj) and name.islower()]
    trainRegimesNames = [name for (name, obj) in trainRegimes.__dict__.items() if isclass(obj) and name.islower()]
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
    parser.add_argument('--workers', type=int, default=1, choices=range(1, 32), help='num of workers')

    parser.add_argument('--optimal_epochs', type=int, default=30, help='stop training weights if there is no new optimum in last optimal_epochs')
    parser.add_argument('--train_portion', type=float, default=0.5, help='portion of training data')
    parser.add_argument('--train_regime', default='TrainRegime', choices=trainRegimesNames, help='Training regime')
    parser.add_argument('--alphas_data_parts', type=int, default=4, help='split alphas training data to parts. each loop uses single part')

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
    args.folderName = '[{}],[{}],[{}],{},[{}]'.format(args.model, args.dataset, args.lmbda, args.width, args.time)
    args.save = '../results/{}'.format(args.folderName)
    create_exp_dir(args.save)
    # set train folder name
    args.trainFolder = 'train'

    # save args to JSON
    saveArgsToJSON(args)

    return args


def logParameters(logger, args, model):
    if not logger:
        return

    # calc number of permutations
    permutationStr = model.nPerms
    for p in [12, 9, 6, 3]:
        v = model.nPerms / (10 ** p)
        if v > 1:
            permutationStr = '{:.3f} * 10<sup>{}</sup>'.format(v, p)
            break
    # log other parameters
    logger.addInfoTable('Parameters', HtmlLogger.dictToRows(
        {
            'Learnable params': len([param for param in model.parameters() if param.requires_grad]),
            'Widths per layer': [layer.nWidths() for layer in model.layersList()],
            'Permutations': permutationStr
        }, nElementPerRow=2))
    # # log baseline model
    # logBaselineModel(args, logger)
    # log args
    logger.addInfoTable('args', HtmlLogger.dictToRows(vars(args), nElementPerRow=3))
    # print args
    print(args)
