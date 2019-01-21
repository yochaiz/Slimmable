from abc import abstractmethod
from argparse import Namespace

# from torch.nn.parallel.data_parallel import DataParallel

from models import ResNetSwitcher
from models.BaseNet.BaseNet import BaseNet
from utils.data import load_data
from utils.args import logParameters
from utils.HtmlLogger import HtmlLogger
from utils.statistics import Statistics


class TrainRegime:
    def __init__(self, args: Namespace, logger: HtmlLogger):
        # init model
        model = self.buildModel(args)
        model = model.cuda()
        # create DataParallel model instance
        self.modelParallel = model
        # self.modelParallel = DataParallel(model, args.gpu)
        # assert (id(model) == id(self.modelParallel.module))

        self.args = args
        self.model = model
        self.logger = logger

        # load data
        self.train_queue, self.search_queue, self.valid_queue = load_data(args)
        # init train folder path, where to save loggers, checkpoints, etc.
        self.trainFolderPath = '{}/{}'.format(args.save, args.trainFolder)

        # build statistics containers
        containers = self.buildStatsContainers()
        # build statistics rules
        rules = self.buildStatsRules()
        # init statistics instance
        self.statistics = Statistics(containers, rules, args.save)

        # log parameters
        logParameters(logger, args, model)

    @abstractmethod
    def train(self):
        raise NotImplementedError('subclasses must override train()!')

    @abstractmethod
    def buildStatsContainers(self) -> dict:
        raise NotImplementedError('subclasses must override buildStatsContainers()!')

    @abstractmethod
    def buildStatsRules(self):
        raise NotImplementedError('subclasses must override buildStatsRules()!')

    @staticmethod
    def buildModel(args: Namespace) -> BaseNet:
        modelsDict = ResNetSwitcher.getModelDict(args.type)
        # get model constructor
        modelKey = '{}_{}'.format(args.model, args.dataset)
        modelClass = modelsDict[modelKey]

        return modelClass(args)

    # ==== TrainWeights default functions ====
    def getModel(self):
        return self.model

    def getModelParallel(self):
        return self.modelParallel

    def getArgs(self):
        return self.args

    def getLogger(self):
        return self.logger

    def getTrainQueue(self):
        return self.train_queue

    def getValidQueue(self):
        return self.valid_queue

    def getTrainFolderPath(self):
        return self.trainFolderPath
