from abc import abstractmethod
from argparse import Namespace

# from torch.nn.parallel.data_parallel import DataParallel

from models import getModelDict
from models.BaseNet.BaseNet import BaseNet, SlimLayer
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
    def buildStatsContainers(self):
        raise NotImplementedError('subclasses must override buildStatsContainers()!')

    @abstractmethod
    def buildStatsRules(self):
        raise NotImplementedError('subclasses must override buildStatsRules()!')

    def buildModel(self, args: Namespace) -> BaseNet:
        modelsDict = getModelDict()
        # get model constructor
        modelKey = '{}_{}'.format(args.model, args.dataset)
        modelClass = modelsDict[modelKey]

        return modelClass(args)

    def _containerPerAlpha(self, model: BaseNet) -> list:
        return [{self._alphaPlotTitle(layer, idx): [] for idx in range(layer.nWidths())} for layer in model.layersList()]

    def _alphaPlotTitle(self, layer: SlimLayer, alphaIdx: int) -> str:
        return '{} ({})'.format(layer.widthRatioByIdx(alphaIdx), layer.widthByIdx(alphaIdx))
