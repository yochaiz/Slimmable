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
        self.train_queue, self.valid_queue, self.createSearchQueue = load_data(args)
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

# ==== COMPARE FLOPS CALCULATION ====
# idxList = [2, 0, 4, 2, 5, 4, 5]
# # idxList = model.currWidthIdx()
# # for i in range(7, len(idxList)):
# #     idxList[i] += 2
# model.setCurrWidthIdx(idxList)
# v1 = model.countFlops()
# from utils.flops_benchmark import count_flops
# from torch.nn.modules.conv import Conv2d
# from torch.nn.modules.pooling import AvgPool2d
# from torch.nn.modules.linear import Linear
# from torch.nn.modules.module import Module
#
# class G(Module):
#     def __init__(self):
#         super(G, self).__init__()
#
#         factor = 0.25
#
#         self.l0 = Conv2d(3, int(16 * 0.5), 3, stride=1, padding=1, bias=False)
#
#         self.b10 = Conv2d(int(0.5 * 16), int(16 * 0.25), 3, stride=1, padding=1, bias=False)
#         self.b11 = Conv2d(int(0.25 * 16), int(16 * 0.75), 3, stride=1, padding=1, bias=False)
#
#         self.b20 = Conv2d(int(0.75 * 16), int(16 * 0.5), 3, stride=1, padding=1, bias=False)
#         self.b21 = Conv2d(int(0.5 * 16), int(16 * 1.0), 3, stride=1, padding=1, bias=False)
#
#         self.b30 = Conv2d(int(1.0 * 16), int(16 * 0.75), 3, stride=1, padding=1, bias=False)
#         self.b31 = Conv2d(int(0.75 * 16), int(16 * 1.0), 3, stride=1, padding=1, bias=False)
#
#         self.e10 = Conv2d(int(0.5 * 16), int(0.75 * 16), 1, stride=1, bias=False)
#         self.e20 = Conv2d(int(0.75 * 16), int(1.0 * 16), 1, stride=1, bias=False)
#         # self.e30 = Conv2d(int(1.0 * 16), int(1.0 * 16), 1, stride=1, bias=False)
#         self.e30 = lambda x: x
#
#         self.avgpool = AvgPool2d(8)
#         self.fc = Linear(int(256 * 1.0), 10).cuda()
#
#         # factor2 = 0.5
#         # self.d1 = Conv2d(int(factor * 16), int(factor2 * 32), 1, stride=2, bias=False)
#         # self.b40 = Conv2d(int(factor * 16), int(factor2 * 32), 3, stride=2, padding=1, bias=False)
#         # factor = 0.5
#         # self.b41 = Conv2d(int(factor * 32), int(factor * 32), 3, stride=1, padding=1, bias=False)
#         #
#         # self.b50 = Conv2d(int(factor * 32), int(factor * 32), 3, stride=1, padding=1, bias=False)
#         # self.b51 = Conv2d(int(factor * 32), int(factor * 32), 3, stride=1, padding=1, bias=False)
#         #
#         # self.b60 = Conv2d(int(factor * 32), int(factor * 32), 3, stride=1, padding=1, bias=False)
#         # self.b61 = Conv2d(int(factor * 32), int(factor * 32), 3, stride=1, padding=1, bias=False)
#         #
#         # self.b70 = Conv2d(int(factor * 32), int(factor * 64), 3, stride=2, padding=1, bias=False)
#         # self.b71 = Conv2d(int(factor * 64), int(factor * 64), 3, stride=1, padding=1, bias=False)
#         # self.d2 = Conv2d(int(factor * 32), int(factor * 64), 1, stride=2, bias=False)
#         #
#         # self.b80 = Conv2d(int(factor * 64), int(factor * 64), 3, stride=1, padding=1, bias=False)
#         # self.b81 = Conv2d(int(factor * 64), int(factor * 64), 3, stride=1, padding=1, bias=False)
#         #
#         # self.b90 = Conv2d(int(factor * 64), int(factor * 64), 3, stride=1, padding=1, bias=False)
#         # self.b91 = Conv2d(int(factor * 64), int(factor * 64), 3, stride=1, padding=1, bias=False)
#
#     def forward(self, x):
#         out0 = self.l0(x)
#
#         out1 = self.b10(out0)
#         out1 = self.b11(out1)
#         out1 += self.e10(out0)
#
#         out2 = self.b20(out1)
#         out2 = self.b21(out2)
#         out2 += self.e20(out1)
#
#         out3 = self.b30(out2)
#         out3 = self.b31(out3)
#         out3 += self.e30(out2)
#
#         # out4 = self.b40(out3)
#         # out4 = self.b41(out4)
#         # out4 += self.d1(out3)
#         #
#         # out5 = self.b50(out4)
#         # out5 = self.b51(out5)
#         # out5 += out4
#         #
#         # out6 = self.b60(out5)
#         # out6 = self.b61(out6)
#         # out6 += out5
#         #
#         # out7 = self.b70(out6)
#         # out7 = self.b71(out7)
#         # out7 += self.d2(out6)
#         #
#         # out8 = self.b80(out7)
#         # out8 = self.b81(out8)
#         # out8 += out7
#         #
#         # out9 = self.b90(out8)
#         # out9 = self.b91(out9)
#         # out9 += out8
#         out9 = out3
#
#         out = self.avgpool(out9)
#         out = out.view(out.size(0), -1)
#         out = self.fc(out)
#
#         return out
#
# flops, output_size = count_flops(G(), args.input_size, 3)
