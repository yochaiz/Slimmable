from torch import tensor, zeros
from torch.nn import functional as F

from .regime import TrainRegime, time

from utils.flopsLoss import FlopsLoss


class SearchRegime(TrainRegime):
    def __init__(self, args, logger):
        super(SearchRegime, self).__init__(args, logger)

        # init flops loss
        self.flopsLoss = FlopsLoss(args, getattr(args, self.model.baselineFlopsKey()))
        self.flopsLoss = self.flopsLoss.cuda()

        self.trainAlphas(0, {})

        # init email time
        self.lastMailTime = time()
        self.secondsBetweenMails = 1 * 3600

    def trainAlphas(self, epoch, loggers):
        print('*** trainAlphas() ***')
        modelParallel = self.modelParallel
        search_queue = self.search_queue[0]

        trainLogger = loggers.get('train')
        if trainLogger:
            trainLogger.createDataTable('Epoch:[{}] - Alphas'.format(epoch), self.colsTrainAlphas)

        for step, (input, target) in enumerate(search_queue):
            startTime = time()

            input = tensor(input, requires_grad=False).cuda()
            target = tensor(target, requires_grad=False).cuda(async=True)

            # switch to inference mode
            modelParallel.eval()
            # evaluate on samples and calc alphas gradients
            self._samplesSamePath(input, target)

    # given paths history dictionary and current path, checks if current path exists in history dictionary
    def _doesPathExist(self, pathsHistoryDict, currPath):
        currDict = pathsHistoryDict
        # init boolean flag
        pathExists = True
        for v in currPath:
            if v not in currDict:
                pathExists = False
                currDict[v] = {}
            # update current dict, we move to next layer
            currDict = currDict[v]

        return pathExists

    # evaluate alphas on same paths
    def _samplesSamePath(self, input, target):
        model = self.model
        modelParallel = self.modelParallel
        nSamples = self.args.nSamplesPerAlpha
        # init samples (paths) history, to make sure we don't select the same sample twice
        pathsHistoryDict = {}

        # init containers to save loss values and variance
        lossValues = [[[] for _ in range(layer.nWidths())] for layer in model.layersList()]

        # iterate over samples. generate a sample (path) and evaluate alphas on sample
        for _ in range(nSamples):
            # select new path based on alphas distribution.
            # check that selected path hasn't been selected before
            pathExists = True
            while pathExists:
                # select path based on alphas distribution
                model.choosePathByAlphas()
                # get selected path indices
                currWidthIdx = model.currWidthIdx()
                # check that selected path hasn't been selected before
                pathExists = self._doesPathExist(pathsHistoryDict, currWidthIdx)

            # iterate over layers. in each layer iterate over alphas
            for layerIdx, layer in enumerate(model.layersList()):
                # init containers to save loss values and variance
                layerLossValues = lossValues[layerIdx]
                # save layer current width idx
                layerCurrWidthIdx = layer.currWidthIdx()
                # iterate over alphas and calc loss
                for idx in range(layer.nWidths()):
                    # set path to go through width[idx] in current layer
                    layer.setCurrWidthIdx(idx)
                    print(model.currWidthIdx())
                    # forward input in model selected path
                    logits = modelParallel(input)
                    # calc loss
                    loss, crossEntropyLoss, flopsLoss = self.flopsLoss(logits, target, model.countFlops())
                    # add loss to container
                    layerLossValues[idx].append(loss.item())

                # restore layer current width idx
                layer.setCurrWidthIdx(layerCurrWidthIdx)

        # init total loss
        totalLoss = 0.0
        # init alphas gradients
        alphasGrad = []
        # init loss variance container
        lossVariance = []
        # after we finished iterating over samples, we can calculate loss average & variance for each alpha
        for layer, layerLossValues in zip(model.layersList(), lossValues):
            # init layer loss variance container
            layerLossVariance = []
            # calc layer alphas probabilities
            probs = F.softmax(layer.alphas(), dim=-1)
            # init layer alphas gradient vector
            layerAlphasGrad = zeros(layer.nWidths()).cuda()
            for idx, alphaLossValues in enumerate(layerLossValues):
                # calc alpha average loss
                alphaLossAvg = tensor(sum(alphaLossValues) / nSamples).cuda()
                # update alpha gradient
                layerAlphasGrad[idx] = alphaLossAvg
                # update total loss
                totalLoss += (alphaLossAvg * probs[idx])
                # calc alpha loss variance
                alphaLossVariance = [((x - alphaLossAvg) ** 2) for x in alphaLossValues]
                alphaLossVariance = sum(alphaLossVariance) / (nSamples - 1)
                layerLossVariance.append(alphaLossVariance)

            alphasGrad.append(layerAlphasGrad)
            lossVariance.append(layerLossVariance)

# ======= compare replicator vs. standard model, who is FASTER ===========
# init model constructor func
# modelConstructor = lambda: self.buildModel(args)
# replicator = ModelReplicator(modelConstructor, args.gpu, self.logger)

# # calc replicator time
# print('Running replicator')
# startTime = time()
# replicator.run()
# endTime = time()
# print('Replicator time:[{}]'.format(endTime - startTime))
#
# # calc standard model time
# data = randn(250, 3, 32, 32).cuda()
# print('Running standard model')
# startTime = time()
# for _ in range(5000):
#     self.model(data)
#     data[0, 0, 0, 0] += 0.001
# endTime = time()
# print('Standard model time:[{}]'.format(endTime - startTime))
