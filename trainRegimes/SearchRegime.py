from time import time
from scipy.stats import entropy

from torch import tensor, zeros
from torch.nn import functional as F

from .regime import TrainRegime

from utils.flopsLoss import FlopsLoss


class SearchRegime(TrainRegime):
    # init table columns
    # k = 2
    # alphasTableTitle = 'Alphas (top [{}])'.format(k)
    # colsTrainAlphas = [batchNumKey, archLossKey, crossEntropyKey, flopsLossKey, alphasTableTitle, forwardCountersKey, timeKey]

    def __init__(self, args, logger):
        super(SearchRegime, self).__init__(args, logger)

        # init flops loss
        self.flopsLoss = FlopsLoss(args, getattr(args, self.model.baselineFlopsKey()))
        self.flopsLoss = self.flopsLoss.cuda()

        self.trainAlphas(0, {})

        # init email time
        self.lastMailTime = time()
        self.secondsBetweenMails = 1 * 3600

    def trainAlphas(self, optimizer, epoch, loggers):
        print('*** trainAlphas() ***')
        model = self.model
        modelParallel = self.modelParallel
        search_queue = self.search_queue[0]

        trainLogger = loggers.get('train')
        if trainLogger:
            trainLogger.createDataTable('Epoch:[{}] - Alphas'.format(epoch), self.colsTrainAlphas)

        for step, (input, target) in enumerate(search_queue):
            startTime = time()

            input = tensor(input, requires_grad=False).cuda()
            target = tensor(target, requires_grad=False).cuda(async=True)

            # reset optimizer gradients
            optimizer.zero_grad()
            # evaluate on samples and calc alphas gradients
            loss = self._samplesSamePath(input, target)
            # perform optimizer step
            optimizer.step()
            # update alphas distribution statistics (after optimizer step)
            self._calcAlphasDistStats(model)
            # update statistics plots
            self.statistics.plotData()

            endTime = time()

    def _calcAlphasDistStats(self, model):
        stats = self.statistics
        for layerIdx, layer in enumerate(model.layersList()):
            # calc layer alphas distribution
            probs = F.softmax(layer.alphas(), dim=-1)
            # add entropy to statistics
            stats.addValue(lambda containers: containers[self.entropyKey][0][layerIdx], entropy(probs))
            # add alphas distribution
            for alphaIdx, p in enumerate(probs):
                alphaTitle = self._alphaPlotTitle(layer, alphaIdx)
                stats.addValue(lambda containers: containers[self.alphaDistributionKey][layerIdx][alphaTitle], p.item())

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
        ceLossValues = [[0.0 for _ in range(layer.nWidths())] for layer in model.layersList()]
        flopsLossValues = [[0.0 for _ in range(layer.nWidths())] for layer in model.layersList()]

        # switch to inference mode
        model.eval()

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
                    ceLossValues[layerIdx][idx] += crossEntropyLoss.item()
                    flopsLossValues[layerIdx][idx] += flopsLoss.item()

                # restore layer current width idx
                layer.setCurrWidthIdx(layerCurrWidthIdx)

        # switch to train mode
        model.train()

        # init total loss
        totalLoss = 0.0
        # init total loss average
        totalLossAvg = 0.0
        # count how many alphas we have sum their loss average
        nAlphas = 0
        # get statistics element with a shorter name
        stats = self.statistics
        # init model probs list for gradient calcs
        probsList = []
        # after we finished iterating over samples, we can calculate loss average & variance for each alpha
        for layerIdx, (layer, layerLossValues) in enumerate(zip(model.layersList(), lossValues)):
            layerAlphas = layer.alphas()
            # calc layer alphas probabilities
            probs = F.softmax(layerAlphas, dim=-1)
            probsList.append(probs)
            # init layer alphas gradient vector
            layerAlphasGrad = zeros(layer.nWidths(), requires_grad=True).cuda()
            # iterate over alphas
            for idx, alphaLossValues in enumerate(layerLossValues):
                # calc alpha average loss
                alphaLossAvg = sum(alphaLossValues) / nSamples
                # add alpha loss average to total loss average
                totalLossAvg += alphaLossAvg
                nAlphas += 1
                # update alpha gradient
                layerAlphasGrad[idx] = alphaLossAvg
                # update total loss
                totalLoss += (alphaLossAvg * probs[idx])
                # calc alpha loss variance
                alphaLossVariance = [((x - alphaLossAvg) ** 2) for x in alphaLossValues]
                alphaLossVariance = sum(alphaLossVariance) / (nSamples - 1)
                # add values to statistics
                alphaTitle = self._alphaPlotTitle(layer, idx)
                # init template for get list function based on container key
                getListFunc = lambda key: lambda containers: containers[key][layerIdx][alphaTitle]
                # add values
                stats.addValue(getListFunc(self.lossAvgKey), alphaLossAvg)
                stats.addValue(getListFunc(self.crossEntropyLossAvgKey), ceLossValues[layerIdx][idx] / nSamples)
                stats.addValue(getListFunc(self.flopsLossAvgKey), flopsLossValues[layerIdx][idx] / nSamples)
                stats.addValue(getListFunc(self.lossVarianceKey), alphaLossVariance)

            # update layer alphas gradient
            layerAlphas.grad = layerAlphasGrad

        # average total loss
        totalLoss /= len(model.layersList())
        # subtract average total loss from every alpha gradient
        for layer, layerProbs in zip(model.layersList(), probsList):
            layerAlphas = layer.alphas()
            layerAlphas.grad -= totalLoss
            # multiply each grad by its probability
            layerAlphas.grad *= probs

        # average (total loss average) by number of alphas
        totalLossAvg /= nAlphas

        return totalLossAvg


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
