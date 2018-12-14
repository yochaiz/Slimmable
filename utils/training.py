class AvgrageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.avg = 0
        self.sum = 0
        self.cnt = 0

    def update(self, val, n=1):
        self.sum += val * n
        self.cnt += n
        self.avg = self.sum / self.cnt


# abstract base class
class TrainingData:
    nRoundDigits = 5
    avgKey = 'Avg'

    @staticmethod
    # returns only average value from dict
    def dictAvg(dict):
        return dict[TrainingData.avgKey]


class TrainingStats(TrainingData):
    def __init__(self, widthRatio):
        super(TrainingStats, self).__init__()

        self._epochLoss = {}
        self._batchLoss = {}
        self._top1 = {}
        self._prec1 = {}

        for ratio in widthRatio:
            self._epochLoss[ratio] = AvgrageMeter()
            self._top1[ratio] = AvgrageMeter()

    # create dict copy with rounded values, for better display
    # in addition, add average value to dict
    def _createDictWithRoundedValues(self, dict, valueFunc=lambda x: x):
        newDict = {}
        sum = 0.0
        for k, v in dict.items():
            value = valueFunc(v)
            newDict[k] = round(value, self.nRoundDigits)
            sum += value
        # add average
        dict[self.avgKey] = sum / len(dict)
        newDict[self.avgKey] = round(dict[self.avgKey], self.nRoundDigits)

        return newDict

    def prec1(self):
        return self._createDictWithRoundedValues(self._prec1)

    def batchLoss(self):
        return self._createDictWithRoundedValues(self._batchLoss)

    def epochLoss(self):
        return self._createDictWithRoundedValues(self._epochLoss, lambda v: v.avg)

    # returns only average value from dict, i.e. returns float
    def epochLossAvg(self):
        return self.dictAvg(self._epochLoss)

    def top1(self):
        return self._createDictWithRoundedValues(self._top1, lambda v: v.avg)

    # returns only average value from dict, i.e. returns float
    def top1Avg(self):
        return self.dictAvg(self._top1)

    # update values for given ratio
    def update(self, ratio, logits, target, loss):
        n = logits.size(0)
        prec1 = self.accuracy(logits, target)[0].item()
        self._epochLoss[ratio].update(loss.item(), n)
        self._top1[ratio].update(prec1, n)
        self._prec1[ratio] = prec1
        self._batchLoss[ratio] = loss.item()

    @staticmethod
    def accuracy(output, target, topk=(1,)):
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


class TrainingOptimum(TrainingData):
    def __init__(self, widthList, tableHeaders):
        super(TrainingOptimum, self).__init__()

        self._tableHeaders = tableHeaders

        self._opt = {}
        # init with invalid values
        for width in widthList + [self.avgKey]:
            self._opt[width] = (-1.0, None)

    # create table for display in HtmlLogger
    def _toTable(self, currEpoch):
        table = [[h] for h in self._tableHeaders]
        for width, (acc, epoch) in self._opt.items():
            values = [width, acc, epoch, currEpoch - epoch]
            for innerTable, v in zip(table, values):
                innerTable.append(v)

        return table

    # update optimum values according to current epoch dict
    def update(self, dict, epoch):
        # update optimal values for each width
        for width, acc in dict.items():
            optAcc, _ = self._opt[width]
            if acc > optAcc:
                self._opt[width] = (acc, epoch)

        # return table
        return self._toTable(epoch)

    # returns if given epoch is average best & average optimum
    def is_best(self, epoch):
        return epoch == self._opt[self.avgKey][-1], self._opt[self.avgKey][0]
