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
    _avgKey = 'Avg'

    def __init__(self, widthRatio, useAvg):
        self.useAvg = useAvg and len(widthRatio) > 1

    @staticmethod
    def avgKey():
        return TrainingData._avgKey

    # returns only average value from dict
    def dictAvg(self, dict):
        return dict[TrainingData._avgKey] if self.useAvg else dict[next(iter(dict))]


class TrainingStats(TrainingData):
    def __init__(self, widthRatio, useAvg=True):
        super(TrainingStats, self).__init__(widthRatio, useAvg)

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
        if self.useAvg:
            newDict[self._avgKey] = round(sum / len(dict), self.nRoundDigits)

        return newDict

    def prec1(self):
        return self._createDictWithRoundedValues(self._prec1)

    def batchLoss(self):
        return self._createDictWithRoundedValues(self._batchLoss)

    def epochLoss(self):
        return self._createDictWithRoundedValues(self._epochLoss, lambda v: v.avg)

    def top1(self):
        return self._createDictWithRoundedValues(self._top1, lambda v: v.avg)

    # update values for given ratio
    def update(self, ratio, logits, target, loss):
        n = logits.size(0)

        self._epochLoss[ratio].update(loss.item(), n)
        self._batchLoss[ratio] = loss.item()

        prec1 = self.accuracy(logits, target)[0].item()
        self._top1[ratio].update(prec1, n)
        self._prec1[ratio] = prec1

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


class AlphaTrainingStats(TrainingStats):
    def __init__(self, widthRatio, useAvg=True):
        super(AlphaTrainingStats, self).__init__(widthRatio, useAvg)

    def update(self, ratio, logits, loss):
        n = logits.size(0)
        self._epochLoss[ratio].update(loss, n)
        self._batchLoss[ratio] = loss


class TrainingOptimum(TrainingData):
    def __init__(self, widthList, tableHeaders, optCompareFunc, useAvg=True):
        super(TrainingOptimum, self).__init__(widthList, useAvg)

        self.optCompareFunc = optCompareFunc
        self._tableHeaders = tableHeaders

        self._opt = {}
        # copy width list
        widthList = widthList.copy()
        # add average key if there multiple width
        if self.useAvg:
            widthList.append(self._avgKey)
        # init with invalid values
        for width in widthList:
            self._opt[width] = (-1.0, None)

    # create table for display in HtmlLogger
    def _toTable(self, currEpoch):
        table = [[h] for h in self._tableHeaders]
        for width, (value, epoch) in self._opt.items():
            values = [width, value, epoch, currEpoch - epoch]
            for innerTable, v in zip(table, values):
                innerTable.append(v)

        return table

    # update optimum values according to current epoch dict
    def update(self, dict, epoch):
        # update optimal values for each width
        for width, value in dict.items():
            optValue, _ = self._opt[width]
            if self.optCompareFunc(value, optValue):
                self._opt[width] = (value, epoch)

        # return table
        return self._toTable(epoch)

    # returns if given epoch is average best
    def is_best(self, epoch):
        # average key is not always in dictionary
        key = self._avgKey if self.useAvg else next(iter(self._opt))
        # return value
        return epoch == self._opt[key][-1]

    # # returns dictionary of (width,accuracy) to save in checkpoint
    # def accuracy(self):
    #     return {width: value for width, (value, epoch) in self._opt.items()}
