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


class TrainingStats:
    nRoundDigits = 5

    def __init__(self, widthRatio):
        self._epochLoss = {}
        self._batchLoss = {}
        self._top1 = {}
        self._prec1 = {}

        for ratio in widthRatio:
            self._epochLoss[ratio] = AvgrageMeter()
            self._top1[ratio] = AvgrageMeter()

    def _createTableWithRoundedValues(self, dict, valueFunc=lambda x: x):
        table = []
        sum = 0.0
        for k, v in dict.items():
            value = valueFunc(v)
            table.append([k, round(value, self.nRoundDigits)])
            sum += value
        # add average
        table.append(['Avg', round(sum / len(dict), self.nRoundDigits)])

        return table

    def prec1(self):
        return self._createTableWithRoundedValues(self._prec1)

    def batchLoss(self):
        return self._createTableWithRoundedValues(self._batchLoss)

    def epochLoss(self):
        return self._createTableWithRoundedValues(self._epochLoss, lambda v: v.avg)

    def top1(self):
        return self._createTableWithRoundedValues(self._top1, lambda v: v.avg)

    def update(self, ratio, logits, target, loss):
        n = logits.size(0)
        prec1 = self.accuracy(logits, target)[0]
        self._epochLoss[ratio].update(loss.item(), n)
        self._top1[ratio].update(prec1.item(), n)
        self._prec1[ratio] = prec1.item()
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
