from torch import tensor, float32
from torch.nn import CrossEntropyLoss, Module
from numpy import linspace
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


# loss function class
class LossFunction:
    def __init__(self, minFlops):
        self.minFlops = minFlops

    def calcLoss(self, modelFlops):
        v = (modelFlops / self.minFlops) ** 2
        return tensor(v, dtype=float32).cuda()


class FlopsLoss(Module):
    # init loss types keys
    _totalKey = 'Total'
    _crossEntropyKey = 'CrossEntropy'
    _flopsKey = 'Flops'
    _lossKeys = [_totalKey, _crossEntropyKey, _flopsKey]

    def __init__(self, args, baselineFlopsDict):
        super(FlopsLoss, self).__init__()

        self.lmbda = args.lmbda
        self.crossEntropyLoss = CrossEntropyLoss().cuda()
        self.baselineFlops = baselineFlopsDict.get(args.baseline)

        self.flopsLoss = LossFunction(self.baselineFlops).calcLoss
        self.flopsLossImgPath = '{}/flops_loss_func.pdf'.format(args.save)
        self._plotFunction(self.flopsLoss, min(baselineFlopsDict.values()), max(baselineFlopsDict.values()))

    @staticmethod
    def lossKeys():
        return FlopsLoss._lossKeys

    @staticmethod
    def totalKey():
        return FlopsLoss._totalKey

    def forward(self, input, target, modelFlops):
        loss = {self._crossEntropyKey: self.crossEntropyLoss(input, target),
                self._flopsKey: self.lmbda * self.flopsLoss(modelFlops)}
        loss[self._totalKey] = sum(loss.values())

        return loss

    def _plotFunction(self, func, xMin, xMax):
        # build data for function
        nPts = (5 * 100) + 1
        ptsGap = int((nPts - 1) / 20)

        pts = linspace(xMin, xMax, nPts).tolist()
        y = [round(func(x).item(), 5) for x in pts]
        data = [[pts, y, 'bo']]
        pts = [pts[x] for x in range(0, nPts, ptsGap)]
        y = [y[k] for k in range(0, nPts, ptsGap)]
        data.append([pts, y, 'go'])

        # plot
        fig, ax = plt.subplots(nrows=1, ncols=1)
        for x, y, style in data:
            ax.plot(x, y, style)

        ax.set_xticks(pts)
        ax.set_yticks(y)
        ax.set_xlabel('flops/baselineFlops')
        ax.set_ylabel('Loss')
        ax.set_title('Flops ratio loss function')
        fig.set_size_inches(25, 10)

        pdf = PdfPages(self.flopsLossImgPath)
        pdf.savefig(fig)
        pdf.close()
