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
        v = (modelFlops / self.minBops) ** 2
        return tensor(v, dtype=float32).cuda()


class FlopsLoss(Module):
    def __init__(self, args, baselineFlops):
        super(FlopsLoss, self).__init__()

        self.lmbda = args.lmbda
        self.crossEntropyLoss = CrossEntropyLoss().cuda()
        self.baselineFlops = baselineFlops

        self.flopsLoss = LossFunction(self.baselineFlops).calcLoss
        self.flopsLossImgPath = '{}/flops_loss_func.pdf'.format(args.save)
        self._plotFunction(self.flopsLoss)

    def forward(self, input, target, modelFlops):
        crossEntropyLoss = self.crossEntropyLoss(input, target)
        flopsLoss = self.lmbda * self.flopsLoss(modelFlops)
        totalLoss = crossEntropyLoss + flopsLoss
        return totalLoss, crossEntropyLoss, flopsLoss

    def _plotFunction(self, func):
        # build data for function
        xMax = 5
        nPts = (xMax * 100) + 1
        ptsGap = int((nPts - 1) / 50)

        pts = linspace(0, xMax, nPts).tolist()
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
