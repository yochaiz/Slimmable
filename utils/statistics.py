from scipy.stats import entropy
from os import makedirs, path
from math import ceil
from io import BytesIO
from base64 import b64encode
from urllib.parse import quote
from numpy import linspace, mean
from abc import abstractmethod
import scipy.stats as st

import torch.nn.functional as F
from torch import save as saveFile

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.backends.backend_pdf import PdfPages
from mpl_toolkits.mplot3d import Axes3D, proj3d


class Statistics:
    _entropyKey = 'alphas_entropy'
    _alphaDistributionKey = 'alphas_distribution'
    _lossVarianceKey = 'loss_variance'
    _lossAvgKey = 'loss_avg'
    _crossEntropyLossAvgKey = 'cross_entropy_loss_avg'
    _flopsLossAvgKey = 'flops_loss_avg'
    _flopsKey = 'flops'
    _weightsLossKey = 'weights_loss'
    _weightsAccKey = 'weights_acc'

    # set plot points style
    ptsStyle = '-'

    # set maxCols & minRows for multiplot
    nColsMax = 7
    nRowsDefault = 3

    def __init__(self, saveFolder):
        # create plot folder
        plotFolderPath = '{}/plots'.format(saveFolder)
        if not path.exists(plotFolderPath):
            makedirs(plotFolderPath)

        self.saveFolder = plotFolderPath
        # init containers
        self.containers = {
            # self._entropyKey: [[] for _ in range(nLayers)],
            # self._lossVarianceKey: [[]], self._alphaDistributionKey: [[[] for _ in range(layer.numOfOps())] for layer in layersList],
            self._lossAvgKey: [[]], self._crossEntropyLossAvgKey: [[]], self._flopsLossAvgKey: [[]],
            self._weightsLossKey: {}, self._weightsAccKey: {}

        }
        # map each list we plot for all layers on single plot to filename
        self.plotAllLayersKeys = [self._entropyKey, self._lossAvgKey, self._crossEntropyLossAvgKey, self._flopsLossAvgKey, self._lossVarianceKey]
        self.plotLayersSeparateKeys = [self._alphaDistributionKey]
        self.containersToPlot = [self._weightsLossKey, self._weightsAccKey]
        # init number of batches
        self.nBatches = 0
        # init colors map
        self.colormap = plt.cm.hot
        # init plots data dictionary
        self.plotsDataFilePath = '{}/plots.data'.format(saveFolder)
        self.plotsData = {}
        # init flopsData, which is a map where keys are labels (pts type) and values are list of tuples (bitwidth, flops, accuracy)
        self.flopsData = {}

    @staticmethod
    def flopsKey():
        return Statistics._flopsKey

    # data is a list of dictionaries
    def addBatchData(self, loss, acc):
        # update number of batches
        self.nBatches += 1
        # add data
        data = [(loss, self._weightsLossKey), (acc, self._weightsAccKey)]
        for dataElement, dataKey in data:
            container = self.containers[dataKey]
            for title, value in dataElement.items():
                # init new list to new title
                if title not in container:
                    container[title] = []
                # add value to title list
                container[title].append(value)

        self.plotData()

    # def addBatchData(self, model):
    #     # update number of batches
    #     self.nBatches += 1
    #     # add data per layer
    #     for i, layer in enumerate(model.layersList):
    #         # calc layer alphas distribution
    #         probs = F.softmax(layer.alphas, dim=-1).detach()
    #         # save distribution
    #         for j, p in enumerate(probs):
    #             self.containers[self._alphaDistributionKey][i][j].append(p.item())
    #         # calc entropy
    #         self.containers[self._entropyKey][i].append(entropy(probs))
    #
    #     # plot data
    #     self.plotData()

    def __saveAndPlotFlops(self):
        # save data to plotData
        self.plotsData[self._flopsKey] = self.flopsData
        # save plots data
        saveFile(self.plotsData, self.plotsDataFilePath)
        # update plot
        self.plotFlops(self.plotsData[self._flopsKey], self.saveFolder)

    # # flopsData_ is a map where keys are bitwidth and values are flops.
    # # we need to find the appropriate checkpoint for accuracy values.
    # def addBaselineFlopsData(self, args, flopsData_):
    #     label = self.baselineLabel
    #     # init label list if label doesn't exist
    #     if label not in self.flopsData.keys():
    #         self.flopsData[label] = []
    #
    #     # add data to list
    #     for bitwidth, flops in flopsData_.items():
    #         # load checkpoint
    #         checkpoint, _ = cnn.utils.loadCheckpoint(args.dataset, args.model, bitwidth)
    #         if checkpoint is not None:
    #             accuracy = checkpoint.get('best_prec1')
    #             if accuracy is not None:
    #                 self.flopsData[label].append((bitwidth, flops, accuracy))
    #
    #     # save & plot flops
    #     self.__saveAndPlotFlops()

    # # flopsData_ is a dictionary where keys are labels and values are list of tuples of (bitwidth, flops, accuracy)
    # def addFlopsData(self, flopsData_):
    #     for label in flopsData_.keys():
    #         # init label list if label doesn't exist
    #         if label not in self.flopsData.keys():
    #             self.flopsData[label] = []
    #
    #         # append values to self.flopsData
    #         self.flopsData[label].extend(flopsData_[label])
    #
    #     # save & plot flops
    #     self.__saveAndPlotFlops()

    @staticmethod
    def saveFigPDF(figs, fileName, saveFolder):
        pdf = PdfPages('{}/{}.pdf'.format(saveFolder, fileName))
        for fig in figs:
            pdf.savefig(fig)

        pdf.close()

    def saveFigHTML(self, figs, fileName):
        # create html page
        htmlCode = '<!DOCTYPE html><html><head><style>' \
                   'table { font-family: gisha; border-collapse: collapse;}' \
                   'td, th { border: 1px solid #dddddd; text-align: center; padding: 8px; white-space:pre;}' \
                   '.collapsible { background-color: #777; color: white; cursor: pointer; padding: 18px; border: none; text-align: left; outline: none; font-size: 15px; }' \
                   '.active, .collapsible:hover { background-color: #555; }' \
                   '.content { max-height: 0; overflow: hidden; transition: max-height 0.2s ease-out;}' \
                   '</style></head>' \
                   '<body>'
        for fig in figs:
            # convert fig to base64
            canvas = FigureCanvas(fig)
            png_output = BytesIO()
            canvas.print_png(png_output)
            img = b64encode(png_output.getvalue())
            img = '<img src="data:image/png;base64,{}">'.format(quote(img))
            # add image to html code
            htmlCode += img
        # close html tags
        htmlCode += '</body></html>'
        # write html code to file
        with open('{}/{}.html'.format(self.saveFolder, fileName), 'w') as f:
            f.write(htmlCode)

    @staticmethod
    def __setAxesProperties(ax, xLabel, yLabel, yMax, title, yMin=0.0):
        # ax.set_xticks(xValues)
        ax.set_xlabel(xLabel)
        ax.set_ylabel(yLabel)
        ax.set_ylim(top=yMax, bottom=yMin)
        ax.set_title(title)
        # put legend in bottom right corner, transparent (framealpha), small font
        ax.legend(loc='lower right', ncol=5, fancybox=True, shadow=True, framealpha=0.1, prop={'size': 6})

    @staticmethod
    def __setFigProperties(fig, figSize=(15, 10)):
        fig.set_size_inches(figSize)
        fig.tight_layout()
        # close plot
        plt.close(fig)

    @staticmethod
    def setPlotProperties(fig, ax, xLabel, yLabel, yMax, title, yMin=0.0):
        Statistics.__setAxesProperties(ax, xLabel, yLabel, yMax, title, yMin)
        Statistics.__setFigProperties(fig)

    def __plotContainer(self, data, xValues, xLabel, yLabel, title, labelFunc, axOther=None, scale=True, annotate=None):
        # create plot
        fig, ax = plt.subplots(nrows=1, ncols=1)
        # init ylim values
        dataMax = 0
        dataSum = []
        # init flag to check whether we have plotted something or not
        isPlotEmpty = True
        # init colors
        colors = [self.colormap(i) for i in linspace(0.6, 0.0, len(data))]
        # reset plot data in plotsData dictionary
        self.plotsData[title] = dict(x=xValues, data=[])

        # for i, layerData in enumerate(data):
        for i, (key, keyDataList) in enumerate(data.items()):
            # set arguments
            label = key
            color = colors[i]
            # plot by shortest length between xValues, layerData
            plotLength = min(len(xValues), len(keyDataList))
            xValues = xValues[:plotLength]
            keyDataList = keyDataList[:plotLength]

            # add data to plotsData
            self.plotsData[title]['data'].append(dict(y=keyDataList, style=self.ptsStyle, label=label, color=color))
            # plot
            ax.plot(xValues, keyDataList, self.ptsStyle, label=label, c=color)
            if axOther:
                axOther.plot(xValues, keyDataList, self.ptsStyle, label=label, c=color)

            isPlotEmpty = False
            dataMax = max(dataMax, max(keyDataList))
            dataSum.append(sum(keyDataList) / len(keyDataList))

        if not isPlotEmpty:
            # add annotations if exists
            if annotate:
                for txt, pt in annotate:
                    ax.annotate(txt, pt)

            # set yMax
            yMax = dataMax * 1.1

            # don't scale axOther
            if axOther:
                if yLabel == self._alphaDistributionKey:
                    yMax = 1.1

                axOther.grid()
                self.__setAxesProperties(axOther, xLabel, yLabel, yMax, title)

            if scale:
                yMax = min(yMax, (sum(dataSum) / len(dataSum)) * 1.5)

            self.setPlotProperties(fig, ax, xLabel, yLabel, yMax, title)

        return fig

    # find optimal grid
    def __findGrid(self, nPlots):
        nRowsOpt, nColsOpt = 0, 0
        optDiff = None

        # iterate over options
        for nRows in range(1, self.nRowsDefault + 1):
            nCols = ceil(nPlots / nRows)
            # calc how many empty plots will be in grid
            diff = nRows * nCols - nPlots
            # update if it is a better grid
            if (nCols <= self.nColsMax) and ((optDiff is None) or (diff < optDiff)):
                nRowsOpt = nRows
                nColsOpt = nCols
                optDiff = diff

        # if we haven't found a valid grid, use nColsMax as number of cols and adjust number of rows accordingly
        if optDiff is None:
            nRowsOpt = ceil(nPlots, self.nColsMax)
            nColsOpt = self.nColsMax

        return nRowsOpt, nColsOpt

    def plotData(self):
        # set x axis values
        xValues = list(range(self.nBatches))
        # generate different plots
        # for fileName in self.plotAllLayersKeys:
        for fileName in self.containersToPlot:
            data = self.containers[fileName]
            fig = self.__plotContainer(data, xValues, xLabel='Batch #', yLabel=fileName, title='{} over epochs'.format(fileName),
                                       labelFunc=lambda x: x)

            self.saveFigPDF([fig], fileName, self.saveFolder)

        return

        for fileName in self.plotLayersSeparateKeys:
            data = self.containers[fileName]
            # build subplot for all plots
            nPlots = len(data)
            nRows, nCols = self.__findGrid(nPlots)
            fig, ax = plt.subplots(nrows=nRows, ncols=nCols)
            axRow, axCol = 0, 0
            figs = [fig]
            # add each layer alphas data to plot
            for i, layerData in enumerate(data):
                layerFig = self.__plotContainer(layerData, xValues, xLabel='Batch #', axOther=ax[axRow, axCol],
                                                title='{} --layer:[{}]-- over epochs'.format(fileName, i), yLabel=fileName,
                                                labelFunc=lambda x: self.layersBitwidths[i][x])
                figs.append(layerFig)
                # update next axes indices
                axCol = (axCol + 1) % nCols
                if axCol == 0:
                    axRow += 1

            # set fig properties
            self.__setFigProperties(fig, figSize=(40, 20))
            # save as HTML
            self.saveFigPDF(figs, fileName, self.saveFolder)

        # save plots data
        saveFile(self.plotsData, self.plotsDataFilePath)

    @staticmethod
    def plotFlops(flopsData, funcs, labelsToConnect, labelsMap, fileName, saveFolder):
        flopsFunc, accFunc, partitionFunc = funcs
        # create plots
        plots = [FlopsStandardPlot(flopsData.keys(), xFunc=flopsFunc, yFunc=accFunc),
                 FlopsAveragePlot(flopsData.keys(), flopsFunc, accFunc, labelsToConnect, labelsMap),
                 FlopsAveragePlot3D(partitionFunc, accFunc, labelsToConnect, labelsMap),
                 FlopsMaxAccuracyPlot(flopsData.keys(), flopsFunc, accFunc, labelsToConnect, labelsMap),
                 MinFlopsPlot(flopsData.keys(), flopsFunc, accFunc, labelsToConnect, labelsMap)
                 ]

        # iterate 1st over non-integer keys
        for label in sorted(flopsData.keys()):
            labelFlopsData = flopsData[label]
            for checkpoint in labelFlopsData:
                for plot in plots:
                    plot.addDataPoint(checkpoint, label)

            for plot in plots:
                plot.plot(label)

        # set plot properties
        for plot in plots:
            plot.setPlotProperties()

        # save as HTML
        Statistics.saveFigPDF([plot.fig for plot in plots], fileName, saveFolder)


class FlopsPlot:
    accuracyFormat = '{:.2f}'
    _titleKey = 'title'

    def __init__(self, title, nKeys, xFunc, yFunc):
        self.title = title
        # create standard flops plot
        fig, ax = plt.subplots(nrows=1, ncols=1)
        ax.grid()

        self.fig = fig
        self.ax = ax

        # init colors
        colormap = plt.cm.hot
        self.colors = [colormap(i) for i in linspace(0.7, 0.0, len(nKeys))]
        self.nextColorIdx = 0

        # init yMax, yMin
        self.yMax = 0.0
        self.yMin = 100.0

        # init values
        self.xValues = []
        self.yValues = []

        # save functions to get x & y values from checkpoint
        self.xFunc = xFunc
        self.yFunc = yFunc

        # set axes labels
        self.xLabel = 'Flops'
        self.yLabel = 'Accuracy'

    def setPlotProperties(self):
        paddingPercentage = 0.02
        paddingSize = (self.yMax - self.yMin) * paddingPercentage
        yMax = self.yMax + paddingSize
        yMin = self.yMin - paddingSize

        self.ax.locator_params(nbins=20, axis='y')
        from matplotlib.ticker import MultipleLocator
        spacing = 0.5
        minorLocator = MultipleLocator(spacing)
        self.ax.yaxis.set_minor_locator(minorLocator)
        # Set grid to use minor tick locations.
        self.ax.grid(which='minor')

        Statistics.setPlotProperties(self.fig, self.ax, xLabel=self.xLabel, yLabel=self.yLabel, title=self.title, yMin=yMin, yMax=yMax)

    @staticmethod
    def getTitleKey():
        return FlopsPlot._titleKey

    def getTitle(self, checkpoint):
        return getattr(checkpoint, self._titleKey, None)

    @abstractmethod
    def addDataPoint(self, checkpoint, label):
        raise NotImplementedError('subclasses must override addDataPoint()!')

    def addStandardDataPoint(self, checkpoint):
        flops = self.xFunc(checkpoint)
        accuracy = self.yFunc(checkpoint)
        title = self.getTitle(checkpoint)

        self.xValues.append(flops)
        self.yValues.append(accuracy)

        # update yMax, yMin
        self.yMax = max(self.yMax, accuracy)
        self.yMin = min(self.yMin, accuracy)
        # title might be None
        # txt = self.accuracyFormat.format(accuracy)
        # if title:
        #     txt = '{},{}'.format(title, txt)
        # # annotate
        # self.ax.annotate(txt, (flops, accuracy), size=6)

        if title:
            self.ax.annotate('{}'.format(title), (flops, accuracy), size=6)

    def resetPlot(self):
        self.xValues.clear()
        self.yValues.clear()

    # plot label values
    def plot(self, label):
        self.ax.plot(self.xValues, self.yValues, 'o', label=label, c=self.colors[self.nextColorIdx])
        self.plotSpecific(label)
        self.nextColorIdx += 1
        self.resetPlot()

    @abstractmethod
    def plotSpecific(self, label):
        raise NotImplementedError('subclasses must override plotSpecific()!')


class FlopsPlotWithConnection(FlopsPlot):
    def __init__(self, title, nKeys, xFunc, yFunc, labelsToConnect, labelsMap):
        super(FlopsPlotWithConnection, self).__init__(title, nKeys, xFunc, yFunc)

        # save labels map
        self.labelsMap = labelsMap
        # save labels to connect list
        self.labelsToConnect = labelsToConnect
        # save previous point per labels list, in order to connect last 2 points with a dashed line
        self.previousPoint = [None] * len(labelsToConnect)

    def connectLabel(self, label, x, y):
        for idx, labelsList in enumerate(self.labelsToConnect):
            if label in labelsList:
                # connect points
                if self.previousPoint[idx] is not None:
                    xPrev, yPrev = self.previousPoint[idx]
                    self.ax.plot([xPrev, x], [yPrev, y], '--', c=self.colors[self.nextColorIdx])
                # save last point as previous point
                self.previousPoint[idx] = (x, y)


class FlopsAveragePlot(FlopsPlotWithConnection):
    # labelsToConnect is list of lists
    # each list contains labels we want to connect with dashed line
    def __init__(self, nKeys, xFunc, yFunc, labelsToConnect, labelsMap):
        # init confidence interval of 1 std
        self.confidence = 0.6827

        title = 'Average accuracy vs. Flops | Confidence:[{}]'.format(self.confidence)
        super(FlopsAveragePlot, self).__init__(title, nKeys, xFunc, yFunc, labelsToConnect, labelsMap)

    def addDataPoint(self, checkpoint, label):
        flops = self.xFunc(checkpoint)
        accuracy = self.yFunc(checkpoint)
        # title = self.getTitle(checkpoint)

        if len(self.xValues) == 0:
            self.xValues.append(flops)
        assert (self.xValues[0] == flops)

        self.yValues.append(accuracy)

    # plot label values
    def plot(self, label):
        # average accuracy
        yMean = mean(self.yValues)
        self.ax.plot(self.xValues, [yMean], 'o', label=label, c=self.colors[self.nextColorIdx])
        # annotate label
        self.ax.annotate(self.labelsMap[label], (self.xValues[0], yMean), size=6)

        # update yMax, yMin
        self.yMax = max(self.yMax, yMean)
        self.yMin = min(self.yMin, yMean)

        # connect label
        self.connectLabel(label, self.xValues[-1], yMean)

        # add error bar if there is more than single value
        if len(self.yValues) > 1:
            # calc standard error of the mean
            sem = st.sem(self.yValues)
            # calc confidence interval
            intervalMin, intervalMax = st.t.interval(self.confidence, len(self.yValues) - 1, loc=yMean, scale=sem)
            confidenceInterval = intervalMax - intervalMin
            self.ax.errorbar(self.xValues, [yMean], yerr=(confidenceInterval / 2),
                             ms=5, marker='X', capsize=4, markeredgewidth=1, elinewidth=2, c=self.colors[self.nextColorIdx])

        # update variables for next plot
        self.nextColorIdx += 1
        self.resetPlot()


class FlopsAveragePlot3D(FlopsAveragePlot):
    # labelsToConnect is list of lists
    # each list contains labels we want to connect with dashed line
    def __init__(self, xFunc, yFunc, labelsToConnect, labelsMap):
        # init accuracy color range
        self.minColorValue = 82.0
        self.maxColorValue = 89.0
        nKeys = 81
        self.colorsValueList = linspace(self.minColorValue, self.maxColorValue, nKeys)

        super(FlopsAveragePlot3D, self).__init__(self.colorsValueList, xFunc, yFunc, labelsToConnect, labelsMap)

        self.ax = self.fig.add_subplot(1, 1, 1, projection='3d')

        colormap = plt.cm.hot
        self.colors = [colormap(i) for i in linspace(0.9, 0.1, nKeys)]

    def setPlotProperties(self):
        Statistics.setPlotProperties(self.fig, self.ax, xLabel='', yLabel='', title=self.title, yMin=0.0, yMax=1.0)
        self.ax.get_legend().remove()

    def valueToColor(self, value):
        lowerIdx = 0
        upperIdx = len(self.colorsValueList)
        while lowerIdx < upperIdx:
            currIdx = lowerIdx + (upperIdx - lowerIdx) // 2
            colorVal = self.colorsValueList[currIdx]
            if value == colorVal:
                break
            elif value > colorVal:
                if lowerIdx == currIdx:  # these two are the actual lines
                    break  # you're looking for
                lowerIdx = currIdx
            elif value < colorVal:
                upperIdx = currIdx

        return self.colors[currIdx]

    # plot label values
    def plot(self, label):
        x1, x2, x3 = self.xValues[0]
        # average accuracy
        yMean = mean(self.yValues)
        # plot 3D, the color is best on accuracy, i.e. yMean
        self.ax.plot([x1], [x2], [x3], 'o', label=label, c=self.valueToColor(yMean))
        # annotate label
        # self.ax.annotate(self.labelsMap[label], (self.xValues[0], yMean), size=6)
        self.ax.text(x1, x2, x3, self.labelsMap[label], size=6, zorder=1)

        self.resetPlot()


class FlopsStandardPlot(FlopsPlot):
    def __init__(self, nKeys, xFunc, yFunc):
        super(FlopsStandardPlot, self).__init__('Accuracy vs. Flops', nKeys, xFunc, yFunc)

    def addDataPoint(self, checkpoint, label):
        self.addStandardDataPoint(checkpoint)

    def plotSpecific(self, label):
        pass


class FlopsPlotWithCondition(FlopsPlotWithConnection):
    def __init__(self, title, nKeys, xFunc, yFunc, labelsToConnect, labelsMap):
        super(FlopsPlotWithCondition, self).__init__(title, nKeys, xFunc, yFunc, labelsToConnect, labelsMap)

    @abstractmethod
    def condition(self, checkpoint):
        raise NotImplementedError('subclasses must override condition()!')

    def addDataPoint(self, checkpoint, label):
        flops = self.xFunc(checkpoint)
        accuracy = self.yFunc(checkpoint)

        if self.condition(checkpoint):
            self.xValues = [flops]
            self.yValues = [accuracy]

            # update yMax, yMin
            self.yMax = max(self.yMax, accuracy)
            self.yMin = min(self.yMin, accuracy)

    def plotSpecific(self, label):
        if (len(self.xValues) > 0) and (len(self.yValues) > 0):
            accuracy = self.yValues[0] if len(self.yValues) > 0 else None
            flops = self.xValues[0] if len(self.xValues) > 0 else None

            # annotate label
            self.ax.annotate(self.labelsMap[label], (flops, accuracy), size=6)
            # connect label
            self.connectLabel(label, self.xValues[0], self.yValues[0])


class FlopsMaxAccuracyPlot(FlopsPlotWithCondition):
    def __init__(self, nKeys, xFunc, yFunc, labelsToConnect, labelsMap):
        super(FlopsMaxAccuracyPlot, self).__init__('Max accuracy vs. Flops', nKeys, xFunc, yFunc, labelsToConnect, labelsMap)

    def condition(self, checkpoint):
        accuracy = self.yFunc(checkpoint)
        return len(self.yValues) == 0 or accuracy > self.yValues[0]


class MinFlopsPlot(FlopsPlotWithCondition):
    def __init__(self, nKeys, xFunc, yFunc, labelsToConnect, labelsMap):
        super(MinFlopsPlot, self).__init__('Accuracy vs. Min Flops', nKeys, xFunc, yFunc, labelsToConnect, labelsMap)

    def condition(self, checkpoint):
        flops = self.xFunc(checkpoint)
        return len(self.xValues) == 0 or flops < self.xValues[0]

# def plotBops(self, layersList):
#     # create plot
#     fig, ax = plt.subplots(nrows=1, ncols=1)
#     # init axis values
#     xValues = [[[] for _ in range(layer.numOfOps())] for layer in layersList]
#     yValues = [[[] for _ in range(layer.numOfOps())] for layer in layersList]
#     # init y axis max value
#     yMax = 0
#     for i, layer in enumerate(layersList):
#         for input_bitwidth in layer.bops.keys():
#             for j, bops in enumerate(layer.bops[input_bitwidth]):
#                 v = bops / 1E6
#                 xValues[i][j].append(i)
#                 yValues[i][j].append(v)
#                 ax.annotate('input:[{}]'.format(input_bitwidth), (i, v))
#                 yMax = max(yMax, v)
#
#     colors = {}
#     for i, (xLayerValues, yLayerValues) in enumerate(zip(xValues, yValues)):
#         for j, (x, y) in enumerate(zip(xLayerValues, yLayerValues)):
#             label = self.layersBitwidths[i][j]
#             if label in colors.keys():
#                 ax.plot(x, y, 'o', label=label, color=colors[label])
#             else:
#                 info = ax.plot(x, y, 'o', label=label)
#                 colors[label] = info[0].get_color()
#
#     yMax *= 1.1
#     self.__setPlotProperties(fig, ax, xLabel='Layer #', yLabel='M-bops', yMax=yMax, title='bops per op in layer')
#     # save as HTML
#     self.saveFigPDF([fig], fileName=self.bopsKey)
