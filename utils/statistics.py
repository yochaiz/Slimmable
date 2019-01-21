from os import makedirs, path
from math import ceil
from io import BytesIO
from base64 import b64encode
from urllib.parse import quote
from numpy import linspace

from torch import save as saveFile

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.backends.backend_pdf import PdfPages


class Statistics:
    # set plot points style
    ptsStyle = '-'

    # set maxCols & minRows for multiplot
    nColsMax = 7
    nRowsDefault = 3

    def __init__(self, containers, rules, saveFolder):
        # create plot folder
        plotFolderPath = '{}/plots'.format(saveFolder)
        if not path.exists(plotFolderPath):
            makedirs(plotFolderPath)

        self.saveFolder = plotFolderPath
        # init containers
        self._containers = containers
        # init rules
        self._rules = rules
        # init colors map
        self.colormap = plt.cm.hot
        # init plots data dictionary
        self.plotsDataFilePath = '{}/plots.data'.format(saveFolder)
        self.plotsData = {}
        # init flopsData, which is a map where keys are labels (pts type) and values are list of tuples (bitwidth, flops, accuracy)
        self.flopsData = {}

    def addValue(self, getListFunc, value):
        list = getListFunc(self._containers)
        list.append(value)

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
    def __setAxesProperties(ax, title, xLabel, yLabel, yMax=None, yMin=0.0):
        ax.set_xlabel(xLabel)
        ax.set_ylabel(yLabel)
        ax.set_ylim(bottom=yMin)
        if yMax:
            ax.set_ylim(top=yMax)
        ax.set_title(title)
        # put legend in bottom right corner, transparent (framealpha), small font
        ax.legend(loc='upper center', ncol=4, fancybox=True, shadow=True, framealpha=0.1, prop={'size': 8})

    @staticmethod
    def __setFigProperties(fig, figSize=(15, 10)):
        fig.set_size_inches(figSize)
        fig.tight_layout()
        # close plot
        plt.close(fig)

    @staticmethod
    def setPlotProperties(fig, ax, title, xLabel, yLabel, yMax=None, yMin=0.0):
        Statistics.__setAxesProperties(ax, title, xLabel, yLabel, yMax, yMin)
        Statistics.__setFigProperties(fig)

    def __plotContainer(self, data, xLabel, yLabel, title, axMerged=None, scale=True, annotate=None):
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
        # self.plotsData[title] = dict(x=xValues, data=[])
        self.plotsData[title] = []

        # for i, layerData in enumerate(data):
        for i, (key, keyDataList) in enumerate(data.items()):
            # move on if there is no data in list
            if len(keyDataList) == 0:
                continue

            # set arguments
            label = key
            color = colors[i]
            # # plot by shortest length between xValues, layerData
            # plotLength = min(len(xValues), len(keyDataList))
            # xValues = xValues[:plotLength]
            # keyDataList = keyDataList[:plotLength]

            # add data to plotsData
            # self.plotsData[title]['data'].append(dict(y=keyDataList, style=self.ptsStyle, label=label, color=color))
            self.plotsData[title].append(dict(y=keyDataList, style=self.ptsStyle, label=label, color=color))
            # plot
            xValues = list(range(len(keyDataList)))
            ax.plot(xValues, keyDataList, self.ptsStyle, label=label, c=color)
            if axMerged:
                axMerged.plot(xValues, keyDataList, self.ptsStyle, label=label, c=color)

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

            # don't scale axMerged
            if axMerged:
                # set grid
                axMerged.grid()
                # set ax properties
                self.__setAxesProperties(axMerged, title, xLabel, yLabel)

            if scale:
                yMax = min(yMax, (sum(dataSum) / len(dataSum)) * 1.5)

            self.setPlotProperties(fig, ax, title, xLabel, yLabel, yMax)

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
        # generate different plots
        # for fileName in self.plotAllLayersKeys:
        for fileName, dataList in self._containers.items():
            # init labels
            xLabel = 'Batch #'
            yLabel = fileName
            # build subplot for all plots
            figs = []
            # init merged plot
            nPlots = len(dataList)
            nRows, nCols = self.__findGrid(nPlots)
            figMerged, axMerged = plt.subplots(nrows=nRows, ncols=nCols, sharey=True)
            axRow, axCol = 0, 0
            figs.append(figMerged)
            # iterate over data elements
            for dataIdx, dataDict in enumerate(dataList):
                # init axMerged sub-plot
                ax = axMerged[axRow, axCol] if nPlots > 1 else axMerged
                # init title
                title = '[{}]-[{}] over epochs'.format(fileName, dataIdx)
                # plot container
                fig = self.__plotContainer(dataDict, xLabel, yLabel, title, axMerged=ax)
                # add fig to figs list
                figs.append(fig)
                # update next axes indices
                if ax:
                    axCol = (axCol + 1) % nCols
                    if axCol == 0:
                        axRow += 1

            if figMerged:
                # apply yMax rule if exists, else retain yMax
                yMax = self._rules.get(yLabel)
                if yMax:
                    ax.set_ylim(top=yMax)
                # set merged fig properties
                self.__setFigProperties(figMerged, figSize=(40, 20))
            # save as PDF
            self.saveFigPDF(figs, fileName, self.saveFolder)
            # save plots data
            saveFile(self.plotsData, self.plotsDataFilePath)

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

# # data is a list of dictionaries
# def addBatchData(self, loss, acc):
#     # update number of batches
#     self.nBatches += 1
#     # add data
#     data = [(loss, self._weightsLossKey), (acc, self._weightsAccKey)]
#     for dataElement, dataKey in data:
#         container = self._containers[dataKey]
#         for title, value in dataElement.items():
#             # init new list to new title
#             if title not in container:
#                 container[title] = []
#             # add value to title list
#             container[title].append(value)
#
#     self.plotData()

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

# def __saveAndPlotFlops(self):
#     # save data to plotData
#     self.plotsData[self._flopsKey] = self.flopsData
#     # save plots data
#     saveFile(self.plotsData, self.plotsDataFilePath)
#     # update plot
#     self.plotFlopsData(self.plotsData[self._flopsKey], self.saveFolder)

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
