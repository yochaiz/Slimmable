from abc import abstractmethod
import scipy.stats as st
from bisect import bisect_left, insort_left
from numpy import mean, linspace

from utils.statistics import Statistics

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D, proj3d


def plotFlopsData(flopsData, funcs, labelsToConnect, fileName, saveFolder):
    flopsFunc, accFunc, partitionFunc = funcs
    # create plots
    plots = [FlopsStandardPlot(xFunc=flopsFunc, yFunc=accFunc),
             FlopsAveragePlot(flopsFunc, accFunc, labelsToConnect),
             FlopsAveragePlot3D(partitionFunc, accFunc, labelsToConnect),
             FlopsMaxAccuracyPlot(flopsFunc, accFunc, labelsToConnect),
             MinFlopsPlot(flopsFunc, accFunc, labelsToConnect)
             ]

    # iterate 1st over non-integer keys
    for labelData in flopsData:
        for checkpoint in labelData.checkpoints():
            for plot in plots:
                plot.addDataPoint(checkpoint)

        for plot in plots:
            plot.plot(labelData)

    # set plot properties
    for plot in plots:
        plot.setPlotProperties()

    # save as HTML
    Statistics.saveFigPDF([plot.fig for plot in plots], fileName, saveFolder)


class PlotLabelData:
    def __init__(self, legendStr, annotateStr, color=None):
        self._legendString = legendStr
        self._annotateString = annotateStr
        self._color = color
        self._checkpoints = []

    def color(self):
        return self._color

    def legendString(self):
        return self._legendString

    def annotateString(self):
        return self._annotateString

    def checkpoints(self):
        for checkpoint in self._checkpoints:
            yield checkpoint

    def addCheckpoint(self, checkpoint):
        self._checkpoints.append(checkpoint)

    def setColor(self, color):
        self._color = color


class FlopsPlot:
    accuracyFormat = '{:.2f}'
    _titleKey = 'title'

    def __init__(self, xFunc, yFunc):
        self.title = self.setTitle()
        # create standard flops plot
        fig, ax = plt.subplots(nrows=1, ncols=1)
        ax.grid()

        self.fig = fig
        self.ax = ax

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
    def addDataPoint(self, checkpoint):
        raise NotImplementedError('subclasses must override addDataPoint()!')

    @abstractmethod
    def setTitle(self):
        raise NotImplementedError('subclasses must override setTitle()!')

    def addStandardDataPoint(self, checkpoint):
        flops = self.xFunc(checkpoint)
        accuracy = self.yFunc(checkpoint)
        title = self.getTitle(checkpoint)

        self.xValues.append(flops)
        self.yValues.append(accuracy)

        # update yMax, yMin
        self.yMax = max(self.yMax, accuracy)
        self.yMin = min(self.yMin, accuracy)

        if title:
            self.ax.annotate('{}'.format(title), (flops, accuracy), size=6)

    def resetPlot(self):
        self.xValues.clear()
        self.yValues.clear()

    # plot label values
    def plot(self, label):
        self.ax.plot(self.xValues, self.yValues, 'o', label=label.legendString(), c=label.color())
        self.plotSpecific(label)
        self.resetPlot()

    @abstractmethod
    def plotSpecific(self, label):
        raise NotImplementedError('subclasses must override plotSpecific()!')


class FlopsPlotWithLine(FlopsPlot):
    def __init__(self, xFunc, yFunc, labelsToConnect):
        super(FlopsPlotWithLine, self).__init__(xFunc, yFunc)

        # init connect line attributes
        self.lineStyle = '--'
        colorMap = plt.get_cmap('Greens')
        self.lineColor = [colorMap(i) for i in linspace(1.0, 0.6, len(labelsToConnect))]
        colorMap = plt.get_cmap('YlGn')
        self.lineErrorBarColor = [colorMap(i) for i in linspace(0.5, 0.2, len(labelsToConnect))]
        # save labels to connect list
        self.labelsToConnect = labelsToConnect
        # save the line points data, in case we want to evaluate other points against the line,
        # i.e. we want to know what is the line y value in any x value
        self.linePoints = [[] for _ in range(len(labelsToConnect))]

    def connectLabel(self, label, x, y):
        # save labelsList indices where label exists
        labelIndices = []
        for idx, labelsList in enumerate(self.labelsToConnect):
            if label in labelsList:
                labelIndices.append(idx)
                # connect points
                if len(self.linePoints[idx]) > 0:
                    xPrev, yPrev = self.linePoints[idx][-1]
                    self.ax.plot([xPrev, x], [yPrev, y], self.lineStyle, c=self.lineColor[idx])
                # add point data to self.linePoints while keeping it sorted
                insort_left(self.linePoints[idx], (x, y))

        return labelIndices


class FlopsAveragePlot(FlopsPlotWithLine):
    # labelsToConnect is list of lists
    # each list contains labels we want to connect with dashed line
    def __init__(self, xFunc, yFunc, labelsToConnect):
        # init confidence interval of 1 std
        self.confidence = 0.6827

        super(FlopsAveragePlot, self).__init__(xFunc, yFunc, labelsToConnect)

        # save previous line point error bar, in order to connect also error bars with line
        self.prevErrorBar = [None] * len(labelsToConnect)

    def setTitle(self):
        return 'Average accuracy vs. Flops | Confidence:[{}]'.format(self.confidence)

    def _connectErrorBars(self, labelIndices, x, y, confidenceHalfInterval):
        # iterate over lines where label is part of
        for idx in labelIndices:
            prevError = self.prevErrorBar[idx]
            # connect bars
            if prevError is not None:
                xPrev, yPrev, confidenceHalfIntervalPrev = prevError
                # connect bar bottom
                self.ax.plot([xPrev, x], [yPrev - confidenceHalfIntervalPrev, y - confidenceHalfInterval],
                             self.lineStyle, c=self.lineErrorBarColor[idx])
                # connect bar top
                self.ax.plot([xPrev, x], [yPrev + confidenceHalfIntervalPrev, y + confidenceHalfInterval],
                             self.lineStyle, c=self.lineErrorBarColor[idx])
            # save last point as previous point
            self.prevErrorBar[idx] = (x, y, confidenceHalfInterval)

    def addDataPoint(self, checkpoint):
        flops = self.xFunc(checkpoint)
        accuracy = self.yFunc(checkpoint)
        # title = self.getTitle(checkpoint)

        if len(self.xValues) == 0:
            self.xValues.append(flops)
        elif (not isinstance(self.xValues[0], list)) and (flops < self.xValues[0]):
            self.xValues = [flops]
        # assert (self.xValues[0] == flops)

        self.yValues.append(accuracy)

    # add error bar to point average
    def plotErrorBar(self, yMean, color, checkConditionsFlag):
        confidenceHalfInterval = 0
        if len(self.yValues) > 1:
            # calc standard error of the mean
            sem = st.sem(self.yValues)
            # calc confidence interval
            intervalMin, intervalMax = st.t.interval(self.confidence, len(self.yValues) - 1, loc=yMean, scale=sem)
            confidenceHalfInterval = (intervalMax - intervalMin) / 2

            if checkConditionsFlag:
                xMean = self.xValues[0]
                for pointsList in self.linePoints:
                    # find current point line interval
                    upperBoundIdx = bisect_left(pointsList, (xMean, yMean))
                    if upperBoundIdx > 0:
                        lowerBoundIdx = upperBoundIdx - 1
                        # calc line slope
                        xUpper, yUpper = pointsList[upperBoundIdx]
                        xLower, yLower = pointsList[lowerBoundIdx]
                        slope = (yUpper - yLower) / (xUpper - xLower)
                        # calc line y value in xMean
                        yLine = yLower + (slope * (xMean - xLower))
                        # do not plot error bar if error bar lower bound is under line y value in xMean
                        if yLine > (yMean - confidenceHalfInterval):
                            return

            self.ax.errorbar(self.xValues, [yMean], yerr=confidenceHalfInterval, c=color,
                             ms=5, marker='X', capsize=4, markeredgewidth=1, elinewidth=2)

        return confidenceHalfInterval

    # plot label values
    def plot(self, label):
        # average accuracy
        yMean = mean(self.yValues)
        self.ax.plot(self.xValues, [yMean], 'o', label=label.legendString(), c=label.color())
        # annotate label
        self.ax.annotate(label.annotateString(), (self.xValues[0], yMean), size=6)

        # update yMax, yMin
        self.yMax = max(self.yMax, yMean)
        self.yMin = min(self.yMin, yMean)

        # connect label
        labelIndices = self.connectLabel(label.legendString(), self.xValues[-1], yMean)
        # plot error bar
        confidenceHalfInterval = self.plotErrorBar(yMean, label.color(), checkConditionsFlag=(len(labelIndices) == 0))
        # connect error bars if we have connected label
        self._connectErrorBars(labelIndices, self.xValues[0], yMean, confidenceHalfInterval)

        # update variables for next plot
        self.resetPlot()


class FlopsAveragePlot3D(FlopsAveragePlot):
    # labelsToConnect is list of lists
    # each list contains labels we want to connect with dashed line
    def __init__(self, xFunc, yFunc, labelsToConnect):
        # init accuracy color range
        self.minColorValue = 82.0
        self.maxColorValue = 89.0
        nKeys = 81
        self.colorsValueList = linspace(self.minColorValue, self.maxColorValue, nKeys)

        super(FlopsAveragePlot3D, self).__init__(xFunc, yFunc, labelsToConnect)

        self.ax = self.fig.add_subplot(1, 1, 1, projection='3d')

        colormap = plt.cm.hot
        self.colors = [colormap(i) for i in linspace(0.9, 0.1, nKeys)]

    def setTitle(self):
        return 'Average accuracy vs. Flops - 3D'

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
        # check all xValues are non-string
        for x in self.xValues[0]:
            if isinstance(x, str):
                return

        x1, x2, x3 = self.xValues[0]
        # average accuracy
        yMean = mean(self.yValues)
        # plot 3D, the color is best on accuracy, i.e. yMean
        self.ax.plot([x1], [x2], [x3], 'o', label=label.legendString(), c=self.valueToColor(yMean))
        # annotate label
        self.ax.text(x1, x2, x3, label.annotateString(), size=6, zorder=1)

        self.resetPlot()


class FlopsStandardPlot(FlopsPlot):
    def __init__(self, xFunc, yFunc):
        super(FlopsStandardPlot, self).__init__(xFunc, yFunc)

    def setTitle(self):
        return 'Accuracy vs. Flops'

    def addDataPoint(self, checkpoint):
        self.addStandardDataPoint(checkpoint)

    def plotSpecific(self, label):
        pass


class FlopsPlotWithCondition(FlopsPlotWithLine):
    def __init__(self, xFunc, yFunc, labelsToConnect):
        super(FlopsPlotWithCondition, self).__init__(xFunc, yFunc, labelsToConnect)

    @abstractmethod
    def condition(self, checkpoint):
        raise NotImplementedError('subclasses must override condition()!')

    def addDataPoint(self, checkpoint):
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
            self.ax.annotate(label.annotateString(), (flops, accuracy), size=6)
            # connect label
            self.connectLabel(label.legendString(), self.xValues[0], self.yValues[0])


class FlopsMaxAccuracyPlot(FlopsPlotWithCondition):
    def __init__(self, xFunc, yFunc, labelsToConnect):
        super(FlopsMaxAccuracyPlot, self).__init__(xFunc, yFunc, labelsToConnect)

    def setTitle(self):
        return 'Max accuracy vs. Flops'

    def condition(self, checkpoint):
        accuracy = self.yFunc(checkpoint)
        return len(self.yValues) == 0 or accuracy > self.yValues[0]


class MinFlopsPlot(FlopsPlotWithCondition):
    def __init__(self, xFunc, yFunc, labelsToConnect):
        super(MinFlopsPlot, self).__init__(xFunc, yFunc, labelsToConnect)

    def setTitle(self):
        return 'Accuracy vs. Min Flops'

    def condition(self, checkpoint):
        flops = self.xFunc(checkpoint)
        return len(self.xValues) == 0 or flops < self.xValues[0]
