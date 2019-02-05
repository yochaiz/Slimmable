from abc import abstractmethod
from pandas import DataFrame


class Alphas:
    _roundDigits = 5
    _alphasKey = 'Alphas'
    _alphasCsvFileName = 'alphas.csv'

    def __init__(self, model, saveFolder):
        # init model alphas
        self._alphas = self.buildAlphas(model)

        # init alphas DataFrame
        self.alphas_df = None
        self._initAlphasDataFrame(model, saveFolder)

    @abstractmethod
    def buildAlphas(self, model):
        raise NotImplementedError('subclasses must override buildAlphas()!')

    @abstractmethod
    def initColumns(self, model):
        raise NotImplementedError('subclasses must override initColumns()!')

    @abstractmethod
    def alphasValues(self, model):
        raise NotImplementedError('subclasses must override alphasValues()!')

    @abstractmethod
    def alphasList(self, model):
        raise NotImplementedError('subclasses must override alphasList()!')

    def alphas(self) -> list:
        return self._alphas

    def update(self, srcModelAlphasList: list):
        for dstAlphas, srcAlphas in zip(self.alphas(), srcModelAlphasList):
            dstAlphas.data.copy_(srcAlphas.data)

    def _initAlphasDataFrame(self, model, saveFolder):
        if saveFolder:
            # update save path if saveFolder exists
            self._alphasCsvFileName = '{}/{}'.format(saveFolder, self._alphasCsvFileName)
            # init DataFrame cols
            cols = ['Epoch', 'Batch'] + self.initColumns(model)
            self.cols = cols
            # init DataFrame
            self.alphas_df = DataFrame([], columns=cols)
            # set init data
            data = ['init', 'init']
            # save alphas data
            self.saveCsv(model, data)

    # save alphas values to csv
    def saveCsv(self, model, data):
        if self.alphas_df is not None:
            data += self.alphasValues(model)
            # create new row
            d = DataFrame([data], columns=self.cols)
            # add row
            self.alphas_df = self.alphas_df.append(d)
            # save DataFrame
            self.alphas_df.to_csv(self._alphasCsvFileName)

    def _topAlphas(self, model, k):
        top = []
        for (alphas, probs, widthList, layer) in self.alphasList(model):
            # sort alphas probabilities
            wSorted, wIndices = probs.sort(descending=True)
            # keep only top-k
            wSorted = wSorted[:k]
            wIndices = wIndices[:k]
            # add to top
            top.append([layer, [(i, w.item(), alphas[i], widthList[i]) for w, i in zip(wSorted, wIndices)]])

        return top

    def logTopAlphas(self, model, k, loggerFuncs, logLayer):
        if (not loggerFuncs) or (len(loggerFuncs) == 0):
            return

        rows = [['Layer #', 'Layer', self._alphasKey]] if logLayer else [['Layer #', self._alphasKey]]
        alphaCols = ['Index', 'Ratio', 'Value', 'Width']

        # init add row functions w/o layer
        addRowWithLayerFunc = lambda i, layer, layerRow: [i, layer, layerRow]
        addRowWithoutLayerFunc = lambda i, layer, layerRow: [i, layerRow]
        # init current addRow function, according to logLayer value
        addRowFunc = addRowWithLayerFunc if logLayer else addRowWithoutLayerFunc

        top = self._topAlphas(model, k)
        for i, (layer, layerTop) in enumerate(top):
            layerRow = [alphaCols]
            for idx, w, alpha, width in layerTop:
                alphaRow = [idx, '{:.5f}'.format(w), '{:.5f}'.format(alpha), width]
                # add alpha data row to layer data table
                layerRow.append(alphaRow)
            # add layer data table to model table as row
            rows.append(addRowFunc(i, layer, layerRow))

        # apply loggers functions
        for f in loggerFuncs:
            f(k, rows)
