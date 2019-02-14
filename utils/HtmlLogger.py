from os import path, makedirs
from datetime import datetime
from io import BytesIO
from base64 import b64encode
from urllib.parse import quote
from time import sleep

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas


class HtmlLogger:
    timestampColumnName = 'Timestamp'
    _maxTableCellLengthDefault = 50

    def __init__(self, save_path, filename, overwrite=False):
        self.save_path = save_path
        self._fullPath = '{}/{}.html'.format(save_path, filename)
        self._maxTableCellLength = self._maxTableCellLengthDefault

        if not path.exists(save_path):
            makedirs(save_path)

        if (not overwrite) and path.exists(self.fullPath):
            with open(self.fullPath, 'r') as f:
                content = f.read()
            # remove close tags in order to allow writing to data table
            for v in ['</body>', '</html>', '</table>']:
                idx = content.rfind(v)
                # remove tag from string
                if idx >= 0:
                    content = content[:idx] + content[idx + len(v):]

            self.head = content
            # script already in self.head now, therefore no need it again
            self.script = ''
        else:
            self.head = '<!DOCTYPE html><html><head><style>' \
                        'table { font-family: gisha; border-collapse: collapse; display: block;}' \
                        'td, th { border: 1px solid #dddddd; text-align: center; padding: 8px; white-space:pre;}' \
                        '.collapsible { background-color: #777; color: white; cursor: pointer; padding: 18px; border: none; text-align: left; outline: none; font-size: 15px; }' \
                        '.active, .collapsible:hover { background-color: #555; }' \
                        '.content { max-height: 0; overflow: hidden; transition: max-height 0.2s ease-out;}' \
                        '</style></head>' \
                        '<body>'
            # init collapse script
            self.script = '<script> var coll = document.getElementsByClassName("collapsible"); var i; for (i = 0; i < coll.length; i++) { coll[i].addEventListener("click", function() { this.classList.toggle("active"); var content = this.nextElementSibling; if (content.style.maxHeight){ content.style.maxHeight = null; } else { content.style.maxHeight = content.scrollHeight + "px"; } }); } </script>'

        self.end = '</body></html>'
        self.infoTables = {}
        self.dataTable = ''
        self.dataTableCols = None
        self.nColsDataTable = None
        self.dataTableRowsNum = 0
        self.nRowsPerColumnsRow = 10

    @property
    def fullPath(self):
        return self._fullPath

    # converts dictionary to rows with nElementPerRow (k,v) elements at most in each row
    @staticmethod
    def dictToRows(_dict, nElementPerRow, currSortFunc=None, sortFuncsDict={}):
        rows = []
        row = []
        counter = 0
        # set default dict sort function
        keySortFuncDefault = lambda x: 10 if isinstance(x, str) else x
        sortFuncDefault = lambda kv: keySortFuncDefault(kv[0])
        # set currSortFunc
        currSortFunc = currSortFunc or sortFuncDefault
        # sort elements by keys name
        for k, v in sorted(_dict.items(), key=currSortFunc):
            value = v
            # recursively transform dictionaries to rows
            if isinstance(v, dict):
                # set dict sort function
                sortFunc = sortFuncsDict[k] if k in sortFuncsDict else sortFuncDefault
                # build table rows from dict
                value = HtmlLogger.dictToRows(v, 1, sortFunc, sortFuncsDict)

            row.append(k)
            row.append(value)
            counter += 1

            if counter == nElementPerRow:
                rows.append(row)
                row = []
                counter = 0

        # add last elements
        if len(row) > 0:
            rows.append(row)

        return rows

    def setMaxTableCellLength(self, length):
        if length > 0:
            self._maxTableCellLength = length

    def resetMaxTableCellLength(self):
        self._maxTableCellLength = self._maxTableCellLengthDefault

    def __writeToFile(self):
        # concat info tables to single string
        infoTablesStr = ''
        for title, table in self.infoTables.items():
            infoTablesStr += table
        # init elements write order to file
        writeOrder = [self.head, infoTablesStr, self.dataTable, '</table>', self.script, self.end]
        # write elements
        with open(self.fullPath, 'w') as f:
            for elem in writeOrder:
                if elem is not '':
                    writeSuccess = False
                    while writeSuccess is False:
                        try:
                            # try to write to file
                            f.write(elem)
                            writeSuccess = True
                        except Exception as e:
                            # if couldn't write for some reason, like no space left on device, wait some time until we will free some space
                            print('HtmlLogger write failed, error:[{}]'.format(e))
                            sleep(10 * 60)

    def __addRow(self, row):
        res = '<tr>'
        for v in row:
            isTable = False
            # check maybe we have a sub-table
            if (type(v) is list) and (len(v) > 0) and isinstance(v[0], list):
                v = self.__createTableFromRows(v)
                isTable = True
            # add element or sub-table to current table
            content = '{}'.format(v)
            # add scroll to cell if content is long
            if (isTable is False) and (len(content) > self._maxTableCellLength):
                content = '<div style="width: 250px; overflow: auto"> {} </div>'.format(content)
            # add content as cell
            res += '<td> {} </td>'.format(content)
        res += '</tr>'

        return res

    # recursive function that supports sub-tables
    def __createTableFromRows(self, rows):
        res = '<table>'
        # create rows
        for row in rows:
            res += self.__addRow(row)
        # close table
        res += '</table>'
        return res

    def createInfoTable(self, title, rows):
        # open a new table
        res = '<button class="collapsible"> {} </button>'.format(title)
        res += '<div class="content" style="overflow: auto">'
        # add rows
        res += self.__createTableFromRows(rows)
        # close table
        res += '</div><h2></h2>'

        return res

    # title - a string for table title
    # rows - array of rows. each row is array of values.
    def addInfoTable(self, title, rows):
        # create new table
        self.infoTables[title] = self.createInfoTable(title, rows)
        # write to file
        self.__writeToFile()

    # add row to existing info table by its title
    def addRowToInfoTableByTitle(self, title, row):
        if title in self.infoTables:
            table = self.infoTables[title]
            valuesToFind = ['</table>']
            idx = 0
            # walk through the string to the desired position
            for v in valuesToFind:
                if idx >= 0:
                    idx = table.find(v, idx)

            if idx >= 0:
                # insert new row in desired position
                table = table[:idx] + self.__addRow(row) + table[idx:]
                # update table in infoTables
                self.infoTables[title] = table
                # write to file
                self.__writeToFile()

    @staticmethod
    def __addColumnsRowToTable(cols):
        res = '<tr bgcolor="gray">'
        for c in cols:
            res += '<td> {} </td>'.format(c)
        res += '</tr>'
        # returns columns row
        return res

    def addColumnsRowToDataTable(self, writeToFile=False):
        self.dataTable += self.__addColumnsRowToTable(self.dataTableCols)
        # write to file
        if writeToFile:
            self.__writeToFile()

    def updateDataTableCols(self, dataTableCols):
        # save copy of columns names
        self.dataTableCols = dataTableCols.copy()
        # add timestamp to columns
        self.dataTableCols.insert(0, self.timestampColumnName)
        # save table number of columns
        self.nColsDataTable = len(self.dataTableCols)

    @staticmethod
    def __addTitleRow(title, nCols):
        return '<tr><th colspan={} bgcolor="gray"> {} </th></tr>'.format(nCols, title)

    def createDataTable(self, title, columns):
        self.dataTableRowsNum = 0
        res = ''
        # check if we need to close last data table in page, before starting a new one
        if len(self.dataTable) > 0:
            res += '</table><h2></h2>'

        res += '<table>'
        # update data table columns
        self.updateDataTableCols(columns)
        # create title row
        res += self.__addTitleRow(title, self.nColsDataTable)
        # add table to body
        self.dataTable += res
        # add columns row
        self.addColumnsRowToDataTable()
        # write to file
        self.__writeToFile()

    def getTimeStr(self):
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # values is a dictionary
    def addDataRow(self, values, trType='<tr>', writeFile=True):
        res = trType
        # add timestamp to values
        values[self.timestampColumnName] = self.getTimeStr()
        # build table row, iterate over dictionary
        for c in self.dataTableCols:
            res += '<td>'
            if c in values:
                if isinstance(values[c], list):
                    res += self.__createTableFromRows(values[c])
                else:
                    content = '{}'.format(values[c])
                    if (len(content) > self._maxTableCellLength) and ('</button>' not in content):
                        content = '<div style="width: 300px; overflow: auto"> {} </div>'.format(content)
                    res += content
            res += '</td>'
        # close row
        res += '</tr>'
        # add data to dataTable
        self.dataTable += res
        # update number of data table rows
        self.dataTableRowsNum += 1
        # add columns row if needed
        if self.dataTableRowsNum % self.nRowsPerColumnsRow == 0:
            self.addColumnsRowToDataTable()
        if writeFile:
            # write to file
            self.__writeToFile()

    # add data summary to data table
    # values is a dictionary
    def addSummaryDataRow(self, values):
        self.addDataRow(values, trType='<tr bgcolor="#27AE60">')

    def addInfoToDataTable(self, line, color='lightblue'):
        res = '<tr>'
        res += '<td> {} </td>'.format(self.getTimeStr())
        res += '<td colspan={} bgcolor="{}"> {} </td>'.format(self.nColsDataTable - 1, color, line)
        res += '</tr>'
        # add table to body
        self.dataTable += res
        # write to file
        self.__writeToFile()

    def replaceValueInDataTable(self, oldVal, newVal):
        self.dataTable = self.dataTable.replace(oldVal, newVal)
        # write to file
        self.__writeToFile()

    def plot(self, **kwargs):
        # data is a list, where each element is [x , y , 'bo' (i.e. pts style)]
        data = kwargs.get('data')

        if not data:
            return

        fig, ax = plt.subplots(nrows=1, ncols=1)
        for x, y, style in data:
            ax.plot(x, y, style)

        # init properties we might want to handle
        properties = [('xticks', ax.set_xticks), ('yticks', ax.set_yticks), ('size', fig.set_size_inches),
                      ('xlabel', ax.set_xlabel), ('ylabel', ax.set_ylabel), ('title', ax.set_title)]

        for key, func in properties:
            if key in kwargs:
                func(kwargs[key])

        # set title
        infoTableTitle = kwargs.get('title', 'Plot')

        # convert fig to base64
        canvas = FigureCanvas(fig)
        png_output = BytesIO()
        canvas.print_png(png_output)
        img = b64encode(png_output.getvalue())
        img = '<img src="data:image/png;base64,{}">'.format(quote(img))

        self.addInfoTable(infoTableTitle, [[img]])

# class SimpleLogger(HtmlLogger):
#     def __init__(self, save_path, filename, overwrite=False):
#         super(SimpleLogger, self).__init__(save_path, filename, overwrite)
#
#         self.tableColumn = 'Description'
#         self.createDataTable('Activity', [self.tableColumn])
#
#     def addRow(self, values):
#         super(SimpleLogger, self).addDataRow({self.tableColumn: values})
#
#     def addSummaryRow(self, values):
#         super(SimpleLogger, self).addSummaryDataRow({self.tableColumn: values})
