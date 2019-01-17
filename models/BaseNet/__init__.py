from .BaseNet_categorical import BaseNet_Categorical
from .BaseNet_multinomial import BaseNet_Multinomial


class BaseNetSwitcher:
    _class = None
    _classesDict = {'categorical': BaseNet_Categorical, 'multinomial': BaseNet_Multinomial}

    @staticmethod
    def choose(className):
        BaseNetSwitcher._class = BaseNetSwitcher._classesDict[className]

    @staticmethod
    def getClass():
        return BaseNetSwitcher._class

    @staticmethod
    def getClassesKeys():
        return BaseNetSwitcher._classesDict.keys()
