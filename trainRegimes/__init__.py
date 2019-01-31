from .MultinomialSearchRegime import MultinomialSearchRegime
from .CategoricalSearchRegime import CategoricalSearchRegime
from .BinomialSearchRegime import BinomialSearchRegime

from utils.args import Switcher


class SearchRegimeSwitcher:
    _categoricalKey = Switcher.categoricalKey()
    _multinomialKey = Switcher.multinomialKey()
    _binomialKey = Switcher.binomialKey()

    _SearchRegimeDict = {_categoricalKey: CategoricalSearchRegime, _multinomialKey: MultinomialSearchRegime, _binomialKey: BinomialSearchRegime}

    @staticmethod
    def getSearchRegimeClass(classKey):
        return SearchRegimeSwitcher._SearchRegimeDict[classKey]
