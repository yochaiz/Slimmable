from .MultinomialSearchRegime import MultinomialSearchRegime
from .CategoricalSearchRegime import CategoricalSearchRegime

from utils.args import Switcher


class SearchRegimeSwitcher:
    _categoricalKey = Switcher.categoricalKey()
    _multinomialKey = Switcher.multinomialKey()

    _SearchRegimeDict = {_categoricalKey: CategoricalSearchRegime, _multinomialKey: MultinomialSearchRegime}

    @staticmethod
    def getSearchRegimeClass(classKey):
        return SearchRegimeSwitcher._SearchRegimeDict[classKey]
