from abc import abstractmethod
from torch.nn import Module


# abstract class for model block
class Block(Module):
    @abstractmethod
    def getOptimizationLayers(self):
        raise NotImplementedError('subclasses must override getOptimizationLayers()!')

    @abstractmethod
    def getFlopsLayers(self):
        raise NotImplementedError('subclasses must override getFlopsLayers()!')

    @abstractmethod
    def getCountersLayers(self):
        raise NotImplementedError('subclasses must override getCountersLayers()!')

    @abstractmethod
    def outputLayer(self):
        raise NotImplementedError('subclasses must override outputLayer()!')

    @abstractmethod
    def countFlops(self):
        raise NotImplementedError('subclasses must override countFlops()!')

    @abstractmethod
    # make some adjustments in model due to current width selected
    def updateCurrWidth(self):
        raise NotImplementedError('subclasses must override updateCurrWidth()!')

    @abstractmethod
    # generate new BNs for current model path, except for given srcLayer
    def generatePathBNs(self, srcLayer):
        raise NotImplementedError('subclasses must override generatePathBNs()!')
