from .BaseNet import BaseNet


class BaseNet_Multinomial(BaseNet):
    def __init__(self, args, initLayersParams):
        super(BaseNet_Multinomial, self).__init__(args, initLayersParams)
