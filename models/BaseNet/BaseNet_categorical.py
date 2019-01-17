from .BaseNet import BaseNet


class BaseNet_Categorical(BaseNet):
    def __init__(self, args, initLayersParams):
        super(BaseNet_Categorical, self).__init__(args, initLayersParams)
