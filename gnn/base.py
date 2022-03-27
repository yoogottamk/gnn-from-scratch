from torch import nn


class BaseGNNLayer(nn.Module):
    def __init__(self):
        super().__init__()

    def aggregate(self):
        pass

    def combine(self):
        pass

    def forward(self):
        pass
