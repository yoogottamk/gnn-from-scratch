import torch
from torch import nn


class BaseGNNLayer(nn.Module):
    def __init__(
        self,
        activation: nn.Module,
    ):
        super().__init__()
        self.activation = activation

    def aggregate(self, x, adj):
        return adj @ x

    def combine(self, x, msg):
        return self.activation(x + msg)

    def forward(self, x, adj):
        msg = self.aggregate(x, adj)
        x = self.combine(x, msg)
        return x
