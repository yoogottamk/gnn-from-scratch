import torch
from torch import nn


class BaseGNNLayer(nn.Module):
    def __init__(
        self,
        n_in: int,
        n_out: int,
        activation: nn.Module,
    ):
        super().__init__()

        self.n_in = n_in
        self.n_out = n_out
        self.w = nn.parameter.Parameter(torch.FloatTensor(n_in, n_out).random_(-1, 1) * 0.01)

        self.activation = activation

    def aggregate(self, x, adj):
        return adj @ x

    def combine(self, x, msg):
        return self.activation(x + msg)

    def forward(self, x, adj):
        x = x @ self.w

        msg = self.aggregate(x, adj)
        x = self.combine(x, msg)

        return x
