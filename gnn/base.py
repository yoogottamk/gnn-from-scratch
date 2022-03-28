from torch import nn


class BaseGNNLayer(nn.Module):
    def __init__(
        self,
        activation: nn.Module,
    ):
        super().__init__()
        self.activation = activation

    def aggregate(self, x, adj):
        # by multiplying with the adjacancy matrix,
        # it collects values of x for neighbours (adj[i][j] = 1) and
        # ignores values of x for nodes that aren't connected (adj[i][j] = 0)
        return adj @ x

    def combine(self, x, msg):
        return self.activation(x + msg)

    def forward(self, x, adj):
        msg = self.aggregate(x, adj)
        x = self.combine(x, msg)
        return x
