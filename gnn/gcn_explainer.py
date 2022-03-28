import random
from typing import List

import numpy as np
import torch
from torch import nn, optim
from tqdm.auto import tqdm


class GCNLayer(nn.Module):
    def __init__(self, n_in: int, n_out: int):
        super().__init__()
        self.w = nn.parameter.Parameter(torch.Tensor(n_in, n_out).random_(-1, 1))

    def aggregate(self, x, adj):
        return adj @ x

    def combine(self, x, msg):
        return x + msg

    def forward(self, x, adj):
        x = x @ self.w

        msg = self.aggregate(x, adj)
        x = self.combine(x, msg)

        return x


class GCN(nn.Module):
    def __init__(self, layer_descriptions: List[int]):
        super().__init__()

        self.module_list = nn.ModuleList()
        for i in range(1, len(layer_descriptions)):
            self.module_list.append(
                GCNLayer(layer_descriptions[i - 1], layer_descriptions[i])
            )

    def forward(self, x, adj):
        for module in self.module_list:
            x = module(x, adj)
        return x


def train(
    seed: int = 42,
    n_epochs: int = 3,
    device="cpu",
):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

    device = torch.device(device)

    adj_matrix = torch.tensor(
        [
            [0, 1, 0, 0, 1, 0, 0, 1, 0, 0],
            [1, 0, 1, 1, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 1, 1, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 0, 0, 0, 1, 1],
            [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
        ]
    ).float()

    class_labels = torch.tensor(
        [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 1, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 1, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
            [0, 0, 0, 1],
            [0, 0, 0, 1],
        ]
    ).float()

    node_features = torch.tensor(
        [
            [0, 1, 0, 0, 1, 0, 0, 1, 0, 0],
            [1, 0, 1, 1, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 1, 1, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 0, 0, 0, 1, 1],
            [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
        ]
    ).float()

    model = GCN([len(node_features[0]), 6, 4])
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1)

    model.to(device)
    node_features = node_features.to(device)
    class_labels = class_labels.to(device)
    adj_matrix = adj_matrix.to(device)

    with tqdm(range(n_epochs)) as t:
        for _ in t:
            optimizer.zero_grad()
            y = model(node_features, adj_matrix)
            loss = loss_fn(y, class_labels.argmax(1))
            loss.backward()
            optimizer.step()

            t.set_postfix({"loss": loss.item()})

    model.eval()
    print(model(node_features, adj_matrix))


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    device = "cpu"
    train(device=device)
