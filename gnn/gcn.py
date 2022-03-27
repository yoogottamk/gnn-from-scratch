import random
from enum import Enum
from typing import List, Optional

import numpy as np
import torch
from torch import nn, optim
from torch.nn.modules.activation import ReLU
from tqdm.auto import tqdm

from gnn.base import BaseGNNLayer
from gnn.datasets import load_citeseer_dataset


class GCNNorm(Enum):
    COL = 0
    ROW = 1
    SYMMETRIC = 2


class GCNLayer(BaseGNNLayer):
    def __init__(
        self,
        n_in: int,
        n_out: int,
        norm: Optional[GCNNorm] = None,
        use_activation: bool = True,
    ):
        super().__init__()

        self.n_in = n_in
        self.n_out = n_out
        self.w = nn.parameter.Parameter(torch.FloatTensor(n_in, n_out).random_(-1, 1))

        self.norm = norm

        self.activation = ReLU() if use_activation else nn.Identity()

    def aggregate(self, x, adj):
        N = adj.size(0)
        a_ = adj + torch.eye(N)

        d = torch.zeros((N, N))
        d[range(N), range(N)] = a_.sum(1)
        d = torch.pow(d, -0.5)

        if self.norm == GCNNorm.COL:
            a_ = d @ a_
        elif self.norm == GCNNorm.ROW:
            a_ = a_ @ d
        elif self.norm == GCNNorm.SYMMETRIC:
            a_ = d @ a_ @ d

        return a_ @ x

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
                GCNLayer(
                    layer_descriptions[i - 1],
                    layer_descriptions[i],
                    GCNNorm.SYMMETRIC,
                    # dont relu the last layer
                    use_activation=i != len(layer_descriptions) - 1,
                )
            )

    def forward(self, x, adj):
        for module in self.module_list:
            x = module(x, adj)
        return x


def train(seed: int = 42, n_epochs: int = 300, train_ratio: float = 0.75, device="cpu"):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

    device = torch.device(device)

    (
        adj_matrix,
        word_attributes,
        class_1hot,
        class_labels2idx,
        paper_ids,
        paper_id2idx,
        train_test_mask,
    ) = load_citeseer_dataset(train_ratio=train_ratio)

    model = GCN([len(word_attributes[0]), 16, len(class_1hot[0])])
    loss_fn = nn.CrossEntropyLoss(reduction="none")
    optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

    model.to(device)
    word_attributes.to(device)
    class_1hot.to(device)

    with tqdm(range(n_epochs)) as t:
        for _ in t:
            optimizer.zero_grad()
            y = model(word_attributes, adj_matrix)
            loss = loss_fn(y, class_1hot.argmax(1))
            loss *= train_test_mask
            loss = loss.sum()
            loss /= train_test_mask.sum()
            loss.backward()
            accuracy = (
                (y.argmax(1).detach() == class_1hot.argmax(1).detach())
                * train_test_mask
            ).sum() / train_test_mask.sum()
            optimizer.step()

            t.set_description_str(f"Loss: {loss.item():0.6} | Accuracy: {accuracy:0.6}")

    if train_ratio == 1:
        return

    model.eval()
    y = model(word_attributes, adj_matrix)
    test_loss = loss_fn(y, class_1hot.argmax(1))
    test_loss *= 1 - train_test_mask
    test_loss = test_loss.sum()
    test_loss /= (1 - train_test_mask).sum()
    test_accuracy = (
        (y.argmax(1).detach() == class_1hot.argmax(1).detach()) * (1 - train_test_mask)
    ).sum() / (1 - train_test_mask).sum()

    print(f"Test loss: {test_loss.item():0.6} | Test accuracy: {test_accuracy:0.6}")


if __name__ == "__main__":
    train(device="cuda" if torch.cuda.is_available() else "cpu")
