import random
from enum import Enum
from typing import List, Optional

import numpy as np
import torch
from torch import nn, optim
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
        activation: nn.Module,
        norm: Optional[GCNNorm] = None,
    ):
        super().__init__(n_in, n_out, activation)

        self.norm = norm

    def aggregate(self, x, adj):
        N = adj.size(0)
        a_ = adj + torch.eye(N).to(adj.get_device())

        d = torch.zeros((N, N)).to(adj.get_device())
        d[range(N), range(N)] = torch.pow(a_.sum(1), -0.5)

        if self.norm == GCNNorm.COL:
            a_ = d @ a_
        elif self.norm == GCNNorm.ROW:
            a_ = a_ @ d
        elif self.norm == GCNNorm.SYMMETRIC:
            a_ = d @ a_ @ d

        return a_ @ x


class GCN(nn.Module):
    def __init__(self, layer_descriptions: List[int], norm: Optional[GCNNorm] = None):
        super().__init__()

        self.module_list = nn.ModuleList()
        for i in range(1, len(layer_descriptions)):
            self.module_list.append(
                GCNLayer(
                    layer_descriptions[i - 1],
                    layer_descriptions[i],
                    # dont relu the last layer
                    nn.ELU() if i != len(layer_descriptions) - 1 else nn.Softmax(dim=1),
                    norm,
                )
            )

    def forward(self, x, adj):
        for module in self.module_list:
            x = module(x, adj)
        return x


def train(
    gcn_norm: Optional[GCNNorm] = None,
    seed: int = 42,
    n_epochs: int = 300,
    train_ratio: float = 0.75,
    device="cpu",
):
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

    model = GCN([len(word_attributes[0]), 16, len(class_1hot[0])], gcn_norm)
    loss_fn = nn.CrossEntropyLoss(reduction="none")
    optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

    model.to(device)
    word_attributes = word_attributes.to(device)
    class_1hot = class_1hot.to(device)
    adj_matrix = adj_matrix.to(device)
    train_test_mask = train_test_mask.to(device)

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

            t.set_postfix({"Loss": loss.item(), "Accuracy": accuracy})

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
    print("No normalization")
    train(gcn_norm=None, device="cuda" if torch.cuda.is_available() else "cpu")

    print("Row normalization")
    train(gcn_norm=GCNNorm.ROW, device="cuda" if torch.cuda.is_available() else "cpu")

    print("Col normalization")
    train(gcn_norm=GCNNorm.COL, device="cuda" if torch.cuda.is_available() else "cpu")

    print("Symmetric normalization")
    train(
        gcn_norm=GCNNorm.SYMMETRIC,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
