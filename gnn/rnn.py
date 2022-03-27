import torch
from torch import nn
from torch.nn.modules.activation import Tanh
from tqdm.auto import tqdm

from gnn.base import BaseGNNLayer
from gnn.datasets import load_imdb_dataset


class RNNLayer(BaseGNNLayer):
    def __init__(self, n_in: int, n_out: int, activation: nn.Module, vocab_size: int):
        super().__init__(activation)
        self.embedding = nn.Embedding(vocab_size, n_in)
        self.w = nn.Linear(n_in, n_out)
        self.h = nn.parameter.Parameter(torch.FloatTensor((n_out)))

    def aggregate(self, x):
        emb = self.embedding(x)
        return self.w(emb)

    def combine(self, x, prev_out):
        return super().combine(x, prev_out)

    def forward(self, x, prev_out):
        x = self.aggregate(x)
        return self.combine(x, prev_out)


def train(
    vocab_size: int = 5000,
    vector_size: int = 256,
    seed: int = 42,
    n_epochs=300,
    device="cpu",
):
    (x_train, y_train), (x_test, y_test) = load_imdb_dataset(
        seed=seed, vocab_size=vocab_size, vector_size=vector_size
    )

    x_train = x_train[:1000, :]

    model = RNNLayer(128, 2, Tanh(), vocab_size)

    out = model(x_train[:, 0], torch.zeros((x_train.size(0), 2)))
    with tqdm(range(n_epochs - 1)) as t:
        for _ in t:
            for j in range(1, vector_size):
                out = model(x_train[:, j], out)

    y = out.argmax(1)


if __name__ == "__main__":
    train()
