from torch import nn

from gnn.base import BaseGNNLayer
from gnn.datasets import load_imdb_dataset


class RNNLayer(BaseGNNLayer):
    def __init__(self, n_in: int, n_out: int, activation: nn.Module, vocab_size: int):
        super().__init__(activation)
        self.embedding = nn.Embedding(vocab_size, n_in)
        self.w = nn.Linear(n_in, n_out)
        self.h = 

    def aggregate(self, x):
        return self.w(self.embedding(x))

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
