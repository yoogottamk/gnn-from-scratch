import torch
from torch import nn, optim
from torch.nn.modules.activation import Tanh
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from gnn.base import BaseGNNLayer
from gnn.datasets import load_imdb_dataset


class RNNLayer(BaseGNNLayer):
    def __init__(
        self,
        n_in: int,
        hidden_size: int,
        activation: nn.Module,
    ):
        super().__init__(activation)

        self.wx = nn.Linear(n_in, hidden_size)
        self.wh = nn.Linear(hidden_size, hidden_size)

    def aggregate(self, x):
        return self.wx(x)

    def combine(self, a, h_):
        return super().combine(a, self.wh(h_))

    def forward(self, a, h_):
        a = self.aggregate(a)
        return self.combine(a, h_)


class RNN(nn.Module):
    def __init__(self, vocab_size: int, n_in: int, hidden_size: int):
        super().__init__()

        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        # TODO: n_in 128
        self.embedding = nn.Embedding(vocab_size, 128, padding_idx=0)

        self.rnn = RNNLayer(128, hidden_size, Tanh())
        self.clf = nn.Linear(hidden_size, 2)

    def forward(self, batch):
        h = torch.zeros((len(batch), self.hidden_size))
        h = h.to(batch.get_device())

        # b x wl x d
        emb = self.embedding(batch)
        for j in range(batch.size(1)):
            h = self.rnn(emb[:, j, :], h)

        return self.clf(h)


def train(
    vocab_size: int = 3000,
    vector_size: int = 512,
    seed: int = 42,
    batch_size: int = 1024,
    n_epochs=30,
    device="cpu",
):
    train_dataset, test_dataset = load_imdb_dataset(
        seed=seed, vocab_size=vocab_size, vector_size=vector_size
    )

    train_data = DataLoader(
        train_dataset, batch_size=batch_size, drop_last=True, shuffle=True
    )
    test_data = DataLoader(
        test_dataset, batch_size=batch_size, drop_last=True, shuffle=True
    )

    device = torch.device(device)

    model = RNN(vocab_size, 128, 1024)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.005)

    model.to(device)

    for _ in range(n_epochs):
        with tqdm(train_data) as t:
            for x, y in t:
                x = x.to(device)
                y = y.to(device)
                optimizer.zero_grad()
                y_ = model(x)
                loss = loss_fn(y_, y)
                loss.backward()
                optimizer.step()

                accuracy = (y_.argmax(1) == y).float().mean()

                t.set_postfix({"Loss": loss.item(), "Accuracy": accuracy.item()})


if __name__ == "__main__":
    train(device="cuda" if torch.cuda.is_available() else "cpu")
