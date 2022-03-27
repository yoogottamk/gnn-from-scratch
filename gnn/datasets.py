import random
from pathlib import Path
from typing import List

import numpy as np
import torch
from keras.datasets import imdb
from torch.nn.functional import one_hot

from gnn.config import DATA_ROOT


def load_citeseer_dataset(
    dataset_dir: Path = DATA_ROOT / "citeseer", train_ratio: float = 1.0
):
    paper_ids: List[str] = []
    word_attributes = []
    class_labels = []

    content_lines = (dataset_dir / "citeseer.content").read_text().splitlines()
    for line in content_lines:
        items = line.split()
        paper_ids.append(items[0])
        word_attributes.append(list(map(int, items[1:-1])))
        class_labels.append(items[-1])

    paper_id2idx = {paper_id: idx for idx, paper_id in enumerate(paper_ids)}
    class_label2idx = {
        class_label: idx for idx, class_label in enumerate(set(class_labels))
    }

    class_idx = [class_label2idx[class_label] for class_label in class_labels]

    # according to the paper, A' = A + I_N is used
    # adj_matrix = np.eye(len(paper_ids), dtype=int)
    adj_matrix = np.zeros((len(paper_ids), len(paper_ids)), dtype=int)

    edge_lines = (dataset_dir / "citeseer.cites").read_text().splitlines()
    for line in edge_lines:
        try:
            u, v = line.split()
            adj_matrix[paper_id2idx[u], paper_id2idx[v]] = 1
            adj_matrix[paper_id2idx[v], paper_id2idx[u]] = 1
        except KeyError:
            # ignore unknown nodes
            pass

    N = len(paper_ids)
    n_train = int(N * train_ratio)
    n_test = N - n_train

    train_test_mask = torch.ones(N)

    for test_idx in random.sample(range(N), n_test):
        train_test_mask[test_idx] = 0

    return (
        torch.Tensor(adj_matrix),
        torch.Tensor(word_attributes),
        one_hot(torch.Tensor(class_idx).to(torch.int64)),
        class_label2idx,
        paper_ids,
        paper_id2idx,
        train_test_mask,
    )


def load_imdb_dataset(
    dataset_path: Path = DATA_ROOT / "imdb.npz",
    seed: int = 42,
    vocab_size: int = 5000,
    vector_size: int = 1024,
):
    (x_train_, y_train), (x_test_, y_test) = imdb.load_data(
        str(dataset_path), num_words=vocab_size, seed=seed
    )

    x_train = []
    x_test = []

    # 0 is the padding character
    for xt in x_train_:
        x_train.append(xt[:vector_size] + ([0] * (vector_size - len(xt))))
    for xt in x_test_:
        x_test.append(xt[:vector_size] + ([0] * (vector_size - len(xt))))

    return (torch.IntTensor(x_train), torch.Tensor(y_train)), (
        torch.IntTensor(x_test),
        torch.Tensor(y_test),
    )
