# Assignment 2

## Setup
1. setup virtualenv
2. `pip install -r requirements.txt`
3. set `PYTHONPATH`, `source .env`
4. Download dataset: `./scripts/download-dataset.sh`

## Report

### q1
Base class is present in `gnn/base.py`.

Any more code conflicted with atleast one of the models, nothing more could be extracted into the base class.

### q2
```sh
python -m gnn.gcn
```

|norm type|loss(&darr;)|accuracy (&uarr;)|
|---|---|---|
|-| 1.29909 | 0.741546 |
|row| 1.29420 | 0.748792 |
|col| 1.29483 | 0.745169 |
|symmetric| **1.29028** | **0.756039** |

#### Explanation behind difference in results
The original paper explains the need for normalization by using eigenvalues. Without normalization, the eigenvalues would be in the range [0, 2] which would lead to exploding gradients upon repeated application. To remove this problem, they modified the Adjacency Matrix by adding I<sub>n</sub> and calculated D' as a diagonal matrix with sum of degrees for that particular node's row/column.

Both row and column normalization remove this problem but they are a little different from each other.

Row normalization effectively takes the mean values of neighbors. Column normalization sums neighbours values while taking into account how many neighbors they have. Intuitively, both of them are incorporating more graph related information compared to no normalization so I expect the performance to be a little better.

Symmetric normalization combines both row and column normalization so it takes into account both the node's neighbors and their neighbors as well.

**References:**

1. [Semi-supervised Classification With Graph Convolutional Networks](https://arxiv.org/pdf/1609.02907.pdf)
2. [Interpretation of Symmetric Normalised Graph Adjacency Matrix?](https://math.stackexchange.com/q/3035968)

### q3
```sh
python -m gnn.gin
```

GIN was chosen because it was the most closest to vanilla GCN in terms of formulation. Theoretically, GIN is supposed to perform better due to their better discriminative power based on the subgraph isomorphisms.

|model|loss(&darr;)|accuracy(&uarr;)|
|---|---|---|
|GCN| 1.29028 | **0.756039** |
|GIN| **1.28577** |0.751208|

GIN manages to do better on loss but loses on accuracy.

#### Experimenting with model

|layers|loss(&darr;)|accuracy(&uarr;)|
|---|---|---|
|2|1.28577|0.751208|
|3|1.30319|0.737923|

Here, the deeper model performs worse compared to the original model. This *might* be due to over-smoothing, although it seems unlikely because such a small change (2 -> 3) shouldn't cross the threshold (assuming citeseer dataset doesn't connect ALL papers within 2-3 hops).

### q4

```sh
python -m gnn.rnn
```

Trained for 50 epochs. Other parameters are described in the function definition.

Train Loss: 0.482  
Train Accuracy: 0.778

Test Accuracy: 0.672
