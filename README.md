# Assignment 2

## Setup
1. setup virtualenv
2. `pip install -r requirements.txt`
3. set `PYTHONPATH`, `source .env`
4. Download dataset: `./scripts/download-dataset.sh`

## Report

### q1
Base class is present in `gnn/base.py`.

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

!!! TODO: why diff

### q3
```sh
python -m gnn.gin
```

!!! TODO: compare with GCN

#### Experimenting with model

|layers|loss(&darr;)|accuracy(&uarr;)|
|---|---|---|
|2|1.28577|0.751208|
|3|1.30319|0.737923|

### q4

```sh
python -m gnn.rnn
```
