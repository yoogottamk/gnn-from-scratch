#!/bin/bash

CURRENT_DIR="$( dirname "$( realpath "$0" )" )"
DATA_ROOT=${DATA_ROOT:-$CURRENT_DIR/../datasets}

source "$CURRENT_DIR/../.env"

[[ -d "$DATA_ROOT" ]] || mkdir -p "$DATA_ROOT"

[[ -f "$DATA_ROOT/citeseer.tgz" ]] || wget -qO- https://linqs-data.soe.ucsc.edu/public/lbc/citeseer.tgz | tar xvz -C "$DATA_ROOT"

python -c 'from gnn.datasets import load_imdb_dataset; tr, te = load_imdb_dataset(); list(tr); list(te)'
