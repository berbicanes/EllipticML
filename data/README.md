# Data

The Elliptic Data Set is not redistributed in this repository (Kaggle terms).
Download it from <https://www.kaggle.com/datasets/ellipticco/elliptic-data-set>
and place the three CSV files in `data/raw/`:

```
data/raw/elliptic_txs_features.csv   203,769 x 167 (no header)
data/raw/elliptic_txs_classes.csv    203,769 x 2
data/raw/elliptic_txs_edgelist.csv   234,355 x 2
```

With the Kaggle CLI:

```bash
kaggle datasets download -d ellipticco/elliptic-data-set -p data/raw --unzip
```

Symlinks work too. `make data` checks that the files are present and builds the
cached parquet file in `data/processed/`.
