"""Loading the Elliptic Data Set into one tidy table.

The features file has no header. Column 0 is the transaction id, column 1 the
time step, columns 2-94 the 93 local features and columns 95-166 the 72
features aggregated over the one-hop neighbourhood. We name them ``local_00``
... ``local_92`` and ``agg_00`` ... ``agg_71`` so that feature groups can be
selected by prefix.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from elliptic import config

LOCAL_COLS = [f"local_{i:02d}" for i in range(config.N_LOCAL)]
AGG_COLS = [f"agg_{i:02d}" for i in range(config.N_AGG)]
FEATURE_COLS = LOCAL_COLS + AGG_COLS

_CLASS_MAP = {"1": config.ILLICIT, "2": config.LICIT, "unknown": config.UNKNOWN}


@dataclass(frozen=True)
class EllipticData:
    """Node table plus edge list.

    ``nodes`` has one row per transaction with columns ``txId``, ``time_step``,
    ``label`` (1 illicit, 0 licit, -1 unknown), ``node`` (0-based row index used
    by the graph code) and the 165 feature columns.
    ``edges`` holds directed edges as 0-based ``src``/``dst`` node indices.
    """

    nodes: pd.DataFrame
    edges: pd.DataFrame

    @property
    def labelled(self) -> pd.DataFrame:
        return self.nodes[self.nodes["label"] != config.UNKNOWN]


def _check_raw(raw_dir: Path) -> None:
    missing = [
        f
        for f in (config.FEATURES_CSV, config.CLASSES_CSV, config.EDGES_CSV)
        if not (raw_dir / f).exists()
    ]
    if missing:
        raise FileNotFoundError(
            f"Missing {missing} in {raw_dir}. See data/README.md for download instructions."
        )


def read_raw(raw_dir: Path = config.DATA_RAW) -> EllipticData:
    """Read the three CSV files and join them on txId."""
    _check_raw(raw_dir)
    names = ["txId", "time_step", *FEATURE_COLS]
    feats = pd.read_csv(
        raw_dir / config.FEATURES_CSV,
        header=None,
        names=names,
        dtype={"txId": np.int64, "time_step": np.int16} | {c: np.float32 for c in FEATURE_COLS},
    )
    classes = pd.read_csv(raw_dir / config.CLASSES_CSV, dtype={"txId": np.int64, "class": str})
    edges = pd.read_csv(raw_dir / config.EDGES_CSV, dtype=np.int64)

    if feats["txId"].duplicated().any():
        raise ValueError("Duplicate txId in features file")
    unknown_classes = set(classes["class"]) - set(_CLASS_MAP)
    if unknown_classes:
        raise ValueError(f"Unexpected class values: {unknown_classes}")

    classes["label"] = classes["class"].map(_CLASS_MAP).astype(np.int8)
    nodes = feats.merge(classes[["txId", "label"]], on="txId", how="left", validate="1:1")
    if nodes["label"].isna().any():
        raise ValueError("Some transactions have no class row")

    nodes = nodes.assign(node=np.arange(len(nodes), dtype=np.int64))
    index = pd.Series(nodes["node"].to_numpy(), index=nodes["txId"].to_numpy())
    src = index.reindex(edges["txId1"].to_numpy()).to_numpy()
    dst = index.reindex(edges["txId2"].to_numpy()).to_numpy()
    if np.isnan(src.astype(float)).any() or np.isnan(dst.astype(float)).any():
        raise ValueError("Edge list references a txId that is not in the features file")
    edge_df = pd.DataFrame({"src": src.astype(np.int64), "dst": dst.astype(np.int64)})

    cols = ["node", "txId", "time_step", "label", *FEATURE_COLS]
    return EllipticData(nodes=nodes[cols], edges=edge_df)


def load(raw_dir: Path = config.DATA_RAW, cache_dir: Path = config.DATA_PROCESSED) -> EllipticData:
    """Load the dataset, using a parquet cache after the first read (~10x faster)."""
    nodes_path = cache_dir / "nodes.parquet"
    edges_path = cache_dir / "edges.parquet"
    if nodes_path.exists() and edges_path.exists():
        return EllipticData(pd.read_parquet(nodes_path), pd.read_parquet(edges_path))
    data = read_raw(raw_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    data.nodes.to_parquet(nodes_path, index=False)
    data.edges.to_parquet(edges_path, index=False)
    return data


def feature_columns(feature_set: str, include_time_step: bool = False) -> list[str]:
    """Column names for a named feature set.

    ``LF``: the 93 local features. ``AF``: local plus the 72 aggregated ones.
    Weber et al. count the time step as a local feature (their "first 94"); we
    exclude it by default because under a temporal split every test step is
    outside the training range, so the column can only mislead the model. Set
    ``include_time_step=True`` to reproduce the paper exactly.
    """
    sets = {"LF": LOCAL_COLS, "AF": FEATURE_COLS}
    if feature_set not in sets:
        raise ValueError(f"Unknown feature set {feature_set!r}; expected one of {list(sets)}")
    cols = list(sets[feature_set])
    return ["time_step", *cols] if include_time_step else cols
