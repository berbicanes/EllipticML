"""Writing results so that every number in the report comes from a file."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from elliptic import config


def _default(o: Any) -> Any:
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"Cannot serialise {type(o)}")


def save_json(obj: Any, name: str, folder: Path = config.RESULTS) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{name}.json"
    path.write_text(json.dumps(obj, indent=2, default=_default) + "\n")
    return path


def load_json(name: str, folder: Path = config.RESULTS) -> Any:
    return json.loads((folder / f"{name}.json").read_text())


def save_scores(scores: dict[str, np.ndarray], test_rows: pd.DataFrame, name: str) -> Path:
    """Store test-set scores (one column per run) so later analyses reuse them."""
    folder = config.RESULTS / "scores"
    folder.mkdir(parents=True, exist_ok=True)
    df = test_rows[["node", "time_step", "label"]].reset_index(drop=True).copy()
    for k, v in scores.items():
        df[k] = np.asarray(v, dtype=np.float32)
    path = folder / f"{name}.parquet"
    df.to_parquet(path, index=False)
    return path


def load_scores(name: str) -> pd.DataFrame:
    return pd.read_parquet(config.RESULTS / "scores" / f"{name}.parquet")


def save_table(df: pd.DataFrame, name: str, folder: Path = config.TABLES, **latex_kw) -> None:
    """Write a table as CSV (for reading), Markdown (README) and LaTeX (report)."""
    folder.mkdir(parents=True, exist_ok=True)
    df.to_csv(folder / f"{name}.csv", index=False)
    (folder / f"{name}.md").write_text(df.to_markdown(index=False, floatfmt=".3f") + "\n")
    latex_kw.setdefault("index", False)
    latex_kw.setdefault("float_format", "%.3f")
    (folder / f"{name}.tex").write_text(df.to_latex(**latex_kw))
