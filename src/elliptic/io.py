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


def save_table(df: pd.DataFrame, name: str, folder: Path = config.TABLES, **latex_kw) -> None:
    """Write a table as CSV (for reading), Markdown (README) and LaTeX (report)."""
    folder.mkdir(parents=True, exist_ok=True)
    df.to_csv(folder / f"{name}.csv", index=False)
    (folder / f"{name}.md").write_text(df.to_markdown(index=False, floatfmt=".3f") + "\n")
    latex_kw.setdefault("index", False)
    latex_kw.setdefault("float_format", "%.3f")
    (folder / f"{name}.tex").write_text(df.to_latex(**latex_kw))
