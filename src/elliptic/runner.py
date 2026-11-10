"""Shared helpers for the experiment scripts."""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from elliptic import config, evaluate, models, split


@dataclass
class XY:
    X: np.ndarray
    y: np.ndarray
    steps: np.ndarray


def xy(table: pd.DataFrame, cols: list[str], mask: np.ndarray) -> XY:
    sub = table.loc[mask]
    return XY(
        X=sub[cols].to_numpy(np.float32),
        y=sub["label"].to_numpy(np.int64),
        steps=sub["time_step"].to_numpy(np.int64),
    )


@dataclass
class RunResult:
    name: str
    score: np.ndarray  # illicit probability on the test rows
    threshold: float
    seconds: float
    extra: dict = field(default_factory=dict)


def fit_predict(model, train: XY, test: XY) -> np.ndarray:
    model.fit(train.X, train.y)
    return model.predict_proba(test.X)[:, 1]


def oof_threshold(
    table: pd.DataFrame,
    cols: list[str],
    build,
    n_folds: int = 5,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Pick the F1-optimal threshold from out-of-fold predictions.

    ``build(y_train)`` returns a fresh model. Each forward fold trains on the
    steps before its validation block, so every validation score is a genuine
    "future" prediction, just like on the test set.
    """
    ys, ss = [], []
    for f in split.forward_folds(table, n_folds=n_folds):
        tr, va = xy(table, cols, f.train), xy(table, cols, f.test)
        ss.append(fit_predict(build(tr.y), tr, va))
        ys.append(va.y)
    y, s = np.concatenate(ys), np.concatenate(ss)
    return evaluate.best_f1_threshold(y, s), y, s


def run_model(
    table: pd.DataFrame,
    cols: list[str],
    name: str,
    imbalance: str = "none",
    seed: int = config.DEFAULT_SEED,
    tune_threshold: bool = False,
    params: dict | None = None,
) -> RunResult:
    """Train on steps 1-34 and score steps 35-49, optionally with a tuned threshold."""
    params = params or {}
    s = split.temporal_split(table)
    train, test = xy(table, cols, s.train), xy(table, cols, s.test)

    def build(y):
        return models.make_model(name, imbalance, seed=seed, y_train=y, **dict(params))

    t0 = time.perf_counter()
    threshold = 0.5
    if tune_threshold:
        threshold, _, _ = oof_threshold(table, cols, build)
    score = fit_predict(build(train.y), train, test)
    return RunResult(name, score, threshold, time.perf_counter() - t0)


def tune(
    table: pd.DataFrame,
    cols: list[str],
    name: str,
    grid: list[dict],
    imbalance: str = "none",
    seed: int = config.DEFAULT_SEED,
) -> tuple[dict, pd.DataFrame]:
    """Grid search over forward folds, ranked by mean validation PR-AUC.

    PR-AUC is threshold-free, so tuning does not depend on the cut-off we pick
    afterwards. Test steps are never used.
    """
    folds = list(split.forward_folds(table))
    rows = []
    for params in grid:
        aps = []
        for f in folds:
            tr, va = xy(table, cols, f.train), xy(table, cols, f.test)
            m = models.make_model(name, imbalance, seed=seed, y_train=tr.y, **dict(params))
            aps.append(evaluate.pr_auc(va.y, fit_predict(m, tr, va)))
        rows.append({"params": params, "pr_auc_mean": np.mean(aps), "pr_auc_sd": np.std(aps)})
    cv = pd.DataFrame(rows).sort_values("pr_auc_mean", ascending=False, ignore_index=True)
    return dict(cv.loc[0, "params"]), cv


def test_xy(table: pd.DataFrame, cols: list[str]) -> XY:
    return xy(table, cols, split.temporal_split(table).test)
