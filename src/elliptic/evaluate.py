"""Metrics, per-time-step scores and uncertainty estimates.

Conventions: ``y`` is 1 for illicit and 0 for licit, ``score`` is a model's
illicit probability (or any score where higher means more suspicious), and
``steps`` is the time step of each row.

Why a *block* bootstrap: transactions from the same two-week step share a
market regime and are linked in the graph, so they are not independent draws.
Resampling whole time steps keeps that dependence and gives honest (wider)
confidence intervals than resampling rows.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)


def best_f1_threshold(y: np.ndarray, score: np.ndarray) -> float:
    """Threshold that maximises illicit F1 on (y, score). Use on validation data only."""
    p, r, t = precision_recall_curve(y, score)
    f1 = np.divide(2 * p * r, p + r, out=np.zeros_like(p), where=(p + r) > 0)
    # precision_recall_curve returns one more (p, r) pair than thresholds.
    return float(t[np.argmax(f1[:-1])])


def metrics(y: np.ndarray, score: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
    pred = (score >= threshold).astype(int)
    return {
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred, zero_division=0),
        "f1": f1_score(y, pred, zero_division=0),
        "micro_f1": f1_score(y, pred, average="micro"),
        "pr_auc": average_precision_score(y, score),
        "roc_auc": roc_auc_score(y, score),
        "threshold": float(threshold),
        "n_predicted_illicit": int(pred.sum()),
    }


def per_step(
    y: np.ndarray, score: np.ndarray, steps: np.ndarray, threshold: float = 0.5
) -> pd.DataFrame:
    """Illicit precision/recall/F1 for every time step separately."""
    rows = []
    for s in np.unique(steps):
        m = steps == s
        ys, ps = y[m], (score[m] >= threshold).astype(int)
        rows.append(
            {
                "time_step": int(s),
                "n": int(m.sum()),
                "n_illicit": int(ys.sum()),
                "precision": precision_score(ys, ps, zero_division=0),
                "recall": recall_score(ys, ps, zero_division=0) if ys.sum() else np.nan,
                "f1": f1_score(ys, ps, zero_division=0) if ys.sum() else np.nan,
            }
        )
    return pd.DataFrame(rows)


def budget_per_step(
    y: np.ndarray, score: np.ndarray, steps: np.ndarray, k: int = 100
) -> dict[str, float]:
    """An analyst reviews the k highest-scored transactions of every time step.

    Returns the share of all test illicit transactions caught that way and the
    share of reviewed transactions that were illicit.
    """
    caught = reviewed = 0
    for s in np.unique(steps):
        m = np.flatnonzero(steps == s)
        top = m[np.argsort(-score[m], kind="stable")[:k]]
        caught += int(y[top].sum())
        reviewed += len(top)
    return {
        "k": k,
        "recall_at_k": caught / max(int(y.sum()), 1),
        "precision_at_k": caught / reviewed,
    }


def _block_indices(steps: np.ndarray, rng: np.random.Generator, n_boot: int):
    uniq = np.unique(steps)
    groups = {s: np.flatnonzero(steps == s) for s in uniq}
    for _ in range(n_boot):
        draw = rng.choice(uniq, size=len(uniq), replace=True)
        yield np.concatenate([groups[s] for s in draw])


Metric = Callable[[np.ndarray, np.ndarray], float]


def f1_at(threshold: float) -> Metric:
    return lambda y, s: f1_score(y, (s >= threshold).astype(int), zero_division=0)


def recall_at(threshold: float) -> Metric:
    return lambda y, s: recall_score(y, (s >= threshold).astype(int), zero_division=0)


def precision_at(threshold: float) -> Metric:
    return lambda y, s: precision_score(y, (s >= threshold).astype(int), zero_division=0)


def pr_auc(y: np.ndarray, s: np.ndarray) -> float:
    return average_precision_score(y, s) if y.any() else np.nan


def bootstrap_ci(
    y: np.ndarray,
    score: np.ndarray,
    steps: np.ndarray,
    metric: Metric,
    n_boot: int = 1000,
    seed: int = 0,
    alpha: float = 0.05,
) -> dict[str, float]:
    """Point estimate and percentile CI from a block bootstrap over time steps."""
    rng = np.random.default_rng(seed)
    vals = np.array([metric(y[i], score[i]) for i in _block_indices(steps, rng, n_boot)])
    vals = vals[~np.isnan(vals)]
    lo, hi = np.quantile(vals, [alpha / 2, 1 - alpha / 2])
    return {"estimate": float(metric(y, score)), "ci_low": float(lo), "ci_high": float(hi)}


def paired_bootstrap(
    y: np.ndarray,
    score_a: np.ndarray,
    score_b: np.ndarray,
    steps: np.ndarray,
    metric_a: Metric,
    metric_b: Metric | None = None,
    n_boot: int = 2000,
    seed: int = 0,
) -> dict[str, float]:
    """Is model A better than model B on the same test rows?

    Both models are scored on the *same* resampled time steps, so the shared
    difficulty of a step cancels out. The two-sided p-value is the share of
    resamples whose difference has the opposite sign to the observed one, times 2.
    """
    metric_b = metric_b or metric_a
    rng = np.random.default_rng(seed)
    diffs = np.array(
        [
            metric_a(y[i], score_a[i]) - metric_b(y[i], score_b[i])
            for i in _block_indices(steps, rng, n_boot)
        ]
    )
    observed = metric_a(y, score_a) - metric_b(y, score_b)
    if observed >= 0:
        p = 2 * np.mean(diffs <= 0)
    else:
        p = 2 * np.mean(diffs >= 0)
    lo, hi = np.quantile(diffs, [0.025, 0.975])
    return {
        "difference": float(observed),
        "ci_low": float(lo),
        "ci_high": float(hi),
        "p_value": float(min(p, 1.0)),
    }
