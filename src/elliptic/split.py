"""Time-aware splits.

A random split would let the model train on transactions from the same two
weeks it is tested on, and from the future. Every split here is by time step:

* ``temporal_split``: the paper's split, steps 1-34 train and 35-49 test.
* ``forward_folds``: expanding-window folds inside the training period, used
  for hyper-parameter tuning and for choosing the decision threshold. The test
  steps are never touched by either.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from elliptic import config


@dataclass(frozen=True)
class Split:
    train: np.ndarray  # boolean mask over rows
    test: np.ndarray

    def __post_init__(self) -> None:
        if (self.train & self.test).any():
            raise ValueError("Train and test overlap")


def _labelled(nodes: pd.DataFrame) -> np.ndarray:
    return (nodes["label"] != config.UNKNOWN).to_numpy()


def steps_mask(nodes: pd.DataFrame, steps: Sequence[int], labelled_only: bool = True) -> np.ndarray:
    mask = nodes["time_step"].isin(list(steps)).to_numpy()
    return mask & _labelled(nodes) if labelled_only else mask


def temporal_split(
    nodes: pd.DataFrame,
    train_steps: Sequence[int] = config.TRAIN_STEPS,
    test_steps: Sequence[int] = config.TEST_STEPS,
    labelled_only: bool = True,
) -> Split:
    if max(train_steps) >= min(test_steps):
        raise ValueError("Training steps must all come before the test steps")
    return Split(
        train=steps_mask(nodes, train_steps, labelled_only),
        test=steps_mask(nodes, test_steps, labelled_only),
    )


def forward_folds(
    nodes: pd.DataFrame,
    train_steps: Sequence[int] = config.TRAIN_STEPS,
    n_folds: int = 5,
    val_size: int = 3,
) -> Iterator[Split]:
    """Expanding-window folds over the training steps.

    With the defaults the validation blocks are steps 20-22, 23-25, 26-28,
    29-31 and 32-34; each fold trains on every labelled step before its block.
    """
    steps = sorted(train_steps)
    last = len(steps)
    for k in range(n_folds, 0, -1):
        val_end = last - (k - 1) * val_size
        val_start = val_end - val_size
        if val_start < 1:
            raise ValueError("Not enough training steps for the requested folds")
        yield temporal_split(nodes, steps[:val_start], steps[val_start:val_end])


def fold_blocks(n_folds: int = 5, val_size: int = 3) -> list[tuple[int, int]]:
    """Human-readable (first, last) validation step of each forward fold."""
    steps = sorted(config.TRAIN_STEPS)
    out = []
    for k in range(n_folds, 0, -1):
        end = len(steps) - (k - 1) * val_size
        out.append((steps[end - val_size], steps[end - 1]))
    return out
