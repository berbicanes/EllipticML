"""Model factory.

Every model is built the same way: ``make_model(name, imbalance, seed, **params)``
returns an object with ``fit(X, y)`` and ``predict_proba(X)``. Scaling and
SMOTE live *inside* the returned pipeline, so they are re-fitted on each
training fold and can never see validation or test rows.

Imbalance strategies:
    none    train on the data as it is
    weight  re-weight the loss so both classes count equally
    smote   oversample illicit rows with SMOTE (training data only)
Threshold tuning is a separate, post-hoc step (see evaluate.best_f1_threshold).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

SUPERVISED = ("LogReg", "RandomForest", "XGBoost", "MLP")
IMBALANCE = ("none", "weight", "smote")


class TorchMLP(ClassifierMixin, BaseEstimator):
    """Small feed-forward network in PyTorch with an sklearn interface.

    sklearn's own MLPClassifier cannot weight classes, which we need for the
    imbalance comparison, hence this wrapper.
    """

    def __init__(
        self,
        hidden: tuple[int, ...] = (50,),
        epochs: int = 200,
        lr: float = 1e-3,
        batch_size: int = 512,
        weight_decay: float = 0.0,
        dropout: float = 0.0,
        pos_weight: float = 1.0,
        seed: int = 0,
    ):
        self.hidden = hidden
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.weight_decay = weight_decay
        self.dropout = dropout
        self.pos_weight = pos_weight
        self.seed = seed

    def _build(self, n_in: int) -> torch.nn.Module:
        layers: list[torch.nn.Module] = []
        prev = n_in
        for h in self.hidden:
            layers += [torch.nn.Linear(prev, h), torch.nn.ReLU()]
            if self.dropout:
                layers.append(torch.nn.Dropout(self.dropout))
            prev = h
        layers.append(torch.nn.Linear(prev, 1))
        return torch.nn.Sequential(*layers)

    def fit(self, X, y):
        torch.manual_seed(self.seed)
        g = torch.Generator().manual_seed(self.seed)
        X_t = torch.as_tensor(np.asarray(X), dtype=torch.float32)
        y_t = torch.as_tensor(np.asarray(y), dtype=torch.float32)
        self.classes_ = np.array([0, 1])
        self.net_ = self._build(X_t.shape[1])
        opt = torch.optim.Adam(self.net_.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(self.pos_weight))
        n = len(X_t)
        self.net_.train()
        for _ in range(self.epochs):
            perm = torch.randperm(n, generator=g)
            for i in range(0, n, self.batch_size):
                idx = perm[i : i + self.batch_size]
                opt.zero_grad()
                loss = loss_fn(self.net_(X_t[idx]).squeeze(1), y_t[idx])
                loss.backward()
                opt.step()
        return self

    @torch.no_grad()
    def predict_proba(self, X):
        self.net_.eval()
        logits = self.net_(torch.as_tensor(np.asarray(X), dtype=torch.float32)).squeeze(1)
        p = torch.sigmoid(logits).numpy()
        return np.column_stack([1 - p, p])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


def _ratio(y: np.ndarray | None) -> float:
    """licit/illicit ratio used for loss weighting (set at fit time by the caller)."""
    if y is None:
        return 1.0
    y = np.asarray(y)
    return float((y == 0).sum() / max((y == 1).sum(), 1))


def make_model(
    name: str,
    imbalance: str = "none",
    seed: int = 0,
    y_train: np.ndarray | None = None,
    **params: Any,
):
    """Build a fresh, unfitted model.

    ``y_train`` is only needed for ``imbalance='weight'`` with XGBoost and the
    MLP, which take the licit/illicit ratio as a number rather than 'balanced'.
    """
    if imbalance not in IMBALANCE:
        raise ValueError(f"imbalance must be one of {IMBALANCE}")
    weight = imbalance == "weight"
    ratio = _ratio(y_train) if weight else 1.0

    if name == "LogReg":
        clf = LogisticRegression(
            max_iter=params.pop("max_iter", 5000),
            class_weight="balanced" if weight else None,
            random_state=seed,
            **params,
        )
        scale = True
    elif name == "RandomForest":
        clf = RandomForestClassifier(
            n_estimators=params.pop("n_estimators", 500),
            class_weight="balanced_subsample" if weight else None,
            n_jobs=-1,
            random_state=seed,
            **params,
        )
        scale = False
    elif name == "XGBoost":
        clf = XGBClassifier(
            n_estimators=params.pop("n_estimators", 600),
            learning_rate=params.pop("learning_rate", 0.05),
            max_depth=params.pop("max_depth", 6),
            subsample=params.pop("subsample", 0.8),
            colsample_bytree=params.pop("colsample_bytree", 0.5),
            scale_pos_weight=ratio,
            tree_method="hist",
            n_jobs=-1,
            random_state=seed,
            eval_metric="aucpr",
            **params,
        )
        scale = False
    elif name == "MLP":
        clf = TorchMLP(pos_weight=ratio, seed=seed, **params)
        scale = True
    else:
        raise ValueError(f"Unknown model {name!r}")

    steps: list[tuple[str, Any]] = []
    if scale:
        steps.append(("scale", StandardScaler()))
    if imbalance == "smote":
        steps.append(("smote", SMOTE(random_state=seed)))
        steps.append(("clf", clf))
        return ImbPipeline(steps)
    steps.append(("clf", clf))
    return Pipeline(steps)


def make_isolation_forest(seed: int = 0, **params: Any) -> IsolationForest:
    return IsolationForest(
        n_estimators=params.pop("n_estimators", 500),
        max_samples=params.pop("max_samples", 4096),
        contamination="auto",
        n_jobs=-1,
        random_state=seed,
        **params,
    )


def anomaly_score(model: IsolationForest, X) -> np.ndarray:
    """Higher = more anomalous (sklearn's score_samples is the opposite)."""
    return -model.score_samples(X)
