import numpy as np
import pytest

from elliptic import evaluate, models


def test_best_threshold_separates_perfect_scores():
    y = np.array([0, 0, 0, 1, 1])
    s = np.array([0.1, 0.2, 0.3, 0.8, 0.9])
    t = evaluate.best_f1_threshold(y, s)
    assert 0.3 < t <= 0.8
    assert evaluate.metrics(y, s, t)["f1"] == 1.0


def test_per_step_handles_steps_without_illicit():
    y = np.array([0, 0, 1, 0])
    s = np.array([0.9, 0.1, 0.9, 0.1])
    steps = np.array([1, 1, 2, 2])
    df = evaluate.per_step(y, s, steps)
    assert np.isnan(df.loc[0, "f1"])
    assert df.loc[1, "f1"] == 1.0


def test_budget_per_step():
    y = np.array([1, 0, 0, 1, 1, 0])
    s = np.array([0.9, 0.8, 0.1, 0.2, 0.7, 0.9])
    steps = np.array([1, 1, 1, 2, 2, 2])
    out = evaluate.budget_per_step(y, s, steps, k=1)
    # Step 1 top-1 is illicit; step 2 top-1 is licit -> 1 of 3 illicit caught.
    assert out["recall_at_k"] == pytest.approx(1 / 3)
    assert out["precision_at_k"] == pytest.approx(1 / 2)


def test_bootstrap_ci_contains_estimate():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 2000)
    s = np.clip(y * 0.6 + rng.normal(0.2, 0.25, 2000), 0, 1)
    steps = np.repeat(np.arange(20), 100)
    ci = evaluate.bootstrap_ci(y, s, steps, evaluate.f1_at(0.5), n_boot=200)
    assert ci["ci_low"] <= ci["estimate"] <= ci["ci_high"]


def test_paired_bootstrap_detects_a_clearly_better_model():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 2000)
    good = np.clip(y * 0.8 + rng.normal(0.1, 0.1, 2000), 0, 1)
    bad = rng.random(2000)
    steps = np.repeat(np.arange(20), 100)
    out = evaluate.paired_bootstrap(y, good, bad, steps, evaluate.f1_at(0.5), n_boot=300)
    assert out["difference"] > 0.3
    assert out["p_value"] < 0.01


@pytest.mark.parametrize("name", models.SUPERVISED)
@pytest.mark.parametrize("imbalance", models.IMBALANCE)
def test_every_model_fits_and_predicts(name, imbalance):
    rng = np.random.default_rng(0)
    X = rng.normal(size=(300, 6)).astype(np.float32)
    y = (X[:, 0] + 0.3 * rng.normal(size=300) > 1.0).astype(int)
    params = {"epochs": 3} if name == "MLP" else {}
    if name in ("RandomForest", "XGBoost"):
        params["n_estimators"] = 20
    m = models.make_model(name, imbalance, seed=0, y_train=y, **params).fit(X, y)
    p = m.predict_proba(X)[:, 1]
    assert p.shape == (300,)
    assert ((0 <= p) & (p <= 1)).all()
