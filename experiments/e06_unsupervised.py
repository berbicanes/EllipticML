"""E6 - Do illicit transactions stand out without labels?

1. Isolation Forest fitted on *every* transaction of steps 1-34, labelled or
   not (136k rows, no labels used), then scored on the labelled test rows.
   Variants: LF vs AF features. As a middle ground we also fit it on known
   licit rows only (novelty detection: "what does normal look like?"), which
   uses labels for one class.
2. Label efficiency: how many labels does supervised XGBoost need before it
   beats the label-free detector? Train on random fractions of the labelled
   training rows, five seeds each, and compare test PR-AUC.

PR-AUC is the main metric here because an unsupervised score has no natural
threshold. A random scorer has PR-AUC equal to the illicit share of the test
set (about 0.065).
"""

from __future__ import annotations

import ast
import sys
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from elliptic import config, data, evaluate, io, models, plots, runner, split

warnings.filterwarnings("ignore", category=UserWarning)

FRACTIONS = [0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.25, 0.5, 1.0]


def isolation_forests(nodes: pd.DataFrame) -> tuple[list[dict], dict[str, np.ndarray]]:
    s = split.temporal_split(nodes)
    all_train = split.steps_mask(nodes, config.TRAIN_STEPS, labelled_only=False)
    licit_train = s.train & (nodes["label"] == config.LICIT).to_numpy()
    y = nodes.loc[s.test, "label"].to_numpy()
    steps = nodes.loc[s.test, "time_step"].to_numpy()
    rows, scores = [], {}
    for fs in ("LF", "AF"):
        cols = data.feature_columns(fs)
        X = nodes[cols].to_numpy(np.float32)
        for fit_name, fit_mask in (
            ("all rows, no labels", all_train),
            ("licit rows only", licit_train),
        ):
            per_seed = []
            for seed in config.SEEDS:
                m = models.make_isolation_forest(seed=seed).fit(X[fit_mask])
                per_seed.append(models.anomaly_score(m, X[s.test]))
            # Two directions: "illicit = unusual" (the textbook assumption) and
            # its inverse "illicit = typical". Choosing the direction is one bit
            # of prior knowledge; no label is used to fit either.
            for direction, sign in (("anomalous", 1.0), ("typical", -1.0)):
                runs = [sign * p for p in per_seed]
                mean = np.mean(runs, axis=0)
                fit_key = "unsup" if fit_name.startswith("all") else "licit"
                key = f"IF|{fs}|{fit_key}|{direction}"
                scores[key] = mean
                ci = evaluate.bootstrap_ci(y, mean, steps, evaluate.pr_auc)
                budget = evaluate.budget_per_step(y, mean, steps, k=100)
                rows.append(
                    {
                        "Detector": "Isolation Forest",
                        "Features": fs,
                        "Fitted on": fit_name,
                        "Flag as illicit": f"most {direction}",
                        "Rows used": int(fit_mask.sum()),
                        "PR-AUC": float(np.mean([evaluate.pr_auc(y, p) for p in runs])),
                        "PR-AUC 95% CI": f"[{ci['ci_low']:.3f}, {ci['ci_high']:.3f}]",
                        "ROC-AUC": float(
                            np.mean([evaluate.metrics(y, p)["roc_auc"] for p in runs])
                        ),
                        "Recall@100/step": budget["recall_at_k"],
                    }
                )
                print(rows[-1])
    return rows, scores


def label_efficiency(nodes: pd.DataFrame, params: dict) -> pd.DataFrame:
    cols = data.feature_columns("AF")
    s = split.temporal_split(nodes)
    train, test = runner.xy(nodes, cols, s.train), runner.xy(nodes, cols, s.test)
    rows = []
    for frac in FRACTIONS:
        for seed in config.SEEDS:
            rng = np.random.default_rng(seed)
            n = max(int(round(frac * len(train.y))), 2)
            idx = rng.choice(len(train.y), size=n, replace=False)
            # Every subsample must contain both classes; resample if not.
            while train.y[idx].min() == train.y[idx].max():
                idx = rng.choice(len(train.y), size=n, replace=False)
            sub = runner.XY(train.X[idx], train.y[idx], train.steps[idx])
            m = models.make_model("XGBoost", "none", seed=seed, y_train=sub.y, **params)
            score = runner.fit_predict(m, sub, test)
            rows.append(
                {
                    "fraction": frac,
                    "n_labels": n,
                    "n_illicit_labels": int(sub.y.sum()),
                    "seed": seed,
                    "pr_auc": evaluate.pr_auc(test.y, score),
                }
            )
        last = [r["pr_auc"] for r in rows[-len(config.SEEDS) :]]
        print(f"fraction {frac:>6.3f} ({n:>5d} labels): PR-AUC {np.mean(last):.3f}")
    return pd.DataFrame(rows)


def fig_label_efficiency(curve: pd.DataFrame, if_rows: list[dict], base_rate: float) -> None:
    plots.setup()
    agg = curve.groupby("n_labels")["pr_auc"].agg(["mean", "std", "min", "max"]).reset_index()
    fig, ax = plt.subplots(figsize=(7.5, 4))
    ax.fill_between(
        agg["n_labels"],
        agg["min"],
        agg["max"],
        color=plots.MODEL_COLORS["XGBoost"],
        alpha=0.2,
        lw=0,
    )
    ax.plot(
        agg["n_labels"],
        agg["mean"],
        "o-",
        color=plots.MODEL_COLORS["XGBoost"],
        label="XGBoost trained on n labels (mean, min-max of 5 seeds)",
    )

    def pick(direction: str) -> dict:
        return next(
            r
            for r in if_rows
            if r["Features"] == "AF"
            and r["Fitted on"].startswith("all")
            and r["Flag as illicit"] == f"most {direction}"
        )

    unsup, typical = pick("anomalous"), pick("typical")
    ax.axhline(
        typical["PR-AUC"],
        color="black",
        ls="-",
        lw=1.2,
        label="Isolation Forest, flag most typical (no labels)",
    )
    ax.axhline(
        unsup["PR-AUC"],
        color="black",
        ls="--",
        lw=1.2,
        label="Isolation Forest, flag most anomalous (no labels)",
    )
    ax.axhline(base_rate, color=plots.MUTED, ls=":", lw=1.2, label="random ranking")
    ax.set_xscale("log")
    ax.set_xlabel("labelled training transactions (log scale)")
    ax.set_ylabel("test PR-AUC")
    ax.set_ylim(0, 1)
    ax.set_title("How many labels beat a label-free detector?")
    ax.legend(loc="lower right", bbox_to_anchor=(1.0, 0.1), fontsize=8)
    plots.save(fig, "e06_label_efficiency")


def fig_score_distribution(nodes: pd.DataFrame, if_scores: dict[str, np.ndarray]) -> None:
    plots.setup()
    s = split.temporal_split(nodes)
    y = nodes.loc[s.test, "label"].to_numpy()
    sc = if_scores["IF|AF|unsup|anomalous"]
    fig, ax = plt.subplots(figsize=(7, 3.4))
    bins = np.linspace(sc.min(), sc.max(), 60)
    ax.hist(sc[y == 0], bins=bins, density=True, alpha=0.6, color=plots.LICIT, label="licit")
    ax.hist(sc[y == 1], bins=bins, density=True, alpha=0.6, color=plots.ILLICIT, label="illicit")
    ax.set_xlabel("Isolation Forest anomaly score (higher = more unusual)")
    ax.set_ylabel("density")
    ax.set_title("Illicit transactions are typical, not anomalous")
    ax.legend()
    plots.save(fig, "e06_anomaly_scores")


def main() -> None:
    if "--figure-only" in sys.argv:
        saved = io.load_json("e06_unsupervised")
        fig_label_efficiency(
            pd.DataFrame(saved["label_efficiency"]),
            saved["isolation_forest"],
            saved["test_base_rate"],
        )
        return
    nodes = data.load().nodes
    s = split.temporal_split(nodes)
    base_rate = float(nodes.loc[s.test, "label"].mean())
    if_rows, if_scores = isolation_forests(nodes)
    params = {
        r["Model"]: ast.literal_eval(r["best_params"])
        for r in io.load_json("e02_models")["summary"]
    }["XGBoost"]
    curve = label_efficiency(nodes, params)

    io.save_scores(if_scores, nodes.loc[s.test], "e06_unsupervised")
    io.save_table(pd.DataFrame(if_rows), "e06_isolation_forest")
    summary = curve.groupby(["fraction", "n_labels"])["pr_auc"].agg(["mean", "std"]).reset_index()
    io.save_table(summary, "e06_label_efficiency")
    io.save_json(
        {
            "isolation_forest": if_rows,
            "label_efficiency": curve.to_dict(orient="records"),
            "test_base_rate": base_rate,
        },
        "e06_unsupervised",
    )
    fig_label_efficiency(curve, if_rows, base_rate)
    fig_score_distribution(nodes, if_scores)


if __name__ == "__main__":
    main()
