"""E3 - Handling the class imbalance.

Three training strategies (none, class weights, SMOTE) crossed with two ways
to turn scores into decisions (fixed 0.5 threshold, threshold tuned on the
out-of-fold validation predictions), for every supervised model, five seeds.

What we want to learn: do re-weighting and synthetic oversampling make the
model *rank* transactions better (PR-AUC), or do they only move the decision
boundary, which threshold tuning does for free?
"""

from __future__ import annotations

import ast
import sys
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from elliptic import config, data, evaluate, io, models, plots, runner

warnings.filterwarnings("ignore", category=UserWarning)


def figure(df: pd.DataFrame) -> None:
    plots.setup()
    names = list(models.SUPERVISED)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), sharey=False)
    colors = {"none": plots.UNKNOWN, "weight": plots.LICIT, "smote": plots.ILLICIT}
    x = np.arange(len(names))
    for i, strat in enumerate(models.IMBALANCE):
        sub = df[df["Imbalance"] == strat].set_index("Model").loc[names]
        off = (i - 1) * 0.27
        axes[0].bar(x + off, sub["F1 @0.5"], 0.27, color=colors[strat], label=strat)
        axes[0].scatter(x + off, sub["F1 @tuned"], marker="_", s=250, color="black", zorder=3)
        axes[1].bar(x + off, sub["PR-AUC"], 0.27, color=colors[strat], label=strat)
    axes[0].set_title("Illicit F1 (bars: threshold 0.5, ticks: tuned)")
    axes[1].set_title("PR-AUC (threshold-free)")
    for ax in axes:
        ax.set_xticks(x, [plots.short(n) for n in names])
        ax.set_ylim(0, 1)
    axes[1].legend(title="training", loc="upper left", bbox_to_anchor=(1.0, 1.0))
    plots.save(fig, "e03_imbalance")


def main() -> None:
    if "--figure-only" in sys.argv:
        figure(pd.read_csv(config.TABLES / "e03_imbalance.csv"))
        return
    table = data.load().nodes
    cols = data.feature_columns("AF")
    test = runner.test_xy(table, cols)
    best_params = {
        r["Model"]: ast.literal_eval(r["best_params"])
        for r in io.load_json("e02_models")["summary"]
    }

    rows, runs = [], []
    for name in models.SUPERVISED:
        for strat in models.IMBALANCE:
            per_seed = []
            for seed in config.SEEDS:
                r = runner.run_model(
                    table,
                    cols,
                    name,
                    imbalance=strat,
                    seed=seed,
                    tune_threshold=True,
                    params=best_params[name],
                )
                fixed = evaluate.metrics(test.y, r.score, 0.5)
                tuned = evaluate.metrics(test.y, r.score, r.threshold)
                per_seed.append((fixed, tuned))
                runs.append(
                    {
                        "model": name,
                        "imbalance": strat,
                        "seed": seed,
                        "fixed": fixed,
                        "tuned": tuned,
                    }
                )
            fx = pd.DataFrame([p[0] for p in per_seed])
            tn = pd.DataFrame([p[1] for p in per_seed])
            rows.append(
                {
                    "Model": name,
                    "Imbalance": strat,
                    "P @0.5": fx["precision"].mean(),
                    "R @0.5": fx["recall"].mean(),
                    "F1 @0.5": fx["f1"].mean(),
                    "P @tuned": tn["precision"].mean(),
                    "R @tuned": tn["recall"].mean(),
                    "F1 @tuned": tn["f1"].mean(),
                    "F1 sd": tn["f1"].std(ddof=1),
                    "PR-AUC": tn["pr_auc"].mean(),
                    "Threshold": tn["threshold"].mean(),
                }
            )
            print(
                f"{name:12s} {strat:6s} F1@0.5 {fx['f1'].mean():.3f}  "
                f"F1@tuned {tn['f1'].mean():.3f}  PR-AUC {tn['pr_auc'].mean():.3f}"
            )
    df = pd.DataFrame(rows)
    io.save_table(df, "e03_imbalance")
    io.save_json({"summary": rows, "runs": runs}, "e03_imbalance")
    figure(df)
    print(df.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
