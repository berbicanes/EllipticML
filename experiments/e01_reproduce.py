"""E1 - Reproduce the baselines of Weber et al. (2019), Table 1.

Same split (train 1-34, test 35-49), same feature sets (LF = time step + 93
local features, AF = LF + 72 aggregated), same models and hyper-parameters as
far as the paper states them, decision threshold 0.5.

    Logistic Regression  scikit-learn defaults of 2019 (liblinear solver, L2,
                         C = 1, no feature scaling)
    Random Forest        50 trees, max_features = 50
    MLP                  one hidden layer of 50, Adam, lr 0.001, 200 epochs

Random Forest and the MLP are run with five seeds; we report mean and standard
deviation, so a gap to the paper can be told apart from seed noise.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from elliptic import config, data, evaluate, io, plots, runner

PAPER = {  # Weber et al. (2019), Table 1, illicit class
    ("LogReg", "AF"): (0.404, 0.593, 0.481, 0.931),
    ("LogReg", "LF"): (0.348, 0.668, 0.457, 0.920),
    ("RandomForest", "AF"): (0.956, 0.670, 0.788, 0.977),
    ("RandomForest", "LF"): (0.803, 0.611, 0.694, 0.966),
    ("MLP", "AF"): (0.694, 0.617, 0.653, 0.962),
    ("MLP", "LF"): (0.637, 0.662, 0.649, 0.958),
}

PAPER_PARAMS = {
    "LogReg": {"solver": "liblinear", "scale": False},
    "RandomForest": {"n_estimators": 50, "max_features": 50},
    "MLP": {"hidden": (50,), "epochs": 200, "lr": 1e-3, "batch_size": 512},
}


def figure(df: pd.DataFrame) -> None:
    plots.setup()
    fig, ax = plt.subplots(figsize=(7.5, 3.4))
    labels = [f"{plots.short(m)}\n{f}" for m, f in zip(df["Model"], df["Features"], strict=True)]
    x = np.arange(len(df))
    ax.bar(x - 0.2, df["Paper F1"], 0.4, color=plots.UNKNOWN, label="Weber et al. (2019)")
    ax.bar(
        x + 0.2,
        df["F1"],
        0.4,
        yerr=df["F1 sd"].fillna(0),
        color=plots.LICIT,
        capsize=3,
        label="this reproduction (mean of 5 seeds)",
    )
    ax.set_xticks(x, labels)
    ax.set_ylabel("illicit F1 (test steps 35-49)")
    ax.set_ylim(0, 1)
    ax.set_title("Reproducing the published baselines")
    ax.legend(loc="upper left")
    plots.save(fig, "e01_reproduction")


def main() -> None:
    table = data.load().nodes
    rows, records = [], []
    for model, params in PAPER_PARAMS.items():
        seeds = [config.DEFAULT_SEED] if model == "LogReg" else list(config.SEEDS)
        for fs in ("AF", "LF"):
            cols = data.feature_columns(fs, include_time_step=True)
            test = runner.test_xy(table, cols)
            per_seed = []
            for seed in seeds:
                r = runner.run_model(table, cols, model, seed=seed, params=params)
                m = evaluate.metrics(test.y, r.score, 0.5)
                per_seed.append(m)
                records.append({"model": model, "features": fs, "seed": seed, **m})
                print(f"{model:12s} {fs} seed={seed} F1={m['f1']:.3f} ({r.seconds:.0f}s)")
            df = pd.DataFrame(per_seed)
            p = PAPER[(model, fs)]
            rows.append(
                {
                    "Model": model,
                    "Features": fs,
                    "Precision": df["precision"].mean(),
                    "Recall": df["recall"].mean(),
                    "F1": df["f1"].mean(),
                    "F1 sd": df["f1"].std(ddof=1) if len(df) > 1 else np.nan,
                    "Micro F1": df["micro_f1"].mean(),
                    "Paper P": p[0],
                    "Paper R": p[1],
                    "Paper F1": p[2],
                    "Paper micro F1": p[3],
                    "F1 gap": df["f1"].mean() - p[2],
                }
            )
    table_out = pd.DataFrame(rows)
    io.save_table(table_out, "e01_reproduction")
    figure(table_out)
    io.save_json({"summary": rows, "runs": records}, "e01_reproduce")
    print(table_out.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
