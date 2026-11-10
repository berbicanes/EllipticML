"""E2 - Main model comparison with a fair protocol.

Differences from the reproduction (E1), each one closing a gap in the
original protocol:

* the time step is not a feature (all test steps lie outside the training range);
* hyper-parameters are tuned on expanding-window folds inside steps 1-34;
* the decision threshold is chosen on out-of-fold predictions, not fixed at 0.5;
* every model runs with five seeds;
* results carry 95% block-bootstrap confidence intervals over test time steps,
  and the best model is compared with every other in a paired bootstrap.

Feature set: AF (93 local + 72 aggregated). Imbalance handling: none here;
E3 studies it separately.
"""

from __future__ import annotations

import itertools
import warnings

import numpy as np
import pandas as pd

from elliptic import config, data, evaluate, io, runner, split

warnings.filterwarnings("ignore", category=UserWarning)


def grid(**axes) -> list[dict]:
    keys = list(axes)
    return [dict(zip(keys, vals, strict=True)) for vals in itertools.product(*axes.values())]


GRIDS = {
    "LogReg": grid(C=[0.01, 0.1, 1.0, 10.0]),
    "RandomForest": grid(
        n_estimators=[500], max_features=["sqrt", 0.3, 50], min_samples_leaf=[1, 3]
    ),
    "XGBoost": grid(
        n_estimators=[600], max_depth=[4, 6, 8], learning_rate=[0.03, 0.1], colsample_bytree=[0.5]
    ),
    "MLP": grid(
        hidden=[(50,), (128, 64)], dropout=[0.0, 0.2], weight_decay=[0.0, 1e-4], epochs=[100]
    ),
}


def main() -> None:
    table = data.load().nodes
    cols = data.feature_columns("AF")
    test = runner.test_xy(table, cols)
    test_rows = table.loc[split.temporal_split(table).test]

    summary, cv_tables, scores, thresholds, runs = [], {}, {}, {}, []
    for name, g in GRIDS.items():
        best, cv = runner.tune(table, cols, name, g)
        cv_tables[name] = cv.assign(params=cv["params"].astype(str)).to_dict(orient="records")
        print(f"{name}: best {best} (CV PR-AUC {cv.loc[0, 'pr_auc_mean']:.3f})")

        per_seed = []
        for seed in config.SEEDS:
            r = runner.run_model(table, cols, name, seed=seed, tune_threshold=True, params=best)
            m = evaluate.metrics(test.y, r.score, r.threshold)
            m05 = evaluate.metrics(test.y, r.score, 0.5)
            per_seed.append(m)
            runs.append({"model": name, "seed": seed, **m, "f1_at_0.5": m05["f1"]})
            scores[f"{name}_s{seed}"] = r.score
            thresholds[f"{name}_s{seed}"] = r.threshold
            print(
                f"  seed {seed}: F1 {m['f1']:.3f} @ t={r.threshold:.3f}, PR-AUC {m['pr_auc']:.3f}"
            )

        # Confidence intervals on the seed-averaged score (an ensemble of the
        # five seeds), so the CI reflects test-set uncertainty, not seed noise.
        mean_score = np.mean([scores[f"{name}_s{s}"] for s in config.SEEDS], axis=0)
        t_mean = float(np.mean([thresholds[f"{name}_s{s}"] for s in config.SEEDS]))
        scores[f"{name}_mean"] = mean_score
        thresholds[f"{name}_mean"] = t_mean
        ci_f1 = evaluate.bootstrap_ci(test.y, mean_score, test.steps, evaluate.f1_at(t_mean))
        ci_ap = evaluate.bootstrap_ci(test.y, mean_score, test.steps, evaluate.pr_auc)
        budget = evaluate.budget_per_step(test.y, mean_score, test.steps, k=100)
        df = pd.DataFrame(per_seed)
        summary.append(
            {
                "Model": name,
                "Precision": df["precision"].mean(),
                "Recall": df["recall"].mean(),
                "F1": df["f1"].mean(),
                "F1 sd (seeds)": df["f1"].std(ddof=1),
                "F1 95% CI": f"[{ci_f1['ci_low']:.3f}, {ci_f1['ci_high']:.3f}]",
                "PR-AUC": df["pr_auc"].mean(),
                "PR-AUC 95% CI": f"[{ci_ap['ci_low']:.3f}, {ci_ap['ci_high']:.3f}]",
                "ROC-AUC": df["roc_auc"].mean(),
                "Recall@100/step": budget["recall_at_k"],
                "Threshold": t_mean,
                "best_params": str(best),
            }
        )

    summary_df = pd.DataFrame(summary).sort_values("PR-AUC", ascending=False, ignore_index=True)

    # Paired comparison of the best model (by PR-AUC) with each of the others.
    best_name = summary_df.loc[0, "Model"]
    sa = scores[f"{best_name}_mean"]
    paired = []
    for other in summary_df["Model"][1:]:
        sb = scores[f"{other}_mean"]
        f1 = evaluate.paired_bootstrap(
            test.y,
            sa,
            sb,
            test.steps,
            evaluate.f1_at(thresholds[f"{best_name}_mean"]),
            evaluate.f1_at(thresholds[f"{other}_mean"]),
        )
        ap = evaluate.paired_bootstrap(test.y, sa, sb, test.steps, evaluate.pr_auc)
        paired.append(
            {
                "Comparison": f"{best_name} vs {other}",
                "F1 diff": f1["difference"],
                "F1 diff CI": f"[{f1['ci_low']:.3f}, {f1['ci_high']:.3f}]",
                "F1 p": f1["p_value"],
                "PR-AUC diff": ap["difference"],
                "PR-AUC diff CI": f"[{ap['ci_low']:.3f}, {ap['ci_high']:.3f}]",
                "PR-AUC p": ap["p_value"],
            }
        )
    paired_df = pd.DataFrame(paired)

    io.save_scores(scores, test_rows, "e02_models")
    io.save_table(summary_df.drop(columns=["best_params"]), "e02_models")
    io.save_table(paired_df, "e02_paired")
    io.save_json(
        {
            "summary": summary_df.to_dict(orient="records"),
            "paired": paired,
            "thresholds": thresholds,
            "runs": runs,
            "cv": cv_tables,
        },
        "e02_models",
    )
    print(summary_df.drop(columns=["best_params"]).round(3).to_string(index=False))
    print(paired_df.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
