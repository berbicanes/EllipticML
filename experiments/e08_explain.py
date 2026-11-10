"""E8 - What did the model learn?

The 165 features are anonymised, so single-feature stories are impossible.
What we *can* say is which *group* of information the model relies on:

    local       93 features of the transaction itself
    aggregated  72 features summarising its one-hop neighbours
    structure   9 label-free graph statistics (E4's "G" features)

Two complementary views on the XGBoost model trained on AF+G:

1. SHAP (TreeExplainer): mean |SHAP| per feature on the test set, summed per
   group, plus the top individual features.
2. Group permutation importance: shuffle all columns of one group together on
   the test set and measure the PR-AUC lost. Repeated 10 times. Permuting a
   whole group avoids the trap of correlated features hiding each other.

We also compare SHAP group shares before and after the dark market shutdown:
does the model lean on different evidence when it fails?
"""

from __future__ import annotations

import ast
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from elliptic import config, data, evaluate, graph, io, models, plots, runner, split

warnings.filterwarnings("ignore")

GROUP_COLORS = {"local": plots.LICIT, "aggregated": "#7FA7D1", "structure": "#E69F00"}


def group_of(col: str) -> str:
    if col.startswith("local_"):
        return "local"
    if col.startswith("agg_"):
        return "aggregated"
    return "structure"


def main() -> None:
    d = data.load()
    nodes = d.nodes
    table = pd.concat(
        [nodes.reset_index(drop=True), graph.structural_features(d.edges, len(nodes))], axis=1
    )
    cols = data.feature_columns("AF") + graph.GRAPH_COLS
    groups = np.array([group_of(c) for c in cols])
    params = {
        r["Model"]: ast.literal_eval(r["best_params"])
        for r in io.load_json("e02_models")["summary"]
    }["XGBoost"]

    s = split.temporal_split(table)
    train, test = runner.xy(table, cols, s.train), runner.xy(table, cols, s.test)
    model = models.make_model("XGBoost", seed=config.DEFAULT_SEED, **params)
    model.fit(train.X, train.y)
    clf = model.named_steps["clf"]
    base_score = evaluate.pr_auc(test.y, model.predict_proba(test.X)[:, 1])

    # ---- SHAP
    sv = shap.TreeExplainer(clf).shap_values(test.X)
    abs_sv = np.abs(sv)
    per_feature = pd.DataFrame({"feature": cols, "group": groups, "mean_abs_shap": abs_sv.mean(0)})
    per_feature = per_feature.sort_values("mean_abs_shap", ascending=False, ignore_index=True)
    share = per_feature.groupby("group")["mean_abs_shap"].sum()
    share = share / share.sum()

    period = np.where(test.steps < config.DARK_MARKET_STEP, "35-42", "43-49")
    by_period = {}
    for p in ("35-42", "43-49"):
        g = pd.Series(abs_sv[period == p].mean(0), index=cols).groupby(groups).sum()
        by_period[p] = (g / g.sum()).to_dict()

    # ---- group permutation importance
    rng = np.random.default_rng(config.DEFAULT_SEED)
    perm_rows = []
    for gname in ("local", "aggregated", "structure"):
        idx = np.flatnonzero(groups == gname)
        drops = []
        for _ in range(10):
            Xp = test.X.copy()
            Xp[:, idx] = Xp[rng.permutation(len(Xp))][:, idx]
            drops.append(base_score - evaluate.pr_auc(test.y, model.predict_proba(Xp)[:, 1]))
        perm_rows.append(
            {
                "group": gname,
                "n_features": len(idx),
                "pr_auc_drop_mean": float(np.mean(drops)),
                "pr_auc_drop_sd": float(np.std(drops)),
            }
        )
    perm = pd.DataFrame(perm_rows)

    # ---- figures
    plots.setup()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), gridspec_kw={"width_ratios": [1.3, 1]})
    top = per_feature.head(20).iloc[::-1]
    axes[0].barh(
        top["feature"], top["mean_abs_shap"], color=[GROUP_COLORS[g] for g in top["group"]]
    )
    axes[0].set_xlabel("mean |SHAP value| on test steps")
    axes[0].set_title("Top 20 features (XGBoost, AF+G)")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in GROUP_COLORS.values()]
    axes[0].legend(handles, GROUP_COLORS.keys(), loc="lower right")
    x = np.arange(3)
    names = ["local", "aggregated", "structure"]
    axes[1].bar(
        x - 0.2,
        [share[n] for n in names],
        0.4,
        color=[GROUP_COLORS[n] for n in names],
        label="share of total |SHAP|",
    )
    axes[1].bar(
        x + 0.2,
        [perm.set_index("group").loc[n, "pr_auc_drop_mean"] for n in names],
        0.4,
        color="none",
        edgecolor=plots.INK,
        hatch="//",
        label="PR-AUC lost when group is shuffled",
    )
    axes[1].set_xticks(x, [f"{n}\n({(groups == n).sum()} feat.)" for n in names])
    axes[1].set_title("Importance by feature group")
    axes[1].legend(loc="upper right", fontsize=8)
    plots.save(fig, "e08_importance")

    io.save_table(per_feature.head(30), "e08_top_features")
    io.save_table(perm, "e08_group_permutation")
    io.save_json(
        {
            "test_pr_auc": base_score,
            "shap_group_share": share.to_dict(),
            "shap_group_share_by_period": by_period,
            "group_permutation": perm_rows,
            "top_features": per_feature.head(30).to_dict(orient="records"),
        },
        "e08_explain",
    )
    print(share, by_period, perm, sep="\n")


if __name__ == "__main__":
    main()
