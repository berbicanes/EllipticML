"""E7 - Why does performance collapse over time?

Weber et al. observed that every model stops catching illicit transactions
after the dark market shutdown at step 43, and that retraining a Random
Forest after every step does not fix it. This experiment measures *why*.

1. Performance per test step for every model (the headline figure).
2. What changed? Adversarial validation: a classifier tries to tell
   illicit transactions from before step 43 apart from those after it. The
   same test on licit transactions is the control. If the illicit AUC is far
   above the licit one, the illicit population itself changed.
3. Are post-shutdown illicit transactions *new*? For every illicit test
   transaction, the distance to its nearest illicit training transaction
   relative to its nearest licit one. Above 1 = it looks more like known
   licit traffic than like any illicit pattern the model has seen.
4. Does retraining help? Walk-forward evaluation: before scoring step t,
   retrain on every labelled step < t (expanding window) or on the last 10
   steps only (sliding window), versus the static model trained once on 1-34.
5. Hidden clocks. A few anonymised features rise almost monotonically with
   time (|Spearman rho| > 0.9 with the time step, measured on training steps
   only). They act like a disguised time stamp: any two periods are perfectly
   separable through them. They are removed before the adversarial and
   novelty analyses (2-3), and we test whether removing them from the model
   changes its test performance.
"""

from __future__ import annotations

import ast
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, spearmanr
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from elliptic import config, data, evaluate, io, models, plots, runner, split

warnings.filterwarnings("ignore", category=UserWarning)


# ---------------------------------------------------------------- 1. per-step F1
def per_step_table(scores: pd.DataFrame, thresholds: dict, keys: dict[str, str]) -> pd.DataFrame:
    y, steps = scores["label"].to_numpy(), scores["time_step"].to_numpy()
    out = []
    for label, key in keys.items():
        df = evaluate.per_step(y, scores[key].to_numpy(), steps, thresholds[key])
        df["model"] = label
        out.append(df)
    return pd.concat(out, ignore_index=True)


def fig_per_step(ps: pd.DataFrame, counts: pd.DataFrame) -> None:
    plots.setup()
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(9, 5.6), sharex=True, gridspec_kw={"height_ratios": [3, 1]}
    )
    for model, df in ps.groupby("model", sort=False):
        ax1.plot(
            df["time_step"],
            df["f1"],
            marker="o",
            ms=3.5,
            lw=1.6,
            color=plots.MODEL_COLORS.get(model.split(" ")[0], plots.MUTED),
            label=model,
        )
    ax1.set_ylabel("illicit F1")
    ax1.set_ylim(-0.02, 1.02)
    ax1.set_title("Illicit F1 per test time step")
    ax1.legend(loc="lower left", ncol=3, fontsize=8)
    plots.mark_timeline(ax1, label=True, shade=False)
    ax2.bar(counts["time_step"], counts["n_illicit"], color=plots.ILLICIT)
    ax2.set_ylabel("illicit\nin step")
    ax2.set_xlabel("time step")
    plots.mark_timeline(ax2, label=False, shade=False)
    ax1.set_xlim(34.5, 49.5)
    plots.save(fig, "e07_f1_per_step")


# ----------------------------------------------------------------- 5. hidden clocks
def clock_features(nodes: pd.DataFrame, cols: list[str], cutoff: float = 0.9) -> dict[str, float]:
    """Features whose rank correlation with the time step exceeds ``cutoff``.

    Measured on all rows of the training steps (no labels, no test data).
    """
    tr = nodes[nodes["time_step"].isin(list(config.TRAIN_STEPS))]
    out = {}
    for c in cols:
        rho = abs(spearmanr(tr["time_step"], tr[c]).statistic)
        if rho > cutoff:
            out[c] = float(rho)
    return dict(sorted(out.items(), key=lambda kv: -kv[1]))


def clock_ablation(nodes, cols, clocks, params_by_model, test) -> list[dict]:
    no_clock = [c for c in cols if c not in clocks]
    rows = []
    for name in ("RandomForest", "XGBoost"):
        res = {}
        for label, use in (("AF", cols), ("AF without clocks", no_clock)):
            ss, ts = [], []
            for seed in config.SEEDS:
                r = runner.run_model(
                    nodes, use, name, seed=seed, tune_threshold=True, params=params_by_model[name]
                )
                ss.append(r.score)
                ts.append(r.threshold)
            res[label] = (np.mean(ss, axis=0), float(np.mean(ts)), ss, ts)
        (sa, ta, _, _), (sb, tb, _, _) = res["AF without clocks"], res["AF"]
        f1 = evaluate.paired_bootstrap(
            test.y, sa, sb, test.steps, evaluate.f1_at(ta), evaluate.f1_at(tb)
        )
        ap = evaluate.paired_bootstrap(test.y, sa, sb, test.steps, evaluate.pr_auc)
        for label, (_, _, ss, ts) in res.items():
            ms = [evaluate.metrics(test.y, s_, t_) for s_, t_ in zip(ss, ts, strict=True)]
            rows.append(
                {
                    "Model": name,
                    "Features": label,
                    "n_features": len(cols) if label == "AF" else len(no_clock),
                    "F1": float(np.mean([m["f1"] for m in ms])),
                    "PR-AUC": float(np.mean([m["pr_auc"] for m in ms])),
                    "F1 diff vs AF": f1["difference"] if label != "AF" else 0.0,
                    "F1 p": f1["p_value"] if label != "AF" else np.nan,
                    "PR-AUC diff vs AF": ap["difference"] if label != "AF" else 0.0,
                    "PR-AUC p": ap["p_value"] if label != "AF" else np.nan,
                }
            )
        print(rows[-2:])
    return rows


# ---------------------------------------------------------- 2. adversarial validation
def adversarial(nodes: pd.DataFrame, cols: list[str], seed: int) -> dict:
    lab = nodes[nodes["label"] != config.UNKNOWN]
    out = {}
    for cls, name in ((config.ILLICIT, "illicit"), (config.LICIT, "licit")):
        sub = lab[lab["label"] == cls]
        X = sub[cols].to_numpy(np.float32)
        after = (sub["time_step"] >= config.DARK_MARKET_STEP).to_numpy().astype(int)
        # Balance the two periods so the AUC is not driven by sample size.
        rng = np.random.default_rng(seed)
        i_after, i_before = np.flatnonzero(after == 1), np.flatnonzero(after == 0)
        n = min(len(i_after), len(i_before))
        idx = np.concatenate(
            [rng.choice(i_after, n, replace=False), rng.choice(i_before, n, replace=False)]
        )
        clf = models.make_model("RandomForest", seed=seed, n_estimators=300)
        cv = StratifiedKFold(5, shuffle=True, random_state=seed)
        p = cross_val_predict(clf, X[idx], after[idx], cv=cv, method="predict_proba")[:, 1]
        ks = sorted(
            (
                (c, ks_2samp(X[after == 1, j], X[after == 0, j]).statistic)
                for j, c in enumerate(cols)
            ),
            key=lambda t: -t[1],
        )
        out[name] = {
            "n_each_period": int(n),
            "auc_before_vs_after_43": float(roc_auc_score(after[idx], p)),
            "top_shifted_features_ks": [{"feature": c, "ks": float(v)} for c, v in ks[:10]],
            "share_features_ks_gt_0.3": float(np.mean([v > 0.3 for _, v in ks])),
        }
        print(name, out[name]["auc_before_vs_after_43"])
    return out


# --------------------------------------------------------------- 3. novelty of illicit
def novelty(nodes: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    s = split.temporal_split(nodes)
    scaler = StandardScaler().fit(nodes.loc[s.train, cols])
    tr = nodes.loc[s.train]
    te = nodes.loc[s.test & (nodes["label"] == config.ILLICIT).to_numpy()]
    Xi = scaler.transform(tr.loc[tr["label"] == config.ILLICIT, cols])
    Xl = scaler.transform(tr.loc[tr["label"] == config.LICIT, cols])
    Xt = scaler.transform(te[cols])
    di = NearestNeighbors(n_neighbors=1).fit(Xi).kneighbors(Xt)[0][:, 0]
    dl = NearestNeighbors(n_neighbors=1).fit(Xl).kneighbors(Xt)[0][:, 0]
    return pd.DataFrame(
        {"time_step": te["time_step"].to_numpy(), "ratio": di / np.maximum(dl, 1e-9)}
    )


def fig_novelty(nv: pd.DataFrame) -> None:
    plots.setup()
    fig, ax = plt.subplots(figsize=(9, 3.4))
    steps = sorted(nv["time_step"].unique())
    data_ = [np.log2(nv.loc[nv["time_step"] == t, "ratio"]) for t in steps]
    bp = ax.boxplot(
        data_,
        positions=steps,
        widths=0.6,
        showfliers=False,
        patch_artist=True,
        medianprops={"color": plots.INK, "lw": 1.5},
    )
    for b, t in zip(bp["boxes"], steps, strict=True):
        b.set_facecolor(plots.ILLICIT if t >= config.DARK_MARKET_STEP else "#E8B4AE")
        b.set_edgecolor(plots.INK)
    ax.axhline(0, color=plots.MUTED, lw=1)
    ax.set_ylabel("log2(dist. to nearest illicit /\n dist. to nearest licit)")
    ax.set_xlabel("test time step")
    ax.set_title("Illicit test transactions: closer to known illicit (<0) or to licit (>0)?")
    ax.set_xlim(34.5, 49.5)
    plots.mark_timeline(ax, label=False, shade=False)
    plots.save(fig, "e07_novelty")


# ------------------------------------------------------------------- 4. walk-forward
def walk_forward(nodes, cols, name, params, threshold, window=None) -> pd.DataFrame:
    rows = []
    for t in config.TEST_STEPS:
        lo = 1 if window is None else max(1, t - window)
        train_mask = split.steps_mask(nodes, range(lo, t))
        test_mask = split.steps_mask(nodes, [t])
        tr, te = runner.xy(nodes, cols, train_mask), runner.xy(nodes, cols, test_mask)
        m = models.make_model(name, seed=config.DEFAULT_SEED, y_train=tr.y, **params)
        sc = runner.fit_predict(m, tr, te)
        r = evaluate.per_step(te.y, sc, te.steps, threshold).iloc[0].to_dict()
        r["pr_auc"] = evaluate.pr_auc(te.y, sc)
        rows.append(r)
    return pd.DataFrame(rows)


def fig_walk_forward(wf: pd.DataFrame) -> None:
    plots.setup()
    fig, ax = plt.subplots(figsize=(9, 3.6))
    styles = {
        "static (train 1-34)": ("-", plots.MUTED),
        "expanding window": ("-", plots.LICIT),
        "sliding window (10 steps)": ("--", plots.ILLICIT),
    }
    for strat, df in wf.groupby("strategy", sort=False):
        ls, c = styles[strat]
        ax.plot(df["time_step"], df["f1"], ls, marker="o", ms=3.5, color=c, label=strat)
    ax.set_ylim(-0.02, 1.02)
    ax.set_ylabel("illicit F1")
    ax.set_xlabel("test time step")
    ax.set_title("Retraining before every step (Random Forest)")
    ax.legend(loc="lower left")
    ax.set_xlim(34.5, 49.5)
    plots.mark_timeline(ax, label=False, shade=False)
    plots.save(fig, "e07_walk_forward")


def main() -> None:
    nodes = data.load().nodes
    cols = data.feature_columns("AF")

    # 1. per-step curves from saved scores
    e02 = io.load_scores("e02_models")
    thr = io.load_json("e02_models")["thresholds"]
    keys = {m: f"{m}_mean" for m in ("RandomForest", "XGBoost", "MLP", "LogReg")}
    labels = {
        "RandomForest": "RandomForest",
        "XGBoost": "XGBoost",
        "MLP": "MLP",
        "LogReg": "LogReg",
    }
    ps = per_step_table(e02, thr, {labels[k]: v for k, v in keys.items()})
    try:
        e04 = io.load_scores("e04_graph")
        thr4 = io.load_json("e04_graph")["thresholds"]
        best_gnn = max(
            ("GCN", "SkipGCN", "DirSAGE"),
            key=lambda g: evaluate.pr_auc(e04["label"], e04[f"{g}_mean"]),
        )
        ps = pd.concat(
            [ps, per_step_table(e04, thr4, {best_gnn: f"{best_gnn}_mean"})], ignore_index=True
        )
    except FileNotFoundError:
        print("E4 scores not found; per-step figure without GNN")
    counts = ps.drop_duplicates("time_step")[["time_step", "n_illicit"]]
    fig_per_step(ps, counts)
    pre = ps[ps["time_step"] < config.DARK_MARKET_STEP].groupby("model")["f1"].mean()
    post = ps[ps["time_step"] >= config.DARK_MARKET_STEP].groupby("model")["f1"].mean()

    # 5a. hidden clocks, then 2-3 without them
    clocks = clock_features(nodes, cols)
    print("clock features:", clocks)
    no_clock = [c for c in cols if c not in clocks]
    adv = {
        "with_clock_features": adversarial(nodes, cols, config.DEFAULT_SEED),
        "without_clock_features": adversarial(nodes, no_clock, config.DEFAULT_SEED),
    }
    nv = novelty(nodes, no_clock)
    fig_novelty(nv)
    nv_summary = {
        "median_ratio_35_42": float(nv.loc[nv["time_step"] < 43, "ratio"].median()),
        "median_ratio_43_49": float(nv.loc[nv["time_step"] >= 43, "ratio"].median()),
        "share_closer_to_licit_35_42": float((nv.loc[nv["time_step"] < 43, "ratio"] > 1).mean()),
        "share_closer_to_licit_43_49": float((nv.loc[nv["time_step"] >= 43, "ratio"] > 1).mean()),
    }

    # 4. retraining
    params = {
        r["Model"]: ast.literal_eval(r["best_params"])
        for r in io.load_json("e02_models")["summary"]
    }["RandomForest"]
    t_rf = thr["RandomForest_mean"]
    static = ps[ps["model"] == "RandomForest"].assign(strategy="static (train 1-34)")
    exp = walk_forward(nodes, cols, "RandomForest", params, t_rf).assign(
        strategy="expanding window"
    )
    sld = walk_forward(nodes, cols, "RandomForest", params, t_rf, window=10).assign(
        strategy="sliding window (10 steps)"
    )
    wf = pd.concat([static, exp, sld], ignore_index=True)
    fig_walk_forward(wf)
    wf_summary = (
        wf.assign(period=np.where(wf["time_step"] < 43, "35-42", "43-49"))
        .groupby(["strategy", "period"])["f1"]
        .mean()
        .unstack()
        .reset_index()
    )

    # 5b. does removing the clocks change the model?
    test = runner.test_xy(nodes, cols)
    params_all = {
        r["Model"]: ast.literal_eval(r["best_params"])
        for r in io.load_json("e02_models")["summary"]
    }
    ablation = clock_ablation(nodes, cols, clocks, params_all, test)
    io.save_table(pd.DataFrame(ablation), "e07_clock_ablation")

    io.save_table(ps, "e07_per_step")
    io.save_table(wf_summary, "e07_walk_forward")
    io.save_json(
        {
            "mean_f1_steps_35_42": pre.to_dict(),
            "mean_f1_steps_43_49": post.to_dict(),
            "clock_features": clocks,
            "clock_ablation": ablation,
            "adversarial_validation": adv,
            "novelty": nv_summary,
            "walk_forward": wf_summary.to_dict(orient="records"),
        },
        "e07_drift",
    )
    print(pre, post, nv_summary, wf_summary, sep="\n")


if __name__ == "__main__":
    main()
