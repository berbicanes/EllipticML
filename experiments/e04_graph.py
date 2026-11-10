"""E4 - Does the position in the transaction graph help?

This is the research question of the project, attacked three ways:

A. Graph neural networks (GCN, Skip-GCN, directed SAGE) that learn from the
   graph directly. Model selection: train on steps 1-29, watch PR-AUC on
   steps 30-34 every 25 epochs, pick the configuration and epoch count; the
   decision threshold is the F1-optimal one on those validation steps. Then
   refit on steps 1-34 with five seeds.
B. Tree models (RF, XGBoost, tuned in E2) on richer feature sets:
     LF      93 local features
     AF      + 72 aggregated neighbour features (the dataset's own graph summary)
     AF+G    + 9 label-free structural features (degree, 2-hop reach, PageRank...)
     AF+NE   + 100-d node embeddings from the best GCN (Weber et al.'s hybrid)
     AF+G+NE everything
C. A paired bootstrap of every graph-enriched variant against plain AF.

Caveat on NE: the embeddings come from a GNN trained on steps 1-34, so on
training rows they have seen the labels. This is how Weber et al. built
them; it can only make the tree model *over*-trust NE on training data, which
would hurt, not flatter, its test score.
"""

from __future__ import annotations

import ast
import itertools
import warnings

import numpy as np
import pandas as pd

from elliptic import config, data, evaluate, gnn, graph, io, runner, split

warnings.filterwarnings("ignore", category=UserWarning)

GNN_KINDS = ("gcn", "skipgcn", "dirsage")
GNN_NAMES = {"gcn": "GCN", "skipgcn": "SkipGCN", "dirsage": "DirSAGE"}
GNN_GRID = [
    {"lr": lr, "dropout": dr, "hidden": 100, "epochs": 1000}
    for lr, dr in itertools.product([1e-3, 3e-3], [0.0, 0.3])
]


def select_gnn(nodes, edges, cols, kind):
    """Pick config + epochs on validation steps 30-34 (training on 1-29)."""
    fit = split.steps_mask(nodes, range(1, 30))
    val = split.steps_mask(nodes, config.VALIDATION_STEPS)
    g = gnn.build_tensors(
        nodes, edges, cols, split.steps_mask(nodes, range(1, 30), False), gnn.default_device()
    )
    y = np.clip(nodes["label"].to_numpy(), 0, 1)  # unknowns never enter the loss
    best = None
    log = []
    for params in GNN_GRID:
        cfg = gnn.TrainConfig(kind=kind, **params)
        model, hist = gnn.train(g, y, fit, cfg, seed=config.DEFAULT_SEED, val_mask=val)
        top = max(hist, key=lambda h: h["val_pr_auc"])
        log.append({**params, "best_epoch": top["epoch"], "val_pr_auc": top["val_pr_auc"]})
        if best is None or top["val_pr_auc"] > best[0]:
            best = (top["val_pr_auc"], params, top["epoch"])
    _, params, epochs = best
    # Threshold: retrain the chosen config for the chosen epochs, F1-optimal on validation.
    cfg = gnn.TrainConfig(kind=kind, **{**params, "epochs": epochs})
    model, _ = gnn.train(g, y, fit, cfg, seed=config.DEFAULT_SEED)
    p = gnn.predict(model, g)
    threshold = evaluate.best_f1_threshold(y[val], p[val])
    return cfg, threshold, log


def main() -> None:
    d = data.load()
    nodes, edges = d.nodes, d.edges
    af = data.feature_columns("AF")
    s = split.temporal_split(nodes)
    test_rows = nodes.loc[s.test]
    y_all = np.clip(nodes["label"].to_numpy(), 0, 1)
    y_test, steps_test = y_all[s.test], nodes["time_step"].to_numpy()[s.test]

    scores, thresholds, summary, selection = {}, {}, [], {}

    # ---- A. Graph neural networks -------------------------------------------------
    g_full = gnn.build_tensors(
        nodes, edges, af, split.steps_mask(nodes, config.TRAIN_STEPS, False), gnn.default_device()
    )
    best_gcn_emb, best_gcn_ap = None, -1.0
    for kind in GNN_KINDS:
        cfg, thr, log = select_gnn(nodes, edges, af, kind)
        selection[kind] = {"chosen": cfg.__dict__, "threshold": thr, "grid": log}
        print(f"{kind}: {cfg} threshold {thr:.3f}")
        per_seed = []
        for seed in config.SEEDS:
            model, _ = gnn.train(g_full, y_all, s.train, cfg, seed=seed)
            p = gnn.predict(model, g_full)
            m = evaluate.metrics(y_test, p[s.test], thr)
            per_seed.append(m)
            name = GNN_NAMES[kind]
            scores[f"{name}_s{seed}"] = p[s.test]
            print(f"  seed {seed}: F1 {m['f1']:.3f} PR-AUC {m['pr_auc']:.3f}")
            if kind == "gcn" and seed == config.DEFAULT_SEED:
                best_gcn_emb = gnn.embeddings(model, g_full)
                best_gcn_ap = m["pr_auc"]
        name = GNN_NAMES[kind]
        scores[f"{name}_mean"] = np.mean([scores[f"{name}_s{k}"] for k in config.SEEDS], axis=0)
        thresholds[f"{name}_mean"] = thr
        summary.append(
            _row(name, "graph (GNN)", per_seed, y_test, scores[f"{name}_mean"], steps_test, thr)
        )
    print(f"GCN embeddings taken from seed {config.DEFAULT_SEED} (test PR-AUC {best_gcn_ap:.3f})")

    # ---- B. Tree models on feature sets -----------------------------------------------
    g_feats = graph.structural_features(edges, len(nodes))
    ne = pd.DataFrame(best_gcn_emb, columns=[f"ne_{i:03d}" for i in range(best_gcn_emb.shape[1])])
    table = pd.concat([nodes.reset_index(drop=True), g_feats, ne], axis=1)
    lf = data.feature_columns("LF")
    sets = {
        "LF": lf,
        "LF+G": lf + graph.GRAPH_COLS,
        "AF": af,
        "AF+G": af + graph.GRAPH_COLS,
        "AF+NE": af + list(ne.columns),
        "AF+G+NE": af + graph.GRAPH_COLS + list(ne.columns),
    }
    best_params = {
        r["Model"]: ast.literal_eval(r["best_params"])
        for r in io.load_json("e02_models")["summary"]
    }
    for model in ("RandomForest", "XGBoost"):
        for fs, cols in sets.items():
            per_seed = []
            for seed in config.SEEDS:
                r = runner.run_model(
                    table, cols, model, seed=seed, tune_threshold=True, params=best_params[model]
                )
                per_seed.append(evaluate.metrics(y_test, r.score, r.threshold))
                scores[f"{model}|{fs}_s{seed}"] = r.score
                thresholds[f"{model}|{fs}_s{seed}"] = r.threshold
            key = f"{model}|{fs}"
            scores[f"{key}_mean"] = np.mean([scores[f"{key}_s{k}"] for k in config.SEEDS], axis=0)
            thresholds[f"{key}_mean"] = float(
                np.mean([thresholds[f"{key}_s{k}"] for k in config.SEEDS])
            )
            row = _row(
                model,
                fs,
                per_seed,
                y_test,
                scores[f"{key}_mean"],
                steps_test,
                thresholds[f"{key}_mean"],
            )
            summary.append(row)
            print(f"{model:12s} {fs:8s} F1 {row['F1']:.3f} PR-AUC {row['PR-AUC']:.3f}")

    # ---- C. Paired tests against plain AF ------------------------------------------------
    paired = []
    for model in ("RandomForest", "XGBoost"):
        base = f"{model}|AF_mean"
        for fs in ("AF+G", "AF+NE", "AF+G+NE", "LF"):
            other = f"{model}|{fs}_mean"
            f1 = evaluate.paired_bootstrap(
                y_test,
                scores[other],
                scores[base],
                steps_test,
                evaluate.f1_at(thresholds[other]),
                evaluate.f1_at(thresholds[base]),
            )
            ap = evaluate.paired_bootstrap(
                y_test, scores[other], scores[base], steps_test, evaluate.pr_auc
            )
            paired.append(
                {
                    "Model": model,
                    "Variant vs AF": fs,
                    "F1 diff": f1["difference"],
                    "F1 diff CI": f"[{f1['ci_low']:.3f}, {f1['ci_high']:.3f}]",
                    "F1 p": f1["p_value"],
                    "PR-AUC diff": ap["difference"],
                    "PR-AUC diff CI": f"[{ap['ci_low']:.3f}, {ap['ci_high']:.3f}]",
                    "PR-AUC p": ap["p_value"],
                }
            )

    summary_df = pd.DataFrame(summary)
    paired_df = pd.DataFrame(paired)
    io.save_scores(scores, test_rows, "e04_graph")
    io.save_table(summary_df, "e04_graph")
    io.save_table(paired_df, "e04_paired")
    io.save_json(
        {
            "summary": summary,
            "paired": paired,
            "thresholds": thresholds,
            "gnn_selection": selection,
        },
        "e04_graph",
    )
    print(summary_df.round(3).to_string(index=False))
    print(paired_df.round(4).to_string(index=False))


def _row(model, features, per_seed, y, mean_score, steps, thr) -> dict:
    df = pd.DataFrame(per_seed)
    ci = evaluate.bootstrap_ci(y, mean_score, steps, evaluate.f1_at(thr))
    return {
        "Model": model,
        "Features": features,
        "Precision": df["precision"].mean(),
        "Recall": df["recall"].mean(),
        "F1": df["f1"].mean(),
        "F1 sd (seeds)": df["f1"].std(ddof=1),
        "F1 95% CI": f"[{ci['ci_low']:.3f}, {ci['ci_high']:.3f}]",
        "PR-AUC": df["pr_auc"].mean(),
        "ROC-AUC": df["roc_auc"].mean(),
    }


if __name__ == "__main__":
    main()
