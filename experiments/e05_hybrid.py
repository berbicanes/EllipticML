"""E5 - Hybrid with our best graph model.

E4 follows Weber et al. and appends *GCN* embeddings to the tree models'
features. But in E4 the directed model (DirSAGE) is clearly the better GNN,
so here we ask whether *its* embeddings help a tree model where the GCN's did
not. Same protocol as E4: DirSAGE with the configuration E4 selected on
validation steps, trained on steps 1-34; its 100-d last hidden layer appended
to AF; RF and XGBoost with E2's hyper-parameters, threshold tuned on forward
folds, five seeds each; paired bootstrap against plain AF.

One embedding model per tree seed (seed k of DirSAGE feeds seed k of the
tree model), so the seed variance includes the GNN's.
"""

from __future__ import annotations

import ast
import warnings

import numpy as np
import pandas as pd

from elliptic import config, data, evaluate, gnn, io, runner, split

warnings.filterwarnings("ignore", category=UserWarning)


def main() -> None:
    d = data.load()
    nodes, edges = d.nodes, d.edges
    af = data.feature_columns("AF")
    s = split.temporal_split(nodes)
    y_all = np.clip(nodes["label"].to_numpy(), 0, 1)
    y_test = y_all[s.test]
    steps_test = nodes["time_step"].to_numpy()[s.test]

    chosen = io.load_json("e04_graph")["gnn_selection"]["dirsage"]["chosen"]
    cfg = gnn.TrainConfig(**chosen)
    g = gnn.build_tensors(
        nodes, edges, af, split.steps_mask(nodes, config.TRAIN_STEPS, False), gnn.default_device()
    )
    best_params = {
        r["Model"]: ast.literal_eval(r["best_params"])
        for r in io.load_json("e02_models")["summary"]
    }
    e04_scores = io.load_scores("e04_graph")
    e04_thr = io.load_json("e04_graph")["thresholds"]

    scores, thresholds, rows = {}, {}, []
    per_model = {m: [] for m in ("RandomForest", "XGBoost")}
    for seed in config.SEEDS:
        model, _ = gnn.train(g, y_all, s.train, cfg, seed=seed)
        emb = gnn.embeddings(model, g)
        ne = pd.DataFrame(emb, columns=[f"dne_{i:03d}" for i in range(emb.shape[1])])
        table = pd.concat([nodes.reset_index(drop=True), ne], axis=1)
        cols = af + list(ne.columns)
        for name in per_model:
            r = runner.run_model(
                table, cols, name, seed=seed, tune_threshold=True, params=best_params[name]
            )
            m = evaluate.metrics(y_test, r.score, r.threshold)
            per_model[name].append(m)
            scores[f"{name}|AF+DirNE_s{seed}"] = r.score
            thresholds[f"{name}|AF+DirNE_s{seed}"] = r.threshold
            print(f"seed {seed} {name:12s} F1 {m['f1']:.3f} PR-AUC {m['pr_auc']:.3f}")

    paired = []
    for name, runs in per_model.items():
        key = f"{name}|AF+DirNE"
        mean = np.mean([scores[f"{key}_s{k}"] for k in config.SEEDS], axis=0)
        thr = float(np.mean([thresholds[f"{key}_s{k}"] for k in config.SEEDS]))
        scores[f"{key}_mean"], thresholds[f"{key}_mean"] = mean, thr
        df = pd.DataFrame(runs)
        ci = evaluate.bootstrap_ci(y_test, mean, steps_test, evaluate.f1_at(thr))
        rows.append(
            {
                "Model": name,
                "Features": "AF+NE (DirSAGE)",
                "Precision": df["precision"].mean(),
                "Recall": df["recall"].mean(),
                "F1": df["f1"].mean(),
                "F1 sd (seeds)": df["f1"].std(ddof=1),
                "F1 95% CI": f"[{ci['ci_low']:.3f}, {ci['ci_high']:.3f}]",
                "PR-AUC": df["pr_auc"].mean(),
                "ROC-AUC": df["roc_auc"].mean(),
            }
        )
        base = e04_scores[f"{name}|AF_mean"].to_numpy()
        base_t = e04_thr[f"{name}|AF_mean"]
        f1 = evaluate.paired_bootstrap(
            y_test, mean, base, steps_test, evaluate.f1_at(thr), evaluate.f1_at(base_t)
        )
        ap = evaluate.paired_bootstrap(y_test, mean, base, steps_test, evaluate.pr_auc)
        paired.append(
            {
                "Model": name,
                "Variant vs AF": "AF+NE (DirSAGE)",
                "F1 diff": f1["difference"],
                "F1 diff CI": f"[{f1['ci_low']:.3f}, {f1['ci_high']:.3f}]",
                "F1 p": f1["p_value"],
                "PR-AUC diff": ap["difference"],
                "PR-AUC diff CI": f"[{ap['ci_low']:.3f}, {ap['ci_high']:.3f}]",
                "PR-AUC p": ap["p_value"],
            }
        )

    io.save_scores(scores, nodes.loc[s.test], "e05_hybrid")
    io.save_table(pd.DataFrame(rows), "e05_hybrid")
    io.save_table(pd.DataFrame(paired), "e05_paired")
    io.save_json({"summary": rows, "paired": paired, "thresholds": thresholds}, "e05_hybrid")
    print(pd.DataFrame(rows).round(3).to_string(index=False))
    print(pd.DataFrame(paired).round(4).to_string(index=False))


if __name__ == "__main__":
    main()
