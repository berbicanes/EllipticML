"""E5b - Why do DirSAGE embeddings hurt the tree model when GCN embeddings do not?

Hypothesis: the embeddings are produced by a GNN trained on the labels of
steps 1-34. On those rows a strong GNN has (nearly) memorised the label, so
the tree model sees an almost perfect feature in training and leans on it;
on the test steps the same feature is only as good as the GNN itself.

Test: for GCN and DirSAGE (the configurations E4 selected, seed 0), compare
PR-AUC of the GNN's own output on the training rows and on the test rows, and
the share of the tree model's split importance that goes to the embedding
columns when they are appended to AF.
"""

from __future__ import annotations

import ast
import warnings

import numpy as np
import pandas as pd

from elliptic import config, data, evaluate, gnn, io, models, runner, split

warnings.filterwarnings("ignore", category=UserWarning)


def main() -> None:
    d = data.load()
    nodes, edges = d.nodes, d.edges
    af = data.feature_columns("AF")
    s = split.temporal_split(nodes)
    y = np.clip(nodes["label"].to_numpy(), 0, 1)
    sel = io.load_json("e04_graph")["gnn_selection"]
    rf_params = {
        r["Model"]: ast.literal_eval(r["best_params"])
        for r in io.load_json("e02_models")["summary"]
    }["RandomForest"]
    g = gnn.build_tensors(
        nodes, edges, af, split.steps_mask(nodes, config.TRAIN_STEPS, False), gnn.default_device()
    )
    rows = []
    for kind in ("gcn", "dirsage"):
        cfg = gnn.TrainConfig(**sel[kind]["chosen"])
        model, _ = gnn.train(g, y, s.train, cfg, seed=config.DEFAULT_SEED)
        p = gnn.predict(model, g)
        emb = gnn.embeddings(model, g)
        ne = pd.DataFrame(emb, columns=[f"ne_{i:03d}" for i in range(emb.shape[1])])
        table = pd.concat([nodes.reset_index(drop=True), ne], axis=1)
        cols = af + list(ne.columns)
        train = runner.xy(table, cols, s.train)
        rf = models.make_model("RandomForest", seed=config.DEFAULT_SEED, **rf_params)
        rf.fit(train.X, train.y)
        imp = rf.named_steps["clf"].feature_importances_
        rows.append(
            {
                "GNN": kind,
                "GNN PR-AUC train rows": evaluate.pr_auc(y[s.train], p[s.train]),
                "GNN PR-AUC test rows": evaluate.pr_auc(y[s.test], p[s.test]),
                "RF importance on embeddings": float(imp[len(af) :].sum()),
                "embedding dims": emb.shape[1],
            }
        )
        print(rows[-1])
    df = pd.DataFrame(rows)
    io.save_table(df, "e05b_memorisation")
    io.save_json({"rows": rows}, "e05b_memorisation")


if __name__ == "__main__":
    main()
