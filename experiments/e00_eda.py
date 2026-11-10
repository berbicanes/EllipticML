"""E0 - Exploratory data analysis.

Questions answered here, each backed by a figure or a number in
results/e00_eda.json:

1. How are the labels spread over time, and how does the illicit share move?
2. How redundant are the 165 features?
3. Do the classes separate in a linear 2-D projection?
4. What does the graph look like (degree, components per step)?
5. Are illicit transactions connected to other illicit transactions (homophily)?
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.sparse.csgraph import connected_components
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from elliptic import config, data, io, plots


def label_counts_per_step(nodes: pd.DataFrame) -> pd.DataFrame:
    counts = (
        nodes.groupby(["time_step", "label"])
        .size()
        .unstack(fill_value=0)
        .rename(
            columns={config.ILLICIT: "illicit", config.LICIT: "licit", config.UNKNOWN: "unknown"}
        )
    )
    counts["illicit_share"] = counts["illicit"] / (counts["illicit"] + counts["licit"])
    return counts


def fig_labels_over_time(counts: pd.DataFrame) -> None:
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(9, 5.2), sharex=True, gridspec_kw={"height_ratios": [2, 1]}
    )
    x = counts.index.to_numpy()
    ax1.bar(x, counts["licit"], color=plots.LICIT, label="licit")
    ax1.bar(x, counts["illicit"], bottom=counts["licit"], color=plots.ILLICIT, label="illicit")
    ax1.set_ylabel("labelled transactions")
    ax1.set_title("Labelled transactions per time step")
    ax1.legend(loc="upper center", ncol=2, bbox_to_anchor=(0.4, 1.0))
    plots.mark_timeline(ax1)

    ax2.plot(x, counts["illicit_share"] * 100, color=plots.ILLICIT, marker="o", ms=3)
    ax2.set_ylabel("illicit share (%)")
    ax2.set_xlabel("time step (about two weeks each)")
    plots.mark_timeline(ax2, label=False)
    plots.save(fig, "eda_labels_over_time")


def fig_correlation(nodes: pd.DataFrame) -> dict:
    X = nodes[data.FEATURE_COLS].to_numpy()
    corr = np.corrcoef(X, rowvar=False)
    corr = np.nan_to_num(corr)
    order = leaves_list(linkage(X.T, method="average", metric="correlation"))
    fig, ax = plt.subplots(figsize=(6.2, 5.4))
    im = ax.imshow(corr[np.ix_(order, order)], cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_title("Feature correlation (165 features, clustered)")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    fig.colorbar(im, ax=ax, shrink=0.8, label="Pearson r")
    plots.save(fig, "eda_feature_correlation")

    upper = np.abs(corr[np.triu_indices_from(corr, k=1)])
    # Greedy count of features left after dropping one of every |r| > 0.95 pair.
    keep: list[int] = []
    for j in range(corr.shape[0]):
        if all(abs(corr[j, k]) <= 0.95 for k in keep):
            keep.append(j)
    return {
        "pairs_abs_r_gt_0.9": int((upper > 0.9).sum()),
        "pairs_abs_r_gt_0.95": int((upper > 0.95).sum()),
        "features_after_0.95_filter": len(keep),
    }


def fig_pca(labelled: pd.DataFrame, seed: int) -> dict:
    out = {}
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.4))
    for ax, fs in zip(axes, ["LF", "AF"], strict=True):
        X = StandardScaler().fit_transform(labelled[data.feature_columns(fs)])
        pca = PCA(n_components=10, random_state=seed).fit(X)
        Z = pca.transform(X)[:, :2]
        y = labelled["label"].to_numpy()
        rng = np.random.default_rng(seed)
        lic = rng.choice(np.flatnonzero(y == config.LICIT), 8000, replace=False)
        ill = np.flatnonzero(y == config.ILLICIT)
        ax.scatter(*Z[lic].T, s=3, alpha=0.35, color=plots.LICIT, label="licit (sample)")
        ax.scatter(*Z[ill].T, s=3, alpha=0.5, color=plots.ILLICIT, label="illicit")
        ev = pca.explained_variance_ratio_.astype(np.float64)
        ax.set_title(f"PCA, {fs} features")
        ax.set_xlabel(f"PC1 ({ev[0]:.0%} of variance)")
        ax.set_ylabel(f"PC2 ({ev[1]:.0%})")
        lo, hi = np.percentile(Z, [0.5, 99.5], axis=0)
        ax.set_xlim(lo[0], hi[0])
        ax.set_ylim(lo[1], hi[1])
        out[fs] = {"explained_variance_top10": ev.round(4).tolist()}
    axes[0].legend(loc="upper right", markerscale=4)
    plots.save(fig, "eda_pca")
    return out


def graph_stats(nodes: pd.DataFrame, edges: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    n = len(nodes)
    A = sp.csr_matrix((np.ones(len(edges)), (edges["src"], edges["dst"])), shape=(n, n))
    out_deg = np.asarray(A.sum(1)).ravel()
    in_deg = np.asarray(A.sum(0)).ravel()
    n_comp, comp = connected_components(A, directed=True, connection="weak")
    comp_size = np.bincount(comp)
    deg = pd.DataFrame(
        {"label": nodes["label"], "in_deg": in_deg, "out_deg": out_deg, "deg": in_deg + out_deg}
    )
    stats = {
        "nodes": n,
        "edges": len(edges),
        "mean_degree": float((in_deg + out_deg).mean()),
        "max_in_degree": int(in_deg.max()),
        "max_out_degree": int(out_deg.max()),
        "isolated_nodes": int(((in_deg + out_deg) == 0).sum()),
        "weak_components": int(n_comp),
        "largest_component": int(comp_size.max()),
        "median_degree_by_class": deg.groupby("label")["deg"].median().to_dict(),
        "mean_degree_by_class": deg.groupby("label")["deg"].mean().round(3).to_dict(),
    }
    return stats, deg


def fig_degree(deg: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    for ax, col, title in zip(
        axes, ["in_deg", "out_deg"], ["In-degree", "Out-degree"], strict=True
    ):
        for lab, color, name in [
            (config.LICIT, plots.LICIT, "licit"),
            (config.ILLICIT, plots.ILLICIT, "illicit"),
        ]:
            vals = np.sort(deg.loc[deg["label"] == lab, col].to_numpy())
            # Complementary CDF: share of transactions with degree >= k.
            ks = np.unique(vals[vals > 0])
            ccdf = 1.0 - np.searchsorted(vals, ks, side="left") / len(vals)
            ax.step(ks, ccdf, where="post", color=color, label=name, lw=1.6)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(f"{title} (complementary CDF)")
        ax.set_xlabel(f"{title.lower()} k")
        ax.set_ylabel("share with degree $\\geq$ k")
    axes[0].legend()
    plots.save(fig, "eda_degree")


def homophily(nodes: pd.DataFrame, edges: pd.DataFrame) -> dict:
    """Of the labelled neighbours of an illicit (licit) node, how many are illicit?"""
    lab = nodes["label"].to_numpy()
    s, d = edges["src"].to_numpy(), edges["dst"].to_numpy()
    both = (lab[s] != config.UNKNOWN) & (lab[d] != config.UNKNOWN)
    s, d = s[both], d[both]
    pairs = pd.crosstab(lab[s], lab[d])
    pairs.index = pairs.index.map({0: "licit", 1: "illicit"})
    pairs.columns = pairs.columns.map({0: "licit", 1: "illicit"})
    # Undirected view: an edge counts for both endpoints.
    a = np.concatenate([lab[s], lab[d]])
    b = np.concatenate([lab[d], lab[s]])
    ill_nb_of_ill = float((b[a == 1] == 1).mean())
    ill_nb_of_lic = float((b[a == 0] == 1).mean())
    base = float((lab[lab != config.UNKNOWN] == 1).mean())
    return {
        "labelled_edges": int(both.sum()),
        "edge_label_pairs_src_to_dst": pairs.to_dict(),
        "p_illicit_neighbour_given_illicit": round(ill_nb_of_ill, 4),
        "p_illicit_neighbour_given_licit": round(ill_nb_of_lic, 4),
        "base_rate_illicit": round(base, 4),
    }


def main() -> None:
    plots.setup()
    d = data.load()
    nodes, edges = d.nodes, d.edges
    labelled = d.labelled

    counts = label_counts_per_step(nodes)
    fig_labels_over_time(counts)

    results = {
        "rows": len(nodes),
        "labelled": len(labelled),
        "label_counts": nodes["label"]
        .map({1: "illicit", 0: "licit", -1: "unknown"})
        .value_counts()
        .to_dict(),
        "illicit_share_train": float(
            counts.loc[1:34, "illicit"].sum() / counts.loc[1:34, ["illicit", "licit"]].sum().sum()
        ),
        "illicit_share_test": float(
            counts.loc[35:49, "illicit"].sum() / counts.loc[35:49, ["illicit", "licit"]].sum().sum()
        ),
        "illicit_after_shutdown_43_49": int(counts.loc[43:49, "illicit"].sum()),
        "illicit_test_35_42": int(counts.loc[35:42, "illicit"].sum()),
        "per_step": counts.reset_index().to_dict(orient="records"),
        "correlation": fig_correlation(nodes),
        "pca": fig_pca(labelled, config.DEFAULT_SEED),
    }
    gstats, deg = graph_stats(nodes, edges)
    results["graph"] = gstats
    fig_degree(deg)
    results["homophily"] = homophily(nodes, edges)
    io.save_json(results, "e00_eda")

    summary = counts.reset_index()[["time_step", "illicit", "licit", "unknown", "illicit_share"]]
    io.save_table(summary, "eda_counts_per_step")
    print({k: v for k, v in results.items() if k not in ("per_step",)})


if __name__ == "__main__":
    main()
