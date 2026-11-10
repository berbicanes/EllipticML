"""Graph structure of the transaction network.

Everything here uses the edge list only, never the labels, so these features
can be computed for test steps without leaking anything. The dataset's own 72
aggregated features already summarise the one-hop neighbourhood's *features*;
what they do not describe is the *shape* of the neighbourhood (how many
inputs and outputs, how central the transaction is, whether its neighbours are
hubs). That is what this module adds.

Every time step is its own connected graph (no edge crosses steps, see
tests/test_data.py), so whole-graph computations such as PageRank never mix
information across time.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.sparse as sp

GRAPH_COLS = [
    "g_in_deg",
    "g_out_deg",
    "g_in_2hop",
    "g_out_2hop",
    "g_nb_mean_deg",
    "g_nb_max_deg",
    "g_clustering",
    "g_pagerank",
    "g_pagerank_rev",
]


def adjacency(edges: pd.DataFrame, n: int) -> sp.csr_matrix:
    """Directed adjacency, A[i, j] = 1 for a flow i -> j."""
    a = sp.csr_matrix(
        (np.ones(len(edges), dtype=np.float32), (edges["src"], edges["dst"])), shape=(n, n)
    )
    a.data[:] = 1.0  # collapse any duplicate edges
    return a


def undirected(a: sp.csr_matrix) -> sp.csr_matrix:
    u = (a + a.T).tocsr()
    u.data[:] = 1.0
    return u


def normalized_adjacency(edges: pd.DataFrame, n: int) -> sp.csr_matrix:
    """Symmetric GCN propagation matrix  D^-1/2 (A + A^T + I) D^-1/2  (Kipf & Welling)."""
    a = undirected(adjacency(edges, n)) + sp.identity(n, dtype=np.float32, format="csr")
    deg = np.asarray(a.sum(1)).ravel()
    d_inv_sqrt = sp.diags(1.0 / np.sqrt(deg))
    return (d_inv_sqrt @ a @ d_inv_sqrt).tocsr().astype(np.float32)


def pagerank(a: sp.csr_matrix, alpha: float = 0.85, tol: float = 1e-10, max_iter: int = 200):
    """Power-iteration PageRank on a directed graph; dangling mass spread uniformly."""
    n = a.shape[0]
    out_deg = np.asarray(a.sum(1)).ravel()
    inv = np.divide(1.0, out_deg, out=np.zeros(n), where=out_deg > 0)
    p = (sp.diags(inv) @ a).T.tocsr()  # column-stochastic transition matrix
    dangling = out_deg == 0
    r = np.full(n, 1.0 / n)
    for _ in range(max_iter):
        r_new = alpha * (p @ r + r[dangling].sum() / n) + (1 - alpha) / n
        if np.abs(r_new - r).sum() < tol:
            return r_new
        r = r_new
    return r


def structural_features(edges: pd.DataFrame, n: int) -> pd.DataFrame:
    """Nine label-free structural features per transaction (row order = node index)."""
    a = adjacency(edges, n)
    at = a.T.tocsr()
    u = undirected(a)

    in_deg = np.asarray(a.sum(0)).ravel()
    out_deg = np.asarray(a.sum(1)).ravel()
    deg = np.asarray(u.sum(1)).ravel()

    # Number of distinct transactions reachable in exactly <= 2 hops (excluding self).
    def two_hop(m: sp.csr_matrix) -> np.ndarray:
        reach = (m + m @ m).tocsr()
        reach.setdiag(0)
        reach.eliminate_zeros()
        return np.diff(reach.indptr)

    nb_sum = u @ deg
    nb_mean = np.divide(nb_sum, deg, out=np.zeros(n), where=deg > 0)
    nb_max = np.zeros(n)
    rows = np.repeat(np.arange(n), np.diff(u.indptr))
    np.maximum.at(nb_max, rows, deg[u.indices])

    # Local clustering coefficient on the undirected graph.
    triangles = np.asarray((u @ u).multiply(u).sum(1)).ravel() / 2.0
    possible = deg * (deg - 1) / 2.0
    clustering = np.divide(triangles, possible, out=np.zeros(n), where=possible > 0)

    # PageRank scaled by n so values are comparable across graph sizes (mean 1).
    pr = pagerank(a) * n
    pr_rev = pagerank(at) * n

    return pd.DataFrame(
        {
            "g_in_deg": in_deg,
            "g_out_deg": out_deg,
            "g_in_2hop": two_hop(at),
            "g_out_2hop": two_hop(a),
            "g_nb_mean_deg": nb_mean,
            "g_nb_max_deg": nb_max,
            "g_clustering": clustering,
            "g_pagerank": pr,
            "g_pagerank_rev": pr_rev,
        }
    ).astype(np.float32)[GRAPH_COLS]
