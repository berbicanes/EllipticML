"""The GNNs must learn a label that is only visible through the graph."""

import numpy as np
import pandas as pd
import pytest

from elliptic import gnn


def toy_graph(n_pairs: int = 200, seed: int = 0, noisy_receivers: bool = False):
    """Pairs u -> v. The label of v equals the sign of u's first feature.

    v's own features are zero, or noise with ``noisy_receivers``. A model that
    ignores edges cannot beat chance on v; a graph model can.
    """
    rng = np.random.default_rng(seed)
    n = 2 * n_pairs
    x = rng.normal(size=(n, 4)).astype(np.float32)
    src = np.arange(0, n, 2)
    dst = src + 1
    if not noisy_receivers:
        x[dst] = 0.0
    y = np.zeros(n, dtype=np.int64)
    y[dst] = (x[src, 0] > 0).astype(int)
    nodes = pd.DataFrame(x, columns=[f"f{i}" for i in range(4)])
    edges = pd.DataFrame({"src": src, "dst": dst})
    return nodes, edges, y, dst


def fit_toy(kind: str, noisy: bool) -> float:
    nodes, edges, y, dst = toy_graph(noisy_receivers=noisy)
    mask = np.zeros(len(nodes), dtype=bool)
    mask[dst] = True
    g = gnn.build_tensors(nodes, edges, list(nodes.columns), np.ones(len(nodes), dtype=bool))
    cfg = gnn.TrainConfig(kind=kind, hidden=16, lr=0.02, epochs=200, illicit_weight=0.5)
    model, _ = gnn.train(g, y, mask, cfg, seed=0)
    p = gnn.predict(model, g)
    return ((p[dst] > 0.5) == y[dst]).mean()


@pytest.mark.parametrize("kind", ["gcn", "skipgcn", "dirsage"])
def test_gnn_uses_the_graph(kind):
    assert fit_toy(kind, noisy=False) > 0.9


def test_only_dirsage_separates_own_and_neighbour_features():
    # GCN averages a node with its neighbours under one weight matrix, so the
    # receiver's own noise is mixed into the sender's signal. DirSAGE keeps
    # separate weights for self and for senders and recovers the label.
    assert fit_toy("dirsage", noisy=True) > 0.9
    assert fit_toy("gcn", noisy=True) < 0.9


def test_mean_operator_rows_sum_to_one_or_zero():
    import scipy.sparse as sp

    a = sp.csr_matrix(np.array([[0, 1, 1], [0, 0, 0], [1, 0, 0]], dtype=np.float32))
    m = gnn.mean_operator(a).toarray()
    assert np.allclose(m.sum(1), [1, 0, 1])
