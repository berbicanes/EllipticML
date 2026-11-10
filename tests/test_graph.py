"""Graph features on a tiny hand-checked graph.

0 -> 1 -> 2
0 -> 2        (triangle 0-1-2 when undirected)
3 -> 2
4             (isolated)
"""

import numpy as np
import pandas as pd
import pytest

from elliptic import graph

EDGES = pd.DataFrame({"src": [0, 1, 0, 3], "dst": [1, 2, 2, 2]})
N = 5


@pytest.fixture(scope="module")
def g():
    return graph.structural_features(EDGES, N)


def test_degrees(g):
    assert g["g_in_deg"].tolist() == [0, 1, 3, 0, 0]
    assert g["g_out_deg"].tolist() == [2, 1, 0, 1, 0]


def test_two_hop(g):
    # Out-reach of 0 is {1, 2}; in-reach of 2 is {0, 1, 3}.
    assert g["g_out_2hop"].tolist() == [2, 1, 0, 1, 0]
    assert g["g_in_2hop"].tolist() == [0, 1, 3, 0, 0]


def test_clustering(g):
    # Node 0 has neighbours {1, 2}, which are linked -> 1.0. Node 2 has {0, 1, 3}:
    # one link (0-1) of three possible -> 1/3. Node 3 has one neighbour -> 0.
    assert g["g_clustering"].tolist() == pytest.approx([1.0, 1.0, 1 / 3, 0.0, 0.0])


def test_neighbour_degree(g):
    # Undirected degrees: [2, 2, 3, 1, 0]. Node 3's only neighbour is node 2.
    assert g.loc[3, "g_nb_mean_deg"] == pytest.approx(3.0)
    assert g.loc[2, "g_nb_max_deg"] == 2.0


def test_pagerank_is_a_distribution():
    a = graph.adjacency(EDGES, N)
    pr = graph.pagerank(a)
    assert pr.sum() == pytest.approx(1.0)
    assert pr.argmax() == 2  # the sink everything flows into


def test_normalized_adjacency_is_symmetric():
    m = graph.normalized_adjacency(EDGES, N).toarray()
    assert np.allclose(m, m.T)
    # Isolated node keeps only its self loop with weight 1.
    assert m[4, 4] == pytest.approx(1.0)
