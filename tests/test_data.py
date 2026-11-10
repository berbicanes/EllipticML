"""Checks on the loaded dataset against the numbers published with it."""

import numpy as np
import pytest

from elliptic import config, data


@pytest.fixture(scope="module")
def d():
    return data.load()


def test_shapes(d):
    assert len(d.nodes) == 203_769
    assert len(d.edges) == 234_355
    assert len(data.FEATURE_COLS) == 165


def test_label_counts(d):
    counts = d.nodes["label"].value_counts().to_dict()
    assert counts[config.ILLICIT] == 4_545
    assert counts[config.LICIT] == 42_019
    assert counts[config.UNKNOWN] == 157_205


def test_time_steps(d):
    assert d.nodes["time_step"].min() == 1
    assert d.nodes["time_step"].max() == 49
    assert d.nodes["time_step"].nunique() == 49


def test_no_missing_features(d):
    assert not d.nodes[data.FEATURE_COLS].isna().any().any()


def test_node_index_is_row_position(d):
    assert np.array_equal(d.nodes["node"].to_numpy(), np.arange(len(d.nodes)))


def test_edges_never_cross_time_steps(d):
    # Each time step is a separate graph; the GNN and graph features rely on it.
    steps = d.nodes["time_step"].to_numpy()
    assert (steps[d.edges["src"]] == steps[d.edges["dst"]]).all()


def test_feature_sets():
    assert len(data.feature_columns("LF")) == 93
    assert len(data.feature_columns("AF")) == 165
    assert data.feature_columns("LF", include_time_step=True)[0] == "time_step"
    with pytest.raises(ValueError):
        data.feature_columns("XYZ")
