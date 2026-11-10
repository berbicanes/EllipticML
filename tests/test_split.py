"""Leakage guards: no test step and no future step may reach training."""

import pytest

from elliptic import config, data, split


@pytest.fixture(scope="module")
def nodes():
    return data.load().nodes


def test_temporal_split_counts(nodes):
    s = split.temporal_split(nodes)
    # 29,894 labelled rows in steps 1-34 and 16,670 in steps 35-49.
    assert s.train.sum() == 29_894
    assert s.test.sum() == 16_670


def test_temporal_split_has_no_overlap_and_no_unknowns(nodes):
    s = split.temporal_split(nodes)
    assert not (s.train & s.test).any()
    assert (nodes.loc[s.train, "label"] != config.UNKNOWN).all()
    assert (nodes.loc[s.test, "label"] != config.UNKNOWN).all()
    assert nodes.loc[s.train, "time_step"].max() == 34
    assert nodes.loc[s.test, "time_step"].min() == 35


def test_split_rejects_future_training(nodes):
    with pytest.raises(ValueError):
        split.temporal_split(nodes, train_steps=range(1, 40), test_steps=range(35, 50))


def test_forward_folds_never_look_ahead(nodes):
    folds = list(split.forward_folds(nodes))
    assert len(folds) == 5
    for f in folds:
        train_max = nodes.loc[f.train, "time_step"].max()
        val_min = nodes.loc[f.test, "time_step"].min()
        assert train_max < val_min
        assert nodes.loc[f.test, "time_step"].max() <= 34


def test_fold_blocks():
    assert split.fold_blocks() == [(20, 22), (23, 25), (26, 28), (29, 31), (32, 34)]
