import numpy as np
import pytest

import epzero


def clustered(n, dim, latent, seed=0):
    """Rows near a `latent`-dimensional subspace: most of the variance in few directions."""
    rng = np.random.default_rng(seed)
    basis = rng.normal(size=(latent, dim))
    centers = rng.normal(size=(30, latent))
    x = centers[rng.integers(0, 30, n)] + 0.4 * rng.normal(size=(n, latent))
    return (x @ basis + 0.05 * rng.normal(size=(n, dim))).astype(np.float32)


def recall(index, base, queries, k):
    truth = np.argsort(((queries[:, None, :] - base[None, :, :]) ** 2).sum(-1), axis=1)[:, :k]
    found, _ = index.search(queries, k=k)
    return np.mean([len(set(f) & set(t)) / k for f, t in zip(found, truth)])


def test_plan_keeps_short_vectors_whole():
    p = epzero.plan(clustered(2000, 64, 8))
    assert p.nav_dim == 0 and p.quantization == "sq8" and p.refine
    assert "below" in p.reasons["nav_dim"]


def test_plan_walks_on_directions_when_few_hold_the_variance():
    p = epzero.plan(clustered(3000, 512, 16))
    assert 0 < p.nav_dim <= 512 // 4 and p.nav_dim % 32 == 0
    assert p.rerank > 0 and p.rerank_z > 0 and p.fill_after


def test_plan_skips_directions_when_variance_is_spread():
    rng = np.random.default_rng(1)
    p = epzero.plan(rng.normal(size=(3000, 512)).astype(np.float32))
    assert p.nav_dim == 0 and "too many" in p.reasons["nav_dim"]


def test_settings_override_the_plan_and_are_marked():
    p = epzero.plan(clustered(2000, 64, 8), refine=False, M=24)
    assert not p.refine and p.M == 24
    assert p.reasons["refine"] == "set by you"
    with pytest.raises(TypeError, match="unknown settings: nav"):
        p.replace(nav=3)


def test_build_reaches_the_recall_target():
    base = clustered(4000, 48, 12)
    queries = clustered(100, 48, 12, seed=7)[:, :]
    index = epzero.build(base, recall=0.95, threads=4)
    assert index.plan.ef is not None and index.plan.prefetch is not None
    assert index.plan.measured["recall at calibration"] >= 0.95
    assert recall(index, base, queries, 10) >= 0.9  # held-out queries: close to the target
    assert len(index) == len(base) and 5 in index


def test_build_with_directions():
    base = clustered(3000, 512, 16)
    index = epzero.build(base, threads=4)
    assert index.plan.nav_dim > 0
    assert index.plan.measured["recall at calibration"] >= 0.95


def test_explicit_settings_skip_calibration_and_tuning():
    base = clustered(2000, 32, 8)
    index = epzero.build(base, ef=40, prefetch=(4, 1), refine=False, threads=2)
    assert index.plan.ef == 40 and index.plan.prefetch == (4, 1)
    assert "recall at calibration" not in index.plan.measured
    assert index.prefetch == (4, 1)


def test_search_arguments_override_the_plan():
    base = clustered(2000, 32, 8)
    index = epzero.build(base, threads=2)
    labels, _ = index.search(base[:3], k=5, ef=100)
    assert labels.shape == (3, 5)
    labels, _ = index.search(base[:3])
    assert labels.shape == (3, index.plan.k)


def test_save_and_load_keep_the_plan(tmp_path):
    base = clustered(2000, 32, 8)
    index = epzero.build(base, threads=2, recall=0.9)
    path = str(tmp_path / "auto.epz")
    index.save(path)
    loaded = epzero.AutoIndex.load(path)
    assert loaded.plan == index.plan
    assert np.array_equal(loaded.search(base[:20])[0], index.search(base[:20])[0])


def test_custom_labels():
    base = clustered(1000, 32, 8)
    labels = np.arange(1000, dtype=np.uint64) * 3 + 7
    index = epzero.build(base, labels=labels, threads=2)
    found, _ = index.search(base[:1], k=1)
    assert found[0, 0] == 7


def test_calibration_holds_rows_out_of_the_graph_then_adds_them():
    base = clustered(3000, 48, 12)
    index = epzero.build(base, threads=2)
    assert index.plan.measured["calibrated on"] == "500 held-out rows"
    assert len(index) == len(base)
    found, _ = index.search(base, k=1)
    assert np.mean(found[:, 0] == np.arange(len(base))) > 0.99  # every row, held out or not, is in


def test_calibration_on_your_queries():
    base = clustered(3000, 48, 12)
    queries = clustered(200, 48, 12, seed=9)
    index = epzero.build(base, queries=queries, recall=0.9, threads=2)
    assert index.plan.measured["calibrated on"] == "your queries"
    assert recall(index, base, queries, 10) >= 0.9
