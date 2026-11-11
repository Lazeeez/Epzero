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
    assert p.nav_dim == 0 and p.quantization == "sq8" and p.refine and p.fast_build
    assert "below" in p.reasons["nav_dim"]
    assert p.ef is None and p.prefetch is None  # measured when building


def test_plan_walks_on_directions_when_few_hold_the_variance():
    p = epzero.plan(clustered(3000, 512, 16))
    assert 0 < p.nav_dim <= 512 // 4 and p.nav_dim % 32 == 0
    assert p.rerank > 0 and p.rerank_z > 0 and p.fill_after


def test_plan_skips_directions_when_variance_is_spread():
    rng = np.random.default_rng(1)
    p = epzero.plan(rng.normal(size=(3000, 512)).astype(np.float32))
    assert p.nav_dim == 0 and "only" in p.reasons["nav_dim"]


def test_settings_override_the_plan_and_are_marked():
    p = epzero.plan(clustered(2000, 64, 8), refine=False, M=24, ef=50)
    assert not p.refine and p.M == 24 and p.ef == 50
    assert p.reasons["refine"] == "set by you"
    with pytest.raises(TypeError, match="unknown settings: nav"):
        p.replace(nav=3)
    with pytest.raises(TypeError, match="unknown settings"):
        epzero.plan(clustered(100, 8, 4), refin=False)


def test_build_reaches_the_recall_target():
    rows = clustered(4100, 48, 12)
    base, queries = rows[:4000], rows[4000:]  # unseen rows of the same distribution
    index = epzero.build(base, recall=0.95, threads=4)
    plan = index.plan
    assert plan.ef is not None and plan.prefetch is not None
    assert plan.measured["recall at calibration"] >= 0.95
    assert recall(index, base, queries, 10) >= 0.93  # unseen queries: close to the target
    assert len(index) == len(base) and 5 in index


def test_build_with_directions():
    base = clustered(3000, 512, 16)
    index = epzero.build(base, threads=4)
    assert index.plan.nav_dim > 0 and index.profile["rerank"] > 0
    assert index.plan.measured["recall at calibration"] >= 0.95


def test_explicit_settings_skip_calibration_and_tuning():
    base = clustered(2000, 32, 8)
    index = epzero.build(base, ef=40, prefetch=(4, 1), refine=False, threads=2)
    assert index.plan.ef == 40 and index.plan.prefetch == (4, 1) and not index.plan.refine
    assert "recall at calibration" not in index.plan.measured
    assert index.prefetch == (4, 1)


def test_a_plan_can_be_edited_and_built():
    base = clustered(2000, 32, 8)
    p = epzero.plan(base).replace(rerank=40, recall=0.9)
    index = epzero.build(base, plan=p, threads=2)
    assert index.plan.rerank == 40 and index.plan.recall == pytest.approx(0.9)
    assert index.plan.reasons["rerank"] == "set by you"


def test_search_uses_the_profile_unless_told_otherwise():
    base = clustered(2000, 32, 8)
    index = epzero.build(base, threads=2, k=5)
    labels, _ = index.search(base[:3])
    assert labels.shape == (3, 5)
    labels, _ = index.search(base[:3], k=20, ef=100)
    assert labels.shape == (3, 20)


def test_hand_built_index_keeps_the_old_defaults():
    base = clustered(500, 16, 4)
    index = epzero.Index(dim=16, max_elements=500)
    index.add(base)
    assert index.profile["ef"] == 64 and index.profile["k"] == 10 and index.profile["notes"] == {}
    assert index.search(base[:2])[0].shape == (2, 10)


def test_save_and_load_keep_the_settings(tmp_path):
    base = clustered(2000, 32, 8)
    index = epzero.build(base, threads=2, recall=0.9)
    path = str(tmp_path / "auto.epz")
    index.save(path)
    loaded = epzero.Index.load(path)
    assert loaded.plan == index.plan and loaded.plan.reasons == index.plan.reasons
    assert loaded.prefetch == index.prefetch  # the same machine
    assert np.array_equal(loaded.search(base[:20])[0], index.search(base[:20])[0])


def test_custom_labels_and_capacity():
    base = clustered(1000, 32, 8)
    labels = np.arange(1000, dtype=np.int64) * 3 + 7
    index = epzero.build(base, labels=labels, capacity=1500, threads=2)
    found, _ = index.search(base[:1], k=1)
    assert found[0, 0] == 7
    index.add(clustered(400, 32, 8, seed=3), labels=np.arange(400) + 10_000)
    assert len(index) == 1400


def test_calibration_holds_rows_out_of_the_graph_then_adds_them():
    base = clustered(3000, 48, 12)
    index = epzero.build(base, threads=2)
    assert len(index) == len(base)
    found, _ = index.search(base, k=1)
    assert np.mean(found[:, 0] == np.arange(len(base))) > 0.99  # every row, held out or not, is in


def test_calibration_on_your_queries():
    rows = clustered(3200, 48, 12)
    base, queries = rows[:3000], rows[3000:]
    index = epzero.build(base, queries=queries, recall=0.9, threads=2)
    assert "your queries" in index.plan.reasons["ef"]
    assert recall(index, base, queries, 10) >= 0.9
    with pytest.raises(ValueError):
        epzero.build(base, queries=queries[:, :10])


def test_bad_input():
    with pytest.raises(ValueError):
        epzero.build(np.zeros((0, 8), dtype=np.float32))
    with pytest.raises(ValueError):
        epzero.build(np.zeros((10, 8), dtype=np.float32), recall=1.5)
    with pytest.raises(ValueError):
        epzero.build(np.ones((10, 8), dtype=np.float32), quantization="sq9")


def test_machine_is_named():
    assert "/" in epzero.machine()


def test_explain_and_trace_use_the_profile():
    base = clustered(3000, 512, 16)
    index = epzero.build(base, threads=4)
    assert index.plan.nav_dim > 0
    labels, _ = index.search(base[:20], k=1)
    hit = int(np.sum(labels[:, 0] == np.arange(20)))
    explained = [index.explain(base[i], 1, i)["cause"] for i in range(20)]
    assert explained.count("returned") == hit  # the same search as search() runs
    assert index.trace(base[0], k=1)["results"][0][0] == labels[0, 0]


def test_recalibrate_and_health_after_growth():
    rows = clustered(8300, 32, 8)
    base, more, queries = rows[:2000], rows[2000:8000], rows[8000:]
    index = epzero.build(base, capacity=8000, threads=2)
    assert index.health()["advice"] == []
    index.add(more, labels=np.arange(2000, 8000))
    health = index.health()
    assert health["growth"] == pytest.approx(3.0) and "recalibrate" in health["advice"][0]
    result = index.recalibrate(queries, threads=2)
    assert result["recall"] >= 0.95 and index.profile["ef"] == result["ef"]
    assert index.plan.reasons["ef"].startswith("recalibrated") and index.health()["advice"] == []
    assert recall(index, np.vstack([base, more]), queries, 10) >= 0.93


def test_recall_guard_starts_from_the_profile():
    base = clustered(3000, 32, 8)
    index = epzero.build(base, recall=0.9, threads=2)
    guard = epzero.RecallGuard(index, sample_rate=1.0, min_samples=10**9)
    guard.search(base[:5], k=10)
    guard.drain()
    report = guard.report()
    assert report[0]["ef"] == index.profile["ef"]


def test_get_vector():
    base = clustered(100, 16, 4)
    index = epzero.build(base, threads=1)
    assert np.allclose(index.get_vector(5), base[5], atol=1e-6)
    with pytest.raises(KeyError):
        index.get_vector(1000)
