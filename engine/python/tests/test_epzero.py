import numpy as np
import pytest

import epzero as epzero


@pytest.fixture(scope="module")
def data():
    rng = np.random.default_rng(0)
    centers = rng.normal(size=(20, 24)).astype(np.float32)
    base = centers[rng.integers(0, 20, 3000)] + 0.3 * rng.normal(size=(3000, 24)).astype(np.float32)
    queries = centers[rng.integers(0, 20, 50)] + 0.3 * rng.normal(size=(50, 24)).astype(np.float32)
    return base.astype(np.float32), queries.astype(np.float32)


def exact_knn(base, queries, k):
    d = ((queries[:, None, :] - base[None, :, :]) ** 2).sum(-1)
    return np.argsort(d, axis=1)[:, :k]


def test_search_matches_exact_neighbors(data):
    base, queries = data
    index = epzero.Index(dim=24, max_elements=len(base))
    index.add(base, threads=4)
    labels, distances = index.search(queries, k=10, ef=100, threads=2)
    assert labels.shape == (50, 10) and distances.shape == (50, 10)
    truth = exact_knn(base, queries, 10)
    recall = np.mean([len(set(l) & set(t)) / 10 for l, t in zip(labels, truth)])
    assert recall >= 0.95
    assert np.all(np.diff(distances, axis=1) >= 0)


def test_filtered_search_returns_only_allowed_labels(data):
    base, queries = data
    index = epzero.Index(dim=24, max_elements=len(base))
    index.add(base, threads=4)
    allowed = np.arange(0, len(base), 7)  # every 7th vector
    keep = np.zeros(len(base), dtype=bool)
    keep[allowed] = True
    # Exact top-10 among the allowed vectors only.
    truth = allowed[exact_knn(base[allowed], queries, 10)]

    compiled = index.filter(allowed)
    assert compiled.allowed == len(allowed)
    assert abs(compiled.selectivity - len(allowed) / len(base)) < 1e-9
    assert "epzero.Filter" in repr(compiled)
    for flt in (compiled, allowed, list(allowed)):  # compiled once, or compiled per call
        for strategy in ("auto", "exact", "inline", "two-hop"):
            labels, _ = index.search(queries, k=10, ef=100, filter=flt, strategy=strategy)
            assert keep[labels].all(), strategy
            recall = np.mean([len(set(l) & set(t)) / 10 for l, t in zip(labels, truth)])
            assert recall >= (1.0 if strategy == "exact" else 0.95), strategy

    labels, distances = index.search(queries[:1], k=10, filter=[3, 5])  # fewer than k allowed
    assert set(labels[0][:2]) == {3, 5} and (labels[0][2:] == -1).all() and np.isinf(distances[0][2:]).all()
    with pytest.raises(ValueError):
        index.search(queries, k=10, filter=allowed, strategy="sideways")


def test_exact_search_explain_and_recall_guard(data):
    base, queries = data
    index = epzero.Index(dim=24, max_elements=len(base), M=8, ef_construction=40)
    index.add(base, threads=4)

    labels, _ = index.exact_search(queries, k=10, threads=2)
    assert (labels == exact_knn(base, queries, 10)).all()

    # Explain: a returned result, a far-away vector, and a missed neighbour at a tiny beam.
    q = queries[0]
    top, _ = index.search(q, k=10, ef=64)
    assert index.explain(q, 10, int(top[0][0]), ef=64)["cause"] == "returned"
    far = int(index.exact_search(q, k=len(base))[0][0][-1])
    why = index.explain(q, 10, far, ef=64)
    assert why["cause"] == "not-near-enough" and why["exact_rank"] == len(base)
    assert index.explain(q, 10, 10**9)["cause"] == "not-in-index"
    assert index.explain(q, 10, int(top[0][0]), filter=[1, 2])["cause"] in ("filtered-out", "returned")

    # A guard starting below its target raises the beam until measured recall reaches it.
    guard = epzero.RecallGuard(index, target=0.97, initial_ef=10, min_ef=10, min_samples=40, sample_rate=1.0, window=200)
    for _ in range(30):
        guard.search(queries, k=10, threads=2)
        guard.drain()
    (row,) = guard.report()
    assert row["segment"] == "unfiltered" and row["ef"] > 10 and row["changes"] > 0
    assert 0 <= row["lower"] <= row["recall"] <= row["upper"] <= 1
    del index  # the guard keeps the index alive
    assert guard.search(queries[:1], k=10)[0].shape == (1, 10)


def test_patience_prefetch_and_per_query_guard(data):
    base, queries = data
    index = epzero.Index(dim=24, max_elements=len(base))
    index.add(base, threads=4)
    plain, _ = index.search(queries, k=10, ef=200)
    # A patience longer than any search changes nothing; a short one still finds the nearest.
    same, _ = index.search(queries, k=10, ef=200, patience=10**6)
    assert (same == plain).all()
    quick, _ = index.search(queries, k=10, ef=200, patience=8)
    assert np.mean(quick[:, 0] == plain[:, 0]) >= 0.9

    # Prefetch settings change speed, never results.
    assert index.prefetch == (None, 1)
    index.set_prefetch(4, 2)
    assert index.prefetch == (4, 2)
    assert (index.search(queries, k=10, ef=200)[0] == plain).all()
    depth, lines = index.tune_prefetch(base[:200])
    assert lines >= 1 and (depth is None or depth >= 1)
    assert (index.search(queries, k=10, ef=200)[0] == plain).all()

    guard = epzero.RecallGuard(index, target=0.9, per_query=True, sample_rate=1.0, min_samples=20, window=100)
    for _ in range(5):
        guard.search(queries, k=10)
        guard.drain()
    (row,) = guard.report()
    assert row["segment"] == "unfiltered" and row["samples"] > 0


def test_text_index_and_hybrid_search(tmp_path, data):
    base, queries = data
    text = epzero.TextIndex()
    text.add([10, 11, 12], ["the red car", "a red red barn", "blue sky"])
    labels, scores = text.search("red", k=5)
    assert list(labels) == [11, 10] and scores[0] > scores[1] > 0
    assert len(text) == 3 and 12 in text and epzero.TextIndex.tokenize("Red-Car!") == ["red", "car"]
    assert text.remove(12) and not text.remove(12)
    text.save(str(tmp_path / "text.idx"))
    again = epzero.TextIndex.load(str(tmp_path / "text.idx"))
    assert list(again.search("red", 5)[0]) == [11, 10]

    # Hybrid: the vector side alone, then a keyword-only match pulled into the results.
    index = epzero.Index(dim=24, max_elements=len(base))
    index.add(base, threads=4)
    words = epzero.TextIndex()
    words.add(None, [f"document {i}" for i in range(len(base))])
    nearest, _ = index.search(queries[0], k=10, ef=100)
    plain, _ = epzero.hybrid_search(index, words, queries[0], "nothing matches this", k=10)
    assert list(plain) == list(nearest[0])
    far = int(index.exact_search(queries[0], k=len(base))[0][0][-1])
    words.add([far], ["invoice INV-7731 overdue"])
    found, _ = epzero.hybrid_search(index, words, queries[0], "INV-7731", k=10)
    assert far in found

    fused, fused_scores = epzero.fuse_rankings([[1, 2, 3], [3, 4]], k=2)
    assert list(fused) == [3, 1] and fused_scores[0] > fused_scores[1]


def test_custom_labels_and_missing_results(data):
    base, _ = data
    index = epzero.Index(dim=24, max_elements=10)
    index.add(base[:3], labels=np.array([100, 200, 300]))
    labels, distances = index.search(base[0], k=5)
    assert labels[0, 0] == 100
    assert list(labels[0, 3:]) == [-1, -1] and np.isinf(distances[0, 3:]).all()


def test_trace_describes_the_path(data):
    base, queries = data
    index = epzero.Index(dim=24, max_elements=len(base))
    index.add(base, threads=4)
    trace = index.trace(queries[0], k=5, ef=32)
    assert trace["entry_layer"] == index.max_level
    assert trace["distance_computations"] == len(trace["events"]) + 2
    layer, frm, to, dist, accepted = trace["events"][-1]
    assert layer == 0
    assert [r[0] for r in trace["results"]] == list(index.search(queries[0], k=5, ef=32)[0][0])


def test_save_load_round_trip(tmp_path, data):
    base, queries = data
    index = epzero.Index(dim=24, metric="cosine", max_elements=len(base))
    index.add(base)
    path = str(tmp_path / "index.epzero")
    index.save(path)
    loaded = epzero.Index.load(path)
    assert len(loaded) == len(index) and loaded.dim == 24
    np.testing.assert_array_equal(index.search(queries, k=5)[0], loaded.search(queries, k=5)[0])


def test_sq8_matches_float_recall(data):
    base, queries = data
    index = epzero.Index(dim=24, max_elements=len(base), quantization="sq8")
    with pytest.raises(Exception, match="trained"):
        index.add(base[:1])
    index.train(base)
    index.add(base, threads=4)
    labels, _ = index.search(queries, k=10, ef=100)
    truth = exact_knn(base, queries, 10)
    assert np.mean([len(set(l) & set(t)) / 10 for l, t in zip(labels, truth)]) >= 0.95


def test_rabitq_matches_float_recall(data, tmp_path):
    base, queries = data
    index = epzero.Index(dim=24, max_elements=len(base), quantization="rabitq")
    with pytest.raises(Exception, match="trained"):
        index.add(base[:1])
    index.train(base)
    index.add(base, threads=4)
    labels, distances = index.search(queries, k=10, ef=100)
    truth = exact_knn(base, queries, 10)
    assert np.mean([len(set(l) & set(t)) / 10 for l, t in zip(labels, truth)]) >= 0.95
    assert np.all(np.diff(distances, axis=1) >= 0)  # exact distances, in order
    index.save(str(tmp_path / "rabitq.epzero"))
    again = epzero.Index.load(str(tmp_path / "rabitq.epzero"))
    assert np.array_equal(again.search(queries, k=10, ef=100)[0], labels)
    with pytest.raises(Exception, match="full vectors"):
        epzero.Index(dim=24, quantization="rabitq", keep_vectors=False)


def test_remove_and_compact(data):
    base, queries = data
    index = epzero.Index(dim=24, max_elements=len(base))
    index.add(base, threads=4)
    for label in range(0, len(base), 2):
        assert index.remove(label)
    assert not index.remove(0)
    assert 0 not in index and 1 in index
    assert len(index) == len(base) // 2 and index.deleted_count == len(base) // 2
    labels, _ = index.search(queries, k=10, ef=64)
    assert np.all(labels % 2 == 1)
    compacted = index.compact(threads=4)
    assert len(compacted) == len(base) // 2 and compacted.deleted_count == 0


def test_database_recovers_after_reopen(tmp_path, data):
    base, _ = data
    path = str(tmp_path / "db")
    db = epzero.Database.open(path, dim=24, max_elements=5000, sync="batch")
    db.upsert(base[:1000], threads=4)
    db.remove(5)
    db.flush()
    del db  # no checkpoint: recovery must replay the write-ahead log

    db = epzero.Database.open(path, dim=24, max_elements=5000)
    assert db.recovery()["replayed"] == 1001
    assert len(db) == 999 and 5 not in db and 6 in db
    labels, distances = db.search(base[6], k=1)
    assert labels[0, 0] == 6 and distances[0, 0] == pytest.approx(0.0, abs=1e-5)
    db.checkpoint()
    assert epzero.Database.open(path, dim=24).recovery()["loaded_snapshot"]


def test_errors_are_python_exceptions(data):
    base, _ = data
    index = epzero.Index(dim=24, max_elements=2)
    with pytest.raises(ValueError):
        index.add(np.zeros((2, 7), dtype=np.float32))
    index.add(base[:2])
    with pytest.raises(Exception, match="full"):
        index.add(base[2:3])
    with pytest.raises(ValueError):
        epzero.Index(dim=24, metric="manhattan")
