import numpy as np
import pytest

import epzero


@pytest.fixture
def meta():
    m = epzero.Metadata()
    m.set_many([1, 2, 3, 4], [
        {"source": "faq", "year": 2021, "public": True},
        {"source": "blog", "year": 2024, "score": 0.5},
        {"source": "faq", "year": 2024.0, "public": False},
        {"source": "docs", "score": 0.9},
    ])
    return m


def test_where_syntax(meta):
    assert meta.select({"source": "faq"}).tolist() == [1, 3]
    assert meta.select({"source": ["faq", "docs"]}).tolist() == [1, 3, 4]
    assert meta.select({"year": 2024}).tolist() == [2, 3]  # 2024 and 2024.0 are the same number
    assert meta.select({"source": "faq", "year": {"$gte": 2022}}).tolist() == [3]
    assert meta.select({"year": {"$gt": 2021, "$lt": 2030}}).tolist() == [2, 3]
    assert meta.select({"source": {"$ne": "faq"}}).tolist() == [2, 4]
    assert meta.select({"source": {"$nin": ["faq", "blog"]}}).tolist() == [4]
    assert meta.select({"score": {"$exists": True}}).tolist() == [2, 4]
    assert meta.select({"public": True}).tolist() == [1]  # a bool is not the integer 1 here
    assert meta.select({}).tolist() == [1, 2, 3, 4]
    assert meta.select({"nothing": "x"}).tolist() == []


def test_bad_filters(meta):
    with pytest.raises(ValueError, match="unknown filter operator"):
        meta.select({"year": {"$approx": 3}})
    with pytest.raises(ValueError, match="needs a number"):
        meta.select({"year": {"$gt": "2020"}})
    with pytest.raises(ValueError):
        meta.set(9, {"tags": ["a", "b"]})


def test_get_remove_save_load(meta, tmp_path):
    assert meta.get(2) == {"source": "blog", "year": 2024, "score": 0.5}
    assert meta.remove(2) and not meta.remove(2) and 2 not in meta and len(meta) == 3
    path = str(tmp_path / "meta.bin")
    meta.save(path)
    loaded = epzero.Metadata.load(path)
    assert loaded.select({"year": {"$gte": 2022}}).tolist() == [3]
    assert loaded.get(1) == {"source": "faq", "year": 2021, "public": True}


def test_filtered_search_matches_brute_force():
    rng = np.random.default_rng(0)
    base = rng.normal(size=(5000, 32)).astype(np.float32)
    queries = rng.normal(size=(20, 32)).astype(np.float32)
    rows = [{"shop": f"s{i % 7}", "price": float(i % 100)} for i in range(5000)]
    meta = epzero.Metadata()
    meta.set_many(None, rows)
    index = epzero.Index(dim=32, max_elements=5000)
    index.add(base, threads=4)
    allowed = meta.select({"shop": ["s1", "s3"], "price": {"$lt": 40}})
    expected = [i for i, r in enumerate(rows) if r["shop"] in ("s1", "s3") and r["price"] < 40]
    assert allowed.tolist() == expected
    labels, _ = index.search(queries, k=10, ef=200, filter=index.filter(allowed))
    truth = np.array(expected)[np.argsort(((queries[:, None, :] - base[expected][None]) ** 2).sum(-1), axis=1)[:, :10]]
    recall = np.mean([len(set(l) & set(t)) / 10 for l, t in zip(labels, truth)])
    assert recall >= 0.95 and set(labels.ravel().tolist()) <= set(expected)
