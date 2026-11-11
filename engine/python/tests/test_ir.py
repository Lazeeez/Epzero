import json
import math

import numpy as np
import pytest

from epzero import ir


def clustered(n, dim=32, seed=0):
    rng = np.random.default_rng(seed)
    centers = rng.normal(size=(20, dim))
    return (centers[rng.integers(0, 20, n)] + 0.3 * rng.normal(size=(n, dim))).astype(np.float32)


WORDS = ["alpha", "beta", "gamma", "delta", "omega", "sigma", "kappa", "theta"]


@pytest.fixture(scope="module")
def collection():
    vectors = clustered(2000)
    ids = [f"doc-{i}" for i in range(len(vectors))]
    texts = [f"{WORDS[i % 8]} {WORDS[(i // 8) % 8]} item{i}" for i in range(len(vectors))]
    return ir.Collection.build(ids, vectors, texts=texts, metric="l2", seed=1), vectors


def test_dense_retrieval_finds_the_document_itself(collection):
    c, vectors = collection
    run = c.retrieve(["q1", "q2"], vectors=vectors[[5, 700]], k=10)
    assert run["q1"][0][0] == "doc-5" and run["q2"][0][0] == "doc-700"
    assert len(run["q1"]) == 10
    scores = [s for _, s in run["q1"]]
    assert scores == sorted(scores, reverse=True)  # higher is better, as TREC expects


def test_keyword_and_hybrid(collection):
    c, vectors = collection
    keyword = c.retrieve(["q"], texts=["item123"], k=5, mode="keyword")
    assert keyword["q"][0][0] == "doc-123"
    hybrid = c.retrieve(["q"], vectors=vectors[[123]], texts=["item123"], k=5, mode="hybrid")
    assert hybrid["q"][0][0] == "doc-123"
    with pytest.raises(ValueError, match="query texts"):
        c.retrieve(["q"], vectors=vectors[[1]], mode="hybrid")
    with pytest.raises(ValueError, match="mode"):
        c.retrieve(["q"], vectors=vectors[[1]], mode="sparse")


def test_add_replace_remove_and_save(tmp_path):
    everything = clustered(350, seed=2)
    vectors, new = everything[:300], everything[300:]
    c = ir.Collection.build([f"d{i}" for i in range(300)], vectors, texts=["t"] * 300, metric="l2")
    c.add([f"n{i}" for i in range(50)], new, texts=["fresh"] * 50)
    assert len(c) == 350 and "n7" in c
    assert c.retrieve(["q"], vectors=new[[7]], k=1)["q"][0][0] == "n7"
    c.add(["d0"], new[[7]] + 0.001, texts=["moved"])  # replaces d0
    assert len(c) == 350
    assert {d for d, _ in c.retrieve(["q"], vectors=new[[7]], k=2)["q"]} == {"n7", "d0"}
    assert c.remove(["n7", "absent"]) == 1 and "n7" not in c
    assert all(d != "n7" for d, _ in c.retrieve(["q"], vectors=new[[7]], k=20)["q"])
    c.save(tmp_path / "c")
    loaded = ir.Collection.load(tmp_path / "c")
    assert len(loaded) == 349
    assert loaded.retrieve(["q"], vectors=new[[7]], k=1) == c.retrieve(["q"], vectors=new[[7]], k=1)
    with pytest.raises(ValueError, match="unique"):
        ir.Collection.build(["a", "a"], vectors[:2])


def test_run_files_round_trip(tmp_path):
    run = {"1": [("a", 3.0), ("b", 2.5)], "2": [("c", -0.1)]}
    ir.write_run(run, tmp_path / "run.txt", tag="test")
    lines = (tmp_path / "run.txt").read_text().splitlines()
    assert lines[0] == "1 Q0 a 1 3.000000 test"
    assert ir.read_run(tmp_path / "run.txt") == run
    with pytest.raises(ValueError):
        ir.write_run(run, tmp_path / "x.txt", tag="two words")


def test_qrels_formats(tmp_path):
    (tmp_path / "trec.txt").write_text("1 0 a 2\n1 0 b 0\n2 0 c 1\n")
    (tmp_path / "beir.tsv").write_text("query-id\tcorpus-id\tscore\n1\ta\t2\n1\tb\t0\n2\tc\t1\n")
    expected = {"1": {"a": 2, "b": 0}, "2": {"c": 1}}
    assert ir.read_qrels(tmp_path / "trec.txt") == expected == ir.read_qrels(tmp_path / "beir.tsv")


def test_measures_by_hand():
    qrels = {"q": {"a": 2, "b": 1, "c": 0}, "none": {"x": 0}}
    run = {"q": [("z", 3), ("a", 2), ("c", 1), ("b", 0)]}
    m = ir.evaluate(run, qrels, ["nDCG@10", "MAP", "MRR@10", "R@2", "P@2"])
    idcg = 2 + 1 / math.log2(3)
    assert m["nDCG@10"] == pytest.approx((2 / math.log2(3) + 1 / math.log2(5)) / idcg)
    assert m["MAP"] == pytest.approx((1 / 2 + 2 / 4) / 2)
    assert m["MRR@10"] == pytest.approx(0.5)
    assert m["R@2"] == pytest.approx(0.5) and m["P@2"] == pytest.approx(0.5)
    # A judged query missing from the run scores 0; one with nothing relevant is not counted.
    assert ir.evaluate({}, qrels)["nDCG@10"] == 0.0
    assert set(ir.evaluate(run, qrels, per_query=True)["MAP"]) == {"q"}


def test_measures_match_trec_eval():
    pytrec_eval = pytest.importorskip("pytrec_eval")
    rng = np.random.default_rng(4)
    docs = [f"d{i}" for i in range(60)]
    qrels = {f"q{i}": {d: int(rng.integers(0, 3)) for d in rng.choice(docs, 15, replace=False)} for i in range(30)}
    qrels = {q: j for q, j in qrels.items() if any(r > 0 for r in j.values())}
    run = {}
    for q in qrels:
        ranked = rng.choice(docs, 40, replace=False)
        run[q] = [(d, float((40 - i) // 3)) for i, d in enumerate(ranked)]  # ties, broken as trec_eval does
    ours = ir.evaluate(run, qrels, ["nDCG@10", "MAP", "R@20", "P@5"], per_query=True)
    theirs = pytrec_eval.RelevanceEvaluator(qrels, {"ndcg_cut.10", "map", "recall.20", "P.5"}).evaluate(
        {q: dict(r) for q, r in run.items()})
    for q in qrels:
        assert ours["nDCG@10"][q] == pytest.approx(theirs[q]["ndcg_cut_10"])
        assert ours["MAP"][q] == pytest.approx(theirs[q]["map"])
        assert ours["R@20"][q] == pytest.approx(theirs[q]["recall_20"])
        assert ours["P@5"][q] == pytest.approx(theirs[q]["P_5"])


def test_load_beir(tmp_path):
    (tmp_path / "qrels").mkdir()
    (tmp_path / "corpus.jsonl").write_text(
        "\n".join(json.dumps(d) for d in [{"_id": "1", "title": "T", "text": "body"}, {"_id": 2, "text": "two"}]) + "\n")
    (tmp_path / "queries.jsonl").write_text(
        "\n".join(json.dumps(q) for q in [{"_id": "a", "text": "first"}, {"_id": "b", "text": "unjudged"}]) + "\n")
    (tmp_path / "qrels" / "test.tsv").write_text("query-id\tcorpus-id\tscore\na\t1\t1\n")
    corpus, queries, qrels = ir.load_beir(tmp_path)
    assert corpus == {"1": "T body", "2": "two"}
    assert queries == {"a": "first"} and qrels == {"a": {"1": 1}}
