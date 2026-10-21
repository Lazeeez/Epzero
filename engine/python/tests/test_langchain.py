import numpy as np
import pytest

pytest.importorskip("langchain_core")
from langchain_core.embeddings import DeterministicFakeEmbedding  # noqa: E402

from epzero.langchain import EpzeroVectorStore  # noqa: E402

TEXTS = [f"document number {i} about {topic}" for i, topic in enumerate(["cats", "dogs", "boats", "trains"] * 10)]
METAS = [{"topic": t.split()[-1], "n": i} for i, t in enumerate(TEXTS)]


@pytest.fixture
def store():
    return EpzeroVectorStore.from_texts(TEXTS, DeterministicFakeEmbedding(size=32), metadatas=METAS, capacity=16)


def test_finds_a_document_by_its_own_text(store):
    for i in (0, 7, 39):
        assert store.similarity_search(TEXTS[i], k=1)[0].page_content == TEXTS[i]
    assert len(store) == len(TEXTS)  # grew past the initial capacity of 16


def test_metadata_filters(store):
    dogs = store.similarity_search(TEXTS[0], k=5, filter={"topic": "dogs"})
    assert len(dogs) == 5 and all(d.metadata["topic"] == "dogs" for d in dogs)
    pets = store.similarity_search(TEXTS[0], k=40, filter={"topic": ["cats", "dogs"]})
    assert {d.metadata["topic"] for d in pets} == {"cats", "dogs"} and len(pets) == 20
    small = store.similarity_search(TEXTS[0], k=40, filter=lambda m: m["n"] < 3)
    assert sorted(d.metadata["n"] for d in small) == [0, 1, 2]
    assert store.similarity_search(TEXTS[0], k=3, filter={"topic": "planes"}) == []


def test_delete_and_replace(store):
    doc = store.similarity_search(TEXTS[5], k=1)[0]
    store.delete([doc.id])
    assert all(d.id != doc.id for d in store.similarity_search(TEXTS[5], k=10))
    assert store.get_by_ids([doc.id]) == []
    (new_id,) = store.add_texts(["replacement text"], [{"topic": "new"}], ids=["fixed-id"])
    store.add_texts(["replacement text, second version"], [{"topic": "newer"}], ids=["fixed-id"])
    (got,) = store.get_by_ids(["fixed-id"])
    assert got.page_content == "replacement text, second version" and got.metadata == {"topic": "newer"}
    assert len(store) == len(TEXTS)


def test_scores_and_relevance(store):
    (doc, distance), *_ = store.similarity_search_with_score(TEXTS[3], k=3)
    assert doc.page_content == TEXTS[3] and distance == pytest.approx(0, abs=1e-5)
    scored = store.similarity_search_with_relevance_scores(TEXTS[3], k=5)
    assert scored[0][1] == pytest.approx(1, abs=1e-5)
    assert all(0 <= s <= 1 for _, s in scored)


def test_max_marginal_relevance(store):
    docs = store.max_marginal_relevance_search(TEXTS[0], k=4, fetch_k=12)
    assert len(docs) == 4 and len({d.id for d in docs}) == 4
    assert docs[0].page_content == TEXTS[0]


def test_retriever(store):
    docs = store.as_retriever(search_kwargs={"k": 2}).invoke(TEXTS[10])
    assert docs[0].page_content == TEXTS[10] and len(docs) == 2


def test_hybrid_search_finds_keywords_the_embedding_misses(store):
    # The fake embedding carries no meaning, so only the keyword side knows which documents are
    # about trains; fused, all ten come first.
    docs = store.hybrid_search("trains", k=10)
    assert len(docs) == 10 and all(d.metadata["topic"] == "trains" for d in docs)
    # Filters apply to both sides; deleted documents are gone from both.
    assert all(d.metadata["n"] < 20 for d in store.hybrid_search("trains", k=5, filter=lambda m: m["n"] < 20))
    gone = docs[0]
    store.delete([gone.id])
    assert gone.id not in {d.id for d in store.hybrid_search("trains", k=40)}
    # Keyword weight 0: the plain vector ranking.
    plain = [d.id for d in store.similarity_search("trains", k=5)]
    assert [d.id for d in store.hybrid_search("trains", k=5, text_weight=0.0)] == plain


def test_save_and_load(store, tmp_path):
    store.delete([store.similarity_search(TEXTS[1], k=1)[0].id])
    store.save_local(str(tmp_path))
    loaded = EpzeroVectorStore.load_local(str(tmp_path), DeterministicFakeEmbedding(size=32))
    assert len(loaded) == len(store)
    for i in (0, 2, 33):
        assert loaded.similarity_search(TEXTS[i], k=1)[0].page_content == TEXTS[i]
    assert loaded.similarity_search(TEXTS[1], k=1)[0].page_content != TEXTS[1]
    loaded.add_texts(["after loading"])
    assert loaded.similarity_search("after loading", k=1)[0].page_content == "after loading"
    assert {d.metadata["topic"] for d in loaded.hybrid_search("boats", k=10)} == {"boats"}


def test_rejects_bad_input():
    store = EpzeroVectorStore(DeterministicFakeEmbedding(size=8))
    assert store.similarity_search("anything") == []
    with pytest.raises(ValueError):
        EpzeroVectorStore(DeterministicFakeEmbedding(size=8), metric="hamming")
    store.add_texts(["one"])
    with pytest.raises(ValueError):
        store.add_vectors(np.zeros((1, 4), dtype=np.float32), ["wrong width"])


# LangChain's own conformance suite for vector stores, when installed (pip install langchain-tests).
try:
    from langchain_tests.integration_tests import VectorStoreIntegrationTests
except ImportError:  # pragma: no cover
    VectorStoreIntegrationTests = None

if VectorStoreIntegrationTests is not None:

    class TestStandardVectorStore(VectorStoreIntegrationTests):
        @pytest.fixture
        def vectorstore(self):
            return EpzeroVectorStore(embedding=self.get_embeddings())
