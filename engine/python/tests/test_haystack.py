import random

import numpy as np
import pytest

pytest.importorskip("haystack")

from haystack import Document, Pipeline  # noqa: E402
from haystack.testing.document_store import DocumentStoreBaseTests  # noqa: E402
from haystack.utils.filters import document_matches_filter  # noqa: E402

from epzero.haystack import EpzeroDocumentStore, EpzeroEmbeddingRetriever  # noqa: E402


class TestEpzeroDocumentStore(DocumentStoreBaseTests):
    """Haystack's own conformance suite: counting, writing, deleting and every filter case."""

    @pytest.fixture
    def document_store(self):
        return EpzeroDocumentStore()

    def test_write_documents(self, document_store):
        """Without a policy, a duplicate id fails, as in Haystack's in-memory store."""
        from haystack.document_stores.errors import DuplicateDocumentError

        docs = [Document(id="1", content="test doc")]
        assert document_store.write_documents(docs) == 1
        with pytest.raises(DuplicateDocumentError):
            document_store.write_documents(docs)


def test_engine_filters_agree_with_haystack():
    rng = random.Random(0)
    store = EpzeroDocumentStore()
    docs = [Document(id=str(i), content=f"doc {i}", meta={"lang": rng.choice(["en", "de", "fr"]), "year": rng.randint(2000, 2024),
                                                           "score": rng.random(), "public": rng.random() < 0.5})
            for i in range(400)]
    store.write_documents(docs)
    leaves = [
        {"field": "meta.lang", "operator": "==", "value": "en"},
        {"field": "meta.lang", "operator": "!=", "value": "de"},
        {"field": "meta.lang", "operator": "in", "value": ["en", "fr"]},
        {"field": "meta.lang", "operator": "not in", "value": ["fr"]},
        {"field": "meta.year", "operator": ">=", "value": 2015},
        {"field": "meta.year", "operator": "<", "value": 2005},
        {"field": "meta.score", "operator": ">", "value": 0.5},
        {"field": "meta.public", "operator": "==", "value": True},
        {"field": "meta.missing", "operator": "!=", "value": "x"},
    ]
    for _ in range(300):
        a, b = rng.sample(leaves, 2)
        filters = {"operator": rng.choice(["AND", "OR", "NOT"]), "conditions": [a, {"operator": rng.choice(["AND", "OR"]), "conditions": [b, rng.choice(leaves)]}]}
        expected = [d.id for d in docs if document_matches_filter(filters, d)]
        assert [d.id for d in store.filter_documents(filters)] == expected, filters
        assert store._engine(filters) is not None  # these stay on the engine path


def test_retriever_in_a_pipeline():
    rng = np.random.default_rng(0)
    store = EpzeroDocumentStore(metric="cosine")
    embeddings = rng.normal(size=(500, 16)).astype(np.float32)
    store.write_documents([Document(id=str(i), content=f"doc {i}", embedding=embeddings[i].tolist(), meta={"group": i % 4})
                           for i in range(500)])
    pipeline = Pipeline()
    pipeline.add_component("retriever", EpzeroEmbeddingRetriever(document_store=store, top_k=5))
    result = pipeline.run({"retriever": {"query_embedding": embeddings[42].tolist()}})
    documents = result["retriever"]["documents"]
    assert documents[0].id == "42" and documents[0].score == pytest.approx(1.0, abs=1e-5)
    assert documents[0].embedding is None
    filtered = pipeline.run({"retriever": {"query_embedding": embeddings[42].tolist(),
                                           "filters": {"field": "meta.group", "operator": "==", "value": 1}}})
    assert all(d.meta["group"] == 1 for d in filtered["retriever"]["documents"])
    restored = EpzeroEmbeddingRetriever.from_dict(EpzeroEmbeddingRetriever(document_store=store, top_k=3).to_dict())
    assert restored.top_k == 3 and isinstance(restored.document_store, EpzeroDocumentStore)
