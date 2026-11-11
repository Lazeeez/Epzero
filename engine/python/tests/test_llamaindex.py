import numpy as np
import pytest

pytest.importorskip("llama_index.core")

from llama_index.core import Document, MockEmbedding, StorageContext, VectorStoreIndex  # noqa: E402
from llama_index.core.schema import TextNode  # noqa: E402
from llama_index.core.vector_stores.types import (  # noqa: E402
    FilterCondition,
    FilterOperator,
    MetadataFilter,
    MetadataFilters,
    VectorStoreQuery,
)

from epzero.llamaindex import EpzeroVectorStore  # noqa: E402


def nodes(n=200, dim=16, seed=0):
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        out.append(TextNode(id_=f"n{i}", text=f"node {i}", embedding=rng.normal(size=dim).tolist(),
                            metadata={"source": ["faq", "blog"][i % 2], "year": 2000 + i % 20},
                            relationships={}))
    return out


def test_query_returns_the_nearest_with_similarities():
    store = EpzeroVectorStore(metric="cosine")
    ns = nodes()
    assert store.add(ns) == [n.node_id for n in ns]
    result = store.query(VectorStoreQuery(query_embedding=ns[7].embedding, similarity_top_k=3))
    assert result.ids[0] == "n7" and result.nodes[0].text == "node 7"
    assert result.similarities[0] == pytest.approx(1.0, abs=1e-5)
    assert result.similarities == sorted(result.similarities, reverse=True)


def test_filters_run_in_the_engine():
    store = EpzeroVectorStore()
    store.add(nodes())
    filters = MetadataFilters(filters=[MetadataFilter(key="source", value="faq"),
                                       MetadataFilter(key="year", value=2010, operator=FilterOperator.GTE)])
    result = store.query(VectorStoreQuery(query_embedding=[0.1] * 16, similarity_top_k=50, filters=filters))
    assert result.nodes and all(n.metadata["source"] == "faq" and n.metadata["year"] >= 2010 for n in result.nodes)
    either = MetadataFilters(filters=[MetadataFilter(key="year", value=2000), MetadataFilter(key="year", value=2001)],
                             condition=FilterCondition.OR)
    got = store.query(VectorStoreQuery(query_embedding=[0.1] * 16, similarity_top_k=100, filters=either))
    assert {n.metadata["year"] for n in got.nodes} == {2000, 2001} and len(got.nodes) == 20
    with pytest.raises(NotImplementedError):
        store.query(VectorStoreQuery(query_embedding=[0.1] * 16, similarity_top_k=5, filters=MetadataFilters(
            filters=[MetadataFilter(key="source", value="fa", operator=FilterOperator.TEXT_MATCH)])))


def test_delete_replace_and_persist(tmp_path):
    store = EpzeroVectorStore()
    ns = nodes(50)
    for n in ns[:10]:
        n.relationships = {}
    store.add(ns)
    store.delete_nodes(["n1", "n2"])
    assert {n.node_id for n in store.get_nodes()} == {f"n{i}" for i in range(50)} - {"n1", "n2"}
    replaced = nodes(1, seed=9)[0]
    replaced.id_ = "n3"
    store.add([replaced])  # the same id replaces the node
    assert len(store.get_nodes(node_ids=["n3"])) == 1
    path = str(tmp_path / "store")
    store.persist(path)
    loaded = EpzeroVectorStore.from_persist_path(path)
    result = loaded.query(VectorStoreQuery(query_embedding=ns[10].embedding, similarity_top_k=1))
    assert result.ids == ["n10"]
    assert len(loaded.get_nodes()) == 48


def test_grows_past_its_first_capacity():
    store = EpzeroVectorStore()
    store.add(nodes(3000))
    assert store.query(VectorStoreQuery(query_embedding=nodes(3000)[2500].embedding, similarity_top_k=1)).ids == ["n2500"]


def test_end_to_end_with_vector_store_index():
    store = EpzeroVectorStore()
    documents = [Document(text=f"document number {i} about topic {i % 5}", metadata={"topic": i % 5}) for i in range(40)]
    index = VectorStoreIndex.from_documents(documents, storage_context=StorageContext.from_defaults(vector_store=store),
                                            embed_model=MockEmbedding(embed_dim=8))
    retriever = index.as_retriever(similarity_top_k=4,
                                   filters=MetadataFilters(filters=[MetadataFilter(key="topic", value=2)]))
    hits = retriever.retrieve("topic 2")
    assert len(hits) == 4 and all(h.node.metadata["topic"] == 2 for h in hits)
    index.delete_ref_doc(documents[2].doc_id)
    assert all(n.ref_doc_id != documents[2].doc_id for n in store.get_nodes())
