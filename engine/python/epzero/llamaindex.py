"""A LlamaIndex vector store backed by Epzero (pip install "epzero[llamaindex]").

    from llama_index.core import StorageContext, VectorStoreIndex
    from epzero.llamaindex import EpzeroVectorStore

    store = EpzeroVectorStore(metric="cosine")
    index = VectorStoreIndex.from_documents(documents, storage_context=StorageContext.from_defaults(vector_store=store))
    answer = index.as_query_engine().query("...")
    store.persist("./epzero-store")                      # and EpzeroVectorStore.from_persist_path(...)

Nodes are kept with their text and metadata (the store holds text, so a docstore isn't needed).
Metadata filters run in the engine (epzero.Metadata): ==, !=, >, >=, <, <=, in, nin on str, int,
float and bool values, combined with AND or OR, nested; other operators raise NotImplementedError.
"""

from __future__ import annotations

import json
import os
from typing import Any

import numpy as np
from llama_index.core.bridge.pydantic import PrivateAttr
from llama_index.core.schema import BaseNode
from llama_index.core.vector_stores.types import (
    BasePydanticVectorStore,
    FilterCondition,
    FilterOperator,
    MetadataFilters,
    VectorStoreQuery,
    VectorStoreQueryResult,
)
from llama_index.core.vector_stores.utils import metadata_dict_to_node, node_to_metadata_dict

from ._core import Index, Metadata

_OPS = {
    FilterOperator.EQ: "$eq", FilterOperator.NE: "$ne", FilterOperator.GT: "$gt", FilterOperator.GTE: "$gte",
    FilterOperator.LT: "$lt", FilterOperator.LTE: "$lte", FilterOperator.IN: "$in", FilterOperator.NIN: "$nin",
}


def _scalar(value: Any) -> bool:
    return isinstance(value, (str, int, float, bool))


class EpzeroVectorStore(BasePydanticVectorStore):
    """Epzero as a LlamaIndex vector store: HNSW search with filters evaluated in the engine."""

    stores_text: bool = True
    flat_metadata: bool = False
    metric: str = "cosine"
    M: int = 16
    ef_construction: int = 200
    ef: int = 64

    _index: Index | None = PrivateAttr(default=None)
    _meta: Metadata = PrivateAttr(default_factory=Metadata)
    _dim: int = PrivateAttr(default=0)
    _capacity: int = PrivateAttr(default=1024)
    _next: int = PrivateAttr(default=0)
    _vectors: np.ndarray = PrivateAttr(default_factory=lambda: np.zeros((0, 0), dtype=np.float32))
    _nodes: dict[int, dict[str, Any]] = PrivateAttr(default_factory=dict)  # label -> node_to_metadata_dict
    _labels: dict[str, int] = PrivateAttr(default_factory=dict)  # node id -> label

    def __init__(self, metric: str = "cosine", M: int = 16, ef_construction: int = 200, ef: int = 64, **kwargs: Any):
        if metric not in ("cosine", "ip", "l2"):
            raise ValueError("metric must be 'cosine', 'ip' or 'l2'")
        super().__init__(metric=metric, M=M, ef_construction=ef_construction, ef=ef, **kwargs)

    @classmethod
    def class_name(cls) -> str:
        return "EpzeroVectorStore"

    @property
    def client(self) -> Index | None:
        return self._index

    # ---- Writing ----

    def add(self, nodes: list[BaseNode], **kwargs: Any) -> list[str]:
        if not nodes:
            return []
        vectors = np.asarray([node.get_embedding() for node in nodes], dtype=np.float32)
        if self._index is None:
            self._dim = vectors.shape[1]
            self._vectors = np.zeros((self._capacity, self._dim), dtype=np.float32)
            self._index = self._new_index(self._capacity)
        elif vectors.shape[1] != self._dim:
            raise ValueError(f"expected {self._dim}-dimensional embeddings, got {vectors.shape[1]}")
        self.delete_nodes([node.node_id for node in nodes if node.node_id in self._labels])
        first = self._next
        self._grow(first + len(nodes))
        labels = np.arange(first, first + len(nodes), dtype=np.int64)
        self._vectors[first:first + len(nodes)] = vectors
        for label, node in zip(labels.tolist(), nodes):
            self._nodes[label] = node_to_metadata_dict(node, remove_text=False, flat_metadata=False)
            self._labels[node.node_id] = label
            scalars = {k: v for k, v in node.metadata.items() if _scalar(v)}
            if node.ref_doc_id is not None:
                scalars["__ref_doc_id"] = node.ref_doc_id
            scalars["__node_id"] = node.node_id
            self._meta.set(label, scalars)
        self._index.add(vectors, labels)
        self._next += len(nodes)
        return [node.node_id for node in nodes]

    def delete(self, ref_doc_id: str, **delete_kwargs: Any) -> None:
        for label in self._meta.select({"__ref_doc_id": ref_doc_id}).tolist():
            self._remove(label)

    def delete_nodes(self, node_ids: list[str] | None = None, filters: MetadataFilters | None = None, **kwargs: Any) -> None:
        labels = set(self._labels[i] for i in (node_ids or []) if i in self._labels)
        if filters is not None:
            matched = self._select(filters).tolist()
            labels = labels & set(matched) if node_ids else set(matched)
        for label in labels:
            self._remove(label)

    def clear(self) -> None:
        for label in list(self._nodes):
            self._remove(label)

    def _remove(self, label: int) -> None:
        node = self._nodes.pop(label, None)
        if node is None:
            return
        self._labels.pop(self._meta.get(label).get("__node_id"), None)
        self._meta.remove(label)
        self._index.remove(label)

    # ---- Reading ----

    def get_nodes(self, node_ids: list[str] | None = None, filters: MetadataFilters | None = None, **kwargs: Any) -> list[BaseNode]:
        labels = list(self._nodes) if node_ids is None and filters is None else []
        if node_ids is not None:
            labels = [self._labels[i] for i in node_ids if i in self._labels]
        if filters is not None:
            allowed = set(self._select(filters).tolist())
            labels = [l for l in (labels if node_ids is not None else self._nodes) if l in allowed]
        return [self._node(label) for label in labels]

    def query(self, query: VectorStoreQuery, **kwargs: Any) -> VectorStoreQueryResult:
        if self._index is None or not self._nodes or query.query_embedding is None:
            return VectorStoreQueryResult(nodes=[], similarities=[], ids=[])
        k = max(1, query.similarity_top_k)
        allowed = None
        if query.filters is not None:
            allowed = set(self._select(query.filters).tolist())
        if query.doc_ids:
            docs = set(self._meta.select({"__ref_doc_id": list(query.doc_ids)}).tolist())
            allowed = docs if allowed is None else allowed & docs
        if query.node_ids:
            nodes = set(self._labels[i] for i in query.node_ids if i in self._labels)
            allowed = nodes if allowed is None else allowed & nodes
        vector = np.asarray(query.query_embedding, dtype=np.float32)[None, :]
        ef = max(kwargs.get("ef", self.ef), k)
        if allowed is not None:
            if not allowed:
                return VectorStoreQueryResult(nodes=[], similarities=[], ids=[])
            labels, distances = self._index.search(vector, k=k, ef=ef, filter=self._index.filter(np.fromiter(allowed, np.int64)))
        else:
            labels, distances = self._index.search(vector, k=k, ef=ef)
        hits = [(int(l), float(d)) for l, d in zip(labels[0], distances[0]) if l >= 0]
        nodes = [self._node(label) for label, _ in hits]
        return VectorStoreQueryResult(nodes=nodes, similarities=[self._similarity(d) for _, d in hits],
                                      ids=[node.node_id for node in nodes])

    def _similarity(self, distance: float) -> float:
        # Engine distances: cosine and ip are 1 - dot, l2 the squared distance.
        return 1.0 - distance if self.metric in ("cosine", "ip") else -distance

    def _node(self, label: int) -> BaseNode:
        node = metadata_dict_to_node(self._nodes[label])
        node.embedding = self._vectors[label].tolist()
        return node

    def _select(self, filters: MetadataFilters) -> np.ndarray:
        """Labels passing `filters`, evaluated in the engine; OR is a union of the parts."""
        parts = []
        for f in filters.filters:
            if isinstance(f, MetadataFilters):
                parts.append(self._select(f))
                continue
            op = _OPS.get(f.operator)
            if op is None:
                raise NotImplementedError(f"filter operator {f.operator.value} is not supported by Epzero")
            parts.append(self._meta.select({f.key: {op: f.value}}))
        if not parts:
            return self._meta.select({})
        if filters.condition == FilterCondition.OR:
            return np.unique(np.concatenate(parts))
        if filters.condition not in (FilterCondition.AND, None):
            raise NotImplementedError(f"filter condition {filters.condition} is not supported by Epzero")
        result = parts[0]
        for part in parts[1:]:
            result = np.intersect1d(result, part, assume_unique=True)
        return result

    # ---- Capacity and persistence ----

    def _new_index(self, capacity: int) -> Index:
        return Index(dim=self._dim, metric=self.metric, max_elements=capacity, M=self.M, ef_construction=self.ef_construction)

    def _grow(self, needed: int) -> None:
        if needed <= self._capacity:
            return
        while self._capacity < needed:
            self._capacity *= 2
        vectors = np.zeros((self._capacity, self._dim), dtype=np.float32)
        vectors[: self._next] = self._vectors[: self._next]
        self._vectors = vectors
        index = self._new_index(self._capacity)  # the index has a fixed capacity: rebuild it larger
        live = np.array(sorted(self._nodes), dtype=np.int64)
        if len(live):
            index.add(self._vectors[live], live)
        self._index = index

    def persist(self, persist_path: str, fs: Any = None) -> None:
        """Saves the store in the folder `persist_path`."""
        os.makedirs(persist_path, exist_ok=True)
        np.save(os.path.join(persist_path, "vectors.npy"), self._vectors[: self._next])
        self._meta.save(os.path.join(persist_path, "metadata.epzero"))
        with open(os.path.join(persist_path, "store.json"), "w", encoding="utf-8") as f:
            json.dump({"metric": self.metric, "M": self.M, "ef_construction": self.ef_construction, "ef": self.ef,
                       "dim": self._dim, "next": self._next, "nodes": {str(k): v for k, v in self._nodes.items()}}, f)

    @classmethod
    def from_persist_path(cls, persist_path: str, fs: Any = None) -> EpzeroVectorStore:
        with open(os.path.join(persist_path, "store.json"), encoding="utf-8") as f:
            state = json.load(f)
        store = cls(metric=state["metric"], M=state["M"], ef_construction=state["ef_construction"], ef=state["ef"])
        store._nodes = {int(k): v for k, v in state["nodes"].items()}
        store._meta = Metadata.load(os.path.join(persist_path, "metadata.epzero"))
        store._next = state["next"]
        if state["dim"]:
            store._dim = state["dim"]
            vectors = np.load(os.path.join(persist_path, "vectors.npy"))
            store._capacity = max(1024, 1 << max(0, store._next - 1).bit_length())
            store._vectors = np.zeros((store._capacity, store._dim), dtype=np.float32)
            store._vectors[: len(vectors)] = vectors
            store._index = store._new_index(store._capacity)
            live = np.array(sorted(store._nodes), dtype=np.int64)
            if len(live):
                store._index.add(store._vectors[live], live)
            store._labels = {store._meta.get(label)["__node_id"]: label for label in store._nodes}
        return store
