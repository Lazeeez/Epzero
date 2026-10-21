"""A LangChain vector store backed by an Oryx HNSW index.

    from langchain_openai import OpenAIEmbeddings
    from oryxdb.langchain import OryxVectorStore

    store = OryxVectorStore.from_texts(texts, OpenAIEmbeddings(), metadatas=metadatas)
    store.similarity_search("how do I reset my password?", k=4, filter={"source": "faq"})
    retriever = store.as_retriever(search_kwargs={"k": 8})

Documents, their metadata and their vectors live in memory beside the index; save_local() and
load_local() persist all of it to a folder. Metadata filters (a dict of required values, a list
meaning "any of", or a callable) are compiled into an Oryx filter, so the engine picks its filtered
search strategy instead of over-fetching and discarding.
"""

from __future__ import annotations

import json
import os
import uuid
from collections.abc import Callable, Iterable, Sequence
from typing import Any

import numpy as np

try:
    from langchain_core.documents import Document
    from langchain_core.embeddings import Embeddings
    from langchain_core.vectorstores import VectorStore
    from langchain_core.vectorstores.utils import maximal_marginal_relevance
except ImportError as error:  # pragma: no cover
    raise ImportError('oryxdb.langchain needs LangChain: pip install "oryxdb[langchain]"') from error

from ._core import Index

MetadataFilter = dict[str, Any] | Callable[[dict[str, Any]], bool]


class OryxVectorStore(VectorStore):
    """LangChain `VectorStore` on an Oryx HNSW index.

    Args:
        embedding: the embedding model.
        metric: "cosine" (default), "ip" (inner product) or "l2".
        M, ef_construction: graph parameters (links per node, build beam width).
        ef: search beam width; higher is more accurate and slower.
        capacity: initial number of vectors the index can hold; it grows (by rebuilding) as needed.
    """

    def __init__(
        self,
        embedding: Embeddings,
        *,
        metric: str = "cosine",
        M: int = 16,
        ef_construction: int = 200,
        ef: int = 64,
        capacity: int = 1024,
    ) -> None:
        if metric not in ("cosine", "ip", "l2"):
            raise ValueError("metric must be 'cosine', 'ip' or 'l2'")
        self._embedding = embedding
        self.metric = metric
        self.M = M
        self.ef_construction = ef_construction
        self.ef = ef
        self._capacity = max(16, capacity)
        self._index: Index | None = None
        self._dim = 0
        self._vectors = np.zeros((0, 0), dtype=np.float32)  # row = label
        self._ids: list[str | None] = []  # label -> id (None once deleted)
        self._labels: dict[str, int] = {}  # id -> label
        self._texts: list[str] = []
        self._metadatas: list[dict[str, Any]] = []

    # ---- LangChain interface ----

    @property
    def embeddings(self) -> Embeddings:
        return self._embedding

    def add_texts(
        self,
        texts: Iterable[str],
        metadatas: list[dict] | None = None,
        *,
        ids: list[str] | None = None,
        **kwargs: Any,
    ) -> list[str]:
        texts = list(texts)
        if not texts:
            return []
        vectors = np.asarray(self._embedding.embed_documents(texts), dtype=np.float32)
        return self.add_vectors(vectors, texts, metadatas, ids=ids)

    def add_vectors(
        self,
        vectors: np.ndarray,
        texts: Sequence[str],
        metadatas: Sequence[dict] | None = None,
        *,
        ids: Sequence[str] | None = None,
    ) -> list[str]:
        """Adds precomputed embeddings with their texts. An existing id is replaced."""
        vectors = np.ascontiguousarray(vectors, dtype=np.float32)
        if vectors.ndim != 2 or len(vectors) != len(texts):
            raise ValueError("vectors must be (len(texts), dim)")
        metadatas = list(metadatas) if metadatas is not None else [{} for _ in texts]
        ids = [str(i) for i in ids] if ids is not None else [str(uuid.uuid4()) for _ in texts]
        if len(metadatas) != len(texts) or len(ids) != len(texts):
            raise ValueError("metadatas and ids must match texts in length")
        if self._index is None:
            self._dim = vectors.shape[1]
            self._vectors = np.zeros((0, self._dim), dtype=np.float32)
            self._index = self._new_index(self._capacity)
        elif vectors.shape[1] != self._dim:
            raise ValueError(f"expected {self._dim}-dimensional vectors, got {vectors.shape[1]}")
        self.delete([i for i in ids if i in self._labels])
        first = len(self._ids)
        self._ensure_capacity(first + len(texts))
        labels = np.arange(first, first + len(texts), dtype=np.uint64)
        self._vectors = np.concatenate([self._vectors, vectors])
        self._ids.extend(ids)
        self._labels.update({id_: int(label) for id_, label in zip(ids, labels)})
        self._texts.extend(texts)
        self._metadatas.extend(dict(m) for m in metadatas)
        self._index.add(vectors, labels)
        return list(ids)

    def delete(self, ids: list[str] | None = None, **kwargs: Any) -> bool | None:
        if ids is None:
            return None
        for id_ in ids:
            label = self._labels.pop(id_, None)
            if label is not None:
                self._index.remove(label)
                self._ids[label] = None
        return True

    def get_by_ids(self, ids: Sequence[str], /) -> list[Document]:
        return [self._document(self._labels[i]) for i in ids if i in self._labels]

    def similarity_search(self, query: str, k: int = 4, filter: MetadataFilter | None = None, **kwargs: Any) -> list[Document]:
        return [doc for doc, _ in self.similarity_search_with_score(query, k, filter=filter, **kwargs)]

    def similarity_search_with_score(
        self, query: str, k: int = 4, filter: MetadataFilter | None = None, **kwargs: Any
    ) -> list[tuple[Document, float]]:
        vector = self._embedding.embed_query(query)
        return self.similarity_search_with_score_by_vector(vector, k, filter=filter, **kwargs)

    def similarity_search_by_vector(
        self, embedding: list[float], k: int = 4, filter: MetadataFilter | None = None, **kwargs: Any
    ) -> list[Document]:
        return [doc for doc, _ in self.similarity_search_with_score_by_vector(embedding, k, filter=filter, **kwargs)]

    def similarity_search_with_score_by_vector(
        self, embedding: Sequence[float], k: int = 4, filter: MetadataFilter | None = None, **kwargs: Any
    ) -> list[tuple[Document, float]]:
        """Documents nearest the vector, with the engine's distance (lower is closer)."""
        hits = self._search(np.asarray(embedding, dtype=np.float32), k, filter, kwargs.get("ef", self.ef))
        return [(self._document(label), distance) for label, distance in hits]

    def max_marginal_relevance_search(
        self,
        query: str,
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        filter: MetadataFilter | None = None,
        **kwargs: Any,
    ) -> list[Document]:
        vector = self._embedding.embed_query(query)
        return self.max_marginal_relevance_search_by_vector(vector, k, fetch_k, lambda_mult, filter=filter, **kwargs)

    def max_marginal_relevance_search_by_vector(
        self,
        embedding: Sequence[float],
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        filter: MetadataFilter | None = None,
        **kwargs: Any,
    ) -> list[Document]:
        query = np.asarray(embedding, dtype=np.float32)
        hits = self._search(query, max(fetch_k, k), filter, kwargs.get("ef", self.ef))
        if not hits:
            return []
        labels = [label for label, _ in hits]
        chosen = maximal_marginal_relevance(query, self._vectors[labels], lambda_mult=lambda_mult, k=k)
        return [self._document(labels[i]) for i in chosen]

    def _select_relevance_score_fn(self) -> Callable[[float], float]:
        # Map the engine's distances to [0, 1], higher is more relevant.
        if self.metric == "cosine":
            return lambda d: 1.0 - d / 2.0  # d = 1 - cos, in [0, 2]
        if self.metric == "ip":
            return lambda d: 1.0 - d / 2.0  # d = 1 - dot; exact in [0, 1] for unit vectors
        return lambda d: 1.0 - float(np.sqrt(max(d, 0.0))) / np.sqrt(2.0)  # d = squared L2, unit vectors

    @classmethod
    def from_texts(
        cls,
        texts: list[str],
        embedding: Embeddings,
        metadatas: list[dict] | None = None,
        *,
        ids: list[str] | None = None,
        **kwargs: Any,
    ) -> OryxVectorStore:
        store = cls(embedding, **kwargs)
        store.add_texts(texts, metadatas, ids=ids)
        return store

    # ---- Persistence ----

    def save_local(self, folder: str) -> None:
        """Writes the index, the vectors and the documents to `folder`."""
        os.makedirs(folder, exist_ok=True)
        if self._index is not None:
            self._index.save(os.path.join(folder, "index.oryx"))
        np.save(os.path.join(folder, "vectors.npy"), self._vectors)
        state = {
            "format": 1,
            "metric": self.metric,
            "M": self.M,
            "ef_construction": self.ef_construction,
            "ef": self.ef,
            "capacity": self._capacity,
            "dim": self._dim,
            "ids": self._ids,
            "texts": self._texts,
            "metadatas": self._metadatas,
        }
        with open(os.path.join(folder, "store.json"), "w", encoding="utf-8") as f:
            json.dump(state, f, ensure_ascii=False)

    @classmethod
    def load_local(cls, folder: str, embedding: Embeddings) -> OryxVectorStore:
        """Opens a store written by save_local()."""
        with open(os.path.join(folder, "store.json"), encoding="utf-8") as f:
            state = json.load(f)
        if state.get("format") != 1:
            raise ValueError(f"{folder}: unsupported store format")
        store = cls(embedding, metric=state["metric"], M=state["M"], ef_construction=state["ef_construction"], ef=state["ef"])
        store._capacity = state["capacity"]
        store._dim = state["dim"]
        store._ids = state["ids"]
        store._labels = {id_: label for label, id_ in enumerate(store._ids) if id_ is not None}
        store._texts = state["texts"]
        store._metadatas = state["metadatas"]
        store._vectors = np.load(os.path.join(folder, "vectors.npy"))
        index_path = os.path.join(folder, "index.oryx")
        if os.path.exists(index_path):
            store._index = Index.load(index_path)
        return store

    # ---- Internals ----

    def __len__(self) -> int:
        return len(self._labels)

    def _new_index(self, capacity: int) -> Index:
        return Index(dim=self._dim, metric=self.metric, max_elements=capacity, M=self.M, ef_construction=self.ef_construction)

    def _ensure_capacity(self, needed: int) -> None:
        if needed <= self._capacity:
            return
        while self._capacity < needed:
            self._capacity *= 2
        # The index has a fixed capacity: rebuild a bigger one from the live vectors.
        index = self._new_index(self._capacity)
        live = np.array(sorted(self._labels.values()), dtype=np.uint64)
        if len(live):
            index.add(self._vectors[live.astype(np.int64)], live)
        self._index = index

    def _allowed(self, filter: MetadataFilter) -> list[int]:
        if callable(filter):
            test = filter
        else:
            def test(metadata: dict[str, Any]) -> bool:
                for key, want in filter.items():
                    value = metadata.get(key)
                    if isinstance(want, (list, tuple, set)):
                        if value not in want:
                            return False
                    elif value != want:
                        return False
                return True
        return [label for label in self._labels.values() if test(self._metadatas[label])]

    def _search(self, query: np.ndarray, k: int, filter: MetadataFilter | None, ef: int) -> list[tuple[int, float]]:
        if self._index is None or not self._labels or k <= 0:
            return []
        if query.shape != (self._dim,):
            raise ValueError(f"expected a {self._dim}-dimensional query")
        kwargs: dict[str, Any] = {}
        if filter is not None:
            allowed = self._allowed(filter)
            if not allowed:
                return []
            kwargs["filter"] = allowed
        labels, distances = self._index.search(query[None, :], k=k, ef=max(ef, k), **kwargs)
        return [(int(label), float(d)) for label, d in zip(labels[0], distances[0]) if label >= 0]

    def _document(self, label: int) -> Document:
        return Document(id=self._ids[label], page_content=self._texts[label], metadata=dict(self._metadatas[label]))
