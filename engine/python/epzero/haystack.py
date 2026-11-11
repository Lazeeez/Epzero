"""A Haystack document store and retriever backed by Epzero (pip install "epzero[haystack]").

    from haystack import Pipeline
    from epzero.haystack import EpzeroDocumentStore, EpzeroEmbeddingRetriever

    store = EpzeroDocumentStore(metric="cosine")
    store.write_documents(documents_with_embeddings)
    pipeline = Pipeline()
    pipeline.add_component("retriever", EpzeroEmbeddingRetriever(document_store=store, top_k=5))
    pipeline.run({"retriever": {"query_embedding": embedding, "filters": {"field": "meta.lang", "operator": "==", "value": "en"}}})

Filters use Haystack's syntax. Each part of a filter runs in the engine (epzero.Metadata) when that
gives exactly Haystack's answer: ==, !=, in and not in on str, int, float and bool values, numeric
ranges, and AND, OR and NOT. Anything else (ranges on strings such as dates, None, fields that mix
kinds of value) is checked by Haystack's own reference implementation, so results and errors match.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from haystack import component, default_from_dict, default_to_dict
from haystack.dataclasses import Document
from haystack.document_stores.errors import DuplicateDocumentError
from haystack.document_stores.types import DuplicatePolicy
from haystack.utils.filters import document_matches_filter

from ._core import Index, Metadata


def _kind(value: Any) -> str | None:
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, (int, float)):
        return "number"
    if isinstance(value, str):
        return "str"
    return None


class _Fallback(Exception):
    """This part of the filter must be evaluated by Haystack's reference implementation."""


class EpzeroDocumentStore:
    """Documents with their embeddings in an Epzero index; metadata filters in the engine."""

    def __init__(self, metric: str = "cosine", M: int = 16, ef_construction: int = 200, ef: int = 64):
        if metric not in ("cosine", "ip", "l2"):
            raise ValueError("metric must be 'cosine', 'ip' or 'l2'")
        self.metric, self.M, self.ef_construction, self.ef = metric, M, ef_construction, ef
        self._documents: dict[str, Document] = {}
        self._labels: dict[str, int] = {}  # document id -> label
        self._ids: dict[int, str] = {}     # label -> document id
        self._next = 0
        self._meta = Metadata()
        self._kinds: dict[str, set[str]] = {}  # per metadata field: the kinds of value it has held
        self._index: Index | None = None
        self._dim = 0
        self._capacity = 1024
        self._vectors = np.zeros((0, 0), dtype=np.float32)

    # ---- DocumentStore protocol ----

    def to_dict(self) -> dict[str, Any]:
        return default_to_dict(self, metric=self.metric, M=self.M, ef_construction=self.ef_construction, ef=self.ef)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EpzeroDocumentStore:
        return default_from_dict(cls, data)

    def count_documents(self) -> int:
        return len(self._documents)

    def filter_documents(self, filters: dict[str, Any] | None = None) -> list[Document]:
        if not filters:
            return list(self._documents.values())
        return [self._documents[self._ids[label]] for label in self._select(filters)]

    def write_documents(self, documents: list[Document], policy: DuplicatePolicy = DuplicatePolicy.NONE) -> int:
        if not isinstance(documents, list) or not all(isinstance(d, Document) for d in documents):
            raise ValueError("write_documents takes a list of Documents")
        if policy == DuplicatePolicy.NONE:
            policy = DuplicatePolicy.FAIL
        written = 0
        for document in documents:
            if document.id in self._documents:
                if policy == DuplicatePolicy.SKIP:
                    continue
                if policy == DuplicatePolicy.FAIL:
                    raise DuplicateDocumentError(f"ID '{document.id}' already exists.")
                self._remove(document.id)
            self._add(document)
            written += 1
        return written

    def delete_documents(self, document_ids: list[str]) -> None:
        for document_id in document_ids:
            self._remove(document_id)

    # ---- Retrieval ----

    def embedding_retrieval(self, query_embedding: list[float], filters: dict[str, Any] | None = None, top_k: int = 10,
                            scale_score: bool = False, return_embedding: bool = False) -> list[Document]:
        if self._index is None or top_k <= 0:
            return []
        query = np.asarray(query_embedding, dtype=np.float32)
        if query.shape != (self._dim,):
            raise ValueError(f"the query embedding must have {self._dim} dimensions")
        allowed = None
        if filters:
            allowed = np.array([label for label in self._select(filters) if self._has_vector(label)], dtype=np.int64)
            if len(allowed) == 0:
                return []
        ef = max(self.ef, top_k)
        options = {} if allowed is None else {"filter": self._index.filter(allowed)}
        labels, distances = self._index.search(query[None, :], k=top_k, ef=ef, **options)
        out = []
        for label, distance in zip(labels[0].tolist(), distances[0].tolist()):
            if label < 0:
                continue
            document = self._documents[self._ids[label]]
            score = self._score(distance, scale_score)
            fields = {"score": score}
            if not return_embedding:
                fields["embedding"] = None
            out.append(_replace(document, **fields))
        return out

    def _score(self, distance: float, scale: bool) -> float:
        # Higher is more similar, as in Haystack's in-memory store: cosine similarity (scaled to 0-1 as
        # (s + 1) / 2), the dot product (scaled by a sigmoid of s / 100), or minus the squared L2 distance
        # (scaled as exp(-d)). Engine distances are 1 - dot for cosine and ip, and the squared L2 distance.
        if self.metric == "l2":
            return float(np.exp(-distance)) if scale else -float(distance)
        similarity = 1.0 - float(distance)
        if not scale:
            return similarity
        return (similarity + 1) / 2 if self.metric == "cosine" else float(1 / (1 + np.exp(-similarity / 100)))

    # ---- Internals ----

    def _has_vector(self, label: int) -> bool:
        return self._documents[self._ids[label]].embedding is not None

    def _add(self, document: Document) -> None:
        label = self._next
        self._next += 1
        self._documents[document.id] = document
        self._labels[document.id] = label
        self._ids[label] = document.id
        scalars = {}
        for key, value in (document.meta or {}).items():
            kind = _kind(value)
            self._kinds.setdefault(key, set()).add(kind or "other")
            if kind is not None:
                scalars[key] = value
        self._meta.set(label, {**scalars, "__doc": True})  # every document is in the metadata index
        if document.embedding is None:
            return
        vector = np.asarray(document.embedding, dtype=np.float32)
        if self._index is None:
            self._dim = len(vector)
            self._vectors = np.zeros((self._capacity, self._dim), dtype=np.float32)
            self._index = self._new_index(self._capacity)
        elif len(vector) != self._dim:
            raise ValueError(f"expected {self._dim}-dimensional embeddings, got {len(vector)}")
        self._grow(label + 1)
        self._vectors[label] = vector
        self._index.add(vector[None, :], np.array([label], dtype=np.int64))

    def _remove(self, document_id: str) -> None:
        label = self._labels.pop(document_id, None)
        if label is None:
            return
        document = self._documents.pop(document_id)
        self._ids.pop(label)
        self._meta.remove(label)
        if document.embedding is not None and self._index is not None:
            self._index.remove(label)

    def _new_index(self, capacity: int) -> Index:
        return Index(dim=self._dim, metric=self.metric, max_elements=capacity, M=self.M, ef_construction=self.ef_construction)

    def _grow(self, needed: int) -> None:
        if needed <= self._capacity:
            return
        while self._capacity < needed:
            self._capacity *= 2
        vectors = np.zeros((self._capacity, self._dim), dtype=np.float32)
        vectors[: len(self._vectors)] = self._vectors
        self._vectors = vectors
        self._index.reserve(self._capacity)  # copies the nodes and keeps the graph

    def _select(self, filters: dict[str, Any]) -> list[int]:
        """Labels of the documents matching `filters`, in insertion order."""
        if "operator" not in filters and "conditions" not in filters:
            raise ValueError("Invalid filter syntax. See https://docs.haystack.deepset.ai/docs/metadata-filtering")
        try:
            labels = self._engine(filters)
            return sorted(labels.tolist())
        except _Fallback:
            return sorted(label for label, document_id in self._ids.items()
                          if document_matches_filter(filters, self._documents[document_id]))

    def _all(self) -> np.ndarray:
        return self._meta.select({"__doc": True})

    def _engine(self, f: dict[str, Any]) -> np.ndarray:
        operator = f.get("operator")
        if "conditions" in f:
            parts = [self._engine(c) for c in f["conditions"]]
            if operator == "AND":
                result = self._all() if not parts else parts[0]
                for part in parts[1:]:
                    result = np.intersect1d(result, part, assume_unique=True)
                return result
            if operator == "OR":
                return np.unique(np.concatenate(parts)) if parts else np.array([], dtype=np.int64)
            if operator == "NOT":
                inner = self._engine({"operator": "AND", "conditions": f["conditions"]})
                return np.setdiff1d(self._all(), inner, assume_unique=True)
            raise _Fallback
        field, value = f.get("field", ""), f.get("value")
        if not field.startswith("meta.") or "." in field[5:]:
            raise _Fallback
        key = field[5:]
        kinds = self._kinds.get(key, set())
        if "other" in kinds or {"bool", "number"} <= kinds:
            raise _Fallback
        if operator in ("==", "!="):
            if _kind(value) is None or (_kind(value) == "bool" and "number" in kinds) or (_kind(value) == "number" and "bool" in kinds):
                raise _Fallback
            equal = self._meta.select({key: value})
            return equal if operator == "==" else np.setdiff1d(self._all(), equal, assume_unique=True)
        if operator in ("in", "not in"):
            if not isinstance(value, list) or not all(_kind(v) is not None for v in value):
                raise _Fallback
            if any(_kind(v) == "bool" for v in value) and "number" in kinds or any(_kind(v) == "number" for v in value) and "bool" in kinds:
                raise _Fallback
            inside = self._meta.select({key: value}) if value else np.array([], dtype=np.int64)
            return inside if operator == "in" else np.setdiff1d(self._all(), inside, assume_unique=True)
        if operator in (">", ">=", "<", "<="):
            if _kind(value) != "number" or kinds - {"number"}:
                raise _Fallback  # strings (dates) and others compare by Haystack's rules
            op = {">": "$gt", ">=": "$gte", "<": "$lt", "<=": "$lte"}[operator]
            return self._meta.select({key: {op: value}})
        raise _Fallback


def _replace(document: Document, **fields: Any) -> Document:
    data = {name: getattr(document, name) for name in ("id", "content", "blob", "meta", "score", "embedding", "sparse_embedding")}
    data.update(fields)
    return Document(**data)


@component
class EpzeroEmbeddingRetriever:
    """Retrieves the documents nearest a query embedding from an EpzeroDocumentStore."""

    def __init__(self, document_store: EpzeroDocumentStore, filters: dict[str, Any] | None = None, top_k: int = 10,
                 scale_score: bool = False, return_embedding: bool = False):
        if not isinstance(document_store, EpzeroDocumentStore):
            raise ValueError("document_store must be an EpzeroDocumentStore")
        self.document_store, self.filters, self.top_k = document_store, filters, top_k
        self.scale_score, self.return_embedding = scale_score, return_embedding

    def to_dict(self) -> dict[str, Any]:
        return default_to_dict(self, document_store=self.document_store.to_dict(), filters=self.filters, top_k=self.top_k,
                               scale_score=self.scale_score, return_embedding=self.return_embedding)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EpzeroEmbeddingRetriever:
        data["init_parameters"]["document_store"] = EpzeroDocumentStore.from_dict(data["init_parameters"]["document_store"])
        return default_from_dict(cls, data)

    @component.output_types(documents=list[Document])
    def run(self, query_embedding: list[float], filters: dict[str, Any] | None = None, top_k: int | None = None) -> dict[str, list[Document]]:
        documents = self.document_store.embedding_retrieval(
            query_embedding, filters=filters if filters is not None else self.filters, top_k=top_k or self.top_k,
            scale_score=self.scale_score, return_embedding=self.return_embedding)
        return {"documents": documents}
