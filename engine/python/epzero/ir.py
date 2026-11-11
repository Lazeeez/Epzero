"""Information retrieval with Epzero: collections keyed by document id, dense, keyword (BM25) and hybrid
retrieval, TREC run files, relevance judgements and the standard measures.

    from epzero import ir

    collection = ir.Collection.build(doc_ids, vectors, texts=texts, metric="cosine")
    run = collection.retrieve(query_ids, vectors=query_vectors, texts=query_texts, k=100, mode="hybrid")
    ir.write_run(run, "run.txt", tag="epzero-hybrid")          # TREC format, for CLEF, TREC or BEIR
    print(ir.evaluate(run, ir.read_qrels("qrels.txt")))         # nDCG@10, R@100, MAP, MRR@10

Measures follow trec_eval: a run is ranked by score (ties by document id, descending), nDCG uses linear
gains, every query with at least one relevant document in the judgements is scored, and a query missing
from the run scores 0.
"""

from __future__ import annotations

import json
import math
import os
from collections.abc import Iterable, Sequence

import numpy as np

from . import auto
from ._core import Index, TextIndex, fuse_rankings

Run = dict[str, list[tuple[str, float]]]   # query id -> [(document id, score)], best first
Qrels = dict[str, dict[str, int]]           # query id -> {document id: relevance}

DEFAULT_MEASURES = ("nDCG@10", "R@100", "MAP", "MRR@10")


class Collection:
    """Documents with string ids, their vectors and optionally their text, searchable by vector, by keyword
    or by both (reciprocal rank fusion). Documents can be added and removed after the build."""

    def __init__(self, index: Index, text: TextIndex | None, doc_ids: list[str | None]):
        self.index = index
        self.text = text
        self._ids = doc_ids                                      # label -> document id (None once removed)
        self._labels = {d: i for i, d in enumerate(doc_ids) if d is not None}

    @classmethod
    def build(cls, doc_ids: Sequence[str], vectors: np.ndarray, texts: Sequence[str] | None = None,
              metric: str = "cosine", **settings) -> Collection:
        """Indexes `vectors` with epzero.build (settings such as recall=, queries= or M= pass through), and
        `texts` with BM25 when given."""
        doc_ids = [str(d) for d in doc_ids]
        if len(set(doc_ids)) != len(doc_ids):
            raise ValueError("document ids must be unique")
        vectors = np.ascontiguousarray(vectors, dtype=np.float32)
        if len(vectors) != len(doc_ids) or (texts is not None and len(texts) != len(doc_ids)):
            raise ValueError("one vector (and text) per document id")
        index = auto.build(vectors, metric=metric, **settings)
        text = None
        if texts is not None:
            text = TextIndex()
            text.add(np.arange(len(doc_ids), dtype=np.int64), list(texts))
        return cls(index, text, list(doc_ids))

    def __len__(self) -> int:
        return len(self._labels)

    def __contains__(self, doc_id: str) -> bool:
        return doc_id in self._labels

    def add(self, doc_ids: Sequence[str], vectors: np.ndarray, texts: Sequence[str] | None = None) -> None:
        """Adds documents; an id already present is replaced. The index grows as needed."""
        doc_ids = [str(d) for d in doc_ids]
        vectors = np.ascontiguousarray(vectors, dtype=np.float32)
        if len(vectors) != len(doc_ids) or len(set(doc_ids)) != len(doc_ids):
            raise ValueError("one vector per unique document id")
        if (texts is None) != (self.text is None):
            raise ValueError("give texts exactly when the collection was built with them")
        self.remove([d for d in doc_ids if d in self._labels])
        first = len(self._ids)
        labels = np.arange(first, first + len(doc_ids), dtype=np.int64)
        self.index.add(vectors, labels=labels)
        if self.text is not None:
            self.text.add(labels, list(texts))
        self._ids.extend(doc_ids)
        self._labels.update({d: int(l) for d, l in zip(doc_ids, labels)})

    def remove(self, doc_ids: Iterable[str]) -> int:
        """Removes documents; returns how many were present."""
        removed = 0
        for doc_id in doc_ids:
            label = self._labels.pop(str(doc_id), None)
            if label is None:
                continue
            self.index.remove(label)
            if self.text is not None:
                self.text.remove(label)
            self._ids[label] = None
            removed += 1
        return removed

    def retrieve(self, query_ids: Sequence[str], vectors: np.ndarray | None = None, texts: Sequence[str] | None = None,
                 k: int = 100, mode: str = "dense", candidates: int | None = None, rrf_k: float = 60.0,
                 weights: tuple[float, float] = (1.0, 1.0), ef: int | None = None, threads: int = 0) -> Run:
        """A run: for each query, the k best documents. mode "dense" uses `vectors`, "keyword" uses `texts`
        (BM25), and "hybrid" fuses both by reciprocal rank fusion, each contributing `candidates` results
        (default 2k), weighted by `weights` (dense, keyword)."""
        query_ids = [str(q) for q in query_ids]
        if mode not in ("dense", "keyword", "hybrid"):
            raise ValueError("mode must be 'dense', 'keyword' or 'hybrid'")
        if mode != "keyword" and (vectors is None or len(vectors) != len(query_ids)):
            raise ValueError(f"mode '{mode}' needs one query vector per query id")
        if mode != "dense" and (texts is None or len(texts) != len(query_ids) or self.text is None):
            raise ValueError(f"mode '{mode}' needs query texts and a collection built with texts")
        depth = k if mode == "dense" else max(k, candidates or 2 * k)
        dense = None
        if mode != "keyword":
            # ef None: the calibrated beam width (at least the depth), with the profile's re-rank.
            dense = self.index.search(np.ascontiguousarray(vectors, dtype=np.float32), k=depth, threads=threads,
                                      ef=None if ef is None else max(ef, depth))
        run: Run = {}
        for i, qid in enumerate(query_ids):
            if mode == "dense":
                pairs = [(int(l), -float(d)) for l, d in zip(*[x[i] for x in dense]) if l >= 0]
            else:
                keyword_labels, keyword_scores = self.text.search(texts[i], k=depth)
                if mode == "keyword":
                    pairs = list(zip(keyword_labels.tolist(), keyword_scores.tolist()))
                else:
                    dense_labels = [int(l) for l in dense[0][i] if l >= 0]
                    fused, scores = fuse_rankings([dense_labels, keyword_labels.tolist()], k=k, rrf_k=rrf_k,
                                                  weights=list(weights))
                    pairs = list(zip(fused.tolist(), scores.tolist()))
            run[qid] = [(self._ids[l], s) for l, s in pairs[:k] if self._ids[l] is not None]
        return run

    def save(self, folder: str) -> None:
        """Saves the index (with its configuration), the keyword index and the document ids."""
        os.makedirs(folder, exist_ok=True)
        self.index.save(os.path.join(folder, "vectors.epzero"))
        if self.text is not None:
            self.text.save(os.path.join(folder, "text.epzero"))
        with open(os.path.join(folder, "ids.json"), "w", encoding="utf-8") as f:
            json.dump(self._ids, f)

    @classmethod
    def load(cls, folder: str) -> Collection:
        with open(os.path.join(folder, "ids.json"), encoding="utf-8") as f:
            ids = json.load(f)
        text_path = os.path.join(folder, "text.epzero")
        text = TextIndex.load(text_path) if os.path.exists(text_path) else None
        return cls(Index.load(os.path.join(folder, "vectors.epzero")), text, ids)


# ---- TREC files ----

def write_run(run: Run, path: str, tag: str = "epzero") -> None:
    """A TREC run file: `query Q0 document rank score tag`, ranks from 1, best first."""
    if any(c.isspace() for c in tag):
        raise ValueError("the run tag must not contain whitespace")
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for qid, ranked in run.items():
            for rank, (doc_id, score) in enumerate(ranked, start=1):
                f.write(f"{qid} Q0 {doc_id} {rank} {score:.6f} {tag}\n")


def read_run(path: str) -> Run:
    """Reads a TREC run file, ordered by each query's ranks."""
    rows: dict[str, list[tuple[int, str, float]]] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.split()
            if len(parts) < 6:
                continue
            qid, _, doc_id, rank, score = parts[:5]
            rows.setdefault(qid, []).append((int(rank), doc_id, float(score)))
    return {qid: [(d, s) for _, d, s in sorted(r)] for qid, r in rows.items()}


def read_qrels(path: str) -> Qrels:
    """Reads relevance judgements: TREC (`query 0 document relevance`) or BEIR's TSV with a header
    (`query-id corpus-id score`)."""
    qrels: Qrels = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.split()
            if not parts or parts[0] in ("query-id", "query_id", "qid"):
                continue
            if len(parts) == 4:
                qid, _, doc_id, rel = parts
            elif len(parts) == 3:
                qid, doc_id, rel = parts
            else:
                raise ValueError(f"{path}: unexpected line {line.strip()!r}")
            qrels.setdefault(qid, {})[doc_id] = int(float(rel))
    return qrels


# ---- Measures ----

def _measure(name: str, ranked: list[str], judged: dict[str, int]) -> float:
    relevant = {d for d, r in judged.items() if r > 0}
    base, _, cut = name.partition("@")
    depth = int(cut) if cut else len(ranked)
    top = ranked[:depth]
    if base == "nDCG":
        dcg = sum(judged.get(d, 0) / math.log2(i + 2) for i, d in enumerate(top) if judged.get(d, 0) > 0)
        ideal = sorted((r for r in judged.values() if r > 0), reverse=True)[:depth]
        idcg = sum(r / math.log2(i + 2) for i, r in enumerate(ideal))
        return dcg / idcg if idcg > 0 else 0.0
    if base == "R":
        return sum(1 for d in top if d in relevant) / len(relevant)
    if base == "P":
        return sum(1 for d in top if d in relevant) / depth
    if base == "MAP":
        hits, total = 0, 0.0
        for i, d in enumerate(top):
            if d in relevant:
                hits += 1
                total += hits / (i + 1)
        return total / len(relevant)
    if base in ("MRR", "RR"):
        return next((1.0 / (i + 1) for i, d in enumerate(top) if d in relevant), 0.0)
    raise ValueError(f"unknown measure {name}: use nDCG@k, R@k, P@k, MAP or MRR@k")


def evaluate(run: Run, qrels: Qrels, measures: Sequence[str] = DEFAULT_MEASURES, per_query: bool = False):
    """Mean of each measure over the queries with at least one relevant document (a query missing from
    the run counts 0), or with per_query, {measure: {query: value}}."""
    queries = [q for q, judged in qrels.items() if any(r > 0 for r in judged.values())]
    values = {m: {} for m in measures}
    for qid in queries:
        # trec_eval's order, whatever the run's: by score, ties by document id, both descending.
        ranked = [d for d, _ in sorted(run.get(qid, []), key=lambda p: (p[1], p[0]), reverse=True)]
        for m in measures:
            values[m][qid] = _measure(m, ranked, qrels[qid])
    if per_query:
        return values
    return {m: (sum(v.values()) / len(v) if v else 0.0) for m, v in values.items()}


# ---- Datasets ----

def load_beir(folder: str, split: str = "test"):
    """A BEIR dataset folder (corpus.jsonl, queries.jsonl, qrels/<split>.tsv): returns (corpus, queries,
    qrels), where corpus maps a document id to "title text" and queries maps a query id to its text,
    keeping only the queries that have judgements in `split`."""
    corpus = {}
    with open(os.path.join(folder, "corpus.jsonl"), encoding="utf-8") as f:
        for line in f:
            doc = json.loads(line)
            corpus[str(doc["_id"])] = (doc.get("title", "") + " " + doc.get("text", "")).strip()
    qrels = read_qrels(os.path.join(folder, "qrels", f"{split}.tsv"))
    queries = {}
    with open(os.path.join(folder, "queries.jsonl"), encoding="utf-8") as f:
        for line in f:
            q = json.loads(line)
            if str(q["_id"]) in qrels:
                queries[str(q["_id"])] = q["text"]
    return corpus, queries, qrels
