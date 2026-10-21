"""Oryx: an HNSW vector database written from scratch in C++20.

    import numpy as np
    import oryxdb

    index = oryxdb.Index(dim=128, metric="cosine")
    index.add(vectors)                      # labels default to 0..n-1
    labels, distances = index.search(queries, k=10, ef=64)

`Index` is the in-memory HNSW index (filtered search, 8-bit quantization, exact search and miss
explanations); `Database` adds a write-ahead log and crash recovery; `RecallGuard` measures recall
on a sample of live queries and moves the search effort to hold a target. `TextIndex` is keyword
search (BM25), and `hybrid_search` merges it with vector search. A LangChain vector store is in
`oryxdb.langchain` (pip install "oryxdb[langchain]").
"""

from importlib.metadata import PackageNotFoundError, version

from ._core import Database, Filter, Index, RecallGuard, TextIndex, fuse_rankings, hybrid_search, simd_level

try:
    __version__ = version("oryxdb")
except PackageNotFoundError:  # running from a source tree
    __version__ = "0.0.0"

__all__ = [
    "Database",
    "Filter",
    "Index",
    "RecallGuard",
    "TextIndex",
    "fuse_rankings",
    "hybrid_search",
    "simd_level",
    "__version__",
]
