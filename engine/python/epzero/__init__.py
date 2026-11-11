"""Epzero: an HNSW vector database written from scratch in C++20.

    import numpy as np
    import epzero

    index = epzero.build(vectors, metric="cosine")   # configured for the data and this machine
    labels, distances = index.search(queries, k=10)  # at the effort calibrated for recall 0.95
    print(index.plan)                                # every setting, and why; pass any to override

`build` chooses compression, the build passes, the re-rank and the beam width from the data, and
times prefetching on this CPU (see epzero.auto); the settings are saved with the index. `Index` is
the same engine configured by hand:
the in-memory HNSW index (filtered search, 8-bit quantization, exact search and miss
explanations); `Database` adds a write-ahead log and crash recovery; `RecallGuard` measures recall
on a sample of live queries and moves the search effort to hold a target. `TextIndex` is keyword
search (BM25), and `hybrid_search` merges it with vector search. A LangChain vector store is in
`epzero.langchain` (pip install "epzero[langchain]").
"""

from importlib.metadata import PackageNotFoundError, version

from ._core import Database, Filter, Index, Metadata, RecallGuard, TextIndex, fuse_rankings, hybrid_search, machine, simd_level
from .auto import Plan, build, plan

try:
    __version__ = version("epzero")
except PackageNotFoundError:  # running from a source tree
    __version__ = "0.0.0"

__all__ = [
    "Database",
    "Filter",
    "Index",
    "Metadata",
    "Plan",
    "RecallGuard",
    "TextIndex",
    "build",
    "fuse_rankings",
    "hybrid_search",
    "machine",
    "plan",
    "simd_level",
    "__version__",
]
