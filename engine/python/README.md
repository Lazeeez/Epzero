# epzero

An HNSW vector database written from scratch in C++20, with Python bindings.

- **Fast search:** hand-written SIMD kernels (AVX2, AVX-512, NEON), chosen at runtime, and
  lock-free reads. On SIFT1M it serves 1.13× hnswlib's and 1.17× FAISS's queries per second at
  equal recall, on the same machine.
- **Filtered search that picks its own strategy:** an exact scan, a filtered graph walk, or a
  two-hop walk (the idea behind ACORN), chosen per query by a cost model.
- **Recall you can see:** `RecallGuard` measures recall on a sample of live queries with a
  confidence interval and moves the search effort to hold a target; `explain()` says why any given
  item was not returned. With `per_query=True` each query also stops as soon as its answer settles,
  so easy queries cost less and hard ones search further.
- **Hybrid search:** `TextIndex` (BM25 keyword search) and `hybrid_search()`, which merges it
  with vector search by reciprocal rank fusion.
- **Durable:** `Database` adds a write-ahead log with crash recovery and checkpoints.
- **8-bit quantization** with exact re-ranking.
- **A LangChain vector store** in `epzero.langchain`.

```bash
pip install epzero                 # the engine
pip install "epzero[langchain]"    # plus the LangChain vector store
```

## Quick start

```python
import numpy as np
import epzero

vectors = np.random.rand(100_000, 128).astype(np.float32)
index = epzero.Index(dim=128, metric="l2", max_elements=100_000, M=16, ef_construction=200)
index.add(vectors)                                   # labels 0..n-1 (or pass labels=)

labels, distances = index.search(vectors[:5], k=10, ef=64)

# Filtered search: compile the allowed labels once, reuse across queries.
allowed = index.filter(np.arange(0, 100_000, 50))
labels, distances = index.search(vectors[:5], k=10, filter=allowed)

# Hold recall at 0.95 as the workload drifts, checking 1% of queries; per_query=True lets each
# query stop once its answer settles (cheaper for the same recall).
guard = epzero.RecallGuard(index, target=0.95, sample_rate=0.01, reference_ef=1024, per_query=True)
labels, distances = guard.search(vectors[:5], k=10)
print(guard.report())

# Optional: measure the best memory-prefetch setting for this machine and data (not saved).
index.tune_prefetch(vectors[:500])

index.save("vectors.epzero")
index = epzero.Index.load("vectors.epzero")

# Hybrid: keyword search on the same labels, merged with the vector search.
text = epzero.TextIndex()
text.add(None, documents)                            # labels 0..n-1, matching the index
labels, scores = epzero.hybrid_search(index, text, query_vector, "error E-1042 at checkout", k=10)
```

## LangChain

```python
from langchain_openai import OpenAIEmbeddings
from epzero.langchain import EpzeroVectorStore

store = EpzeroVectorStore.from_texts(texts, OpenAIEmbeddings(), metadatas=metadatas)
store.similarity_search("how do I reset my password?", k=4, filter={"source": "faq"})
store.max_marginal_relevance_search("pricing", k=4, fetch_k=20)
store.hybrid_search("error E-1042 at checkout", k=4)   # keywords + vectors
retriever = store.as_retriever(search_kwargs={"k": 8})

store.save_local("./store")
store = EpzeroVectorStore.load_local("./store", OpenAIEmbeddings())
```

Metadata filters are a dict of required values (a list means "any of"), or a callable on the
metadata. They are compiled into an Epzero filter, so the engine chooses how to search the allowed
documents instead of over-fetching and discarding. The store passes LangChain's standard
vector-store test suite.

## Platforms

Wheels for Linux (x86-64 and ARM64, glibc 2.28+) and macOS (Apple silicon and Intel), Python 3.10
to 3.13. Elsewhere, `pip install epzero` builds from source (CMake 3.20+, a C++20 compiler).

Live demo, benchmarks and design notes: https://epzero.dev/ · MIT license.
