# epzero

An HNSW vector database written from scratch in C++20, with Python bindings.

- **Configures itself:** `epzero.build(vectors)` picks compression, the graph build and the search
  effort for your data and recall target, times memory prefetching on your CPU, saves the choices
  inside the index and tells you why it made each one. Any setting can be overridden.
- **Fast search:** hand-written SIMD kernels (AVX2, AVX-512, VNNI, NEON) chosen at runtime, 8-bit
  codes, an optional walk on principal directions, and lock-free reads. On dedicated Intel Xeon Gold
  6226 and AMD EPYC 7513 servers, one thread, at equal recall: 1.4-4.2× the queries per second of
  Glass, the fastest library we compared, on GIST1M, OpenAI DBpedia and DEEP10M, and 1.0-1.3× on
  GloVe-100 (with each dataset's best settings; see the project's scoreboard for every loss).
- **Filtered search that picks its own strategy:** an exact scan, a filtered graph walk, or a
  two-hop walk (the idea behind ACORN), chosen per query by a cost model.
- **Recall you can see:** `RecallGuard` measures recall on a sample of live queries with a
  confidence interval and moves the search effort to hold a target; `explain()` says why any given
  item was not returned. With `per_query=True` each query also stops as soon as its answer settles,
  so easy queries cost less and hard ones search further.
- **Hybrid search:** `TextIndex` (BM25 keyword search) and `hybrid_search()`, which merges it
  with vector search by reciprocal rank fusion.
- **Durable:** `Database` adds a write-ahead log with crash recovery and checkpoints.
- **8-bit quantization** with exact re-ranking, stopped early per query once the rest can't reach
  the top k.
- **A LangChain vector store** in `epzero.langchain`.

```bash
pip install epzero                 # the engine
pip install "epzero[langchain]"    # plus the LangChain vector store
pip install "epzero[llamaindex]"   # plus the LlamaIndex vector store
pip install "epzero[haystack]"     # plus the Haystack document store and retriever
```

## Quick start: let it configure itself

```python
import numpy as np
import epzero

vectors = np.random.rand(100_000, 768).astype(np.float32)
index = epzero.build(vectors, metric="cosine")       # chooses every setting (below)
labels, distances = index.search(vectors[:5], k=10)  # at the effort calibrated for recall 0.95
print(index.plan)                                    # what it chose, and why
```

`build` decides from the data: 8-bit compression, whether to walk on principal directions (high
dimensions with a concentrated spectrum), the build passes, the re-rank, and the beam width that
reaches your recall target, measured on rows held out of the graph. From the machine: the SIMD
kernel, the prefetch setting (timed on this CPU) and every core. Change anything:

```python
index = epzero.build(vectors, metric="cosine", recall=0.99)           # a different target
index = epzero.build(vectors, metric="cosine", queries=my_queries)    # calibrate on real queries
index = epzero.build(vectors, metric="cosine", nav_dim=0, refine=False, M=32)  # override choices
plan = epzero.plan(vectors, metric="cosine")         # look before building, edit, then pass it in
index = epzero.build(vectors, plan=plan.replace(ef_construction=400))
labels, distances = index.search(queries, k=50, ef=200)               # per-call overrides
index.save("my.epz")                                 # the settings and reasons are saved inside it
index = epzero.Index.load("my.epz")                  # search() uses them again
```

`build` returns an ordinary `epzero.Index`: filters, deletes, `explain()` and `RecallGuard` all use
its settings. A saved index re-applies its prefetch setting only on the same kind of CPU (call
`tune_prefetch()` on another). The decisions are made in the C++ engine (`epzero/auto.h`), so C++
users get the same `build_auto()`.

If your queries differ from your data (text queries against image vectors, say), pass a few
hundred of them as `queries=`: recall calibrated on data rows is optimistic for those.

**Metadata.** Fields per label, filtered in the engine:

```python
meta = epzero.Metadata()
meta.set_many(labels, [{"source": "faq", "year": 2024}, ...])
allowed = meta.select({"source": ["faq", "docs"], "year": {"$gte": 2022}})
labels, distances = index.search(queries, k=10, filter=index.filter(allowed))
```

**After the build.** An index keeps changing; its calibration doesn't, by itself:

```python
print(index.health())          # growth since calibration, deleted share, fit of the directions, advice
print(index.check_queries(queries))   # do these queries look like the data? (text vs images: no)
index.recalibrate(queries)     # re-measure the beam width on real queries; saved in the profile
guard = epzero.RecallGuard(index, sample_rate=0.01)   # or measure live recall and adjust as you go
```

## Configured by hand

```python
vectors = np.random.rand(100_000, 128).astype(np.float32)
index = epzero.Index(dim=128, metric="l2", max_elements=100_000, M=16, ef_construction=200)
index.add(vectors)                                   # labels 0..n-1 (or pass labels=); grows as needed

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

## LlamaIndex

```python
from llama_index.core import StorageContext, VectorStoreIndex
from epzero.llamaindex import EpzeroVectorStore

store = EpzeroVectorStore(metric="cosine")
index = VectorStoreIndex.from_documents(documents, storage_context=StorageContext.from_defaults(vector_store=store))
index.as_query_engine().query("How do I reset my password?")
store.persist("./epzero-store")        # EpzeroVectorStore.from_persist_path("./epzero-store")
```

Metadata filters (`==`, `!=`, `>`, `>=`, `<`, `<=`, `in`, `nin`, combined with AND or OR) run in the
engine, as do `doc_ids` and `node_ids`; `delete_ref_doc` removes a document's nodes.

## Haystack

```python
from haystack import Pipeline
from epzero.haystack import EpzeroDocumentStore, EpzeroEmbeddingRetriever

store = EpzeroDocumentStore(metric="cosine")
store.write_documents(documents)       # Documents with embeddings
pipeline = Pipeline()
pipeline.add_component("retriever", EpzeroEmbeddingRetriever(document_store=store, top_k=5))
pipeline.run({"retriever": {"query_embedding": embedding,
                            "filters": {"field": "meta.lang", "operator": "==", "value": "en"}}})
```

Filters use Haystack's syntax and run in the engine wherever that gives exactly Haystack's answer;
the rest (ranges on dates, `None`) go to Haystack's reference implementation. The store passes
Haystack's document-store test suite.

## Platforms

Wheels for Linux (x86-64 and ARM64, glibc 2.28+), macOS (Apple silicon and Intel) and, from 0.5,
Windows (x64), Python 3.10 to 3.13. Elsewhere, `pip install epzero` builds from source (CMake 3.20+, a C++20 compiler).

Live demo, benchmarks and design notes: https://epzero.dev/ · MIT license.
