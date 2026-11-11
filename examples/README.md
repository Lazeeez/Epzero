# Examples

Notebooks that run as they are (executed before they were committed, outputs included):

| Notebook | What it shows |
|---|---|
| [`01_quickstart.ipynb`](01_quickstart.ipynb) | `epzero.build()` and the plan it chooses, search and recall, overrides, metadata filters, `health()` and `recalibrate()` after the index grows, saving and loading |
| [`02_rag_langchain.ipynb`](02_rag_langchain.ipynb) | Retrieval for RAG with the LangChain vector store and a local sentence-embedding model: similarity with scores, metadata filters, maximal marginal relevance, and a prompt built from the results |
| [`03_llamaindex_and_haystack.ipynb`](03_llamaindex_and_haystack.ipynb) | The same corpus through LlamaIndex (`VectorStoreIndex` with filters) and a Haystack pipeline |
| [`04_retrieval_evaluation.ipynb`](04_retrieval_evaluation.ipynb) | `epzero.ir` on BEIR SciFact: keyword, dense and hybrid runs scored with nDCG@10, MAP and recall as trec_eval does, per-query comparison, a TREC run file, and a collection that changes |

```bash
pip install "epzero[langchain,llamaindex,haystack]" sentence-transformers jupyter
jupyter lab examples/
```

The second, third and fourth download `all-MiniLM-L6-v2` (about 90 MB) on first use; the fourth also
downloads SciFact (3 MB) and embeds it, about two minutes on a laptop CPU.
