<script lang="ts">
  import { highlight } from "../lib/highlight";
  import { reveal } from "../lib/reveal";
  import { router } from "../lib/router.svelte";
  import { spotlight } from "../lib/spotlight";

  // What 0.5 adds (on main, in the next release), and what 0.4 already does. Each with the figure that backs it.
  const features = [
    {
      title: "Configures itself",
      since: "0.4",
      text: "epzero.build chooses 8-bit codes, principal directions, build passes, the re-rank and the beam width for your recall target, and times prefetching on your CPU.",
      figure: "Within 1–5% of hand-tuned settings on three CPUs",
    },
    {
      title: "Keeps itself accurate",
      since: "0.5",
      text: "health() says when an index needs recalibrating, requantizing, compacting or rebuilding, and each fix runs in place.",
      figure: "After a 4 sd drift: beam width 200 → 12, in half a rebuild's time",
    },
    {
      title: "Spots unlike queries",
      since: "0.5",
      text: "check_queries() measures whether queries look like the data. Text searching images doesn't, and recall calibrated on the data then overstates theirs.",
      figure: "Recall 0.74–0.80 → 0.945–0.960 once recalibrated on the queries",
    },
    {
      title: "Grows in place",
      since: "0.5",
      text: "add() past capacity copies the nodes into larger blocks and keeps the graph, so stores no longer rebuild themselves as they fill.",
      figure: "40,000 documents into LangChain: 6.0 s → 3.4 s",
    },
    {
      title: "Metadata in the engine",
      since: "0.5",
      text: "Typed fields per label with an inverted index; equality, set and range filters compile into the search's own filter.",
      figure: "{\"year\": {\"$gte\": 2022}, \"lang\": [\"en\", \"de\"]}",
    },
    {
      title: "LangChain, LlamaIndex, Haystack",
      since: "0.5",
      text: "A vector store for each, with filters run in the engine. LlamaIndex and Haystack are new in 0.5.",
      figure: "Each passes its framework's own test suite",
    },
    {
      title: "Retrieval evaluation",
      since: "0.5",
      text: "epzero.ir: documents under string ids, dense, keyword and hybrid runs, TREC run files, and nDCG, MAP, recall and MRR scored exactly as trec_eval does.",
      figure: "Hybrid nDCG@10 0.713 on BEIR SciFact",
    },
    {
      title: "Stemmed keyword search",
      since: "0.5",
      text: "BM25 reduces English words to their Porter stems, so “connections” finds “connected”; the same algorithm Lucene uses.",
      figure: "BM25 nDCG@10 0.660 → 0.688 on SciFact",
    },
    {
      title: "Linux, macOS and Windows",
      since: "0.5",
      text: "Wheels for x86-64 and ARM64 Linux, Apple silicon and Intel Macs, and from 0.5 Windows, for Python 3.10–3.13. 16-bit full vectors halve memory, opt-in.",
      figure: "One C++ engine, built with GCC, Clang and MSVC",
    },
  ];

  const examples = [
    {
      id: "build",
      name: "Build and search",
      note: "No settings needed. The plan lists every choice and the reason for it, and is saved inside the index.",
      code: `import numpy as np
import epzero

vectors = np.load("embeddings.npy")                  # (n, dim) float32
index = epzero.build(vectors, metric="cosine")       # configured for this data and this machine
labels, distances = index.search(queries, k=10)      # at the effort calibrated for recall 0.95
print(index.plan)                                    # every choice, and why

index = epzero.build(vectors, metric="cosine", recall=0.99)          # another target
index = epzero.build(vectors, metric="cosine", queries=my_queries)   # calibrate on real queries
index.save("my.epz")                                 # the settings travel with the index
index = epzero.Index.load("my.epz")`,
    },
    {
      id: "upkeep",
      name: "Keep it accurate",
      since: "0.5",
      note: "An index keeps changing after the build. health() says what it needs, in plain words; each fix runs in place.",
      code: `print(index.health())                 # growth, deletes, code drift, fit of the directions, advice
print(index.check_queries(queries))   # do these queries look like the data?
index.recalibrate(queries)            # re-measure the beam width on real queries
index.requantize()                    # refit the 8-bit codes after drift, reconnect the graph
index.add(new_vectors)                # grows in place past its capacity`,
    },
    {
      id: "filters",
      name: "Metadata filters",
      since: "0.5",
      note: "Fields per label, filtered in the engine. The search then picks its own strategy for the allowed set: a scan, a filtered walk or a two-hop walk.",
      code: `meta = epzero.Metadata()
meta.set_many(labels, [{"lang": "en", "year": 2024, "source": "docs"}, ...])
allowed = meta.select({"lang": ["en", "de"], "year": {"$gte": 2022}})
labels, distances = index.search(queries, k=10, filter=index.filter(allowed))`,
    },
    {
      id: "hybrid",
      name: "Hybrid search",
      note: "Keyword search beside the vectors, merged by reciprocal rank fusion, for the exact terms embeddings blur.",
      code: `text = epzero.TextIndex()              # BM25; English words stemmed from 0.5
text.add(None, documents)              # labels 0..n-1, matching the index
labels, scores = epzero.hybrid_search(index, text, query_vector, "error E-1042 at checkout", k=10)`,
    },
    {
      id: "ir",
      name: "Evaluate retrieval",
      since: "0.5",
      note: "Retrieval experiments end to end, scored exactly as trec_eval scores them, with run files ready for TREC or CLEF.",
      code: `from epzero import ir

corpus, queries, qrels = ir.load_beir("scifact")              # a BEIR dataset folder
collection = ir.Collection.build(list(corpus), doc_vectors, texts=list(corpus.values()), metric="cosine")
run = collection.retrieve(list(queries), vectors=query_vectors, texts=list(queries.values()),
                          k=100, mode="hybrid")                # or "dense", "keyword"
print(ir.evaluate(run, qrels))                                 # nDCG@10, R@100, MAP, MRR@10
ir.write_run(run, "run.txt", tag="my-system")                  # TREC format`,
    },
    {
      id: "langchain",
      name: "LangChain",
      install: 'pip install "epzero[langchain]"',
      note: "Similarity with scores, maximal marginal relevance, metadata filters in the engine, hybrid search, and saving to a folder.",
      code: `from epzero.langchain import EpzeroVectorStore

store = EpzeroVectorStore.from_texts(texts, embeddings, metadatas=metadatas)
store.similarity_search("reset my password", k=4, filter={"source": "faq"})
store.hybrid_search("error E-1042 at checkout", k=4)
retriever = store.as_retriever(search_kwargs={"k": 8})`,
    },
    {
      id: "llamaindex",
      name: "LlamaIndex",
      since: "0.5",
      install: 'pip install "epzero[llamaindex]"',
      note: "Nodes with their text and metadata, LlamaIndex's filters run in the engine, delete by document, and persistence.",
      code: `from llama_index.core import StorageContext, VectorStoreIndex
from epzero.llamaindex import EpzeroVectorStore

store = EpzeroVectorStore(metric="cosine")
index = VectorStoreIndex.from_documents(documents, storage_context=StorageContext.from_defaults(vector_store=store))
index.as_query_engine().query("How do I reset my password?")
store.persist("./epzero-store")`,
    },
    {
      id: "haystack",
      name: "Haystack",
      since: "0.5",
      install: 'pip install "epzero[haystack]"',
      note: "A document store and an embedding retriever for pipelines, with Haystack's filter syntax run in the engine.",
      code: `from haystack import Pipeline
from epzero.haystack import EpzeroDocumentStore, EpzeroEmbeddingRetriever

store = EpzeroDocumentStore(metric="cosine")
store.write_documents(documents)       # Documents with embeddings
pipeline = Pipeline()
pipeline.add_component("retriever", EpzeroEmbeddingRetriever(document_store=store, top_k=5))
pipeline.run({"retriever": {"query_embedding": embedding,
                            "filters": {"field": "meta.lang", "operator": "==", "value": "en"}}})`,
    },
  ];

  // BEIR, all-MiniLM-L6-v2 embeddings, scored with epzero.ir (examples/04_retrieval_evaluation.ipynb).
  const beir = [
    { name: "SciFact", docs: "5,183", queries: 300, keyword: [0.688, 0.931], dense: [0.645, 0.925], hybrid: [0.713, 0.964] },
    { name: "NFCorpus", docs: "3,633", queries: 323, keyword: [0.321, 0.248], dense: [0.313, 0.299], hybrid: [0.354, 0.325] },
  ];

  let active = $state("build");
  const current = $derived(examples.find((e) => e.id === active)!);

  let copied = $state("");
  async function copy(text: string, key: string) {
    try {
      await navigator.clipboard.writeText(text);
      copied = key;
      setTimeout(() => copied === key && (copied = ""), 1600);
    } catch {
      copied = "";
    }
  }

  // Arrow keys move between tabs, as in any tab list.
  function onTabKey(event: KeyboardEvent) {
    const i = examples.findIndex((e) => e.id === active);
    const next = event.key === "ArrowRight" ? i + 1 : event.key === "ArrowLeft" ? i - 1 : event.key === "Home" ? 0 : event.key === "End" ? examples.length - 1 : -2;
    if (next === -2) return;
    event.preventDefault();
    active = examples[(next + examples.length) % examples.length].id;
    (event.currentTarget as HTMLElement).querySelector<HTMLElement>(`#tab-${active}`)?.focus();
  }
</script>

<div class="download">
  <header class="head container">
    <p class="eyebrow">Download</p>
    <h1><span class="gradient-text">pip install epzero</span><br />and it configures itself.</h1>
    <p class="section-lede">
      The engine behind this site, from Python: it chooses its own settings for your data and your machine, says why, keeps itself accurate as
      the data changes, and plugs into LangChain, LlamaIndex and Haystack. Open source, MIT licensed.
    </p>
    <div class="install" use:spotlight>
      <code><span class="prompt" aria-hidden="true">$</span> pip install epzero</code>
      <button class="copy" onclick={() => copy("pip install epzero", "install")} aria-label="Copy the install command">
        {copied === "install" ? "Copied" : "Copy"}
      </button>
    </div>
    <p class="release">
      <span class="tag now">0.4</span> on PyPI now ·
      <span class="tag next">0.5</span> marks what the next release adds, finished and tested on the main branch.
    </p>
  </header>

  <section class="container block" aria-labelledby="features-title">
    <h2 id="features-title" class="block-title" use:reveal>What it does</h2>
    <ul class="features">
      {#each features as f, i (f.title)}
        <li class="card feature" use:reveal style:--reveal-delay="{(i % 3) * 0.06}s" use:spotlight>
          <span class="tag" class:next={f.since === "0.5"} class:now={f.since !== "0.5"}>{f.since}</span>
          <h3>{f.title}</h3>
          <p>{f.text}</p>
          <p class="figure num">{f.figure}</p>
        </li>
      {/each}
    </ul>
  </section>

  <section class="container block" aria-labelledby="code-title">
    <h2 id="code-title" class="block-title" use:reveal>In code</h2>
    <div class="card panel code-panel" use:reveal>
      <!-- svelte-ignore a11y_interactive_supports_focus -->
      <div class="tabs" role="tablist" aria-label="Examples" tabindex="-1" onkeydown={onTabKey}>
        {#each examples as e (e.id)}
          <button
            id="tab-{e.id}"
            role="tab"
            aria-selected={active === e.id}
            aria-controls="panel-code"
            tabindex={active === e.id ? 0 : -1}
            class:on={active === e.id}
            onclick={() => (active = e.id)}
          >
            {e.name}{#if e.since}<span class="tag next small">{e.since}</span>{/if}
          </button>
        {/each}
      </div>
      <div id="panel-code" role="tabpanel" aria-labelledby="tab-{current.id}">
        <p class="note">{current.note}</p>
        {#if current.install}
          <p class="extra"><code>{current.install}</code></p>
        {/if}
        <div class="snippet">
          <button class="copy" onclick={() => copy(current.code, current.id)} aria-label="Copy this example">
            {copied === current.id ? "Copied" : "Copy"}
          </button>
          <pre><code>{@html highlight(current.code)}</code></pre>
        </div>
      </div>
    </div>
  </section>

  <section class="container block" aria-labelledby="beir-title">
    <h2 id="beir-title" class="block-title" use:reveal>Retrieval quality, scored as TREC scores it</h2>
    <p class="caption lead" use:reveal>
      Two BEIR datasets, embedded with the small all-MiniLM-L6-v2 model and searched with <code>epzero.ir</code>. The dense scores match the model's
      published ones exactly; hybrid search beats both of its halves.
    </p>
    <div class="card panel" use:reveal use:spotlight>
      <div class="table-wrap">
        <table>
          <thead>
            <tr><th>Dataset</th><th>Keyword (BM25)</th><th>Dense</th><th>Hybrid</th></tr>
          </thead>
          <tbody>
            {#each beir as b (b.name)}
              <tr>
                <td><b>{b.name}</b><span class="meta">{b.docs} documents · {b.queries} queries</span></td>
                {#each [b.keyword, b.dense, b.hybrid] as [ndcg, recall], i (i)}
                  <td class="num" class:best={i === 2}>
                    <span class="score">{ndcg.toFixed(3)}</span><span class="meta">recall@100 {recall.toFixed(3)}</span>
                  </td>
                {/each}
              </tr>
            {/each}
          </tbody>
        </table>
      </div>
    </div>
    <p class="caption" use:reveal>
      nDCG@10, the measure BEIR and TREC rank by. With English words stemmed, keyword search is above BEIR's Anserini baseline on SciFact (0.665) and
      close to it on NFCorpus (0.325); on hybrid, the claims share exact terms with their abstracts (gene names, drugs) that the small
      model blurs, while the embeddings find paraphrases that share no word. The same toolkit carries Epzero's research on how retrieval holds up as
      a collection changes over time.
    </p>
  </section>

  <section class="container block next-steps" aria-label="Where to go next">
    <div class="card panel ctas" use:reveal use:spotlight>
      <div>
        <h2>See it before you install it</h2>
        <p>Search 29,825 photos with the same engine in your browser, or benchmark it on your own embeddings, configured automatically.</p>
      </div>
      <div class="actions">
        <a class="btn btn-primary" href={router.href("try")}>Open the playground <span aria-hidden="true">→</span></a>
        <a class="btn btn-ghost" href={router.href("benchmarks", "#lab")}>Benchmark your vectors</a>
        <a class="btn btn-ghost" href="https://pypi.org/project/epzero/" rel="noopener">epzero on PyPI</a>
      </div>
    </div>
  </section>
</div>

<style>
  .download {
    padding: calc(var(--nav-h) + 72px) 0 40px;
  }
  .head h1 {
    font-size: clamp(38px, 6vw, 72px);
    letter-spacing: -0.045em;
    line-height: 1.02;
    text-wrap: balance;
  }
  .install {
    display: inline-flex;
    align-items: center;
    gap: 16px;
    margin-top: 32px;
    padding: 10px 10px 10px 18px;
    border: 1px solid var(--border-strong);
    border-radius: 14px;
    background: var(--surface);
  }
  .install code {
    font: 500 16px var(--mono);
    color: var(--text);
  }
  .install .prompt {
    margin-right: 8px;
    color: var(--muted);
  }
  .copy {
    padding: 6px 12px;
    border: 1px solid var(--border);
    border-radius: 9px;
    background: var(--glass-fill);
    color: var(--text-2);
    font: 600 12.5px var(--font);
    cursor: pointer;
    transition:
      color 0.2s,
      border-color 0.2s;
  }
  .copy:hover {
    color: var(--text);
    border-color: var(--text-2);
  }
  .release {
    margin: 16px 0 0;
    color: var(--text-2);
    font-size: 14px;
  }
  .tag {
    display: inline-block;
    padding: 1px 7px;
    border-radius: 999px;
    font: 600 11.5px var(--mono);
    vertical-align: 1px;
  }
  .tag.now {
    color: var(--text-2);
    background: color-mix(in srgb, var(--text-2) 12%, transparent);
  }
  .tag.next {
    color: var(--accent-strong);
    background: color-mix(in srgb, var(--accent) 16%, transparent);
  }
  .tag.small {
    margin-left: 7px;
    padding: 0 6px;
    font-size: 10.5px;
  }
  .block {
    margin-top: clamp(56px, 8vw, 96px);
  }
  .block-title {
    font-size: clamp(26px, 3.4vw, 38px);
  }
  .features {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 16px;
    margin: 28px 0 0;
    padding: 0;
    list-style: none;
  }
  .feature {
    display: flex;
    flex-direction: column;
    gap: 8px;
    padding: 22px;
  }
  .feature h3 {
    margin: 4px 0 0;
    font-size: 18px;
    letter-spacing: -0.02em;
  }
  .feature p {
    margin: 0;
    color: var(--text-2);
    font-size: 14.5px;
    line-height: 1.6;
  }
  .feature .figure {
    margin-top: auto;
    padding-top: 12px;
    border-top: 1px solid var(--border);
    color: var(--text);
    font: 500 12.5px/1.5 var(--mono);
  }
  .feature .tag {
    align-self: flex-start;
  }
  .panel {
    padding: clamp(18px, 3vw, 28px);
    background: color-mix(in srgb, var(--surface-solid) 72%, transparent);
  }
  .code-panel {
    margin-top: 28px;
  }
  .tabs {
    display: flex;
    flex-wrap: wrap;
    gap: 6px;
    margin-bottom: 18px;
  }
  .tabs button {
    display: inline-flex;
    align-items: center;
    padding: 8px 13px;
    border: 1px solid var(--border);
    border-radius: 10px;
    background: transparent;
    color: var(--text-2);
    font: 550 14px var(--font);
    cursor: pointer;
    transition:
      color 0.2s,
      border-color 0.2s,
      background 0.2s;
  }
  .tabs button:hover {
    color: var(--text);
  }
  .tabs button.on {
    border-color: var(--border-strong);
    background: var(--surface);
    color: var(--text);
  }
  .note {
    max-width: 760px;
    margin: 0 0 12px;
    color: var(--text-2);
    font-size: 15px;
    line-height: 1.6;
  }
  .extra {
    margin: 0 0 12px;
  }
  .extra code {
    font: 13px var(--mono);
    color: var(--text);
  }
  .snippet {
    position: relative;
  }
  .snippet .copy {
    position: absolute;
    top: 10px;
    right: 10px;
  }
  pre {
    margin: 0;
    padding: 18px 20px;
    overflow-x: auto;
    border: 1px solid var(--border);
    border-radius: 12px;
    background: var(--surface-solid);
    font: 13.5px/1.7 var(--mono);
    color: var(--text);
    tab-size: 4;
  }
  /* The site's inline-code chip, undone inside a block. */
  pre code {
    padding: 0;
    border: 0;
    border-radius: 0;
    background: none;
    font: inherit;
    color: inherit;
  }
  pre :global(.c) {
    color: var(--muted);
  }
  pre :global(.s) {
    color: var(--good);
  }
  pre :global(.k) {
    color: var(--accent-strong);
  }
  .caption {
    max-width: 760px;
    margin: 18px 0 0;
    color: var(--text-2);
    font-size: 15px;
    line-height: 1.65;
  }
  .caption.lead {
    margin: 10px 0 24px;
  }
  table {
    width: 100%;
    border-collapse: collapse;
    font-size: 14px;
  }
  th,
  td {
    padding: 12px 14px;
    border-bottom: 1px solid var(--border);
    text-align: right;
    white-space: nowrap;
  }
  th {
    color: var(--muted);
    font: 500 12.5px var(--mono);
  }
  th:first-child,
  td:first-child {
    text-align: left;
  }
  tbody tr:last-child td {
    border-bottom: 0;
  }
  td .meta {
    display: block;
    margin-top: 3px;
    color: var(--text-2);
    font-size: 12px;
  }
  td .score {
    font-size: 18px;
    font-weight: 650;
  }
  td.best .score {
    color: var(--accent-strong);
  }
  .ctas {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    justify-content: space-between;
    gap: 20px;
  }
  .ctas h2 {
    margin: 0;
    font-size: 22px;
  }
  .ctas p {
    max-width: 520px;
    margin: 6px 0 0;
    color: var(--text-2);
    font-size: 15px;
  }
  .actions {
    display: flex;
    flex-wrap: wrap;
    gap: 10px;
  }
  @media (max-width: 980px) {
    .features {
      grid-template-columns: 1fr 1fr;
    }
  }
  @media (max-width: 640px) {
    .features {
      grid-template-columns: 1fr;
    }
    .install {
      display: flex;
      justify-content: space-between;
    }
    pre {
      font-size: 12.5px;
    }
  }
</style>
