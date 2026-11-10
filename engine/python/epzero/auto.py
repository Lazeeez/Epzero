"""Automatic configuration: an index set up for your data and your machine, every choice overridable.

    import epzero

    index = epzero.build(vectors, metric="cosine")         # chooses everything
    labels, distances = index.search(queries, k=10)        # at the effort calibrated for recall 0.95
    print(index.plan)                                      # what it chose, and why

    index = epzero.build(vectors, metric="cosine", nav_dim=0, refine=False)   # override any choice
    plan = epzero.plan(vectors, metric="cosine")           # or look first, edit, then build
    index = epzero.build(vectors, metric="cosine", plan=plan.replace(recall=0.99))

Two kinds of choice:
- Data: compression, walking on principal directions, the build passes, the re-rank, and the beam
  width (ef) that reaches the recall target. Decided from the data, by the rules below.
- Machine: the SIMD kernel (picked when the library loads), the prefetch setting (timed on this
  CPU) and the thread count (every core). A saved index re-times prefetching when it is loaded on
  a different CPU.

The rules come from measurements in docs/performance.md, and are provisional where they say so.
"""

import dataclasses
import json
import math
import os
import platform

import numpy as np

from ._core import Index, simd_level

# Walking on principal directions paid off at 960 and 1536 dimensions (GIST1M, DBpedia OpenAI) and
# lost at 96-128 (DEEP, GloVe, SIFT), where an 8-bit record is already one or two cache lines.
# Nothing between was measured, so this threshold is provisional.
NAV_MIN_DIM = 256
# The directions kept: enough for this share of the variance. The winners kept about 80% (GIST at
# 96 directions: 0.83; DBpedia at 256: 0.80), as the re-rank repairs what the walk gets wrong.
NAV_VARIANCE = 0.80
# Below a 4x smaller record, the walk saves too little to pay for projecting the query and re-ranking.
NAV_MIN_SAVING = 4
NAV_ROUND = 32                # whole SIMD blocks
SPECTRUM_SAMPLE = 20_000      # rows used to measure the variance spectrum
TRAIN_SAMPLE = 200_000        # rows used to learn the 8-bit ranges
CALIBRATION_QUERIES = 500     # rows of the data used as queries to calibrate ef
TUNE_QUERIES = 1_000          # rows used to time prefetch settings
MAX_EF = 4_096


@dataclasses.dataclass(frozen=True)
class Plan:
    """Every setting of an automatically configured index. None means "decide when building"."""

    metric: str = "l2"
    quantization: str = "sq8"
    M: int = 16
    ef_construction: int = 200
    nav_dim: int = 0            # walk on 8-bit codes of this many principal directions (0: off)
    fast_build: bool = True     # build on the 8-bit codes
    refine: bool = True         # a second pass: every node searches the finished graph and reconnects
    fill_after: bool = False    # give free link slots to the closest two-hop neighbours
    rerank: int = 0             # candidates re-scored with full vectors (0: twice k)
    rerank_z: float = 0.0       # stop the re-rank early at this many standard deviations (0: off)
    recall: float = 0.95        # the target the beam width is calibrated for
    k: int = 10
    ef: int | None = None       # None: calibrated for `recall` at `k`
    prefetch: tuple | None = None  # (depth, lines); None: timed on this machine
    threads: int = 0            # 0: every core
    reasons: dict = dataclasses.field(default_factory=dict, compare=False)
    measured: dict = dataclasses.field(default_factory=dict, compare=False)

    def replace(self, **changes):
        """A copy with `changes`; a changed setting is recorded as yours, not the plan's."""
        unknown = set(changes) - {f.name for f in dataclasses.fields(self)}
        if unknown:
            raise TypeError(f"unknown settings: {', '.join(sorted(unknown))}")
        reasons = {**self.reasons, **{name: "set by you" for name in changes if name not in ("reasons", "measured")}}
        return dataclasses.replace(self, **changes, reasons=reasons)

    def __str__(self):
        width = max(len(f.name) for f in dataclasses.fields(self))
        lines = []
        for f in dataclasses.fields(self):
            if f.name in ("reasons", "measured"):
                continue
            value = getattr(self, f.name)
            why = self.reasons.get(f.name)
            lines.append(f"{f.name:<{width}}  {value!r:<14}" + (f"  {why}" if why else ""))
        lines += [f"{name:<{width}}  {value!r}" for name, value in self.measured.items()]
        return "\n".join(lines)


def system():
    """The facts about this machine that automatic settings depend on."""
    cpu = platform.processor() or platform.machine()
    try:
        with open("/proc/cpuinfo") as f:
            cpu = next((line.split(":", 1)[1].strip() for line in f if line.startswith("model name")), cpu)
    except OSError:
        pass
    return {"cpu": cpu, "cores": os.cpu_count() or 1, "simd": simd_level()}


def _rows(data):
    data = np.ascontiguousarray(data, dtype=np.float32)
    if data.ndim != 2 or len(data) == 0:
        raise ValueError("data must be a non-empty 2-D array (rows of vectors)")
    return data


def _sample(n, size, seed=0):
    return np.sort(np.random.default_rng(seed).choice(n, size=min(n, size), replace=False))


def _directions_for(data, metric, share):
    """How many principal directions hold `share` of the variance, on a sample of the rows."""
    x = data[_sample(len(data), SPECTRUM_SAMPLE)].astype(np.float64)
    if metric == "cosine":
        x /= np.linalg.norm(x, axis=1, keepdims=True) + 1e-12
    x -= x.mean(axis=0)
    values = np.clip(np.linalg.eigvalsh(x.T @ x)[::-1], 0, None)
    total = values.sum()
    return len(values) if total == 0 else int(np.searchsorted(np.cumsum(values) / total, share) + 1)


def plan(data, metric="l2", **settings):
    """The settings `build` would use for `data`, with the reason for each. Pass any setting to fix it."""
    data = _rows(data)
    n, dim = data.shape
    base = Plan(metric=metric)
    reasons = {
        "quantization": "8-bit codes: a quarter of the memory traffic, re-ranked with the full vectors",
        "fast_build": "built on the codes; faster, and as good a graph in our measurements",
        "refine": "a second pass gave higher recall at equal speed on every dataset measured",
        "ef": "calibrated at build time for the recall target",
        "prefetch": "timed on this machine at build time",
        "threads": "every core",
    }
    nav = 0
    if dim < NAV_MIN_DIM:
        reasons["nav_dim"] = f"off: below {NAV_MIN_DIM} dimensions an 8-bit record is already short"
    else:
        needed = _directions_for(data, metric, NAV_VARIANCE)
        rounded = min(dim, math.ceil(needed / NAV_ROUND) * NAV_ROUND)
        if rounded * NAV_MIN_SAVING > dim:
            reasons["nav_dim"] = (f"off: {needed} directions hold {NAV_VARIANCE:.0%} of the variance, "
                                  f"too many for a {NAV_MIN_SAVING}x smaller record")
        else:
            nav = rounded
            reasons["nav_dim"] = f"{needed} principal directions hold {NAV_VARIANCE:.0%} of the variance"
    if nav:
        chosen = base.replace(nav_dim=nav, fill_after=True, rerank=300, rerank_z=3.0)
        reasons.update(fill_after="free link slots filled after the build (measured with walks on directions)",
                       rerank="re-ranks up to 300 candidates, as the walk on directions is approximate",
                       rerank_z="stops the re-rank once the rest are 3 standard deviations past the k-th best")
    else:
        chosen = base
        reasons.update(rerank="twice k (the default): enough when the walk uses every dimension")
    chosen = dataclasses.replace(chosen, reasons=reasons)
    return chosen.replace(**settings) if settings else chosen


_plan_for = plan  # build() takes a `plan` argument, which hides the function


class AutoIndex:
    """An index with the settings of its Plan: search() uses the plan's k, ef and re-rank unless you
    pass your own. Everything else (filter, remove, explain, ...) is the underlying epzero.Index."""

    def __init__(self, index, plan):
        self.index = index
        self.plan = plan

    def __getattr__(self, name):
        return getattr(self.index, name)

    def __len__(self):
        return len(self.index)

    def __contains__(self, label):
        return label in self.index

    def __repr__(self):
        return f"<epzero.AutoIndex of {len(self)} vectors, nav_dim={self.plan.nav_dim}, ef={self.plan.ef}>"

    def search(self, queries, k=None, ef=None, **options):
        p = self.plan
        k = p.k if k is None else k
        # The calibrated beam was for the plan's k; a larger k needs at least that many candidates.
        ef = max(p.ef or 64, k) if ef is None else ef
        options.setdefault("rerank", p.rerank)
        options.setdefault("rerank_z", p.rerank_z)
        return self.index.search(queries, k=k, ef=ef, **options)

    def save(self, path):
        """Saves the index at `path` and its plan beside it (`path`.plan.json)."""
        self.index.save(path)
        state = dataclasses.asdict(self.plan)
        state["system"] = system()
        with open(path + ".plan.json", "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2)

    @classmethod
    def load(cls, path, queries=None):
        """Loads an index saved by save(). On a different CPU the saved prefetch setting may be wrong,
        so it is re-timed when `queries` (a few hundred rows) are given, and reset otherwise."""
        index = Index.load(path)
        with open(path + ".plan.json", encoding="utf-8") as f:
            state = json.load(f)
        saved_on = state.pop("system", {})
        if state.get("prefetch") is not None:
            state["prefetch"] = tuple(state["prefetch"])
        p = Plan(**state)
        if saved_on.get("cpu") == system()["cpu"] and p.prefetch is not None:
            index.set_prefetch(*p.prefetch)
        elif queries is not None:
            p = dataclasses.replace(p, prefetch=tuple(index.tune_prefetch(_rows(queries), k=p.k, ef=p.ef or 64)))
        return cls(index, p)


def _calibrate(index, queries, p, threads):
    """The smallest beam width reaching the recall target on `queries`, which must not be in the
    index: a row already in the graph is linked to its own nearest neighbours, which makes its search
    far easier than a real query's (GloVe: 0.95 that way, 0.87 on rows held out)."""
    k = p.k
    truth, _ = index.exact_search(queries, k=k, threads=threads)

    def recall(ef):
        found, _ = index.search(queries, k=k, ef=max(ef, k), rerank=p.rerank, rerank_z=p.rerank_z, threads=threads)
        return float(np.mean([len(set(f) & set(t)) / k for f, t in zip(found, truth)]))

    low, ef = 0, max(16, k)
    reached = recall(ef)
    while reached < p.recall and ef < MAX_EF:
        low, ef = ef, ef * 2
        reached = recall(ef)
    if reached < p.recall:
        return ef, reached  # the target is out of reach at these graph settings: the widest beam
    while ef - low > max(2, ef // 16):  # narrow to within about 6%
        middle = (low + ef) // 2
        r = recall(middle)
        if r >= p.recall:
            ef, reached = middle, r
        else:
            low = middle
    return ef, reached


def build(data, metric="l2", labels=None, plan=None, capacity=None, queries=None, **settings):
    """An index of `data` (rows of vectors) configured for it and for this machine. Any Plan setting
    passed here overrides the automatic choice; `plan` starts from a Plan you made with plan().
    `capacity` is how many vectors the index can hold (default: the rows of `data`).

    `queries`: a few hundred typical queries, if you have them. The beam width is then calibrated on
    them, which is the most faithful; otherwise on rows of the data held out of the graph until
    calibration is done, which is accurate when your queries look like your data and optimistic
    when they don't (text queries against image vectors, say)."""
    data = _rows(data)
    n, dim = data.shape
    p = plan or _plan_for(data, metric)
    if settings:
        p = p.replace(**settings)
    threads = p.threads or os.cpu_count() or 1
    labels = np.arange(n, dtype=np.uint64) if labels is None else np.asarray(labels, dtype=np.uint64)
    if len(labels) != n:
        raise ValueError(f"{len(labels)} labels for {n} rows")

    # Rows to calibrate on go in last, so the graph has never seen them when they are searched.
    held = np.zeros(n, dtype=bool)
    if p.ef is None and queries is None and n >= 4 * p.k:
        held[_sample(n, min(CALIBRATION_QUERIES, n // 4), seed=1)] = True

    index = Index(dim, p.metric, max(capacity or n, n), p.M, p.ef_construction, quantization=p.quantization,
                  nav_dim=p.nav_dim, fast_build=p.fast_build and p.quantization != "none")
    if p.quantization != "none":
        index.train(data[_sample(n, TRAIN_SAMPLE)])
    index.add(data[~held], labels[~held], threads=threads)
    if p.refine:
        index.refine_links(threads)
    if p.fill_after:
        index.fill_free_links(threads)

    measured = {}
    ef = p.ef
    if ef is None:
        sample = _rows(queries) if queries is not None else data[held]
        if len(sample):
            ef, reached = _calibrate(index, sample, p, threads)
            measured["recall at calibration"] = round(reached, 4)
            measured["calibrated on"] = "your queries" if queries is not None else f"{len(sample)} held-out rows"
        else:
            ef = max(64, p.k)
    if held.any():
        index.add(data[held], labels[held], threads=threads)
    prefetch = p.prefetch
    if prefetch is None:
        prefetch = tuple(index.tune_prefetch(data[_sample(n, TUNE_QUERIES, seed=2)], k=p.k, ef=ef))
    else:
        index.set_prefetch(*prefetch)
    measured.update(system())
    return AutoIndex(index, dataclasses.replace(p, ef=ef, prefetch=prefetch, measured=measured))
