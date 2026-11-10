"""Automatic configuration: an index set up for your data and your machine, every choice overridable.

    import epzero

    index = epzero.build(vectors, metric="cosine")         # chooses everything
    labels, distances = index.search(queries, k=10)        # at the effort calibrated for recall 0.95
    print(index.plan)                                      # what it chose, and why

    index = epzero.build(vectors, metric="cosine", nav_dim=0, refine=False)   # override any choice
    plan = epzero.plan(vectors, metric="cosine")           # or look first, edit, then build
    index = epzero.build(vectors, plan=plan.replace(recall=0.99))

Two kinds of choice:
- Data: compression, walking on principal directions, the build passes, the re-rank, and the beam
  width (ef) that reaches the recall target. Decided from the data.
- Machine: the SIMD kernel (picked when the library loads), the prefetch setting (timed on this
  CPU) and the thread count (every core).

The decisions are made in C++ (epzero/auto.h), so every binding shares them, and the result is saved
inside the index file: Index.save / Index.load keep it, and a loaded index re-applies its prefetch
setting only on the same kind of CPU (call tune_prefetch() elsewhere). The rules come from
measurements in docs/performance.md.
"""

import dataclasses

import numpy as np

from ._core import Index, _build_auto, _plan_auto

# Settings the C++ planner decides, which a caller can fix instead.
_DECIDED = ("quantization", "nav_dim", "fast_build", "refine", "fill_after", "rerank", "rerank_z", "ef", "prefetch")


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
    seed: int = 42
    reasons: dict = dataclasses.field(default_factory=dict, compare=False)
    measured: dict = dataclasses.field(default_factory=dict, compare=False)

    def replace(self, **changes):
        """A copy with `changes`; a changed setting is recorded as yours, not the plan's."""
        unknown = set(changes) - {f.name for f in dataclasses.fields(self)}
        if unknown:
            raise TypeError(f"unknown settings: {', '.join(sorted(unknown))}")
        reasons = {**self.reasons, **{name: "set by you" for name in changes if name not in ("reasons", "measured")}}
        return dataclasses.replace(self, **changes, reasons=reasons)

    def fixed(self):
        """The settings you chose, which building must keep."""
        return {name: getattr(self, name) for name in _DECIDED if self.reasons.get(name) == "set by you"}

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


def _from_core(params, profile, recall, threads, seed, built):
    """A Plan from the C++ planner's (or a built index's) params and profile."""
    measured = {}
    if built and profile["recall_target"] > 0:
        measured["recall at calibration"] = round(float(profile["recall_measured"]), 4)
    if built and profile["machine"]:
        measured["machine"] = profile["machine"]
    return Plan(metric=params["metric"], quantization=params["quantization"], M=params["M"],
                ef_construction=params["ef_construction"], nav_dim=params["nav_dim"],
                fast_build=profile["fast_build"] if built else params["fast_build"],
                refine=profile["refine"], fill_after=profile["fill_after"], rerank=profile["rerank"],
                rerank_z=round(float(profile["rerank_z"]), 4), recall=round(float(profile["recall_target"] or recall), 4),
                k=profile["k"], ef=profile["ef"] or None, prefetch=tuple(profile["prefetch"]) if built else None,
                threads=threads, seed=seed, reasons=dict(profile["notes"]), measured=measured)


def _rows(data):
    data = np.ascontiguousarray(data, dtype=np.float32)
    if data.ndim != 2 or len(data) == 0 or data.shape[1] == 0:
        raise ValueError("data must be a non-empty 2-D array (rows of vectors)")
    return data


def plan(data, metric="l2", queries=None, **settings):
    """The settings build() would use for `data`, with the reason for each, without building. Pass any
    Plan setting to fix it; ef and prefetch stay None until build() measures them."""
    p = Plan(metric=metric).replace(**settings) if settings else Plan(metric=metric)
    core = _plan_auto(_rows(data), p.metric, p.M, p.ef_construction, p.k, p.recall, p.threads, 0, p.seed,
                      queries is not None, p.fixed())
    out = _from_core(core["params"], core["profile"], p.recall, p.threads, p.seed, built=False)
    # The planner records your settings as "set by you"; keep the measured ones you fixed too.
    return dataclasses.replace(out, **{name: getattr(p, name) for name in ("ef", "prefetch") if name in p.fixed()})


def build(data, metric="l2", labels=None, plan=None, capacity=None, queries=None, **settings):
    """An index of `data` (rows of vectors) configured for it and for this machine. Any Plan setting
    passed here overrides the automatic choice; `plan` starts from a Plan you made with plan().
    `capacity` is how many vectors the index can hold (default: the rows of `data`). `labels`
    default to 0..n-1.

    `queries`: a few hundred typical queries, if you have them. The beam width is then calibrated on
    them, which is the most faithful; otherwise on rows of the data held out of the graph until
    calibration is done, which is accurate when your queries look like your data and optimistic
    when they don't (text queries against image vectors, say).

    Returns an epzero.Index whose search() uses the calibrated settings by default; index.plan says
    what was chosen and why. The settings are saved with the index."""
    data = _rows(data)
    p = plan if plan is not None else Plan(metric=metric)
    if settings:
        p = p.replace(**settings)
    if queries is not None:
        queries = np.ascontiguousarray(queries, dtype=np.float32)
        if queries.ndim != 2 or queries.shape[1] != data.shape[1] or len(queries) == 0:
            raise ValueError(f"queries must be a non-empty array of shape (n, {data.shape[1]})")
    return _build_auto(data, labels, p.metric, p.M, p.ef_construction, p.k, p.recall, p.threads, capacity or 0, p.seed,
                       queries, p.fixed())


def _index_plan(index):
    return _from_core(index.params, index.profile, 0.95, 0, index.params["seed"], built=True)


Index.plan = property(_index_plan, doc="How this index is configured, and why: an epzero.Plan (saved with it).")
