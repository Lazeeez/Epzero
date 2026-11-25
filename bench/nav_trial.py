"""Which direction count would a trial on a sample choose? The planner's candidate rule, tested offline.

    python bench/nav_trial.py --data DATA.hdf5 --k 100 --out trial.json [--sample 100000] [--candidates 0,128,...]

For each candidate count (0: no walk) it builds epzero.build on a sample of the data, with that count fixed
and 8-bit codes, calibrates the beam width for recall 0.95 at k on rows held out of the sample, and times
those rows' search at that width on one thread (the fastest of three passes). The count with the cheapest
search at the target is the trial's choice. Rounds 12b and 13 measured every count's speed on the full
datasets, so the choice can be compared with the full-scale best before the planner makes it (round 15).
"""
import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from compete import environment, load_dataset  # noqa: E402

HELD = 1000  # rows held out of the sample: the first half calibrates, all of them are timed


def trial(sample, held, k, nav, threads, recall=0.95):
    import epzero

    start = time.perf_counter()
    index = epzero.build(sample, "l2", M=16, ef_construction=200, threads=threads, queries=held[: HELD // 2],
                         nav_dim=nav, code_bits=8, k=k, recall=recall)
    build = time.perf_counter() - start
    plan = index.plan
    best = float("inf")
    for _ in range(3):
        start = time.perf_counter()
        index.search(held, k=k, threads=1)
        best = min(best, time.perf_counter() - start)
    return {"nav_dim": nav, "ef": plan.ef, "recall_at_calibration": plan.measured.get("recall at calibration"),
            "rerank": plan.rerank, "rerank_z": plan.rerank_z, "build_seconds": round(build, 1),
            "us_per_query": round(best / len(held) * 1e6, 1)}


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", required=True)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--sample", type=int, default=100_000)
    parser.add_argument("--candidates", default="0,128,192,256,384,512")
    parser.add_argument("--threads", type=int, default=len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count(),
                        help="build threads (default: the CPUs this process may run on)")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    base, _, _, metric = load_dataset(args.data, 1)
    n, dim = base.shape
    rows = np.random.default_rng(0).permutation(n)[: args.sample + HELD]
    sample, held = np.ascontiguousarray(base[np.sort(rows[:-HELD])]), np.ascontiguousarray(base[rows[-HELD:]])
    counts = [c for c in (int(x) for x in args.candidates.split(",")) if c == 0 or c <= dim // 2]
    results = []
    for nav in counts:
        r = trial(sample, held, args.k, nav, args.threads)
        print(json.dumps(r), flush=True)
        results.append(r)
    reached = [r for r in results if (r["recall_at_calibration"] or 0) >= 0.95]
    choice = min(reached, key=lambda r: r["us_per_query"])["nav_dim"] if reached else None
    out = {"dataset": os.path.splitext(os.path.basename(args.data))[0], "n": n, "dim": dim, "metric": metric, "k": args.k,
           "sample": len(sample), "held_out": HELD, "trials": results, "choice": choice, "environment": environment()}
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print("choice:", choice)


if __name__ == "__main__":
    main()
