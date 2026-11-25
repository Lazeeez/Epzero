"""4-bit codes on this CPU: every integer kernel the engine can pick must return the same search.

    EPZERO_SQ8_KERNEL=avx2 python bench/sq4_check.py OUT.json      (then =avx512, then compare)
    python bench/sq4_check.py --compare A.json B.json

The kernels are exact integer sums, so two runs that differ only in the kernel (EPZERO_SQ8_KERNEL, read
when an index is made) must give identical labels and distances. Lengths cover whole 128-code iterations,
a 64-code block and a 32-code tail, with and without principal directions.
"""
import json
import sys

import numpy as np


def run(out):
    import epzero

    rng = np.random.default_rng(7)
    result = {"simd": None, "cases": {}}
    for dim, nav in ((512, 0), (224, 0), (200, 0), (1024, 192), (768, 160)):
        centers = rng.standard_normal((20, dim)).astype(np.float32)
        data = (centers[rng.integers(0, 20, 6000)] + 0.3 * rng.standard_normal((6000, dim))).astype(np.float32)
        queries = (centers[rng.integers(0, 20, 50)] + 0.3 * rng.standard_normal((50, dim))).astype(np.float32)
        # One build thread: a seeded single-threaded build is deterministic, so only the kernel differs.
        index = epzero.build(data, "l2", threads=1, nav_dim=nav, code_bits=4, ef=64, prefetch=(4, 2))
        labels, distances = index.search(queries, k=10, ef=64)
        result["simd"] = index.simd
        result["cases"][f"{dim}/{nav}"] = {"labels": labels.tolist(), "distances": np.round(distances, 5).tolist()}
    with open(out, "w") as f:
        json.dump(result, f)
    print(out, result["simd"], "cases", len(result["cases"]))


def compare(a, b):
    x, y = (json.load(open(p)) for p in (a, b))
    same = x["cases"] == y["cases"]
    print("identical" if same else "DIFFERENT", f"({x['simd']} vs {y['simd']}, {len(x['cases'])} cases)")
    sys.exit(0 if same else 1)


if __name__ == "__main__":
    if sys.argv[1] == "--compare":
        compare(sys.argv[2], sys.argv[3])
    else:
        run(sys.argv[1])
