"""Checks that search quality hasn't regressed: recall and distance computations per query on fixed
synthetic data, built on one thread so the graph is deterministic. Unlike queries per second, both are
the same on any machine (within float rounding between SIMD paths), so CI can gate on them.

    python scripts/check_regression.py build/bench_recall               # check against the bounds
    python scripts/check_regression.py build/bench_recall --update      # re-record them (say why in the PR)

A configuration fails when recall at any beam width drops by more than RECALL_SLACK, or distance
computations per query rise by more than WORK_SLACK.
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile

BOUNDS = os.path.join(os.path.dirname(__file__), "..", "engine", "bench", "regression.json")
RECALL_SLACK = 0.01
WORK_SLACK = 0.05

COMMON = ["--synthetic", "--n", "20000", "--queries", "500", "--threads", "1", "--no-baseline", "--repeat", "1"]
# Beam widths where recall is still below 1, so a regression shows. Run on x86 (the integer 8-bit kernels
# give exactly the same sums there on any CPU); other kernels round differently and drift by a few percent.
CONFIGS = {
    "float-l2": ["--dim", "64", "--ef", "4,8,16"],
    "float-cosine": ["--dim", "64", "--metric", "cosine", "--ef", "4,8,16"],
    "sq8-second-pass": ["--dim", "64", "--quant", "sq8", "--fast-build", "--refine", "1", "--ef", "4,8,16"],
    "sq8-directions": ["--dim", "256", "--quant", "sq8", "--nav-dim", "64", "--fast-build", "--fill-after", "--refine", "1",
                       "--rerank", "300", "--rerank-z", "3", "--ef", "32,64,128"],
}


def measure(binary, args):
    with tempfile.TemporaryDirectory() as tmp:
        out = os.path.join(tmp, "result.json")
        subprocess.run([binary, *COMMON, *args, "--json", out], check=True, stdout=subprocess.DEVNULL)
        with open(out, encoding="utf-8") as f:
            sweep = json.load(f)["sweep"]
    return {str(s["ef"]): {"recall": s["recall"], "distances": s["distances_per_query"]} for s in sweep}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("bench_recall")
    parser.add_argument("--update", action="store_true")
    args = parser.parse_args()
    measured = {name: measure(args.bench_recall, config) for name, config in CONFIGS.items()}
    if args.update:
        with open(BOUNDS, "w", encoding="utf-8", newline="\n") as f:
            json.dump(measured, f, indent=2)
            f.write("\n")
        print(f"recorded {BOUNDS}")
        return 0
    with open(BOUNDS, encoding="utf-8") as f:
        expected = json.load(f)
    failures = []
    for name, sweep in measured.items():
        for ef, got in sweep.items():
            want = expected[name][ef]
            line = (f"{name:16s} ef {ef:>4s}: recall {got['recall']:.4f} (recorded {want['recall']:.4f}), "
                    f"distances {got['distances']:8.1f} (recorded {want['distances']:8.1f})")
            bad = got["recall"] < want["recall"] - RECALL_SLACK or got["distances"] > want["distances"] * (1 + WORK_SLACK)
            print(("FAIL " if bad else "ok   ") + line)
            if bad:
                failures.append(line)
    if failures:
        print(f"\n{len(failures)} regression(s); if intended, re-record with --update and explain why in the pull request")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
