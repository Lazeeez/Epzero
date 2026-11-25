"""Score round 15's recorded predictions (scripts/pace/round15.sh, R1-R3) from its result files.

    python bench/round15_score.py [RESULTS_DIR]          (default: bench/results/pace/gold6226)

Speeds at a recall target are read as in round14_score.py. Every rule was written before any round 15 result.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from round14_score import HERE, OPENAI, TARGETS, at, load  # noqa: E402

WALKS = ("agnews-mxbai-1024-euclidean", "arxiv-nomic-768-normalized", "ccnews-nomic-768-normalized",
         "dbpedia-openai-1000k-angular", "gist-960-euclidean", "simplewiki-openai-3072-normalized")


def score(folder):
    runs = load(folder, "*-speed-round15-k*.json")
    lines, verdict = [], {}
    lib = lambda d, name: d["libraries"].get(name)

    # R1: 16-bit vectors against 8-bit, every dataset with both.
    r1 = {}
    for (ds, k), d in sorted(runs.items()):
        h, b = lib(d, "epzero-auto-half"), lib(d, "epzero-auto-8bit")
        if not (h and b):
            continue
        fails = []
        if h["build_seconds"] > 1.10 * b["build_seconds"]:
            fails.append(f"build {h['build_seconds'] / b['build_seconds']:.2f}x")
        saving = 1 - h["memory_mb"] / b["memory_mb"]
        if not 0.28 <= saving <= 0.50:
            fails.append(f"memory -{saving:.0%}")
        recall_b = {p[b["param"]]: p["recall"] for p in b["sweep"]}
        gaps = [p["recall"] - recall_b[p[h["param"]]] for p in h["sweep"] if p[h["param"]] in recall_b]
        if gaps and (min(gaps) < -0.005 if ds in OPENAI else max(abs(g) for g in gaps) > 0.002):
            fails.append(f"recall gap {min(gaps):+.4f}")
        for t in TARGETS:
            a, c = at(h["sweep"], t), at(b["sweep"], t)
            if a and c and ((k == 100 and d["dim"] >= 768 and a / c < 0.97) or (k == 10 and not 0.90 <= a / c <= 1.03)):
                fails.append(f"{t}: {a / c:.2f}x")
        r1.setdefault(ds, []).extend(f"k{k} {f}" for f in fails)
        lines.append(f"  R1 {ds.split('-')[0]} k{k}: build {h['build_seconds'] / b['build_seconds']:.2f}x, memory -{saving:.0%}"
                     + ("" if not fails else "  FAILS: " + "; ".join(fails)))
    verdict["R1 (16-bit on every dataset)"] = bool(r1) and all(not f for f in r1.values())

    # R2: refine width 100 on the six walks.
    r2 = {}
    for ds in WALKS:
        ok, worst, notes = True, 9.0, []
        for k in (100, 10):
            d = runs.get((ds, k))
            r, b = (lib(d, "epzero-auto-refine100"), lib(d, "epzero-auto-8bit")) if d else (None, None)
            if not (r and b):
                ok = False
                notes.append(f"k{k} missing")
                continue
            faster = 1 - r["build_seconds"] / b["build_seconds"]
            ok &= 0.15 <= faster <= 0.35
            notes.append(f"k{k} build -{faster:.0%}")
            for t in TARGETS:
                a, c = at(r["sweep"], t), at(b["sweep"], t)
                if a and c:
                    worst = min(worst, a / c)
                    ok &= 0.97 <= a / c <= 1.05
                    notes.append(f"{t}: {a / c:.2f}x")
        r2[ds] = (ok, worst)
        lines.append(f"  R2 {ds.split('-')[0]}: {'holds' if ok else 'fails'} ({', '.join(notes)})")
    held = sum(ok for ok, _ in r2.values())
    verdict["R2 (refine 100 on >= 5 of 6)"] = held >= 5
    verdict["refine 100 decision (all six, or five with the sixth >= 0.97x)"] = held == 6 or (
        held == 5 and all(w >= 0.97 for ok, w in r2.values() if not ok))

    # R3: check_walk on COCO.
    r3 = []
    for k in (100, 10):
        d = runs.get(("coco-nomic-768-normalized", k))
        c, o = (lib(d, "epzero-auto-ood-checkwalk"), lib(d, "epzero-auto-ood")) if d else (None, None)
        if not (c and o):
            r3.append(False)
            lines.append(f"  R3 k{k}: missing")
            continue
        no_walk = "nav_dim 0," in c.get("note", "")
        reaches = max(p["recall"] for p in c["sweep"]) >= 0.95
        a, b = at(c["sweep"], 0.90), at(o["sweep"], 0.90)
        ratio = a / b if a and b else (float("inf") if a else 0.0)
        build = c["build_seconds"] / o["build_seconds"]
        r3.append(no_walk and reaches and ratio >= 2 and build <= 1.5)
        lines.append(f"  R3 k{k}: no walk {no_walk}, reaches 0.95 {reaches}, speed at 0.90 {ratio:.2f}x, build {build:.2f}x")
    verdict["R3 (check_walk on COCO)"] = len(r3) == 2 and all(r3)
    return lines, verdict, len(runs)


def main():
    folder = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "results", "pace", "gold6226")
    lines, verdict, runs = score(folder)
    print(f"round 15: {runs} speed results")
    print("\n".join(lines))
    print("\nverdicts:")
    for name, ok in verdict.items():
        print(f"  {name}: {'held' if ok else 'FAILED'}")


if __name__ == "__main__":
    main()
