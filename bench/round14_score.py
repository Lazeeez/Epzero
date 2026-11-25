"""Score round 14's recorded predictions (scripts/pace/round14.sh, P1-P8) from its result files.

    python bench/round14_score.py [RESULTS_DIR]          (default: bench/results/pace/gold6226)

Queries per second at a recall target are read as in docs/performance.md: the fastest point at or above the
target, or the log-linear interpolation across the first crossing, whichever is higher; a library that never
reaches the target has no value there. Every rule below was written before any round 14 result existed.

- P1: glass-sq4 against glass at recall 0.95, k = 10 (faster where glass never reaches 0.95 and it does);
  glass-sq4's highest recall on LAION at both k; glass-sq4 against epzero-auto-8bit at 0.95, k = 100, on AG News
  and ImageNet-CLIP.
- P2-P4: epzero-auto-half against its 8-bit baseline (epzero-auto-8bit, or epzero-auto on GloVe and DEEP, where
  4 bits are never tried): memory; recall at every beam width both swept; speed at 0.95, 0.98 and 0.99 where
  both reach them. A dataset passes P3 and P4 only if every check on it passes, at both k.
- P5-P7: epzero-auto-4bit against epzero-auto-8bit, and epzero-auto against the faster of the two.
- The 8-bit path: epzero-auto-8bit against the latest earlier round's epzero-auto at 0.95, same dataset and k.
- P8: the full-scale speed of each count at 0.95 comes from rounds 12b and 13. In round 12b the planner's own
  count was built twice, by epzero.build and by the control path every other count used; the control is used,
  so that a dataset's counts are all built the same way. Round 13 has one build per count.
"""
import glob
import json
import math
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
TARGETS = (0.95, 0.98, 0.99)
OPENAI = ("dbpedia-openai-1000k-angular", "simplewiki-openai-3072-normalized")
NAV_WALK = ("agnews-mxbai-1024-euclidean", "arxiv-nomic-768-normalized", "ccnews-nomic-768-normalized",
            "dbpedia-openai-1000k-angular", "gist-960-euclidean", "simplewiki-openai-3072-normalized")
FULL_WALK = ("gooaq-distilroberta-768-normalized", "yahoo-minilm-384-normalized", "imagenet-clip-512-normalized",
             "laion-clip-512-normalized", "imagenet-align-640-normalized")
EARLIER = ("round13", "round12b", "round12", "round11", "round10", "round8", "round7")


def at(sweep, target):
    points = sorted((p["recall"], p["qps"]) for p in sweep)
    found = [q for r, q in points if r >= target]
    for (r0, q0), (r1, q1) in zip(points, points[1:]):
        if r0 < target <= r1 and q0 > 0 and q1 > 0:
            found.append(math.exp(math.log(q0) + (target - r0) / (r1 - r0) * (math.log(q1) - math.log(q0))))
    return max(found) if found else None


def load(folder, pattern):
    out = {}
    for path in sorted(glob.glob(os.path.join(folder, pattern))):
        d = json.load(open(path))
        out[(d["dataset"], d["k"])] = d
    return out


def faster(a, b):
    """a faster than b at a target: a reaches it and b doesn't, or both do and a is faster."""
    return a is not None and (b is None or a > b)


def score(folder):
    runs = load(folder, "*-speed-round14-k*.json")
    lines, verdict = [], {}
    lib = lambda d, name: d["libraries"].get(name)
    q = lambda d, name, t: at(lib(d, name)["sweep"], t) if lib(d, name) else None
    datasets = sorted({ds for ds, _ in runs})

    # P1
    k10 = [d for (ds, k), d in runs.items() if k == 10 and lib(d, "glass") and lib(d, "glass-sq4")]
    wins = sum(faster(q(d, "glass-sq4", 0.95), q(d, "glass", 0.95)) for d in k10)
    laion = [max(p["recall"] for p in lib(d, "glass-sq4")["sweep"]) for (ds, k), d in runs.items()
             if ds.startswith("laion") and lib(d, "glass-sq4")]
    beats = [faster(q(runs[(ds, 100)], "glass-sq4", 0.95), q(runs[(ds, 100)], "epzero-auto-8bit", 0.95))
             for ds in ("agnews-mxbai-1024-euclidean", "imagenet-clip-512-normalized") if (ds, 100) in runs]
    verdict["P1"] = wins >= 9 and len(laion) == 2 and all(r < 0.90 for r in laion) and len(beats) == 2 and all(beats)
    lines.append(f"P1: glass-sq4 faster than glass at 0.95 (k=10) on {wins}/{len(k10)}; LAION top recall "
                 f"{[round(r, 3) for r in laion]}; beats epzero-auto-8bit at 0.95 (k=100) on AG News, ImageNet-CLIP: {beats}")

    # P2-P4
    half_ok, memory_ok = {}, {}
    for ds in datasets:
        checks = []
        for k in (10, 100):
            d = runs.get((ds, k))
            if not d or not lib(d, "epzero-auto-half"):
                continue
            base = "epzero-auto-8bit" if lib(d, "epzero-auto-8bit") else "epzero-auto"
            h, b = lib(d, "epzero-auto-half"), lib(d, base)
            saving = 1 - h["memory_mb"] / b["memory_mb"] if b["memory_mb"] else None
            checks.append(("P2", saving is not None and 0.28 <= saving <= 0.50, f"k={k} memory -{saving:.0%}" if saving is not None else "no memory"))
            recall_b = {p[b["param"]]: p["recall"] for p in b["sweep"]}
            gaps = [p["recall"] - recall_b[p[h["param"]]] for p in h["sweep"] if p[h["param"]] in recall_b]
            worst = min(gaps) if gaps else 0.0
            ok3 = all(g >= -0.005 for g in gaps) if ds in OPENAI else all(abs(g) <= 0.002 for g in gaps)
            checks.append(("P3", ok3, f"k={k} worst recall gap {worst:+.4f}"))
            for t in TARGETS:
                a, c = at(h["sweep"], t), at(b["sweep"], t)
                if a is None or c is None:
                    continue
                ratio = a / c
                if k == 100 and d["dim"] >= 768:
                    checks.append(("P4", ratio >= 0.97, f"k={k} {t}: {ratio:.2f}x"))
                elif k == 10:
                    checks.append(("P4", 0.90 <= ratio <= 1.03, f"k={k} {t}: {ratio:.2f}x"))
        if not checks:
            continue
        p2 = memory_ok[ds] = all(ok for name, ok, _ in checks if name == "P2")
        half_ok[ds] = all(ok for name, ok, _ in checks if name in ("P3", "P4"))
        lines.append(f"  half {ds}: P2 {'held' if p2 else 'FAILED'}, P3+P4 {'held' if half_ok[ds] else 'FAILED'}; "
                     + "; ".join(text for name, ok, text in checks if not ok or name == "P2"))
    verdict["P2"] = bool(memory_ok) and all(memory_ok.values())
    verdict["half_vectors decision (P3+P4 on >= 12 of 14)"] = sum(half_ok.values()) >= 12
    lines.append(f"P3+P4 held on {sum(half_ok.values())}/{len(half_ok)} datasets")

    # P5-P7
    p5 = []
    for ds in NAV_WALK:
        for k in (10, 100):
            d = runs.get((ds, k))
            if d and lib(d, "epzero-auto-4bit"):
                p5.append((ds, k, faster(q(d, "epzero-auto-8bit", 0.95), q(d, "epzero-auto-4bit", 0.95))))
    verdict["P5"] = len(p5) == 2 * len(NAV_WALK) and all(ok for *_, ok in p5)
    lines.append("P5 (4-bit slower on direction walks, 0.95): " + ", ".join(f"{ds.split('-')[0]} k{k} {'yes' if ok else 'NO'}" for ds, k, ok in p5))
    p6 = [(ds, faster(q(runs[(ds, 100)], "epzero-auto-4bit", 0.98), q(runs[(ds, 100)], "epzero-auto-8bit", 0.98)))
          for ds in FULL_WALK if (ds, 100) in runs and lib(runs[(ds, 100)], "epzero-auto-4bit")]
    verdict["P6"] = sum(ok for _, ok in p6) >= 3
    lines.append("P6 (4-bit faster on full walks, 0.98, k=100): " + ", ".join(f"{ds.split('-')[0]} {'yes' if ok else 'no'}" for ds, ok in p6))
    p7 = []
    for (ds, k), d in runs.items():
        if k == 10 and lib(d, "epzero-auto-4bit"):
            best = max(filter(None, (q(d, "epzero-auto-8bit", 0.95), q(d, "epzero-auto-4bit", 0.95))), default=None)
            mine = q(d, "epzero-auto", 0.95)
            p7.append((ds, best is not None and mine is not None and mine >= 0.97 * best,
                       f"{mine / best:.2f}x" if best and mine else "-"))
    verdict["P7"] = sum(ok for _, ok, _ in p7) >= 10
    lines.append(f"P7 (trial within 3% of the better, k=10): {sum(ok for _, ok, _ in p7)}/{len(p7)}: "
                 + ", ".join(f"{ds.split('-')[0]} {r}" for ds, _, r in sorted(p7)))

    # The 8-bit path against the latest earlier round
    regress = []
    for (ds, k), d in runs.items():
        if not lib(d, "epzero-auto-8bit"):
            continue
        for tag in EARLIER:
            path = os.path.join(folder, f"{ds}-speed-{tag}-k{k}.json")
            if k == 10 and not os.path.exists(path):
                path = os.path.join(folder, f"{ds}-speed-{tag}.json")
            if os.path.exists(path):
                old = json.load(open(path))["libraries"].get("epzero-auto")
                new, was = q(d, "epzero-auto-8bit", 0.95), at(old["sweep"], 0.95) if old else None
                if new and was:
                    regress.append((ds, k, tag, new / was))
                break
    no_loss = all(0.95 <= r for *_, r in regress)
    verdict["4-bit merge (P7 and no 8-bit loss)"] = verdict["P7"] and no_loss
    lines.append("8-bit path against earlier rounds at 0.95: " + ", ".join(f"{ds.split('-')[0]} k{k} {r:.2f}x ({tag})" for ds, k, tag, r in sorted(regress)))

    # P8
    trials = load(folder, "*-nav-trial-k*.json")
    p8 = []
    for (ds, k), t in sorted(trials.items()):
        earlier = next((p for tag in ("round12b", "round13") for p in [os.path.join(folder, f"{ds}-speed-{tag}-k{k}.json")] if os.path.exists(p)), None)
        if earlier is None or t["choice"] is None:
            p8.append((ds, k, False, "no full-scale file or no choice"))
            continue
        speed = {}
        for name, v in json.load(open(earlier))["libraries"].items():
            m = re.search(r"nav(\d+)-", name) or (re.search(r"nav_dim (\d+)", v.get("note", "")) if name.startswith("epzero-auto") else None)
            if not m:
                continue
            count = int(m.group(1))
            if name.startswith("epzero-sq8-nav") or count not in speed:  # the control path wins for a repeated count
                speed[count] = at(v["sweep"], 0.95)
        best = max(filter(None, speed.values()), default=None)
        mine = speed.get(t["choice"])
        ok = best is not None and mine is not None and mine >= 0.95 * best
        p8.append((ds, k, ok, f"chose {t['choice']}, full-scale best {max(speed, key=lambda c: speed[c] or 0)}"
                              f" ({mine / best:.2f}x of it)" if best and mine else f"chose {t['choice']}"))
    verdict["P8"] = sum(ok for _, _, ok, _ in p8) >= 12
    lines.append(f"P8: {sum(ok for _, _, ok, _ in p8)}/{len(p8)}: " + "; ".join(f"{ds.split('-')[0]} k{k} {text}" for ds, k, _, text in p8))

    return lines, verdict, len(runs), len(trials)


def main():
    folder = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "results", "pace", "gold6226")
    lines, verdict, runs, trials = score(folder)
    print(f"round 14: {runs} speed results, {trials} trials")
    print("\n".join(lines))
    print("\nverdicts:")
    for name, ok in verdict.items():
        print(f"  {name}: {'held' if ok else 'FAILED'}")


if __name__ == "__main__":
    main()
