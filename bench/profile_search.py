"""Hardware counters for one library's search loop, at the setting where it reaches a target recall:
where the time per query goes (instructions, cycles, cache and TLB misses), to compare Oryx with
the faster libraries in bench/compete.py on equal terms.

    run      one library, under `perf stat --control` (the counters only run during the timed loop):
             perf stat -x, -e <events> --control fifo:ctl,ack -D -1 -o counts.csv -- \\
               python bench/profile_search.py run --data X.hdf5 --cache DIR --lib glass --param 120 \\
                 --ctl ctl --ack ack --info info.json
    pick     the smallest search parameter at which each library reaches the target recall, from a
             compete.py result: python bench/profile_search.py pick --results X-compete.json --target 0.95
    report   per-query counts from the runs' CSV files: python bench/profile_search.py report --dir D --out P.json

Indexes come from compete.py's cache (Glass, which isn't saved, is rebuilt).
"""

import argparse
import glob
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def run(args):
    import numpy as np
    from compete import LIBRARIES, load_dataset, recall_at_k

    dataset = os.path.splitext(os.path.basename(args.data))[0]
    base, queries, truth, _ = load_dataset(args.data, args.queries)
    lib = LIBRARIES[args.lib](base.shape[1], args)
    path = os.path.join(args.cache, f"{dataset}-{args.lib}-M{args.M}-efc{args.efc}.idx")
    if lib.saves and os.path.exists(path):
        lib.load(path, base)
    else:
        lib.build(base)
    recall = recall_at_k(lib.search(queries, args.k, args.param), truth, args.k)  # also warms the caches
    del base  # only the index stays in memory

    def perf(command):  # tells perf stat to start or stop counting, and waits until it has
        if args.ctl:
            with open(args.ctl, "w") as ctl:
                ctl.write(command + "\n")
            with open(args.ack) as ack:
                ack.readline()

    perf("enable")
    start = time.perf_counter()
    for _ in range(args.rounds):
        lib.search(queries, args.k, args.param)
    seconds = time.perf_counter() - start
    perf("disable")
    searched = args.rounds * len(queries)
    info = {"dataset": dataset, "lib": args.lib, "param": args.param, "recall": recall, "queries": searched,
            "qps": searched / seconds}
    print(json.dumps(info))
    if args.info:
        with open(args.info, "w") as f:
            json.dump(info, f)


def pick(args):
    with open(args.results) as f:
        libs = json.load(f)["libraries"]
    for name in args.libs.split(","):
        lib = libs[name]
        ok = [s for s in lib["sweep"] if s["recall"] >= args.target]
        if ok:
            print(name, min(ok, key=lambda s: s[lib["param"]])[lib["param"]])


def report(args):
    rows = []
    for info_path in sorted(glob.glob(os.path.join(args.dir, "*.info.json"))):
        with open(info_path) as f:
            info = json.load(f)
        counts, coverage = {}, {}
        with open(info_path.replace(".info.json", ".csv")) as f:
            for line in f:  # perf stat -x, : value,unit,event,run time,percent running,...
                fields = line.strip().split(",")
                if len(fields) < 5 or not fields[0] or fields[0].startswith("<"):
                    continue
                try:
                    value = float(fields[0])
                except ValueError:
                    continue
                event = fields[2].removesuffix(":u")
                counts[event] = value / info["queries"]
                coverage[event] = float(fields[4] or 100)
        if "cycles" in counts and "instructions" in counts:
            counts["ipc"] = counts["instructions"] / counts["cycles"]
        rows.append({**info, "per_query": counts, "percent_counted": coverage})
        line = "  ".join(f"{k} {v:,.0f}" if v >= 10 else f"{k} {v:.2f}" for k, v in counts.items())
        print(f"{info['dataset']:28s} {info['lib']:10s} {info['param']:5} recall {info['recall']:.3f} qps {info['qps']:8.0f}  {line}")
    with open(args.out, "w") as f:
        json.dump({"note": "hardware counter totals divided by queries searched; user space only", "runs": rows}, f, indent=2)
    print(f"wrote {args.out}")


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    r = sub.add_parser("run")
    r.add_argument("--data", required=True)
    r.add_argument("--cache", required=True)
    r.add_argument("--lib", required=True)
    r.add_argument("--param", type=int, required=True)
    r.add_argument("--rounds", type=int, default=3)
    r.add_argument("--queries", type=int, default=10_000)
    r.add_argument("--k", type=int, default=10)
    r.add_argument("--M", type=int, default=16)
    r.add_argument("--efc", type=int, default=200)
    r.add_argument("--threads", type=int, default=os.cpu_count())  # for a Glass rebuild
    r.add_argument("--ctl")
    r.add_argument("--ack")
    r.add_argument("--info")
    p = sub.add_parser("pick")
    p.add_argument("--results", required=True)
    p.add_argument("--target", type=float, default=0.95)
    p.add_argument("--libs", default="oryx,oryx-sq8,hnswlib,glass")
    s = sub.add_parser("report")
    s.add_argument("--dir", required=True)
    s.add_argument("--out", required=True)
    args = parser.parse_args()
    {"run": run, "pick": pick, "report": report}[args.command](args)


if __name__ == "__main__":
    main()
