"""Renders benchmark results as SVG charts in light and dark variants for the README.

SVG is written directly (no plotting library) so marks follow the chart spec exactly:
2px lines, 8px markers with a surface-colored ring, hairline grid, 4px rounded bar ends, and
the system UI font. Colors come from a CVD-validated categorical palette (first three slots
validate for every pair in both modes).

    python bench/plot_results.py --compare bench/results/sift1m-compare.json \
        --optimizations bench/results/optimizations.json --out docs/img
"""

import argparse
import json
import math
import os
from html import escape

FONT = "system-ui, -apple-system, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif"

THEMES = {
    "light": {
        "surface": "#fcfcfb", "border": "rgba(11,11,11,0.10)", "primary": "#0b0b0b", "secondary": "#52514e",
        "muted": "#898781", "grid": "#e1e0d9", "axis": "#c3c2b7",
        "series": ["#2a78d6", "#eb6834", "#1baf7a"],
    },
    "dark": {
        "surface": "#1a1a19", "border": "rgba(255,255,255,0.10)", "primary": "#ffffff", "secondary": "#c3c2b7",
        "muted": "#898781", "grid": "#2c2c2a", "axis": "#383835",
        "series": ["#3987e5", "#d95926", "#199e70"],
    },
}

LIB_NAMES = {"oryx": "Oryx", "hnswlib": "hnswlib", "faiss": "FAISS HNSW"}


def text(x, y, s, size, color, weight=400, anchor="start", extra=""):
    return (f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" font-weight="{weight}" fill="{color}" '
            f'text-anchor="{anchor}" {extra}>{escape(s)}</text>')


def svg_doc(width, height, theme, title, desc, body):
    t = THEMES[theme]
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" '
            f'role="img" aria-labelledby="t d" font-family="{FONT}">\n'
            f'<title id="t">{escape(title)}</title>\n<desc id="d">{escape(desc)}</desc>\n'
            f'<rect x="0.5" y="0.5" width="{width - 1}" height="{height - 1}" rx="12" fill="{t["surface"]}" '
            f'stroke="{t["border"]}"/>\n' + "\n".join(body) + "\n</svg>\n")


def interpolate_qps(sweep, recall):
    """QPS at a target recall, interpolating log(QPS) linearly between measured points."""
    points = sorted((p["recall"], p["qps"]) for p in sweep)
    for (r0, q0), (r1, q1) in zip(points, points[1:]):
        if r0 <= recall <= r1 and r1 > r0:
            f = (recall - r0) / (r1 - r0)
            return math.exp(math.log(q0) + f * (math.log(q1) - math.log(q0)))
    return None


def nice_125(value, up):
    """Nearest 1-2-5 step at or above (up=True) / below (up=False) value."""
    decade = 10 ** math.floor(math.log10(value))
    steps = [decade * m for m in (1, 2, 5, 10)]
    return min(s for s in steps if s >= value) if up else max(s for s in steps if s <= value)


def compare_chart(data, theme):
    t = THEMES[theme]
    width, height = 760, 480
    left, right, top, bottom = 70, 28, 136, 58
    pw, ph = width - left - right, height - top - bottom
    # "Nines" axis: position by -log10(1 - recall), so 0.9 / 0.99 / 0.999 are evenly spaced and the
    # high-recall region where production systems operate is not crushed against the edge.
    x_ticks = [0.7, 0.8, 0.9, 0.95, 0.98, 0.99, 0.995, 0.999]
    x_min, x_max = 0.7, 0.9995
    libs = [name for name in ("oryx", "hnswlib", "faiss") if name in data["libraries"]]
    all_qps = [p["qps"] for name in libs for p in data["libraries"][name]["sweep"] if p["recall"] >= x_min]
    y_min, y_max = nice_125(min(all_qps), up=False), nice_125(max(all_qps), up=True)

    def nines(r):
        return -math.log10(1 - min(r, x_max))

    def sx(r):
        return left + (nines(r) - nines(x_min)) / (nines(x_max) - nines(x_min)) * pw

    def sy(q):
        return top + ph - (math.log10(q) - math.log10(y_min)) / (math.log10(y_max) - math.log10(y_min)) * ph

    body = [
        text(left - 46, 38, "Oryx vs. hnswlib vs. FAISS on SIFT1M", 17, t["primary"], 600),
        text(left - 46, 60, f"{data['n']:,} × {data['dim']}-d vectors · {data['queries']:,} queries · 1 search thread",
             12.5, t["secondary"]),
        text(left - 46, 78, f"Same graph parameters (M={data['M']}, efConstruction={data['ef_construction']}) · "
             f"median of {data['rounds']} interleaved runs · up and to the right is better", 12.5, t["secondary"]),
    ]
    # Legend row.
    lx = left - 46
    for i, name in enumerate(libs):
        color = t["series"][i]
        body.append(f'<line x1="{lx}" y1="106" x2="{lx + 18}" y2="106" stroke="{color}" stroke-width="2" stroke-linecap="round"/>')
        body.append(f'<circle cx="{lx + 9}" cy="106" r="4" fill="{color}" stroke="{t["surface"]}" stroke-width="2"/>')
        body.append(text(lx + 26, 110.5, LIB_NAMES[name], 12.5, t["secondary"]))
        lx += 26 + 8 * len(LIB_NAMES[name]) + 28

    # Grid: log-scale y ticks at 1-2-5 steps, x ticks every 0.05.
    decade = 10 ** math.floor(math.log10(y_min))
    while decade <= y_max:
        for m in (1, 2, 5):
            q = decade * m
            if y_min <= q <= y_max:
                y = sy(q)
                body.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left + pw}" y2="{y:.1f}" stroke="{t["grid"]}" stroke-width="1"/>')
                label = f"{q / 1000:g}k" if q >= 1000 else f"{q:g}"
                body.append(text(left - 10, y + 4, label, 11.5, t["muted"], anchor="end",
                                 extra='style="font-variant-numeric: tabular-nums"'))
        decade *= 10
    for r in x_ticks:
        x = sx(r)
        body.append(f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{top + ph}" stroke="{t["grid"]}" stroke-width="1"/>')
        body.append(text(x, top + ph + 20, f"{r:g}", 11.5, t["muted"], anchor="middle",
                         extra='style="font-variant-numeric: tabular-nums"'))
    body.append(f'<line x1="{left}" y1="{top + ph}" x2="{left + pw}" y2="{top + ph}" stroke="{t["axis"]}" stroke-width="1"/>')
    body.append(text(left + pw / 2, height - 14, "Recall@10 (log scale of 1 − recall)", 12, t["secondary"],
                     anchor="middle"))
    body.append(text(18, top + ph / 2, "Queries per second (log scale)", 12, t["secondary"], anchor="middle",
                     extra=f'transform="rotate(-90 18 {top + ph / 2})"'))

    # Series: draw the story series (Oryx) last so it sits on top.
    for i, name in reversed(list(enumerate(libs))):
        color = t["series"][i]
        pts = [(p["recall"], p["qps"]) for p in data["libraries"][name]["sweep"] if p["recall"] >= x_min]
        path = " ".join(f"{'M' if j == 0 else 'L'}{sx(rc):.1f},{sy(q):.1f}" for j, (rc, q) in enumerate(pts))
        body.append(f'<path d="{path}" fill="none" stroke="{color}" stroke-width="2" stroke-linejoin="round" stroke-linecap="round"/>')
        for rc, q in pts:
            body.append(f'<circle cx="{sx(rc):.1f}" cy="{sy(q):.1f}" r="4" fill="{color}" stroke="{t["surface"]}" stroke-width="2">'
                        f'<title>{LIB_NAMES[name]}: recall {rc:.4f}, {q:,.0f} QPS</title></circle>')

    # Callout: speedup at a fixed recall target, derived from the data.
    target = 0.95
    ours = interpolate_qps(data["libraries"]["oryx"]["sweep"], target)
    notes = []
    for name in libs[1:]:
        theirs = interpolate_qps(data["libraries"][name]["sweep"], target)
        if ours and theirs:
            notes.append(f"{ours / theirs:.2f}× the QPS of {LIB_NAMES[name]}")
    if notes:
        x = sx(target)
        body.append(f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{top + ph}" stroke="{t["axis"]}" stroke-width="1"/>')
        body.append(text(x + 8, top + 18, f"At recall {target:.2f}, Oryx serves", 12, t["secondary"]))
        body.append(text(x + 8, top + 35, " · ".join(notes), 12.5, t["primary"], 600))
    desc = "Recall versus queries per second. " + "; ".join(
        f"{LIB_NAMES[n]}: " + ", ".join(f"recall {p['recall']:.3f} at {p['qps']:,.0f} QPS" for p in data["libraries"][n]["sweep"])
        for n in libs)
    return svg_doc(width, height, theme, "Oryx vs hnswlib vs FAISS on SIFT1M", desc, body)


def rounded_bar(x, y, w, h, radius, color):
    """Horizontal bar, square at the baseline (left) and rounded at the data end (right)."""
    r = min(radius, w, h / 2)
    return (f'<path d="M{x:.1f},{y:.1f} H{x + w - r:.1f} Q{x + w:.1f},{y:.1f} {x + w:.1f},{y + r:.1f} '
            f'V{y + h - r:.1f} Q{x + w:.1f},{y + h:.1f} {x + w - r:.1f},{y + h:.1f} H{x:.1f} Z" fill="{color}"/>')


def optimization_chart(data, theme):
    t = THEMES[theme]
    steps = data["steps"]
    width, row, bar = 760, 44, 22
    left, right, top = 210, 150, 96
    height = top + row * len(steps) + 50
    pw = width - left - right
    q_max = max(s["qps"] for s in steps)
    base = steps[0]["qps"]
    body = [
        text(24, 38, "Search throughput after each optimization", 17, t["primary"], 600),
        text(24, 60, data["subtitle"], 12.5, t["secondary"]),
    ]
    for i, s in enumerate(steps):
        y = top + i * row
        w = s["qps"] / q_max * pw
        body.append(text(left - 14, y + bar / 2 + 4.5, s["label"], 13, t["secondary"], anchor="end"))
        body.append(rounded_bar(left, y, w, bar, 4, t["series"][0]))
        gain = (s["qps"] / base - 1) * 100
        delta = "" if i == 0 else f'<tspan dx="10" fill="{t["secondary"]}" font-weight="400">{gain:+.0f}%</tspan>'
        body.append(f'<text x="{left + w + 10:.1f}" y="{y + bar / 2 + 4.5:.1f}" font-size="13" font-weight="500" '
                    f'fill="{t["primary"]}" style="font-variant-numeric: tabular-nums">{s["qps"]:,.0f} QPS{delta}</text>')
    body.append(f'<line x1="{left}" y1="{top - 10}" x2="{left}" y2="{top + row * len(steps) - row + bar + 10}" '
                f'stroke="{t["axis"]}" stroke-width="1"/>')
    body.append(text(24, height - 18, "Changes under ~3% are within run-to-run noise. Recall is identical at every step.",
                     11.5, t["muted"]))
    desc = "; ".join(f"{s['label']}: {s['qps']:,.0f} QPS" for s in steps)
    return svg_doc(width, height, theme, "Search throughput after each optimization", desc, body)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--compare", required=True)
    parser.add_argument("--optimizations", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)
    with open(args.compare) as f:
        compare = json.load(f)
    with open(args.optimizations) as f:
        optimizations = json.load(f)
    for theme in THEMES:
        for name, svg in (("sift1m-recall-qps", compare_chart(compare, theme)),
                          ("optimizations", optimization_chart(optimizations, theme))):
            path = os.path.join(args.out, f"{name}-{theme}.svg")
            with open(path, "w", encoding="utf-8", newline="\n") as f:
                f.write(svg)
            print("wrote", path)


if __name__ == "__main__":
    main()
