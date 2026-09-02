#!/usr/bin/env python3
"""Aggregate the recall@10 / p50 / p99 sweep CSVs into a markdown table + SVG curve.

Usage: aggregate_plot.py <csv1> <csv2> ... [-o <out_prefix>]
  Each CSV line: label,engine,param,param_value,recall_at_10,p50_ms,p99_ms
  Output: <out_prefix>.md and <out_prefix>.svg
"""
import csv
import sys
from pathlib import Path

OUT = "bench_results"
files = []
args = sys.argv[1:]
while args:
    a = args.pop(0)
    if a == "-o":
        OUT = args.pop(0)
    else:
        files.append(a)

rows = []
for f in files:
    with open(f) as fh:
        for r in csv.DictReader(fh, fieldnames=[
                "label", "engine", "param", "param_value",
                "recall_at_10", "p50_ms", "p99_ms"]):
            r["recall_at_10"] = float(r["recall_at_10"])
            r["p50_ms"] = float(r["p50_ms"])
            r["p99_ms"] = float(r["p99_ms"])
            rows.append(r)

rows.sort(key=lambda r: (r["label"], r["engine"], float(r["param_value"])))

# ---- markdown table ----
md = ["# recall@10 vs latency — ivfrq & hnsw: vanilla PG17 vs Neon on k8s (x86)",
      "",
      "| config | engine | param | value | recall@10 | p50 (ms) | p99 (ms) |",
      "|---|---|---|---|---|---|---|"]
for r in rows:
    md.append(f"| {r['label']} | {r['engine']} | {r['param']} | {r['param_value']} "
              f"| {r['recall_at_10']:.2f} | {r['p50_ms']:.3f} | {r['p99_ms']:.3f} |")
md.append("")
Path(OUT + ".md").write_text("\n".join(md))

# ---- SVG plot: x = latency ms (log), y = recall@10 % (linear, 80-100),
#      line per config, p50 solid / p99 dashed, recall >= 0.8 only ----
import math

plot_rows = [r for r in rows if r["recall_at_10"] >= 80.0]

W, H = 900, 560
ML, MR, MT, MB = 70, 240, 30, 60
X0, Y0 = ML, H - MB
XW, YH = W - ML - MR, H - MT - MB

labels = sorted({r["label"] for r in plot_rows})

# Distinct colors per label.  Explicit map for the benchmark's known labels;
# unknown labels fall back to a deterministic cycle so nothing ever shares
# (or silently defaults to) a color.
PALETTE = {
    "ivfrq-vanilla": "#1f77b4",  # blue
    "ivfrq-k8s":     "#d62728",  # red
    "hnsw-vanilla":  "#2ca02c",  # green
    "hnsw-k8s":      "#ff7f0e",  # orange
}
CYCLE = ["#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]

def color(lab):
    if lab in PALETTE:
        return PALETTE[lab]
    i = labels.index(lab)
    return CYCLE[i % len(CYCLE)]

xmin, xmax = 0.5, 500.0     # latency ms (log)
ymin, ymax = 80.0, 100.0    # recall % (linear)

def X(v):  # log latency
    lx0, lx1 = math.log10(xmin), math.log10(xmax)
    return X0 + (math.log10(max(v, xmin)) - lx0) / (lx1 - lx0) * XW

def Y(v):  # linear recall
    return Y0 - (v - ymin) / (ymax - ymin) * YH

parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
         f'viewBox="0 0 {W} {H}">',
         f'<rect width="{W}" height="{H}" fill="white"/>']
# grid: latency (log) verticals, recall horizontals
for g in [0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500]:
    if g < xmin or g > xmax:
        continue
    x = X(g)
    parts.append(f'<line x1="{x:.1f}" y1="{Y0}" x2="{x:.1f}" y2="{MT}" '
                 f'stroke="#e0e0e0" stroke-width="1"/>')
    parts.append(f'<text x="{x:.1f}" y="{Y0+16}" font-size="11" text-anchor="middle" '
                 f'fill="#666">{g:g}</text>')
for g in [80, 85, 90, 95, 100]:
    y = Y(g)
    parts.append(f'<line x1="{X0}" y1="{y:.1f}" x2="{X0+XW}" y2="{y:.1f}" '
                 f'stroke="#e0e0e0" stroke-width="1"/>')
    parts.append(f'<text x="{X0-6}" y="{y+4:.1f}" font-size="11" text-anchor="end" '
                 f'fill="#666">{g}</text>')
parts.append(f'<text x="{X0+XW/2:.0f}" y="{H-12}" font-size="13" text-anchor="middle">'
             f'latency (ms, log)</text>')
parts.append(f'<text x="16" y="{MT+YH/2:.0f}" font-size="13" text-anchor="middle" '
             f'transform="rotate(-90 16 {MT+YH/2:.0f})">recall@10 (%)</text>')

# legend in the right margin (solid = p50, dashed = p99)
parts.append(f'<text x="{X0+XW+14}" y="{MT+4}" font-size="12" fill="#333">solid p50 / dashed p99</text>')
for i, lab in enumerate(labels):
    col = color(lab)
    ly = MT + 24 + i * 34
    parts.append(f'<line x1="{X0+XW+14}" y1="{ly}" x2="{X0+XW+42}" y2="{ly}" '
                 f'stroke="{col}" stroke-width="2"/>')
    parts.append(f'<line x1="{X0+XW+14}" y1="{ly+13}" x2="{X0+XW+42}" y2="{ly+13}" '
                 f'stroke="{col}" stroke-width="2" stroke-dasharray="8,4"/>')
    parts.append(f'<text x="{X0+XW+50}" y="{ly+16}" font-size="12" fill="{col}">{lab}</text>')

for lab in labels:
    for pct, dash in (("p50", ""), ("p99", "8,4")):
        pts = [(r[pct + "_ms"], r["recall_at_10"]) for r in plot_rows if r["label"] == lab]
        if not pts:
            continue
        col = color(lab)
        path = "M" + " L".join(f"{X(x):.1f} {Y(y):.1f}" for x, y in pts)
        parts.append(f'<path d="{path}" fill="none" stroke="{col}" stroke-width="2" '
                     f'stroke-dasharray="{dash}"/>')
        for x, y in pts:
            parts.append(f'<circle cx="{X(x):.1f}" cy="{Y(y):.1f}" r="3" fill="{col}"/>')

parts.append("</svg>")
Path(OUT + ".svg").write_text("\n".join(parts))
print(f"wrote {OUT}.md and {OUT}.svg ({len(rows)} rows)")
