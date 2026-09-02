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
md = ["# recall@10 vs latency — ivfrq & hnsw on vanilla PG17 vs Neon (x86)",
      "",
      "| config | engine | param | value | recall@10 | p50 (ms) | p99 (ms) |",
      "|---|---|---|---|---|---|---|"]
for r in rows:
    md.append(f"| {r['label']} | {r['engine']} | {r['param']} | {r['param_value']} "
              f"| {r['recall_at_10']:.2f} | {r['p50_ms']:.3f} | {r['p99_ms']:.3f} |")
md.append("")
Path(OUT + ".md").write_text("\n".join(md))

# ---- SVG plot: x = recall@10 (%), y = latency ms (log scale), line per config, p50 solid / p99 dashed ----
W, H = 900, 560
ML, MR, MT, MB = 70, 20, 30, 60
X0, Y0 = ML, H - MB
XW, YH = W - ML - MR, H - MT - MB

labels = sorted({r["label"] for r in rows})
palette = {"ivfrq-vanilla": "#1f77b4", "hnsw-vanilla": "#2ca02c",
           "ivfrq-neon": "#d62728", "hnsw-neon": "#ff7f0e"}

def color(lab):
    return palette.get(lab, "#000000")

xmin, xmax = 0.0, 100.0
ymin, ymax = 0.1, 2000.0

def X(v):  # linear recall
    return X0 + (v - xmin) / (xmax - xmin) * XW

def Y(v):  # log latency
    import math
    ly0, ly1 = math.log10(ymin), math.log10(ymax)
    return Y0 - (math.log10(max(v, ymin)) - ly0) / (ly1 - ly0) * YH

parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
         f'viewBox="0 0 {W} {H}">',
         f'<rect width="{W}" height="{H}" fill="white"/>']
# grid
import math
for g in [0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000]:
    if g < ymin or g > ymax:
        continue
    y = Y(g)
    parts.append(f'<line x1="{X0}" y1="{y:.1f}" x2="{X0+XW}" y2="{y:.1f}" '
                 f'stroke="#e0e0e0" stroke-width="1"/>')
    parts.append(f'<text x="{X0-6}" y="{y+4:.1f}" font-size="11" text-anchor="end" '
                 f'fill="#666">{g:g}</text>')
for g in [0, 20, 40, 60, 80, 100]:
    x = X(g)
    parts.append(f'<line x1="{x:.1f}" y1="{Y0}" x2="{x:.1f}" y2="{MT}" '
                 f'stroke="#e0e0e0" stroke-width="1"/>')
    parts.append(f'<text x="{x:.1f}" y="{Y0+16}" font-size="11" text-anchor="middle" '
                 f'fill="#666">{g}</text>')
parts.append(f'<text x="{X0+XW/2:.0f}" y="{H-12}" font-size="13" text-anchor="middle">'
             f'recall@10 (%)</text>')
parts.append(f'<text x="16" y="{MT+YH/2:.0f}" font-size="13" text-anchor="middle" '
             f'transform="rotate(-90 16 {MT+YH/2:.0f})">latency (ms, log)</text>')

for lab in labels:
    for pct, dash in (("p50", ""), ("p99", "8,4")):
        pts = [(r["recall_at_10"], r[pct + "_ms"]) for r in rows if r["label"] == lab]
        if not pts:
            continue
        col = color(lab)
        path = "M" + " L".join(f"{X(x):.1f} {Y(y):.1f}" for x, y in pts)
        parts.append(f'<path d="{path}" fill="none" stroke="{col}" stroke-width="2" '
                     f'stroke-dasharray="{dash}"/>')
        for x, y in pts:
            parts.append(f'<circle cx="{X(x):.1f}" cy="{Y(y):.1f}" r="3" fill="{col}"/>')
        # legend
        i = labels.index(lab)
        lx, ly = X0 + XW - 250, MT + 16 + i * 34
        parts.append(f'<line x1="{lx}" y1="{ly}" x2="{lx+28}" y2="{ly}" stroke="{col}" '
                     f'stroke-width="2" stroke-dasharray="{dash}"/>')
        parts.append(f'<text x="{lx+36}" y="{ly+4}" font-size="12" fill="{col}">'
                     f'{lab} {pct}</text>')

parts.append("</svg>")
Path(OUT + ".svg").write_text("\n".join(parts))
print(f"wrote {OUT}.md and {OUT}.svg ({len(rows)} rows)")
