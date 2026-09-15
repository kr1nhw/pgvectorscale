#!/usr/bin/env python3
"""Per-query EXPLAIN (ANALYZE, BUFFERS) comparison for one hnsw-family index.

Usage: gap_sweep.py <engine:hnsw|hnswsq> <ef list csv> [table] [queries] [out.csv]

Runs, in ONE backend per (ef, limit) configuration:
  * the user-visible query   -- ORDER BY <-> q LIMIT 10
  * the full drain           -- ORDER BY <-> q LIMIT 100000 (all candidates)
and reports mean/p50 execution time, mean shared blocks (index + heap), and the
mean number of rows the index scan actually produced.

Keeps the same connection for all queries in a configuration so per-backend
startup cost does not dominate; `enable_seqscan = off` forces the ANN index.
"""
import json
import os
import statistics
import subprocess
import sys

PSQL = "/root/.pgrx-hnswsq/17.11/pgrx-install/bin/psql"
CONN = ["-h", "127.0.0.1", "-p", "54330", "-U", "pgtest", "-d", "postgres", "-X", "-q", "-At"]

engine = sys.argv[1] if len(sys.argv) > 1 else "hnswsq"
efs = [int(x) for x in (sys.argv[2] if len(sys.argv) > 2 else "10,40,160,640").split(",")]
table = sys.argv[3] if len(sys.argv) > 3 else "items_1m"
nq = int(sys.argv[4]) if len(sys.argv) > 4 else 100
out_csv = sys.argv[5] if len(sys.argv) > 5 else f"/tmp/gap_sweep_{engine}.csv"

guc = "hnswsq.ef_search" if engine == "hnswsq" else "hnsw.ef_search"


def run_sql(sql_path):
    with open(sql_path) as f:
        p = subprocess.run(["sudo", "-u", "pgtest", PSQL] + CONN + ["-f", sql_path],
                           capture_output=True, text=True)
    if p.returncode != 0:
        raise SystemExit(f"psql failed:\n{p.stderr[:800]}")
    return p.stdout


def parse_json_stream(text):
    """psql prints one JSON array per EXPLAIN; decode them back to back."""
    dec = json.JSONDecoder()
    out, i, n = [], 0, len(text)
    while i < n:
        while i < n and text[i] in " \t\r\n":
            i += 1
        if i >= n:
            break
        obj, j = dec.raw_decode(text, i)
        out.append(obj)
        i = j
    return out


def scan_stats(plan):
    """Walk the plan tree, return the Index Scan node's own counters."""
    stack, best = [plan], None
    while stack:
        node = stack.pop()
        if node.get("Node Type", "").endswith("Index Scan"):
            best = node
        stack.extend(node.get("Plans", []))
    return best


def sweep(ef, limit):
    stmts = ["SET enable_seqscan = off;", f"SET {guc} = {ef};"]
    # EXTRA_SETS: additional SET statements (e.g. hnswsq.sq8_distance) for
    # experiment runs.
    for extra in os.environ.get("EXTRA_SETS", "").split(";"):
        extra = extra.strip()
        if extra:
            stmts.append(f"SET {extra};")
    for qid in range(nq):
        stmts.append(
            f"EXPLAIN (ANALYZE, BUFFERS, FORMAT JSON) SELECT id FROM {table} "
            f"ORDER BY embedding <-> (SELECT q FROM bench_queries WHERE qid = {qid}) "
            f"LIMIT {limit};"
        )
    path = f"/tmp/gap_{engine}_{ef}_{limit}.sql"
    with open(path, "w") as f:
        f.write("\n".join(stmts) + "\n")
    plans = parse_json_stream(run_sql(path))
    times, blocks, rows = [], [], []
    for p in plans:
        plan = p[0]
        times.append(plan["Execution Time"])
        rows.append(plan["Plan"].get("Actual Rows", 0))
        idx = scan_stats(plan["Plan"]) or plan["Plan"]
        blk = idx.get("Shared Hit Blocks", 0) + idx.get("Shared Read Blocks", 0)
        blk += idx.get("Shared Dirtied Blocks", 0)
        blocks.append(blk)
    return times, blocks, rows


print("engine,ef,limit,queries,mean_ms,p50_ms,blocks_mean,index_rows_mean,index_rows_max")
out = open(out_csv, "w")
out.write("engine,ef,limit,queries,mean_ms,p50_ms,blocks_mean,index_rows_mean,index_rows_max\n")
for ef in efs:
    for limit in (10, 100000):
        times, blocks, rows = sweep(ef, limit)
        line = (
            f"{engine},{ef},{limit},{len(times)},{statistics.mean(times):.3f},"
            f"{statistics.median(times):.3f},{statistics.mean(blocks):.1f},"
            f"{statistics.mean(rows):.1f},{max(rows)}"
        )
        print(line)
        out.write(line + "\n")
        sys.stdout.flush()
out.close()
