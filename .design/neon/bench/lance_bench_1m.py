#!/usr/bin/env python
"""Lance (lancedb) benchmark on the BIGANN 1M/100K subsets.

Compares against the hnswsq/pgvector A/B protocol: first 100 queries,
L2, top-10, subset-exact numpy ground truth, recall@10 + p50/p99/mean
latency.  Runs on 113.44.106.182 with /root/miniconda3/bin/python.
"""
import json
import shutil
import time

import lancedb
import numpy as np
import pyarrow as pa

BASE = "/data1/lance_bench"
OUT = "/tmp/lance_bench_1m_results.json"
QUERIES = 100
TOP_K = 10


def load(n):
    v = np.load(f"{BASE}/vectors.npy", mmap_mode="r")[:n].astype(np.float32)
    q = np.load(f"{BASE}/queries.npy", mmap_mode="r")[:QUERIES].astype(np.float32)
    return v, q


def exact_gt(v, q, k=TOP_K):
    # chunked matmul to bound memory: (100, 128) x (n, 128)^T
    dist = np.empty((q.shape[0], v.shape[0]), dtype=np.float32)
    chunk = 1 << 20
    q2 = np.einsum("ij->i", q * q)  # |q|^2
    for s in range(0, v.shape[0], chunk):
        e = min(s + chunk, v.shape[0])
        d = q @ v[s:e].T
        d = -2.0 * d
        d += q2[:, None]
        d += np.einsum("ij->j", v[s:e] * v[s:e])[None, :]
        dist[:, s:e] = d
    return np.argsort(dist, axis=1)[:, :k]


def bench(n, label, results):
    t0 = time.time()
    v, q = load(n)
    t1 = time.time()
    gt = exact_gt(v, q)
    t2 = time.time()

    uri = f"/tmp/lance_bench_{label}"
    shutil.rmtree(uri, ignore_errors=True)
    db = lancedb.connect(uri)
    tids = np.arange(n, dtype=np.int64)
    tbl = db.create_table(
        "bench",
        data=pa.table({"id": pa.array(tids), "vector": pa.FixedSizeListArray.from_arrays(v.ravel(), 128)}),
    )
    t3 = time.time()
    size_pre = sum(f.stat().st_size for f in __import__("pathlib").Path(uri).rglob("*") if f.is_file())

    for cfg_name, cfg in [
        ("hnsq_p16_sq8", dict(index_type="IVF_HNSW_SQ", num_partitions=16, num_sub_vectors=8)),
        ("hnsq_p64_sq8", dict(index_type="IVF_HNSW_SQ", num_partitions=64, num_sub_vectors=8)),
        ("sq_p64", dict(index_type="IVF_SQ", num_partitions=64, num_sub_vectors=8)),
    ]:
        tb = time.time()
        tbl.create_index(metric="l2", **cfg)
        te = time.time()
        size_post = sum(f.stat().st_size for f in __import__("pathlib").Path(uri).rglob("*") if f.is_file())
        for nprobes in (1, 2, 4, 8, 16, 32):
            found = np.empty((QUERIES, TOP_K), dtype=np.int64)
            lat = []
            for i in range(QUERIES):
                qs = time.perf_counter()
                res = tbl.search(q[i]).metric("l2").limit(TOP_K).nprobes(nprobes).to_arrow()
                lat.append((time.perf_counter() - qs) * 1000.0)
                ids = res["id"].to_numpy()
                found[i, : len(ids)] = ids[:TOP_K]
            hits = sum(
                1
                for i in range(QUERIES)
                for j in found[i]
                if j in set(gt[i].tolist())
            )
            recall = hits / (QUERIES * TOP_K)
            lat = np.array(lat)
            results.append(
                {
                    "dataset": label,
                    "n": n,
                    "index": cfg_name,
                    "build_s": round(te - tb, 2),
                    "size_bytes": size_post,
                    "nprobes": nprobes,
                    "recall@10": round(recall, 4),
                    "p50_ms": round(float(np.percentile(lat, 50)), 3),
                    "p95_ms": round(float(np.percentile(lat, 95)), 3),
                    "p99_ms": round(float(np.percentile(lat, 99)), 3),
                    "mean_ms": round(float(lat.mean()), 3),
                }
            )
            print(results[-1], flush=True)
    results.append(
        {
            "dataset": label,
            "n": n,
            "index": "none",
            "build_s": 0,
            "size_bytes": size_pre,
            "nprobes": 0,
            "recall@10": 1.0,
            "p50_ms": 0,
            "p95_ms": 0,
            "p99_ms": 0,
            "mean_ms": 0,
            "meta": f"load_s={t1-t0:.2f} gt_s={t2-t1:.2f} write_s={t3-t2:.2f}",
        }
    )
    shutil.rmtree(uri, ignore_errors=True)


def main():
    results = []
    bench(100_000, "bigann_100k", results)
    bench(1_000_000, "bigann_1m", results)
    with open(OUT, "w") as f:
        json.dump(results, f, indent=2)
    print("WROTE", OUT)


if __name__ == "__main__":
    main()
