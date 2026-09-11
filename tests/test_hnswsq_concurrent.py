"""
Concurrent-operation tests for the hnswsq access method.

These exercise the pgvector-style concurrent insert protocol (no global
writer lock; per-page locking + two-phase optimistic neighbor updates)
against racing readers, deleters, and VACUUM — the same style of stress
used to validate the IVF append-only segment design:

- concurrent INSERTs from multiple backends (no deadlocks/hangs/lost rows),
- concurrent INSERT + SELECT + DELETE + VACUUM (mixed workload),
- the same for the reduced-precision layouts (ieeefp8, f8/SQ8).

A row is considered "found" when an exact-match probe
(ORDER BY embedding <-> row LIMIT k) returns its id among the top-k.
Plain storage must find >= 98% of probed rows at ef_search=400; quantized
layouts allow a small tolerance for code ties/quantization noise.
"""
import pytest
import threading
import time
import numpy as np
import psycopg2
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError

DIMENSIONS = 8
# Overall wall-clock budget for the thread pools (a deadlock shows up here).
POOL_TIMEOUT_S = 600


def _vec_str(v):
    return "[" + ",".join(f"{x:.6f}" for x in v) + "]"


def _connect(db_setup):
    conn = psycopg2.connect(**db_setup)
    conn.autocommit = True
    return conn


def _setup_table(db_setup, table, layout, dim=DIMENSIONS):
    conn = _connect(db_setup)
    try:
        with conn.cursor() as cur:
            cur.execute(f"DROP TABLE IF EXISTS {table} CASCADE")
            cur.execute(
                f"CREATE TABLE {table} (id BIGSERIAL PRIMARY KEY, embedding vector({dim}))"
            )
            cur.execute(
                f"CREATE INDEX {table}_idx ON {table} "
                f"USING hnswsq (embedding vector_l2_ops) "
                f"WITH (storage_layout = {layout}, m = 16, ef_construction = 64)"
            )
    finally:
        conn.close()


def _insert_batch(db_setup, table, n):
    """Insert n random vectors; returns the (id, vector) pairs inserted."""
    conn = _connect(db_setup)
    inserted = []
    try:
        with conn.cursor() as cur:
            for _ in range(n):
                v = np.random.rand(DIMENSIONS) * 2 - 1
                cur.execute(
                    f"INSERT INTO {table} (embedding) VALUES (%s) RETURNING id",
                    (_vec_str(v),),
                )
                inserted.append((cur.fetchone()[0], v))
    finally:
        conn.close()
    return inserted


def _probe_rows(db_setup, table, rows, topk=3, ef_search=400):
    """Exact-match probes; returns the fraction of rows found in the top-k."""
    if not rows:
        return 1.0
    conn = _connect(db_setup)
    found = 0
    try:
        with conn.cursor() as cur:
            cur.execute(f"SET hnswsq.ef_search = {ef_search}")
            cur.execute("SET enable_seqscan = off")
            for rid, v in rows:
                cur.execute(
                    f"SELECT id FROM {table} ORDER BY embedding <-> %s LIMIT {topk}",
                    (_vec_str(v),),
                )
                ids = [r[0] for r in cur.fetchall()]
                if rid in ids:
                    found += 1
    finally:
        conn.close()
    return found / len(rows)


@pytest.mark.concurrency
def test_hnswsq_concurrent_inserts_plain(db_setup):
    """4 writer backends race inserts into a plain-layout hnswsq index."""
    table = "hs_conc_plain"
    _setup_table(db_setup, table, "plain")

    writers = 4
    batches = 10
    per_batch = 50
    all_inserted = []
    lock = threading.Lock()
    errors = []

    def writer():
        try:
            for _ in range(batches):
                rows = _insert_batch(db_setup, table, per_batch)
                with lock:
                    all_inserted.extend(rows)
        except Exception as e:  # noqa: BLE001
            with lock:
                errors.append(e)

    with ThreadPoolExecutor(max_workers=writers) as ex:
        futures = [ex.submit(writer) for _ in range(writers)]
        for f in futures:
            f.result(timeout=POOL_TIMEOUT_S)

    assert not errors, f"concurrent inserts failed: {errors[:3]}"

    conn = _connect(db_setup)
    try:
        with conn.cursor() as cur:
            cur.execute(f"SELECT count(*) FROM {table}")
            count = cur.fetchone()[0]
    finally:
        conn.close()
    assert count == writers * batches * per_batch, f"lost rows: {count}"

    # No lost index entries: probe a sample of the concurrently inserted rows.
    sample = all_inserted[:: max(1, len(all_inserted) // 150)]
    hit = _probe_rows(db_setup, table, sample)
    assert hit >= 0.98, f"plain layout lost index entries: hit rate {hit}"

    conn = _connect(db_setup)
    try:
        with conn.cursor() as cur:
            cur.execute(f"DROP TABLE {table} CASCADE")
    finally:
        conn.close()


@pytest.mark.concurrency
def test_hnswsq_concurrent_inserts_fp8(db_setup):
    """Same race on the training-free ieeefp8 layout (top-k tolerance for ties)."""
    table = "hs_conc_fp8"
    _setup_table(db_setup, table, "ieeefp8")

    writers = 4
    batches = 8
    per_batch = 50
    all_inserted = []
    lock = threading.Lock()
    errors = []

    def writer():
        try:
            for _ in range(batches):
                rows = _insert_batch(db_setup, table, per_batch)
                with lock:
                    all_inserted.extend(rows)
        except Exception as e:  # noqa: BLE001
            with lock:
                errors.append(e)

    with ThreadPoolExecutor(max_workers=writers) as ex:
        futures = [ex.submit(writer) for _ in range(writers)]
        for f in futures:
            f.result(timeout=POOL_TIMEOUT_S)

    assert not errors, f"concurrent inserts failed: {errors[:3]}"

    conn = _connect(db_setup)
    try:
        with conn.cursor() as cur:
            cur.execute(f"SELECT count(*) FROM {table}")
            count = cur.fetchone()[0]
    finally:
        conn.close()
    assert count == writers * batches * per_batch, f"lost rows: {count}"

    sample = all_inserted[:: max(1, len(all_inserted) // 150)]
    hit = _probe_rows(db_setup, table, sample, topk=5)
    assert hit >= 0.95, f"ieeefp8 layout lost index entries: hit rate {hit}"

    conn = _connect(db_setup)
    try:
        with conn.cursor() as cur:
            cur.execute(f"DROP TABLE {table} CASCADE")
    finally:
        conn.close()


@pytest.mark.concurrency
@pytest.mark.slow
def test_hnswsq_concurrent_mixed_operations(db_setup):
    """INSERT + SELECT + DELETE + VACUUM racing on the same hnswsq index.

    The IVF validation workload, adapted: 3 inserters, 2 continuous readers,
    1 deleter, and a periodic VACUUM — all against one index.  Success means
    zero errors/deadlocks, exact final counts, and (after a settling VACUUM)
    a high probe hit rate for surviving rows.
    """
    table = "hs_conc_mixed"
    _setup_table(db_setup, table, "plain")

    inserted_ids = []
    deleted_ids = set()
    lock = threading.Lock()
    errors = []
    stop = threading.Event()

    def inserter(n_batches, per_batch):
        try:
            for _ in range(n_batches):
                if stop.is_set():
                    return
                rows = _insert_batch(db_setup, table, per_batch)
                with lock:
                    inserted_ids.extend(rid for rid, _ in rows)
                time.sleep(0.02)
        except Exception as e:  # noqa: BLE001
            with lock:
                errors.append(("inserter", e))

    def reader(n_queries):
        conn = _connect(db_setup)
        try:
            with conn.cursor() as cur:
                cur.execute("SET hnswsq.ef_search = 100")
                cur.execute("SET enable_seqscan = off")
                for _ in range(n_queries):
                    if stop.is_set():
                        return
                    q = _vec_str(np.random.rand(DIMENSIONS) * 2 - 1)
                    cur.execute(
                        f"SELECT id, embedding <-> %s FROM {table} ORDER BY embedding <-> %s LIMIT 10",
                        (q, q),
                    )
                    cur.fetchall()
                    time.sleep(0.01)
        except Exception as e:  # noqa: BLE001
            with lock:
                errors.append(("reader", e))
        finally:
            conn.close()

    def deleter(n_rounds, per_round):
        conn = _connect(db_setup)
        try:
            for _ in range(n_rounds):
                if stop.is_set():
                    return
                with lock:
                    candidates = [i for i in inserted_ids if i not in deleted_ids]
                    victims = candidates[:per_round]
                    deleted_ids.update(victims)
                if victims:
                    with conn.cursor() as cur:
                        cur.execute(
                            f"DELETE FROM {table} WHERE id = ANY(%s)", (victims,)
                        )
                time.sleep(0.05)
        except Exception as e:  # noqa: BLE001
            with lock:
                errors.append(("deleter", e))
        finally:
            conn.close()

    def vaccumer(rounds, interval_s):
        conn = _connect(db_setup)
        try:
            for _ in range(rounds):
                if stop.is_set():
                    return
                time.sleep(interval_s)
                with conn.cursor() as cur:
                    cur.execute(f"VACUUM {table}")
        except Exception as e:  # noqa: BLE001
            with lock:
                errors.append(("vaccumer", e))
        finally:
            conn.close()

    try:
        with ThreadPoolExecutor(max_workers=7) as ex:
            futures = []
            for _ in range(3):
                futures.append(ex.submit(inserter, 12, 40))
            for _ in range(2):
                futures.append(ex.submit(reader, 60))
            futures.append(ex.submit(deleter, 15, 20))
            futures.append(ex.submit(vaccumer, 4, 3.0))
            for f in futures:
                f.result(timeout=POOL_TIMEOUT_S)
    finally:
        stop.set()

    assert not errors, f"mixed workload failed: {errors[:3]}"

    # Exact final count.
    conn = _connect(db_setup)
    try:
        with conn.cursor() as cur:
            cur.execute(f"SELECT count(*) FROM {table}")
            count = cur.fetchone()[0]
            # Let the last vacuum settle, then re-check surviving rows are
            # still findable through the index.
            cur.execute(f"VACUUM {table}")
    finally:
        conn.close()

    with lock:
        expected = len(inserted_ids) - len(deleted_ids & set(inserted_ids))
    assert count == expected, f"row count mismatch: {count} != {expected}"

    # Re-read the surviving rows (ids + vectors) and probe a sample.
    conn = _connect(db_setup)
    rows = []
    try:
        with conn.cursor() as cur:
            cur.execute(f"SELECT id, embedding FROM {table} ORDER BY id")
            for rid, emb in cur.fetchall():
                v = np.array(
                    [float(x) for x in str(emb).strip("[]").split(",")], dtype=np.float32
                )
                rows.append((rid, v))
    finally:
        conn.close()
    sample = rows[:: max(1, len(rows) // 150)]
    hit = _probe_rows(db_setup, table, sample)
    assert hit >= 0.98, f"mixed workload lost index entries: hit rate {hit}"

    conn = _connect(db_setup)
    try:
        with conn.cursor() as cur:
            cur.execute(f"DROP TABLE {table} CASCADE")
    finally:
        conn.close()
