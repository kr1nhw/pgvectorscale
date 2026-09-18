-- Local development dataset for the hnswsq build/perf gates (single host).
--
-- 100k rows, dim 16, 200 gaussian-ish clusters of 500 rows (the same shape the
-- in-memory `mem_tests` harness uses, so recall numbers are comparable), plus
-- 200 queries drawn from the cluster centres with a small perturbation and
-- their exact top-10 (ground truth computed by a seq scan).
--
-- Usage:
--   psql -h 127.0.0.1 -p 54331 -U $USER -d postgres -f local_dataset.sql
--
-- Deterministic: `setseed` fixes the whole stream, so rebuilds give the same
-- data and an A/B on two code revisions is apples-to-apples.

\set ON_ERROR_STOP on

SELECT setseed(0.42);

DROP DATABASE IF EXISTS t100kdb;
CREATE DATABASE t100kdb;

\connect t100kdb

CREATE EXTENSION vector;
CREATE EXTENSION vectorscale CASCADE;

CREATE TABLE t100k (id int PRIMARY KEY, embedding vector(16));

-- 200 cluster centres, 500 rows each.
WITH centers AS (
    SELECT g AS cid,
           (SELECT array_agg(random() * 2 - 1) FROM generate_series(1, 16)) AS c
    FROM generate_series(1, 200) g
)
INSERT INTO t100k (id, embedding)
SELECT (row_number() OVER ())::int - 1,
       (SELECT array_agg((c.c[j] + (random() - 0.5) * 0.05)::float4)
        FROM generate_series(1, 16) j)::float4[]::vector(16)
FROM centers c, generate_series(1, 500);

CREATE TABLE bench_queries (qid int PRIMARY KEY, q vector(16));

WITH centers AS (
    SELECT g AS cid,
           (SELECT array_agg(random() * 2 - 1) FROM generate_series(1, 16)) AS c
    FROM generate_series(1, 200) g
)
INSERT INTO bench_queries (qid, q)
SELECT (row_number() OVER ())::int - 1,
       (SELECT array_agg((c.c[j] + (random() - 0.5) * 0.01)::float4)
        FROM generate_series(1, 16) j)::float4[]::vector(16)
FROM centers c;

-- Exact top-10 for every query (seq scan, no index involved).
CREATE TABLE gt_100k AS
SELECT qid, array_agg(id ORDER BY d) AS ids
FROM (
    SELECT q.qid, t.id, t.embedding <-> q.q AS d,
           row_number() OVER (PARTITION BY q.qid ORDER BY t.embedding <-> q.q) AS rn
    FROM bench_queries q, t100k t
) s
WHERE rn <= 10
GROUP BY qid;

ANALYZE t100k;
ANALYZE bench_queries;
SELECT count(*) AS rows, (SELECT count(*) FROM bench_queries) AS queries,
       (SELECT count(*) FROM gt_100k) AS gt_rows FROM t100k;
