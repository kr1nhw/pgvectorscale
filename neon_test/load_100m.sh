#!/bin/bash
# Load BIGANN-100M into a dedicated bench100m database (tables named to match
# the sweep harness: items_10m / bench_queries / gt_10m).
set -euxo pipefail
PSQL="/data1/pg17/bin/psql -h 127.0.0.1 -p 5432 -U postgres -X -v ON_ERROR_STOP=1"

$PSQL -c "DROP DATABASE IF EXISTS bench100m;" -c "CREATE DATABASE bench100m;" >/dev/null

$PSQL -d bench100m <<'SQL'
CREATE EXTENSION IF NOT EXISTS vector;
CREATE TABLE items_10m (id bigserial PRIMARY KEY, embedding vector(128) NOT NULL);
CREATE TABLE gt_staging (qid int, rank int, vid int);
CREATE TABLE q_staging (q vector(128));
SQL

echo "== loading 100M vectors (text COPY) =="
time $PSQL -d bench100m -c "\copy items_10m (embedding) FROM '/data1/bigann_100m_vectors.txt'"
echo "== loading ground truth =="
time $PSQL -d bench100m -c "\copy gt_staging FROM '/data1/bigann_ground_truth.txt'"
echo "== loading first 100 queries =="
time $PSQL -d bench100m -c "\copy q_staging FROM PROGRAM 'head -100 /data1/bigann_queries.txt'"

$PSQL -d bench100m <<'SQL'
-- ground truth: top-10 per query (rank 1..10), vid 0-based -> id 1-based
CREATE TABLE gt_10m AS
SELECT qid, array_agg(vid + 1 ORDER BY rank) AS ids
FROM gt_staging WHERE rank <= 10 GROUP BY qid;
ALTER TABLE gt_10m ADD PRIMARY KEY (qid);

CREATE TABLE bench_queries AS
SELECT (row_number() OVER () - 1)::int AS qid, q FROM q_staging;
ALTER TABLE bench_queries ADD PRIMARY KEY (qid);

DROP TABLE gt_staging, q_staging;
ANALYZE items_10m;
SELECT count(*) AS nrows FROM items_10m;
SELECT count(*) AS ngt FROM gt_10m;
SELECT count(*) AS nq FROM bench_queries;
SQL
echo "== 100M load done =="
