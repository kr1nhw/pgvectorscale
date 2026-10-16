-- Setup for the agentvec insertion-throughput benchmark.
--
-- Creates, in the CURRENT database (the 1B BIGANN database), tables with the
-- `avb_` prefix:
--   avb_pool    : a pool of 128-d vectors sampled from the real BIGANN items
--   avb_qpool   : a pool of query vectors (for the combined insert+query test)
--   avb_t_heap  : insert target with NO index          (pure heap+WAL cost)
--   avb_t_btree : insert target with a btree pkey      (+ btree maintenance)
--   avb_t_av    : insert target with an agentvec index (+ hnswsq hot segment)
--   avb_t_pgv   : insert target with a pgvector HNSW   (reference: same commit)
--
-- All targets share the same shape so the only difference is index maintenance.
--
-- Usage:
--   psql -v pool_rows=200000 -v qpool_rows=200 -f 00_setup.sql
--
-- NOTE: must run against the *release* build of the extension.

\set ON_ERROR_STOP on
\if :{?pool_rows}
\else
  \set pool_rows 200000
\endif
\if :{?qpool_rows}
\else
  \set qpool_rows 10000
\endif
\if :{?src_table}
\else
  \set src_table items_1b
\endif
\if :{?build_pgv}
\else
  \set build_pgv 0
\endif

CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS vectorscale;

DROP TABLE IF EXISTS avb_t_av CASCADE;
DROP TABLE IF EXISTS avb_t_pgv CASCADE;
DROP TABLE IF EXISTS avb_t_btree CASCADE;
DROP TABLE IF EXISTS avb_t_heap CASCADE;

-- ---------------------------------------------------------------------------
-- Vector pools, sampled from the real BIGANN items / queries.
-- ---------------------------------------------------------------------------
DROP TABLE IF EXISTS avb_pool;
CREATE TABLE avb_pool (k integer PRIMARY KEY, embedding vector(128));
INSERT INTO avb_pool (k, embedding)
SELECT row_number() OVER (), embedding
FROM (SELECT embedding FROM :src_table ORDER BY id LIMIT :pool_rows) s;
ANALYZE avb_pool;
SELECT count(*) AS pool_rows FROM avb_pool;

DROP TABLE IF EXISTS avb_qpool;
CREATE TABLE avb_qpool (qk integer PRIMARY KEY, q vector(128));
INSERT INTO avb_qpool (qk, q)
SELECT row_number() OVER (), q
FROM (SELECT q FROM bench_queries ORDER BY qid LIMIT :qpool_rows) s;
ANALYZE avb_qpool;
SELECT count(*) AS qpool_rows FROM avb_qpool;

-- ---------------------------------------------------------------------------
-- Insert targets.  bigserial id keeps the row-identity work identical across
-- variants (a sequence nextval plus a heap tuple).
-- ---------------------------------------------------------------------------
CREATE TABLE avb_t_heap  (id bigserial PRIMARY KEY, embedding vector(128));
CREATE TABLE avb_t_btree (id bigserial, embedding vector(128), PRIMARY KEY (id));
CREATE TABLE avb_t_av    (id bigserial PRIMARY KEY, embedding vector(128));
CREATE TABLE avb_t_pgv   (id bigserial PRIMARY KEY, embedding vector(128));

-- The agentvec options mirror the production 1B index (items_1b_av):
--   ivf_lists=2000 rabitq_bits=1 hot_segment_max_rows=50000
--   ivf_probes=256 search_candidates=8000
CREATE INDEX avb_t_av_idx ON avb_t_av USING agentvec (embedding)
    WITH (ivf_lists='2000', rabitq_bits='1', hot_segment_max_rows='50000',
          ivf_probes='256', search_candidates='8000');

\if :build_pgv
CREATE INDEX avb_t_pgv_idx ON avb_t_pgv USING hnsw (embedding vector_l2_ops)
    WITH (m='16', ef_construction='64');
\endif

SELECT 'setup complete' AS status;
