-- pgvectorscale IVF/RaBitQ functional + recall + DML test for Neon (compute 55432)
\set ON_ERROR_STOP on
SET client_min_messages = warning;

CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS vectorscale;
SELECT extname, extversion FROM pg_extension WHERE extname IN ('vector', 'vectorscale') ORDER BY 1;

DROP TABLE IF EXISTS items CASCADE;

-- 20k rows, dim 8, 8 well-separated clusters (centroids at +/-3 per coordinate)
CREATE TABLE items (id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY, embedding vector(8) NOT NULL);

WITH centroids AS (
  SELECT v FROM (VALUES
    (ARRAY[ 3, 3, 3, 3, 3, 3, 3, 3]::real[]),
    (ARRAY[ 3, 3, 3, 3,-3,-3,-3,-3]::real[]),
    (ARRAY[ 3, 3,-3,-3, 3, 3,-3,-3]::real[]),
    (ARRAY[ 3, 3,-3,-3,-3,-3, 3, 3]::real[]),
    (ARRAY[ 3,-3, 3,-3, 3,-3, 3,-3]::real[]),
    (ARRAY[ 3,-3, 3,-3,-3, 3,-3, 3]::real[]),
    (ARRAY[ 3,-3,-3, 3, 3,-3,-3, 3]::real[]),
    (ARRAY[ 3,-3,-3, 3,-3, 3, 3,-3]::real[])
  ) c(v)
), pts AS (
  SELECT g, (SELECT v FROM centroids OFFSET floor(random()*8)::int LIMIT 1) AS c
  FROM generate_series(1, 20000) g
)
INSERT INTO items (embedding)
SELECT (
  ARRAY[
    c[1] + (random()-0.5), c[2] + (random()-0.5), c[3] + (random()-0.5), c[4] + (random()-0.5),
    c[5] + (random()-0.5), c[6] + (random()-0.5), c[7] + (random()-0.5), c[8] + (random()-0.5)
  ]::real[]
)::vector
FROM pts;

ANALYZE items;
SELECT count(*) AS nrows FROM items;

-- ---------- Part 1: build an index for every supported num_bits ----------
CREATE INDEX items_ivf_1 ON items USING ivf (embedding vector_l2_ops) WITH (lists = 8, num_bits = 1);
CREATE INDEX items_ivf_2 ON items USING ivf (embedding vector_l2_ops) WITH (lists = 8, num_bits = 2);
CREATE INDEX items_ivf_4 ON items USING ivf (embedding vector_l2_ops) WITH (lists = 8, num_bits = 4);
CREATE INDEX items_ivf_8 ON items USING ivf (embedding vector_l2_ops) WITH (lists = 8, num_bits = 8);

SELECT indexrelid::regclass AS idx, indisvalid, indisready
FROM pg_index WHERE indrelid = 'items'::regclass ORDER BY 1;

-- ---------- Part 2: functional queries through the index ----------
SET enable_seqscan = off;
SET ivf.probes = 4;

EXPLAIN (COSTS OFF, SUMMARY OFF)
SELECT id FROM items ORDER BY embedding <-> '[3,3,3,3,3,3,3,3]'::vector LIMIT 5;

-- top-3 distances to centroid (3,3,3,3,3,3,3,3) must be small (< ~1.0)
SELECT id, round((embedding <-> '[3,3,3,3,3,3,3,3]'::vector)::numeric, 3) AS dist
FROM items ORDER BY embedding <-> '[3,3,3,3,3,3,3,3]'::vector LIMIT 3;

-- ---------- Part 3: DML + VACUUM + MVCC smoke ----------
UPDATE items SET embedding = (
  ARRAY[
    -3 + (random()-0.5), -3 + (random()-0.5), -3 + (random()-0.5), -3 + (random()-0.5),
    -3 + (random()-0.5), -3 + (random()-0.5), -3 + (random()-0.5), -3 + (random()-0.5)
  ]::real[]
)::vector WHERE id % 40 = 0;   -- 500 rows move to the opposite cluster

DELETE FROM items WHERE id % 41 = 0;  -- ~488 rows deleted

VACUUM items;
ANALYZE items;

-- after DML, the query against the moved cluster must still work
SELECT id, round((embedding <-> '[-3,-3,-3,-3,-3,-3,-3,-3]'::vector)::numeric, 3) AS dist
FROM items ORDER BY embedding <-> '[-3,-3,-3,-3,-3,-3,-3,-3]'::vector LIMIT 3;

-- fresh inserts land in the index (not committed yet -> invisible, then committed)
BEGIN;
INSERT INTO items (embedding) VALUES ('[3,3,3,3,3,3,3,3]');
SELECT 'uncommitted_visible' AS check, count(*) FROM items WHERE id = (SELECT max(id) FROM items);
ROLLBACK;

INSERT INTO items (embedding) VALUES ('[3,3,3,3,3,3,3,3]');
SELECT id, round((embedding <-> '[3,3,3,3,3,3,3,3]'::vector)::numeric, 3) AS dist
FROM items ORDER BY embedding <-> '[3,3,3,3,3,3,3,3]'::vector LIMIT 1;

-- ---------- Part 4: per-num_bits recall (index rebuilt with a single bit width) ----------
-- drop the Part 1 indexes so exactly one ivf index exists during each recall pass
DROP INDEX items_ivf_1, items_ivf_2, items_ivf_4, items_ivf_8;

CREATE TEMP TABLE probes AS SELECT id, embedding FROM items WHERE id % 1000 = 0 LIMIT 20;

-- exact baseline via brute-force seq scan (index scans disabled)
SET enable_indexscan = off;
SET enable_bitmapscan = off;
SET enable_seqscan = on;

CREATE TEMP TABLE exact_top AS
SELECT p.id AS qid, t.id AS nid
FROM probes p
CROSS JOIN LATERAL (
  SELECT id FROM items ORDER BY embedding <-> p.embedding LIMIT 10
) t;

-- now route ORDER BY <-> through the ivf index
SET enable_indexscan = on;
SET enable_bitmapscan = on;
SET enable_seqscan = off;
SET ivf.probes = 8;   -- probe all lists so recall measures quantization loss

CREATE TEMP TABLE recall_results (num_bits int, recall_pct numeric);

DO $$
DECLARE
  nb int;
  q record;
  hit int;
  total_hits int;
  total_exact int;
BEGIN
  FOREACH nb IN ARRAY ARRAY[1,2,4,8] LOOP
    DROP INDEX IF EXISTS items_ivf_bits;
    EXECUTE format('CREATE INDEX items_ivf_bits ON items USING ivf (embedding vector_l2_ops) WITH (lists = 8, num_bits = %s)', nb);
    total_hits := 0;
    total_exact := 0;
    FOR q IN SELECT id FROM probes LOOP
      total_exact := total_exact + 10;
      SELECT count(*) INTO hit
      FROM exact_top e
      WHERE e.qid = q.id
        AND e.nid IN (
          SELECT i3.id FROM items i3
          ORDER BY i3.embedding <-> (SELECT embedding FROM items WHERE id = q.id)
          LIMIT 10
        );
      total_hits := total_hits + hit;
    END LOOP;
    INSERT INTO recall_results VALUES (nb, round(100.0 * total_hits / total_exact, 1));
  END LOOP;
END
$$;

SELECT * FROM recall_results ORDER BY num_bits;

\echo DONE
