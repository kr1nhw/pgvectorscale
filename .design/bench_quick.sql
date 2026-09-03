EXPLAIN (COSTS OFF)
SELECT id FROM items_10m ORDER BY embedding <-> (SELECT q FROM bench_queries WHERE qid=0) LIMIT 10;

-- quick 20-query recall sample, one probes setting per run (edit via -v)
SET ivf.probes = :probes;
SET ivf.top_k = 1000;

DROP TABLE IF EXISTS res_quick;
CREATE TABLE res_quick (qid int PRIMARY KEY, ids int[]);

\timing on
INSERT INTO res_quick
SELECT qid, (SELECT array_agg(id) FROM (
  SELECT id FROM items_10m ORDER BY embedding <-> q.q LIMIT 10) t)
FROM (SELECT qid, q FROM bench_queries WHERE qid < 20) q;
\timing off
