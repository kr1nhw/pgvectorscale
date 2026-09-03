-- Load library + set search params in one session.
EXPLAIN (COSTS OFF)
SELECT id FROM items_10m ORDER BY embedding <-> (SELECT q FROM bench_queries WHERE qid=0) LIMIT 10;

SET ivf.probes = 10;
SET ivf.top_k = 1000;

DROP TABLE IF EXISTS res_final_topk1000;
CREATE TABLE res_final_topk1000 (qid int PRIMARY KEY, ids int[]);

\timing on
INSERT INTO res_final_topk1000
SELECT qid, (SELECT array_agg(id) FROM (
  SELECT id FROM items_10m ORDER BY embedding <-> q.q LIMIT 10) t)
FROM (SELECT qid, q FROM bench_queries WHERE qid < 100) q;
\timing off

SELECT count(*) AS results FROM res_final_topk1000;
