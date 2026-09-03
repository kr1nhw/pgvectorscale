EXPLAIN (COSTS OFF)
SELECT id FROM items_10m ORDER BY embedding <-> (SELECT q FROM bench_queries WHERE qid=0) LIMIT 10;

SET ivf.probes = 10;

-- top_k = 100
SET ivf.top_k = 100;
DROP TABLE IF EXISTS res_final_topk100;
CREATE TABLE res_final_topk100 (qid int PRIMARY KEY, ids int[]);
INSERT INTO res_final_topk100
SELECT qid, (SELECT array_agg(id) FROM (
  SELECT id FROM items_10m ORDER BY embedding <-> q.q LIMIT 10) t)
FROM (SELECT qid, q FROM bench_queries WHERE qid < 100) q;

-- top_k = 500
SET ivf.top_k = 500;
DROP TABLE IF EXISTS res_final_topk500;
CREATE TABLE res_final_topk500 (qid int PRIMARY KEY, ids int[]);
INSERT INTO res_final_topk500
SELECT qid, (SELECT array_agg(id) FROM (
  SELECT id FROM items_10m ORDER BY embedding <-> q.q LIMIT 10) t)
FROM (SELECT qid, q FROM bench_queries WHERE qid < 100) q;

-- top_k = 10000
SET ivf.top_k = 10000;
DROP TABLE IF EXISTS res_final_topk10000;
CREATE TABLE res_final_topk10000 (qid int PRIMARY KEY, ids int[]);
INSERT INTO res_final_topk10000
SELECT qid, (SELECT array_agg(id) FROM (
  SELECT id FROM items_10m ORDER BY embedding <-> q.q LIMIT 10) t)
FROM (SELECT qid, q FROM bench_queries WHERE qid < 100) q;
