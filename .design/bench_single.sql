-- Load the vectorscale library via planning (registers ivf.* GUCs).
EXPLAIN (COSTS OFF)
SELECT id FROM items_10m ORDER BY embedding <-> (SELECT q FROM bench_queries WHERE qid=0) LIMIT 10;

SET ivf.probes = 10;
SET ivf.top_k = 1000;
SHOW ivf.probes;
SHOW ivf.top_k;

\timing on
SELECT id FROM items_10m ORDER BY embedding <-> (SELECT q FROM bench_queries WHERE qid=0) LIMIT 10;
\timing off
