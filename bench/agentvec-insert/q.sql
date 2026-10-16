-- pgbench query script: one 10-NN query per transaction against :tbl.
--
-- :nq is the query-pool cardinality (upper bound of random()).
\set qk random(1, :nq)
SELECT id FROM :tbl
 ORDER BY embedding <-> (SELECT q FROM avb_qpool WHERE qk = :qk)
 LIMIT 10;
