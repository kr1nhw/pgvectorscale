-- pgbench insert script: one row per transaction.
--
-- :tbl selects the target (avb_t_heap | avb_t_btree | avb_t_av | avb_t_pgv).
-- :nk  is the pool cardinality (upper bound of random()).
--
-- The same two-statement shape is used for every target, so the pool lookup
-- is a constant additive cost present in all variants (and is quantified by
-- the avb_t_heap baseline, which has no secondary index maintenance beyond
-- its own btree pkey).
\set k random(1, :nk)
INSERT INTO :tbl (embedding) SELECT embedding FROM avb_pool WHERE k = :k;
