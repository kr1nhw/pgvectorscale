-- Recall/latency sweep for the local hnswsq gates.
--
-- Creates `bench_sweep(efs int[], k int)`: for every ef in `efs` it runs the k-NN
-- query over all `bench_queries` rows through the *index* (seq scans disabled),
-- compares against `gt_<n>`, and returns recall@k plus p50/p99 latency.
--
-- Usage:
--   psql ... -d t100kdb -f local_sweep.sql
--   psql ... -d t100kdb -c "SELECT * FROM bench_sweep(ARRAY[10,40,160,640])"

\set ON_ERROR_STOP on

CREATE OR REPLACE FUNCTION bench_sweep(efs int[], k int DEFAULT 10)
RETURNS TABLE(ef int, recall numeric, p50_ms numeric, p99_ms numeric)
LANGUAGE plpgsql AS $$
DECLARE
    e int;
    rec record;
    approx int[];
    exact int[];
    lat numeric[];
    hits bigint;
    nq bigint;
BEGIN
    PERFORM set_config('enable_seqscan', 'off', true);
    SELECT count(*) INTO nq FROM bench_queries;
    FOREACH e IN ARRAY efs LOOP
        PERFORM set_config('hnswsq.ef_search', e::text, true);
        lat := ARRAY[]::numeric[];
        hits := 0;
        FOR rec IN SELECT qid, q FROM bench_queries ORDER BY qid LOOP
            DECLARE
                t0 timestamptz := clock_timestamp();
            BEGIN
                SELECT array_agg(id) INTO approx FROM (
                    SELECT t.id FROM t100k t ORDER BY t.embedding <-> rec.q LIMIT k
                ) s;
                lat := lat || (EXTRACT(EPOCH FROM clock_timestamp() - t0) * 1000)::numeric;
            END;
            SELECT ids INTO exact FROM gt_100k WHERE qid = rec.qid;
            SELECT hits + count(*) INTO hits
            FROM (SELECT unnest(exact) INTERSECT SELECT unnest(approx)) x;
        END LOOP;
        ef := e;
        recall := round(hits::numeric / (k * nq), 4);
        SELECT round(percentile_cont(0.5) WITHIN GROUP (ORDER BY l)::numeric, 3),
               round(percentile_cont(0.99) WITHIN GROUP (ORDER BY l)::numeric, 3)
        INTO p50_ms, p99_ms FROM unnest(lat) l;
        RETURN NEXT;
    END LOOP;
END $$;
