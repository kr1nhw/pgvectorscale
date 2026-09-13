-- Recall/latency sweep for the local hnswsq gates.
--
-- Creates `bench_sweep(tbl text, gtbl text, qtbl text, efs int[], k int)`:
-- for every ef in `efs` it runs the k-NN query over all rows of `qtbl` through
-- the *index* (seq scans disabled), compares against `gtbl`, and returns
-- recall@k plus p50/p99 latency.  Table names are arguments, so one function
-- serves every dataset tag/dimension in the database.
--
-- Usage:
--   psql ... -d t100kdb -f local_sweep.sql
--   psql ... -d t100kdb -c "SELECT * FROM bench_sweep('t100k','gt_100k','bench_queries_16',ARRAY[10,40,160,640])"

\set ON_ERROR_STOP on

DROP FUNCTION IF EXISTS bench_sweep(int[], int);

CREATE OR REPLACE FUNCTION bench_sweep(
    tbl text,
    gtbl text,
    qtbl text,
    efs int[],
    k int DEFAULT 10
)
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
    EXECUTE format('SELECT count(*) FROM %I', qtbl) INTO nq;
    FOREACH e IN ARRAY efs LOOP
        PERFORM set_config('hnswsq.ef_search', e::text, true);
        lat := ARRAY[]::numeric[];
        hits := 0;
        FOR rec IN EXECUTE format('SELECT qid, q FROM %I ORDER BY qid', qtbl) LOOP
            DECLARE
                t0 timestamptz := clock_timestamp();
            BEGIN
                EXECUTE format(
                    'SELECT array_agg(id) FROM (SELECT id FROM %I ORDER BY embedding <-> $1 LIMIT %s) s',
                    tbl, k)
                INTO approx USING rec.q;
                lat := lat || (EXTRACT(EPOCH FROM clock_timestamp() - t0) * 1000)::numeric;
            END;
            EXECUTE format('SELECT ids FROM %I WHERE qid = $1', gtbl) INTO exact USING rec.qid;
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
