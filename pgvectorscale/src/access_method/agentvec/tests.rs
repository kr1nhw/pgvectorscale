//! Tests for the `agentvec` access method.
//!
//! Two kinds of coverage:
//!
//! * **Behavioural** — create/insert/update/delete/vacuum/reindex through SQL,
//!   always with `enable_seqscan = off` so the plan really exercises the index
//!   (results are compared against explicitly known exact distances).
//! * **Structural** — the on-disk model is inspected directly (meta page,
//!   directory, segment headers) so the tests fail if sealing or the directory
//!   stop being what the design says they are.

#[cfg(any(test, feature = "pg_test"))]
#[pgrx::pg_schema]
mod tests {
    use pgrx::*;

    use crate::access_method::agentvec::directory::{
        AgentVecDirectory, AgentVecSegmentHeader, SegmentOwnership, SegmentState,
    };
    use crate::access_method::agentvec::meta_page::AgentVecMetaPage;
    use crate::access_method::agentvec::options::TSVAgentVecOptions;
    use crate::access_method::agentvec::router;
    use rand::rngs::SmallRng;
    use rand::{Rng, SeedableRng};

    /// Open an index relation by name.
    fn index_relation(name: &str) -> PgRelation {
        let oid = Spi::get_one::<pg_sys::Oid>(&format!("SELECT '{name}'::regclass::oid"))
            .unwrap()
            .expect("index oid was null");
        unsafe { PgRelation::from_pg(pg_sys::RelationIdGetRelation(oid)) }
    }

    /// The ids returned by an approximate-nearest-neighbour query, in order.
    fn search_ids(table: &str, column: &str, op: &str, query: &str, limit: usize) -> Vec<i32> {
        Spi::connect(|client| {
            let sql =
                format!("SELECT id FROM {table} ORDER BY {column} {op} '{query}' LIMIT {limit}");
            let table = client.select(&sql, None, &[])?;
            let mut ids = Vec::new();
            for row in table {
                ids.push(row.get::<i32>(1)?.expect("id was null"));
            }
            Ok::<Vec<i32>, spi::Error>(ids)
        })
        .unwrap()
    }

    /// Force index usage for the rest of the test transaction.
    fn force_index_scan() {
        Spi::run("SET enable_seqscan = off").unwrap();
    }

    #[pg_test]
    fn test_agentvec_l2_exact_search() -> spi::Result<()> {
        Spi::run(
            "CREATE TABLE t_av_l2(id int, v vector(3));
             INSERT INTO t_av_l2 VALUES
                (1, '[1,0,0]'), (2, '[2,0,0]'), (3, '[3,0,0]'), (4, '[4,0,0]');
             CREATE INDEX idx_av_l2 ON t_av_l2 USING agentvec (v vector_l2_ops);",
        )?;
        force_index_scan();

        assert_eq!(
            search_ids("t_av_l2", "v", "<->", "[0,0,0]", 4),
            vec![1, 2, 3, 4],
            "L2 order must be exact"
        );
        assert_eq!(
            search_ids("t_av_l2", "v", "<->", "[100,0,0]", 2),
            vec![4, 3],
            "closest first"
        );
        Ok(())
    }

    /// A row inserted after the index exists is searchable immediately, with no
    /// background step (design §32.1).
    #[pg_test]
    fn test_agentvec_insert_is_immediately_searchable() -> spi::Result<()> {
        Spi::run(
            "CREATE TABLE t_av_ins(id int, v vector(3));
             CREATE INDEX idx_av_ins ON t_av_ins USING agentvec (v vector_l2_ops);",
        )?;
        force_index_scan();

        // Empty index first: the plan must still work (no rows).
        assert!(search_ids("t_av_ins", "v", "<->", "[0,0,0]", 5).is_empty());

        Spi::run(
            "INSERT INTO t_av_ins VALUES (1, '[5,0,0]');
             INSERT INTO t_av_ins VALUES (2, '[1,0,0]');
             INSERT INTO t_av_ins VALUES (3, '[3,0,0]');",
        )?;
        assert_eq!(
            search_ids("t_av_ins", "v", "<->", "[0,0,0]", 3),
            vec![2, 3, 1],
            "inserted rows must be immediately visible to the index"
        );
        Ok(())
    }

    /// UPDATE inserts the new version and leaves the old one to heap visibility,
    /// so the query sees only the new vector.
    #[pg_test]
    fn test_agentvec_update_reflects_new_vector() -> spi::Result<()> {
        Spi::run(
            "CREATE TABLE t_av_upd(id int, v vector(3));
             INSERT INTO t_av_upd VALUES (1, '[9,0,0]'), (2, '[8,0,0]');
             CREATE INDEX idx_av_upd ON t_av_upd USING agentvec (v vector_l2_ops);",
        )?;
        force_index_scan();
        assert_eq!(search_ids("t_av_upd", "v", "<->", "[0,0,0]", 2), vec![2, 1]);

        Spi::run("UPDATE t_av_upd SET v = '[0.5,0,0]' WHERE id = 1")?;
        force_index_scan();
        assert_eq!(
            search_ids("t_av_upd", "v", "<->", "[0,0,0]", 2),
            vec![1, 2],
            "the updated vector must be the one the index returns"
        );
        Ok(())
    }

    /// A tombstoned entry is skipped by scans and stops consuming candidate
    /// slots.
    ///
    /// The tombstone goes through `agentvec`'s `ambulkdelete` (HNSW dispatch:
    /// the embedded hnswsq region vacuum) with a synthetic callback, so the
    /// assertion does not depend on PostgreSQL's vacuum cutoff.
    #[pg_test]
    fn test_agentvec_tombstoned_entries_are_skipped() -> spi::Result<()> {
        Spi::run(
            "CREATE TABLE t_av_tomb(id int, v vector(2));
             INSERT INTO t_av_tomb VALUES (1, '[1,0]'), (2, '[2,0]'), (3, '[3,0]');
             CREATE INDEX idx_av_tomb ON t_av_tomb USING agentvec (v vector_l2_ops)
                WITH (search_candidates = 10);",
        )?;
        force_index_scan();
        assert_eq!(
            search_ids("t_av_tomb", "v", "<->", "[0,0]", 2),
            vec![1, 2],
            "the bound keeps the two nearest entries"
        );

        // Kill the heap row with id = 1 through the AM's own bulkdelete.
        let ctid =
            Spi::get_one::<String>("SELECT ctid::text FROM t_av_tomb WHERE id = 1")?.expect("ctid");
        let (block, offset) = parse_ctid(&ctid);

        struct KillOne {
            block: u32,
            offset: u16,
        }
        unsafe extern "C-unwind" fn kill_one(
            tid: *mut pg_sys::ItemPointerData,
            state: *mut std::os::raw::c_void,
        ) -> bool {
            let target = &*(state as *const KillOne);
            pgrx::itemptr::item_pointer_get_block_number(tid) == target.block
                && pgrx::itemptr::item_pointer_get_offset_number(tid) == target.offset
        }

        let index = index_relation("idx_av_tomb");
        let mut kill = KillOne { block, offset };
        let mut info = pg_sys::IndexVacuumInfo::default();
        info.index = index.as_ptr();
        let results = unsafe {
            crate::access_method::agentvec::vacuum::ambulkdelete(
                &mut info,
                std::ptr::null_mut(),
                Some(kill_one),
                &mut kill as *mut KillOne as *mut std::os::raw::c_void,
            )
        };
        unsafe {
            assert_eq!((*results).tuples_removed, 1.0, "one dead entry reported");
        }

        assert_eq!(
            search_ids("t_av_tomb", "v", "<->", "[0,0]", 2),
            vec![2, 3],
            "a tombstoned entry must not consume a candidate slot"
        );
        let dead: i64 = Spi::get_one::<i64>(
            "SELECT sum(dead_entries)::bigint FROM agentvec_index_info('idx_av_tomb')",
        )?
        .expect("dead count");
        assert_eq!(dead, 1);
        Ok(())
    }

    /// Parse PostgreSQL's `(block,offset)` ctid text.
    fn parse_ctid(ctid: &str) -> (u32, u16) {
        let inner = ctid.trim_start_matches('(').trim_end_matches(')');
        let mut parts = inner.split(',');
        let block = parts.next().expect("ctid block").parse().expect("block");
        let offset = parts.next().expect("ctid offset").parse().expect("offset");
        (block, offset)
    }

    /// VACUUM runs against the index and leaves its answers correct.
    ///
    /// VACUUM cannot run inside the SPI test transaction, so this test brings
    /// up the test instance and drives everything through a second connection
    /// (the pattern `access_method::vacuum` uses for the diskann AM).  The
    /// assertions are deliberately independent of PostgreSQL's vacuum cutoff;
    /// the tombstone semantics themselves are covered by
    /// `test_agentvec_tombstoned_entries_are_skipped`.
    #[cfg(test)]
    #[test]
    fn test_agentvec_vacuum_keeps_results_correct() {
        // This test owns its own connection and DDL, so it must not overlap
        // with itself (pg_test's rollback protection does not apply to it).
        static VAC_MUTEX: once_cell::sync::Lazy<std::sync::Mutex<()>> =
            once_cell::sync::Lazy::new(std::sync::Mutex::default);
        let _lock = VAC_MUTEX
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());

        pgrx_tests::run_test(
            "agentvec_vacuum_mock_fn",
            None,
            crate::pg_test::postgresql_conf_options(),
        )
        .unwrap();

        let (mut client, _) = pgrx_tests::client().unwrap();
        client
            .batch_execute(
                "CREATE TABLE t_av_vac(id int, v vector(2));
                 INSERT INTO t_av_vac VALUES (1, '[1,0]'), (2, '[2,0]'), (3, '[3,0]');
                 CREATE INDEX idx_av_vac ON t_av_vac USING agentvec (v vector_l2_ops);",
            )
            .unwrap();
        client.execute("SET enable_seqscan = off", &[]).unwrap();

        let query = "SELECT id FROM t_av_vac ORDER BY v <-> '[0,0]' LIMIT 3";
        let ids: Vec<i32> = client
            .query(query, &[])
            .unwrap()
            .iter()
            .map(|row| row.get::<_, i32>(0))
            .collect();
        assert_eq!(ids, vec![1, 2, 3]);

        client
            .execute("DELETE FROM t_av_vac WHERE id = 1", &[])
            .unwrap();
        let ids: Vec<i32> = client
            .query(query, &[])
            .unwrap()
            .iter()
            .map(|row| row.get::<_, i32>(0))
            .collect();
        assert_eq!(ids, vec![2, 3], "a deleted row must not be returned");

        // VACUUM cannot run inside a transaction block, so it needs its own
        // statement (batch_execute would wrap it with the DELETE).
        client.execute("VACUUM t_av_vac", &[]).unwrap();

        let ids: Vec<i32> = client
            .query(query, &[])
            .unwrap()
            .iter()
            .map(|row| row.get::<_, i32>(0))
            .collect();
        assert_eq!(ids, vec![2, 3], "results are unchanged by vacuum");

        client.batch_execute("DROP TABLE t_av_vac").unwrap();
    }

    /// Only a mock: it brings up the test instance that the connection-based
    /// test above drives.
    #[pg_test]
    fn agentvec_vacuum_mock_fn() -> spi::Result<()> {
        Ok(())
    }

    /// Deterministic clustered rows + one query per cluster center.
    fn gen_clustered(
        n_clusters: usize,
        per_cluster: usize,
        dim: usize,
        noise: f32,
        seed: u64,
    ) -> (Vec<Vec<f32>>, Vec<Vec<f32>>) {
        let mut rng = SmallRng::seed_from_u64(seed);
        let centers: Vec<Vec<f32>> = (0..n_clusters)
            .map(|_| (0..dim).map(|_| rng.gen::<f32>() * 2.0 - 1.0).collect())
            .collect();
        let mut rows = Vec::with_capacity(n_clusters * per_cluster);
        for c in &centers {
            for _ in 0..per_cluster {
                rows.push(
                    c.iter()
                        .map(|x| x + (rng.gen::<f32>() - 0.5) * noise)
                        .collect(),
                );
            }
        }
        let queries: Vec<Vec<f32>> = centers
            .iter()
            .map(|c| {
                c.iter()
                    .map(|x| x + (rng.gen::<f32>() - 0.5) * noise * 0.2)
                    .collect()
            })
            .collect();
        (rows, queries)
    }

    fn vec_literal(v: &[f32]) -> String {
        let inner = v
            .iter()
            .map(|x| x.to_string())
            .collect::<Vec<_>>()
            .join(",");
        format!("[{inner}]")
    }

    /// Exact top-k ids for a query against an in-memory row set (the ground
    /// truth the approximate segments are measured against).
    fn exact_topk(rows: &[Vec<f32>], q: &[f32], k: usize) -> Vec<usize> {
        let mut scored: Vec<(f32, usize)> = rows
            .iter()
            .enumerate()
            .map(|(i, v)| {
                let d: f32 = v.iter().zip(q).map(|(a, b)| (a - b) * (a - b)).sum();
                (d, i)
            })
            .collect();
        scored.sort_by(|a, b| a.0.total_cmp(&b.0));
        scored.into_iter().take(k).map(|(_, i)| i + 1).collect()
    }

    /// Whole-segment conversion: the oldest sealed HOT segment becomes one
    /// immutable IVF-RaBitQ segment, the source is retired, no row is lost,
    /// and searches span HOT + WARM in one query with exact results (the
    /// executor recheck restores ordering of the returned candidates).
    #[pg_test]
    fn test_agentvec_consolidate_lifecycle_and_row_identity() -> spi::Result<()> {
        Spi::run(
            "CREATE TABLE t_av_conv(id int, v vector(8));
             CREATE INDEX idx_av_conv ON t_av_conv USING agentvec (v vector_l2_ops)
                WITH (hot_segment_max_rows = 10);
             INSERT INTO t_av_conv
                SELECT g, ('[' || g || ',0,0,0,0,0,0,0]')::vector FROM generate_series(1, 30) AS g;",
        )?;
        force_index_scan();

        // Bulk loads build IVF segments directly; the HNSW HOT path belongs
        // to inserts, which seal at hot_segment_max_rows (two 10-row sealed
        // segments here).  Converting the oldest sealed segment...
        let converted =
            Spi::get_one::<i64>("SELECT agentvec_consolidate('idx_av_conv')")?.expect("converted");
        assert_eq!(converted, 10, "the oldest sealed segment holds 10 rows");

        let warm = Spi::get_one::<i64>(
            "SELECT count(*) FROM agentvec_index_info('idx_av_conv') WHERE algorithm = 'ivf_rabitq'",
        )?
        .expect("warm count");
        let retired = Spi::get_one::<i64>(
            "SELECT count(*) FROM agentvec_index_info('idx_av_conv') WHERE state = 'retired'",
        )?
        .expect("retired count");
        let total = Spi::get_one::<i64>(
            "SELECT sum(num_entries)::bigint FROM agentvec_index_info('idx_av_conv')",
        )?
        .expect("total count");
        // 1 warm segment, 1 retired segment. `num_entries` counts physical
        // rows: the retired HOT segment keeps its 10 entries until phase-10
        // reclamation, so the index holds 30 HOT + 10 WARM copies. Retired
        // segments are not searched, so no duplicate TIDs reach the scan.
        assert_eq!(
            (warm, retired, total),
            (1, 1, 40),
            "1 warm, 1 retired, no row lost"
        );

        // The converted segment is immutable and searchable.
        assert_eq!(
            search_ids("t_av_conv", "v", "<->", "[0,0,0,0,0,0,0,0]", 3),
            vec![1, 2, 3],
            "HOT + WARM in one query, exact order"
        );
        Ok(())
    }

    /// Recall of the converted IVF-RaBitQ segments against exact ground truth.
    #[pg_test]
    fn test_agentvec_consolidate_recall_vs_exact() -> spi::Result<()> {
        let (rows, queries) = gen_clustered(4, 60, 16, 0.6, 424242);
        let values = rows
            .iter()
            .map(|v| format!("('{}')", vec_literal(v)))
            .collect::<Vec<_>>()
            .join(",");
        Spi::run(&format!(
            "CREATE TABLE t_av_recall(id serial primary key, v vector(16));
             CREATE INDEX idx_av_recall ON t_av_recall USING agentvec (v vector_l2_ops)
                WITH (hot_segment_max_rows = 60, ivf_lists = 8, ivf_probes = 8);
             INSERT INTO t_av_recall(v) VALUES {values};"
        ))?;
        force_index_scan();

        // 240 rows at 60/segment: three sealed HOT segments, each converted.
        for _ in 0..3 {
            let n = Spi::get_one::<i64>("SELECT agentvec_consolidate('idx_av_recall')")?
                .expect("converted");
            assert_eq!(n, 60);
        }

        let mut hits = 0usize;
        let mut total = 0usize;
        for q in &queries {
            let exact = exact_topk(&rows, q, 10);
            let got = search_ids("t_av_recall", "v", "<->", &vec_literal(q), 10);
            total += 10;
            hits += got
                .iter()
                .filter(|id| exact.contains(&(**id as usize)))
                .count();
        }
        let recall = hits as f64 / total as f64;
        assert!(
            recall >= 0.9,
            "recall@10 of the converted segments must be >= 0.9, got {recall}"
        );
        Ok(())
    }

    /// The router activates only a subset of the owned WARM segments:
    /// `router::route` returns the top `router_top_m` segment ids, drawn
    /// exclusively from owned IvfRaBitQ segments, and its size is capped.
    #[pg_test]
    fn test_agentvec_router_returns_segment_subset() -> spi::Result<()> {
        Spi::run(
            "CREATE TABLE t_av_route(id int, v vector(8));
             CREATE INDEX idx_av_route ON t_av_route USING agentvec (v vector_l2_ops)
                WITH (hot_segment_max_rows = 40, router_top_m = 2);
             INSERT INTO t_av_route
                SELECT g, ('[' || g || ',0,0,0,0,0,0,0]')::vector FROM generate_series(1, 120) AS g;",
        )?;

        // Before any conversion there is no router region: the router must
        // fall back to "search everything".
        {
            let index = index_relation("idx_av_route");
            let meta = AgentVecMetaPage::fetch(&index);
            assert_eq!(meta.get_router_base(), 0, "no router region yet");
            let directory = meta.load_directory(&index);
            let options = TSVAgentVecOptions::from_relation(&index);
            let decided = unsafe {
                router::route(
                    &index,
                    &[0.0; 8],
                    &directory,
                    &options,
                    meta.get_router_base(),
                )
            };
            assert!(decided.is_none(), "no region means no decision");
        }

        for _ in 0..2 {
            assert_eq!(
                Spi::get_one::<i64>("SELECT agentvec_consolidate('idx_av_route')")?.expect("n"),
                40
            );
        }

        let index = index_relation("idx_av_route");
        let meta = AgentVecMetaPage::fetch(&index);
        let router_base = meta.get_router_base();
        assert_ne!(router_base, 0, "the router region exists after conversion");
        let directory = meta.load_directory(&index);
        let options = TSVAgentVecOptions::from_relation(&index);
        let warm: Vec<u64> = directory
            .segments
            .iter()
            .filter(|s| {
                s.algorithm()
                    == crate::access_method::agentvec::directory::SegmentAlgorithm::IvfRaBitQ
                    && s.ownership() == SegmentOwnership::Owned
            })
            .map(|s| s.segment_id)
            .collect();
        assert_eq!(warm.len(), 2, "two converted segments");

        let decided =
            unsafe { router::route(&index, &[0.0; 8], &directory, &options, router_base) }
                .expect("the router must decide once it has nodes");
        assert_eq!(decided.len(), 2, "top_m = 2 activates at most 2");
        assert!(
            decided.iter().all(|id| warm.contains(id)),
            "only owned WARM segments may be activated, got {decided:?}"
        );

        // The routed query still returns exact top-N (2 of 3 segments stay
        // searchable: 1 HOT + 2 WARM, all activated).
        force_index_scan();
        assert_eq!(
            search_ids("t_av_route", "v", "<->", "[0,0,0,0,0,0,0,0]", 3),
            vec![1, 2, 3]
        );
        Ok(())
    }

    /// Recall of routed queries: with more WARM segments than `router_top_m`,
    /// the router prunes segment searches and the results stay good.
    #[pg_test]
    fn test_agentvec_router_multi_segment_recall() -> spi::Result<()> {
        let (rows, queries) = gen_clustered(6, 60, 16, 0.6, 131313);
        let values = rows
            .iter()
            .map(|v| format!("('{}')", vec_literal(v)))
            .collect::<Vec<_>>()
            .join(",");
        Spi::run(&format!(
            "CREATE TABLE t_av_rr(id serial primary key, v vector(16));
             CREATE INDEX idx_av_rr ON t_av_rr USING agentvec (v vector_l2_ops)
                WITH (hot_segment_max_rows = 60, ivf_lists = 8, ivf_probes = 8,
                      router_top_m = 3);
             INSERT INTO t_av_rr(v) VALUES {values};"
        ))?;
        force_index_scan();

        // Insert-driven segmentation: five sealed 60-row segments (the
        // sixth stays HOT), each converted.
        for _ in 0..5 {
            assert_eq!(
                Spi::get_one::<i64>("SELECT agentvec_consolidate('idx_av_rr')")?.expect("n"),
                60
            );
        }

        let mut hits = 0usize;
        let mut total = 0usize;
        for q in &queries {
            let exact = exact_topk(&rows, q, 10);
            let got = search_ids("t_av_rr", "v", "<->", &vec_literal(q), 10);
            total += 10;
            hits += got
                .iter()
                .filter(|id| exact.contains(&(**id as usize)))
                .count();
        }
        let recall = hits as f64 / total as f64;
        assert!(
            recall >= 0.85,
            "recall@10 with 3 of 5 WARM segments routed must be >= 0.85, got {recall}"
        );
        Ok(())
    }

    /// The synchronous maintenance entry point converts every sealed
    /// segment of an index, bounded by the migration budget.
    #[pg_test]
    fn test_agentvec_run_maintenance_converts_all_sealed() -> spi::Result<()> {
        Spi::run(
            "CREATE TABLE t_av_maint(id int, v vector(8));
             CREATE INDEX idx_av_maint ON t_av_maint USING agentvec (v vector_l2_ops)
                WITH (hot_segment_max_rows = 30);
             INSERT INTO t_av_maint
                SELECT g, ('[' || g || ',0,0,0,0,0,0,0]')::vector FROM generate_series(1, 90) AS g;",
        )?;
        force_index_scan();

        let converted =
            Spi::get_one::<i64>("SELECT agentvec_run_maintenance('idx_av_maint')")?.expect("n");
        assert_eq!(converted, 60, "two sealed segments of 30 rows");
        let warm: i64 = Spi::get_one::<i64>(
            "SELECT count(*) FROM agentvec_index_info('idx_av_maint') WHERE algorithm = 'ivf_rabitq'",
        )?
        .expect("count");
        assert_eq!(warm, 2, "both sealed segments converted");

        // Idempotent: nothing left to convert.
        let again =
            Spi::get_one::<i64>("SELECT agentvec_run_maintenance('idx_av_maint')")?.expect("n");
        assert_eq!(again, 0);

        assert_eq!(
            search_ids("t_av_maint", "v", "<->", "[0,0,0,0,0,0,0,0]", 3),
            vec![1, 2, 3]
        );
        Ok(())
    }

    /// The per-index maintenance schedule gate: `agentvec_maybe_maintain`
    /// converts only when the index's `maintenance_interval` has elapsed
    /// since the last pass (fresh indexes start as just-maintained).  The
    /// worker itself calls this same gate; it is exercised end-to-end in the
    /// scratch instance because the pg_test harness runs every test in a
    /// transaction that never commits, so a separate worker backend cannot
    /// see the test's index.
    #[pg_test]
    fn test_agentvec_maybe_maintain_schedule() -> spi::Result<()> {
        Spi::run(
            "CREATE TABLE t_av_sched(id int, v vector(8));
             CREATE INDEX idx_av_sched ON t_av_sched USING agentvec (v vector_l2_ops)
                WITH (hot_segment_max_rows = 30);
             INSERT INTO t_av_sched
                SELECT g, ('[' || g || ',0,0,0,0,0,0,0]')::vector FROM generate_series(1, 60) AS g;",
        )?;

        // Fresh index: not due yet (default 60 s interval).
        assert_eq!(
            Spi::get_one::<i64>("SELECT agentvec_maybe_maintain('idx_av_sched')")?.expect("n"),
            0,
            "a fresh index must not be maintained before its interval"
        );
        Spi::run("SELECT pg_sleep(0.2)")?;
        assert_eq!(
            Spi::get_one::<i64>("SELECT agentvec_maybe_maintain('idx_av_sched')")?.expect("n"),
            0,
            "still not due"
        );

        // Force maintenance (marks the index maintained and converts the
        // sealed 30-row segment).
        assert_eq!(
            Spi::get_one::<i64>("SELECT agentvec_run_maintenance('idx_av_sched')")?.expect("n"),
            30
        );
        // 31 more rows: the active HOT segment seals on the 31st insert.
        Spi::run(
            "INSERT INTO t_av_sched
                SELECT g, ('[' || g || ',0,0,0,0,0,0,0]')::vector FROM generate_series(61, 91) AS g;",
        )?;
        assert_eq!(
            Spi::get_one::<i64>("SELECT agentvec_maybe_maintain('idx_av_sched')")?.expect("n"),
            0,
            "not due again after the forced pass"
        );

        // An interval of 0 means "always due"; the 31-row insert sealed two
        // 30-row segments, so the batch converts both (60).
        Spi::run("ALTER INDEX idx_av_sched SET (maintenance_interval = 0)")?;
        assert_eq!(
            Spi::get_one::<i64>("SELECT agentvec_maybe_maintain('idx_av_sched')")?.expect("n"),
            60,
            "due immediately with interval 0"
        );
        Ok(())
    }

    /// A failing index must not take the maintenance pass (or the worker
    /// process) down with it.  `maintain_one_index` runs the index's step in
    /// its own subtransaction: the error is caught, logged and rolled back, the
    /// enclosing transaction stays usable, and the pass goes on to the next
    /// index.  (Before this was isolated, the raw `Spi::get_one` raised: the
    /// whole pass — and, in the worker, the backend — was lost.)
    #[pg_test]
    fn test_agentvec_maintenance_isolates_failing_index() -> spi::Result<()> {
        use crate::access_method::agentvec::maintenance;

        Spi::run(
            "CREATE TABLE t_av_iso(id int, v vector(8));
             CREATE INDEX idx_av_iso ON t_av_iso USING agentvec (v vector_l2_ops)
                WITH (hot_segment_max_rows = 30, maintenance_interval = 0);
             INSERT INTO t_av_iso
                SELECT g, ('[' || g || ',0,0,0,0,0,0,0]')::vector FROM generate_series(1, 60) AS g;",
        )?;

        // A relation that does not exist: the step fails inside its
        // subtransaction (the `regclass` cast inside the SPI call raises).
        force_index_scan();
        let missing = Spi::get_one::<pg_sys::Oid>("SELECT 987654321::oid")?.expect("oid");
        assert_eq!(
            maintenance::maintain_one_index(missing),
            None,
            "a failing index must be reported as failed, not raised"
        );

        // ... and the pass survived it: the next index runs on the same
        // transaction and converts the segment the insert above sealed.
        let oid = Spi::get_one::<pg_sys::Oid>("SELECT 'idx_av_iso'::regclass::oid")?.expect("oid");
        assert_eq!(
            maintenance::maintain_one_index(oid),
            Some(30),
            "the pass continues after a failing index"
        );

        // The transaction is still usable (a raised error would have left it
        // aborted, failing everything from here on).
        assert_eq!(Spi::get_one::<i32>("SELECT 1")?, Some(1));
        assert_eq!(
            search_ids("t_av_iso", "v", "<->", "[0,0,0,0,0,0,0,0]", 3),
            vec![1, 2, 3],
            "the index converted after the failure is searchable"
        );
        Ok(())
    }

    /// A sealed segment that spans several entry blocks: the burst reader
    /// compacts one payload per page into the SoA stream, so a search must see
    /// exactly the entries that were written (a single-block segment cannot
    /// catch a compaction that only handles the first page).
    #[pg_test]
    fn test_agentvec_multiblock_segment_search() -> spi::Result<()> {
        Spi::run(
            "CREATE TABLE t_av_mb(id int, v vector(8));
             INSERT INTO t_av_mb
                SELECT g, ('[' || (g % 7) || ',' || (g % 5) || ',' || (g % 3) || ',0,0,0,0,0]')::vector
                  FROM generate_series(1, 4000) AS g;
             CREATE INDEX idx_av_mb ON t_av_mb USING agentvec (v vector_l2_ops)
                WITH (hot_segment_max_rows = 4000);",
        )?;
        force_index_scan();
        // Reference order without the index.
        Spi::run("SET enable_seqscan = on")?;
        let exact = search_ids("t_av_mb", "v", "<->", "[1,1,1,0,0,0,0,0]", 5);
        force_index_scan();
        let approx = search_ids("t_av_mb", "v", "<->", "[1,1,1,0,0,0,0,0]", 5);
        assert_eq!(approx.len(), exact.len());
        for id in &approx {
            assert!(exact.contains(id), "id {id} not in the exact top-5");
        }
        Ok(())
    }

    /// With nothing sealed, the conversion is a no-op.
    #[pg_test]
    fn test_agentvec_consolidate_noop() -> spi::Result<()> {
        Spi::run(
            "CREATE TABLE t_av_noop(id int, v vector(2));
             INSERT INTO t_av_noop VALUES (1, '[1,0]'), (2, '[2,0]');
             CREATE INDEX idx_av_noop ON t_av_noop USING agentvec (v vector_l2_ops)
                WITH (hot_segment_max_rows = 100);",
        )?;
        let converted =
            Spi::get_one::<i64>("SELECT agentvec_consolidate('idx_av_noop')")?.expect("n");
        assert_eq!(converted, 0, "nothing sealed, nothing converted");
        let warm: i64 = Spi::get_one::<i64>(
            "SELECT count(*) FROM agentvec_index_info('idx_av_noop') WHERE algorithm = 'ivf_rabitq'",
        )?
        .expect("count");
        assert_eq!(warm, 0);
        Ok(())
    }

    /// A segment left `Retiring` by a crashed claim self-heals: the next
    /// conversion call picks it up.
    #[pg_test]
    fn test_agentvec_consolidate_selfheals_retiring() -> spi::Result<()> {
        Spi::run(
            "CREATE TABLE t_av_heal(id int, v vector(8));
             CREATE INDEX idx_av_heal ON t_av_heal USING agentvec (v vector_l2_ops)
                WITH (hot_segment_max_rows = 10);
             INSERT INTO t_av_heal
                SELECT g, ('[' || g || ',0,0,0,0,0,0,0]')::vector FROM generate_series(1, 25) AS g;",
        )?;
        // Flip the oldest sealed segment to Retiring directly (simulating a
        // claim whose build never published).
        let index = index_relation("idx_av_heal");
        let meta = AgentVecMetaPage::fetch(&index);
        let directory = meta.load_directory(&index);
        let victim = directory
            .segments
            .iter()
            .find(|s| s.state() == SegmentState::QueuedForMigration)
            .expect("a sealed segment")
            .segment_id;
        unsafe {
            AgentVecMetaPage::update(&index, |m| {
                let mut d = m.load_directory(&index);
                let seg = d.get_mut(victim).expect("victim");
                seg.state = SegmentState::Retiring as u8;
                let (ptr, blocks) = d.store(&index);
                m.set_directory(ptr, blocks);
            });
        }

        let converted =
            Spi::get_one::<i64>("SELECT agentvec_consolidate('idx_av_heal')")?.expect("n");
        assert_eq!(
            converted, 10,
            "the Retiring sealed segment is reclaimed and converted"
        );
        let warm: i64 = Spi::get_one::<i64>(
            "SELECT count(*) FROM agentvec_index_info('idx_av_heal') WHERE algorithm = 'ivf_rabitq'",
        )?
        .expect("count");
        assert_eq!(warm, 1);
        Ok(())
    }

    /// Reaching the HOT threshold seals the segment and opens a new one; the
    /// sealed segment stays searchable, so results span all segments in order.
    #[pg_test]
    fn test_agentvec_hot_seal_and_multi_segment_search() -> spi::Result<()> {
        Spi::run(
            "CREATE TABLE t_av_seal(id int, v vector(2));
             CREATE INDEX idx_av_seal ON t_av_seal USING agentvec (v vector_l2_ops)
                WITH (hot_segment_max_rows = 2);
             INSERT INTO t_av_seal
                SELECT g, ('[' || g || ',0]')::vector FROM generate_series(1, 7) AS g;",
        )?;
        force_index_scan();

        // 7 rows at 2 rows per HOT segment: the last insert observes the
        // threshold, so segments are created at rows 3, 5 and 7.
        let index = index_relation("idx_av_seal");
        let meta = AgentVecMetaPage::fetch(&index);
        let directory: AgentVecDirectory = meta.load_directory(&index);
        assert_eq!(
            directory.segments.len(),
            4,
            "expected four HOT segments, got {:?}",
            directory
                .segments
                .iter()
                .map(|s| (s.segment_id, s.state, s.vector_count))
                .collect::<Vec<_>>()
        );
        for (i, segment) in directory.segments.iter().enumerate() {
            let header = AgentVecSegmentHeader::load(&index, segment.header);
            let expected_state = if segment.segment_id == meta.get_hot_segment_id() {
                SegmentState::Published
            } else {
                SegmentState::QueuedForMigration
            };
            assert_eq!(segment.state(), expected_state, "segment {i} state");
            if segment.segment_id != meta.get_hot_segment_id() {
                // HNSW segments have no FLAT chains: the seal flips the state
                // only; the row count is the lifecycle record.
                assert_eq!(
                    header.num_entries, 2,
                    "a sealed segment must have exactly two entries"
                );
            }
        }

        // The same structure is visible from SQL, which is how a user (or a
        // later phase's maintenance worker) inspects the directory.
        let hot_segments = Spi::get_one::<i64>(
            "SELECT count(*) FROM agentvec_index_info('idx_av_seal') WHERE level = 'hot'",
        )?
        .expect("count");
        assert_eq!(hot_segments, 4);
        let queued: i64 = Spi::get_one::<i64>(
            "SELECT count(*) FROM agentvec_index_info('idx_av_seal') WHERE state = 'queued_for_migration'",
        )?
        .expect("count");
        assert_eq!(queued, 3, "every sealed segment awaits maintenance");
        let indexed: i64 = Spi::get_one::<i64>(
            "SELECT sum(num_entries)::bigint FROM agentvec_index_info('idx_av_seal')",
        )?
        .expect("sum");
        assert_eq!(indexed, 7, "every inserted row is accounted for");

        // Distance from [0,0] is |g|, so exact order is 1..7 across segments.
        assert_eq!(
            search_ids("t_av_seal", "v", "<->", "[0,0]", 7),
            vec![1, 2, 3, 4, 5, 6, 7],
            "search must merge every segment in distance order"
        );
        Ok(())
    }

    /// The number of appendable HOT segments in a directory.
    fn appendable_hot_count(directory: &AgentVecDirectory) -> usize {
        directory
            .segments
            .iter()
            .filter(|s| {
                s.level() == crate::access_method::agentvec::directory::SegmentLevel::Hot
                    && s.state() == SegmentState::Published
            })
            .count()
    }

    /// `hot_segment_count > 1` keeps several HOT segments appendable at once.
    ///
    /// Each member is an independent embedded hnswsq region, but the
    /// transaction contract is unchanged: every entry is written into a
    /// published segment inside the inserting transaction, and a scan merges
    /// all of them, so every committed row is immediately findable.
    #[pg_test]
    fn test_agentvec_multi_hot_pool() -> spi::Result<()> {
        Spi::run(
            "CREATE TABLE t_av_pool(id int, v vector(2));
             CREATE INDEX idx_av_pool ON t_av_pool USING agentvec (v vector_l2_ops)
                WITH (hot_segment_count = 4);
             INSERT INTO t_av_pool
                SELECT g, ('[' || g || ',0]')::vector FROM generate_series(1, 32) AS g;",
        )?;
        force_index_scan();

        let index = index_relation("idx_av_pool");
        let meta = AgentVecMetaPage::fetch(&index);
        let directory: AgentVecDirectory = meta.load_directory(&index);
        assert_eq!(
            appendable_hot_count(&directory),
            4,
            "the pool should publish hot_segment_count appendable segments, got {:?}",
            directory
                .segments
                .iter()
                .map(|s| (s.segment_id, s.level, s.state, s.algorithm))
                .collect::<Vec<_>>()
        );

        // Every member received rows: the writer spreads by cached load rather
        // than always taking the first segment.
        for segment in directory
            .segments
            .iter()
            .filter(|s| s.state() == SegmentState::Published)
        {
            let header = AgentVecSegmentHeader::load(&index, segment.header);
            assert!(
                header.num_entries > 0,
                "segment {} received no rows",
                segment.segment_id
            );
        }

        // Every row is still found: the scan must merge all four graphs.
        let found: i64 = Spi::get_one::<i64>("SELECT count(*) FROM t_av_pool")?.expect("count");
        assert_eq!(found, 32);
        assert_eq!(
            search_ids("t_av_pool", "v", "<->", "[16,0]", 32).len(),
            32,
            "a scan must return rows from every member of the HOT pool"
        );
        Ok(())
    }

    /// Sealing one member of a multi-segment pool leaves the pool at its
    /// configured size and keeps every committed row searchable — the property
    /// that lets the write side continue while a segment is being handed to
    /// maintenance.
    #[pg_test]
    fn test_agentvec_multi_hot_seal_keeps_writers_going() -> spi::Result<()> {
        Spi::run(
            "CREATE TABLE t_av_pool2(id int, v vector(2));
             CREATE INDEX idx_av_pool2 ON t_av_pool2 USING agentvec (v vector_l2_ops)
                WITH (hot_segment_count = 2, hot_segment_max_rows = 3);
             INSERT INTO t_av_pool2
                SELECT g, ('[' || g || ',0]')::vector FROM generate_series(1, 40) AS g;",
        )?;
        force_index_scan();

        let index = index_relation("idx_av_pool2");
        let meta = AgentVecMetaPage::fetch(&index);
        let directory: AgentVecDirectory = meta.load_directory(&index);

        // 40 rows at 3 rows per segment over a 2-segment pool: every seal hands
        // its writers to a sibling and immediately opens a replacement, so the
        // appendable pool is still exactly two members.
        assert_eq!(
            appendable_hot_count(&directory),
            2,
            "a seal must be followed by a replacement, got {:?}",
            directory
                .segments
                .iter()
                .map(|s| (s.segment_id, s.level, s.state))
                .collect::<Vec<_>>()
        );
        let queued: i64 = Spi::get_one::<i64>(
            "SELECT count(*) FROM agentvec_index_info('idx_av_pool2')
              WHERE state = 'queued_for_migration'",
        )?
        .expect("count");
        assert!(queued >= 1, "expected sealed segments, got {queued}");

        // No committed row may be lost across the seals.
        let indexed: i64 = Spi::get_one::<i64>(
            "SELECT sum(num_entries)::bigint FROM agentvec_index_info('idx_av_pool2')",
        )?
        .expect("sum");
        assert_eq!(indexed, 40, "every inserted row is accounted for");
        assert_eq!(
            search_ids("t_av_pool2", "v", "<->", "[0,0]", 40),
            (1..=40).collect::<Vec<i32>>(),
            "search must merge every segment, sealed ones included"
        );
        Ok(())
    }

    /// Cosine and inner-product operator classes, including the cosine
    /// normalization the stored representation relies on.
    #[pg_test]
    fn test_agentvec_cosine_and_inner_product_ops() -> spi::Result<()> {
        Spi::run(
            "CREATE TABLE t_av_cos(id int, v vector(2));
             INSERT INTO t_av_cos VALUES
                (1, '[1,0]'), (2, '[1,1]'), (3, '[0,1]'), (4, '[-1,0]');
             CREATE INDEX idx_av_cos ON t_av_cos USING agentvec (v vector_cosine_ops);",
        )?;
        force_index_scan();
        assert_eq!(
            search_ids("t_av_cos", "v", "<=>", "[1,0]", 4),
            vec![1, 2, 3, 4],
            "cosine order must ignore magnitude"
        );

        Spi::run(
            "CREATE TABLE t_av_ip(id int, v vector(2));
             INSERT INTO t_av_ip VALUES (1, '[1,0]'), (2, '[2,0]'), (3, '[3,0]');
             CREATE INDEX idx_av_ip ON t_av_ip USING agentvec (v vector_ip_ops);",
        )?;
        force_index_scan();
        assert_eq!(
            search_ids("t_av_ip", "v", "<#>", "[1,0]", 3),
            vec![3, 2, 1],
            "inner product orders by descending dot product"
        );
        Ok(())
    }

    /// `search_candidates` bounds the candidate stream (0 = exhaustive).
    #[pg_test]
    fn test_agentvec_search_candidates_bound() -> spi::Result<()> {
        Spi::run(
            "CREATE TABLE t_av_bound(id int, v vector(2));
             INSERT INTO t_av_bound
                SELECT g, ('[' || g || ',0]')::vector FROM generate_series(1, 10) AS g;
             CREATE INDEX idx_av_bound ON t_av_bound USING agentvec (v vector_l2_ops)
                WITH (search_candidates = 3);",
        )?;
        force_index_scan();

        let ids = search_ids("t_av_bound", "v", "<->", "[0,0]", 10);
        assert_eq!(
            ids.len(),
            3,
            "a bounded scan returns at most `search_candidates` rows"
        );
        assert!(ids.iter().all(|id| (1..=10).contains(id)));
        Ok(())
    }

    /// REINDEX rebuilds from the heap and produces the same answers.
    #[pg_test]
    fn test_agentvec_reindex() -> spi::Result<()> {
        Spi::run(
            "CREATE TABLE t_av_reindex(id int, v vector(2));
             INSERT INTO t_av_reindex
                SELECT g, ('[' || g || ',0]')::vector FROM generate_series(1, 5) AS g;
             CREATE INDEX idx_av_reindex ON t_av_reindex USING agentvec (v vector_l2_ops)
                WITH (hot_segment_max_rows = 2);",
        )?;
        force_index_scan();
        let before = search_ids("t_av_reindex", "v", "<->", "[0,0]", 5);
        assert_eq!(before, vec![1, 2, 3, 4, 5]);

        Spi::run("REINDEX INDEX idx_av_reindex")?;
        force_index_scan();
        assert_eq!(
            search_ids("t_av_reindex", "v", "<->", "[0,0]", 5),
            before,
            "REINDEX must reproduce the same index contents"
        );
        Ok(())
    }

    /// The index answers ORDER BY queries only: a plan that would need an
    /// unordered answer must not pick it.
    #[pg_test]
    fn test_agentvec_planner_uses_index_only_for_order_by() -> spi::Result<()> {
        Spi::run(
            "CREATE TABLE t_av_plan(id int, v vector(2));
             INSERT INTO t_av_plan VALUES (1, '[1,0]'), (2, '[2,0]');
             CREATE INDEX idx_av_plan ON t_av_plan USING agentvec (v vector_l2_ops);",
        )?;
        force_index_scan();

        let ordered = plan_for("SELECT id FROM t_av_plan ORDER BY v <-> '[0,0]' LIMIT 1")?;
        assert!(
            ordered.contains("Index Scan") && ordered.contains("idx_av_plan"),
            "ORDER BY query must use the agentvec index, got:\n{ordered}"
        );

        // A count(*) cannot be answered by this AM at all (it has no ordered
        // candidate stream), so it must be planned without the index and must
        // return the true count -- including under `enable_seqscan = off`,
        // where the planner has to fall back to a disabled seq scan rather
        // than to an infinite-cost index path.
        let unordered = plan_for("SELECT count(*) FROM t_av_plan")?;
        assert!(
            !unordered.contains("idx_av_plan"),
            "a count(*) plan must not use the agentvec index, got:\n{unordered}"
        );
        let count = Spi::get_one::<i64>("SELECT count(*) FROM t_av_plan")?.expect("count");
        assert_eq!(count, 2, "count(*) must return the true row count");
        Ok(())
    }

    /// `EXPLAIN (COSTS OFF)` as a single string.
    fn plan_for(query: &str) -> spi::Result<String> {
        Spi::connect(|client| {
            let table = client.select(&format!("EXPLAIN (COSTS OFF) {query}"), None, &[])?;
            let mut plan = String::new();
            for row in table {
                if let Some(line) = row.get::<String>(1)? {
                    plan.push_str(&line);
                    plan.push('\n');
                }
            }
            Ok::<String, spi::Error>(plan)
        })
    }

    /// The on-disk state a fresh index starts from: meta at block 0 with a
    /// directory holding exactly one published, empty HOT segment.
    #[pg_test]
    fn test_agentvec_empty_index_layout() -> spi::Result<()> {
        Spi::run(
            "CREATE TABLE t_av_empty(id int, v vector(4));
             CREATE INDEX idx_av_empty ON t_av_empty USING agentvec (v vector_l2_ops);",
        )?;

        let index = index_relation("idx_av_empty");
        let meta = AgentVecMetaPage::fetch(&index);
        assert_eq!(meta.get_num_dimensions(), 4);
        assert_eq!(meta.get_num_tuples(), 0);

        let directory = meta.load_directory(&index);
        assert_eq!(directory.segments.len(), 1);
        let segment = &directory.segments[0];
        assert_eq!(segment.segment_id, meta.get_hot_segment_id());
        assert_eq!(segment.state(), SegmentState::Published);
        assert_eq!(segment.algorithm().as_str(), "hnsw");
        assert_eq!(segment.ownership().as_str(), "owned");
        assert!(
            segment.code_root.block_number > 0,
            "the embedded hnswsq region has a base block"
        );

        let header = AgentVecSegmentHeader::load(&index, segment.header);
        assert_eq!(header.num_entries, 0);
        assert!(header.sealed.is_empty());
        assert!(header.active.is_none());
        Ok(())
    }

    /// The Flat branch of `ambulkdelete`, with real entries and several chains.
    ///
    /// Flat segments are no longer produced (`open_new_hot_segment` always
    /// builds an HNSW region since 1a85350), so no index created today has one
    /// and the legacy sweep would otherwise be dead code that never runs in a
    /// test.  This test turns the empty HOT segment the build publishes into a
    /// Flat one, then lets the *product's own* insert path append to it, seals
    /// the active chain between batches to leave the segment with three chains,
    /// and drives the sweep directly with a synthetic dead-verdict callback
    /// (VACUUM's own verdict depends on the cutoff, which concurrent backends
    /// can pin — the same reason `hnswsq_bulkdelete_tombstones_direct` drives
    /// the AM directly).
    ///
    /// Four chains is what makes this a test of the sweep's per-chain buffers:
    /// if the tombstones or the entry list of one chain leaked into the next,
    /// `flat::mark_dead` would report an entry tombstoned twice, or the tombstones
    /// would land on the wrong entries — both of which the assertions below
    /// catch, against ids read back from the heap.
    #[pg_test]
    fn test_agentvec_flat_segment_bulkdelete_across_chains() -> spi::Result<()> {
        use crate::access_method::agentvec::directory::SegmentAlgorithm;
        use crate::access_method::agentvec::flat;
        use crate::access_method::agentvec::insert::SEGMENT_FORMAT_FLAT_V1;

        /// Parse PostgreSQL's `(block,offset)` ctid text.
        fn parse_ctid(text: &str) -> (u32, u16) {
            let inner = text.trim().trim_start_matches('(').trim_end_matches(')');
            let (block, offset) = inner.split_once(',').expect("ctid is (block,offset)");
            (
                block.trim().parse().expect("ctid block"),
                offset.trim().parse().expect("ctid offset"),
            )
        }

        struct KillSet(Vec<(u32, u16)>);
        unsafe extern "C-unwind" fn kill_set(
            tid: *mut pg_sys::ItemPointerData,
            state: *mut std::os::raw::c_void,
        ) -> bool {
            let targets = &*(state as *const KillSet);
            let here = (
                pgrx::itemptr::item_pointer_get_block_number(tid),
                pgrx::itemptr::item_pointer_get_offset_number(tid),
            );
            targets.0.iter().any(|&t| t == here)
        }

        force_index_scan();
        Spi::run(
            "CREATE TABLE t_av_flat(id serial primary key, embedding vector(128));
             CREATE INDEX idx_av_flat ON t_av_flat USING agentvec (embedding vector_l2_ops)
                 WITH (hot_segment_max_rows = 1000000);",
        )?;

        let index = index_relation("idx_av_flat");

        // Legacy-format flag: both the directory entry and the header carry it.
        let header_pointer = unsafe {
            let meta = AgentVecMetaPage::fetch(&index);
            let mut directory = meta.load_directory(&index);
            let hot_id = meta.get_hot_segment_id();
            let segment = directory.get_mut(hot_id).expect("the build's HOT segment");
            assert_eq!(
                segment.algorithm(),
                SegmentAlgorithm::Hnsw,
                "the build publishes an HNSW HOT segment"
            );
            let header_pointer = segment.header;
            segment.algorithm = SegmentAlgorithm::Flat as u8;
            segment.format_version = SEGMENT_FORMAT_FLAT_V1;
            let (directory_pointer, directory_blocks) = directory.store(&index);
            AgentVecMetaPage::update(&index, |meta| {
                meta.set_directory(directory_pointer, directory_blocks);
            });
            AgentVecSegmentHeader::update(&index, header_pointer.block_number, |header| {
                header.algorithm = SegmentAlgorithm::Flat as u8;
            });
            header_pointer
        };

        // Four batches, sealing the active chain in between: 2000 + 1500 +
        // 1000 + 500.  A dim-128 entry is 519 bytes, so each chain spans ~130
        // pages — long enough that a walk of one chain has real state to carry.
        let mut last_id = 0;
        for batch in [2000usize, 1500, 1000, 500] {
            let first = last_id + 1;
            last_id += batch;
            Spi::run(&format!(
                "INSERT INTO t_av_flat(embedding)
                 SELECT ('[' || g || repeat(',0', 127) || ']')::vector
                 FROM generate_series({first}, {last_id}) g"
            ))?;
            if batch != 500 {
                unsafe {
                    AgentVecSegmentHeader::update(&index, header_pointer.block_number, |header| {
                        assert!(
                            header.seal_active(),
                            "the inserts above must have opened a chain"
                        );
                    });
                }
            }
        }

        let header = unsafe { AgentVecSegmentHeader::load(&index, header_pointer) };
        let chains = header.chain_starts();
        assert_eq!(chains.len(), 4, "one chain per batch: {chains:?}");
        assert_eq!(header.num_entries, 5000);
        assert_eq!(header.dead_entries, 0);

        // Every entry, per chain, with the heap TID the insert path stored.
        let mut per_chain: Vec<usize> = Vec::new();
        for &chain in &chains {
            let mut n = 0;
            unsafe {
                flat::for_each_entry(&index, chain, |_block, _offset, _bytes| n += 1);
            }
            per_chain.push(n);
        }
        assert_eq!(
            per_chain,
            vec![2000, 1500, 1000, 500],
            "entries land in the chain they were appended to"
        );

        // The index is exact and complete before the sweep.
        let zero = format!(
            "[{}]",
            std::iter::repeat("0")
                .take(128)
                .collect::<Vec<_>>()
                .join(",")
        );
        let all_ids: Vec<i32> = (1..=5000).collect();
        assert_eq!(
            search_ids("t_av_flat", "embedding", "<->", &zero, 5000),
            all_ids,
            "all 5000 entries are live and searchable"
        );

        // Kill one row in each chain, by the ctid the heap reports.
        let kill_ids = [1000, 3000, 4200, 4800];
        let mut targets = Vec::new();
        for id in kill_ids {
            let ctid: String =
                Spi::get_one(&format!("SELECT ctid::text FROM t_av_flat WHERE id = {id}"))?
                    .expect("ctid");
            targets.push(parse_ctid(&ctid));
        }

        let mut kill = KillSet(targets.clone());
        let mut info = pg_sys::IndexVacuumInfo::default();
        info.index = index.as_ptr();
        let results = unsafe {
            crate::access_method::agentvec::vacuum::ambulkdelete(
                &mut info,
                std::ptr::null_mut(),
                Some(kill_set),
                &mut kill as *mut KillSet as *mut std::os::raw::c_void,
            )
        };
        unsafe {
            assert_eq!((*results).tuples_removed, 4.0, "four tombstones");
            assert_eq!((*results).num_index_tuples, 4996.0, "4996 live entries");
            pg_sys::pfree(results as *mut std::os::raw::c_void);
        }

        // Exactly the killed entries are dead — nothing from another chain, and
        // nothing tombstoned twice (which `mark_dead` would have raised on).
        let header = unsafe { AgentVecSegmentHeader::load(&index, header_pointer) };
        assert_eq!(
            header.num_entries, 5000,
            "the sweep does not remove entries"
        );
        assert_eq!(header.dead_entries, 4);
        let mut dead: Vec<(u32, u16)> = Vec::new();
        let mut live = 0usize;
        for &chain in &header.chain_starts() {
            unsafe {
                flat::for_each_entry(&index, chain, |_block, _offset, bytes| {
                    if flat::decode_state(bytes) == flat::STATE_DEAD {
                        let tid = flat::decode_tid(bytes);
                        dead.push((tid.block_number, tid.offset));
                    } else {
                        live += 1;
                    }
                });
            }
        }
        dead.sort_unstable();
        let mut expected_dead = targets.clone();
        expected_dead.sort_unstable();
        assert_eq!(dead, expected_dead, "the killed entries are the dead ones");
        assert_eq!(live, 4996);

        // … and the scan agrees with the heap.
        let expected_live: Vec<i32> = (1..=5000).filter(|id| !kill_ids.contains(id)).collect();
        assert_eq!(
            search_ids("t_av_flat", "embedding", "<->", &zero, 4996),
            expected_live,
            "tombstoned rows are gone from results, live rows are kept"
        );

        // A second sweep has nothing left to do.
        let mut kill = KillSet(targets);
        let results = unsafe {
            crate::access_method::agentvec::vacuum::ambulkdelete(
                &mut info,
                std::ptr::null_mut(),
                Some(kill_set),
                &mut kill as *mut KillSet as *mut std::os::raw::c_void,
            )
        };
        unsafe {
            assert_eq!(
                (*results).tuples_removed,
                0.0,
                "no entry is tombstoned twice"
            );
            assert_eq!((*results).num_index_tuples, 4996.0);
            pg_sys::pfree(results as *mut std::os::raw::c_void);
        }
        Ok(())
    }
}
