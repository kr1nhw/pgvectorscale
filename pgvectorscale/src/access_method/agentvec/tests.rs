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
            let sql = format!(
                "SELECT id FROM {table} ORDER BY {column} {op} '{query}' LIMIT {limit}"
            );
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
        let ctid = Spi::get_one::<String>("SELECT ctid::text FROM t_av_tomb WHERE id = 1")?
            .expect("ctid");
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
             INSERT INTO t_av_conv
                SELECT g, ('[' || g || ',0,0,0,0,0,0,0]')::vector FROM generate_series(1, 30) AS g;
             CREATE INDEX idx_av_conv ON t_av_conv USING agentvec (v vector_l2_ops)
                WITH (hot_segment_max_rows = 10);",
        )?;
        force_index_scan();

        let converted = Spi::get_one::<i64>("SELECT agentvec_consolidate('idx_av_conv')")?
            .expect("converted");
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
        assert_eq!((warm, retired, total), (1, 1, 40), "1 warm, 1 retired, no row lost");

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
             INSERT INTO t_av_recall(v) VALUES {values};
             CREATE INDEX idx_av_recall ON t_av_recall USING agentvec (v vector_l2_ops)
                WITH (hot_segment_max_rows = 60, ivf_lists = 8, ivf_probes = 8);"
        ))?;
        force_index_scan();

        // 240 rows at 60/segment: three sealed HOT segments.
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
            hits += got.iter().filter(|id| exact.contains(&(**id as usize))).count();
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
             INSERT INTO t_av_route
                SELECT g, ('[' || g || ',0,0,0,0,0,0,0]')::vector FROM generate_series(1, 120) AS g;
             CREATE INDEX idx_av_route ON t_av_route USING agentvec (v vector_l2_ops)
                WITH (hot_segment_max_rows = 40, router_top_m = 2);",
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
                router::route(&index, &[0.0; 8], &directory, &options, meta.get_router_base())
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
                s.algorithm() == crate::access_method::agentvec::directory::SegmentAlgorithm::IvfRaBitQ
                    && s.ownership() == SegmentOwnership::Owned
            })
            .map(|s| s.segment_id)
            .collect();
        assert_eq!(warm.len(), 2, "two converted segments");

        let decided = unsafe {
            router::route(&index, &[0.0; 8], &directory, &options, router_base)
        }
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
             INSERT INTO t_av_rr(v) VALUES {values};
             CREATE INDEX idx_av_rr ON t_av_rr USING agentvec (v vector_l2_ops)
                WITH (hot_segment_max_rows = 60, ivf_lists = 8, ivf_probes = 8,
                      router_top_m = 3);"
        ))?;
        force_index_scan();

        // 360 rows: five sealed segments, each converted (the sixth stays HOT).
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
            hits += got.iter().filter(|id| exact.contains(&(**id as usize))).count();
        }
        let recall = hits as f64 / total as f64;
        assert!(
            recall >= 0.85,
            "recall@10 with 3 of 5 WARM segments routed must be >= 0.85, got {recall}"
        );
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
             INSERT INTO t_av_heal
                SELECT g, ('[' || g || ',0,0,0,0,0,0,0]')::vector FROM generate_series(1, 25) AS g;
             CREATE INDEX idx_av_heal ON t_av_heal USING agentvec (v vector_l2_ops)
                WITH (hot_segment_max_rows = 10);",
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
        assert_eq!(converted, 10, "the Retiring segment is reclaimed and converted");
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
        assert_eq!(ids.len(), 3, "a bounded scan returns at most `search_candidates` rows");
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
}
