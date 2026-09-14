//! Node levels for a parallel build, drawn before any worker starts.
//!
//! A single-builder build draws a node's level when it inserts it, so "the i-th node"
//! and "the i-th random draw" coincide.  That stops being true the moment several
//! workers claim ids concurrently: which worker draws next depends on scheduling, so
//! the *level stream* -- and with it the graph, and with it the fingerprint -- would
//! change from run to run.  A parallel build therefore draws the whole table up front,
//! in the leader, where the seed is pinned: the levels become a function of the row
//! count alone, and the fingerprint gate keeps working for a parallel build.
//!
//! **An ordinal-indexed table is not enough for a parallel scan**, and that is worth
//! stating because it is the obvious first design: with `table_index_build_scan`, which
//! worker sees which row -- and in which order -- depends on scheduling, so "the i-th
//! row processed" is no more stable than "the i-th random draw".  What *is* stable is the
//! row itself, so the parallel path derives a level from the row's heap TID
//! ([`level_for_tid`]), which is a pure function of `(seed, tid)` and therefore
//! independent of who processes it.  The ordinal table remains useful for a
//! single-builder build (and for tests that want to assert a level directly), and both
//! use the same `random_level`, so the two agree when the order is the same.

use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};

use super::graph::random_level;

/// One level per node id, in id order.
#[derive(Clone, Debug)]
pub struct LevelTable {
    levels: Vec<u8>,
}

impl LevelTable {
    /// Draw `count` levels from `seed` (the same `build_seed` the single-builder path
    /// pins, so the two agree for the same row count).
    pub fn draw(count: usize, ml: f32, max_level: u8, seed: u64) -> Self {
        let mut rng = SmallRng::seed_from_u64(seed);
        Self {
            levels: (0..count)
                .map(|_| random_level(ml, max_level, &mut rng))
                .collect(),
        }
    }

    /// Level of node `id`; `0` for an id past the table (that node was never drawn, and
    /// level 0 is the only safe answer for a slot nobody claimed).
    #[inline]
    pub fn level(&self, id: u32) -> u8 {
        self.levels.get(id as usize).copied().unwrap_or(0)
    }

    /// Levels drawn so far.
    #[inline]
    pub fn len(&self) -> usize {
        self.levels.len()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.levels.is_empty()
    }

    /// Slabs the table's levels need, `sum(level + 1)` -- what the arena's slab budget
    /// has to cover for these nodes, and the number `plan_capacity` reasons about.
    pub fn slabs(&self) -> usize {
        self.levels.iter().map(|&l| l as usize + 1).sum()
    }

    /// Extend by `extra` more draws.  Only the leader does this, before any worker
    /// starts, which is why it may keep the generator's state to itself.
    pub fn extend(&mut self, extra: usize, ml: f32, max_level: u8, seed: u64) {
        let mut rng = SmallRng::seed_from_u64(seed);
        // Replay the draws already made, then continue: the table stays a pure function
        // of (count, seed), so extending it cannot change the levels already handed out.
        for _ in 0..self.levels.len() {
            random_level(ml, max_level, &mut rng);
        }
        for _ in 0..extra {
            self.levels.push(random_level(ml, max_level, &mut rng));
        }
    }
}

/// A row's level, derived from its heap TID rather than from when it was processed.
///
/// This is what makes a **parallel** build deterministic: `table_index_build_scan` hands
/// rows to workers in an order that depends on scheduling, so neither a shared counter
/// nor an RNG stream is stable.  The heap TID is: the same row yields the same level no
/// matter which worker gets it, and the whole level assignment becomes a function of the
/// table's contents and the pinned seed.  That is what lets the fingerprint gate -- and
/// with it the acceptance test that `workers = 0` is bit-identical -- apply to a parallel
/// build at all.
///
/// Distinct TIDs must give independent draws, so the seed is mixed with a hash of the TID
/// rather than merely offset by it: sequential TIDs are adjacent numbers, and feeding
/// those straight into a generator would correlate the levels of neighbouring rows.
pub fn level_for_tid(seed: u64, ml: f32, max_level: u8, tid: (u32, u16)) -> u8 {
    let mut rng = SmallRng::seed_from_u64(seed ^ fnv1a(tid));
    random_level(ml, max_level, &mut rng)
}

/// FNV-1a over the TID's block and offset, mixed so that neighbouring TIDs land far
/// apart in the generator's seed space.
fn fnv1a((block, offset): (u32, u16)) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for byte in block
        .to_le_bytes()
        .into_iter()
        .chain(offset.to_le_bytes())
    {
        h ^= byte as u64;
        h = h.wrapping_mul(0x100_0000_01b3);
    }
    // One avalanche round, so small TID differences do not leave small seed differences.
    h ^= h >> 33;
    h = h.wrapping_mul(0xff51_afd7_ed55_8ccd);
    h ^= h >> 33;
    h
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_same_seed_draws_the_same_levels() {
        // This is the whole point: a parallel build's level stream must not depend on
        // which worker claims an id.
        let a = LevelTable::draw(1000, 1.0 / 8f32.ln(), 7, 20240912);
        let b = LevelTable::draw(1000, 1.0 / 8f32.ln(), 7, 20240912);
        assert_eq!(a.levels, b.levels, "same seed, same table");
        assert_eq!(a.len(), 1000);

        let c = LevelTable::draw(1000, 1.0 / 8f32.ln(), 7, 20240913);
        assert_ne!(a.levels, c.levels, "a different seed is a different table");
    }

    #[test]
    fn levels_stay_within_the_configured_range() {
        let (ml, max_level) = (1.0 / 8f32.ln(), 3u8);
        let t = LevelTable::draw(5000, ml, max_level, 7);
        assert!(t.levels.iter().all(|&l| l <= max_level));
        // The distribution is geometric with p = 1/8, so most nodes are level 0 and the
        // top level is rare -- the same shape `random_level` produces on the fly.
        let zero = t.levels.iter().filter(|&&l| l == 0).count();
        assert!(zero > 5000 * 3 / 4, "most nodes are level 0: {}", zero);
        assert!(t.levels.iter().any(|&l| l > 0), "some are not");

        // A cap of zero forces every node to level 0, which is the degenerate graph.
        let flat = LevelTable::draw(100, ml, 0, 7);
        assert!(flat.levels.iter().all(|&l| l == 0));
        assert_eq!(flat.slabs(), 100, "one slab per node");
    }

    #[test]
    fn slabs_match_the_levels_and_ids_past_the_table_read_zero() {
        let t = LevelTable::draw(10, 1.0 / 8f32.ln(), 7, 3);
        let expected: usize = t.levels.iter().map(|&l| l as usize + 1).sum();
        assert_eq!(t.slabs(), expected);
        assert!(t.slabs() >= 10, "every node owns at least its layer-0 slab");
        assert_eq!(t.level(0), t.levels[0]);
        assert_eq!(t.level(9), t.levels[9]);
        assert_eq!(t.level(10), 0, "an id nobody drew is level 0, not a panic");
    }

    #[test]
    fn a_rows_level_does_not_depend_on_when_it_is_processed() {
        // The property the parallel build rests on.  Two "workers" processing the same
        // rows in different orders must assign the same level to each row -- which an
        // ordinal-indexed table or a shared RNG stream cannot promise.
        let (ml, max_level, seed) = (1.0 / 8f32.ln(), 7u8, 20240912u64);
        let rows: Vec<(u32, u16)> = (1..=200u32).map(|i| (i / 8 + 1, (i % 8 + 1) as u16)).collect();

        let in_order: Vec<u8> = rows
            .iter()
            .map(|&tid| level_for_tid(seed, ml, max_level, tid))
            .collect();
        let mut reversed = rows.clone();
        reversed.reverse();
        let mut out_of_order: Vec<u8> = vec![0; rows.len()];
        for &tid in &reversed {
            let i = rows.iter().position(|&r| r == tid).unwrap();
            out_of_order[i] = level_for_tid(seed, ml, max_level, tid);
        }
        assert_eq!(in_order, out_of_order, "processing order cannot change a level");

        // A different seed is a different assignment (the seed is actually used).
        let other: Vec<u8> = rows
            .iter()
            .map(|&tid| level_for_tid(seed + 1, ml, max_level, tid))
            .collect();
        assert_ne!(in_order, other);
    }

    #[test]
    fn neighbouring_rows_do_not_get_correlated_levels() {
        // Sequential TIDs must not produce a visibly patterned assignment: without the
        // hash, adjacent seeds come from the same generator region and the levels drift
        // together (all-zero runs, or long stretches at the cap).
        let (ml, max_level) = (1.0 / 8f32.ln(), 7u8);
        let levels: Vec<u8> = (1..=400u32)
            .map(|i| level_for_tid(7, ml, max_level, (1, i as u16)))
            .collect();
        let zero = levels.iter().filter(|&&l| l == 0).count();
        assert!(
            zero > levels.len() * 3 / 4,
            "the geometric shape must survive the hash: {} of {} are level 0",
            zero,
            levels.len()
        );
        assert!(levels.iter().any(|&l| l > 0), "some rows are above level 0");
        // No long constant runs, which is what correlated seeds would show.
        let longest = levels
            .windows(2)
            .fold((1usize, 1usize), |(best, run), w| {
                if w[0] == w[1] {
                    (best.max(run + 1), run + 1)
                } else {
                    (best, 1)
                }
            })
            .0;
        assert!(longest < 60, "longest constant run was {}", longest);
    }

    #[test]
    fn extending_replays_rather_than_redraws() {
        // Extending must not change levels already handed out: a worker may already
        // have used them.
        let mut t = LevelTable::draw(50, 1.0 / 8f32.ln(), 7, 99);
        let before = t.levels.clone();
        t.extend(25, 1.0 / 8f32.ln(), 7, 99);
        assert_eq!(t.len(), 75);
        assert_eq!(&t.levels[..50], &before[..], "the first 50 are untouched");
        // ... and it is the same as drawing 75 from the start.
        let whole = LevelTable::draw(75, 1.0 / 8f32.ln(), 7, 99);
        assert_eq!(t.levels, whole.levels, "extend == draw(n) for the same seed");
    }
}
