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
//! Storing the table also makes the level of an id something a test can assert
//! directly, instead of re-deriving it by replaying the RNG.

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
