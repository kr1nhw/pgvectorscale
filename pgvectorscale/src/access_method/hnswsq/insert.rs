//! hnswsq insert — pgvector-style concurrent protocol.
//!
//! There is NO global writer lock.  Correctness rests on three rules:
//!
//! 1. **Single-content-lock rule**: a backend holds at most one buffer content
//!    lock at a time and never acquires any other lock (content, extension,
//!    meta) while holding an exclusive content lock.  Reads snapshot-and-
//!    release.  Buffer content locks are not deadlock-detected by PostgreSQL,
//!    so this rule is what makes concurrent inserts hang-free.
//! 2. **Two-phase optimistic neighbor updates**: a backlink write snapshots
//!    the target's list + identity under a share lock (phase 1), computes the
//!    pruned list with no locks held, then re-validates identity and list
//!    equality under the exclusive lock (phase 2), retrying on a racing
//!    change and falling back to append-if-room.
//! 3. **Crash safety via transaction atomicity**: the node, its lists, and its
//!    backlinks are all written before the inserting transaction commits, so a
//!    crash rolls the heap row back with them; an orphaned index node then
//!    points at a dead TID and is tombstoned by vacuum's linear page walk.
//!
//! Ordering within one insert (mirrors pgvector): allocate slot → write node
//! with empty (padded) lists → search → fill own lists → add backlinks →
//! promote entry point if this node's level is higher.

use std::cell::RefCell;

use pgrx::*;
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};

use crate::access_method::distance::{preprocess_cosine, DistanceFn, DistanceType};
use crate::access_method::hnswsq::graph::{
    distance_encoded, greedy_descent, random_level, search_layer, select_neighbors_heuristic,
    DiskGraph, GraphAccess,
};
use crate::access_method::hnswsq::meta_page::HnswMetaPage;
use crate::access_method::hnswsq::node::{
    item_fits, load_node_view, modify_node, HnswNode,
};
use crate::access_method::hnswsq::quantize::{Codec, HnswPrecision, Sq8Calibration};
use crate::access_method::node::WriteableNode;
use crate::access_method::pg_vector::PgVectorInternal;
use crate::util::page::{PageType, WritablePage};
use crate::util::ItemPointer;

// Per-backend RNG for level assignment (insert is not parallel-aware).
thread_local! {
    static INSERT_RNG: RefCell<SmallRng> = RefCell::new(SmallRng::from_entropy());
}

/// Immutable parameters for one insert (or a disk-mode build row stream).
pub struct InsertCtx<'a> {
    pub index: &'a PgRelation,
    pub codec: &'a Codec,
    pub dist_fn: DistanceFn,
    pub distance_type: DistanceType,
    pub m: usize,
    pub m0: usize,
    pub ef_construction: usize,
    pub ml: f32,
    pub max_level: u8,
}

impl<'a> InsertCtx<'a> {
    pub fn from_meta(index: &'a PgRelation, meta: &HnswMetaPage, codec: &'a Codec) -> Self {
        Self {
            index,
            codec,
            dist_fn: meta.get_distance_type().get_distance_function(),
            distance_type: meta.get_distance_type(),
            m: meta.get_m(),
            m0: meta.get_m0(),
            ef_construction: meta.get_ef_construction(),
            ml: meta.get_ml(),
            max_level: meta.get_max_level(),
        }
    }

    /// Neighbor capacity for `layer` (layer 0 → m0, upper layers → m).
    pub fn cap_for_layer(&self, layer: usize) -> usize {
        if layer == 0 {
            self.m0
        } else {
            self.m
        }
    }
}

/// Build the codec for an index (loads the SQ8 calibration chain when the
/// layout is `f8`; the training-free layouts are stateless).
pub fn codec_for(index: &PgRelation, meta: &HnswMetaPage) -> Codec {
    match meta.get_precision() {
        HnswPrecision::Sq8 => {
            let ptr = meta
                .get_calibration_pointer()
                .expect("hnswsq: f8 (sq8) index is missing its calibration");
            Codec::new_sq8(&Sq8Calibration::load(index, ptr))
        }
        p => Codec::new(p, meta.get_num_dimensions() as usize),
    }
}

/// Insert one tuple into the hnswsq index.
#[pg_guard]
pub unsafe extern "C-unwind" fn aminsert(
    index: pg_sys::Relation,
    values: *mut pg_sys::Datum,
    isnull: *mut bool,
    heap_tid: pg_sys::ItemPointer,
    _heap: pg_sys::Relation,
    _check_unique: pg_sys::IndexUniqueCheck::Type,
    _index_unchanged: bool,
    _index_info: *mut pg_sys::IndexInfo,
) -> bool {
    // Skip null vectors.
    if *isnull {
        return false;
    }

    let index_rel = unsafe { PgRelation::from_pg(index) };
    let meta = HnswMetaPage::fetch(&index_rel);
    let codec = codec_for(&index_rel, &meta);
    let dim = meta.get_num_dimensions() as usize;

    // Extract the vector (detoast-copy pattern shared with the IVF paths).
    let datum = *values;
    let detoasted = pg_sys::pg_detoast_datum_copy(datum.cast_mut_ptr());
    let pg_vec = detoasted.cast::<PgVectorInternal>();
    let mut vector = (*pg_vec).to_slice().to_vec();
    pg_sys::pfree(detoasted.cast());

    if vector.len() != dim {
        error!(
            "hnswsq: vector has {} dimensions but the index was built for {}",
            vector.len(),
            dim
        );
    }
    if meta.get_distance_type() == DistanceType::Cosine {
        preprocess_cosine(&mut vector);
    }

    let ctx = InsertCtx::from_meta(&index_rel, &meta, &codec);
    let tid = ItemPointer::with_item_pointer_data(*heap_tid);
    INSERT_RNG.with(|r| insert_vector(&ctx, tid, &vector, &mut *r.borrow_mut()));
    false
}

/// Core insert routine, shared by `aminsert` and the disk-mode build path.
///
/// `vector` must already be normalized when the index distance type is cosine.
pub unsafe fn insert_vector(
    ctx: &InsertCtx,
    heap_tid: ItemPointer,
    vector: &[f32],
    rng: &mut impl Rng,
) -> ItemPointer {
    let index = ctx.index;
    let meta = HnswMetaPage::fetch(index);

    // 1. Draw the level and write the node with empty (capacity-padded)
    //    neighbor lists.  Until backlinks exist the node is unreachable, so no
    //    concurrent search can observe it with half-filled lists.
    let level = random_level(ctx.ml, ctx.max_level, rng);
    let mut encoded = Vec::with_capacity(ctx.codec.vector_bytes());
    let clamped = ctx.codec.encode_into(vector, &mut encoded);
    let node = HnswNode::new_clamped(
        heap_tid,
        level,
        encoded.clone(),
        vec![Vec::new(); level as usize + 1],
        ctx.m,
        ctx.m0,
        clamped,
    );
    let bytes = node.serialize_to_vec();
    let (self_ptr, new_hint) = allocate_node_slot(index, meta.get_insert_page(), &bytes);
    if let Some(block) = new_hint {
        if meta.get_insert_page() != Some(block) {
            HnswMetaPage::set_insert_page(index, block);
        }
    }

    // 2. Resolve the entry point.  For an empty index, try to claim it; the
    //    loser of a first-insert race re-reads and links into the winner's
    //    graph (so no committed row is left unreachable).
    let mut entry = resolve_entry(index, &meta);
    if entry.is_none() {
        if HnswMetaPage::claim_entry_point_if_empty(index, self_ptr, level) {
            // Sole first node: empty lists are final.
            return self_ptr;
        }
        let meta2 = HnswMetaPage::fetch(index);
        entry = resolve_entry(index, &meta2);
        if entry.is_none() {
            // Pathological (entry vanished between the claim and the fetch);
            // leave the node unlinked — vacuum cleans it if the row dies, and
            // a later insert's promotion reconnects the entry.
            return self_ptr;
        }
    }
    let (ep, entry_level) = entry.expect("entry resolved above");

    // 3. Search: greedy ef=1 descent to `level + 1`, then ef_construction
    //    search at each layer from min(level, entry_level) down to 0.
    let subject = ctx.codec.decode(&encoded);
    let access = DiskGraph { index };
    let top = (level as usize).min(entry_level);

    let ep_view = load_node_view(index, ep).expect("entry resolved above");
    let mut cur = (
        distance_encoded(ctx.codec, ctx.distance_type, &subject, &ep_view.vector),
        ep,
    );
    if entry_level > top {
        cur = greedy_descent(
            ctx.codec,
            ctx.distance_type,
            &subject,
            &access,
            cur,
            entry_level,
            top + 1,
        );
    }

    let mut layer_neighbors: Vec<Vec<ItemPointer>> = vec![Vec::new(); level as usize + 1];
    for l in (0..=top).rev() {
        let hits = search_layer(
            ctx.codec,
            ctx.distance_type,
            &subject,
            &access,
            vec![cur],
            ctx.ef_construction,
            l,
        );
        if let Some(best) = hits.first() {
            cur = (best.dist, best.id);
        }
        // Tombstones stay out of fresh links but keep routing the search.
        let cands: Vec<(f32, ItemPointer)> = hits
            .iter()
            .filter(|h| !h.deleted && h.id != self_ptr)
            .map(|h| (h.dist, h.id))
            .collect();
        layer_neighbors[l] = select_neighbors_heuristic(
            ctx.codec,
            ctx.distance_type,
            &access,
            cands,
            ctx.cap_for_layer(l),
        );
    }

    // 4. Fill our own neighbor lists (single exclusive lock, one commit).
    let _ = modify_node(index, self_ptr, |mut archived| {
        for l in 0..=top {
            archived
                .as_mut()
                .set_neighbors(l, &layer_neighbors[l], ctx.cap_for_layer(l));
        }
    });

    // 5. Backlinks (two-phase optimistic, per plan §F.5).
    for l in 0..=top {
        let cap = ctx.cap_for_layer(l);
        for &nb in &layer_neighbors[l] {
            add_backlink(ctx, nb, self_ptr, &encoded, l, cap);
        }
    }

    // 6. Entry-point promotion (RMW re-checks the level under the meta lock;
    //    the highest concurrent promoter wins).
    if (level as usize) > entry_level {
        HnswMetaPage::promote_entry_point(index, self_ptr, level);
    }

    self_ptr
}

/// Resolve a loadable entry point (None when the index is empty or the entry
/// vanished — a crash orphan freed by vacuum).
unsafe fn resolve_entry(
    index: &PgRelation,
    meta: &HnswMetaPage,
) -> Option<(ItemPointer, usize)> {
    let ep = meta.get_entry_point()?;
    load_node_view(index, ep)?;
    Some((ep, meta.get_entry_level().max(0) as usize))
}

/// Allocate a slot for `bytes` and append the item, returning its pointer and
/// a new insert-page hint when the allocation rotated pages.
///
/// Order: hint page (share→exclusive, fits check) → free-list page (meta RMW
/// pop, reinit) → relation extension.  Concurrent allocators serialize on the
/// page's exclusive lock; a loser whose page filled up rotates and republishes
/// the hint (last writer wins — bounded transient space waste, pgvector
/// parity).
unsafe fn allocate_node_slot(
    index: &PgRelation,
    hint: Option<pg_sys::BlockNumber>,
    bytes: &[u8],
) -> (ItemPointer, Option<pg_sys::BlockNumber>) {
    // 1. The hinted append page.
    if let Some(block) = hint {
        let mut page = WritablePage::modify(index, block);
        if page.get_type() == PageType::HnswNode && item_fits(page.get_aligned_free_space(), bytes.len())
        {
            let off = page.add_item(bytes);
            let b = page.get_block_number();
            page.commit();
            return (ItemPointer::new(b, off), None);
        }
        // Dropping the page aborts its GenericXLog state and releases the lock.
    }

    // 2. A page recycled by vacuum.
    if let Some(block) = HnswMetaPage::pop_free_page(index) {
        let mut page = WritablePage::modify(index, block);
        page.reinit(PageType::HnswNode);
        // A fresh page always fits: CREATE INDEX rejects shapes whose largest
        // node (max_level) does not fit one item.
        let off = page.add_item(bytes);
        let b = page.get_block_number();
        page.commit();
        return (ItemPointer::new(b, off), Some(b));
    }

    // 3. Extend the relation (LockedBufferExclusive::new takes the extension
    //    lock before the content lock — the only heavyweight-lock ordering).
    let mut page = WritablePage::new(index, PageType::HnswNode);
    let off = page.add_item(bytes);
    let b = page.get_block_number();
    page.commit();
    (ItemPointer::new(b, off), Some(b))
}

/// Two-phase optimistic backlink: add `self_ptr` to `target`'s neighbor list
/// at `layer` (capacity `cap`), pruning with the HNSW heuristic.
///
/// Phase 1 (share locks, one page at a time): snapshot the target's identity
/// (heap TID + level + deleted flag) and list, decode the vectors of the list
/// members, and compute the pruned list.  Phase 2 (single exclusive lock):
/// re-validate identity and list equality, then write.  A racing change
/// retries phase 1 (bounded), then falls back to append-if-room — never a
/// hang, never a torn list.
unsafe fn add_backlink(
    ctx: &InsertCtx,
    target: ItemPointer,
    self_ptr: ItemPointer,
    self_encoded: &[u8],
    layer: usize,
    cap: usize,
) {
    let index = ctx.index;
    let access = DiskGraph { index };

    enum Outcome {
        Wrote,
        Changed,
        Gone,
    }

    for _attempt in 0..3 {
        // ---- Phase 1: snapshot (no locks held across loads) ----
        let Some(view) = load_node_view(index, target) else {
            return; // recycled page: skip this backlink
        };
        if view.deleted || (view.level as usize) < layer {
            return; // never backlink into tombstones or missing layers
        }
        let current: Vec<ItemPointer> = view
            .neighbors
            .get(layer)
            .cloned()
            .unwrap_or_default();
        if current.contains(&self_ptr) {
            return; // already linked (racing duplicate insert of the same row)
        }

        let subject = ctx.codec.decode(&view.vector);
        let mut cands: Vec<(f32, ItemPointer)> = Vec::with_capacity(current.len() + 1);
        let d_self = distance_encoded(
            ctx.codec,
            ctx.distance_type,
            &subject,
            self_encoded,
        );
        cands.push((d_self, self_ptr));
        for &m in &current {
            if let Some(enc) = access.vector(m) {
                let d = distance_encoded(ctx.codec, ctx.distance_type, &subject, &enc);
                cands.push((d, m));
            }
            // Vanished member: dropped from the new list (implicit repair).
        }
        let new_list = select_neighbors_heuristic(
            ctx.codec,
            ctx.distance_type,
            &access,
            cands,
            cap,
        );

        // ---- Phase 2: validate + write under one exclusive lock ----
        let identity_tid = view.heap_tid;
        let identity_level = view.level;
        let outcome = modify_node(index, target, |mut archived| {
            if archived.is_deleted() {
                return Outcome::Gone;
            }
            let tid = archived.heap_tid.deserialize_item_pointer();
            if tid != identity_tid || archived.level != identity_level {
                return Outcome::Gone; // page recycled into a different node
            }
            let cur: Vec<ItemPointer> = archived.iter_valid_neighbors(layer).collect();
            if cur != current {
                return Outcome::Changed; // racing inserter won; retry phase 1
            }
            archived.as_mut().set_neighbors(layer, &new_list, cap);
            Outcome::Wrote
        });
        match outcome {
            Some(Outcome::Wrote) | Some(Outcome::Gone) | None => return,
            Some(Outcome::Changed) => continue,
        }
    }

    // Fallback after repeated races: append-if-room (closure reads/writes only
    // the target page — no external loads, so the single-lock rule holds).
    let _ = modify_node(index, target, |mut archived| {
        if archived.is_deleted() {
            return;
        }
        let cur: Vec<ItemPointer> = archived.iter_valid_neighbors(layer).collect();
        if cur.len() < cap && !cur.contains(&self_ptr) {
            let mut list = cur;
            list.push(self_ptr);
            archived.as_mut().set_neighbors(layer, &list, cap);
        }
    });
}
