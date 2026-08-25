//! SIMD-optimized distance computation for IVF search.
//!
//! Provides batch distance functions optimized for IVF search patterns:
//! - Finding nearest centroids (compare query to all centroids)
//! - Scanning posting lists (compare query to all vectors in a list)
//!
//! Uses the existing SIMD infrastructure from the `distance` module
//! (AVX2+FMA on x86_64, NEON on aarch64) via runtime dispatch.

use crate::access_method::distance::{self, DistanceType};

/// SIMD-optimized L2 (squared Euclidean) distance.
/// Delegates to the platform-specific SIMD implementation.
#[inline]
pub fn simd_distance_l2(a: &[f32], b: &[f32]) -> f32 {
    distance::distance_l2(a, b)
}

/// SIMD-optimized cosine distance.
/// Delegates to the platform-specific SIMD implementation.
#[inline]
pub fn simd_distance_cosine(a: &[f32], b: &[f32]) -> f32 {
    distance::distance_cosine(a, b)
}

/// SIMD-optimized inner product distance (negative inner product).
/// Delegates to the platform-specific SIMD implementation.
#[inline]
pub fn simd_distance_ip(a: &[f32], b: &[f32]) -> f32 {
    distance::distance_inner_product(a, b)
}

/// Get the SIMD distance function for a given distance type.
#[inline]
pub fn get_simd_distance_fn(distance_type: DistanceType) -> fn(&[f32], &[f32]) -> f32 {
    match distance_type {
        DistanceType::Cosine => simd_distance_cosine,
        DistanceType::L2 => simd_distance_l2,
        DistanceType::InnerProduct => simd_distance_ip,
    }
}

/// A (distance, index) pair for ranking results.
#[derive(Clone, Copy, Debug)]
pub struct DistanceIndex {
    pub distance: f32,
    pub index: usize,
}

/// Compute distances from a query vector to all centroids, returning
/// the indices of the `n_probe` nearest centroids sorted by distance.
///
/// This is the first step of IVF search: find which posting lists to scan.
/// Uses SIMD distance computation for each centroid comparison.
pub fn find_nearest_centroids(
    query: &[f32],
    centroids: &[Vec<f32>],
    distance_type: DistanceType,
    n_probe: usize,
) -> Vec<usize> {
    if centroids.is_empty() || n_probe == 0 {
        return Vec::new();
    }

    let dist_fn = get_simd_distance_fn(distance_type);
    let n_probe = n_probe.min(centroids.len());

    // Compute all distances using SIMD
    let mut distances: Vec<DistanceIndex> = centroids
        .iter()
        .enumerate()
        .map(|(i, centroid)| DistanceIndex {
            distance: dist_fn(query, centroid),
            index: i,
        })
        .collect();

    // Partial sort to find the n_probe nearest
    distances.select_nth_unstable_by(n_probe - 1, |a, b| {
        a.distance
            .partial_cmp(&b.distance)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    // Sort only the top n_probe
    distances[..n_probe].sort_by(|a, b| {
        a.distance
            .partial_cmp(&b.distance)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    distances[..n_probe].iter().map(|d| d.index).collect()
}

/// A candidate result from scanning a posting list.
#[derive(Clone, Copy, Debug)]
pub struct SearchResult {
    pub distance: f32,
    pub list_index: usize,
    pub entry_index: usize,
}

/// Scan a posting list (set of vectors) and return the top-k nearest
/// to the query vector, using SIMD distance computation.
///
/// `vectors` is the list of vectors in the posting list.
/// `list_index` identifies which posting list this is (for result tracking).
/// Returns up to `k` results sorted by distance.
pub fn scan_posting_list(
    query: &[f32],
    vectors: &[Vec<f32>],
    list_index: usize,
    distance_type: DistanceType,
    k: usize,
) -> Vec<SearchResult> {
    if vectors.is_empty() || k == 0 {
        return Vec::new();
    }

    let dist_fn = get_simd_distance_fn(distance_type);
    let k = k.min(vectors.len());

    // Compute all distances using SIMD
    let mut results: Vec<SearchResult> = vectors
        .iter()
        .enumerate()
        .map(|(entry_index, vec)| SearchResult {
            distance: dist_fn(query, vec),
            list_index,
            entry_index,
        })
        .collect();

    // Partial sort to find top-k
    results.select_nth_unstable_by(k - 1, |a, b| {
        a.distance
            .partial_cmp(&b.distance)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    // Sort only the top-k
    results[..k].sort_by(|a, b| {
        a.distance
            .partial_cmp(&b.distance)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    results.truncate(k);
    results
}

/// Merge results from multiple posting list scans and return the global top-k.
///
/// Takes a list of already-sorted result sets (one per posting list)
/// and merges them into a single sorted list of the top-k results.
pub fn merge_search_results(list_results: Vec<Vec<SearchResult>>, k: usize) -> Vec<SearchResult> {
    if k == 0 {
        return Vec::new();
    }

    // Flatten all results
    let mut all: Vec<SearchResult> = list_results.into_iter().flatten().collect();

    if all.is_empty() {
        return Vec::new();
    }

    let k = k.min(all.len());

    // Partial sort to find top-k
    all.select_nth_unstable_by(k - 1, |a, b| {
        a.distance
            .partial_cmp(&b.distance)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    // Sort only the top-k
    all[..k].sort_by(|a, b| {
        a.distance
            .partial_cmp(&b.distance)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    all.truncate(k);
    all
}

/// Perform a complete IVF search: find nearest centroids, scan their
/// posting lists, and return the global top-k results.
///
/// This is the main entry point for IVF search with SIMD optimization.
pub fn ivf_search(
    query: &[f32],
    centroids: &[Vec<f32>],
    posting_lists: &[Vec<Vec<f32>>],
    distance_type: DistanceType,
    n_probe: usize,
    k: usize,
) -> Vec<SearchResult> {
    // Step 1: Find nearest centroids using SIMD
    let nearest = find_nearest_centroids(query, centroids, distance_type, n_probe);

    // Step 2: Scan each selected posting list using SIMD
    let mut list_results = Vec::with_capacity(nearest.len());
    for &list_idx in &nearest {
        if list_idx < posting_lists.len() {
            let results =
                scan_posting_list(query, &posting_lists[list_idx], list_idx, distance_type, k);
            list_results.push(results);
        }
    }

    // Step 3: Merge results from all lists
    merge_search_results(list_results, k)
}

#[cfg(test)]
mod tests {
    use super::*;

    // Helper: scalar L2 distance for comparison
    fn scalar_l2(a: &[f32], b: &[f32]) -> f32 {
        a.iter().zip(b.iter()).map(|(x, y)| (x - y) * (x - y)).sum()
    }

    // Helper: scalar cosine distance for comparison
    fn scalar_cosine(a: &[f32], b: &[f32]) -> f32 {
        let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        (1.0 - dot).max(0.0)
    }

    // Helper: scalar inner product distance for comparison
    fn scalar_ip(a: &[f32], b: &[f32]) -> f32 {
        -a.iter().zip(b.iter()).map(|(x, y)| x * y).sum::<f32>()
    }

    // ===================== SIMD distance function tests =====================

    #[test]
    fn test_simd_distance_l2_matches_scalar() {
        let a: Vec<f32> = (0..128).map(|i| i as f32 * 0.1).collect();
        let b: Vec<f32> = (0..128).map(|i| (i as f32 + 5.0) * 0.1).collect();

        let simd_result = simd_distance_l2(&a, &b);
        let scalar_result = scalar_l2(&a, &b);

        assert!(
            (simd_result - scalar_result).abs() < 1e-3,
            "SIMD L2 {} != scalar L2 {}",
            simd_result,
            scalar_result
        );
    }

    #[test]
    fn test_simd_distance_cosine_matches_scalar() {
        // Use unit vectors for clean cosine comparison
        let a: Vec<f32> = vec![1.0, 0.0, 0.0, 0.0];
        let b: Vec<f32> = vec![0.0, 1.0, 0.0, 0.0];

        let simd_result = simd_distance_cosine(&a, &b);
        let scalar_result = scalar_cosine(&a, &b);

        assert!(
            (simd_result - scalar_result).abs() < 1e-5,
            "SIMD cosine {} != scalar cosine {}",
            simd_result,
            scalar_result
        );
    }

    #[test]
    fn test_simd_distance_ip_matches_scalar() {
        let a: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
        let b: Vec<f32> = vec![5.0, 6.0, 7.0, 8.0];

        let simd_result = simd_distance_ip(&a, &b);
        let scalar_result = scalar_ip(&a, &b);

        assert!(
            (simd_result - scalar_result).abs() < 1e-3,
            "SIMD IP {} != scalar IP {}",
            simd_result,
            scalar_result
        );
    }

    #[test]
    fn test_simd_distance_l2_various_dimensions() {
        // Test with dimensions that exercise different SIMD code paths
        for dim in [1, 2, 3, 4, 7, 8, 15, 16, 31, 32, 64, 128, 256, 768] {
            let a: Vec<f32> = (0..dim).map(|i| (i as f32) * 0.01).collect();
            let b: Vec<f32> = (0..dim).map(|i| (i as f32 + 1.0) * 0.01).collect();

            let simd_result = simd_distance_l2(&a, &b);
            let scalar_result = scalar_l2(&a, &b);

            let tolerance = (scalar_result.abs() * 1e-3).max(1e-5);
            assert!(
                (simd_result - scalar_result).abs() < tolerance,
                "dim={}: SIMD L2 {} != scalar L2 {}",
                dim,
                simd_result,
                scalar_result
            );
        }
    }

    #[test]
    fn test_simd_distance_l2_same_vector() {
        let a: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let result = simd_distance_l2(&a, &a);
        assert!(
            result.abs() < 1e-6,
            "Distance to self should be 0, got {}",
            result
        );
    }

    #[test]
    fn test_simd_distance_l2_empty_vectors() {
        let a: Vec<f32> = vec![];
        let b: Vec<f32> = vec![];
        let result = simd_distance_l2(&a, &b);
        assert_eq!(result, 0.0);
    }

    #[test]
    fn test_get_simd_distance_fn() {
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![5.0, 6.0, 7.0, 8.0];

        let l2_fn = get_simd_distance_fn(DistanceType::L2);
        assert!((l2_fn(&a, &b) - scalar_l2(&a, &b)).abs() < 1e-3);

        let cos_fn = get_simd_distance_fn(DistanceType::Cosine);
        // Use unit vectors for cosine
        let ua = vec![1.0, 0.0, 0.0, 0.0];
        let ub = vec![0.0, 1.0, 0.0, 0.0];
        assert!((cos_fn(&ua, &ub) - scalar_cosine(&ua, &ub)).abs() < 1e-5);

        let ip_fn = get_simd_distance_fn(DistanceType::InnerProduct);
        assert!((ip_fn(&a, &b) - scalar_ip(&a, &b)).abs() < 1e-3);
    }

    // ===================== Centroid selection tests =====================

    #[test]
    fn test_find_nearest_centroids_basic() {
        let query = vec![0.0, 0.0];
        let centroids = vec![
            vec![1.0, 0.0],   // distance = 1
            vec![10.0, 10.0], // distance = 200
            vec![0.5, 0.5],   // distance = 0.5
            vec![2.0, 0.0],   // distance = 4
        ];

        let result = find_nearest_centroids(&query, &centroids, DistanceType::L2, 2);
        assert_eq!(result.len(), 2);
        assert_eq!(result[0], 2); // closest: [0.5, 0.5]
        assert_eq!(result[1], 0); // second: [1.0, 0.0]
    }

    #[test]
    fn test_find_nearest_centroids_n_probe_larger_than_centroids() {
        let query = vec![0.0, 0.0];
        let centroids = vec![vec![1.0, 0.0], vec![2.0, 0.0]];

        let result = find_nearest_centroids(&query, &centroids, DistanceType::L2, 10);
        assert_eq!(result.len(), 2); // capped at centroids.len()
    }

    #[test]
    fn test_find_nearest_centroids_empty() {
        let query = vec![0.0, 0.0];
        let centroids: Vec<Vec<f32>> = vec![];

        let result = find_nearest_centroids(&query, &centroids, DistanceType::L2, 3);
        assert!(result.is_empty());
    }

    #[test]
    fn test_find_nearest_centroids_n_probe_zero() {
        let query = vec![0.0, 0.0];
        let centroids = vec![vec![1.0, 0.0]];

        let result = find_nearest_centroids(&query, &centroids, DistanceType::L2, 0);
        assert!(result.is_empty());
    }

    #[test]
    fn test_find_nearest_centroids_cosine() {
        let query = vec![1.0, 0.0];
        let centroids = vec![
            vec![1.0, 0.0],  // cosine distance = 0
            vec![0.0, 1.0],  // cosine distance = 1
            vec![-1.0, 0.0], // cosine distance = 2
        ];

        let result = find_nearest_centroids(&query, &centroids, DistanceType::Cosine, 2);
        assert_eq!(result.len(), 2);
        assert_eq!(result[0], 0); // same direction
        assert_eq!(result[1], 1); // orthogonal
    }

    // ===================== Posting list scan tests =====================

    #[test]
    fn test_scan_posting_list_basic() {
        let query = vec![0.0, 0.0];
        let vectors = vec![
            vec![1.0, 0.0],
            vec![0.1, 0.1],
            vec![10.0, 10.0],
            vec![0.5, 0.0],
        ];

        let results = scan_posting_list(&query, &vectors, 0, DistanceType::L2, 2);
        assert_eq!(results.len(), 2);
        // Closest should be [0.1, 0.1] (dist=0.02), then [0.5, 0.0] (dist=0.25)
        assert_eq!(results[0].entry_index, 1);
        assert_eq!(results[1].entry_index, 3);
        assert_eq!(results[0].list_index, 0);
    }

    #[test]
    fn test_scan_posting_list_empty() {
        let query = vec![0.0, 0.0];
        let vectors: Vec<Vec<f32>> = vec![];

        let results = scan_posting_list(&query, &vectors, 0, DistanceType::L2, 5);
        assert!(results.is_empty());
    }

    #[test]
    fn test_scan_posting_list_k_larger_than_list() {
        let query = vec![0.0, 0.0];
        let vectors = vec![vec![1.0, 0.0], vec![2.0, 0.0]];

        let results = scan_posting_list(&query, &vectors, 0, DistanceType::L2, 10);
        assert_eq!(results.len(), 2); // capped at vectors.len()
    }

    #[test]
    fn test_scan_posting_list_inner_product() {
        let query = vec![1.0, 0.0];
        let vectors = vec![
            vec![1.0, 0.0],  // IP = 1, distance = -1
            vec![0.0, 1.0],  // IP = 0, distance = 0
            vec![-1.0, 0.0], // IP = -1, distance = 1
        ];

        let results = scan_posting_list(&query, &vectors, 0, DistanceType::InnerProduct, 2);
        assert_eq!(results.len(), 2);
        // Closest (most negative) should be [1.0, 0.0] with distance -1
        assert_eq!(results[0].entry_index, 0);
        assert!(results[0].distance < 0.0);
    }

    // ===================== Merge results tests =====================

    #[test]
    fn test_merge_search_results_basic() {
        let list1 = vec![
            SearchResult {
                distance: 1.0,
                list_index: 0,
                entry_index: 0,
            },
            SearchResult {
                distance: 3.0,
                list_index: 0,
                entry_index: 1,
            },
        ];
        let list2 = vec![
            SearchResult {
                distance: 0.5,
                list_index: 1,
                entry_index: 0,
            },
            SearchResult {
                distance: 2.0,
                list_index: 1,
                entry_index: 1,
            },
        ];

        let merged = merge_search_results(vec![list1, list2], 3);
        assert_eq!(merged.len(), 3);
        assert_eq!(merged[0].distance, 0.5);
        assert_eq!(merged[1].distance, 1.0);
        assert_eq!(merged[2].distance, 2.0);
    }

    #[test]
    fn test_merge_search_results_empty() {
        let merged = merge_search_results(vec![], 5);
        assert!(merged.is_empty());
    }

    #[test]
    fn test_merge_search_results_k_zero() {
        let list1 = vec![SearchResult {
            distance: 1.0,
            list_index: 0,
            entry_index: 0,
        }];
        let merged = merge_search_results(vec![list1], 0);
        assert!(merged.is_empty());
    }

    // ===================== Full IVF search tests =====================

    #[test]
    fn test_ivf_search_basic() {
        let query = vec![0.0, 0.0];
        let centroids = vec![
            vec![0.1, 0.1],     // close to query
            vec![100.0, 100.0], // far from query
        ];
        let posting_lists = vec![
            // List 0: vectors near origin
            vec![vec![0.1, 0.0], vec![0.0, 0.2], vec![0.3, 0.3]],
            // List 1: vectors far away
            vec![vec![99.0, 99.0], vec![101.0, 101.0]],
        ];

        let results = ivf_search(&query, &centroids, &posting_lists, DistanceType::L2, 1, 2);
        assert_eq!(results.len(), 2);
        // All results should come from list 0 (nearest centroid)
        for r in &results {
            assert_eq!(r.list_index, 0);
        }
        // Results should be sorted by distance
        assert!(results[0].distance <= results[1].distance);
    }

    #[test]
    fn test_ivf_search_multiple_lists() {
        let query = vec![5.0, 5.0];
        let centroids = vec![vec![0.0, 0.0], vec![10.0, 10.0]];
        let posting_lists = vec![
            vec![vec![0.0, 0.0], vec![1.0, 1.0]],
            vec![vec![9.0, 9.0], vec![10.0, 10.0]],
        ];

        // Probe both lists
        let results = ivf_search(&query, &centroids, &posting_lists, DistanceType::L2, 2, 4);
        assert_eq!(results.len(), 4);
        // Should have results from both lists
        let has_list_0 = results.iter().any(|r| r.list_index == 0);
        let has_list_1 = results.iter().any(|r| r.list_index == 1);
        assert!(has_list_0 && has_list_1);
    }

    #[test]
    fn test_ivf_search_empty_index() {
        let query = vec![0.0, 0.0];
        let centroids: Vec<Vec<f32>> = vec![];
        let posting_lists: Vec<Vec<Vec<f32>>> = vec![];

        let results = ivf_search(&query, &centroids, &posting_lists, DistanceType::L2, 3, 10);
        assert!(results.is_empty());
    }

    #[test]
    fn test_ivf_search_high_dimensional() {
        // Test with realistic embedding dimensions
        let dim = 128;
        let query: Vec<f32> = (0..dim).map(|i| (i as f32) * 0.01).collect();

        let centroids: Vec<Vec<f32>> = (0..4)
            .map(|c| {
                (0..dim)
                    .map(|i| (i as f32 + c as f32 * 10.0) * 0.01)
                    .collect()
            })
            .collect();

        let posting_lists: Vec<Vec<Vec<f32>>> = (0..4)
            .map(|c| {
                (0..10)
                    .map(|j| {
                        (0..dim)
                            .map(|i| (i as f32 + c as f32 * 10.0 + j as f32 * 0.1) * 0.01)
                            .collect()
                    })
                    .collect()
            })
            .collect();

        let results = ivf_search(&query, &centroids, &posting_lists, DistanceType::L2, 2, 5);
        assert_eq!(results.len(), 5);
        // Results should be sorted by distance
        for w in results.windows(2) {
            assert!(w[0].distance <= w[1].distance);
        }
    }
}
