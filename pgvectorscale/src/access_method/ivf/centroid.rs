//! IVF centroid computation and management.
//!
//! Implements K-means++ initialization and Lloyd's algorithm for computing
//! centroids used in IVF (Inverted File) indexes.

use crate::access_method::distance::DistanceType;
use rand::prelude::*;
use rand::rngs::SmallRng;

/// K-means++ initialization: select k initial centroids from the data.
///
/// Algorithm:
/// 1. Choose the first centroid uniformly at random from the data points.
/// 2. For each data point, compute its distance to the nearest existing centroid.
/// 3. Choose the next centroid with probability proportional to distance².
/// 4. Repeat steps 2-3 until k centroids are chosen.
///
/// Returns k centroids as Vec<Vec<f32>>.
pub fn kmeans_plus_plus_init(
    vectors: &[Vec<f32>],
    k: usize,
    distance_type: DistanceType,
) -> Vec<Vec<f32>> {
    if vectors.is_empty() || k == 0 {
        return Vec::new();
    }

    let n = vectors.len();
    let k = k.min(n); // Can't have more centroids than data points
    let dist_fn = distance_type.get_distance_function();

    let mut rng = SmallRng::from_entropy();
    let mut centroids = Vec::with_capacity(k);

    // Step 1: Choose first centroid uniformly at random
    let first_idx = rng.gen_range(0..n);
    centroids.push(vectors[first_idx].clone());

    // Steps 2-4: Choose remaining centroids with probability proportional to distance²
    let mut min_distances = vec![f32::MAX; n];

    for _ in 1..k {
        // Update minimum distances with the latest centroid
        let last_centroid = centroids.last().unwrap();
        for (i, vec) in vectors.iter().enumerate() {
            let d = dist_fn(vec, last_centroid);
            if d < min_distances[i] {
                min_distances[i] = d;
            }
        }

        // Compute cumulative distribution for weighted sampling
        // Note: distance functions already return squared distances for L2,
        // and valid distance metrics for cosine/inner product, so we use them directly.
        let total: f32 = min_distances.iter().copied().sum();

        if total <= 0.0 {
            // All points are at distance 0 from existing centroids; pick randomly
            let idx = rng.gen_range(0..n);
            centroids.push(vectors[idx].clone());
            continue;
        }

        // Sample proportional to distance
        let threshold = rng.gen::<f32>() * total;
        let mut cumulative = 0.0f32;
        let mut chosen_idx = n - 1; // fallback
        for (i, &d) in min_distances.iter().enumerate() {
            cumulative += d;
            if cumulative >= threshold {
                chosen_idx = i;
                break;
            }
        }

        centroids.push(vectors[chosen_idx].clone());
    }

    centroids
}

/// Lloyd's algorithm: iteratively refine centroids by assignment and update.
///
/// Algorithm:
/// 1. Assign each vector to the nearest centroid.
/// 2. Recompute each centroid as the mean of its assigned vectors.
/// 3. Repeat until convergence (no reassignments) or max_iterations reached.
///
/// Returns the final centroids.
pub fn lloyds_algorithm(
    vectors: &[Vec<f32>],
    centroids: &mut [Vec<f32>],
    max_iterations: usize,
    distance_type: DistanceType,
) -> Vec<Vec<f32>> {
    if vectors.is_empty() || centroids.is_empty() {
        return centroids.to_vec();
    }

    let dist_fn = distance_type.get_distance_function();
    let k = centroids.len();
    let dim = vectors[0].len();

    // Allocate once and reuse across iterations: re-allocating these on every
    // pass is pure churn (the buffers are zeroed/reset in place below).
    let mut assignments = vec![0usize; vectors.len()];
    let mut new_centroids = vec![vec![0.0f32; dim]; k];
    let mut counts = vec![0usize; k];

    for _iter in 0..max_iterations {
        // Step 1: Assign each vector to nearest centroid
        for (i, vec) in vectors.iter().enumerate() {
            let mut best_cluster = 0;
            let mut best_dist = dist_fn(vec, &centroids[0]);
            for c in 1..k {
                let d = dist_fn(vec, &centroids[c]);
                if d < best_dist {
                    best_dist = d;
                    best_cluster = c;
                }
            }
            assignments[i] = best_cluster;
        }

        // Step 2: Recompute centroids as mean of assigned vectors
        for row in &mut new_centroids {
            row.fill(0.0);
        }
        counts.fill(0);

        for (i, vec) in vectors.iter().enumerate() {
            let c = assignments[i];
            counts[c] += 1;
            for (j, &val) in vec.iter().enumerate() {
                new_centroids[c][j] += val;
            }
        }

        // Divide by count to get mean; keep old centroid if cluster is empty
        let mut changed = false;
        for c in 0..k {
            if counts[c] > 0 {
                let count = counts[c] as f32;
                for j in 0..dim {
                    new_centroids[c][j] /= count;
                }
                // Check if centroid changed
                if new_centroids[c] != centroids[c] {
                    changed = true;
                    centroids[c] = new_centroids[c].clone();
                }
            }
            // If counts[c] == 0, keep the old centroid (no change)
        }

        // Step 3: Check convergence
        if !changed {
            break;
        }
    }

    centroids.to_vec()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::access_method::distance::DistanceType;

    #[test]
    fn test_kmeans_plus_plus_init_basic() {
        // 4 well-separated 2D clusters
        let vectors: Vec<Vec<f32>> = vec![
            vec![0.0, 0.0],
            vec![0.1, 0.1],
            vec![10.0, 10.0],
            vec![10.1, 10.1],
            vec![20.0, 0.0],
            vec![20.1, 0.1],
            vec![0.0, 20.0],
            vec![0.1, 20.1],
        ];

        let centroids = kmeans_plus_plus_init(&vectors, 4, DistanceType::L2);
        assert_eq!(centroids.len(), 4);
        // Each centroid should be one of the input vectors
        for c in &centroids {
            assert!(vectors.contains(c));
        }
    }

    #[test]
    fn test_kmeans_plus_plus_init_k_equals_1() {
        let vectors: Vec<Vec<f32>> = vec![vec![1.0, 2.0], vec![3.0, 4.0], vec![5.0, 6.0]];
        let centroids = kmeans_plus_plus_init(&vectors, 1, DistanceType::L2);
        assert_eq!(centroids.len(), 1);
        assert!(vectors.contains(&centroids[0]));
    }

    #[test]
    fn test_kmeans_plus_plus_init_k_greater_than_n() {
        let vectors: Vec<Vec<f32>> = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        let centroids = kmeans_plus_plus_init(&vectors, 10, DistanceType::L2);
        // Should cap at n
        assert_eq!(centroids.len(), 2);
    }

    #[test]
    fn test_kmeans_plus_plus_init_empty() {
        let vectors: Vec<Vec<f32>> = vec![];
        let centroids = kmeans_plus_plus_init(&vectors, 3, DistanceType::L2);
        assert!(centroids.is_empty());
    }

    #[test]
    fn test_kmeans_plus_plus_init_k_zero() {
        let vectors: Vec<Vec<f32>> = vec![vec![1.0, 2.0]];
        let centroids = kmeans_plus_plus_init(&vectors, 0, DistanceType::L2);
        assert!(centroids.is_empty());
    }

    #[test]
    fn test_kmeans_plus_plus_init_cosine() {
        let vectors: Vec<Vec<f32>> = vec![
            vec![1.0, 0.0],
            vec![0.0, 1.0],
            vec![-1.0, 0.0],
            vec![0.0, -1.0],
        ];
        let centroids = kmeans_plus_plus_init(&vectors, 2, DistanceType::Cosine);
        assert_eq!(centroids.len(), 2);
    }

    #[test]
    fn test_lloyds_algorithm_converges() {
        // Two clear clusters
        let vectors: Vec<Vec<f32>> = vec![
            vec![0.0, 0.0],
            vec![0.1, 0.1],
            vec![0.2, 0.0],
            vec![10.0, 10.0],
            vec![10.1, 10.1],
            vec![10.2, 10.0],
        ];

        // Start with poor initial centroids
        let mut centroids = vec![vec![0.0, 0.0], vec![10.0, 10.0]];
        let result = lloyds_algorithm(&vectors, &mut centroids, 100, DistanceType::L2);

        assert_eq!(result.len(), 2);

        // After convergence, centroids should be near the cluster means
        // Cluster 1 mean: (0.1, 0.0333...), Cluster 2 mean: (10.1, 10.0333...)
        let c0 = &result[0];
        let c1 = &result[1];

        // One centroid should be near (0.1, 0.033) and other near (10.1, 10.033)
        let near_origin = |c: &[f32]| c[0] < 5.0 && c[1] < 5.0;
        let near_ten = |c: &[f32]| c[0] > 5.0 && c[1] > 5.0;

        assert!(
            (near_origin(c0) && near_ten(c1)) || (near_ten(c0) && near_origin(c1)),
            "Centroids should converge to cluster centers: {:?}",
            result
        );

        // Check actual values are close to expected means
        let expected_c1 = vec![0.1, 0.033333334];
        let expected_c2 = vec![10.1, 10.033333];

        let (actual_c1, actual_c2) = if near_origin(c0) {
            (c0, c1)
        } else {
            (c1, c0)
        };

        for (a, e) in actual_c1.iter().zip(expected_c1.iter()) {
            assert!((a - e).abs() < 1e-4, "Expected ~{:?}, got {:?}", expected_c1, actual_c1);
        }
        for (a, e) in actual_c2.iter().zip(expected_c2.iter()) {
            assert!((a - e).abs() < 1e-4, "Expected ~{:?}, got {:?}", expected_c2, actual_c2);
        }
    }

    #[test]
    fn test_lloyds_algorithm_k_equals_1() {
        let vectors: Vec<Vec<f32>> = vec![vec![1.0, 2.0], vec![3.0, 4.0], vec![5.0, 6.0]];
        let mut centroids = vec![vec![0.0, 0.0]];
        let result = lloyds_algorithm(&vectors, &mut centroids, 100, DistanceType::L2);

        assert_eq!(result.len(), 1);
        // Mean of all vectors: (3.0, 4.0)
        assert!((result[0][0] - 3.0).abs() < 1e-5);
        assert!((result[0][1] - 4.0).abs() < 1e-5);
    }

    #[test]
    fn test_lloyds_algorithm_empty_vectors() {
        let vectors: Vec<Vec<f32>> = vec![];
        let mut centroids = vec![vec![1.0, 2.0]];
        let result = lloyds_algorithm(&vectors, &mut centroids, 100, DistanceType::L2);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0], vec![1.0, 2.0]);
    }

    #[test]
    fn test_lloyds_algorithm_empty_centroids() {
        let vectors: Vec<Vec<f32>> = vec![vec![1.0, 2.0]];
        let mut centroids: Vec<Vec<f32>> = vec![];
        let result = lloyds_algorithm(&vectors, &mut centroids, 100, DistanceType::L2);
        assert!(result.is_empty());
    }

    #[test]
    fn test_lloyds_algorithm_max_iterations() {
        // Should terminate even with max_iterations = 1
        let vectors: Vec<Vec<f32>> = vec![vec![0.0, 0.0], vec![10.0, 10.0]];
        let mut centroids = vec![vec![0.0, 0.0], vec![10.0, 10.0]];
        let result = lloyds_algorithm(&vectors, &mut centroids, 1, DistanceType::L2);
        assert_eq!(result.len(), 2);
    }

    #[test]
    fn test_lloyds_algorithm_cosine() {
        // Vectors pointing in different directions
        let vectors: Vec<Vec<f32>> = vec![
            vec![1.0, 0.0],
            vec![0.99, 0.01],
            vec![0.0, 1.0],
            vec![0.01, 0.99],
        ];
        let mut centroids = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let result = lloyds_algorithm(&vectors, &mut centroids, 100, DistanceType::Cosine);
        assert_eq!(result.len(), 2);
    }

    #[test]
    fn test_lloyds_algorithm_inner_product() {
        let vectors: Vec<Vec<f32>> = vec![
            vec![1.0, 0.0],
            vec![0.9, 0.1],
            vec![-1.0, 0.0],
            vec![-0.9, -0.1],
        ];
        let mut centroids = vec![vec![1.0, 0.0], vec![-1.0, 0.0]];
        let result = lloyds_algorithm(&vectors, &mut centroids, 100, DistanceType::InnerProduct);
        assert_eq!(result.len(), 2);
    }

    #[test]
    fn test_full_pipeline_init_then_lloyds() {
        // Test the full pipeline: init + refinement
        let vectors: Vec<Vec<f32>> = vec![
            vec![0.0, 0.0],
            vec![0.1, 0.0],
            vec![0.0, 0.1],
            vec![100.0, 100.0],
            vec![100.1, 100.0],
            vec![100.0, 100.1],
        ];

        let mut centroids = kmeans_plus_plus_init(&vectors, 2, DistanceType::L2);
        assert_eq!(centroids.len(), 2);

        let result = lloyds_algorithm(&vectors, &mut centroids, 100, DistanceType::L2);
        assert_eq!(result.len(), 2);

        // Verify centroids converged to cluster means
        let mean1 = vec![0.033333334, 0.033333334];
        let mean2 = vec![100.03333, 100.03333];

        let matches = |c: &[f32], expected: &[f32]| {
            c.iter()
                .zip(expected.iter())
                .all(|(a, e)| (a - e).abs() < 0.1)
        };

        assert!(
            (matches(&result[0], &mean1) && matches(&result[1], &mean2))
                || (matches(&result[0], &mean2) && matches(&result[1], &mean1)),
            "Pipeline result should converge to cluster means: {:?}",
            result
        );
    }
}
