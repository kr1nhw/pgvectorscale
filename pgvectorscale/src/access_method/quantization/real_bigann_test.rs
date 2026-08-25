#[cfg(test)]
mod real_bigann_tests {
    use super::*;
    use std::fs;

/// Loads /tmp/est_test.csv: "id,\"v1,v2,...\"" and checks the 8-bit
/// estimator ranking on REAL bigann vectors (query 0 = id -1).
#[test]
fn real_bigann_8bit_ranking() {
    let raw = fs::read_to_string("/tmp/est_test.csv").expect("est_test.csv");
    let mut vectors: Vec<(i32, Vec<f32>)> = Vec::new();
    for line in raw.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let (id, rest) = line.split_once(',').expect("csv split");
        let id: i32 = id.trim().parse().expect("id parse");
        let vals: Vec<f32> = rest
            .trim_matches('"')
            .split(',')
            .filter_map(|v| v.trim().parse::<f32>().ok())
            .collect();
        vectors.push((id, vals));
    }
    let query = vectors
        .iter()
        .find(|(id, _)| *id == -1)
        .expect("query vector")
        .1
        .clone();
    let q = crate::access_method::quantization::rabitq::RabitqQuantizer::new(8, 42, 128);

    // exact + estimated L2 for each candidate
    let mut exact: Vec<(i32, f32)> = Vec::new();
    let mut est: Vec<(i32, f32)> = Vec::new();
    for (id, v) in &vectors {
        if *id == -1 {
            continue;
        }
        let exact_d: f32 = v
            .iter()
            .zip(query.iter())
            .map(|(a, b)| (a - b) * (a - b))
            .sum();
        let qv = q.quantize(v);
        let rq = q.rotate_query(&query);
        let m = qv.dot_with_rotated(&rq.rotated);
        let cos = m / qv.l1_of_rotated;
        let est_d = crate::access_method::quantization::rabitq::RabitqVector::distance_from_cos(
            cos,
            qv.sum_of_x2,
            rq.sum_of_x2,
            crate::access_method::distance::DistanceType::L2,
        );
        exact.push((*id, exact_d));
        est.push((*id, est_d));
    }
    exact.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
    est.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
    let exact_top = exact[0].0;
    let est_top = est[0].0;
    eprintln!("exact top5: {:?}", &exact[..5]);
    eprintln!("est   top5: {:?}", &est[..5]);
    assert_eq!(est_top, exact_top, "8-bit estimator must rank the exact NN first");
    }
}
