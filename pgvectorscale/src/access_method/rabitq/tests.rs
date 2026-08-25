#[cfg(test)]
mod unit {
    use crate::access_method::distance::DistanceType;
    use crate::access_method::rabitq::quantize::{
        binary_dot, ex_dot, pack_sign_bits, RabitqQuantizer,
    };
    use crate::access_method::rabitq::rotation::random_fast_rotation_signs;
    use rand::{rngs::SmallRng, SeedableRng};

    fn make_test_quantizer(
        dim: usize,
        num_bits: u8,
        distance_type: DistanceType,
    ) -> RabitqQuantizer {
        let mut q = RabitqQuantizer::new_with_distance_type(distance_type, num_bits);
        let mut rng = SmallRng::seed_from_u64(42);
        let mean = vec![0.1f32; dim];
        let signs = random_fast_rotation_signs(dim, &mut rng);
        q.load(1, mean, signs, num_bits);
        q
    }

    #[test]
    fn test_pack_sign_bits_round_trip() {
        let rotated = vec![1.0f32, -1.0, 3.0, -4.0, 0.0, 2.0, -7.0, 8.0];
        let code = pack_sign_bits(&rotated);
        assert_eq!(code.len(), 1);
        assert_eq!(code[0] & 0b0000_0001, 1);
        assert_eq!(code[0] & 0b0000_0010, 0);
        assert_eq!(code[0] & 0b0000_0100, 4);
        assert_eq!(code[0] & 0b0000_1000, 0);
        assert_eq!(code[0] & 0b0001_0000, 16);
        assert_eq!(code[0] & 0b0010_0000, 32);
        assert_eq!(code[0] & 0b0100_0000, 0);
        assert_eq!(code[0] & 0b1000_0000, 128);
    }

    #[test]
    fn test_binary_dot_matches_reference() {
        let q = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let code = pack_sign_bits(&[1.0, -1.0, 1.0, -1.0, 1.0, 1.0, -1.0, 1.0]);
        assert_eq!(binary_dot(&code, &q), 23.0);
    }

    #[test]
    fn test_ex_dot_byte_packed() {
        let q = vec![1.0f32, 2.0, 3.0, 4.0];
        let ex = vec![2u8, 3, 5, 7];
        assert_eq!(ex_dot(&ex, &q, 7), 51.0);
    }

    #[test]
    fn test_quantize_produces_valid_codes() {
        let dim = 64;
        let q = make_test_quantizer(dim, 1, DistanceType::L2);
        let v: Vec<f32> = (0..dim).map(|i| (i as f32).sin()).collect();
        let code = q.quantize(&v);
        assert_eq!(code.code.len(), 1);
        assert!(code.ex_code.is_empty());
        assert!(code.f_rescale.is_finite());
    }

    #[test]
    fn test_quantize_multibit_produces_ex_codes() {
        let dim = 64;
        let q4 = make_test_quantizer(dim, 4, DistanceType::L2);
        let q8 = make_test_quantizer(dim, 8, DistanceType::L2);
        let v: Vec<f32> = (0..dim).map(|i| (i as f32).cos()).collect();
        let code4 = q4.quantize(&v);
        let code8 = q8.quantize(&v);
        assert_eq!(code4.ex_code.len(), dim / 2); // nibble packed
        assert_eq!(code8.ex_code.len(), dim); // byte packed
    }

    #[test]
    fn test_estimator_orders_near_far() {
        let dim = 128;
        let q = make_test_quantizer(dim, 1, DistanceType::L2);
        let base: Vec<f32> = (0..dim).map(|i| (i as f32).sin()).collect();
        let near: Vec<f32> = base.iter().map(|v| v + 0.001).collect();
        let far: Vec<f32> = base.iter().map(|v| v + 1.0).collect();

        let qm = q.query_measure(&base);
        let code_near = q.quantize(&near);
        let code_far = q.quantize(&far);

        let d_near = q.estimate_distance(&qm, &code_near);
        let d_far = q.estimate_distance(&qm, &code_far);
        assert!(d_near < d_far, "near={d_near}, far={d_far}");
    }

    #[test]
    fn test_cosine_estimator_orders_near_far() {
        let dim = 128;
        let q = make_test_quantizer(dim, 1, DistanceType::Cosine);
        let base: Vec<f32> = (0..dim).map(|i| (i as f32).sin()).collect();
        let base_norm = base.iter().map(|v| v * v).sum::<f32>().sqrt();
        let base: Vec<f32> = base.iter().map(|v| v / base_norm).collect();
        let near: Vec<f32> = base.iter().map(|v| v + 0.001).collect();
        let far: Vec<f32> = (0..dim).map(|i| (i as f32).cos()).collect();

        let qm = q.query_measure(&base);
        let code_near = q.quantize(&near);
        let code_far = q.quantize(&far);

        let d_near = q.estimate_distance(&qm, &code_near);
        let d_far = q.estimate_distance(&qm, &code_far);
        assert!(d_near < d_far, "cos near={d_near}, far={d_far}");
    }

    #[test]
    fn test_inner_product_estimator_orders_near_far() {
        let dim = 128;
        let q = make_test_quantizer(dim, 1, DistanceType::InnerProduct);
        let base: Vec<f32> = (0..dim).map(|i| (i as f32).sin()).collect();
        let near: Vec<f32> = base.iter().map(|v| v + 0.001).collect();
        let far: Vec<f32> = base.iter().map(|v| -v).collect();

        let qm = q.query_measure(&base);
        let code_near = q.quantize(&near);
        let code_far = q.quantize(&far);

        // Inner-product distance is -<q, v>; near (higher ip) must be smaller.
        let d_near = q.estimate_distance(&qm, &code_near);
        let d_far = q.estimate_distance(&qm, &code_far);
        assert!(d_near < d_far, "ip near={d_near}, far={d_far}");
    }
}

#[cfg(any(test, feature = "pg_test"))]
#[pgrx::pg_schema]
mod tests {
    use pgrx::*;

    use crate::access_method::distance::DistanceType;

    #[pg_test]
    unsafe fn test_rabitq_index_creation_default_neighbors() -> spi::Result<()> {
        crate::access_method::build::tests::test_index_creation_and_accuracy_scaffold(
            DistanceType::Cosine,
            "storage_layout = rabitq",
            "rabitq_default_neighbors",
            1536,
        )?;
        Ok(())
    }

    #[pg_test]
    unsafe fn test_rabitq_index_creation_few_neighbors() -> spi::Result<()> {
        crate::access_method::build::tests::test_index_creation_and_accuracy_scaffold(
            DistanceType::Cosine,
            "num_neighbors=10, storage_layout = rabitq",
            "rabitq_few_neighbors",
            1536,
        )?;
        Ok(())
    }

    #[pg_test]
    unsafe fn test_rabitq_4bit_index_creation() -> spi::Result<()> {
        crate::access_method::build::tests::test_index_creation_and_accuracy_scaffold(
            DistanceType::L2,
            "storage_layout = rabitq, num_bits_per_dimension = 4",
            "rabitq_4bit",
            1536,
        )?;
        Ok(())
    }

    #[pg_test]
    unsafe fn test_rabitq_8bit_index_creation() -> spi::Result<()> {
        crate::access_method::build::tests::test_index_creation_and_accuracy_scaffold(
            DistanceType::L2,
            "storage_layout = rabitq, num_bits_per_dimension = 8",
            "rabitq_8bit",
            1536,
        )?;
        Ok(())
    }

    #[pg_test]
    unsafe fn test_rabitq_index_creation_low_memory() -> spi::Result<()> {
        crate::access_method::build::tests::test_sized_index_scaffold(
            "num_neighbors=40, storage_layout = rabitq",
            1536,
            2000,
            Some(1024),
        )?;
        Ok(())
    }

    #[test]
    fn test_rabitq_storage_delete_vacuum_plain() {
        crate::access_method::vacuum::tests::test_delete_vacuum_plain_scaffold(
            "num_neighbors = 10, storage_layout = rabitq",
        );
    }

    #[test]
    fn test_rabitq_storage_delete_vacuum_full() {
        crate::access_method::vacuum::tests::test_delete_vacuum_full_scaffold(
            "num_neighbors = 38, storage_layout = rabitq",
        );
    }

    #[test]
    fn test_rabitq_storage_update_with_null() {
        crate::access_method::vacuum::tests::test_update_with_null_scaffold(
            "num_neighbors = 38, storage_layout = rabitq",
        );
    }

    #[pg_test]
    unsafe fn test_rabitq_storage_empty_table_insert() -> spi::Result<()> {
        crate::access_method::build::tests::test_empty_table_insert_scaffold(
            "num_neighbors=38, storage_layout = rabitq",
        )
    }

    #[pg_test]
    unsafe fn test_rabitq_storage_insert_empty_insert() -> spi::Result<()> {
        crate::access_method::build::tests::test_insert_empty_insert_scaffold(
            "num_neighbors=38, storage_layout = rabitq",
        )
    }
}
