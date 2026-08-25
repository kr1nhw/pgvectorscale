//! General quantization framework.
//!
//! Quantization compresses full-precision `f32` vectors so index nodes can be
//! stored and compared more cheaply (both in space and in CPU cost).  This
//! module defines the *framework*: a [`Quantizer`] trait that any concrete
//! quantization scheme implements (e.g. Sign-Bit Quantization (SBQ) and
//! RaBitQ), and a storage-agnostic [`QuantizedVector`] value type that any
//! index algorithm can persist and compare without knowing which quantizer
//! produced it.
//!
//! The framework is deliberately independent of the DiskANN graph internals:
//! a new access method or storage layout can plug in any `Quantizer` and
//! store/compare its `QuantizedVector`s without forking index-specific code.

use super::{distance::DistanceType, meta_page::MetaPage};

pub mod rabitq;

/// Identifies which quantization scheme a vector (or an index) uses.
///
/// Stored as a `u8` on disk; do not renumber existing values.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[repr(u8)]
pub enum QuantizerType {
    /// Sign-Bit Quantization: 1 bit per dimension (optionally mean-centered).
    #[default]
    Sbq = 0,
    /// RaBitQ: randomized sign-bit quantization with a norm-aware estimator.
    Rabitq = 1,
}

impl QuantizerType {
    pub fn from_u8(value: u8) -> Self {
        match value {
            0 => QuantizerType::Sbq,
            1 => QuantizerType::Rabitq,
            _ => pgrx::error!("Invalid quantizer type: {}", value),
        }
    }
}

/// A quantized vector in an index-agnostic form.
///
/// Owns its data so it can be cached, serialized, or passed between storage
/// layers.  Distance computation is dispatched here so index algorithms only
/// ever deal with this enum, never with a specific quantizer's layout.
#[derive(Clone, Debug, PartialEq)]
pub enum QuantizedVector {
    /// SBQ code words (`u64` bit-pack per `BITS_STORE_TYPE_SIZE` bits).
    Sbq(Vec<u64>),
    /// RaBitQ code (sign bits) plus norm metadata.
    Rabitq(crate::access_method::quantization::rabitq::RabitqVector),
}

impl QuantizedVector {
    pub fn quantizer_type(&self) -> QuantizerType {
        match self {
            QuantizedVector::Sbq(_) => QuantizerType::Sbq,
            QuantizedVector::Rabitq(_) => QuantizerType::Rabitq,
        }
    }

    /// Size in bytes of the quantized representation (code words only, not
    /// including heap/vec overhead).
    pub fn quantized_size_bytes(&self) -> usize {
        match self {
            QuantizedVector::Sbq(code) => code.len() * std::mem::size_of::<u64>(),
            QuantizedVector::Rabitq(v) => v.quantized_size_bytes(),
        }
    }

    /// Estimated distance between two quantized vectors for the given
    /// distance type.  Both vectors must come from the same quantizer.
    pub fn distance(&self, other: &Self, distance_type: DistanceType) -> f32 {
        match (self, other) {
            (QuantizedVector::Sbq(a), QuantizedVector::Sbq(b)) => {
                super::distance::distance_xor_optimized(a, b) as f32
            }
            (QuantizedVector::Rabitq(a), QuantizedVector::Rabitq(b)) => {
                a.estimated_distance(b, distance_type)
            }
            _ => pgrx::error!("Mismatched quantized vector types in distance computation"),
        }
    }
}

/// Metadata that a quantizer may need to persist across sessions (e.g. the
/// SBQ means page pointer, or the RaBitQ rotation seed).
#[derive(Clone, Debug, Default)]
pub struct QuantizerMetadata {
    pub quantizer_type: QuantizerType,
}

/// A quantization scheme.
///
/// Implementations are responsible for their own training (if any) and for
/// producing [`QuantizedVector`]s that are comparable via
/// [`QuantizedVector::distance`].
pub trait Quantizer: Send + Sync {
    fn quantize(&self, full_vector: &[f32]) -> QuantizedVector;
    fn quantized_size(&self, full_vector_size: usize) -> usize;
    fn num_bits_per_dimension(&self) -> u8;
    fn quantizer_type(&self) -> QuantizerType;
    fn metadata(&self) -> QuantizerMetadata {
        QuantizerMetadata {
            quantizer_type: self.quantizer_type(),
        }
    }

    /// Begin a training pass.  No-op for data-independent quantizers.
    fn start_training(&mut self, _meta_page: &MetaPage) {}
    /// Feed one training sample.  No-op for data-independent quantizers.
    fn add_sample(&mut self, _sample: &[f32]) {}
    /// End the training pass.  No-op for data-independent quantizers.
    fn finish_training(&mut self) {}

    /// The quantized representation of a vector for a new index node.
    /// Defaults to `quantize`.
    fn vector_for_new_node(&self, _meta_page: &MetaPage, full_vector: &[f32]) -> QuantizedVector {
        self.quantize(full_vector)
    }
}
