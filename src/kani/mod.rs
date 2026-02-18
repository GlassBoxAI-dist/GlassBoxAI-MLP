//! @file
//! @ingroup MLP_Core_Verified
//! Kani Verification Test Suite for GlassBoxAI MLP
//!
//! Formal verification harnesses following CISA "Secure by Design" standards.
//! Each harness uses symbolic inputs to mathematically prove safety properties.
//!
//! Run with: `cargo kani --tests`

pub mod core_types;
pub mod bounds_checks;
pub mod pointer_validity;
pub mod no_panic;
pub mod integer_overflow;
pub mod division_by_zero;
pub mod state_consistency;
pub mod deadlock_free;
pub mod input_sanitization;
pub mod result_coverage;
pub mod memory_leaks;
pub mod constant_time;
pub mod state_machine;
pub mod enum_exhaustion;
pub mod floating_point;
pub mod resource_limits;
pub mod ffi_cuda;
pub mod ffi_opencl;
pub mod ffi_c_boundary;

pub use core_types::*;

