/*
 * MIT License
 * Copyright (c) 2025 Matthew Abbott
 */
#![allow(non_snake_case)]
#![allow(non_camel_case_types)]
#![allow(dead_code)]

//! # Facaded MLP CUDA/OpenCL
//!
//! A GPU-accelerated Multi-Layer Perceptron implementation in Rust with support for CUDA and OpenCL.
//!
//! ## Example
//!
//! ```no_run
//! use facaded_mlp_cuda::{TMultiLayerPerceptronCUDA, TActivationType, TGPUBackend};
//!
//! # fn main() -> Result<(), String> {
//! let mut mlp = TMultiLayerPerceptronCUDA::new_with_backend(
//!     2,                              // input size
//!     &vec![8],                       // hidden layers
//!     1,                              // output size
//!     TActivationType::atSigmoid,     // hidden activation
//!     TActivationType::atSigmoid,     // output activation
//!     TGPUBackend::CUDA,              // GPU backend
//! )?;
//!
//! let input = vec![1.0, 0.0];
//! let target = vec![1.0];
//! mlp.Train(&input, &target)?;
//!
//! let output = mlp.Predict(&input)?;
//! println!("Output: {:?}", output);
//! # Ok(())
//! # }
//! ```

pub const EPSILON: f64 = 1e-15;
pub const BLOCK_SIZE: u32 = 256;
pub const MODEL_MAGIC: &str = "MLPCUDA1";

mod kernels;
pub mod mlp;
mod onnx;
pub mod gpu_backend;

#[cfg(feature = "opencl")]
mod opencl_kernels;
#[cfg(feature = "opencl")]
mod opencl_mlp;

#[cfg(feature = "cli")]
pub mod cli;

#[cfg(feature = "python")]
mod python;

#[cfg(feature = "nodejs")]
mod nodejs;

// Re-export main types
pub use mlp::{
    TMultiLayerPerceptronCUDA,
    TActivationType,
    TOptimizerType,
    TDataPoint,
    Darray,
    TIntArray,
};

pub use gpu_backend::{TGPUBackend, detect_available_backends, select_best_backend};

// Re-export utility functions
pub use mlp::{
    LoadDataCSV,
    NormalizeData,
    ShuffleData,
    ParseActivation,
    ParseOptimizer,
    ParseDoubleArray,
    ParseIntArray,
    ActivationToStr,
    OptimizerToStr,
    MaxIndex,
};
