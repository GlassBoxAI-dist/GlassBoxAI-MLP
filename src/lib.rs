/*
 * MIT License
 * Copyright (c) 2025 Matthew Abbott
 */

//! # GlassBoxAI MLP
//!
//! A transparent, verifiable multi-layer perceptron library with GPU acceleration.
//!
//! This library provides a GPU-accelerated neural network implementation designed for
//! explainable AI. Unlike black-box neural networks, GlassBoxAI-MLP gives you full
//! introspection into weights, gradients, activations, and optimizer state.
//!
//! ## Features
//!
//! - **GPU Acceleration**: CUDA, OpenCL, and CPU backends
//! - **Full Introspection**: Access weights, biases, gradients, and optimizer state
//! - **Explainable AI**: Feature importance, activation histograms, gradient analysis
//! - **Multiple Optimizers**: SGD, Adam, RMSProp
//! - **Regularization**: Dropout, L2 regularization, batch normalization
//! - **Serialization**: Save/load models to JSON
//! - **Language Bindings**: Python, Node.js, Julia, and C++ wrappers
//!
//! ## Quick Start
//!
//! ```no_run
//! use glassboxai_mlp::{MLP, MlpConfig, ActivationType, OptimizerType};
//!
//! // Create a simple network with defaults
//! let mut mlp = MLP::new(2, &[8], 1).unwrap();
//!
//! // XOR training data
//! let inputs = vec![
//!     vec![0.0, 0.0], vec![0.0, 1.0],
//!     vec![1.0, 0.0], vec![1.0, 1.0],
//! ];
//! let targets = vec![
//!     vec![0.0], vec![1.0], vec![1.0], vec![0.0],
//! ];
//!
//! // Train
//! mlp.set_learning_rate(0.5);
//! mlp.set_optimizer(OptimizerType::Adam);
//!
//! let result = mlp.fit(&inputs, &targets, 1000, true).unwrap();
//! println!("Final loss: {:.6}", result.final_loss);
//!
//! // Predict
//! let output = mlp.predict(&[1.0, 0.0]).unwrap();
//! println!("Prediction: {:.4}", output[0]);
//!
//! // Introspect
//! let importance = mlp.feature_importance();
//! for fi in &importance {
//!     println!("Feature {}: {:.4}", fi.index, fi.score);
//! }
//! ```
//!
//! ## Custom Configuration
//!
//! ```no_run
//! use glassboxai_mlp::{MLP, MlpConfig, ActivationType, OptimizerType, BackendType};
//!
//! let config = MlpConfig {
//!     hidden_activation: ActivationType::ReLU,
//!     output_activation: ActivationType::Softmax,
//!     learning_rate: 0.001,
//!     optimizer: OptimizerType::Adam,
//!     backend: BackendType::CUDA,
//!     dropout_rate: 0.2,
//!     l2_lambda: 0.0001,
//!     ..Default::default()
//! };
//!
//! let mlp = MLP::with_config(784, &[256, 128], 10, config).unwrap();
//! ```
//!
//! ## Introspection / Glass Box Features
//!
//! ```no_run
//! use glassboxai_mlp::MLP;
//!
//! let mut mlp = MLP::new(2, &[4], 1).unwrap();
//! mlp.train(&[1.0, 0.0], &[1.0]).unwrap();
//!
//! // View layer information
//! let info = mlp.layer_info(1);
//! println!("Layer {}: {} neurons, {} activation", info.index, info.size, info.activation);
//!
//! // View individual neurons
//! let neuron = mlp.neuron_view(1, 0);
//! println!("Weights: {:?}", neuron.weights);
//! println!("Bias: {}", neuron.bias);
//!
//! // Access optimizer state (Adam M and V values)
//! let m = mlp.get_weight_m(1, 0, 0);
//! let v = mlp.get_weight_v(1, 0, 0);
//! println!("Adam state: M={}, V={}", m, v);
//!
//! // Activation histogram
//! let hist = mlp.activation_histogram(1, 10);
//! println!("Activation distribution: {:?}", hist);
//! ```
//!
//! ## GPU Backend Selection
//!
//! ```no_run
//! use glassboxai_mlp::{MLP, BackendType, available_backends};
//!
//! // Check available backends
//! println!("Available: {:?}", available_backends());
//!
//! // Create with specific backend
//! let mut mlp = MLP::new(2, &[8], 1).unwrap();
//! mlp.set_backend(BackendType::CUDA).unwrap();
//! ```
//!
//! ## Feature Flags
//!
//! - `cuda` - Enable NVIDIA CUDA support (requires CUDA toolkit)
//! - `opencl` - Enable OpenCL support
//! - `python` - Build Python bindings (PyO3)
//! - `nodejs` - Build Node.js bindings (NAPI)
//! - `julia` - Build Julia/C FFI bindings

#![allow(non_snake_case)]
#![allow(non_camel_case_types)]
#![allow(dead_code)]

/// Model file magic number for version identification.
pub const MODEL_MAGIC: &str = "MLPCUDA1";

/// Small epsilon for numerical stability.
pub const EPSILON: f64 = 1e-15;

/// CUDA/OpenCL kernel block size.
pub const BLOCK_SIZE: u32 = 256;

// Internal modules
mod kernels;
mod onnx;

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

#[cfg(feature = "julia")]
pub mod julia;

// Core modules
pub mod mlp;
pub mod gpu_backend;
pub mod types;
pub mod facade;

// ========== Primary API (recommended) ==========

/// The main MLP type - a transparent, verifiable neural network.
pub use facade::MLP;

/// Utility functions for backend detection.
pub use facade::{available_backends, select_best_backend};

// ========== Type re-exports ==========

pub use types::{
    ActivationType,
    OptimizerType,
    BackendType,
    MlpConfig,
    DataPoint,
    TrainResult,
    LayerInfo,
    NeuronView,
    FeatureImportance,
};

// ========== Low-level API (for advanced users) ==========

/// Low-level internal types (use `MLP` facade instead when possible).
pub mod internal {
    pub use crate::mlp::{
        TMultiLayerPerceptronCUDA,
        TActivationType,
        TOptimizerType,
        TDataPoint,
        TNeuron,
        TLayer,
        Darray,
        TIntArray,
    };
    
    pub use crate::gpu_backend::{
        TGPUBackend,
        detect_available_backends,
        select_best_backend,
    };
    
    pub use crate::mlp::{
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
}

// For backwards compatibility, also expose at crate root
// (will be deprecated in future versions)
#[doc(hidden)]
pub use mlp::{
    TMultiLayerPerceptronCUDA,
    TActivationType,
    TOptimizerType,
    TDataPoint,
    Darray,
    TIntArray,
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

#[doc(hidden)]
pub use gpu_backend::{TGPUBackend, detect_available_backends as detect_backends};

#[cfg(any(kani, test))]
pub mod kani;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_mlp() {
        let mlp = MLP::new(2, &[4], 1).unwrap();
        assert_eq!(mlp.input_size(), 2);
        assert_eq!(mlp.output_size(), 1);
        assert_eq!(mlp.hidden_sizes(), vec![4]);
    }

    #[test]
    fn test_predict() {
        let mut mlp = MLP::new(2, &[4], 1).unwrap();
        let output = mlp.predict(&[1.0, 0.0]).unwrap();
        assert_eq!(output.len(), 1);
        assert!(output[0] >= 0.0 && output[0] <= 1.0);
    }

    #[test]
    fn test_train() {
        let mut mlp = MLP::new(2, &[4], 1).unwrap();
        mlp.train(&[1.0, 0.0], &[1.0]).unwrap();
    }

    #[test]
    fn test_xor() {
        let config = MlpConfig {
            learning_rate: 0.5,
            optimizer: OptimizerType::Adam,
            ..Default::default()
        };
        let mut mlp = MLP::with_config(2, &[8], 1, config).unwrap();

        let inputs = vec![
            vec![0.0, 0.0], vec![0.0, 1.0],
            vec![1.0, 0.0], vec![1.0, 1.0],
        ];
        let targets = vec![
            vec![0.0], vec![1.0], vec![1.0], vec![0.0],
        ];

        let result = mlp.fit(&inputs, &targets, 1000, false).unwrap();
        // XOR should converge, but loss threshold is lenient due to random init
        assert!(result.final_loss < 0.5, "Loss {} should be < 0.5", result.final_loss);
    }

    #[test]
    fn test_introspection() {
        let mut mlp = MLP::new(2, &[4], 1).unwrap();
        mlp.train(&[1.0, 0.0], &[1.0]).unwrap();

        let info = mlp.layer_info(1);
        assert_eq!(info.size, 4);

        let neuron = mlp.neuron_view(1, 0);
        assert_eq!(neuron.weights.len(), 2);

        let importance = mlp.feature_importance();
        assert_eq!(importance.len(), 2);
    }
}
