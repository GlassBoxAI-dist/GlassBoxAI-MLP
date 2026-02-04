/*
 * MIT License
 * Copyright (c) 2025 Matthew Abbott
 */

//! Core types for the MLP library.

use serde::{Deserialize, Serialize};

/// Activation function type for neural network layers.
///
/// # Example
/// ```
/// use glassboxai_mlp::ActivationType;
///
/// let activation = ActivationType::ReLU;
/// ```
#[derive(Clone, Copy, PartialEq, Eq, Debug, Serialize, Deserialize)]
pub enum ActivationType {
    /// Sigmoid activation: f(x) = 1 / (1 + e^(-x))
    Sigmoid = 0,
    /// Hyperbolic tangent: f(x) = tanh(x)
    Tanh = 1,
    /// Rectified Linear Unit: f(x) = max(0, x)
    ReLU = 2,
    /// Softmax activation for multi-class classification
    Softmax = 3,
}

impl Default for ActivationType {
    fn default() -> Self {
        Self::Sigmoid
    }
}

impl std::fmt::Display for ActivationType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ActivationType::Sigmoid => write!(f, "sigmoid"),
            ActivationType::Tanh => write!(f, "tanh"),
            ActivationType::ReLU => write!(f, "relu"),
            ActivationType::Softmax => write!(f, "softmax"),
        }
    }
}

impl ActivationType {
    /// Parse activation type from string.
    pub fn from_str(s: &str) -> Self {
        match s.to_lowercase().as_str() {
            "sigmoid" => Self::Sigmoid,
            "tanh" => Self::Tanh,
            "relu" => Self::ReLU,
            "softmax" => Self::Softmax,
            _ => Self::Sigmoid,
        }
    }
}

/// Optimizer type for training.
///
/// # Example
/// ```
/// use glassboxai_mlp::OptimizerType;
///
/// let optimizer = OptimizerType::Adam;
/// ```
#[derive(Clone, Copy, PartialEq, Eq, Debug, Serialize, Deserialize)]
pub enum OptimizerType {
    /// Stochastic Gradient Descent
    SGD = 0,
    /// Adam optimizer (recommended for most cases)
    Adam = 1,
    /// RMSProp optimizer
    RMSProp = 2,
}

impl Default for OptimizerType {
    fn default() -> Self {
        Self::Adam
    }
}

impl std::fmt::Display for OptimizerType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            OptimizerType::SGD => write!(f, "sgd"),
            OptimizerType::Adam => write!(f, "adam"),
            OptimizerType::RMSProp => write!(f, "rmsprop"),
        }
    }
}

impl OptimizerType {
    /// Parse optimizer type from string.
    pub fn from_str(s: &str) -> Self {
        match s.to_lowercase().as_str() {
            "sgd" => Self::SGD,
            "adam" => Self::Adam,
            "rmsprop" => Self::RMSProp,
            _ => Self::SGD,
        }
    }
}

/// GPU backend type for computation.
///
/// # Example
/// ```
/// use glassboxai_mlp::BackendType;
///
/// let backend = BackendType::CPU;
/// ```
#[derive(Clone, Copy, PartialEq, Eq, Debug, Serialize, Deserialize)]
pub enum BackendType {
    /// Pure Rust CPU implementation
    CPU,
    /// NVIDIA CUDA acceleration
    CUDA,
    /// OpenCL acceleration (AMD, Intel, NVIDIA)
    OpenCL,
}

impl Default for BackendType {
    fn default() -> Self {
        Self::CPU
    }
}

impl std::fmt::Display for BackendType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BackendType::CPU => write!(f, "cpu"),
            BackendType::CUDA => write!(f, "cuda"),
            BackendType::OpenCL => write!(f, "opencl"),
        }
    }
}

impl BackendType {
    /// Parse backend type from string.
    pub fn from_str(s: &str) -> Self {
        match s.to_lowercase().as_str() {
            "cuda" => Self::CUDA,
            "opencl" | "ocl" => Self::OpenCL,
            "cpu" => Self::CPU,
            _ => Self::CPU,
        }
    }
}

/// Configuration options for creating an MLP.
///
/// # Example
/// ```
/// use glassboxai_mlp::{MlpConfig, ActivationType, OptimizerType, BackendType};
///
/// let config = MlpConfig {
///     hidden_activation: ActivationType::ReLU,
///     output_activation: ActivationType::Sigmoid,
///     learning_rate: 0.001,
///     optimizer: OptimizerType::Adam,
///     backend: BackendType::CPU,
///     ..Default::default()
/// };
/// ```
#[derive(Clone, Debug)]
pub struct MlpConfig {
    /// Activation function for hidden layers (default: Sigmoid)
    pub hidden_activation: ActivationType,
    /// Activation function for output layer (default: Sigmoid)
    pub output_activation: ActivationType,
    /// Learning rate (default: 0.01)
    pub learning_rate: f64,
    /// Optimizer type (default: Adam)
    pub optimizer: OptimizerType,
    /// GPU backend (default: CPU)
    pub backend: BackendType,
    /// Dropout rate for regularization (default: 0.0)
    pub dropout_rate: f64,
    /// L2 regularization lambda (default: 0.0)
    pub l2_lambda: f64,
    /// Use batch normalization (default: false)
    pub batch_norm: bool,
    /// Adam beta1 parameter (default: 0.9)
    pub beta1: f64,
    /// Adam beta2 parameter (default: 0.999)
    pub beta2: f64,
}

impl Default for MlpConfig {
    fn default() -> Self {
        Self {
            hidden_activation: ActivationType::Sigmoid,
            output_activation: ActivationType::Sigmoid,
            learning_rate: 0.01,
            optimizer: OptimizerType::Adam,
            backend: BackendType::CPU,
            dropout_rate: 0.0,
            l2_lambda: 0.0,
            batch_norm: false,
            beta1: 0.9,
            beta2: 0.999,
        }
    }
}

/// A single data point for training.
#[derive(Clone, Debug)]
pub struct DataPoint {
    /// Input features
    pub input: Vec<f64>,
    /// Target output
    pub target: Vec<f64>,
}

impl DataPoint {
    /// Create a new data point.
    pub fn new(input: Vec<f64>, target: Vec<f64>) -> Self {
        Self { input, target }
    }
}

/// Training result containing loss history.
#[derive(Clone, Debug)]
pub struct TrainResult {
    /// Loss value for each epoch
    pub losses: Vec<f64>,
    /// Final loss value
    pub final_loss: f64,
}

/// Information about a neural network layer (for introspection).
#[derive(Clone, Debug)]
pub struct LayerInfo {
    /// Layer index (0 = input, 1+ = hidden/output)
    pub index: usize,
    /// Number of neurons in this layer
    pub size: usize,
    /// Activation function
    pub activation: ActivationType,
    /// Number of weights per neuron (connections from previous layer)
    pub weights_per_neuron: usize,
}

/// View into a single neuron (for introspection).
#[derive(Clone, Debug)]
pub struct NeuronView {
    /// Layer index
    pub layer: usize,
    /// Neuron index within layer
    pub index: usize,
    /// Current weights
    pub weights: Vec<f64>,
    /// Current bias
    pub bias: f64,
    /// Last computed output
    pub output: f64,
    /// Last computed error (gradient)
    pub error: f64,
}

/// Feature importance result.
#[derive(Clone, Debug)]
pub struct FeatureImportance {
    /// Feature index
    pub index: usize,
    /// Importance score (higher = more important)
    pub score: f64,
}
