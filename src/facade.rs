/*
 * MIT License
 * Copyright (c) 2025 Matthew Abbott
 */

//! High-level facade API for the MLP library.
//!
//! This module provides a clean, idiomatic Rust interface to the MLP library.

use crate::mlp::{TMultiLayerPerceptronCUDA, TActivationType, TOptimizerType};
use crate::gpu_backend::TGPUBackend;
use crate::types::*;

/// GPU-accelerated Multi-Layer Perceptron.
///
/// A transparent, verifiable neural network implementation with support for
/// CUDA, OpenCL, and CPU backends. Designed for explainable AI with full
/// introspection of weights, gradients, and activations.
///
/// # Example
///
/// ```no_run
/// use glassboxai_mlp::{MLP, MlpConfig, ActivationType, OptimizerType};
///
/// // Create a simple network
/// let mut mlp = MLP::new(2, &[8], 1).unwrap();
///
/// // Or with custom configuration
/// let config = MlpConfig {
///     hidden_activation: ActivationType::ReLU,
///     learning_rate: 0.01,
///     optimizer: OptimizerType::Adam,
///     ..Default::default()
/// };
/// let mut mlp = MLP::with_config(2, &[16, 8], 1, config).unwrap();
///
/// // Train on XOR
/// let inputs = vec![
///     vec![0.0, 0.0], vec![0.0, 1.0],
///     vec![1.0, 0.0], vec![1.0, 1.0],
/// ];
/// let targets = vec![
///     vec![0.0], vec![1.0], vec![1.0], vec![0.0],
/// ];
///
/// for epoch in 0..1000 {
///     for (input, target) in inputs.iter().zip(targets.iter()) {
///         mlp.train(input, target).unwrap();
///     }
/// }
///
/// // Predict
/// let output = mlp.predict(&[1.0, 0.0]).unwrap();
/// println!("Output: {:.4}", output[0]);
/// ```
pub struct MLP {
    inner: TMultiLayerPerceptronCUDA,
}

impl MLP {
    /// Create a new MLP with default configuration.
    ///
    /// # Arguments
    /// * `input_size` - Number of input neurons
    /// * `hidden_sizes` - Sizes of hidden layers (e.g., `&[8]` or `&[16, 8]`)
    /// * `output_size` - Number of output neurons
    ///
    /// # Example
    /// ```no_run
    /// use glassboxai_mlp::MLP;
    ///
    /// let mlp = MLP::new(2, &[8], 1).unwrap();
    /// ```
    pub fn new(input_size: usize, hidden_sizes: &[usize], output_size: usize) -> Result<Self, String> {
        Self::with_config(input_size, hidden_sizes, output_size, MlpConfig::default())
    }

    /// Create a new MLP with custom configuration.
    ///
    /// # Arguments
    /// * `input_size` - Number of input neurons
    /// * `hidden_sizes` - Sizes of hidden layers
    /// * `output_size` - Number of output neurons
    /// * `config` - Configuration options
    ///
    /// # Example
    /// ```no_run
    /// use glassboxai_mlp::{MLP, MlpConfig, ActivationType, OptimizerType};
    ///
    /// let config = MlpConfig {
    ///     hidden_activation: ActivationType::ReLU,
    ///     learning_rate: 0.001,
    ///     optimizer: OptimizerType::Adam,
    ///     ..Default::default()
    /// };
    ///
    /// let mlp = MLP::with_config(784, &[256, 128], 10, config).unwrap();
    /// ```
    pub fn with_config(
        input_size: usize,
        hidden_sizes: &[usize],
        output_size: usize,
        config: MlpConfig,
    ) -> Result<Self, String> {
        let hidden: Vec<i32> = hidden_sizes.iter().map(|&s| s as i32).collect();
        
        let mut inner = TMultiLayerPerceptronCUDA::new_with_backend(
            input_size as i32,
            &hidden,
            output_size as i32,
            activation_to_internal(config.hidden_activation),
            activation_to_internal(config.output_activation),
            backend_to_internal(config.backend),
        )?;
        
        inner.LearningRate = config.learning_rate;
        inner.Optimizer = optimizer_to_internal(config.optimizer);
        inner.DropoutRate = config.dropout_rate;
        inner.L2Lambda = config.l2_lambda;
        inner.UseBatchNorm = config.batch_norm;
        inner.Beta1 = config.beta1;
        inner.Beta2 = config.beta2;
        
        Ok(Self { inner })
    }

    /// Make a prediction.
    ///
    /// # Arguments
    /// * `input` - Input values (must match input_size)
    ///
    /// # Returns
    /// Output values from the network.
    ///
    /// # Example
    /// ```no_run
    /// use glassboxai_mlp::MLP;
    ///
    /// let mut mlp = MLP::new(2, &[4], 1).unwrap();
    /// let output = mlp.predict(&[1.0, 0.0]).unwrap();
    /// ```
    pub fn predict(&mut self, input: &[f64]) -> Result<Vec<f64>, String> {
        self.inner.Predict(&input.to_vec())
    }

    /// Train on a single sample.
    ///
    /// # Arguments
    /// * `input` - Input values
    /// * `target` - Target output values
    ///
    /// # Example
    /// ```no_run
    /// use glassboxai_mlp::MLP;
    ///
    /// let mut mlp = MLP::new(2, &[4], 1).unwrap();
    /// mlp.train(&[1.0, 0.0], &[1.0]).unwrap();
    /// ```
    pub fn train(&mut self, input: &[f64], target: &[f64]) -> Result<(), String> {
        self.inner.Train(&input.to_vec(), &target.to_vec())
    }

    /// Train on a dataset for multiple epochs.
    ///
    /// # Arguments
    /// * `inputs` - Vector of input samples
    /// * `targets` - Vector of target outputs
    /// * `epochs` - Number of training epochs
    /// * `verbose` - Print progress every 100 epochs
    ///
    /// # Returns
    /// Training result with loss history.
    ///
    /// # Example
    /// ```no_run
    /// use glassboxai_mlp::MLP;
    ///
    /// let mut mlp = MLP::new(2, &[8], 1).unwrap();
    /// let inputs = vec![vec![0.0, 0.0], vec![0.0, 1.0], vec![1.0, 0.0], vec![1.0, 1.0]];
    /// let targets = vec![vec![0.0], vec![1.0], vec![1.0], vec![0.0]];
    ///
    /// let result = mlp.fit(&inputs, &targets, 1000, true).unwrap();
    /// println!("Final loss: {:.6}", result.final_loss);
    /// ```
    pub fn fit(
        &mut self,
        inputs: &[Vec<f64>],
        targets: &[Vec<f64>],
        epochs: usize,
        verbose: bool,
    ) -> Result<TrainResult, String> {
        if inputs.len() != targets.len() {
            return Err("inputs and targets must have same length".to_string());
        }

        let mut losses = Vec::with_capacity(epochs);

        for epoch in 0..epochs {
            let mut epoch_loss = 0.0;

            for (input, target) in inputs.iter().zip(targets.iter()) {
                self.train(input, target)?;
                let output = self.predict(input)?;
                epoch_loss += self.compute_loss(&output, target);
            }

            epoch_loss /= inputs.len() as f64;
            losses.push(epoch_loss);

            if verbose && (epoch % 100 == 0 || epoch == epochs - 1) {
                println!("Epoch {}/{} - Loss: {:.6}", epoch + 1, epochs, epoch_loss);
            }
        }

        let final_loss = losses.last().copied().unwrap_or(0.0);
        Ok(TrainResult { losses, final_loss })
    }

    /// Compute loss between output and target.
    pub fn compute_loss(&self, output: &[f64], target: &[f64]) -> f64 {
        self.inner.ComputeLoss(&output.to_vec(), &target.to_vec())
    }

    /// Save model to JSON file.
    ///
    /// # Example
    /// ```no_run
    /// use glassboxai_mlp::MLP;
    ///
    /// let mlp = MLP::new(2, &[4], 1).unwrap();
    /// mlp.save("model.json").unwrap();
    /// ```
    pub fn save(&self, filename: &str) -> Result<(), String> {
        self.inner.Save(filename)
    }

    /// Load model from JSON file.
    ///
    /// # Example
    /// ```no_run
    /// use glassboxai_mlp::MLP;
    ///
    /// let mlp = MLP::load("model.json").unwrap();
    /// ```
    pub fn load(filename: &str) -> Result<Self, String> {
        let inner = TMultiLayerPerceptronCUDA::Load(filename)?;
        Ok(Self { inner })
    }

    // ========== Properties ==========

    /// Get the learning rate.
    pub fn learning_rate(&self) -> f64 {
        self.inner.LearningRate
    }

    /// Set the learning rate.
    pub fn set_learning_rate(&mut self, value: f64) {
        self.inner.LearningRate = value;
    }

    /// Get the optimizer type.
    pub fn optimizer(&self) -> OptimizerType {
        optimizer_from_internal(self.inner.Optimizer)
    }

    /// Set the optimizer type.
    pub fn set_optimizer(&mut self, optimizer: OptimizerType) {
        self.inner.Optimizer = optimizer_to_internal(optimizer);
    }

    /// Get the dropout rate.
    pub fn dropout_rate(&self) -> f64 {
        self.inner.DropoutRate
    }

    /// Set the dropout rate.
    pub fn set_dropout_rate(&mut self, value: f64) {
        self.inner.DropoutRate = value;
    }

    /// Get the L2 regularization lambda.
    pub fn l2_lambda(&self) -> f64 {
        self.inner.L2Lambda
    }

    /// Set the L2 regularization lambda.
    pub fn set_l2_lambda(&mut self, value: f64) {
        self.inner.L2Lambda = value;
    }

    /// Get batch normalization state.
    pub fn batch_norm(&self) -> bool {
        self.inner.UseBatchNorm
    }

    /// Set batch normalization state.
    pub fn set_batch_norm(&mut self, value: bool) {
        self.inner.UseBatchNorm = value;
    }

    /// Get the current GPU backend.
    pub fn backend(&self) -> BackendType {
        backend_from_internal(self.inner.get_backend())
    }

    /// Set the GPU backend.
    pub fn set_backend(&mut self, backend: BackendType) -> Result<(), String> {
        self.inner.set_backend(backend_to_internal(backend))
    }

    /// Get the input size.
    pub fn input_size(&self) -> usize {
        self.inner.GetInputSize() as usize
    }

    /// Get the output size.
    pub fn output_size(&self) -> usize {
        self.inner.GetOutputSize() as usize
    }

    /// Get the hidden layer sizes.
    pub fn hidden_sizes(&self) -> Vec<usize> {
        self.inner.GetHiddenSizes().iter().map(|&s| s as usize).collect()
    }

    /// Get the number of layers (including input and output).
    pub fn num_layers(&self) -> usize {
        self.inner.GetNumLayers() as usize
    }

    /// Get the current timestep (for Adam optimizer).
    pub fn timestep(&self) -> i32 {
        self.inner.Timestep
    }

    // ========== Introspection (Facade) ==========

    /// Get information about a layer.
    ///
    /// # Arguments
    /// * `layer` - Layer index (0 = input, 1+ = hidden/output)
    pub fn layer_info(&self, layer: usize) -> LayerInfo {
        let size = self.inner.GetLayerSize(layer) as usize;
        let activation = activation_from_internal(self.inner.GetLayerActivation(layer as i32));
        let weights_per_neuron = if layer == 0 {
            0
        } else {
            self.inner.GetLayerSize(layer - 1) as usize
        };

        LayerInfo {
            index: layer,
            size,
            activation,
            weights_per_neuron,
        }
    }

    /// Get a view of a specific neuron.
    ///
    /// # Arguments
    /// * `layer` - Layer index (1+ for hidden/output, 0 = input has no weights)
    /// * `neuron` - Neuron index within the layer
    pub fn neuron_view(&self, layer: usize, neuron: usize) -> NeuronView {
        let weights = self.inner.GetNeuronWeights(layer as i32, neuron as i32);
        let bias = self.inner.GetNeuronBias(layer as i32, neuron as i32);
        let outputs = self.inner.GetLayerOutputs(layer as i32);
        let errors = self.inner.GetLayerErrors(layer as i32);

        NeuronView {
            layer,
            index: neuron,
            weights,
            bias,
            output: outputs.get(neuron).copied().unwrap_or(0.0),
            error: errors.get(neuron).copied().unwrap_or(0.0),
        }
    }

    /// Get weights for a specific neuron.
    pub fn get_weights(&self, layer: usize, neuron: usize) -> Vec<f64> {
        self.inner.GetNeuronWeights(layer as i32, neuron as i32)
    }

    /// Set a specific weight.
    pub fn set_weight(&mut self, layer: usize, neuron: usize, weight_idx: usize, value: f64) {
        self.inner.SetNeuronWeight(layer as i32, neuron as i32, weight_idx as i32, value);
    }

    /// Get bias for a specific neuron.
    pub fn get_bias(&self, layer: usize, neuron: usize) -> f64 {
        self.inner.GetNeuronBias(layer as i32, neuron as i32)
    }

    /// Set bias for a specific neuron.
    pub fn set_bias(&mut self, layer: usize, neuron: usize, value: f64) {
        self.inner.SetNeuronBias(layer as i32, neuron as i32, value);
    }

    /// Get layer outputs (after a prediction).
    pub fn layer_outputs(&self, layer: usize) -> Vec<f64> {
        self.inner.GetLayerOutputs(layer as i32)
    }

    /// Get layer errors/gradients (after training).
    pub fn layer_errors(&self, layer: usize) -> Vec<f64> {
        self.inner.GetLayerErrors(layer as i32)
    }

    /// Get Adam optimizer's first moment (M) for a weight.
    pub fn get_weight_m(&self, layer: usize, neuron: usize, weight_idx: usize) -> f64 {
        self.inner.GetWeightM(layer as i32, neuron as i32, weight_idx as i32)
    }

    /// Get Adam optimizer's second moment (V) for a weight.
    pub fn get_weight_v(&self, layer: usize, neuron: usize, weight_idx: usize) -> f64 {
        self.inner.GetWeightV(layer as i32, neuron as i32, weight_idx as i32)
    }

    /// Get Adam optimizer's first moment (M) for a bias.
    pub fn get_bias_m(&self, layer: usize, neuron: usize) -> f64 {
        self.inner.GetBiasM(layer as i32, neuron as i32)
    }

    /// Get Adam optimizer's second moment (V) for a bias.
    pub fn get_bias_v(&self, layer: usize, neuron: usize) -> f64 {
        self.inner.GetBiasV(layer as i32, neuron as i32)
    }

    /// Get activation histogram for a layer.
    pub fn activation_histogram(&self, layer: usize, bins: usize) -> Vec<i32> {
        self.inner.GetActivationHistogram(layer as i32, bins)
    }

    /// Get gradient histogram for a layer.
    pub fn gradient_histogram(&self, layer: usize, bins: usize) -> Vec<i32> {
        self.inner.GetGradientHistogram(layer as i32, bins)
    }

    /// Compute feature importance based on input layer weights.
    ///
    /// Returns features sorted by importance (highest first).
    pub fn feature_importance(&self) -> Vec<FeatureImportance> {
        self.inner
            .compute_feature_importance()
            .into_iter()
            .map(|(index, score)| FeatureImportance { index, score })
            .collect()
    }

    /// Get access to the internal MLP (for advanced use).
    pub fn inner(&self) -> &TMultiLayerPerceptronCUDA {
        &self.inner
    }

    /// Get mutable access to the internal MLP (for advanced use).
    pub fn inner_mut(&mut self) -> &mut TMultiLayerPerceptronCUDA {
        &mut self.inner
    }
}

impl std::fmt::Debug for MLP {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MLP")
            .field("input_size", &self.input_size())
            .field("hidden_sizes", &self.hidden_sizes())
            .field("output_size", &self.output_size())
            .field("learning_rate", &self.learning_rate())
            .field("optimizer", &self.optimizer())
            .field("backend", &self.backend())
            .finish()
    }
}

impl std::fmt::Display for MLP {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "MLP(input={}, hidden={:?}, output={}, lr={:.4}, optimizer={}, backend={})",
            self.input_size(),
            self.hidden_sizes(),
            self.output_size(),
            self.learning_rate(),
            self.optimizer(),
            self.backend()
        )
    }
}

// ========== Utility functions ==========

/// Detect available GPU backends.
pub fn available_backends() -> Vec<BackendType> {
    crate::gpu_backend::detect_available_backends()
        .into_iter()
        .map(backend_from_internal)
        .collect()
}

/// Select the best available backend.
pub fn select_best_backend() -> BackendType {
    backend_from_internal(crate::gpu_backend::select_best_backend())
}

// ========== Internal conversions ==========

fn activation_to_internal(act: ActivationType) -> TActivationType {
    match act {
        ActivationType::Sigmoid => TActivationType::atSigmoid,
        ActivationType::Tanh => TActivationType::atTanh,
        ActivationType::ReLU => TActivationType::atReLU,
        ActivationType::Softmax => TActivationType::atSoftmax,
    }
}

fn activation_from_internal(act: TActivationType) -> ActivationType {
    match act {
        TActivationType::atSigmoid => ActivationType::Sigmoid,
        TActivationType::atTanh => ActivationType::Tanh,
        TActivationType::atReLU => ActivationType::ReLU,
        TActivationType::atSoftmax => ActivationType::Softmax,
    }
}

fn optimizer_to_internal(opt: OptimizerType) -> TOptimizerType {
    match opt {
        OptimizerType::SGD => TOptimizerType::otSGD,
        OptimizerType::Adam => TOptimizerType::otAdam,
        OptimizerType::RMSProp => TOptimizerType::otRMSProp,
    }
}

fn optimizer_from_internal(opt: TOptimizerType) -> OptimizerType {
    match opt {
        TOptimizerType::otSGD => OptimizerType::SGD,
        TOptimizerType::otAdam => OptimizerType::Adam,
        TOptimizerType::otRMSProp => OptimizerType::RMSProp,
    }
}

fn backend_to_internal(backend: BackendType) -> TGPUBackend {
    match backend {
        BackendType::CPU => TGPUBackend::CPU,
        BackendType::CUDA => TGPUBackend::CUDA,
        BackendType::OpenCL => TGPUBackend::OpenCL,
    }
}

fn backend_from_internal(backend: TGPUBackend) -> BackendType {
    match backend {
        TGPUBackend::CPU => BackendType::CPU,
        TGPUBackend::CUDA => BackendType::CUDA,
        TGPUBackend::OpenCL => BackendType::OpenCL,
    }
}
