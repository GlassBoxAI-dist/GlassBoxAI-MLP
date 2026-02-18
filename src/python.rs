//! @file
//! @ingroup MLP_Internal_Logic
/*
 * MIT License
 * Copyright (c) 2025 Matthew Abbott
 */

use pyo3::prelude::*;
use pyo3::exceptions::{PyValueError, PyIOError, PyRuntimeError};
use crate::mlp::*;
use crate::gpu_backend::*;

/// Convert Rust error to Python exception
fn to_py_err(err: String) -> PyErr {
    PyRuntimeError::new_err(err)
}

/// Activation function types
#[pyclass(eq, eq_int)]
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum PyActivationType {
    Sigmoid = 0,
    Tanh = 1,
    ReLU = 2,
    Softmax = 3,
}

impl From<PyActivationType> for TActivationType {
    fn from(val: PyActivationType) -> Self {
        match val {
            PyActivationType::Sigmoid => TActivationType::atSigmoid,
            PyActivationType::Tanh => TActivationType::atTanh,
            PyActivationType::ReLU => TActivationType::atReLU,
            PyActivationType::Softmax => TActivationType::atSoftmax,
        }
    }
}

impl From<TActivationType> for PyActivationType {
    fn from(val: TActivationType) -> Self {
        match val {
            TActivationType::atSigmoid => PyActivationType::Sigmoid,
            TActivationType::atTanh => PyActivationType::Tanh,
            TActivationType::atReLU => PyActivationType::ReLU,
            TActivationType::atSoftmax => PyActivationType::Softmax,
        }
    }
}

/// Optimizer types
#[pyclass(eq, eq_int)]
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum PyOptimizerType {
    SGD = 0,
    Adam = 1,
    RMSProp = 2,
}

impl From<PyOptimizerType> for TOptimizerType {
    fn from(val: PyOptimizerType) -> Self {
        match val {
            PyOptimizerType::SGD => TOptimizerType::otSGD,
            PyOptimizerType::Adam => TOptimizerType::otAdam,
            PyOptimizerType::RMSProp => TOptimizerType::otRMSProp,
        }
    }
}

impl From<TOptimizerType> for PyOptimizerType {
    fn from(val: TOptimizerType) -> Self {
        match val {
            TOptimizerType::otSGD => PyOptimizerType::SGD,
            TOptimizerType::otAdam => PyOptimizerType::Adam,
            TOptimizerType::otRMSProp => PyOptimizerType::RMSProp,
        }
    }
}

/// Multi-Layer Perceptron with GPU acceleration
#[pyclass(name = "MLP")]
pub struct PyMLP {
    inner: TMultiLayerPerceptronCUDA,
}

#[pymethods]
impl PyMLP {
    /// Create a new MLP model
    ///
    /// Args:
    ///     input_size (int): Number of input neurons
    ///     hidden_sizes (list[int]): List of hidden layer sizes
    ///     output_size (int): Number of output neurons
    ///     hidden_activation (PyActivationType): Activation for hidden layers (default: Sigmoid)
    ///     output_activation (PyActivationType): Activation for output layer (default: Sigmoid)
    ///     gpu_backend (str): GPU backend: "auto", "cpu", "cuda", "opencl" (default: "auto")
    ///
    /// Returns:
    ///     MLP: A new MLP instance
    ///
    /// Example:
    ///     >>> mlp = MLP(2, [8, 8], 1)
    ///     >>> mlp = MLP(4, [16], 3, hidden_activation=PyActivationType.ReLU, gpu_backend="cuda")
    #[new]
    #[pyo3(signature = (input_size, hidden_sizes, output_size, hidden_activation=PyActivationType::Sigmoid, output_activation=PyActivationType::Sigmoid, gpu_backend=None))]
    fn new(
        input_size: i32,
        hidden_sizes: Vec<i32>,
        output_size: i32,
        hidden_activation: PyActivationType,
        output_activation: PyActivationType,
        gpu_backend: Option<String>,
    ) -> PyResult<Self> {
        let backend = match gpu_backend.as_deref() {
            Some("cuda") => TGPUBackend::CUDA,
            Some("opencl") | Some("ocl") => TGPUBackend::OpenCL,
            Some("cpu") => TGPUBackend::CPU,
            Some("auto") | None => select_best_backend(),
            Some(other) => return Err(PyValueError::new_err(format!("Invalid GPU backend: {}", other))),
        };
        
        let mlp = TMultiLayerPerceptronCUDA::new_with_backend(
            input_size,
            &hidden_sizes,
            output_size,
            hidden_activation.into(),
            output_activation.into(),
            backend,
        ).map_err(to_py_err)?;
        
        Ok(PyMLP { inner: mlp })
    }

    /// Train the model on a single sample
    ///
    /// Args:
    ///     input (list[float]): Input values
    ///     target (list[float]): Target output values
    ///
    /// Example:
    ///     >>> mlp.train([1.0, 0.0], [1.0])
    fn train(&mut self, input: Vec<f64>, target: Vec<f64>) -> PyResult<()> {
        self.inner.Train(&input, &target).map_err(to_py_err)
    }

    /// Make a prediction
    ///
    /// Args:
    ///     input (list[float]): Input values
    ///
    /// Returns:
    ///     list[float]: Model output
    ///
    /// Example:
    ///     >>> output = mlp.predict([1.0, 0.0])
    fn predict(&mut self, input: Vec<f64>) -> PyResult<Vec<f64>> {
        self.inner.Predict(&input).map_err(to_py_err)
    }

    /// Train on multiple epochs with a dataset
    ///
    /// Args:
    ///     inputs (list[list[float]]): List of input samples
    ///     targets (list[list[float]]): List of target outputs
    ///     epochs (int): Number of training epochs (default: 100)
    ///     verbose (bool): Print progress (default: False)
    ///
    /// Returns:
    ///     list[float]: Loss per epoch
    ///
    /// Example:
    ///     >>> X = [[0,0], [0,1], [1,0], [1,1]]
    ///     >>> y = [[0], [1], [1], [0]]
    ///     >>> losses = mlp.fit(X, y, epochs=1000, verbose=True)
    #[pyo3(signature = (inputs, targets, epochs=100, verbose=false))]
    fn fit(
        &mut self,
        inputs: Vec<Vec<f64>>,
        targets: Vec<Vec<f64>>,
        epochs: usize,
        verbose: bool,
    ) -> PyResult<Vec<f64>> {
        if inputs.len() != targets.len() {
            return Err(PyValueError::new_err("inputs and targets must have same length"));
        }

        let mut losses = Vec::with_capacity(epochs);

        for epoch in 0..epochs {
            let mut epoch_loss = 0.0;
            
            for (input, target) in inputs.iter().zip(targets.iter()) {
                self.inner.Train(input, target).map_err(to_py_err)?;
                let output = self.inner.Predict(input).map_err(to_py_err)?;
                epoch_loss += self.inner.ComputeLoss(&output, target);
            }
            
            epoch_loss /= inputs.len() as f64;
            losses.push(epoch_loss);
            
            if verbose && (epoch % 100 == 0 || epoch == epochs - 1) {
                println!("Epoch {}/{} - Loss: {:.6}", epoch + 1, epochs, epoch_loss);
            }
        }

        Ok(losses)
    }

    /// Predict on multiple samples
    ///
    /// Args:
    ///     inputs (list[list[float]]): List of input samples
    ///
    /// Returns:
    ///     list[list[float]]: List of predictions
    fn predict_batch(&mut self, inputs: Vec<Vec<f64>>) -> PyResult<Vec<Vec<f64>>> {
        inputs.iter()
            .map(|input| self.inner.Predict(input).map_err(to_py_err))
            .collect()
    }

    /// Save model to file
    ///
    /// Args:
    ///     filename (str): Path to save the model
    fn save(&self, filename: &str) -> PyResult<()> {
        self.inner.Save(filename).map_err(to_py_err)
    }

    /// Load model from file
    ///
    /// Args:
    ///     filename (str): Path to the model file
    ///
    /// Returns:
    ///     MLP: Loaded model
    #[staticmethod]
    fn load(filename: &str) -> PyResult<Self> {
        let mlp = TMultiLayerPerceptronCUDA::Load(filename).map_err(to_py_err)?;
        Ok(PyMLP { inner: mlp })
    }

    /// Export model to ONNX format
    ///
    /// Args:
    ///     filename (str): Path to save ONNX file
    fn export_onnx(&self, filename: &str) -> PyResult<()> {
        self.inner.export_to_onnx(filename).map_err(to_py_err)
    }

    /// Import model from ONNX format
    ///
    /// Args:
    ///     filename (str): Path to ONNX file
    ///
    /// Returns:
    ///     MLP: Loaded model
    #[staticmethod]
    fn import_onnx(filename: &str) -> PyResult<Self> {
        let mlp = TMultiLayerPerceptronCUDA::import_from_onnx(filename).map_err(to_py_err)?;
        Ok(PyMLP { inner: mlp })
    }

    /// Compute feature importance
    ///
    /// Returns:
    ///     list[tuple[int, float]]: List of (feature_index, importance_score) tuples
    fn feature_importance(&self) -> PyResult<Vec<(usize, f64)>> {
        Ok(self.inner.compute_feature_importance())
    }

    /// Get learning rate
    #[getter]
    fn learning_rate(&self) -> f64 {
        self.inner.LearningRate
    }

    /// Set learning rate
    #[setter]
    fn set_learning_rate(&mut self, value: f64) {
        self.inner.LearningRate = value;
    }

    /// Get optimizer type
    #[getter]
    fn optimizer(&self) -> PyOptimizerType {
        self.inner.Optimizer.into()
    }

    /// Set optimizer type
    #[setter]
    fn set_optimizer(&mut self, value: PyOptimizerType) {
        self.inner.Optimizer = value.into();
    }

    /// Get dropout rate
    #[getter]
    fn dropout_rate(&self) -> f64 {
        self.inner.DropoutRate
    }

    /// Set dropout rate
    #[setter]
    fn set_dropout_rate(&mut self, value: f64) {
        self.inner.DropoutRate = value;
    }

    /// Get L2 regularization lambda
    #[getter]
    fn l2_lambda(&self) -> f64 {
        self.inner.L2Lambda
    }

    /// Set L2 regularization lambda
    #[setter]
    fn set_l2_lambda(&mut self, value: f64) {
        self.inner.L2Lambda = value;
    }

    /// Get batch normalization flag
    #[getter]
    fn batch_norm(&self) -> bool {
        self.inner.UseBatchNorm
    }

    /// Set batch normalization flag
    #[setter]
    fn set_batch_norm(&mut self, value: bool) {
        self.inner.UseBatchNorm = value;
    }

    /// Get current GPU backend
    #[getter]
    fn gpu_backend(&self) -> String {
        self.inner.get_backend().to_str().to_string()
    }

    /// Set GPU backend
    ///
    /// Args:
    ///     backend (str): "cpu", "cuda", or "opencl"
    fn set_backend(&mut self, backend: &str) -> PyResult<()> {
        let backend_enum = TGPUBackend::from_str(backend);
        self.inner.set_backend(backend_enum).map_err(to_py_err)
    }

    /// Get list of available GPU backends
    ///
    /// Returns:
    ///     list[str]: Available backends
    #[staticmethod]
    fn available_backends() -> Vec<String> {
        detect_available_backends().iter().map(|b| b.to_str().to_string()).collect()
    }

    /// Get input size
    #[getter]
    fn input_size(&self) -> i32 {
        self.inner.GetInputSize()
    }

    /// Get output size
    #[getter]
    fn output_size(&self) -> i32 {
        self.inner.GetOutputSize()
    }

    /// Get hidden layer sizes
    #[getter]
    fn hidden_sizes(&self) -> Vec<i32> {
        self.inner.GetHiddenSizes().clone()
    }

    /// Get number of layers (including input and output)
    #[getter]
    fn num_layers(&self) -> i32 {
        self.inner.GetNumLayers()
    }

    /// Get weights for a specific neuron
    ///
    /// Args:
    ///     layer (int): Layer index
    ///     neuron (int): Neuron index
    ///
    /// Returns:
    ///     list[float]: Weights
    fn get_neuron_weights(&self, layer: i32, neuron: i32) -> Vec<f64> {
        self.inner.GetNeuronWeights(layer, neuron)
    }

    /// Get bias for a specific neuron
    ///
    /// Args:
    ///     layer (int): Layer index
    ///     neuron (int): Neuron index
    ///
    /// Returns:
    ///     float: Bias value
    fn get_neuron_bias(&self, layer: i32, neuron: i32) -> f64 {
        self.inner.GetNeuronBias(layer, neuron)
    }

    /// Set a specific weight
    ///
    /// Args:
    ///     layer (int): Layer index
    ///     neuron (int): Neuron index
    ///     weight_idx (int): Weight index
    ///     value (float): New weight value
    fn set_neuron_weight(&mut self, layer: i32, neuron: i32, weight_idx: i32, value: f64) {
        self.inner.SetNeuronWeight(layer, neuron, weight_idx, value);
    }

    /// Set a neuron's bias
    ///
    /// Args:
    ///     layer (int): Layer index
    ///     neuron (int): Neuron index
    ///     value (float): New bias value
    fn set_neuron_bias(&mut self, layer: i32, neuron: i32, value: f64) {
        self.inner.SetNeuronBias(layer, neuron, value);
    }

    /// Get layer outputs
    ///
    /// Args:
    ///     layer (int): Layer index
    ///
    /// Returns:
    ///     list[float]: Output values for all neurons in the layer
    fn get_layer_outputs(&mut self, layer: i32) -> Vec<f64> {
        self.inner.GetLayerOutputs(layer)
    }

    /// Get Adam optimizer timestep (number of training steps completed)
    ///
    /// Returns:
    ///     int: Current timestep
    #[getter]
    fn timestep(&self) -> i32 {
        self.inner.Timestep
    }

    /// Compute loss between output and target vectors
    ///
    /// Args:
    ///     output (list[float]): Model output
    ///     target (list[float]): Target values
    ///
    /// Returns:
    ///     float: Loss value
    fn compute_loss(&self, output: Vec<f64>, target: Vec<f64>) -> f64 {
        self.inner.ComputeLoss(&output, &target)
    }

    /// Get per-neuron error/gradient values after the last training step
    ///
    /// Args:
    ///     layer (int): Layer index
    ///
    /// Returns:
    ///     list[float]: Error values for all neurons in the layer
    fn get_layer_errors(&mut self, layer: i32) -> Vec<f64> {
        self.inner.GetLayerErrors(layer)
    }

    /// Get the number of neurons in a layer
    ///
    /// Args:
    ///     layer (int): Layer index
    ///
    /// Returns:
    ///     int: Number of neurons
    fn get_layer_size(&self, layer: i32) -> i32 {
        self.inner.GetLayerSize(layer as usize) as i32
    }

    /// Get the activation function used by a layer
    ///
    /// Args:
    ///     layer (int): Layer index
    ///
    /// Returns:
    ///     PyActivationType: Activation type enum value
    fn get_layer_activation(&self, layer: i32) -> PyActivationType {
        PyActivationType::from(self.inner.GetLayerActivation(layer))
    }

    /// Get combined layer metadata
    ///
    /// Args:
    ///     layer (int): Layer index
    ///
    /// Returns:
    ///     dict: Dictionary with keys ``index`` (int), ``size`` (int),
    ///           ``activation`` (PyActivationType)
    fn get_layer_info<'py>(&self, py: Python<'py>, layer: i32) -> PyResult<pyo3::Bound<'py, pyo3::types::PyDict>> {
        let dict = pyo3::types::PyDict::new_bound(py);
        dict.set_item("index", layer)?;
        dict.set_item("size", self.inner.GetLayerSize(layer as usize) as i32)?;
        dict.set_item("activation", PyActivationType::from(self.inner.GetLayerActivation(layer)))?;
        Ok(dict)
    }

    /// Get detailed per-neuron introspection view
    ///
    /// Args:
    ///     layer (int): Layer index
    ///     neuron (int): Neuron index within the layer
    ///
    /// Returns:
    ///     dict: Dictionary with keys ``layer``, ``neuron``, ``weights`` (list[float]),
    ///           ``bias`` (float), ``output`` (float), ``error`` (float)
    fn get_neuron_view<'py>(&mut self, py: Python<'py>, layer: i32, neuron: i32) -> PyResult<pyo3::Bound<'py, pyo3::types::PyDict>> {
        let dict = pyo3::types::PyDict::new_bound(py);
        let weights = self.inner.GetNeuronWeights(layer, neuron);
        let bias = self.inner.GetNeuronBias(layer, neuron);
        let outputs = self.inner.GetLayerOutputs(layer);
        let errors = self.inner.GetLayerErrors(layer);
        let output_val = outputs.get(neuron as usize).copied().unwrap_or(0.0);
        let error_val = errors.get(neuron as usize).copied().unwrap_or(0.0);
        dict.set_item("layer", layer)?;
        dict.set_item("neuron", neuron)?;
        dict.set_item("weights", weights)?;
        dict.set_item("bias", bias)?;
        dict.set_item("output", output_val)?;
        dict.set_item("error", error_val)?;
        Ok(dict)
    }

    /// Get Adam first moment (M) for a specific weight
    ///
    /// Args:
    ///     layer (int): Layer index
    ///     neuron (int): Neuron index
    ///     weight_idx (int): Weight index within the neuron
    ///
    /// Returns:
    ///     float: First moment estimate
    fn get_weight_m(&self, layer: i32, neuron: i32, weight_idx: i32) -> f64 {
        self.inner.GetWeightM(layer, neuron, weight_idx)
    }

    /// Get Adam second moment (V) for a specific weight
    ///
    /// Args:
    ///     layer (int): Layer index
    ///     neuron (int): Neuron index
    ///     weight_idx (int): Weight index within the neuron
    ///
    /// Returns:
    ///     float: Second moment estimate
    fn get_weight_v(&self, layer: i32, neuron: i32, weight_idx: i32) -> f64 {
        self.inner.GetWeightV(layer, neuron, weight_idx)
    }

    /// Get Adam first moment (M) for a specific bias
    ///
    /// Args:
    ///     layer (int): Layer index
    ///     neuron (int): Neuron index
    ///
    /// Returns:
    ///     float: First moment estimate
    fn get_bias_m(&self, layer: i32, neuron: i32) -> f64 {
        self.inner.GetBiasM(layer, neuron)
    }

    /// Get Adam second moment (V) for a specific bias
    ///
    /// Args:
    ///     layer (int): Layer index
    ///     neuron (int): Neuron index
    ///
    /// Returns:
    ///     float: Second moment estimate
    fn get_bias_v(&self, layer: i32, neuron: i32) -> f64 {
        self.inner.GetBiasV(layer, neuron)
    }

    /// Get a histogram of activation values across all neurons in a layer
    ///
    /// Args:
    ///     layer (int): Layer index
    ///     bins (int): Number of histogram bins
    ///
    /// Returns:
    ///     list[int]: Per-bin counts
    fn get_activation_histogram(&self, layer: i32, bins: i32) -> Vec<i32> {
        self.inner.GetActivationHistogram(layer, bins as usize)
    }

    /// Get a histogram of gradient values across all neurons in a layer
    ///
    /// Args:
    ///     layer (int): Layer index
    ///     bins (int): Number of histogram bins
    ///
    /// Returns:
    ///     list[int]: Per-bin counts
    fn get_gradient_histogram(&self, layer: i32, bins: i32) -> Vec<i32> {
        self.inner.GetGradientHistogram(layer, bins as usize)
    }

    /// Get model info as string
    fn __repr__(&self) -> String {
        format!(
            "MLP(input={}, hidden={:?}, output={}, lr={:.4}, optimizer={:?}, backend={})",
            self.inner.GetInputSize(),
            self.inner.GetHiddenSizes(),
            self.inner.GetOutputSize(),
            self.inner.LearningRate,
            OptimizerToStr(self.inner.Optimizer),
            self.inner.get_backend().to_str()
        )
    }

    /// Get model info as string
    fn __str__(&self) -> String {
        self.__repr__()
    }
}

/// Load dataset from CSV file
///
/// Args:
///     filename (str): Path to CSV file
///     input_size (int): Number of input features
///     output_size (int): Number of output features
///
/// Returns:
///     tuple[list[list[float]], list[list[float]]]: (inputs, targets)
#[pyfunction]
fn load_csv(filename: &str, input_size: i32, output_size: i32) -> PyResult<(Vec<Vec<f64>>, Vec<Vec<f64>>)> {
    let data = LoadDataCSV(filename, input_size, output_size);
    
    if data.is_empty() {
        return Err(PyIOError::new_err("No data loaded from CSV"));
    }

    let inputs: Vec<Vec<f64>> = data.iter().map(|dp| dp.Input.clone()).collect();
    let targets: Vec<Vec<f64>> = data.iter().map(|dp| dp.Target.clone()).collect();

    Ok((inputs, targets))
}

/// Normalize data (zero mean, unit variance)
///
/// Args:
///     inputs (list[list[float]]): Input data to normalize
///
/// Returns:
///     list[list[float]]: Normalized data
#[pyfunction]
fn normalize(inputs: Vec<Vec<f64>>) -> PyResult<Vec<Vec<f64>>> {
    if inputs.is_empty() {
        return Ok(inputs);
    }

    let mut data: Vec<TDataPoint> = inputs.iter().map(|input| {
        TDataPoint {
            Input: input.clone(),
            Target: vec![],
        }
    }).collect();

    NormalizeData(&mut data);

    Ok(data.iter().map(|dp| dp.Input.clone()).collect())
}

/// Python module
#[pymodule]
fn facaded_mlp_cuda(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyMLP>()?;
    m.add_class::<PyActivationType>()?;
    m.add_class::<PyOptimizerType>()?;
    m.add_function(wrap_pyfunction!(load_csv, m)?)?;
    m.add_function(wrap_pyfunction!(normalize, m)?)?;
    Ok(())
}

