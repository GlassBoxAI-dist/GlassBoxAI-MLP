//! @file
//! @ingroup MLP_Wrappers
/*
 * MIT License
 * Copyright (c) 2025 Matthew Abbott
 */

use napi::bindgen_prelude::*;
use napi_derive::napi;
use crate::mlp::*;
use crate::gpu_backend::*;

/// Activation function types
#[napi]
pub enum JsActivationType {
    Sigmoid = 0,
    Tanh = 1,
    ReLU = 2,
    Softmax = 3,
}

impl From<JsActivationType> for TActivationType {
    fn from(val: JsActivationType) -> Self {
        match val {
            JsActivationType::Sigmoid => TActivationType::atSigmoid,
            JsActivationType::Tanh => TActivationType::atTanh,
            JsActivationType::ReLU => TActivationType::atReLU,
            JsActivationType::Softmax => TActivationType::atSoftmax,
        }
    }
}

impl From<TActivationType> for JsActivationType {
    fn from(val: TActivationType) -> Self {
        match val {
            TActivationType::atSigmoid => JsActivationType::Sigmoid,
            TActivationType::atTanh => JsActivationType::Tanh,
            TActivationType::atReLU => JsActivationType::ReLU,
            TActivationType::atSoftmax => JsActivationType::Softmax,
        }
    }
}

/// Optimizer types
#[napi]
pub enum JsOptimizerType {
    SGD = 0,
    Adam = 1,
    RMSProp = 2,
}

impl From<JsOptimizerType> for TOptimizerType {
    fn from(val: JsOptimizerType) -> Self {
        match val {
            JsOptimizerType::SGD => TOptimizerType::otSGD,
            JsOptimizerType::Adam => TOptimizerType::otAdam,
            JsOptimizerType::RMSProp => TOptimizerType::otRMSProp,
        }
    }
}

impl From<TOptimizerType> for JsOptimizerType {
    fn from(val: TOptimizerType) -> Self {
        match val {
            TOptimizerType::otSGD => JsOptimizerType::SGD,
            TOptimizerType::otAdam => JsOptimizerType::Adam,
            TOptimizerType::otRMSProp => JsOptimizerType::RMSProp,
        }
    }
}

/// Training result containing loss history
#[napi(object)]
pub struct TrainResult {
    pub losses: Vec<f64>,
    pub final_loss: f64,
}

/// Feature importance result
#[napi(object)]
pub struct FeatureImportance {
    pub feature_index: u32,
    pub score: f64,
}

/// Model information
#[napi(object)]
pub struct ModelInfo {
    pub input_size: i32,
    pub output_size: i32,
    pub hidden_sizes: Vec<i32>,
    pub num_layers: i32,
    pub learning_rate: f64,
    pub optimizer: String,
    pub gpu_backend: String,
}

/// CSV data result
#[napi(object)]
pub struct CsvData {
    pub inputs: Vec<Vec<f64>>,
    pub targets: Vec<Vec<f64>>,
}

/// MLP creation options
#[napi(object)]
pub struct MlpOptions {
    pub hidden_activation: Option<JsActivationType>,
    pub output_activation: Option<JsActivationType>,
    pub gpu_backend: Option<String>,
    pub learning_rate: Option<f64>,
    pub optimizer: Option<JsOptimizerType>,
    pub dropout_rate: Option<f64>,
    pub l2_lambda: Option<f64>,
    pub batch_norm: Option<bool>,
}

/// Multi-Layer Perceptron with GPU acceleration
#[napi]
pub struct MLP {
    inner: TMultiLayerPerceptronCUDA,
}

#[napi]
impl MLP {
    /// Create a new MLP model
    ///
    /// @param inputSize - Number of input neurons
    /// @param hiddenSizes - Array of hidden layer sizes
    /// @param outputSize - Number of output neurons
    /// @param options - Optional configuration (activation, backend, etc.)
    #[napi(constructor)]
    pub fn new(
        input_size: i32,
        hidden_sizes: Vec<i32>,
        output_size: i32,
        options: Option<MlpOptions>,
    ) -> Result<Self> {
        let opts = options.unwrap_or(MlpOptions {
            hidden_activation: None,
            output_activation: None,
            gpu_backend: None,
            learning_rate: None,
            optimizer: None,
            dropout_rate: None,
            l2_lambda: None,
            batch_norm: None,
        });

        let hidden_act = opts.hidden_activation
            .map(|a| a.into())
            .unwrap_or(TActivationType::atSigmoid);
        let output_act = opts.output_activation
            .map(|a| a.into())
            .unwrap_or(TActivationType::atSigmoid);

        let backend = match opts.gpu_backend.as_deref() {
            Some("cuda") => TGPUBackend::CUDA,
            Some("opencl") | Some("ocl") => TGPUBackend::OpenCL,
            Some("cpu") => TGPUBackend::CPU,
            Some("auto") | None => select_best_backend(),
            Some(other) => return Err(Error::new(
                Status::InvalidArg,
                format!("Invalid GPU backend: {}", other),
            )),
        };

        let mut mlp = TMultiLayerPerceptronCUDA::new_with_backend(
            input_size,
            &hidden_sizes,
            output_size,
            hidden_act,
            output_act,
            backend,
        ).map_err(|e| Error::new(Status::GenericFailure, e))?;

        if let Some(lr) = opts.learning_rate {
            mlp.LearningRate = lr;
        }
        if let Some(opt) = opts.optimizer {
            mlp.Optimizer = opt.into();
        }
        if let Some(dr) = opts.dropout_rate {
            mlp.DropoutRate = dr;
        }
        if let Some(l2) = opts.l2_lambda {
            mlp.L2Lambda = l2;
        }
        if let Some(bn) = opts.batch_norm {
            mlp.UseBatchNorm = bn;
        }

        Ok(MLP { inner: mlp })
    }

    /// Train on a single sample
    #[napi]
    pub fn train(&mut self, input: Vec<f64>, target: Vec<f64>) -> Result<()> {
        self.inner.Train(&input, &target)
            .map_err(|e| Error::new(Status::GenericFailure, e))
    }

    /// Make a prediction
    #[napi]
    pub fn predict(&mut self, input: Vec<f64>) -> Result<Vec<f64>> {
        self.inner.Predict(&input)
            .map_err(|e| Error::new(Status::GenericFailure, e))
    }

    /// Train on a dataset for multiple epochs
    #[napi]
    pub fn fit(
        &mut self,
        inputs: Vec<Vec<f64>>,
        targets: Vec<Vec<f64>>,
        epochs: Option<u32>,
        verbose: Option<bool>,
    ) -> Result<TrainResult> {
        let epochs = epochs.unwrap_or(100) as usize;
        let verbose = verbose.unwrap_or(false);

        if inputs.len() != targets.len() {
            return Err(Error::new(
                Status::InvalidArg,
                "inputs and targets must have same length",
            ));
        }

        let mut losses = Vec::with_capacity(epochs);

        for epoch in 0..epochs {
            let mut epoch_loss = 0.0;

            for (input, target) in inputs.iter().zip(targets.iter()) {
                self.inner.Train(input, target)
                    .map_err(|e| Error::new(Status::GenericFailure, e))?;
                let output = self.inner.Predict(input)
                    .map_err(|e| Error::new(Status::GenericFailure, e))?;
                epoch_loss += self.inner.ComputeLoss(&output, target);
            }

            epoch_loss /= inputs.len() as f64;
            losses.push(epoch_loss);

            if verbose && (epoch % 100 == 0 || epoch == epochs - 1) {
                println!("Epoch {}/{} - Loss: {:.6}", epoch + 1, epochs, epoch_loss);
            }
        }

        let final_loss = *losses.last().unwrap_or(&0.0);
        Ok(TrainResult { losses, final_loss })
    }

    /// Predict on multiple samples
    #[napi]
    pub fn predict_batch(&mut self, inputs: Vec<Vec<f64>>) -> Result<Vec<Vec<f64>>> {
        inputs.iter()
            .map(|input| self.inner.Predict(input)
                .map_err(|e| Error::new(Status::GenericFailure, e)))
            .collect()
    }

    /// Save model to file
    #[napi]
    pub fn save(&self, filename: String) -> Result<()> {
        self.inner.Save(&filename)
            .map_err(|e| Error::new(Status::GenericFailure, e))
    }

    /// Load model from file
    #[napi(factory)]
    pub fn load(filename: String) -> Result<Self> {
        let mlp = TMultiLayerPerceptronCUDA::Load(&filename)
            .map_err(|e| Error::new(Status::GenericFailure, e))?;
        Ok(MLP { inner: mlp })
    }

    /// Export model to ONNX format
    #[napi]
    pub fn export_onnx(&self, filename: String) -> Result<()> {
        self.inner.export_to_onnx(&filename)
            .map_err(|e| Error::new(Status::GenericFailure, e))
    }

    /// Import model from ONNX format
    #[napi(factory)]
    pub fn import_onnx(filename: String) -> Result<Self> {
        let mlp = TMultiLayerPerceptronCUDA::import_from_onnx(&filename)
            .map_err(|e| Error::new(Status::GenericFailure, e))?;
        Ok(MLP { inner: mlp })
    }

    /// Compute feature importance
    #[napi]
    pub fn feature_importance(&self) -> Vec<FeatureImportance> {
        self.inner.compute_feature_importance()
            .into_iter()
            .map(|(idx, score)| FeatureImportance {
                feature_index: idx as u32,
                score,
            })
            .collect()
    }

    /// Get learning rate
    #[napi(getter)]
    pub fn learning_rate(&self) -> f64 {
        self.inner.LearningRate
    }

    /// Set learning rate
    #[napi(setter)]
    pub fn set_learning_rate(&mut self, value: f64) {
        self.inner.LearningRate = value;
    }

    /// Get optimizer type
    #[napi(getter)]
    pub fn optimizer(&self) -> JsOptimizerType {
        self.inner.Optimizer.into()
    }

    /// Set optimizer type
    #[napi(setter)]
    pub fn set_optimizer(&mut self, value: JsOptimizerType) {
        self.inner.Optimizer = value.into();
    }

    /// Get dropout rate
    #[napi(getter)]
    pub fn dropout_rate(&self) -> f64 {
        self.inner.DropoutRate
    }

    /// Set dropout rate
    #[napi(setter)]
    pub fn set_dropout_rate(&mut self, value: f64) {
        self.inner.DropoutRate = value;
    }

    /// Get L2 regularization lambda
    #[napi(getter)]
    pub fn l2_lambda(&self) -> f64 {
        self.inner.L2Lambda
    }

    /// Set L2 regularization lambda
    #[napi(setter)]
    pub fn set_l2_lambda(&mut self, value: f64) {
        self.inner.L2Lambda = value;
    }

    /// Get batch normalization flag
    #[napi(getter)]
    pub fn batch_norm(&self) -> bool {
        self.inner.UseBatchNorm
    }

    /// Set batch normalization flag
    #[napi(setter)]
    pub fn set_batch_norm(&mut self, value: bool) {
        self.inner.UseBatchNorm = value;
    }

    /// Get current GPU backend
    #[napi(getter)]
    pub fn gpu_backend(&self) -> String {
        self.inner.get_backend().to_str().to_string()
    }

    /// Set GPU backend
    #[napi]
    pub fn set_backend(&mut self, backend: String) -> Result<()> {
        let backend_enum = TGPUBackend::from_str(&backend);
        self.inner.set_backend(backend_enum)
            .map_err(|e| Error::new(Status::GenericFailure, e))
    }

    /// Get list of available GPU backends
    #[napi]
    pub fn available_backends() -> Vec<String> {
        detect_available_backends().iter().map(|b| b.to_str().to_string()).collect()
    }

    /// Get input size
    #[napi(getter)]
    pub fn input_size(&self) -> i32 {
        self.inner.GetInputSize()
    }

    /// Get output size
    #[napi(getter)]
    pub fn output_size(&self) -> i32 {
        self.inner.GetOutputSize()
    }

    /// Get hidden layer sizes
    #[napi(getter)]
    pub fn hidden_sizes(&self) -> Vec<i32> {
        self.inner.GetHiddenSizes().clone()
    }

    /// Get number of layers
    #[napi(getter)]
    pub fn num_layers(&self) -> i32 {
        self.inner.GetNumLayers()
    }

    /// Get weights for a specific neuron
    #[napi]
    pub fn get_neuron_weights(&self, layer: i32, neuron: i32) -> Vec<f64> {
        self.inner.GetNeuronWeights(layer, neuron)
    }

    /// Get bias for a specific neuron
    #[napi]
    pub fn get_neuron_bias(&self, layer: i32, neuron: i32) -> f64 {
        self.inner.GetNeuronBias(layer, neuron)
    }

    /// Set a specific weight
    #[napi]
    pub fn set_neuron_weight(&mut self, layer: i32, neuron: i32, weight_idx: i32, value: f64) {
        self.inner.SetNeuronWeight(layer, neuron, weight_idx, value);
    }

    /// Set a neuron's bias
    #[napi]
    pub fn set_neuron_bias(&mut self, layer: i32, neuron: i32, value: f64) {
        self.inner.SetNeuronBias(layer, neuron, value);
    }

    /// Get layer outputs
    #[napi]
    pub fn get_layer_outputs(&mut self, layer: i32) -> Vec<f64> {
        self.inner.GetLayerOutputs(layer)
    }

    /// Get model info
    #[napi]
    pub fn info(&self) -> ModelInfo {
        ModelInfo {
            input_size: self.inner.GetInputSize(),
            output_size: self.inner.GetOutputSize(),
            hidden_sizes: self.inner.GetHiddenSizes().clone(),
            num_layers: self.inner.GetNumLayers(),
            learning_rate: self.inner.LearningRate,
            optimizer: OptimizerToStr(self.inner.Optimizer).to_string(),
            gpu_backend: self.inner.get_backend().to_str().to_string(),
        }
    }

    /// String representation
    #[napi]
    pub fn to_string(&self) -> String {
        format!(
            "MLP(input={}, hidden={:?}, output={}, lr={:.4}, optimizer={}, backend={})",
            self.inner.GetInputSize(),
            self.inner.GetHiddenSizes(),
            self.inner.GetOutputSize(),
            self.inner.LearningRate,
            OptimizerToStr(self.inner.Optimizer),
            self.inner.get_backend().to_str()
        )
    }
}

/// Load dataset from CSV file
#[napi]
pub fn load_csv(filename: String, input_size: i32, output_size: i32) -> Result<CsvData> {
    let data = LoadDataCSV(&filename, input_size, output_size);

    if data.is_empty() {
        return Err(Error::new(Status::GenericFailure, "No data loaded from CSV"));
    }

    let inputs: Vec<Vec<f64>> = data.iter().map(|dp| dp.Input.clone()).collect();
    let targets: Vec<Vec<f64>> = data.iter().map(|dp| dp.Target.clone()).collect();

    Ok(CsvData { inputs, targets })
}

/// Normalize data (zero mean, unit variance)
#[napi]
pub fn normalize(inputs: Vec<Vec<f64>>) -> Vec<Vec<f64>> {
    if inputs.is_empty() {
        return inputs;
    }

    let mut data: Vec<TDataPoint> = inputs.iter().map(|input| {
        TDataPoint {
            Input: input.clone(),
            Target: vec![],
        }
    }).collect();

    NormalizeData(&mut data);

    data.iter().map(|dp| dp.Input.clone()).collect()
}

