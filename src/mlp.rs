//! @file
//! @ingroup MLP_Internal_Logic
/*
 * MIT License
 * Copyright (c) 2025 Matthew Abbott
 */

#![allow(non_snake_case)]
#![allow(non_camel_case_types)]
#![allow(dead_code)]
#![allow(unused_variables)]
#![allow(unused_assignments)]

use std::fs::File;
use std::io::{BufRead, BufReader, Write};
use serde::{Deserialize, Serialize};
use rand::Rng;

use crate::gpu_backend::TGPUBackend;

#[cfg(feature = "cuda")]
use cudarc::driver::CudaDevice;
#[cfg(feature = "cuda")]

#[cfg(feature = "opencl")]
use crate::opencl_mlp::TOpenCLContext as OpenCLContext;

pub type Darray = Vec<f64>;
pub type TIntArray = Vec<i32>;

#[derive(Clone, Copy, PartialEq, Eq, Debug, Serialize, Deserialize)]
pub enum TActivationType {
    atSigmoid = 0,
    atTanh = 1,
    atReLU = 2,
    atSoftmax = 3,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug, Serialize, Deserialize)]
pub enum TOptimizerType {
    otSGD = 0,
    otAdam = 1,
    otRMSProp = 2,
}

#[derive(Clone, Debug)]
pub struct TDataPoint {
    pub Input: Darray,
    pub Target: Darray,
}

#[derive(Serialize, Deserialize)]
struct TNeuronData {
    weights: Vec<f64>,
    bias: f64,
    #[serde(default)]
    m: Vec<f64>,
    #[serde(default)]
    v: Vec<f64>,
    #[serde(default)]
    m_bias: f64,
    #[serde(default)]
    v_bias: f64,
}

#[derive(Serialize, Deserialize)]
struct TLayerData {
    neurons: Vec<TNeuronData>,
    activation: i32,
}

#[derive(Serialize, Deserialize)]
struct TModelData {
    magic: String,
    version: String,
    input_size: i32,
    output_size: i32,
    hidden_sizes: Vec<i32>,
    learning_rate: f64,
    optimizer: i32,
    hidden_activation: i32,
    output_activation: i32,
    dropout_rate: f64,
    l2_lambda: f64,
    beta1: f64,
    beta2: f64,
    timestep: i32,
    use_batch_norm: bool,
    gpu_backend: String,
    input_layer: TLayerData,
    hidden_layers: Vec<TLayerData>,
    output_layer: TLayerData,
}

pub struct TNeuron {
    pub Weights: Vec<f64>,
    pub Bias: f64,
    pub Output: f64,
    pub Error: f64,
    pub M: Vec<f64>,
    pub V: Vec<f64>,
    pub MBias: f64,
    pub VBias: f64,
}

pub struct TLayer {
    pub Neurons: Vec<TNeuron>,
    pub ActivationType: TActivationType,
    pub DropoutMask: Vec<bool>,
}

pub struct TMultiLayerPerceptronCUDA {
    FInputLayer: TLayer,
    FHiddenLayers: Vec<TLayer>,
    FOutputLayer: TLayer,
    FHiddenSizes: TIntArray,
    FInputSize: i32,
    FOutputSize: i32,
    
    pub LearningRate: f64,
    pub Optimizer: TOptimizerType,
    pub HiddenActivation: TActivationType,
    pub OutputActivation: TActivationType,
    pub DropoutRate: f64,
    pub L2Lambda: f64,
    pub Beta1: f64,
    pub Beta2: f64,
    pub Timestep: i32,
    pub EnableLRDecay: bool,
    pub LRDecayRate: f64,
    pub LRDecayEpochs: i32,
    pub EnableEarlyStopping: bool,
    pub EarlyStoppingPatience: i32,
    pub UseBatchNorm: bool,
    pub GPUBackend: TGPUBackend,
    
    #[cfg(feature = "cuda")]
    cuda_device: Option<std::sync::Arc<CudaDevice>>,
    
    #[cfg(feature = "opencl")]
    opencl_context: Option<OpenCLContext>,
}

impl TMultiLayerPerceptronCUDA {
    pub fn new(
        input_size: i32,
        hidden_sizes: &TIntArray,
        output_size: i32,
        hidden_activation: TActivationType,
        output_activation: TActivationType,
    ) -> Result<Self, String> {
        Self::new_with_backend(
            input_size,
            hidden_sizes,
            output_size,
            hidden_activation,
            output_activation,
            TGPUBackend::CPU,
        )
    }
    
    pub fn new_with_backend(
        input_size: i32,
        hidden_sizes: &TIntArray,
        output_size: i32,
        hidden_activation: TActivationType,
        output_activation: TActivationType,
        backend: TGPUBackend,
    ) -> Result<Self, String> {
        if input_size <= 0 {
            return Err("Input size must be positive".to_string());
        }
        if output_size <= 0 {
            return Err("Output size must be positive".to_string());
        }
        if hidden_sizes.is_empty() {
            return Err("Must have at least one hidden layer".to_string());
        }
        for &size in hidden_sizes {
            if size <= 0 {
                return Err("Hidden layer sizes must be positive".to_string());
            }
        }
        
        let input_layer = Self::create_layer(input_size, 0, TActivationType::atSigmoid);
        
        let mut hidden_layers = Vec::new();
        for (i, &size) in hidden_sizes.iter().enumerate() {
            let prev_size = if i == 0 { input_size } else { hidden_sizes[i - 1] };
            hidden_layers.push(Self::create_layer(size, prev_size, hidden_activation));
        }
        
        let last_hidden_size = *hidden_sizes.last().unwrap();
        let output_layer = Self::create_layer(output_size, last_hidden_size, output_activation);
        
        #[cfg(feature = "cuda")]
        let cuda_device = if backend == TGPUBackend::CUDA {
            match CudaDevice::new(0) {
                Ok(dev) => Some(dev),
                Err(e) => {
                    eprintln!("Warning: CUDA initialization failed: {}. Falling back to CPU.", e);
                    None
                }
            }
        } else {
            None
        };
        
        #[cfg(not(feature = "cuda"))]
        let cuda_device = None;
        
        #[cfg(feature = "opencl")]
        let opencl_context = if backend == TGPUBackend::OpenCL {
            match OpenCLContext::new() {
                Ok(ctx) => Some(ctx),
                Err(e) => {
                    eprintln!("Warning: OpenCL initialization failed: {}. Falling back to CPU.", e);
                    None
                }
            }
        } else {
            None
        };
        
        #[cfg(not(feature = "opencl"))]
        let opencl_context: Option<()> = None;
        
        let actual_backend = match backend {
            #[cfg(feature = "cuda")]
            TGPUBackend::CUDA => {
                if cuda_device.is_some() {
                    TGPUBackend::CUDA
                } else {
                    TGPUBackend::CPU
                }
            }
            #[cfg(feature = "opencl")]
            TGPUBackend::OpenCL => {
                if opencl_context.is_some() {
                    TGPUBackend::OpenCL
                } else {
                    TGPUBackend::CPU
                }
            }
            _ => TGPUBackend::CPU,
        };
        
        Ok(TMultiLayerPerceptronCUDA {
            FInputLayer: input_layer,
            FHiddenLayers: hidden_layers,
            FOutputLayer: output_layer,
            FHiddenSizes: hidden_sizes.to_vec(),
            FInputSize: input_size,
            FOutputSize: output_size,
            LearningRate: 0.1,
            Optimizer: TOptimizerType::otSGD,
            HiddenActivation: hidden_activation,
            OutputActivation: output_activation,
            DropoutRate: 0.0,
            L2Lambda: 0.0,
            Beta1: 0.9,
            Beta2: 0.999,
            Timestep: 0,
            EnableLRDecay: false,
            LRDecayRate: 0.95,
            LRDecayEpochs: 10,
            EnableEarlyStopping: false,
            EarlyStoppingPatience: 10,
            UseBatchNorm: false,
            GPUBackend: actual_backend,
            #[cfg(feature = "cuda")]
            cuda_device,
            #[cfg(feature = "opencl")]
            opencl_context,
        })
    }
    
    pub fn set_backend(&mut self, backend: TGPUBackend) -> Result<(), String> {
        match backend {
            TGPUBackend::CPU => {
                self.GPUBackend = TGPUBackend::CPU;
                Ok(())
            }
            TGPUBackend::CUDA => {
                #[cfg(feature = "cuda")]
                {
                    if self.cuda_device.is_none() {
                        match CudaDevice::new(0) {
                            Ok(dev) => {
                                self.cuda_device = Some(dev);
                                self.GPUBackend = TGPUBackend::CUDA;
                                Ok(())
                            }
                            Err(e) => Err(format!("CUDA initialization failed: {}", e))
                        }
                    } else {
                        self.GPUBackend = TGPUBackend::CUDA;
                        Ok(())
                    }
                }
                #[cfg(not(feature = "cuda"))]
                Err("CUDA support not compiled".to_string())
            }
            TGPUBackend::OpenCL => {
                #[cfg(feature = "opencl")]
                {
                    if self.opencl_context.is_none() {
                        self.opencl_context = Some(OpenCLContext::new()?);
                    }
                    self.GPUBackend = TGPUBackend::OpenCL;
                    Ok(())
                }
                #[cfg(not(feature = "opencl"))]
                Err("OpenCL support not compiled".to_string())
            }
        }
    }
    
    pub fn get_backend(&self) -> TGPUBackend {
        self.GPUBackend
    }
    
    fn create_layer(size: i32, input_size: i32, activation: TActivationType) -> TLayer {
        let mut neurons = Vec::new();
        let mut rng = rand::thread_rng();
        
        let limit = if input_size > 0 {
            (6.0_f64 / (input_size + size) as f64).sqrt()
        } else {
            0.1
        };
        
        for _ in 0..size {
            let mut weights = Vec::new();
            for _ in 0..input_size {
                weights.push(rng.gen_range(-limit..limit));
            }
            
            neurons.push(TNeuron {
                Weights: weights.clone(),
                Bias: 0.0,
                Output: 0.0,
                Error: 0.0,
                M: vec![0.0; input_size as usize],
                V: vec![0.0; input_size as usize],
                MBias: 0.0,
                VBias: 0.0,
            });
        }
        
        TLayer {
            Neurons: neurons,
            ActivationType: activation,
            DropoutMask: vec![true; size as usize],
        }
    }
    
    pub fn Predict(&mut self, input: &Darray) -> Result<Darray, String> {
        if input.len() != self.FInputSize as usize {
            return Err(format!(
                "Input size mismatch: expected {}, got {}",
                self.FInputSize,
                input.len()
            ));
        }
        
        for (i, &val) in input.iter().enumerate() {
            self.FInputLayer.Neurons[i].Output = val;
        }
        
        self.FeedForward();
        
        let mut output = Vec::new();
        for neuron in &self.FOutputLayer.Neurons {
            output.push(neuron.Output);
        }
        
        Ok(output)
    }
    
    fn FeedForward(&mut self) {
        for layer_idx in 0..self.FHiddenLayers.len() {
            let prev_outputs = if layer_idx == 0 {
                self.FInputLayer.Neurons.iter().map(|n| n.Output).collect::<Vec<_>>()
            } else {
                self.FHiddenLayers[layer_idx - 1].Neurons.iter().map(|n| n.Output).collect::<Vec<_>>()
            };
            
            self.FeedForwardLayer(layer_idx, &prev_outputs);
        }
        
        let prev_outputs: Vec<f64> = self.FHiddenLayers
            .last()
            .unwrap()
            .Neurons
            .iter()
            .map(|n| n.Output)
            .collect();
        
        self.FeedForwardOutputLayer(&prev_outputs);
    }
    
    fn FeedForwardLayer(&mut self, layer_idx: usize, prev_outputs: &[f64]) {
        let layer = &mut self.FHiddenLayers[layer_idx];
        
        for neuron in &mut layer.Neurons {
            let mut sum = neuron.Bias;
            for (j, &prev_out) in prev_outputs.iter().enumerate() {
                sum += neuron.Weights[j] * prev_out;
            }
            neuron.Output = ApplyActivation(sum, layer.ActivationType);
        }
    }
    
    fn FeedForwardOutputLayer(&mut self, prev_outputs: &[f64]) {
        for neuron in &mut self.FOutputLayer.Neurons {
            let mut sum = neuron.Bias;
            for (j, &prev_out) in prev_outputs.iter().enumerate() {
                sum += neuron.Weights[j] * prev_out;
            }
            neuron.Output = ApplyActivation(sum, self.FOutputLayer.ActivationType);
        }
        
        if self.FOutputLayer.ActivationType == TActivationType::atSoftmax {
            let outputs: Vec<f64> = self.FOutputLayer.Neurons.iter().map(|n| n.Output).collect();
            let softmax_outputs = Softmax(&outputs);
            for (i, &val) in softmax_outputs.iter().enumerate() {
                self.FOutputLayer.Neurons[i].Output = val;
            }
        }
    }
    
    pub fn Train(&mut self, input: &Darray, target: &Darray) -> Result<(), String> {
        if input.len() != self.FInputSize as usize {
            return Err(format!("Input size mismatch: expected {}, got {}", self.FInputSize, input.len()));
        }
        if target.len() != self.FOutputSize as usize {
            return Err(format!("Target size mismatch: expected {}, got {}", self.FOutputSize, target.len()));
        }
        
        self.Predict(input)?;
        self.BackPropagate(target);
        self.Timestep += 1;  // Increment before UpdateWeights for Adam bias correction
        self.UpdateWeights();

        Ok(())
    }
    
    fn BackPropagate(&mut self, target: &Darray) {
        let is_softmax = self.FOutputLayer.ActivationType == TActivationType::atSoftmax;
        
        for (i, neuron) in self.FOutputLayer.Neurons.iter_mut().enumerate() {
            if is_softmax {
                neuron.Error = target[i] - neuron.Output;
            } else {
                let derivative = ApplyActivationDerivative(neuron.Output, self.FOutputLayer.ActivationType);
                neuron.Error = derivative * (target[i] - neuron.Output);
            }
        }
        
        for layer_idx in (0..self.FHiddenLayers.len()).rev() {
            let next_layer_weights: Vec<Vec<f64>>;
            let next_layer_errors: Vec<f64>;

            if layer_idx == self.FHiddenLayers.len() - 1 {
                next_layer_weights = self.FOutputLayer.Neurons.iter().map(|n| n.Weights.clone()).collect();
                next_layer_errors = self.FOutputLayer.Neurons.iter().map(|n| n.Error).collect();
            } else {
                next_layer_weights = self.FHiddenLayers[layer_idx + 1].Neurons.iter().map(|n| n.Weights.clone()).collect();
                next_layer_errors = self.FHiddenLayers[layer_idx + 1].Neurons.iter().map(|n| n.Error).collect();
            }

            let activation_type = self.FHiddenLayers[layer_idx].ActivationType;
            for (i, neuron) in self.FHiddenLayers[layer_idx].Neurons.iter_mut().enumerate() {
                let mut error_sum = 0.0;
                for (j, next_neuron_weights) in next_layer_weights.iter().enumerate() {
                    error_sum += next_layer_errors[j] * next_neuron_weights[i];
                }

                let derivative = ApplyActivationDerivative(neuron.Output, activation_type);
                neuron.Error = derivative * error_sum;
            }
        }
    }
    
    fn UpdateWeights(&mut self) {
        // Extract optimizer parameters to avoid borrow issues
        let optimizer = self.Optimizer;
        let learning_rate = self.LearningRate;
        let l2_lambda = self.L2Lambda;
        let beta1 = self.Beta1;
        let beta2 = self.Beta2;
        let timestep = self.Timestep;

        for layer_idx in 0..self.FHiddenLayers.len() {
            let prev_outputs = if layer_idx == 0 {
                self.FInputLayer.Neurons.iter().map(|n| n.Output).collect::<Vec<_>>()
            } else {
                self.FHiddenLayers[layer_idx - 1].Neurons.iter().map(|n| n.Output).collect::<Vec<_>>()
            };

            for neuron in &mut self.FHiddenLayers[layer_idx].Neurons {
                Self::update_neuron_weights_static(neuron, &prev_outputs, optimizer, learning_rate, l2_lambda, beta1, beta2, timestep);
            }
        }

        let prev_outputs: Vec<f64> = self.FHiddenLayers
            .last()
            .unwrap()
            .Neurons
            .iter()
            .map(|n| n.Output)
            .collect();

        for neuron in &mut self.FOutputLayer.Neurons {
            Self::update_neuron_weights_static(neuron, &prev_outputs, optimizer, learning_rate, l2_lambda, beta1, beta2, timestep);
        }
    }

    fn update_neuron_weights_static(
        neuron: &mut TNeuron,
        prev_outputs: &[f64],
        optimizer: TOptimizerType,
        learning_rate: f64,
        l2_lambda: f64,
        beta1: f64,
        beta2: f64,
        timestep: i32,
    ) {
        match optimizer {
            TOptimizerType::otSGD => Self::update_neuron_weights_sgd_static(neuron, prev_outputs, learning_rate, l2_lambda),
            TOptimizerType::otAdam => Self::update_neuron_weights_adam_static(neuron, prev_outputs, learning_rate, l2_lambda, beta1, beta2, timestep),
            TOptimizerType::otRMSProp => Self::update_neuron_weights_rmsprop_static(neuron, prev_outputs, learning_rate, l2_lambda),
        }
    }

    fn update_neuron_weights_sgd_static(neuron: &mut TNeuron, prev_outputs: &[f64], learning_rate: f64, l2_lambda: f64) {
        for (j, &prev_out) in prev_outputs.iter().enumerate() {
            let mut gradient = neuron.Error * prev_out;
            if l2_lambda > 0.0 {
                gradient -= l2_lambda * neuron.Weights[j];
            }
            neuron.Weights[j] += learning_rate * gradient;
        }
        neuron.Bias += learning_rate * neuron.Error;
    }

    fn update_neuron_weights_adam_static(neuron: &mut TNeuron, prev_outputs: &[f64], learning_rate: f64, l2_lambda: f64, beta1: f64, beta2: f64, timestep: i32) {
        let eps = 1e-8;
        let beta1_t = beta1.powi(timestep);
        let beta2_t = beta2.powi(timestep);

        for (j, &prev_out) in prev_outputs.iter().enumerate() {
            let mut gradient = -neuron.Error * prev_out;
            if l2_lambda > 0.0 {
                gradient += l2_lambda * neuron.Weights[j];
            }

            neuron.M[j] = beta1 * neuron.M[j] + (1.0 - beta1) * gradient;
            neuron.V[j] = beta2 * neuron.V[j] + (1.0 - beta2) * gradient * gradient;

            let m_hat = neuron.M[j] / (1.0 - beta1_t);
            let v_hat = neuron.V[j] / (1.0 - beta2_t);

            neuron.Weights[j] -= learning_rate * m_hat / (v_hat.sqrt() + eps);
        }

        let gradient = -neuron.Error;
        neuron.MBias = beta1 * neuron.MBias + (1.0 - beta1) * gradient;
        neuron.VBias = beta2 * neuron.VBias + (1.0 - beta2) * gradient * gradient;
        let m_hat = neuron.MBias / (1.0 - beta1_t);
        let v_hat = neuron.VBias / (1.0 - beta2_t);
        neuron.Bias -= learning_rate * m_hat / (v_hat.sqrt() + eps);
    }

    fn update_neuron_weights_rmsprop_static(neuron: &mut TNeuron, prev_outputs: &[f64], learning_rate: f64, l2_lambda: f64) {
        let eps = 1e-8;
        let decay = 0.9;

        for (j, &prev_out) in prev_outputs.iter().enumerate() {
            let mut gradient = -neuron.Error * prev_out;
            if l2_lambda > 0.0 {
                gradient += l2_lambda * neuron.Weights[j];
            }

            neuron.V[j] = decay * neuron.V[j] + (1.0 - decay) * gradient * gradient;
            neuron.Weights[j] -= learning_rate * gradient / (neuron.V[j].sqrt() + eps);
        }

        let gradient = -neuron.Error;
        neuron.VBias = decay * neuron.VBias + (1.0 - decay) * gradient * gradient;
        neuron.Bias -= learning_rate * gradient / (neuron.VBias.sqrt() + eps);
    }
    
    pub fn ComputeLoss(&self, predicted: &Darray, target: &Darray) -> f64 {
        let mut loss = 0.0;
        for i in 0..predicted.len() {
            let diff = predicted[i] - target[i];
            loss += diff * diff;
        }
        loss / predicted.len() as f64
    }
    
    pub fn Save(&self, filename: &str) -> Result<(), String> {
        let input_layer_data = TLayerData {
            neurons: self.FInputLayer.Neurons.iter().map(|n| TNeuronData {
                weights: n.Weights.clone(),
                bias: n.Bias,
                m: n.M.clone(),
                v: n.V.clone(),
                m_bias: n.MBias,
                v_bias: n.VBias,
            }).collect(),
            activation: self.FInputLayer.ActivationType as i32,
        };
        
        let hidden_layers_data: Vec<TLayerData> = self.FHiddenLayers.iter().map(|layer| {
            TLayerData {
                neurons: layer.Neurons.iter().map(|n| TNeuronData {
                    weights: n.Weights.clone(),
                    bias: n.Bias,
                    m: n.M.clone(),
                    v: n.V.clone(),
                    m_bias: n.MBias,
                    v_bias: n.VBias,
                }).collect(),
                activation: layer.ActivationType as i32,
            }
        }).collect();
        
        let output_layer_data = TLayerData {
            neurons: self.FOutputLayer.Neurons.iter().map(|n| TNeuronData {
                weights: n.Weights.clone(),
                bias: n.Bias,
                m: n.M.clone(),
                v: n.V.clone(),
                m_bias: n.MBias,
                v_bias: n.VBias,
            }).collect(),
            activation: self.FOutputLayer.ActivationType as i32,
        };
        
        let model_data = TModelData {
            magic: crate::MODEL_MAGIC.to_string(),
            version: "1.0".to_string(),
            input_size: self.FInputSize,
            output_size: self.FOutputSize,
            hidden_sizes: self.FHiddenSizes.clone(),
            learning_rate: self.LearningRate,
            optimizer: self.Optimizer as i32,
            hidden_activation: self.HiddenActivation as i32,
            output_activation: self.OutputActivation as i32,
            dropout_rate: self.DropoutRate,
            l2_lambda: self.L2Lambda,
            beta1: self.Beta1,
            beta2: self.Beta2,
            timestep: self.Timestep,
            use_batch_norm: self.UseBatchNorm,
            gpu_backend: self.GPUBackend.to_str().to_string(),
            input_layer: input_layer_data,
            hidden_layers: hidden_layers_data,
            output_layer: output_layer_data,
        };
        
        let json = serde_json::to_string_pretty(&model_data)
            .map_err(|e| format!("Serialization error: {}", e))?;
        
        let mut file = File::create(filename)
            .map_err(|e| format!("File creation error: {}", e))?;
        
        file.write_all(json.as_bytes())
            .map_err(|e| format!("File write error: {}", e))?;
        
        Ok(())
    }
    
    pub fn Load(filename: &str) -> Result<Self, String> {
        let file = File::open(filename)
            .map_err(|e| format!("File open error: {}", e))?;
        
        let reader = BufReader::new(file);
        let model_data: TModelData = serde_json::from_reader(reader)
            .map_err(|e| format!("Deserialization error: {}", e))?;
        
        if model_data.magic != crate::MODEL_MAGIC {
            return Err(format!("Invalid model file magic: {}", model_data.magic));
        }
        
        let hidden_activation = match model_data.hidden_activation {
            0 => TActivationType::atSigmoid,
            1 => TActivationType::atTanh,
            2 => TActivationType::atReLU,
            3 => TActivationType::atSoftmax,
            _ => TActivationType::atSigmoid,
        };
        
        let output_activation = match model_data.output_activation {
            0 => TActivationType::atSigmoid,
            1 => TActivationType::atTanh,
            2 => TActivationType::atReLU,
            3 => TActivationType::atSoftmax,
            _ => TActivationType::atSigmoid,
        };
        
        let backend = TGPUBackend::from_str(&model_data.gpu_backend);
        
        let mut mlp = Self::new_with_backend(
            model_data.input_size,
            &model_data.hidden_sizes,
            model_data.output_size,
            hidden_activation,
            output_activation,
            backend,
        )?;
        
        mlp.LearningRate = model_data.learning_rate;
        mlp.Optimizer = match model_data.optimizer {
            0 => TOptimizerType::otSGD,
            1 => TOptimizerType::otAdam,
            2 => TOptimizerType::otRMSProp,
            _ => TOptimizerType::otSGD,
        };
        mlp.DropoutRate = model_data.dropout_rate;
        mlp.L2Lambda = model_data.l2_lambda;
        mlp.Beta1 = model_data.beta1;
        mlp.Beta2 = model_data.beta2;
        mlp.Timestep = model_data.timestep;
        mlp.UseBatchNorm = model_data.use_batch_norm;
        
        for (i, neuron_data) in model_data.input_layer.neurons.iter().enumerate() {
            if i < mlp.FInputLayer.Neurons.len() {
                mlp.FInputLayer.Neurons[i].Weights = neuron_data.weights.clone();
                mlp.FInputLayer.Neurons[i].Bias = neuron_data.bias;
                mlp.FInputLayer.Neurons[i].M = neuron_data.m.clone();
                mlp.FInputLayer.Neurons[i].V = neuron_data.v.clone();
                mlp.FInputLayer.Neurons[i].MBias = neuron_data.m_bias;
                mlp.FInputLayer.Neurons[i].VBias = neuron_data.v_bias;
            }
        }
        
        for (layer_idx, layer_data) in model_data.hidden_layers.iter().enumerate() {
            if layer_idx < mlp.FHiddenLayers.len() {
                for (neuron_idx, neuron_data) in layer_data.neurons.iter().enumerate() {
                    if neuron_idx < mlp.FHiddenLayers[layer_idx].Neurons.len() {
                        mlp.FHiddenLayers[layer_idx].Neurons[neuron_idx].Weights = neuron_data.weights.clone();
                        mlp.FHiddenLayers[layer_idx].Neurons[neuron_idx].Bias = neuron_data.bias;
                        mlp.FHiddenLayers[layer_idx].Neurons[neuron_idx].M = neuron_data.m.clone();
                        mlp.FHiddenLayers[layer_idx].Neurons[neuron_idx].V = neuron_data.v.clone();
                        mlp.FHiddenLayers[layer_idx].Neurons[neuron_idx].MBias = neuron_data.m_bias;
                        mlp.FHiddenLayers[layer_idx].Neurons[neuron_idx].VBias = neuron_data.v_bias;
                    }
                }
            }
        }
        
        for (i, neuron_data) in model_data.output_layer.neurons.iter().enumerate() {
            if i < mlp.FOutputLayer.Neurons.len() {
                mlp.FOutputLayer.Neurons[i].Weights = neuron_data.weights.clone();
                mlp.FOutputLayer.Neurons[i].Bias = neuron_data.bias;
                mlp.FOutputLayer.Neurons[i].M = neuron_data.m.clone();
                mlp.FOutputLayer.Neurons[i].V = neuron_data.v.clone();
                mlp.FOutputLayer.Neurons[i].MBias = neuron_data.m_bias;
                mlp.FOutputLayer.Neurons[i].VBias = neuron_data.v_bias;
            }
        }
        
        Ok(mlp)
    }
    
    pub fn GetInputSize(&self) -> i32 {
        self.FInputSize
    }
    
    pub fn GetOutputSize(&self) -> i32 {
        self.FOutputSize
    }
    
    pub fn GetHiddenLayerCount(&self) -> i32 {
        self.FHiddenLayers.len() as i32
    }
    
    pub fn GetHiddenSizes(&self) -> &TIntArray {
        &self.FHiddenSizes
    }
    
    pub fn GetNumLayers(&self) -> i32 {
        2 + self.FHiddenLayers.len() as i32
    }
    
    pub fn GetLayerSize(&self, layer_idx: usize) -> i32 {
        if layer_idx == 0 {
            self.FInputSize
        } else if layer_idx <= self.FHiddenLayers.len() {
            self.FHiddenLayers[layer_idx - 1].Neurons.len() as i32
        } else if layer_idx == self.FHiddenLayers.len() + 1 {
            self.FOutputSize
        } else {
            0
        }
    }
    
    pub fn GetLayerActivation(&self, layer_idx: i32) -> TActivationType {
        if layer_idx == 0 {
            self.FInputLayer.ActivationType
        } else if layer_idx > 0 && (layer_idx as usize) <= self.FHiddenLayers.len() {
            self.FHiddenLayers[layer_idx as usize - 1].ActivationType
        } else {
            self.FOutputLayer.ActivationType
        }
    }
    
    pub fn GetNeuronWeight(&self, layer_idx: i32, neuron_idx: i32, weight_idx: i32) -> f64 {
        if layer_idx > 0 && (layer_idx as usize) <= self.FHiddenLayers.len() {
            let layer = &self.FHiddenLayers[layer_idx as usize - 1];
            if neuron_idx >= 0 && (neuron_idx as usize) < layer.Neurons.len() {
                let neuron = &layer.Neurons[neuron_idx as usize];
                if weight_idx >= 0 && (weight_idx as usize) < neuron.Weights.len() {
                    return neuron.Weights[weight_idx as usize];
                }
            }
        }
        0.0
    }
    
    pub fn SetNeuronWeight(&mut self, layer_idx: i32, neuron_idx: i32, weight_idx: i32, value: f64) {
        if layer_idx > 0 && (layer_idx as usize) <= self.FHiddenLayers.len() {
            let layer = &mut self.FHiddenLayers[layer_idx as usize - 1];
            if neuron_idx >= 0 && (neuron_idx as usize) < layer.Neurons.len() {
                let neuron = &mut layer.Neurons[neuron_idx as usize];
                if weight_idx >= 0 && (weight_idx as usize) < neuron.Weights.len() {
                    neuron.Weights[weight_idx as usize] = value;
                }
            }
        }
    }
    
    pub fn GetNeuronWeights(&self, layer_idx: i32, neuron_idx: i32) -> Vec<f64> {
        if layer_idx > 0 && (layer_idx as usize) <= self.FHiddenLayers.len() {
            let layer = &self.FHiddenLayers[layer_idx as usize - 1];
            if neuron_idx >= 0 && (neuron_idx as usize) < layer.Neurons.len() {
                return layer.Neurons[neuron_idx as usize].Weights.clone();
            }
        }
        Vec::new()
    }
    
    pub fn GetNeuronBias(&self, layer_idx: i32, neuron_idx: i32) -> f64 {
        if layer_idx > 0 && (layer_idx as usize) <= self.FHiddenLayers.len() {
            let layer = &self.FHiddenLayers[layer_idx as usize - 1];
            if neuron_idx >= 0 && (neuron_idx as usize) < layer.Neurons.len() {
                return layer.Neurons[neuron_idx as usize].Bias;
            }
        }
        0.0
    }
    
    pub fn SetNeuronBias(&mut self, layer_idx: i32, neuron_idx: i32, value: f64) {
        if layer_idx > 0 && (layer_idx as usize) <= self.FHiddenLayers.len() {
            let layer = &mut self.FHiddenLayers[layer_idx as usize - 1];
            if neuron_idx >= 0 && (neuron_idx as usize) < layer.Neurons.len() {
                layer.Neurons[neuron_idx as usize].Bias = value;
            }
        }
    }
    
    pub fn GetLayerOutputs(&self, layer_idx: i32) -> Vec<f64> {
        if layer_idx == 0 {
            self.FInputLayer.Neurons.iter().map(|n| n.Output).collect()
        } else if layer_idx > 0 && (layer_idx as usize) <= self.FHiddenLayers.len() {
            self.FHiddenLayers[layer_idx as usize - 1].Neurons.iter().map(|n| n.Output).collect()
        } else if layer_idx as usize == self.FHiddenLayers.len() + 1 {
            self.FOutputLayer.Neurons.iter().map(|n| n.Output).collect()
        } else {
            Vec::new()
        }
    }
    
    pub fn GetLayerErrors(&self, layer_idx: i32) -> Vec<f64> {
        if layer_idx > 0 && (layer_idx as usize) <= self.FHiddenLayers.len() {
            self.FHiddenLayers[layer_idx as usize - 1].Neurons.iter().map(|n| n.Error).collect()
        } else if layer_idx as usize == self.FHiddenLayers.len() + 1 {
            self.FOutputLayer.Neurons.iter().map(|n| n.Error).collect()
        } else {
            Vec::new()
        }
    }
    
    pub fn GetWeightsPerNeuron(&self, layer_idx: i32, neuron_idx: i32) -> i32 {
        if layer_idx > 0 && (layer_idx as usize) <= self.FHiddenLayers.len() {
            let layer = &self.FHiddenLayers[layer_idx as usize - 1];
            if neuron_idx >= 0 && (neuron_idx as usize) < layer.Neurons.len() {
                return layer.Neurons[neuron_idx as usize].Weights.len() as i32;
            }
        }
        0
    }
    
    pub fn GetActivationHistogram(&self, layer_idx: i32, bins: usize) -> Vec<i32> {
        let outputs = self.GetLayerOutputs(layer_idx);
        if outputs.is_empty() {
            return vec![0; bins];
        }
        
        let min = outputs.iter().cloned().fold(f64::INFINITY, f64::min);
        let max = outputs.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let range = max - min;
        
        let mut histogram = vec![0; bins];
        
        for &output in &outputs {
            let bin = if range > 1e-10 {
                let normalized = (output - min) / range;
                let bin_idx = (normalized * bins as f64).floor() as usize;
                bin_idx.min(bins - 1)
            } else {
                0
            };
            histogram[bin] += 1;
        }
        
        histogram
    }
    
    pub fn GetGradientHistogram(&self, layer_idx: i32, bins: usize) -> Vec<i32> {
        let errors = self.GetLayerErrors(layer_idx);
        if errors.is_empty() {
            return vec![0; bins];
        }
        
        let min = errors.iter().cloned().fold(f64::INFINITY, f64::min);
        let max = errors.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let range = max - min;
        
        let mut histogram = vec![0; bins];
        
        for &error in &errors {
            let bin = if range > 1e-10 {
                let normalized = (error - min) / range;
                let bin_idx = (normalized * bins as f64).floor() as usize;
                bin_idx.min(bins - 1)
            } else {
                0
            };
            histogram[bin] += 1;
        }
        
        histogram
    }
    
    pub fn GetWeightM(&self, layer_idx: i32, neuron_idx: i32, weight_idx: i32) -> f64 {
        if layer_idx > 0 && (layer_idx as usize) <= self.FHiddenLayers.len() {
            let layer = &self.FHiddenLayers[layer_idx as usize - 1];
            if neuron_idx >= 0 && (neuron_idx as usize) < layer.Neurons.len() {
                let neuron = &layer.Neurons[neuron_idx as usize];
                if weight_idx >= 0 && (weight_idx as usize) < neuron.M.len() {
                    return neuron.M[weight_idx as usize];
                }
            }
        }
        0.0
    }
    
    pub fn GetWeightV(&self, layer_idx: i32, neuron_idx: i32, weight_idx: i32) -> f64 {
        if layer_idx > 0 && (layer_idx as usize) <= self.FHiddenLayers.len() {
            let layer = &self.FHiddenLayers[layer_idx as usize - 1];
            if neuron_idx >= 0 && (neuron_idx as usize) < layer.Neurons.len() {
                let neuron = &layer.Neurons[neuron_idx as usize];
                if weight_idx >= 0 && (weight_idx as usize) < neuron.V.len() {
                    return neuron.V[weight_idx as usize];
                }
            }
        }
        0.0
    }
    
    pub fn GetBiasM(&self, layer_idx: i32, neuron_idx: i32) -> f64 {
        if layer_idx > 0 && (layer_idx as usize) <= self.FHiddenLayers.len() {
            let layer = &self.FHiddenLayers[layer_idx as usize - 1];
            if neuron_idx >= 0 && (neuron_idx as usize) < layer.Neurons.len() {
                return layer.Neurons[neuron_idx as usize].MBias;
            }
        }
        0.0
    }
    
    pub fn GetBiasV(&self, layer_idx: i32, neuron_idx: i32) -> f64 {
        if layer_idx > 0 && (layer_idx as usize) <= self.FHiddenLayers.len() {
            let layer = &self.FHiddenLayers[layer_idx as usize - 1];
            if neuron_idx >= 0 && (neuron_idx as usize) < layer.Neurons.len() {
                return layer.Neurons[neuron_idx as usize].VBias;
            }
        }
        0.0
    }
    
    pub fn SetWeightM(&mut self, layer_idx: i32, neuron_idx: i32, weight_idx: i32, value: f64) {
        if layer_idx > 0 && (layer_idx as usize) <= self.FHiddenLayers.len() {
            let layer = &mut self.FHiddenLayers[layer_idx as usize - 1];
            if neuron_idx >= 0 && (neuron_idx as usize) < layer.Neurons.len() {
                let neuron = &mut layer.Neurons[neuron_idx as usize];
                if weight_idx >= 0 && (weight_idx as usize) < neuron.M.len() {
                    neuron.M[weight_idx as usize] = value;
                }
            }
        }
    }
    
    pub fn SetWeightV(&mut self, layer_idx: i32, neuron_idx: i32, weight_idx: i32, value: f64) {
        if layer_idx > 0 && (layer_idx as usize) <= self.FHiddenLayers.len() {
            let layer = &mut self.FHiddenLayers[layer_idx as usize - 1];
            if neuron_idx >= 0 && (neuron_idx as usize) < layer.Neurons.len() {
                let neuron = &mut layer.Neurons[neuron_idx as usize];
                if weight_idx >= 0 && (weight_idx as usize) < neuron.V.len() {
                    neuron.V[weight_idx as usize] = value;
                }
            }
        }
    }
    
    pub fn SetBiasM(&mut self, layer_idx: i32, neuron_idx: i32, value: f64) {
        if layer_idx > 0 && (layer_idx as usize) <= self.FHiddenLayers.len() {
            let layer = &mut self.FHiddenLayers[layer_idx as usize - 1];
            if neuron_idx >= 0 && (neuron_idx as usize) < layer.Neurons.len() {
                layer.Neurons[neuron_idx as usize].MBias = value;
            }
        }
    }
    
    pub fn SetBiasV(&mut self, layer_idx: i32, neuron_idx: i32, value: f64) {
        if layer_idx > 0 && (layer_idx as usize) <= self.FHiddenLayers.len() {
            let layer = &mut self.FHiddenLayers[layer_idx as usize - 1];
            if neuron_idx >= 0 && (neuron_idx as usize) < layer.Neurons.len() {
                layer.Neurons[neuron_idx as usize].VBias = value;
            }
        }
    }
    
    pub fn SetTimestep(&mut self, value: i32) {
        self.Timestep = value;
    }
    
    pub fn SetLayerActivation(&mut self, layer_idx: i32, activation: TActivationType) {
        if layer_idx == 0 {
            self.FInputLayer.ActivationType = activation;
        } else if layer_idx > 0 && (layer_idx as usize) <= self.FHiddenLayers.len() {
            self.FHiddenLayers[layer_idx as usize - 1].ActivationType = activation;
            self.HiddenActivation = activation;
        } else if layer_idx as usize == self.FHiddenLayers.len() + 1 {
            self.FOutputLayer.ActivationType = activation;
            self.OutputActivation = activation;
        }
    }
    
    pub fn SetNeuronWeights(&mut self, layer_idx: i32, neuron_idx: i32, weights: &[f64]) {
        if layer_idx > 0 && (layer_idx as usize) <= self.FHiddenLayers.len() {
            let layer = &mut self.FHiddenLayers[layer_idx as usize - 1];
            if neuron_idx >= 0 && (neuron_idx as usize) < layer.Neurons.len() {
                let neuron = &mut layer.Neurons[neuron_idx as usize];
                let len = weights.len().min(neuron.Weights.len());
                neuron.Weights[..len].copy_from_slice(&weights[..len]);
            }
        }
    }
    
    pub fn export_to_onnx(&self, _filename: &str) -> Result<(), String> {
        Err("ONNX export not yet implemented".to_string())
    }
    
    pub fn import_from_onnx(_filename: &str) -> Result<Self, String> {
        Err("ONNX import not yet implemented".to_string())
    }
    
    pub fn compute_feature_importance(&self) -> Vec<(usize, f64)> {
        let mut importance = vec![0.0; self.FInputSize as usize];
        
        if self.FHiddenLayers.is_empty() {
            for neuron in &self.FOutputLayer.Neurons {
                for (i, &weight) in neuron.Weights.iter().enumerate() {
                    importance[i] += weight.abs();
                }
            }
        } else {
            let first_layer = &self.FHiddenLayers[0];
            for neuron in &first_layer.Neurons {
                for (i, &weight) in neuron.Weights.iter().enumerate() {
                    importance[i] += weight.abs();
                }
            }
        }
        
        let mut ranked: Vec<(usize, f64)> = importance.iter().enumerate().map(|(i, &v)| (i, v)).collect();
        ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        
        ranked
    }
}

fn ApplyActivation(x: f64, activation: TActivationType) -> f64 {
    match activation {
        TActivationType::atSigmoid => Sigmoid(x),
        TActivationType::atTanh => x.tanh(),
        TActivationType::atReLU => x.max(0.0),
        TActivationType::atSoftmax => x,
    }
}

fn ApplyActivationDerivative(output: f64, activation: TActivationType) -> f64 {
    match activation {
        TActivationType::atSigmoid => output * (1.0 - output),
        TActivationType::atTanh => 1.0 - output * output,
        TActivationType::atReLU => if output > 0.0 { 1.0 } else { 0.0 },
        TActivationType::atSoftmax => 1.0,
    }
}

fn Sigmoid(x: f64) -> f64 {
    if x < -500.0 {
        0.0
    } else if x > 500.0 {
        1.0
    } else {
        1.0 / (1.0 + (-x).exp())
    }
}

fn Softmax(x: &[f64]) -> Vec<f64> {
    let max_val = x.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let exp_values: Vec<f64> = x.iter().map(|&v| (v - max_val).exp()).collect();
    let sum: f64 = exp_values.iter().sum();
    
    exp_values.iter().map(|&v| {
        let val = v / sum;
        if val < crate::EPSILON {
            crate::EPSILON
        } else if val > 1.0 - crate::EPSILON {
            1.0 - crate::EPSILON
        } else {
            val
        }
    }).collect()
}

pub fn LoadDataCSV(filename: &str, input_size: i32, output_size: i32) -> Vec<TDataPoint> {
    let mut data = Vec::new();
    
    let file = match File::open(filename) {
        Ok(f) => f,
        Err(_) => return data,
    };
    
    let reader = BufReader::new(file);
    
    for line in reader.lines() {
        let line = match line {
            Ok(l) => l,
            Err(_) => continue,
        };
        
        if line.trim().is_empty() {
            continue;
        }
        
        let values: Vec<f64> = line
            .split(',')
            .filter_map(|s| s.trim().parse().ok())
            .collect();
        
        if values.len() < (input_size + output_size) as usize {
            continue;
        }
        
        let input = values[..input_size as usize].to_vec();
        let target = values[input_size as usize..(input_size + output_size) as usize].to_vec();
        
        data.push(TDataPoint {
            Input: input,
            Target: target,
        });
    }
    
    data
}

pub fn NormalizeData(data: &mut Vec<TDataPoint>) {
    if data.is_empty() {
        return;
    }
    
    let input_size = data[0].Input.len();
    let mut mins = vec![f64::INFINITY; input_size];
    let mut maxs = vec![f64::NEG_INFINITY; input_size];
    
    for dp in data.iter() {
        for (i, &val) in dp.Input.iter().enumerate() {
            if val < mins[i] {
                mins[i] = val;
            }
            if val > maxs[i] {
                maxs[i] = val;
            }
        }
    }
    
    for dp in data.iter_mut() {
        for (i, val) in dp.Input.iter_mut().enumerate() {
            let range = maxs[i] - mins[i];
            if range > 1e-10 {
                *val = (*val - mins[i]) / range;
            }
        }
    }
}

pub fn ShuffleData(data: &mut Vec<TDataPoint>) {
    use rand::seq::SliceRandom;
    let mut rng = rand::thread_rng();
    data.shuffle(&mut rng);
}

pub fn ParseActivation(s: &str) -> TActivationType {
    match s.to_lowercase().as_str() {
        "sigmoid" => TActivationType::atSigmoid,
        "tanh" => TActivationType::atTanh,
        "relu" => TActivationType::atReLU,
        "softmax" => TActivationType::atSoftmax,
        _ => TActivationType::atSigmoid,
    }
}

pub fn ParseOptimizer(s: &str) -> TOptimizerType {
    match s.to_lowercase().as_str() {
        "sgd" => TOptimizerType::otSGD,
        "adam" => TOptimizerType::otAdam,
        "rmsprop" => TOptimizerType::otRMSProp,
        _ => TOptimizerType::otSGD,
    }
}

pub fn ParseDoubleArray(s: &str) -> Vec<f64> {
    s.split(',')
        .filter_map(|s| s.trim().parse().ok())
        .collect()
}

pub fn ParseIntArray(s: &str) -> Vec<i32> {
    s.split(',')
        .filter_map(|s| s.trim().parse().ok())
        .collect()
}

pub fn ActivationToStr(act: TActivationType) -> &'static str {
    match act {
        TActivationType::atSigmoid => "sigmoid",
        TActivationType::atTanh => "tanh",
        TActivationType::atReLU => "relu",
        TActivationType::atSoftmax => "softmax",
    }
}

pub fn OptimizerToStr(opt: TOptimizerType) -> &'static str {
    match opt {
        TOptimizerType::otSGD => "sgd",
        TOptimizerType::otAdam => "adam",
        TOptimizerType::otRMSProp => "rmsprop",
    }
}

pub fn MaxIndex(arr: &[f64]) -> usize {
    let mut max_idx = 0;
    let mut max_val = arr[0];
    
    for (i, &val) in arr.iter().enumerate().skip(1) {
        if val > max_val {
            max_val = val;
            max_idx = i;
        }
    }
    
    max_idx
}

