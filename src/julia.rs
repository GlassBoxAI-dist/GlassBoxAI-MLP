/*
 * MIT License
 * Copyright (c) 2025 Matthew Abbott
 */

//! C FFI bindings for Julia integration

use std::ffi::{c_char, CStr, CString};
use std::ptr;
use crate::mlp::*;
use crate::gpu_backend::*;

/// Opaque handle to MLP
pub struct JuliaMLP {
    inner: TMultiLayerPerceptronCUDA,
}

/// Result status codes
#[repr(i32)]
pub enum JuliaStatus {
    Ok = 0,
    InvalidArg = -1,
    RuntimeError = -2,
    IoError = -3,
}

// Last error message (thread-local)
thread_local! {
    static LAST_ERROR: std::cell::RefCell<Option<CString>> = std::cell::RefCell::new(None);
}

fn set_error(msg: String) {
    let safe_msg = msg.replace('\0', "");
    LAST_ERROR.with(|e| {
        *e.borrow_mut() = CString::new(safe_msg).ok();
    });
}

/// Get last error message
#[no_mangle]
pub extern "C" fn mlp_get_last_error() -> *const c_char {
    LAST_ERROR.with(|e| {
        match &*e.borrow() {
            Some(cstr) => cstr.as_ptr(),
            None => ptr::null(),
        }
    })
}

/// Free error string (no-op, kept for API symmetry)
#[no_mangle]
pub extern "C" fn mlp_free_error(_ptr: *const c_char) {
    // No-op: error is stored in thread-local
}

/// Create a new MLP
/// 
/// # Arguments
/// * `input_size` - Number of input neurons
/// * `hidden_sizes` - Pointer to array of hidden layer sizes
/// * `hidden_count` - Number of hidden layers
/// * `output_size` - Number of output neurons
/// * `hidden_activation` - Hidden activation: 0=Sigmoid, 1=Tanh, 2=ReLU, 3=Softmax
/// * `output_activation` - Output activation: 0=Sigmoid, 1=Tanh, 2=ReLU, 3=Softmax
/// * `gpu_backend` - Backend string: "auto", "cpu", "cuda", "opencl"
#[no_mangle]
pub extern "C" fn mlp_create(
    input_size: i32,
    hidden_sizes: *const i32,
    hidden_count: i32,
    output_size: i32,
    hidden_activation: i32,
    output_activation: i32,
    gpu_backend: *const c_char,
) -> *mut JuliaMLP {
    if hidden_sizes.is_null() || hidden_count <= 0 {
        set_error("Invalid hidden layer configuration".to_string());
        return ptr::null_mut();
    }
    if input_size <= 0 || output_size <= 0 {
        set_error("Input and output sizes must be positive".to_string());
        return ptr::null_mut();
    }
    if hidden_count > 64 {
        set_error("Too many hidden layers".to_string());
        return ptr::null_mut();
    }

    let hidden: Vec<i32> = unsafe {
        std::slice::from_raw_parts(hidden_sizes, hidden_count as usize).to_vec()
    };

    let hidden_act = int_to_activation(hidden_activation);
    let output_act = int_to_activation(output_activation);

    let backend = if gpu_backend.is_null() {
        select_best_backend()
    } else {
        let backend_str = unsafe { CStr::from_ptr(gpu_backend) }
            .to_str()
            .unwrap_or("auto");
        match backend_str {
            "cuda" => TGPUBackend::CUDA,
            "opencl" | "ocl" => TGPUBackend::OpenCL,
            "cpu" => TGPUBackend::CPU,
            _ => select_best_backend(),
        }
    };

    match TMultiLayerPerceptronCUDA::new_with_backend(
        input_size,
        &hidden,
        output_size,
        hidden_act,
        output_act,
        backend,
    ) {
        Ok(mlp) => Box::into_raw(Box::new(JuliaMLP { inner: mlp })),
        Err(e) => {
            set_error(e);
            ptr::null_mut()
        }
    }
}

/// Destroy an MLP instance
#[no_mangle]
pub extern "C" fn mlp_destroy(mlp: *mut JuliaMLP) {
    if !mlp.is_null() {
        unsafe { drop(Box::from_raw(mlp)) };
    }
}

/// Train on a single sample
#[no_mangle]
pub extern "C" fn mlp_train(
    mlp: *mut JuliaMLP,
    input: *const f64,
    input_len: i32,
    target: *const f64,
    target_len: i32,
) -> i32 {
    if mlp.is_null() || input.is_null() || target.is_null() {
        set_error("Null pointer".to_string());
        return JuliaStatus::InvalidArg as i32;
    }
    if input_len <= 0 || target_len <= 0 {
        set_error("Input and target lengths must be positive".to_string());
        return JuliaStatus::InvalidArg as i32;
    }
    if input_len > 4096 || target_len > 4096 {
        set_error("Input or target length exceeds maximum".to_string());
        return JuliaStatus::InvalidArg as i32;
    }

    let mlp = unsafe { &mut *mlp };
    let input_vec: Vec<f64> = unsafe {
        std::slice::from_raw_parts(input, input_len as usize).to_vec()
    };
    let target_vec: Vec<f64> = unsafe {
        std::slice::from_raw_parts(target, target_len as usize).to_vec()
    };

    match mlp.inner.Train(&input_vec, &target_vec) {
        Ok(()) => JuliaStatus::Ok as i32,
        Err(e) => {
            set_error(e);
            JuliaStatus::RuntimeError as i32
        }
    }
}

/// Make a prediction
/// 
/// # Returns
/// Output array length, or negative on error. Caller must provide output buffer.
#[no_mangle]
pub extern "C" fn mlp_predict(
    mlp: *mut JuliaMLP,
    input: *const f64,
    input_len: i32,
    output: *mut f64,
    output_capacity: i32,
) -> i32 {
    if mlp.is_null() || input.is_null() || output.is_null() {
        set_error("Null pointer".to_string());
        return JuliaStatus::InvalidArg as i32;
    }
    if input_len <= 0 || output_capacity <= 0 {
        set_error("Lengths must be positive".to_string());
        return JuliaStatus::InvalidArg as i32;
    }
    if input_len > 4096 || output_capacity > 4096 {
        set_error("Length exceeds maximum".to_string());
        return JuliaStatus::InvalidArg as i32;
    }

    let mlp = unsafe { &mut *mlp };
    let input_vec: Vec<f64> = unsafe {
        std::slice::from_raw_parts(input, input_len as usize).to_vec()
    };

    match mlp.inner.Predict(&input_vec) {
        Ok(result) => {
            let len = result.len().min(output_capacity as usize);
            unsafe {
                ptr::copy_nonoverlapping(result.as_ptr(), output, len);
            }
            len as i32
        }
        Err(e) => {
            set_error(e);
            JuliaStatus::RuntimeError as i32
        }
    }
}

/// Compute loss for given output and target
#[no_mangle]
pub extern "C" fn mlp_compute_loss(
    mlp: *const JuliaMLP,
    output: *const f64,
    output_len: i32,
    target: *const f64,
    target_len: i32,
) -> f64 {
    if mlp.is_null() || output.is_null() || target.is_null() {
        return f64::NAN;
    }
    if output_len <= 0 || target_len <= 0 || output_len > 4096 || target_len > 4096 {
        return f64::NAN;
    }

    let mlp = unsafe { &*mlp };
    let output_vec: Vec<f64> = unsafe {
        std::slice::from_raw_parts(output, output_len as usize).to_vec()
    };
    let target_vec: Vec<f64> = unsafe {
        std::slice::from_raw_parts(target, target_len as usize).to_vec()
    };

    mlp.inner.ComputeLoss(&output_vec, &target_vec)
}

/// Save model to file
#[no_mangle]
pub extern "C" fn mlp_save(mlp: *const JuliaMLP, filename: *const c_char) -> i32 {
    if mlp.is_null() || filename.is_null() {
        set_error("Null pointer".to_string());
        return JuliaStatus::InvalidArg as i32;
    }

    let mlp = unsafe { &*mlp };
    let path = unsafe { CStr::from_ptr(filename) }
        .to_str()
        .unwrap_or("");

    match mlp.inner.Save(path) {
        Ok(()) => JuliaStatus::Ok as i32,
        Err(e) => {
            set_error(e);
            JuliaStatus::IoError as i32
        }
    }
}

/// Load model from file
#[no_mangle]
pub extern "C" fn mlp_load(filename: *const c_char) -> *mut JuliaMLP {
    if filename.is_null() {
        set_error("Null filename".to_string());
        return ptr::null_mut();
    }

    let path = unsafe { CStr::from_ptr(filename) }
        .to_str()
        .unwrap_or("");

    match TMultiLayerPerceptronCUDA::Load(path) {
        Ok(mlp) => Box::into_raw(Box::new(JuliaMLP { inner: mlp })),
        Err(e) => {
            set_error(e);
            ptr::null_mut()
        }
    }
}

/// Get learning rate
#[no_mangle]
pub extern "C" fn mlp_get_learning_rate(mlp: *const JuliaMLP) -> f64 {
    if mlp.is_null() { return 0.0; }
    unsafe { (*mlp).inner.LearningRate }
}

/// Set learning rate
#[no_mangle]
pub extern "C" fn mlp_set_learning_rate(mlp: *mut JuliaMLP, value: f64) {
    if !mlp.is_null() && !value.is_nan() && !value.is_infinite() && value >= 0.0 {
        unsafe { (*mlp).inner.LearningRate = value; }
    }
}

/// Get optimizer (0=SGD, 1=Adam, 2=RMSProp)
#[no_mangle]
pub extern "C" fn mlp_get_optimizer(mlp: *const JuliaMLP) -> i32 {
    if mlp.is_null() { return 0; }
    unsafe { (*mlp).inner.Optimizer as i32 }
}

/// Set optimizer (0=SGD, 1=Adam, 2=RMSProp)
#[no_mangle]
pub extern "C" fn mlp_set_optimizer(mlp: *mut JuliaMLP, value: i32) {
    if !mlp.is_null() {
        unsafe { (*mlp).inner.Optimizer = int_to_optimizer(value); }
    }
}

/// Get dropout rate
#[no_mangle]
pub extern "C" fn mlp_get_dropout_rate(mlp: *const JuliaMLP) -> f64 {
    if mlp.is_null() { return 0.0; }
    unsafe { (*mlp).inner.DropoutRate }
}

/// Set dropout rate
#[no_mangle]
pub extern "C" fn mlp_set_dropout_rate(mlp: *mut JuliaMLP, value: f64) {
    if !mlp.is_null() && !value.is_nan() && !value.is_infinite() && value >= 0.0 && value <= 1.0 {
        unsafe { (*mlp).inner.DropoutRate = value; }
    }
}

/// Get L2 lambda
#[no_mangle]
pub extern "C" fn mlp_get_l2_lambda(mlp: *const JuliaMLP) -> f64 {
    if mlp.is_null() { return 0.0; }
    unsafe { (*mlp).inner.L2Lambda }
}

/// Set L2 lambda
#[no_mangle]
pub extern "C" fn mlp_set_l2_lambda(mlp: *mut JuliaMLP, value: f64) {
    if !mlp.is_null() && !value.is_nan() && !value.is_infinite() && value >= 0.0 {
        unsafe { (*mlp).inner.L2Lambda = value; }
    }
}

/// Get batch norm flag
#[no_mangle]
pub extern "C" fn mlp_get_batch_norm(mlp: *const JuliaMLP) -> i32 {
    if mlp.is_null() { return 0; }
    if unsafe { (*mlp).inner.UseBatchNorm } { 1 } else { 0 }
}

/// Set batch norm flag
#[no_mangle]
pub extern "C" fn mlp_set_batch_norm(mlp: *mut JuliaMLP, value: i32) {
    if !mlp.is_null() {
        unsafe { (*mlp).inner.UseBatchNorm = value != 0; }
    }
}

/// Get input size
#[no_mangle]
pub extern "C" fn mlp_get_input_size(mlp: *const JuliaMLP) -> i32 {
    if mlp.is_null() { return 0; }
    unsafe { (*mlp).inner.GetInputSize() }
}

/// Get output size
#[no_mangle]
pub extern "C" fn mlp_get_output_size(mlp: *const JuliaMLP) -> i32 {
    if mlp.is_null() { return 0; }
    unsafe { (*mlp).inner.GetOutputSize() }
}

/// Get number of layers
#[no_mangle]
pub extern "C" fn mlp_get_num_layers(mlp: *const JuliaMLP) -> i32 {
    if mlp.is_null() { return 0; }
    unsafe { (*mlp).inner.GetNumLayers() }
}

/// Get hidden layer sizes
/// Returns number of hidden layers. Caller provides buffer.
#[no_mangle]
pub extern "C" fn mlp_get_hidden_sizes(
    mlp: *const JuliaMLP,
    output: *mut i32,
    capacity: i32,
) -> i32 {
    if mlp.is_null() || output.is_null() || capacity <= 0 { return 0; }
    
    let sizes = unsafe { (*mlp).inner.GetHiddenSizes() };
    let len = sizes.len().min(capacity as usize);
    unsafe {
        ptr::copy_nonoverlapping(sizes.as_ptr(), output, len);
    }
    len as i32
}

/// Get GPU backend string
#[no_mangle]
pub extern "C" fn mlp_get_backend(mlp: *const JuliaMLP) -> *const c_char {
    if mlp.is_null() { return ptr::null(); }
    
    let backend = unsafe { (*mlp).inner.get_backend() };
    match backend {
        TGPUBackend::CPU => b"cpu\0".as_ptr() as *const c_char,
        TGPUBackend::CUDA => b"cuda\0".as_ptr() as *const c_char,
        TGPUBackend::OpenCL => b"opencl\0".as_ptr() as *const c_char,
    }
}

/// Set GPU backend
#[no_mangle]
pub extern "C" fn mlp_set_backend(mlp: *mut JuliaMLP, backend: *const c_char) -> i32 {
    if mlp.is_null() || backend.is_null() {
        set_error("Null pointer".to_string());
        return JuliaStatus::InvalidArg as i32;
    }

    let backend_str = unsafe { CStr::from_ptr(backend) }
        .to_str()
        .unwrap_or("cpu");
    let backend_enum = TGPUBackend::from_str(backend_str);

    match unsafe { (*mlp).inner.set_backend(backend_enum) } {
        Ok(()) => JuliaStatus::Ok as i32,
        Err(e) => {
            set_error(e);
            JuliaStatus::RuntimeError as i32
        }
    }
}

/// Get available backends (comma-separated string)
#[no_mangle]
pub extern "C" fn mlp_available_backends() -> *mut c_char {
    let backends: Vec<&str> = detect_available_backends()
        .iter()
        .map(|b| b.to_str())
        .collect();
    
    let result = backends.join(",");
    CString::new(result)
        .map(|s| s.into_raw())
        .unwrap_or(ptr::null_mut())
}

/// Free a string returned by the library
#[no_mangle]
pub extern "C" fn mlp_free_string(s: *mut c_char) {
    if !s.is_null() {
        unsafe { drop(CString::from_raw(s)); }
    }
}

/// Get neuron weights
#[no_mangle]
pub extern "C" fn mlp_get_neuron_weights(
    mlp: *const JuliaMLP,
    layer: i32,
    neuron: i32,
    output: *mut f64,
    capacity: i32,
) -> i32 {
    if mlp.is_null() || output.is_null() || capacity <= 0 || layer < 0 || neuron < 0 { return 0; }
    
    let weights = unsafe { (*mlp).inner.GetNeuronWeights(layer, neuron) };
    let len = weights.len().min(capacity as usize);
    unsafe {
        ptr::copy_nonoverlapping(weights.as_ptr(), output, len);
    }
    len as i32
}

/// Get neuron bias
#[no_mangle]
pub extern "C" fn mlp_get_neuron_bias(mlp: *const JuliaMLP, layer: i32, neuron: i32) -> f64 {
    if mlp.is_null() || layer < 0 || neuron < 0 { return 0.0; }
    unsafe { (*mlp).inner.GetNeuronBias(layer, neuron) }
}

/// Set neuron weight
#[no_mangle]
pub extern "C" fn mlp_set_neuron_weight(
    mlp: *mut JuliaMLP,
    layer: i32,
    neuron: i32,
    weight_idx: i32,
    value: f64,
) {
    if !mlp.is_null() && layer >= 0 && neuron >= 0 && weight_idx >= 0 && !value.is_nan() && !value.is_infinite() {
        unsafe { (*mlp).inner.SetNeuronWeight(layer, neuron, weight_idx, value); }
    }
}

/// Set neuron bias
#[no_mangle]
pub extern "C" fn mlp_set_neuron_bias(mlp: *mut JuliaMLP, layer: i32, neuron: i32, value: f64) {
    if !mlp.is_null() && layer >= 0 && neuron >= 0 && !value.is_nan() && !value.is_infinite() {
        unsafe { (*mlp).inner.SetNeuronBias(layer, neuron, value); }
    }
}

/// Get layer outputs
#[no_mangle]
pub extern "C" fn mlp_get_layer_outputs(
    mlp: *mut JuliaMLP,
    layer: i32,
    output: *mut f64,
    capacity: i32,
) -> i32 {
    if mlp.is_null() || output.is_null() || capacity <= 0 || layer < 0 { return 0; }
    
    let outputs = unsafe { (*mlp).inner.GetLayerOutputs(layer) };
    let len = outputs.len().min(capacity as usize);
    unsafe {
        ptr::copy_nonoverlapping(outputs.as_ptr(), output, len);
    }
    len as i32
}

/// Compute feature importance
#[no_mangle]
pub extern "C" fn mlp_feature_importance(
    mlp: *const JuliaMLP,
    indices: *mut i32,
    scores: *mut f64,
    capacity: i32,
) -> i32 {
    if mlp.is_null() || indices.is_null() || scores.is_null() || capacity <= 0 { return 0; }
    
    let importance = unsafe { (*mlp).inner.compute_feature_importance() };
    let len = importance.len().min(capacity as usize);
    
    for (i, (idx, score)) in importance.iter().take(len).enumerate() {
        unsafe {
            *indices.add(i) = *idx as i32;
            *scores.add(i) = *score;
        }
    }
    
    len as i32
}

/// Get layer errors/gradients (after training)
#[no_mangle]
pub extern "C" fn mlp_get_layer_errors(
    mlp: *mut JuliaMLP,
    layer: i32,
    output: *mut f64,
    capacity: i32,
) -> i32 {
    if mlp.is_null() || output.is_null() || capacity <= 0 || layer < 0 { return 0; }

    let errors = unsafe { (*mlp).inner.GetLayerErrors(layer) };
    let len = errors.len().min(capacity as usize);
    unsafe {
        ptr::copy_nonoverlapping(errors.as_ptr(), output, len);
    }
    len as i32
}

/// Get the size of a layer
#[no_mangle]
pub extern "C" fn mlp_get_layer_size(mlp: *const JuliaMLP, layer: i32) -> i32 {
    if mlp.is_null() || layer < 0 { return 0; }
    unsafe { (*mlp).inner.GetLayerSize(layer as usize) as i32 }
}

/// Get the activation type of a layer (0=Sigmoid, 1=Tanh, 2=ReLU, 3=Softmax)
#[no_mangle]
pub extern "C" fn mlp_get_layer_activation(mlp: *const JuliaMLP, layer: i32) -> i32 {
    if mlp.is_null() || layer < 0 { return 0; }
    unsafe { (*mlp).inner.GetLayerActivation(layer) as i32 }
}

/// Get Adam optimizer's first moment (M) for a weight
#[no_mangle]
pub extern "C" fn mlp_get_weight_m(
    mlp: *const JuliaMLP,
    layer: i32,
    neuron: i32,
    weight_idx: i32,
) -> f64 {
    if mlp.is_null() || layer < 0 || neuron < 0 || weight_idx < 0 { return 0.0; }
    unsafe { (*mlp).inner.GetWeightM(layer, neuron, weight_idx) }
}

/// Get Adam optimizer's second moment (V) for a weight
#[no_mangle]
pub extern "C" fn mlp_get_weight_v(
    mlp: *const JuliaMLP,
    layer: i32,
    neuron: i32,
    weight_idx: i32,
) -> f64 {
    if mlp.is_null() || layer < 0 || neuron < 0 || weight_idx < 0 { return 0.0; }
    unsafe { (*mlp).inner.GetWeightV(layer, neuron, weight_idx) }
}

/// Get Adam optimizer's first moment (M) for a bias
#[no_mangle]
pub extern "C" fn mlp_get_bias_m(mlp: *const JuliaMLP, layer: i32, neuron: i32) -> f64 {
    if mlp.is_null() || layer < 0 || neuron < 0 { return 0.0; }
    unsafe { (*mlp).inner.GetBiasM(layer, neuron) }
}

/// Get Adam optimizer's second moment (V) for a bias
#[no_mangle]
pub extern "C" fn mlp_get_bias_v(mlp: *const JuliaMLP, layer: i32, neuron: i32) -> f64 {
    if mlp.is_null() || layer < 0 || neuron < 0 { return 0.0; }
    unsafe { (*mlp).inner.GetBiasV(layer, neuron) }
}

/// Get activation histogram for a layer
#[no_mangle]
pub extern "C" fn mlp_get_activation_histogram(
    mlp: *const JuliaMLP,
    layer: i32,
    bins: i32,
    output: *mut i32,
    capacity: i32,
) -> i32 {
    if mlp.is_null() || output.is_null() || capacity <= 0 || bins <= 0 || layer < 0 { return 0; }

    let hist = unsafe { (*mlp).inner.GetActivationHistogram(layer, bins as usize) };
    let len = hist.len().min(capacity as usize);
    unsafe {
        ptr::copy_nonoverlapping(hist.as_ptr(), output, len);
    }
    len as i32
}

/// Get gradient histogram for a layer
#[no_mangle]
pub extern "C" fn mlp_get_gradient_histogram(
    mlp: *const JuliaMLP,
    layer: i32,
    bins: i32,
    output: *mut i32,
    capacity: i32,
) -> i32 {
    if mlp.is_null() || output.is_null() || capacity <= 0 || bins <= 0 || layer < 0 { return 0; }

    let hist = unsafe { (*mlp).inner.GetGradientHistogram(layer, bins as usize) };
    let len = hist.len().min(capacity as usize);
    unsafe {
        ptr::copy_nonoverlapping(hist.as_ptr(), output, len);
    }
    len as i32
}

/// Get Adam optimizer timestep
#[no_mangle]
pub extern "C" fn mlp_get_timestep(mlp: *const JuliaMLP) -> i32 {
    if mlp.is_null() { return 0; }
    unsafe { (*mlp).inner.Timestep }
}

// Helper functions

fn int_to_activation(val: i32) -> TActivationType {
    match val {
        0 => TActivationType::atSigmoid,
        1 => TActivationType::atTanh,
        2 => TActivationType::atReLU,
        3 => TActivationType::atSoftmax,
        _ => TActivationType::atSigmoid,
    }
}

fn int_to_optimizer(val: i32) -> TOptimizerType {
    match val {
        0 => TOptimizerType::otSGD,
        1 => TOptimizerType::otAdam,
        2 => TOptimizerType::otRMSProp,
        _ => TOptimizerType::otSGD,
    }
}
