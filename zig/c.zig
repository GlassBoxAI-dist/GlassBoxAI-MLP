// MIT License
// Copyright (c) 2025 Matthew Abbott
//
// C FFI declarations for the GlassBoxAI MLP native library.

pub const MlpHandle = ?*anyopaque;

// Status codes
pub const MLP_OK: i32 = 0;
pub const MLP_INVALID_ARG: i32 = -1;
pub const MLP_RUNTIME_ERROR: i32 = -2;
pub const MLP_IO_ERROR: i32 = -3;

// Activation types
pub const MLP_ACTIVATION_SIGMOID: i32 = 0;
pub const MLP_ACTIVATION_TANH: i32 = 1;
pub const MLP_ACTIVATION_RELU: i32 = 2;
pub const MLP_ACTIVATION_SOFTMAX: i32 = 3;

// Optimizer types
pub const MLP_OPTIMIZER_SGD: i32 = 0;
pub const MLP_OPTIMIZER_ADAM: i32 = 1;
pub const MLP_OPTIMIZER_RMSPROP: i32 = 2;

pub extern fn mlp_get_last_error() ?[*:0]const u8;
pub extern fn mlp_free_error(ptr: ?[*:0]const u8) void;

// Lifecycle
pub extern fn mlp_create(
    input_size: i32,
    hidden_sizes: [*]const i32,
    hidden_count: i32,
    output_size: i32,
    hidden_activation: i32,
    output_activation: i32,
    gpu_backend: ?[*:0]const u8,
) MlpHandle;
pub extern fn mlp_destroy(mlp: MlpHandle) void;

// Core operations
pub extern fn mlp_train(
    mlp: MlpHandle,
    input: [*]const f64,
    input_len: i32,
    target: [*]const f64,
    target_len: i32,
) i32;
pub extern fn mlp_predict(
    mlp: MlpHandle,
    input: [*]const f64,
    input_len: i32,
    output: [*]f64,
    output_capacity: i32,
) i32;
pub extern fn mlp_compute_loss(
    mlp: MlpHandle,
    output: [*]const f64,
    output_len: i32,
    target: [*]const f64,
    target_len: i32,
) f64;

// Serialization
pub extern fn mlp_save(mlp: MlpHandle, filename: [*:0]const u8) i32;
pub extern fn mlp_load(filename: [*:0]const u8) MlpHandle;

// Property getters
pub extern fn mlp_get_learning_rate(mlp: MlpHandle) f64;
pub extern fn mlp_get_optimizer(mlp: MlpHandle) i32;
pub extern fn mlp_get_dropout_rate(mlp: MlpHandle) f64;
pub extern fn mlp_get_l2_lambda(mlp: MlpHandle) f64;
pub extern fn mlp_get_batch_norm(mlp: MlpHandle) i32;
pub extern fn mlp_get_input_size(mlp: MlpHandle) i32;
pub extern fn mlp_get_output_size(mlp: MlpHandle) i32;
pub extern fn mlp_get_num_layers(mlp: MlpHandle) i32;
pub extern fn mlp_get_hidden_sizes(mlp: MlpHandle, output: [*]i32, capacity: i32) i32;
pub extern fn mlp_get_backend(mlp: MlpHandle) ?[*:0]const u8;
pub extern fn mlp_get_timestep(mlp: MlpHandle) i32;

// Property setters
pub extern fn mlp_set_learning_rate(mlp: MlpHandle, value: f64) void;
pub extern fn mlp_set_optimizer(mlp: MlpHandle, value: i32) void;
pub extern fn mlp_set_dropout_rate(mlp: MlpHandle, value: f64) void;
pub extern fn mlp_set_l2_lambda(mlp: MlpHandle, value: f64) void;
pub extern fn mlp_set_batch_norm(mlp: MlpHandle, value: i32) void;
pub extern fn mlp_set_backend(mlp: MlpHandle, backend: [*:0]const u8) i32;

// Backend detection
pub extern fn mlp_available_backends() ?[*:0]u8;
pub extern fn mlp_free_string(s: ?[*:0]u8) void;

// Neuron access
pub extern fn mlp_get_neuron_weights(
    mlp: MlpHandle,
    layer: i32,
    neuron: i32,
    output: [*]f64,
    capacity: i32,
) i32;
pub extern fn mlp_get_neuron_bias(mlp: MlpHandle, layer: i32, neuron: i32) f64;
pub extern fn mlp_set_neuron_weight(
    mlp: MlpHandle,
    layer: i32,
    neuron: i32,
    weight_idx: i32,
    value: f64,
) void;
pub extern fn mlp_set_neuron_bias(mlp: MlpHandle, layer: i32, neuron: i32, value: f64) void;

// Layer introspection
pub extern fn mlp_get_layer_outputs(
    mlp: MlpHandle,
    layer: i32,
    output: [*]f64,
    capacity: i32,
) i32;
pub extern fn mlp_get_layer_errors(
    mlp: MlpHandle,
    layer: i32,
    output: [*]f64,
    capacity: i32,
) i32;
pub extern fn mlp_get_layer_size(mlp: MlpHandle, layer: i32) i32;
pub extern fn mlp_get_layer_activation(mlp: MlpHandle, layer: i32) i32;

// Feature importance
pub extern fn mlp_feature_importance(
    mlp: MlpHandle,
    indices: [*]i32,
    scores: [*]f64,
    capacity: i32,
) i32;

// Adam optimizer state
pub extern fn mlp_get_weight_m(mlp: MlpHandle, layer: i32, neuron: i32, weight_idx: i32) f64;
pub extern fn mlp_get_weight_v(mlp: MlpHandle, layer: i32, neuron: i32, weight_idx: i32) f64;
pub extern fn mlp_get_bias_m(mlp: MlpHandle, layer: i32, neuron: i32) f64;
pub extern fn mlp_get_bias_v(mlp: MlpHandle, layer: i32, neuron: i32) f64;

// Histograms
pub extern fn mlp_get_activation_histogram(
    mlp: MlpHandle,
    layer: i32,
    bins: i32,
    output: [*]i32,
    capacity: i32,
) i32;
pub extern fn mlp_get_gradient_histogram(
    mlp: MlpHandle,
    layer: i32,
    bins: i32,
    output: [*]i32,
    capacity: i32,
) i32;
