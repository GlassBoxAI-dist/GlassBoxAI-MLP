/**
 * @file
 * @ingroup MLP_Internal_Logic
 */
/*
 * MIT License
 * Copyright (c) 2025 Matthew Abbott
 *
 * C API for Facaded MLP CUDA/OpenCL library
 */

#ifndef FACADED_MLP_H
#define FACADED_MLP_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Opaque handle to MLP */
typedef void* mlp_handle_t;

/* Status codes */
typedef enum {
    MLP_OK = 0,
    MLP_INVALID_ARG = -1,
    MLP_RUNTIME_ERROR = -2,
    MLP_IO_ERROR = -3
} mlp_status_t;

/* Activation types */
typedef enum {
    MLP_ACTIVATION_SIGMOID = 0,
    MLP_ACTIVATION_TANH = 1,
    MLP_ACTIVATION_RELU = 2,
    MLP_ACTIVATION_SOFTMAX = 3
} mlp_activation_t;

/* Optimizer types */
typedef enum {
    MLP_OPTIMIZER_SGD = 0,
    MLP_OPTIMIZER_ADAM = 1,
    MLP_OPTIMIZER_RMSPROP = 2
} mlp_optimizer_t;

/*
 * Get last error message.
 * Returns NULL if no error.
 */
const char* mlp_get_last_error(void);

/*
 * Free error string (no-op, provided for API symmetry).
 */
void mlp_free_error(const char* ptr);

/*
 * Create a new MLP.
 *
 * @param input_size Number of input neurons
 * @param hidden_sizes Array of hidden layer sizes
 * @param hidden_count Number of hidden layers
 * @param output_size Number of output neurons
 * @param hidden_activation Activation for hidden layers
 * @param output_activation Activation for output layer
 * @param gpu_backend Backend: "auto", "cpu", "cuda", "opencl" (NULL for auto)
 * @return Handle to MLP, or NULL on error
 */
mlp_handle_t mlp_create(
    int32_t input_size,
    const int32_t* hidden_sizes,
    int32_t hidden_count,
    int32_t output_size,
    int32_t hidden_activation,
    int32_t output_activation,
    const char* gpu_backend
);

/*
 * Destroy an MLP instance.
 */
void mlp_destroy(mlp_handle_t mlp);

/*
 * Train on a single sample.
 *
 * @return 0 on success, negative on error
 */
int32_t mlp_train(
    mlp_handle_t mlp,
    const double* input,
    int32_t input_len,
    const double* target,
    int32_t target_len
);

/*
 * Make a prediction.
 *
 * @param output Buffer to receive output (must be at least output_size)
 * @param output_capacity Size of output buffer
 * @return Number of outputs written, or negative on error
 */
int32_t mlp_predict(
    mlp_handle_t mlp,
    const double* input,
    int32_t input_len,
    double* output,
    int32_t output_capacity
);

/*
 * Compute loss for given output and target.
 *
 * @return Loss value, or NaN on error
 */
double mlp_compute_loss(
    mlp_handle_t mlp,
    const double* output,
    int32_t output_len,
    const double* target,
    int32_t target_len
);

/*
 * Save model to file.
 *
 * @return 0 on success, negative on error
 */
int32_t mlp_save(mlp_handle_t mlp, const char* filename);

/*
 * Load model from file.
 *
 * @return Handle to MLP, or NULL on error
 */
mlp_handle_t mlp_load(const char* filename);

/* Property getters */
double mlp_get_learning_rate(mlp_handle_t mlp);
int32_t mlp_get_optimizer(mlp_handle_t mlp);
double mlp_get_dropout_rate(mlp_handle_t mlp);
double mlp_get_l2_lambda(mlp_handle_t mlp);
int32_t mlp_get_batch_norm(mlp_handle_t mlp);
int32_t mlp_get_input_size(mlp_handle_t mlp);
int32_t mlp_get_output_size(mlp_handle_t mlp);
int32_t mlp_get_num_layers(mlp_handle_t mlp);

/* Property setters */
void mlp_set_learning_rate(mlp_handle_t mlp, double value);
void mlp_set_optimizer(mlp_handle_t mlp, int32_t value);
void mlp_set_dropout_rate(mlp_handle_t mlp, double value);
void mlp_set_l2_lambda(mlp_handle_t mlp, double value);
void mlp_set_batch_norm(mlp_handle_t mlp, int32_t value);

/*
 * Get hidden layer sizes.
 *
 * @param output Buffer to receive sizes
 * @param capacity Size of buffer
 * @return Number of hidden layers
 */
int32_t mlp_get_hidden_sizes(mlp_handle_t mlp, int32_t* output, int32_t capacity);

/*
 * Get current GPU backend.
 *
 * @return Static string: "cpu", "cuda", or "opencl"
 */
const char* mlp_get_backend(mlp_handle_t mlp);

/*
 * Set GPU backend.
 *
 * @param backend "cpu", "cuda", or "opencl"
 * @return 0 on success, negative on error
 */
int32_t mlp_set_backend(mlp_handle_t mlp, const char* backend);

/*
 * Get available backends (comma-separated).
 *
 * @return Allocated string (caller must free with mlp_free_string)
 */
char* mlp_available_backends(void);

/*
 * Free a string returned by the library.
 */
void mlp_free_string(char* s);

/*
 * Get neuron weights.
 *
 * @param layer Layer index (0 = input, 1+ = hidden/output)
 * @param neuron Neuron index within layer
 * @param output Buffer to receive weights
 * @param capacity Size of buffer
 * @return Number of weights
 */
int32_t mlp_get_neuron_weights(
    mlp_handle_t mlp,
    int32_t layer,
    int32_t neuron,
    double* output,
    int32_t capacity
);

/*
 * Get neuron bias.
 */
double mlp_get_neuron_bias(mlp_handle_t mlp, int32_t layer, int32_t neuron);

/*
 * Set a specific weight.
 */
void mlp_set_neuron_weight(
    mlp_handle_t mlp,
    int32_t layer,
    int32_t neuron,
    int32_t weight_idx,
    double value
);

/*
 * Set a neuron's bias.
 */
void mlp_set_neuron_bias(mlp_handle_t mlp, int32_t layer, int32_t neuron, double value);

/*
 * Get layer outputs (after prediction).
 *
 * @param layer Layer index
 * @param output Buffer to receive outputs
 * @param capacity Size of buffer
 * @return Number of outputs
 */
int32_t mlp_get_layer_outputs(
    mlp_handle_t mlp,
    int32_t layer,
    double* output,
    int32_t capacity
);

/*
 * Compute feature importance.
 *
 * @param indices Buffer to receive feature indices (sorted by importance)
 * @param scores Buffer to receive importance scores
 * @param capacity Size of buffers
 * @return Number of features
 */
int32_t mlp_feature_importance(
    mlp_handle_t mlp,
    int32_t* indices,
    double* scores,
    int32_t capacity
);

/*
 * Get layer errors/gradients (after training).
 *
 * @param layer Layer index
 * @param output Buffer to receive errors
 * @param capacity Size of buffer
 * @return Number of errors
 */
int32_t mlp_get_layer_errors(
    mlp_handle_t mlp,
    int32_t layer,
    double* output,
    int32_t capacity
);

/*
 * Get the size (number of neurons) of a layer.
 *
 * @param layer Layer index (0 = input, 1+ = hidden/output)
 * @return Number of neurons in the layer
 */
int32_t mlp_get_layer_size(mlp_handle_t mlp, int32_t layer);

/*
 * Get the activation type of a layer.
 *
 * @param layer Layer index
 * @return Activation type (0=Sigmoid, 1=Tanh, 2=ReLU, 3=Softmax)
 */
int32_t mlp_get_layer_activation(mlp_handle_t mlp, int32_t layer);

/*
 * Get Adam optimizer's first moment (M) for a weight.
 */
double mlp_get_weight_m(mlp_handle_t mlp, int32_t layer, int32_t neuron, int32_t weight_idx);

/*
 * Get Adam optimizer's second moment (V) for a weight.
 */
double mlp_get_weight_v(mlp_handle_t mlp, int32_t layer, int32_t neuron, int32_t weight_idx);

/*
 * Get Adam optimizer's first moment (M) for a bias.
 */
double mlp_get_bias_m(mlp_handle_t mlp, int32_t layer, int32_t neuron);

/*
 * Get Adam optimizer's second moment (V) for a bias.
 */
double mlp_get_bias_v(mlp_handle_t mlp, int32_t layer, int32_t neuron);

/*
 * Get activation histogram for a layer.
 *
 * @param layer Layer index
 * @param bins Number of histogram bins
 * @param output Buffer to receive histogram counts
 * @param capacity Size of buffer
 * @return Number of bins written
 */
int32_t mlp_get_activation_histogram(
    mlp_handle_t mlp,
    int32_t layer,
    int32_t bins,
    int32_t* output,
    int32_t capacity
);

/*
 * Get gradient histogram for a layer.
 *
 * @param layer Layer index
 * @param bins Number of histogram bins
 * @param output Buffer to receive histogram counts
 * @param capacity Size of buffer
 * @return Number of bins written
 */
int32_t mlp_get_gradient_histogram(
    mlp_handle_t mlp,
    int32_t layer,
    int32_t bins,
    int32_t* output,
    int32_t capacity
);

/*
 * Get Adam optimizer timestep.
 *
 * @return Current timestep value
 */
int32_t mlp_get_timestep(mlp_handle_t mlp);

/*
 * Set Adam optimizer's first moment (M) for a weight.
 */
void mlp_set_weight_m(mlp_handle_t mlp, int32_t layer, int32_t neuron, int32_t weight_idx, double value);

/*
 * Set Adam optimizer's second moment (V) for a weight.
 */
void mlp_set_weight_v(mlp_handle_t mlp, int32_t layer, int32_t neuron, int32_t weight_idx, double value);

/*
 * Set Adam optimizer's first moment (M) for a bias.
 */
void mlp_set_bias_m(mlp_handle_t mlp, int32_t layer, int32_t neuron, double value);

/*
 * Set Adam optimizer's second moment (V) for a bias.
 */
void mlp_set_bias_v(mlp_handle_t mlp, int32_t layer, int32_t neuron, double value);

/*
 * Set Adam optimizer timestep.
 */
void mlp_set_timestep(mlp_handle_t mlp, int32_t value);

/*
 * Set the activation type of a layer.
 *
 * @param layer Layer index
 * @param activation Activation type (0=Sigmoid, 1=Tanh, 2=ReLU, 3=Softmax)
 */
void mlp_set_layer_activation(mlp_handle_t mlp, int32_t layer, int32_t activation);

/*
 * Set all weights for a neuron.
 *
 * @param layer Layer index
 * @param neuron Neuron index
 * @param weights Array of weight values
 * @param weights_len Number of weights
 * @return 0 on success, negative on error
 */
int32_t mlp_set_neuron_weights(mlp_handle_t mlp, int32_t layer, int32_t neuron,
    const double* weights, int32_t weights_len);

/*
 * Export model to an ONNX file.
 *
 * @param mlp      Model handle
 * @param filename Destination path (UTF-8)
 * @return 0 on success, negative on error. Call mlp_get_last_error() for details.
 */
int32_t mlp_export_onnx(mlp_handle_t mlp, const char* filename);

/*
 * Import a model from an ONNX file and create a new handle.
 *
 * @param filename Path to the ONNX file (UTF-8)
 * @param backend  GPU backend string: "auto", "cpu", "cuda", "opencl"
 * @return New model handle, or NULL on error. Call mlp_get_last_error() for details.
 *         Caller must free with mlp_destroy().
 */
mlp_handle_t mlp_import_onnx(const char* filename, const char* backend);

#ifdef __cplusplus
}
#endif

#endif /* FACADED_MLP_H */
