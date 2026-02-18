/**
 * @file
 * @ingroup MLP_Wrappers
 */
// Package facadedmlp provides Go bindings for the GlassBoxAI MLP library.
//
// This package provides a GPU-accelerated Multi-Layer Perceptron implementation
// with support for CUDA, OpenCL, and CPU backends. It includes full introspection
// capabilities for explainable AI.
//
// # Quick Start
//
//	mlp, err := facadedmlp.New(2, []int{8}, 1, nil)
//	if err != nil {
//	    log.Fatal(err)
//	}
//	defer mlp.Close()
//
//	// Train on XOR
//	inputs := [][]float64{{0, 0}, {0, 1}, {1, 0}, {1, 1}}
//	targets := [][]float64{{0}, {1}, {1}, {0}}
//
//	result, err := mlp.Fit(inputs, targets, 1000, true)
//	if err != nil {
//	    log.Fatal(err)
//	}
//
//	// Predict
//	output, err := mlp.Predict([]float64{1.0, 0.0})
//	fmt.Printf("Output: %.4f\n", output[0])
package facadedmlp

/*
#cgo LDFLAGS: -L${SRCDIR}/../../target/release -lfacaded_mlp_cuda -lm
#cgo CFLAGS: -I${SRCDIR}/../../cpp/include

#include <stdlib.h>
#include <stdint.h>

// Opaque handle
typedef void* mlp_handle_t;

// FFI functions
extern const char* mlp_get_last_error(void);
extern mlp_handle_t mlp_create(int32_t input_size, const int32_t* hidden_sizes,
    int32_t hidden_count, int32_t output_size, int32_t hidden_activation,
    int32_t output_activation, const char* gpu_backend);
extern void mlp_destroy(mlp_handle_t mlp);
extern int32_t mlp_train(mlp_handle_t mlp, const double* input, int32_t input_len,
    const double* target, int32_t target_len);
extern int32_t mlp_predict(mlp_handle_t mlp, const double* input, int32_t input_len,
    double* output, int32_t output_capacity);
extern double mlp_compute_loss(mlp_handle_t mlp, const double* output, int32_t output_len,
    const double* target, int32_t target_len);
extern int32_t mlp_save(mlp_handle_t mlp, const char* filename);
extern mlp_handle_t mlp_load(const char* filename);

// Property getters
extern double mlp_get_learning_rate(mlp_handle_t mlp);
extern int32_t mlp_get_optimizer(mlp_handle_t mlp);
extern double mlp_get_dropout_rate(mlp_handle_t mlp);
extern double mlp_get_l2_lambda(mlp_handle_t mlp);
extern int32_t mlp_get_batch_norm(mlp_handle_t mlp);
extern int32_t mlp_get_input_size(mlp_handle_t mlp);
extern int32_t mlp_get_output_size(mlp_handle_t mlp);
extern int32_t mlp_get_num_layers(mlp_handle_t mlp);
extern int32_t mlp_get_hidden_sizes(mlp_handle_t mlp, int32_t* output, int32_t capacity);
extern const char* mlp_get_backend(mlp_handle_t mlp);

// Property setters
extern void mlp_set_learning_rate(mlp_handle_t mlp, double value);
extern void mlp_set_optimizer(mlp_handle_t mlp, int32_t value);
extern void mlp_set_dropout_rate(mlp_handle_t mlp, double value);
extern void mlp_set_l2_lambda(mlp_handle_t mlp, double value);
extern void mlp_set_batch_norm(mlp_handle_t mlp, int32_t value);
extern int32_t mlp_set_backend(mlp_handle_t mlp, const char* backend);

// Backend detection
extern char* mlp_available_backends(void);
extern void mlp_free_string(char* s);

// Neuron access
extern int32_t mlp_get_neuron_weights(mlp_handle_t mlp, int32_t layer, int32_t neuron,
    double* output, int32_t capacity);
extern double mlp_get_neuron_bias(mlp_handle_t mlp, int32_t layer, int32_t neuron);
extern void mlp_set_neuron_weight(mlp_handle_t mlp, int32_t layer, int32_t neuron,
    int32_t weight_idx, double value);
extern void mlp_set_neuron_bias(mlp_handle_t mlp, int32_t layer, int32_t neuron, double value);
extern int32_t mlp_get_layer_outputs(mlp_handle_t mlp, int32_t layer, double* output,
    int32_t capacity);
extern int32_t mlp_feature_importance(mlp_handle_t mlp, int32_t* indices, double* scores,
    int32_t capacity);

// Advanced introspection
extern int32_t mlp_get_layer_errors(mlp_handle_t mlp, int32_t layer, double* output,
    int32_t capacity);
extern int32_t mlp_get_layer_size(mlp_handle_t mlp, int32_t layer);
extern int32_t mlp_get_layer_activation(mlp_handle_t mlp, int32_t layer);
extern double mlp_get_weight_m(mlp_handle_t mlp, int32_t layer, int32_t neuron,
    int32_t weight_idx);
extern double mlp_get_weight_v(mlp_handle_t mlp, int32_t layer, int32_t neuron,
    int32_t weight_idx);
extern double mlp_get_bias_m(mlp_handle_t mlp, int32_t layer, int32_t neuron);
extern double mlp_get_bias_v(mlp_handle_t mlp, int32_t layer, int32_t neuron);
extern int32_t mlp_get_activation_histogram(mlp_handle_t mlp, int32_t layer, int32_t bins,
    int32_t* output, int32_t capacity);
extern int32_t mlp_get_gradient_histogram(mlp_handle_t mlp, int32_t layer, int32_t bins,
    int32_t* output, int32_t capacity);
extern int32_t mlp_get_timestep(mlp_handle_t mlp);
extern int32_t mlp_export_onnx(mlp_handle_t mlp, const char* filename);
extern mlp_handle_t mlp_import_onnx(const char* filename, const char* backend);
*/
import "C"

import (
	"errors"
	"fmt"
	"strings"
	"unsafe"
)

// ActivationType represents the activation function for a layer.
type ActivationType int

const (
	Sigmoid ActivationType = iota
	Tanh
	ReLU
	Softmax
)

func (a ActivationType) String() string {
	switch a {
	case Sigmoid:
		return "sigmoid"
	case Tanh:
		return "tanh"
	case ReLU:
		return "relu"
	case Softmax:
		return "softmax"
	default:
		return "unknown"
	}
}

// OptimizerType represents the optimizer algorithm.
type OptimizerType int

const (
	SGD OptimizerType = iota
	Adam
	RMSProp
)

func (o OptimizerType) String() string {
	switch o {
	case SGD:
		return "sgd"
	case Adam:
		return "adam"
	case RMSProp:
		return "rmsprop"
	default:
		return "unknown"
	}
}

// BackendType represents the compute backend.
type BackendType string

const (
	BackendAuto   BackendType = "auto"
	BackendCPU    BackendType = "cpu"
	BackendCUDA   BackendType = "cuda"
	BackendOpenCL BackendType = "opencl"
)

// Config holds configuration options for creating an MLP.
type Config struct {
	HiddenActivation ActivationType
	OutputActivation ActivationType
	Backend          BackendType
	LearningRate     float64
	Optimizer        OptimizerType
	DropoutRate      float64
	L2Lambda         float64
	BatchNorm        bool
}

// DefaultConfig returns the default configuration.
func DefaultConfig() *Config {
	return &Config{
		HiddenActivation: Sigmoid,
		OutputActivation: Sigmoid,
		Backend:          BackendAuto,
		LearningRate:     0.01,
		Optimizer:        Adam,
		DropoutRate:      0.0,
		L2Lambda:         0.0,
		BatchNorm:        false,
	}
}

// TrainResult holds the results of training.
type TrainResult struct {
	Losses    []float64
	FinalLoss float64
}

// FeatureImportance holds feature importance data.
type FeatureImportance struct {
	Index int
	Score float64
}

// LayerInfo holds information about a layer.
type LayerInfo struct {
	Index            int
	Size             int
	WeightsPerNeuron int
	Activation       ActivationType
}

// NeuronView provides a view into a single neuron.
type NeuronView struct {
	Layer   int
	Index   int
	Weights []float64
	Bias    float64
	Output  float64
	Error   float64
}

// MLP represents a Multi-Layer Perceptron neural network.
type MLP struct {
	handle     C.mlp_handle_t
	inputSize  int
	outputSize int
	hiddenSizes []int
}

// getLastError returns the last error message from the library.
func getLastError() error {
	errPtr := C.mlp_get_last_error()
	if errPtr == nil {
		return errors.New("unknown error")
	}
	return errors.New(C.GoString(errPtr))
}

// New creates a new MLP with the specified architecture.
//
// Parameters:
//   - inputSize: number of input neurons
//   - hiddenSizes: sizes of hidden layers (e.g., []int{8, 8} for two layers of 8 neurons)
//   - outputSize: number of output neurons
//   - config: optional configuration (pass nil for defaults)
//
// Example:
//
//	mlp, err := facadedmlp.New(2, []int{8}, 1, nil)
//	if err != nil {
//	    log.Fatal(err)
//	}
//	defer mlp.Close()
func New(inputSize int, hiddenSizes []int, outputSize int, config *Config) (*MLP, error) {
	if config == nil {
		config = DefaultConfig()
	}

	if len(hiddenSizes) == 0 {
		return nil, errors.New("must have at least one hidden layer")
	}

	// Convert hidden sizes to C array
	cHidden := make([]C.int32_t, len(hiddenSizes))
	for i, s := range hiddenSizes {
		cHidden[i] = C.int32_t(s)
	}

	// Convert backend string
	var cBackend *C.char
	if config.Backend != "" {
		cBackend = C.CString(string(config.Backend))
		defer C.free(unsafe.Pointer(cBackend))
	}

	handle := C.mlp_create(
		C.int32_t(inputSize),
		&cHidden[0],
		C.int32_t(len(hiddenSizes)),
		C.int32_t(outputSize),
		C.int32_t(config.HiddenActivation),
		C.int32_t(config.OutputActivation),
		cBackend,
	)

	if handle == nil {
		return nil, getLastError()
	}

	mlp := &MLP{
		handle:     handle,
		inputSize:  inputSize,
		outputSize: outputSize,
		hiddenSizes: hiddenSizes,
	}

	// Apply additional configuration
	mlp.SetLearningRate(config.LearningRate)
	mlp.SetOptimizer(config.Optimizer)
	mlp.SetDropoutRate(config.DropoutRate)
	mlp.SetL2Lambda(config.L2Lambda)
	mlp.SetBatchNorm(config.BatchNorm)

	return mlp, nil
}

// Load loads an MLP from a JSON file.
func Load(filename string) (*MLP, error) {
	cFilename := C.CString(filename)
	defer C.free(unsafe.Pointer(cFilename))

	handle := C.mlp_load(cFilename)
	if handle == nil {
		return nil, getLastError()
	}

	mlp := &MLP{handle: handle}
	mlp.inputSize = int(C.mlp_get_input_size(handle))
	mlp.outputSize = int(C.mlp_get_output_size(handle))
	mlp.hiddenSizes = mlp.getHiddenSizesInternal()

	return mlp, nil
}

// Close releases the MLP resources. Always call this when done.
func (m *MLP) Close() {
	if m.handle != nil {
		C.mlp_destroy(m.handle)
		m.handle = nil
	}
}

// Predict makes a prediction for the given input.
func (m *MLP) Predict(input []float64) ([]float64, error) {
	if len(input) != m.inputSize {
		return nil, fmt.Errorf("input size mismatch: expected %d, got %d", m.inputSize, len(input))
	}

	output := make([]float64, m.outputSize)
	
	result := C.mlp_predict(
		m.handle,
		(*C.double)(unsafe.Pointer(&input[0])),
		C.int32_t(len(input)),
		(*C.double)(unsafe.Pointer(&output[0])),
		C.int32_t(m.outputSize),
	)

	if result < 0 {
		return nil, getLastError()
	}

	return output[:result], nil
}

// Train trains the network on a single sample.
func (m *MLP) Train(input, target []float64) error {
	if len(input) != m.inputSize {
		return fmt.Errorf("input size mismatch: expected %d, got %d", m.inputSize, len(input))
	}
	if len(target) != m.outputSize {
		return fmt.Errorf("target size mismatch: expected %d, got %d", m.outputSize, len(target))
	}

	result := C.mlp_train(
		m.handle,
		(*C.double)(unsafe.Pointer(&input[0])),
		C.int32_t(len(input)),
		(*C.double)(unsafe.Pointer(&target[0])),
		C.int32_t(len(target)),
	)

	if result != 0 {
		return getLastError()
	}

	return nil
}

// ComputeLoss computes the loss between output and target.
func (m *MLP) ComputeLoss(output, target []float64) float64 {
	return float64(C.mlp_compute_loss(
		m.handle,
		(*C.double)(unsafe.Pointer(&output[0])),
		C.int32_t(len(output)),
		(*C.double)(unsafe.Pointer(&target[0])),
		C.int32_t(len(target)),
	))
}

// Fit trains the network on a dataset for multiple epochs.
func (m *MLP) Fit(inputs, targets [][]float64, epochs int, verbose bool) (*TrainResult, error) {
	if len(inputs) != len(targets) {
		return nil, errors.New("inputs and targets must have same length")
	}

	losses := make([]float64, 0, epochs)

	for epoch := 0; epoch < epochs; epoch++ {
		var epochLoss float64

		for i := range inputs {
			if err := m.Train(inputs[i], targets[i]); err != nil {
				return nil, err
			}
			output, err := m.Predict(inputs[i])
			if err != nil {
				return nil, err
			}
			epochLoss += m.ComputeLoss(output, targets[i])
		}

		epochLoss /= float64(len(inputs))
		losses = append(losses, epochLoss)

		if verbose && (epoch%100 == 0 || epoch == epochs-1) {
			fmt.Printf("Epoch %d/%d - Loss: %.6f\n", epoch+1, epochs, epochLoss)
		}
	}

	finalLoss := 0.0
	if len(losses) > 0 {
		finalLoss = losses[len(losses)-1]
	}

	return &TrainResult{
		Losses:    losses,
		FinalLoss: finalLoss,
	}, nil
}

// PredictBatch makes predictions for multiple inputs.
func (m *MLP) PredictBatch(inputs [][]float64) ([][]float64, error) {
	outputs := make([][]float64, len(inputs))
	for i, input := range inputs {
		output, err := m.Predict(input)
		if err != nil {
			return nil, err
		}
		outputs[i] = output
	}
	return outputs, nil
}

// Save saves the model to a JSON file.
func (m *MLP) Save(filename string) error {
	cFilename := C.CString(filename)
	defer C.free(unsafe.Pointer(cFilename))

	result := C.mlp_save(m.handle, cFilename)
	if result != 0 {
		return getLastError()
	}
	return nil
}

// Properties

// InputSize returns the number of input neurons.
func (m *MLP) InputSize() int {
	return m.inputSize
}

// OutputSize returns the number of output neurons.
func (m *MLP) OutputSize() int {
	return m.outputSize
}

// HiddenSizes returns the sizes of hidden layers.
func (m *MLP) HiddenSizes() []int {
	return m.hiddenSizes
}

// NumLayers returns the total number of layers (including input and output).
func (m *MLP) NumLayers() int {
	return int(C.mlp_get_num_layers(m.handle))
}

func (m *MLP) getHiddenSizesInternal() []int {
	sizes := make([]C.int32_t, 100)
	count := C.mlp_get_hidden_sizes(m.handle, &sizes[0], 100)
	result := make([]int, count)
	for i := 0; i < int(count); i++ {
		result[i] = int(sizes[i])
	}
	return result
}

// LearningRate returns the current learning rate.
func (m *MLP) LearningRate() float64 {
	return float64(C.mlp_get_learning_rate(m.handle))
}

// SetLearningRate sets the learning rate.
func (m *MLP) SetLearningRate(value float64) {
	C.mlp_set_learning_rate(m.handle, C.double(value))
}

// Optimizer returns the current optimizer type.
func (m *MLP) Optimizer() OptimizerType {
	return OptimizerType(C.mlp_get_optimizer(m.handle))
}

// SetOptimizer sets the optimizer type.
func (m *MLP) SetOptimizer(opt OptimizerType) {
	C.mlp_set_optimizer(m.handle, C.int32_t(opt))
}

// DropoutRate returns the dropout rate.
func (m *MLP) DropoutRate() float64 {
	return float64(C.mlp_get_dropout_rate(m.handle))
}

// SetDropoutRate sets the dropout rate.
func (m *MLP) SetDropoutRate(value float64) {
	C.mlp_set_dropout_rate(m.handle, C.double(value))
}

// L2Lambda returns the L2 regularization lambda.
func (m *MLP) L2Lambda() float64 {
	return float64(C.mlp_get_l2_lambda(m.handle))
}

// SetL2Lambda sets the L2 regularization lambda.
func (m *MLP) SetL2Lambda(value float64) {
	C.mlp_set_l2_lambda(m.handle, C.double(value))
}

// BatchNorm returns whether batch normalization is enabled.
func (m *MLP) BatchNorm() bool {
	return C.mlp_get_batch_norm(m.handle) != 0
}

// SetBatchNorm enables or disables batch normalization.
func (m *MLP) SetBatchNorm(value bool) {
	v := C.int32_t(0)
	if value {
		v = 1
	}
	C.mlp_set_batch_norm(m.handle, v)
}

// Backend returns the current GPU backend.
func (m *MLP) Backend() string {
	ptr := C.mlp_get_backend(m.handle)
	if ptr == nil {
		return "unknown"
	}
	return C.GoString(ptr)
}

// SetBackend sets the GPU backend.
func (m *MLP) SetBackend(backend BackendType) error {
	cBackend := C.CString(string(backend))
	defer C.free(unsafe.Pointer(cBackend))

	result := C.mlp_set_backend(m.handle, cBackend)
	if result != 0 {
		return getLastError()
	}
	return nil
}

// Introspection

// GetNeuronWeights returns the weights for a specific neuron.
func (m *MLP) GetNeuronWeights(layer, neuron int) []float64 {
	// Estimate max weights based on previous layer size
	maxWeights := m.inputSize
	if layer > 1 && layer-1 <= len(m.hiddenSizes) {
		maxWeights = m.hiddenSizes[layer-2]
	}

	weights := make([]float64, maxWeights)
	count := C.mlp_get_neuron_weights(
		m.handle,
		C.int32_t(layer),
		C.int32_t(neuron),
		(*C.double)(unsafe.Pointer(&weights[0])),
		C.int32_t(maxWeights),
	)
	return weights[:count]
}

// GetNeuronBias returns the bias for a specific neuron.
func (m *MLP) GetNeuronBias(layer, neuron int) float64 {
	return float64(C.mlp_get_neuron_bias(m.handle, C.int32_t(layer), C.int32_t(neuron)))
}

// SetNeuronWeight sets a specific weight.
func (m *MLP) SetNeuronWeight(layer, neuron, weightIdx int, value float64) {
	C.mlp_set_neuron_weight(m.handle, C.int32_t(layer), C.int32_t(neuron),
		C.int32_t(weightIdx), C.double(value))
}

// SetNeuronBias sets a neuron's bias.
func (m *MLP) SetNeuronBias(layer, neuron int, value float64) {
	C.mlp_set_neuron_bias(m.handle, C.int32_t(layer), C.int32_t(neuron), C.double(value))
}

// GetLayerOutputs returns the outputs of all neurons in a layer (after prediction).
func (m *MLP) GetLayerOutputs(layer int) []float64 {
	var layerSize int
	if layer == 0 {
		layerSize = m.inputSize
	} else if layer <= len(m.hiddenSizes) {
		layerSize = m.hiddenSizes[layer-1]
	} else {
		layerSize = m.outputSize
	}

	outputs := make([]float64, layerSize)
	count := C.mlp_get_layer_outputs(
		m.handle,
		C.int32_t(layer),
		(*C.double)(unsafe.Pointer(&outputs[0])),
		C.int32_t(layerSize),
	)
	return outputs[:count]
}

// NeuronView returns a detailed view of a specific neuron.
func (m *MLP) NeuronView(layer, neuron int) *NeuronView {
	outputs := m.GetLayerOutputs(layer)
	errors := m.GetLayerErrors(layer)
	var outputVal, errorVal float64
	if neuron < len(outputs) {
		outputVal = outputs[neuron]
	}
	if neuron < len(errors) {
		errorVal = errors[neuron]
	}
	return &NeuronView{
		Layer:   layer,
		Index:   neuron,
		Weights: m.GetNeuronWeights(layer, neuron),
		Bias:    m.GetNeuronBias(layer, neuron),
		Output:  outputVal,
		Error:   errorVal,
	}
}

// FeatureImportance computes feature importance based on input layer weights.
func (m *MLP) FeatureImportance() []FeatureImportance {
	indices := make([]C.int32_t, m.inputSize)
	scores := make([]float64, m.inputSize)

	count := C.mlp_feature_importance(
		m.handle,
		&indices[0],
		(*C.double)(unsafe.Pointer(&scores[0])),
		C.int32_t(m.inputSize),
	)

	result := make([]FeatureImportance, count)
	for i := 0; i < int(count); i++ {
		result[i] = FeatureImportance{
			Index: int(indices[i]),
			Score: scores[i],
		}
	}
	return result
}

// Timestep returns the current Adam optimizer timestep.
func (m *MLP) Timestep() int {
	return int(C.mlp_get_timestep(m.handle))
}

// GetLayerErrors returns the error/gradient values for all neurons in a layer.
func (m *MLP) GetLayerErrors(layer int) []float64 {
	size := m.GetLayerSize(layer)
	if size <= 0 {
		return nil
	}
	errs := make([]float64, size)
	count := C.mlp_get_layer_errors(
		m.handle,
		C.int32_t(layer),
		(*C.double)(unsafe.Pointer(&errs[0])),
		C.int32_t(size),
	)
	return errs[:count]
}

// GetLayerSize returns the number of neurons in a layer.
func (m *MLP) GetLayerSize(layer int) int {
	return int(C.mlp_get_layer_size(m.handle, C.int32_t(layer)))
}

// GetLayerActivation returns the activation function used by a layer.
func (m *MLP) GetLayerActivation(layer int) ActivationType {
	return ActivationType(C.mlp_get_layer_activation(m.handle, C.int32_t(layer)))
}

// GetLayerInfo returns combined metadata about a layer.
func (m *MLP) GetLayerInfo(layer int) LayerInfo {
	size := m.GetLayerSize(layer)
	weightsPerNeuron := 0
	if layer > 0 {
		weightsPerNeuron = m.GetLayerSize(layer - 1)
	}
	return LayerInfo{
		Index:            layer,
		Size:             size,
		WeightsPerNeuron: weightsPerNeuron,
		Activation:       m.GetLayerActivation(layer),
	}
}

// GetWeightM returns the Adam first moment (M) for a specific weight.
func (m *MLP) GetWeightM(layer, neuron, weightIdx int) float64 {
	return float64(C.mlp_get_weight_m(m.handle, C.int32_t(layer), C.int32_t(neuron),
		C.int32_t(weightIdx)))
}

// GetWeightV returns the Adam second moment (V) for a specific weight.
func (m *MLP) GetWeightV(layer, neuron, weightIdx int) float64 {
	return float64(C.mlp_get_weight_v(m.handle, C.int32_t(layer), C.int32_t(neuron),
		C.int32_t(weightIdx)))
}

// GetBiasM returns the Adam first moment (M) for a specific bias.
func (m *MLP) GetBiasM(layer, neuron int) float64 {
	return float64(C.mlp_get_bias_m(m.handle, C.int32_t(layer), C.int32_t(neuron)))
}

// GetBiasV returns the Adam second moment (V) for a specific bias.
func (m *MLP) GetBiasV(layer, neuron int) float64 {
	return float64(C.mlp_get_bias_v(m.handle, C.int32_t(layer), C.int32_t(neuron)))
}

// GetActivationHistogram returns a histogram of activation values for a layer.
func (m *MLP) GetActivationHistogram(layer, bins int) []int {
	hist := make([]C.int32_t, bins)
	count := C.mlp_get_activation_histogram(
		m.handle,
		C.int32_t(layer),
		C.int32_t(bins),
		&hist[0],
		C.int32_t(bins),
	)
	result := make([]int, count)
	for i := 0; i < int(count); i++ {
		result[i] = int(hist[i])
	}
	return result
}

// GetGradientHistogram returns a histogram of gradient values for a layer.
func (m *MLP) GetGradientHistogram(layer, bins int) []int {
	hist := make([]C.int32_t, bins)
	count := C.mlp_get_gradient_histogram(
		m.handle,
		C.int32_t(layer),
		C.int32_t(bins),
		&hist[0],
		C.int32_t(bins),
	)
	result := make([]int, count)
	for i := 0; i < int(count); i++ {
		result[i] = int(hist[i])
	}
	return result
}

// ExportONNX exports the model to an ONNX file.
func (m *MLP) ExportONNX(filename string) error {
	cFilename := C.CString(filename)
	defer C.free(unsafe.Pointer(cFilename))
	result := C.mlp_export_onnx(m.handle, cFilename)
	if result != 0 {
		return getLastError()
	}
	return nil
}

// ImportONNX loads an MLP from an ONNX file and returns a new instance.
func ImportONNX(filename string, backend BackendType) (*MLP, error) {
	cFilename := C.CString(filename)
	defer C.free(unsafe.Pointer(cFilename))
	cBackend := C.CString(string(backend))
	defer C.free(unsafe.Pointer(cBackend))

	handle := C.mlp_import_onnx(cFilename, cBackend)
	if handle == nil {
		return nil, getLastError()
	}

	mlp := &MLP{handle: handle}
	mlp.inputSize = int(C.mlp_get_input_size(handle))
	mlp.outputSize = int(C.mlp_get_output_size(handle))
	mlp.hiddenSizes = mlp.getHiddenSizesInternal()
	return mlp, nil
}

// Utility functions

// AvailableBackends returns a list of available GPU backends.
func AvailableBackends() []string {
	ptr := C.mlp_available_backends()
	if ptr == nil {
		return []string{"cpu"}
	}
	result := C.GoString(ptr)
	C.mlp_free_string(ptr)
	return strings.Split(result, ",")
}

// String returns a string representation of the MLP.
func (m *MLP) String() string {
	return fmt.Sprintf("MLP(input=%d, hidden=%v, output=%d, lr=%.4f, optimizer=%s, backend=%s)",
		m.inputSize, m.hiddenSizes, m.outputSize, m.LearningRate(), m.Optimizer(), m.Backend())
}
