# FacadedMLP Go Wrapper

Go bindings for the GPU-accelerated Multi-Layer Perceptron library with CUDA/OpenCL/CPU backends.

## Features

- Full MLP functionality (create, train, predict, save/load)
- GPU acceleration (CUDA, OpenCL) with CPU fallback
- Complete introspection API (weights, biases, activations)
- Feature importance analysis
- Idiomatic Go API with proper error handling

## Installation

### 1. Build the Rust library

```bash
cd /path/to/GlassBoxAI-MLP
cargo build --release --features julia
```

### 2. Set up Go module

```bash
cd go
go mod tidy
```

### 3. Set library path

```bash
export LD_LIBRARY_PATH=/path/to/GlassBoxAI-MLP/target/release:$LD_LIBRARY_PATH
```

Or on macOS:
```bash
export DYLD_LIBRARY_PATH=/path/to/GlassBoxAI-MLP/target/release:$DYLD_LIBRARY_PATH
```

## Quick Start

```go
package main

import (
    "fmt"
    "log"
    
    "github.com/GlassBoxAI-dist/GlassBoxAI-MLP/go/facadedmlp"
)

func main() {
    // Create a network: 2 inputs, 8 hidden neurons, 1 output
    mlp, err := facadedmlp.New(2, []int{8}, 1, nil)
    if err != nil {
        log.Fatal(err)
    }
    defer mlp.Close()  // Always close when done!
    
    // XOR training data
    inputs := [][]float64{{0, 0}, {0, 1}, {1, 0}, {1, 1}}
    targets := [][]float64{{0}, {1}, {1}, {0}}
    
    // Configure and train
    mlp.SetLearningRate(0.5)
    mlp.SetOptimizer(facadedmlp.Adam)
    
    result, err := mlp.Fit(inputs, targets, 1000, true)
    if err != nil {
        log.Fatal(err)
    }
    fmt.Printf("Final loss: %.6f\n", result.FinalLoss)
    
    // Predict
    output, _ := mlp.Predict([]float64{1.0, 0.0})
    fmt.Printf("Prediction: %.4f\n", output[0])
}
```

## API Reference

### Creating Models

```go
// With defaults
mlp, err := facadedmlp.New(inputSize, hiddenSizes, outputSize, nil)

// With custom configuration
config := &facadedmlp.Config{
    HiddenActivation: facadedmlp.ReLU,
    OutputActivation: facadedmlp.Sigmoid,
    Backend:          facadedmlp.BackendCUDA,
    LearningRate:     0.001,
    Optimizer:        facadedmlp.Adam,
    DropoutRate:      0.2,
    L2Lambda:         0.0001,
    BatchNorm:        true,
}
mlp, err := facadedmlp.New(784, []int{256, 128}, 10, config)
```

### Training

```go
// Single sample
err := mlp.Train(input, target)

// Full dataset
result, err := mlp.Fit(inputs, targets, epochs, verbose)
fmt.Printf("Final loss: %f\n", result.FinalLoss)
```

### Prediction

```go
output, err := mlp.Predict(input)
outputs, err := mlp.PredictBatch(inputs)
```

### Properties

```go
// Getters
lr := mlp.LearningRate()
opt := mlp.Optimizer()
dr := mlp.DropoutRate()
backend := mlp.Backend()
inputSize := mlp.InputSize()
outputSize := mlp.OutputSize()
hidden := mlp.HiddenSizes()

// Setters
mlp.SetLearningRate(0.001)
mlp.SetOptimizer(facadedmlp.SGD)
mlp.SetDropoutRate(0.2)
mlp.SetBackend(facadedmlp.BackendCUDA)
```

### Save/Load

```go
err := mlp.Save("model.json")
loadedMLP, err := facadedmlp.Load("model.json")
defer loadedMLP.Close()
```

### Introspection

```go
// Feature importance
importance := mlp.FeatureImportance()
for _, fi := range importance {
    fmt.Printf("Feature %d: %.4f\n", fi.Index, fi.Score)
}

// Neuron access
weights := mlp.GetNeuronWeights(layer, neuron)
bias := mlp.GetNeuronBias(layer, neuron)
mlp.SetNeuronWeight(layer, neuron, weightIdx, value)
mlp.SetNeuronBias(layer, neuron, value)

// Layer outputs (after prediction)
outputs := mlp.GetLayerOutputs(layer)

// Neuron view
view := mlp.NeuronView(layer, neuron)
fmt.Printf("Weights: %v, Bias: %f\n", view.Weights, view.Bias)
```

### Utility Functions

```go
// Check available backends
backends := facadedmlp.AvailableBackends()

// String representation
fmt.Println(mlp)  // MLP(input=2, hidden=[8], output=1, ...)
```

## Types

### ActivationType

```go
facadedmlp.Sigmoid
facadedmlp.Tanh
facadedmlp.ReLU
facadedmlp.Softmax
```

### OptimizerType

```go
facadedmlp.SGD
facadedmlp.Adam
facadedmlp.RMSProp
```

### BackendType

```go
facadedmlp.BackendAuto   // Auto-select best available
facadedmlp.BackendCPU    // Pure CPU
facadedmlp.BackendCUDA   // NVIDIA CUDA
facadedmlp.BackendOpenCL // OpenCL (AMD, Intel, NVIDIA)
```

## Running Tests

```bash
cd go
LD_LIBRARY_PATH=../target/release go test ./facadedmlp -v
```

## Running Examples

```bash
cd go/examples/xor
go build
LD_LIBRARY_PATH=../../../target/release ./xor
```

## Error Handling

All fallible operations return errors that should be checked:

```go
mlp, err := facadedmlp.New(2, []int{8}, 1, nil)
if err != nil {
    log.Fatalf("Failed to create MLP: %v", err)
}
defer mlp.Close()

output, err := mlp.Predict(input)
if err != nil {
    log.Printf("Prediction failed: %v", err)
}
```

## Memory Management

Always call `Close()` when done with an MLP to free resources:

```go
mlp, _ := facadedmlp.New(...)
defer mlp.Close()  // Ensure cleanup
```

## CGO Notes

This package uses cgo to interface with the Rust library. Ensure:

1. The Rust library is built: `cargo build --release --features julia`
2. The library path is set: `LD_LIBRARY_PATH=.../target/release`
3. CGO is enabled: `CGO_ENABLED=1` (default)

## License

MIT License - Copyright (c) 2025 Matthew Abbott
