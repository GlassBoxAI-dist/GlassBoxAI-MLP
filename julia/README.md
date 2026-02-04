# FacadedMLP.jl

Julia wrapper for the GPU-accelerated Multi-Layer Perceptron library with CUDA/OpenCL/CPU backends.

## Installation

### 1. Build the Rust library

```bash
cd /path/to/GlassBoxAI-MLP
cargo build --release --features julia
```

### 2. Add the Julia package

```julia
using Pkg
Pkg.develop(path="/path/to/GlassBoxAI-MLP/julia")
```

Or add to your Julia environment:

```julia
] dev /path/to/GlassBoxAI-MLP/julia
```

## Quick Start

```julia
using FacadedMLP

# Create a network: 2 inputs, 8 hidden neurons, 1 output
mlp = MLP(2, [8], 1)

# Train on XOR problem
X = [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]
y = [[0.0], [1.0], [1.0], [0.0]]

losses = fit!(mlp, X, y; epochs=1000, verbose=true)

# Predict
for x in X
    output = predict(mlp, x)
    println("$x => $(round(output[1], digits=2))")
end
```

## API Reference

### Creating Models

```julia
# Basic creation
mlp = MLP(input_size, hidden_sizes, output_size)

# With options
mlp = MLP(2, [16, 8], 1;
    hidden_activation = ReLU,      # Sigmoid, Tanh, ReLU, Softmax
    output_activation = Sigmoid,
    backend = "auto",              # "auto", "cpu", "cuda", "opencl"
    learning_rate = 0.01,
    optimizer = Adam,              # SGD, Adam, RMSProp
    dropout_rate = 0.0,
    l2_lambda = 0.0,
    batch_norm = false
)
```

### Training

```julia
# Single sample
train!(mlp, input, target)

# Full dataset
losses = fit!(mlp, inputs, targets; epochs=100, verbose=true)
```

### Prediction

```julia
output = predict(mlp, input)
outputs = predict_batch(mlp, inputs)
```

### Model Properties

```julia
mlp.learning_rate = 0.001
mlp.optimizer = Adam
mlp.dropout_rate = 0.1
mlp.l2_lambda = 0.0001
mlp.batch_norm = true

# Read-only
mlp.input_size
mlp.output_size
mlp.hidden_sizes
mlp.num_layers
mlp.backend
```

### Save/Load

```julia
save(mlp, "model.json")
mlp = load("model.json")
```

### Advanced

```julia
# Feature importance
importance = feature_importance(mlp)

# GPU backends
backends = available_backends()
set_backend!(mlp, "cuda")

# Neuron access
weights = get_neuron_weights(mlp, layer, neuron)
bias = get_neuron_bias(mlp, layer, neuron)
set_neuron_weight!(mlp, layer, neuron, weight_idx, value)
set_neuron_bias!(mlp, layer, neuron, value)

# Layer outputs (after prediction)
outputs = get_layer_outputs(mlp, layer)
```

## Activation Types

- `Sigmoid` (default)
- `Tanh`
- `ReLU`
- `Softmax`

## Optimizer Types

- `SGD`
- `Adam` (default)
- `RMSProp`

## GPU Backends

- `"cpu"` - Pure Rust CPU implementation
- `"cuda"` - NVIDIA CUDA acceleration
- `"opencl"` - OpenCL acceleration
- `"auto"` - Automatically select best available

## Example: MNIST-like Classification

```julia
using FacadedMLP

# 784 inputs (28x28 images), 2 hidden layers, 10 outputs (digits 0-9)
mlp = MLP(784, [256, 128], 10;
    hidden_activation = ReLU,
    output_activation = Softmax,
    optimizer = Adam,
    learning_rate = 0.001,
    dropout_rate = 0.2,
    backend = "cuda"
)

# Train
losses = fit!(mlp, train_X, train_y; epochs=50, verbose=true)

# Predict class
function predict_class(mlp, x)
    output = predict(mlp, x)
    argmax(output) - 1  # 0-indexed class
end
```

## Running Tests

```bash
cd julia
julia --project=. -e 'using Pkg; Pkg.test()'
```

## License

MIT License - Copyright (c) 2025 Matthew Abbott
