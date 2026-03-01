# GlassBoxAI-MLP

## **Multi-Layer Perceptron Suite**

### *GPU-Accelerated MLP with Python, Node.js, Julia, C++, Go, C# & Zig Bindings and Formal Verification*

---

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![CUDA](https://img.shields.io/badge/CUDA-12.0-green.svg)](https://developer.nvidia.com/cuda-toolkit)
[![OpenCL](https://img.shields.io/badge/OpenCL-3.0-blue.svg)](https://www.khronos.org/opencl/)
[![Rust](https://img.shields.io/badge/Rust-1.75+-orange.svg)](https://www.rust-lang.org/)
[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![Node.js](https://img.shields.io/badge/Node.js-16+-339933.svg)](https://nodejs.org/)
[![Go](https://img.shields.io/badge/Go-1.21+-00ADD8.svg)](https://go.dev/)
[![C#](https://img.shields.io/badge/C%23-.NET%208.0-512BD4.svg)](https://dotnet.microsoft.com/)
[![Zig](https://img.shields.io/badge/Zig-0.13+-F7A41D.svg)](https://ziglang.org/)
[![Kani](https://img.shields.io/badge/Kani-Verified-brightgreen.svg)](https://model-checking.github.io/kani/)
[![CISA Compliant](https://img.shields.io/badge/CISA-Secure%20by%20Design-blue.svg)](https://www.cisa.gov/securebydesign)

---

## **Overview**

GlassBoxAI-MLP is a comprehensive, production-ready Multi-Layer Perceptron implementation suite featuring:

- **Multiple GPU backends**: CUDA and OpenCL acceleration with automatic backend selection
- **Python bindings**: Full-featured Python API via PyO3 and maturin
- **Node.js bindings**: Full-featured Node.js API via napi-rs
- **Go bindings**: Idiomatic Go API via cgo
- **C# bindings**: .NET API via P/Invoke
- **Zig bindings**: Zig API via C FFI
- **Julia bindings**: Julia API via ccall FFI
- **C++ bindings**: Header-only C++ wrapper
- **Rust implementation**: Memory-safe, high-performance core
- **Facade pattern architecture**: Clean API separation for maintainability and introspection
- **Formal verification**: Kani-verified Rust implementation for memory safety guarantees
- **CISA/NSA Secure by Design compliance**: Built following government cybersecurity standards

This project demonstrates enterprise-grade software engineering practices including comprehensive testing, formal verification, cross-platform compatibility, and security-first development.

---

## **Table of Contents**

1. [Features](#features)
2. [Architecture](#architecture)
3. [File Structure](#file-structure)
4. [Prerequisites](#prerequisites)
5. [Installation](#installation)
6. [Rust API Reference](#rust-api-reference)
7. [Python API Reference](#python-api-reference)
8. [Node.js API Reference](#nodejs-api-reference)
9. [Julia API Reference](#julia-api-reference)
10. [C++ API Reference](#c-api-reference)
11. [Go API Reference](#go-api-reference)
12. [C# API Reference](#c-api-reference-1)
13. [Zig API Reference](#zig-api-reference)
14. [CLI Reference](#cli-reference)
15. [Testing](#testing)
16. [Formal Verification with Kani](#formal-verification-with-kani)
17. [CISA/NSA Compliance](#cisansa-compliance)
18. [License](#license)
19. [Author](#author)

---

## **Features**

### Core Capabilities

| Feature | Description |
|---------|-------------|
| **Multi-Layer Architecture** | Configurable hidden layers with flexible depth and width |
| **Activation Functions** | Sigmoid, Tanh, ReLU, Softmax |
| **Optimizers** | SGD, Adam, RMSProp with configurable hyperparameters |
| **Regularization** | Dropout and L2 regularization |
| **Batch Normalization** | Stabilize training with learnable scale/shift parameters |
| **Training Features** | Learning rate decay, early stopping, batch training |
| **Model Persistence** | JSON serialization for model save/load |
| **ONNX Export/Import** | Interoperability with the global AI ecosystem |
| **Feature Importance** | GlassBox interpretability - understand which inputs matter most |
| **Mutation API** | Full get/set for optimizer state (Adam M/V), timestep, activations, and bulk weights |

### GPU Acceleration

| Backend | Implementation | Performance |
|---------|---------------|-------------|
| **CUDA** | Native CUDA via cudarc | Optimal for NVIDIA GPUs |
| **OpenCL** | Cross-platform GPU via ocl | AMD, Intel, NVIDIA support |
| **CPU** | Pure Rust fallback | Always available |
| **Auto** | Automatic backend selection | Best available |

### Safety & Security

| Feature | Technology |
|---------|------------|
| **Memory Safety** | Rust ownership model |
| **Formal Verification** | Kani proof harnesses |
| **Bounds Checking** | Verified array access |
| **Input Validation** | CLI and API argument validation |

---

## **Architecture**

```
┌─────────────────────────────────────────────────────────────────────┐
│                         GlassBoxAI-MLP                              │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │              Core Rust Library  (src/)                      │    │
│  │  • network.rs    — Layer, MLP, forward/backward             │    │
│  │  • matrix.rs     — TMatrix ops (add/mul/norm/clip/…)        │    │
│  │  • loss.rs       — MSE, CrossEntropy, Huber                 │    │
│  │  • activations.rs — ReLU, Sigmoid, Tanh, Leaky, Softmax     │    │
│  │  • optimizer.rs  — Adam, SGD, RMSProp, cosine_anneal        │    │
│  │  • random.rs     — Gaussian, uniform, weight init           │    │
│  │  • main.rs       — CLI entry point                          │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                                                                     │
│  ┌────────────────────────┐  ┌────────────────────────────────┐     │
│  │    CUDA Backend        │  │    OpenCL Backend              │     │
│  ├────────────────────────┤  ├────────────────────────────────┤     │
│  │ cudarc (NVIDIA GPUs)   │  │ ocl (NVIDIA / AMD / Intel)     │     │
│  │ Feature: cuda          │  │ Feature: opencl                │     │
│  └────────────────────────┘  └────────────────────────────────┘     │
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │             C ABI Layer  (c_bindings/)                      │    │
│  │  • lib.rs           — extern "C" functions                  │    │
│  │  • kani_ffi_tests.rs — Kani boundary proof harnesses        │    │
│  │  include/glassbox_mlp.h   — C header                        │    │
│  │  include/glassbox_mlp.hpp — C++ RAII wrapper                │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                                                                     │
│  ┌────────┐ ┌────────┐ ┌──────┐ ┌───────┐ ┌──────┐ ┌──────┐         │
│  │ Python │ │Node.js │ │  Go  │ │ Julia │ │  C#  │ │ Zig  │         │
│  ├────────┤ ├────────┤ ├──────┤ ├───────┤ ├──────┤ ├──────┤         │
│  │ PyO3   │ │napi-rs │ │ CGo  │ │ ccall │ │ P/I  │ │C FFI │         │
│  └────────┘ └────────┘ └──────┘ └───────┘ └──────┘ └──────┘         │
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │              Kani Formal Verification                       │    │
│  │  • JSON model format  • ONNX import/export                  │    │
│  │  • Consistent API     • Comprehensive tests                 │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## **File Structure**

```
GlassBoxAI-MLP/
│
├── src/                        # Rust source code
│   ├── lib.rs                  # Library entry point
│   ├── main.rs                 # CLI entry point
│   ├── cli.rs                  # Command-line interface
│   ├── mlp.rs                  # Core MLP implementation
│   ├── gpu_backend.rs          # GPU backend abstraction
│   ├── kernels.rs              # CUDA kernels
│   ├── opencl_kernels.rs       # OpenCL kernels
│   ├── opencl_mlp.rs           # OpenCL MLP implementation
│   ├── onnx.rs                 # ONNX import/export
│   └── python.rs               # Python bindings (PyO3)
│
├── python/                     # Python package
│   └── facaded_mlp_cuda/
│       └── __init__.py         # Python module init
│
├── go/                         # Go wrapper
│   ├── facadedmlp/             # Go package (cgo bindings)
│   ├── examples/               # Go examples
│   ├── go.mod
│   └── README.md
│
├── csharp/                     # C# wrapper
│   ├── GlassBoxAI.MLP/         # .NET project (P/Invoke)
│   ├── examples/               # C# examples
│   └── README.md
│
├── zig/                        # Zig wrapper
│   ├── mlp.zig                 # Zig bindings (C FFI)
│   ├── c.zig                   # C interop layer
│   ├── build.zig               # Build configuration
│   ├── examples/               # Zig examples
│   └── README.md
│
├── kani/                       # Formal verification proofs
│   ├── Cargo.toml
│   ├── lib.rs
│   ├── core_types.rs
│   ├── harnesses.rs            # Kani proofs (categories 1-15)
│   ├── ffi_boundary.rs         # FFI boundary safety proofs (category 16)
│   └── README.md
│
├── examples/                   # Example scripts
│   ├── xor_example.py          # Python XOR example
│   ├── gpu_example_backend.py  # Python GPU backend demo
│   ├── xor_example.js          # Node.js XOR example
│   └── gpu_backend_example.js  # Node.js GPU backend demo
│
├── tests/                      # Python tests
│   └── test_mlp.py             # pytest test suite
│
├── Cargo.toml                  # Rust dependencies
├── pyproject.toml              # Python build config
├── package.json                # Node.js package config
├── index.js                    # Node.js module entry
├── index.d.ts                  # TypeScript definitions
├── Makefile                    # Build automation
├── BUILD.md                    # Detailed build instructions
└── README.md                   # This file
```

---

## **Prerequisites**

### Required

| Dependency | Version | Purpose |
|------------|---------|---------|
| **Python** | 3.8+ | Python bindings |
| **Node.js** | 16+ | Node.js bindings |
| **Rust** | 1.75+ | Core compilation |
| **maturin** | 1.0+ | Python package building |
| **@napi-rs/cli** | 2.0+ | Node.js package building |

### Optional

| Dependency | Version | Purpose |
|------------|---------|---------|
| **CUDA Toolkit** | 11.0+ | CUDA GPU acceleration |
| **OpenCL SDK** | 3.0 | OpenCL GPU acceleration |
| **Kani** | 0.67+ | Formal verification |
| **pytest** | latest | Python testing |

---

## **Installation**

### **Quick Install (Python)**

```bash
# Install maturin
pip install maturin

# Clone and install
git clone <repo-url>
cd GlassBoxAI-MLP

# Build with all GPU backends
maturin develop --release --features python,cuda,opencl

# Or CPU-only (no GPU dependencies)
maturin develop --release --features python
```

### **Using Makefile**

```bash
# See all available commands
make help

# Build and install Python package with all backends
make install

# Run tests
make test

# Run examples
make run-xor
make run-backends
```

### **Rust Crate**

```toml
# Cargo.toml
[dependencies]
glassboxai-mlp = "0.1"
```

Or with specific features:
```toml
[dependencies]
glassboxai-mlp = { version = "0.1", features = ["cuda", "opencl"] }
```

### **Quick Install (Node.js)**

```bash
# Install napi-rs CLI
npm install -g @napi-rs/cli

# Clone and install
git clone <repo-url>
cd GlassBoxAI-MLP

# Build with all GPU backends
npm run build

# Or CPU-only (no GPU dependencies)
npm run build:cpu
```

### **Feature Flags**

| Feature | Description |
|---------|-------------|
| `cuda` | Enable CUDA support |
| `opencl` | Enable OpenCL support |
| `cli` | Build command-line interface |
| `python` | Build Python bindings |
| `nodejs` | Build Node.js bindings |

```bash
# Python: CUDA only
maturin develop --release --features python,cuda

# Python: OpenCL only
maturin develop --release --features python,opencl

# Python: Both CUDA and OpenCL
maturin develop --release --features python,cuda,opencl

# Node.js: All backends
npm run build

# Node.js: CPU only
npm run build:cpu
```

---

## **Rust API Reference**

### **Quick Start**

```rust
use glassboxai_mlp::{MLP, MlpConfig, ActivationType, OptimizerType};

fn main() -> Result<(), String> {
    // Create a simple network with defaults
    let mut mlp = MLP::new(2, &[8], 1)?;

    // XOR training data
    let inputs = vec![
        vec![0.0, 0.0], vec![0.0, 1.0],
        vec![1.0, 0.0], vec![1.0, 1.0],
    ];
    let targets = vec![
        vec![0.0], vec![1.0], vec![1.0], vec![0.0],
    ];

    // Configure and train
    mlp.set_learning_rate(0.5);
    mlp.set_optimizer(OptimizerType::Adam);
    
    let result = mlp.fit(&inputs, &targets, 1000, true)?;
    println!("Final loss: {:.6}", result.final_loss);

    // Predict
    let output = mlp.predict(&[1.0, 0.0])?;
    println!("Prediction: {:.4}", output[0]);

    // Introspection
    let importance = mlp.feature_importance();
    for fi in &importance {
        println!("Feature {}: {:.4}", fi.index, fi.score);
    }
    
    Ok(())
}
```

### **Custom Configuration**

```rust
use glassboxai_mlp::{MLP, MlpConfig, ActivationType, OptimizerType, BackendType};

let config = MlpConfig {
    hidden_activation: ActivationType::ReLU,
    output_activation: ActivationType::Softmax,
    learning_rate: 0.001,
    optimizer: OptimizerType::Adam,
    backend: BackendType::CUDA,
    dropout_rate: 0.2,
    l2_lambda: 0.0001,
    ..Default::default()
};

let mlp = MLP::with_config(784, &[256, 128], 10, config)?;
```

### **Glass Box Introspection**

```rust
// View layer information
let info = mlp.layer_info(1);
println!("Layer {}: {} neurons", info.index, info.size);

// View individual neurons
let neuron = mlp.neuron_view(1, 0);
println!("Weights: {:?}", neuron.weights);
println!("Bias: {}", neuron.bias);

// Access optimizer state (Adam M and V values)
let m = mlp.get_weight_m(1, 0, 0);
let v = mlp.get_weight_v(1, 0, 0);

// Activation histogram
let hist = mlp.activation_histogram(1, 10);
```

### **Mutation API**

Full control over internal model state for checkpointing, transfer learning, and research:

```rust
// Set Adam optimizer moments
mlp.set_weight_m(1, 0, 0, 0.001);
mlp.set_weight_v(1, 0, 0, 0.0001);
mlp.set_bias_m(1, 0, 0.001);
mlp.set_bias_v(1, 0, 0.0001);

// Set optimizer timestep
mlp.set_timestep(100);

// Change layer activation at runtime
mlp.set_layer_activation(1, ActivationType::ReLU);

// Bulk-set all weights for a neuron
mlp.set_weights(1, 0, &[0.1, 0.2, 0.3]);
```

### **Feature Flags**

| Feature | Description |
|---------|-------------|
| `cuda` | NVIDIA CUDA GPU acceleration |
| `opencl` | OpenCL GPU acceleration (AMD, Intel, NVIDIA) |
| `python` | Python bindings (PyO3) |
| `nodejs` | Node.js bindings (NAPI) |
| `julia` | Julia/C FFI bindings |
| `cli` | Command-line interface |

---

## **Python API Reference**

### **Quick Start**

```python
from facaded_mlp_cuda import MLP, PyActivationType, PyOptimizerType

# Check available backends
print("Available:", MLP.available_backends())

# Create MLP with auto-selected backend
mlp = MLP(2, [8], 1, gpu_backend="auto")
print(f"Using: {mlp.gpu_backend}")

# XOR problem
X = [[0,0], [0,1], [1,0], [1,1]]
y = [[0], [1], [1], [0]]
losses = mlp.fit(X, y, epochs=1000, verbose=True)

predictions = mlp.predict_batch(X)
print(predictions)
```

### **MLP Class**

#### Constructor

```python
MLP(
    input_size: int,
    hidden_sizes: list[int],
    output_size: int,
    hidden_activation: PyActivationType = PyActivationType.Sigmoid,
    output_activation: PyActivationType = PyActivationType.Sigmoid,
    gpu_backend: str = "auto"  # "auto", "cuda", "opencl", "cpu"
)
```

#### Training Methods

| Method | Description |
|--------|-------------|
| `fit(inputs, targets, epochs=100, verbose=False)` | Train on dataset, returns loss history |
| `train(input, target)` | Train on single sample |
| `predict(input)` | Predict single sample |
| `predict_batch(inputs)` | Predict multiple samples |

#### Model I/O

| Method | Description |
|--------|-------------|
| `save(filename)` | Save model to JSON file |
| `MLP.load(filename)` | Load model from JSON file (static method) |
| `export_onnx(filename)` | Export to ONNX format |
| `MLP.import_onnx(filename)` | Import from ONNX format (static method) |

#### Properties

| Property | Type | Description |
|----------|------|-------------|
| `learning_rate` | float | Learning rate (get/set) |
| `optimizer` | PyOptimizerType | Optimizer type (get/set) |
| `dropout_rate` | float | Dropout rate (get/set) |
| `l2_lambda` | float | L2 regularization (get/set) |
| `batch_norm` | bool | Batch normalization (get/set) |
| `gpu_backend` | str | Current GPU backend (get) |
| `input_size` | int | Input layer size (get) |
| `output_size` | int | Output layer size (get) |
| `hidden_sizes` | list[int] | Hidden layer sizes (get) |
| `num_layers` | int | Total layer count (get) |

#### Facade Methods (Introspection)

| Method | Description |
|--------|-------------|
| `get_neuron_weights(layer, neuron)` | Get weights for a specific neuron |
| `get_neuron_bias(layer, neuron)` | Get bias for a specific neuron |
| `set_neuron_weight(layer, neuron, weight_idx, value)` | Set a specific weight |
| `set_neuron_bias(layer, neuron, value)` | Set a neuron's bias |
| `get_layer_outputs(layer)` | Get layer outputs after forward pass |
| `get_layer_errors(layer)` | Get layer errors/gradients after training |
| `get_layer_size(layer)` | Get number of neurons in a layer |
| `get_layer_activation(layer)` | Get activation type of a layer |
| `feature_importance()` | Calculate feature importance scores |

#### Mutation Methods

| Method | Description |
|--------|-------------|
| `set_weight_m(layer, neuron, weight_idx, value)` | Set Adam weight first moment (M) |
| `set_weight_v(layer, neuron, weight_idx, value)` | Set Adam weight second moment (V) |
| `set_bias_m(layer, neuron, value)` | Set Adam bias first moment (M) |
| `set_bias_v(layer, neuron, value)` | Set Adam bias second moment (V) |
| `set_timestep(value)` | Set Adam optimizer timestep |
| `set_layer_activation(layer, activation)` | Set activation function for a layer |
| `set_neuron_weights(layer, neuron, weights)` | Set all weights for a neuron (bulk) |

#### Backend Methods

| Method | Description |
|--------|-------------|
| `MLP.available_backends()` | List available backends (static method) |
| `set_backend(backend)` | Switch GPU backend dynamically |

### **Enums**

```python
# Activation functions
PyActivationType.Sigmoid
PyActivationType.Tanh
PyActivationType.ReLU
PyActivationType.Softmax

# Optimizers
PyOptimizerType.SGD
PyOptimizerType.Adam
PyOptimizerType.RMSProp
```

### **Utility Functions**

```python
from facaded_mlp_cuda import load_csv, normalize

# Load dataset from CSV
inputs, targets = load_csv("data.csv", input_size=4, output_size=1)

# Normalize inputs
normalized = normalize(inputs)
```

### **Complete Example**

```python
from facaded_mlp_cuda import MLP, PyActivationType, PyOptimizerType

# Check available backends
print("Available GPU backends:", MLP.available_backends())

# Create MLP for XOR problem
mlp = MLP(
    input_size=2,
    hidden_sizes=[8],
    output_size=1,
    hidden_activation=PyActivationType.Sigmoid,
    output_activation=PyActivationType.Sigmoid,
    gpu_backend="auto"
)

# Set hyperparameters
mlp.learning_rate = 0.5
mlp.optimizer = PyOptimizerType.Adam

print(f"Created model: {mlp}")
print(f"Using backend: {mlp.gpu_backend}")

# XOR training data
X = [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]
y = [[0.0], [1.0], [1.0], [0.0]]

# Train
losses = mlp.fit(X, y, epochs=2000, verbose=True)
print(f"Final loss: {losses[-1]:.6f}")

# Predictions
predictions = mlp.predict_batch(X)
for inp, tgt, pred in zip(X, y, predictions):
    print(f"Input: {inp} -> Target: {tgt[0]:.1f}, Prediction: {pred[0]:.4f}")

# Save model
mlp.save("xor_model.json")

# Load and test
mlp2 = MLP.load("xor_model.json")
output = mlp2.predict([1.0, 0.0])
print(f"Loaded model prediction: {output[0]:.4f}")

# Feature importance
importance = mlp.feature_importance()
print("Feature importance:")
for feature_idx, score in importance:
    print(f"  Feature {feature_idx}: {score:.6f}")
```

---

## **Node.js API Reference**

### **Quick Start**

```javascript
const { MLP, JsActivationType, JsOptimizerType } = require('facaded-mlp-cuda');

// Check available backends
console.log('Available:', MLP.availableBackends());

// Create MLP with auto-selected backend
const mlp = new MLP(2, [8], 1, { gpuBackend: 'auto' });
console.log(`Using: ${mlp.gpuBackend}`);

// XOR problem
const X = [[0,0], [0,1], [1,0], [1,1]];
const y = [[0], [1], [1], [0]];
const result = mlp.fit(X, y, 1000, true);

const predictions = mlp.predictBatch(X);
console.log(predictions);
```

### **MLP Class**

#### Constructor

```javascript
new MLP(inputSize, hiddenSizes, outputSize, options?)
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `inputSize` | `number` | Number of input neurons |
| `hiddenSizes` | `number[]` | Array of hidden layer sizes |
| `outputSize` | `number` | Number of output neurons |
| `options` | `MlpOptions` | Optional configuration object |

#### Options Object

```typescript
interface MlpOptions {
  hiddenActivation?: JsActivationType;  // default: Sigmoid
  outputActivation?: JsActivationType;  // default: Sigmoid
  gpuBackend?: 'auto' | 'cuda' | 'opencl' | 'cpu';  // default: 'auto'
  learningRate?: number;
  optimizer?: JsOptimizerType;
  dropoutRate?: number;
  l2Lambda?: number;
  batchNorm?: boolean;
}
```

#### Training Methods

| Method | Description |
|--------|-------------|
| `fit(inputs, targets, epochs?, verbose?)` | Train on dataset, returns `{ losses, finalLoss }` |
| `train(input, target)` | Train on single sample |
| `predict(input)` | Predict single sample, returns `number[]` |
| `predictBatch(inputs)` | Predict multiple samples, returns `number[][]` |

#### Model I/O

| Method | Description |
|--------|-------------|
| `save(filename)` | Save model to JSON file |
| `MLP.load(filename)` | Load model from JSON file (static) |
| `exportOnnx(filename)` | Export to ONNX format |
| `MLP.importOnnx(filename)` | Import from ONNX format (static) |

#### Properties

| Property | Type | Description |
|----------|------|-------------|
| `learningRate` | `number` | Learning rate (get/set) |
| `optimizer` | `JsOptimizerType` | Optimizer type (get/set) |
| `dropoutRate` | `number` | Dropout rate (get/set) |
| `l2Lambda` | `number` | L2 regularization (get/set) |
| `batchNorm` | `boolean` | Batch normalization (get/set) |
| `gpuBackend` | `string` | Current GPU backend (get) |
| `inputSize` | `number` | Input layer size (get) |
| `outputSize` | `number` | Output layer size (get) |
| `hiddenSizes` | `number[]` | Hidden layer sizes (get) |
| `numLayers` | `number` | Total layer count (get) |

#### Facade Methods (Introspection)

| Method | Description |
|--------|-------------|
| `getNeuronWeights(layer, neuron)` | Get weights for a specific neuron |
| `getNeuronBias(layer, neuron)` | Get bias for a specific neuron |
| `setNeuronWeight(layer, neuron, weightIdx, value)` | Set a specific weight |
| `setNeuronBias(layer, neuron, value)` | Set a neuron's bias |
| `getLayerOutputs(layer)` | Get layer outputs after forward pass |
| `getLayerErrors(layer)` | Get layer errors/gradients after training |
| `getLayerSize(layer)` | Get number of neurons in a layer |
| `getLayerActivation(layer)` | Get activation type of a layer |
| `getWeightM(layer, neuron, weightIdx)` | Get Adam weight first moment (M) |
| `getWeightV(layer, neuron, weightIdx)` | Get Adam weight second moment (V) |
| `getBiasM(layer, neuron)` | Get Adam bias first moment (M) |
| `getBiasV(layer, neuron)` | Get Adam bias second moment (V) |
| `getTimestep()` | Get Adam optimizer timestep |
| `getActivationHistogram(layer, bins)` | Get activation histogram for a layer |
| `getGradientHistogram(layer, bins)` | Get gradient histogram for a layer |
| `featureImportance()` | Returns `{ featureIndex, score }[]` |

#### Mutation Methods

| Method | Description |
|--------|-------------|
| `setWeightM(layer, neuron, weightIdx, value)` | Set Adam weight first moment (M) |
| `setWeightV(layer, neuron, weightIdx, value)` | Set Adam weight second moment (V) |
| `setBiasM(layer, neuron, value)` | Set Adam bias first moment (M) |
| `setBiasV(layer, neuron, value)` | Set Adam bias second moment (V) |
| `setTimestep(value)` | Set Adam optimizer timestep |
| `setLayerActivation(layer, activation)` | Set activation function for a layer |
| `setNeuronWeights(layer, neuron, weights)` | Set all weights for a neuron (bulk) |

#### Backend Methods

| Method | Description |
|--------|-------------|
| `MLP.availableBackends()` | List available backends (static) |
| `setBackend(backend)` | Switch GPU backend dynamically |
| `info()` | Get model info object |
| `toString()` | Get string representation |

### **Enums**

```javascript
// Activation functions
JsActivationType.Sigmoid  // 0
JsActivationType.Tanh     // 1
JsActivationType.ReLU     // 2
JsActivationType.Softmax  // 3

// Optimizers
JsOptimizerType.SGD       // 0
JsOptimizerType.Adam      // 1
JsOptimizerType.RMSProp   // 2
```

### **Utility Functions**

```javascript
const { loadCsv, normalize } = require('facaded-mlp-cuda');

// Load dataset from CSV
const { inputs, targets } = loadCsv('data.csv', 4, 1);

// Normalize inputs
const normalized = normalize(inputs);
```

### **Complete Example**

```javascript
const { MLP, JsActivationType, JsOptimizerType } = require('facaded-mlp-cuda');

// Check available backends
console.log('Available GPU backends:', MLP.availableBackends());

// Create MLP for XOR problem
const mlp = new MLP(2, [8], 1, {
  hiddenActivation: JsActivationType.Sigmoid,
  outputActivation: JsActivationType.Sigmoid,
  gpuBackend: 'auto',
});

// Set hyperparameters
mlp.learningRate = 0.5;
mlp.optimizer = JsOptimizerType.Adam;

console.log(`Created model: ${mlp.toString()}`);
console.log(`Using backend: ${mlp.gpuBackend}`);

// XOR training data
const X = [[0, 0], [0, 1], [1, 0], [1, 1]];
const y = [[0], [1], [1], [0]];

// Train
const result = mlp.fit(X, y, 2000, true);
console.log(`Final loss: ${result.finalLoss.toFixed(6)}`);

// Predictions
const predictions = mlp.predictBatch(X);
for (let i = 0; i < X.length; i++) {
  console.log(`Input: [${X[i]}] -> Target: ${y[i][0]}, Prediction: ${predictions[i][0].toFixed(4)}`);
}

// Save model
mlp.save('xor_model.json');

// Load and test
const mlp2 = MLP.load('xor_model.json');
const output = mlp2.predict([1.0, 0.0]);
console.log(`Loaded model prediction: ${output[0].toFixed(4)}`);

// Feature importance
const importance = mlp.featureImportance();
console.log('Feature importance:');
for (const { featureIndex, score } of importance) {
  console.log(`  Feature ${featureIndex}: ${score.toFixed(6)}`);
}
```

### **TypeScript Support**

The package includes full TypeScript definitions in `index.d.ts`:

```typescript
import { MLP, JsActivationType, JsOptimizerType, MlpOptions, TrainResult } from 'facaded-mlp-cuda';

const options: MlpOptions = {
  hiddenActivation: JsActivationType.ReLU,
  gpuBackend: 'cuda',
  learningRate: 0.01,
};

const mlp = new MLP(4, [16, 8], 2, options);
const result: TrainResult = mlp.fit(inputs, targets, 1000);
```

---

## **Julia API Reference**

See [julia/README.md](julia/README.md) for complete documentation.

### **Quick Start**

```julia
using FacadedMLP

# Create a network
mlp = MLP(2, [8], 1; learning_rate=0.5, optimizer=Adam)

# Train on XOR
X = [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]
y = [[0.0], [1.0], [1.0], [0.0]]

losses = fit!(mlp, X, y; epochs=1000, verbose=true)

# Predict
output = predict(mlp, [1.0, 0.0])
println("Output: $(output[1])")

# Save/load
save(mlp, "model.json")
mlp2 = load("model.json")
```

### **Mutation API**

```julia
# Set Adam optimizer state
set_weight_m!(mlp, 1, 1, 1, 0.001)
set_weight_v!(mlp, 1, 1, 1, 0.0001)
set_bias_m!(mlp, 1, 1, 0.001)
set_bias_v!(mlp, 1, 1, 0.0001)
set_timestep!(mlp, 100)

# Change activation and bulk-set weights
set_layer_activation!(mlp, 1, ReLU)
set_neuron_weights!(mlp, 1, 1, [0.1, 0.2, 0.3])
```

### **Installation**

```bash
# Build Rust library
cargo build --release --features julia

# In Julia
using Pkg
Pkg.develop(path="julia")
```

---

## **C++ API Reference**

See [cpp/README.md](cpp/README.md) for complete documentation.

### **Quick Start**

```cpp
#include "facaded_mlp.hpp"

using namespace facaded;

int main() {
    // Create network
    MLP mlp(2, {8}, 1);
    
    // Configure
    mlp.set_learning_rate(0.5);
    mlp.set_optimizer(Optimizer::Adam);
    
    // Train on XOR
    std::vector<std::vector<double>> X = {{0,0}, {0,1}, {1,0}, {1,1}};
    std::vector<std::vector<double>> y = {{0}, {1}, {1}, {0}};
    
    auto result = mlp.fit(X, y, 1000, true);
    std::cout << "Final loss: " << result.final_loss << std::endl;
    
    // Predict
    auto output = mlp.predict({1.0, 0.0});
    std::cout << "Prediction: " << output[0] << std::endl;
    
    return 0;
}
```

### **Mutation API**

```cpp
// Set Adam optimizer state
mlp.set_weight_m(1, 0, 0, 0.001);
mlp.set_weight_v(1, 0, 0, 0.0001);
mlp.set_bias_m(1, 0, 0.001);
mlp.set_bias_v(1, 0, 0.0001);
mlp.set_timestep(100);

// Change activation and bulk-set weights
mlp.set_layer_activation(1, Activation::ReLU);
mlp.set_neuron_weights(1, 0, {0.1, 0.2, 0.3});
```

### **Building**

```bash
# Build Rust library
cargo build --release --features julia

# Compile C++ example
g++ -std=c++17 -O2 -I cpp/include example.cpp \
    -L target/release -lfacaded_mlp_cuda -o example
```

---

## **Go API Reference**

See [go/README.md](go/README.md) for complete documentation.

### **Quick Start**

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
    defer mlp.Close()
    
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

### **Mutation API**

```go
// Set Adam optimizer state
mlp.SetWeightM(1, 0, 0, 0.001)
mlp.SetWeightV(1, 0, 0, 0.0001)
mlp.SetBiasM(1, 0, 0.001)
mlp.SetBiasV(1, 0, 0.0001)
mlp.SetTimestep(100)

// Change activation and bulk-set weights
mlp.SetLayerActivation(1, facadedmlp.ReLU)
mlp.SetNeuronWeights(1, 0, []float64{0.1, 0.2, 0.3})
```

### **Installation**

```bash
# Build Rust library
cargo build --release --features julia

# Set library path
export LD_LIBRARY_PATH=/path/to/GlassBoxAI-MLP/target/release:$LD_LIBRARY_PATH

# Run Go example
cd go/examples/xor
go build
LD_LIBRARY_PATH=../../../target/release ./xor
```

---

## **C# API Reference**

See [csharp/README.md](csharp/README.md) for complete documentation.

### **Quick Start**

```csharp
using GlassBoxAI.MLP;

using var mlp = new MLP(2, new[] { 8 }, 1);
mlp.LearningRate = 0.5;
mlp.Optimizer = OptimizerType.Adam;

var inputs = new[] {
    new[] { 0.0, 0.0 }, new[] { 0.0, 1.0 },
    new[] { 1.0, 0.0 }, new[] { 1.0, 1.0 }
};
var targets = new[] {
    new[] { 0.0 }, new[] { 1.0 },
    new[] { 1.0 }, new[] { 0.0 }
};

var result = mlp.Fit(inputs, targets, 1000, verbose: true);
Console.WriteLine($"Final loss: {result.FinalLoss:F6}");

var output = mlp.Predict(new[] { 1.0, 0.0 });
Console.WriteLine($"Prediction: {output[0]:F4}");
```

### **Mutation API**

```csharp
// Set Adam optimizer state
mlp.SetWeightM(1, 0, 0, 0.001);
mlp.SetWeightV(1, 0, 0, 0.0001);
mlp.SetBiasM(1, 0, 0.001);
mlp.SetBiasV(1, 0, 0.0001);
mlp.SetTimestep(100);

// Change activation and bulk-set weights
mlp.SetLayerActivation(1, ActivationType.ReLU);
mlp.SetNeuronWeights(1, 0, new[] { 0.1, 0.2, 0.3 });
```

### **Building**

```bash
# Build the native library
cargo build --release --features julia

# Build the C# project
cd csharp/examples
dotnet build

# Run the example
LD_LIBRARY_PATH=../../target/release dotnet run
```

---

## **Zig API Reference**

See [zig/README.md](zig/README.md) for complete documentation.

### **Quick Start**

```zig
const std = @import("std");
const mlp_mod = @import("glassboxai-mlp");
const MLP = mlp_mod.MLP;

pub fn main() !void {
    const allocator = std.heap.page_allocator;

    const hidden = [_]i32{8};
    var net = try MLP.init(allocator, 2, &hidden, 1, .{
        .learning_rate = 0.5,
        .optimizer = .adam,
    });
    defer net.deinit();

    // Train on XOR
    const inputs = [_][]const f64{
        &[_]f64{ 0.0, 0.0 }, &[_]f64{ 0.0, 1.0 },
        &[_]f64{ 1.0, 0.0 }, &[_]f64{ 1.0, 1.0 },
    };
    const targets = [_][]const f64{
        &[_]f64{0.0}, &[_]f64{1.0},
        &[_]f64{1.0}, &[_]f64{0.0},
    };

    var result = try net.fit(allocator, &inputs, &targets, 1000, true);
    defer result.deinit();

    // Predict
    var output: [1]f64 = undefined;
    _ = try net.predict(&[_]f64{ 1.0, 0.0 }, &output);
    std.debug.print("Output: {d:.4}\n", .{output[0]});
}
```

### **Mutation API**

```zig
// Set Adam optimizer state
net.setWeightM(1, 0, 0, 0.001);
net.setWeightV(1, 0, 0, 0.0001);
net.setBiasM(1, 0, 0.001);
net.setBiasV(1, 0, 0.0001);
net.setTimestep(100);

// Change activation and bulk-set weights
net.setLayerActivation(1, .relu);
try net.setNeuronWeights(1, 0, &[_]f64{ 0.1, 0.2, 0.3 });
```

### **Building**

```bash
# Build the native library
cargo build --release --features julia

# Build the Zig example
cd zig
zig build

# Run the example
LD_LIBRARY_PATH=../target/release zig-out/bin/xor_example
```

---

## **CLI Reference**

### Usage

```
glassboxai-mlp <command> [options]
```

### Commands

| Command | Description |
|---------|-------------|
| `create` | Create a new model |
| `info` | Display model information |
| `train` | Train on dataset |
| `predict` | Make a prediction |
| `batch-predict` | Make batch predictions |
| `export-onnx` | Export to ONNX format |
| `import-onnx` | Import from ONNX format |
| `feature-importance` | Calculate feature importance |
| `get-weight` | Get a single weight value |
| `set-weight` | Set a single weight value |
| `get-weights` | Get all weights for a neuron |
| `set-weights` | Set all weights for a neuron (bulk) |
| `get-bias` | Get bias for a neuron |
| `set-bias` | Set bias for a neuron |
| `layer-info` | Display layer information |
| `histogram` | Display activation/error histogram |
| `set-weight-m` | Set Adam weight first moment (M) |
| `set-weight-v` | Set Adam weight second moment (V) |
| `set-bias-m` | Set Adam bias first moment (M) |
| `set-bias-v` | Set Adam bias second moment (V) |
| `set-timestep` | Set Adam optimizer timestep |
| `set-activation` | Set layer activation function |
| `set-learning-rate` | Set model learning rate |
| `set-optimizer-type` | Set optimizer type (sgd/adam/rmsprop) |
| `set-dropout` | Set dropout rate |
| `set-l2` | Set L2 regularization lambda |
| `set-batch-norm` | Set batch normalization on/off |

### Options

| Option | Description |
|--------|-------------|
| `-i, --inputs=N` | Number of input neurons |
| `-H, --hidden=N,M,...` | Hidden layer sizes |
| `-o, --outputs=N` | Number of output neurons |
| `-m, --model=FILE` | Model file path |
| `-s, --save=FILE` | Save output/modified model to file |
| `-d, --data=FILE` | Training data CSV |
| `--epochs=N` | Training epochs |
| `--lr=VALUE` | Learning rate |
| `--activation=TYPE` | Activation function (sigmoid/tanh/relu/softmax) |
| `--optimizer=TYPE` | Optimizer type (sgd/adam/rmsprop) |
| `--gpu=BACKEND` | GPU backend: auto, cuda, opencl, cpu |
| `--layer=N` | Layer index (for weight/bias/activation commands) |
| `--neuron=N` | Neuron index |
| `--weight=N` | Weight index |
| `--value=V` | Value to set (for set-* commands) |
| `--values=v1,v2,...` | Comma-separated values (for set-weights) |
| `--timestep=N` | Timestep value (for set-timestep) |
| `--on / --off` | Toggle for set-batch-norm |

### CLI Examples

```bash
# Create a new model (auto-detect GPU)
facaded_mlp_cuda create -i 2 -H 8 -o 1 -s model.json

# Create with specific backend
facaded_mlp_cuda create -i 2 -H 8 -o 1 -s model.json --gpu=cuda

# Train
facaded_mlp_cuda train -m model.json -d data.csv -s trained.json --epochs=1000

# Predict
facaded_mlp_cuda predict -m trained.json -i 1.0,0.0

# Feature importance
facaded_mlp_cuda feature-importance -m trained.json

# Export to ONNX
facaded_mlp_cuda export-onnx -m trained.json -s model.onnx

# Set Adam optimizer state
facaded_mlp_cuda set-weight-m -m model.json --layer=1 --neuron=0 --weight=0 --value=0.001 -s model.json
facaded_mlp_cuda set-weight-v -m model.json --layer=1 --neuron=0 --weight=0 --value=0.0001 -s model.json
facaded_mlp_cuda set-bias-m -m model.json --layer=1 --neuron=0 --value=0.001 -s model.json
facaded_mlp_cuda set-bias-v -m model.json --layer=1 --neuron=0 --value=0.0001 -s model.json

# Set timestep, activation, and hyperparameters
facaded_mlp_cuda set-timestep -m model.json --timestep=100 -s model.json
facaded_mlp_cuda set-activation -m model.json --layer=1 --activation=relu -s model.json
facaded_mlp_cuda set-learning-rate -m model.json --value=0.001 -s model.json
facaded_mlp_cuda set-optimizer-type -m model.json --optimizer=adam -s model.json
facaded_mlp_cuda set-dropout -m model.json --value=0.2 -s model.json
facaded_mlp_cuda set-l2 -m model.json --value=0.0001 -s model.json
facaded_mlp_cuda set-batch-norm -m model.json --on -s model.json

# Bulk-set all weights for a neuron
facaded_mlp_cuda set-weights -m model.json --layer=1 --neuron=0 --values=0.1,0.2,0.3 -s model.json
```

---

## **Testing**

### Running Python Tests

```bash
# Install test dependencies
pip install pytest numpy

# Run all tests
pytest tests/ -v

# Run specific test
pytest tests/test_mlp.py::test_xor_training -v
```

### Running Node.js Examples

```bash
# Run XOR example
npm test

# Or run directly
node examples/xor_example.js
node examples/gpu_backend_example.js
```

### Using Makefile

```bash
make test
```

### Test Categories

| Category | Tests |
|----------|-------|
| **Model Creation** | Architecture configurations |
| **Backend Detection** | Available backends, switching |
| **Training** | XOR problem, loss reduction |
| **Model I/O** | Save/load persistence |
| **Properties** | Hyperparameter get/set |
| **Introspection** | Weight/bias access |

---

## **Formal Verification with Kani**

### Overview

The Rust implementation includes **Kani formal verification proofs** that mathematically prove the absence of certain classes of bugs. This goes beyond traditional testing to provide **mathematical guarantees** about code correctness.

### Running Kani Verification

```bash
cd kani

# Run all proofs
cargo kani --tests

# Run specific proof
cargo kani --harness verify_array_bounds_weight_access

# Run unit tests
cargo test
```

### Why Formal Verification Matters

Traditional testing can only verify specific test cases. Formal verification with Kani:

- **Exhaustively checks all possible inputs** within defined bounds
- **Mathematically proves** absence of panics, buffer overflows, and undefined behavior
- **Catches edge cases** that random testing might miss
- **Provides cryptographic-level assurance** for safety-critical code

### Verification Harnesses (Categories 1–15)

#### 1. Strict Bound Checks
- `verify_array_bounds_layer_access`
- `verify_array_bounds_weight_access`
- `verify_array_bounds_bias_access`
- `verify_array_bounds_output_access`
- `verify_validate_bounds_generic`

#### 2. Pointer Validity Proofs
- `verify_no_null_pointer_in_layer_creation`
- `verify_mlp_initialization_validity`

#### 3. No-Panic Guarantee
- `verify_activation_functions_no_panic`
- `verify_max_index_no_panic`
- `verify_mlp_construction_no_panic`
- `verify_parse_activation_no_panic`
- `verify_parse_optimizer_no_panic`

#### 4. Integer Overflow Prevention
- `verify_safe_add_no_overflow`
- `verify_safe_sub_no_overflow`
- `verify_safe_mul_no_overflow`
- `verify_layer_size_calculation_no_overflow`

#### 5. Division-by-Zero Exclusion
- `verify_safe_div_no_zero`
- `verify_normalization_no_div_by_zero`
- `verify_softmax_denominator_non_zero`

#### 6. Global State Consistency
- `verify_mlp_invariants_after_mutation`
- `verify_layer_invariants_preserved`

#### 7. Deadlock-Free Logic
- `verify_no_reentrant_locking_pattern`

#### 8. Input Sanitization Bounds
- `verify_bounded_loop_terminates`
- `verify_training_epoch_bounded`
- `verify_hidden_layer_count_bounded`

#### 9. Result Coverage Audit
- `verify_layer_access_result_handling`
- `verify_mlp_creation_result_handling`
- `verify_compute_loss_result_handling`

#### 10. Memory Leak/Leakage Proofs
- `verify_layer_data_owned_vectors`
- `verify_allocation_with_limit_respects_budget`

#### 11. Constant-Time Execution
- `verify_sigmoid_constant_time_bounds`
- `verify_relu_constant_time_output`
- `verify_activation_selection_public_key`

#### 12. State Machine Integrity
- `verify_privilege_escalation_blocked`
- `verify_unprivileged_cannot_escalate`

#### 13. Enum Exhaustion
- `verify_activation_type_exhaustive`
- `verify_optimizer_type_exhaustive`
- `verify_command_type_exhaustive`

#### 14. Floating-Point Sanity
- `verify_fp_sanity_check`
- `verify_clamp_fp_handles_special_values`
- `verify_sigmoid_never_nan_or_inf`
- `verify_relu_never_nan`
- `verify_compute_loss_nan_handling`

#### 15. Resource Limit Compliance
- `verify_memory_budget_enforcement`
- `verify_layer_allocation_within_budget`
- `verify_mlp_total_memory_bounded`

### FFI Boundary Safety Harnesses (Category 16)

Proofs that all data crossing the C FFI boundary is validated before use. Covers the complete `extern "C"` surface consumed by C++, Go, C#, Julia, Zig, and Python wrappers.

#### A. Signed-to-Unsigned Conversion Safety
- `verify_i32_to_usize_rejects_negative`
- `verify_i32_positive_rejects_zero_and_negative`
- `verify_ffi_len_validates_range`
- `verify_ffi_len_i32_min_rejected`
- `verify_ffi_len_negative_one_rejected`

#### B. Output Buffer Overflow Prevention
- `verify_negative_capacity_prevents_buffer_write`
- `verify_zero_capacity_prevents_buffer_write`
- `verify_output_write_bounded_by_validated_capacity`

#### C. NaN/Infinity Parameter Rejection
- `verify_f64_param_rejects_nan`
- `verify_f64_param_rejects_infinity`
- `verify_f64_param_accepts_finite`
- `verify_learning_rate_validation`
- `verify_dropout_rate_validation`
- `verify_l2_lambda_validation`

#### D. Enum Variant Validation from Foreign Callers
- `verify_activation_i32_validation_exhaustive`
- `verify_activation_i32_negative_rejected`
- `verify_optimizer_i32_validation_exhaustive`
- `verify_optimizer_i32_negative_rejected`

#### E. MLP Creation Preconditions
- `verify_ffi_create_rejects_zero_input`
- `verify_ffi_create_rejects_zero_output`
- `verify_ffi_create_rejects_zero_hidden`
- `verify_ffi_create_rejects_excessive_hidden_layers`
- `verify_ffi_create_rejects_oversized_hidden`
- `verify_ffi_hidden_count_i32_negative_as_usize_huge`
- `verify_ffi_i32_min_as_usize_huge`

#### F. Train/Predict Length Validation
- `verify_ffi_train_input_len_validated`
- `verify_ffi_predict_capacity_validated`
- `verify_ffi_predict_output_bounded_by_capacity`

#### G. Layer/Neuron Index Validation
- `verify_ffi_layer_index_negative_rejected`
- `verify_ffi_layer_index_out_of_bounds_safe`
- `verify_ffi_neuron_index_negative_rejected`
- `verify_ffi_weight_index_negative_rejected`

#### H. Histogram Parameter Validation
- `verify_ffi_histogram_bins_negative_rejected`
- `verify_ffi_histogram_bins_zero_rejected`

#### I. Error String Safety
- `verify_ffi_error_nul_byte_sanitized`

#### J. No-Panic Guarantee for All FFI Validators
- `verify_validate_i32_as_usize_no_panic`
- `verify_validate_i32_positive_no_panic`
- `verify_validate_ffi_len_no_panic`
- `verify_validate_f64_param_no_panic`
- `verify_validate_f64_param_range_no_panic`
- `verify_validate_learning_rate_no_panic`
- `verify_validate_dropout_rate_no_panic`
- `verify_validate_l2_lambda_no_panic`
- `verify_validate_activation_i32_no_panic`
- `verify_validate_optimizer_i32_no_panic`

#### K. ABI Type Compatibility
- `verify_activation_type_repr_i32_abi`
- `verify_optimizer_type_repr_i32_abi`
- `verify_f64_abi_compatibility`
- `verify_i32_abi_compatibility`

#### L. Input Array NaN/Infinity Detection
- `verify_ffi_input_array_nan_detection`
- `verify_ffi_input_array_infinity_detection`

#### M. Resource Limits at Boundary
- `verify_ffi_allocation_respects_budget_at_boundary`
- `verify_ffi_to_vec_copy_bounded`

#### N. State Consistency After Parameter Mutation
- `verify_ffi_parameter_mutation_preserves_structure`

#### O. End-to-End FFI Pipeline Validation
- `verify_ffi_complete_train_validation_pipeline`
- `verify_ffi_complete_predict_validation_pipeline`
- `verify_ffi_complete_create_validation_pipeline`

#### P. FFI Setter Value Validation
- `verify_ffi_setter_rejects_nan_value`
- `verify_ffi_setter_rejects_inf_value`
- `verify_ffi_setter_negative_index_rejected`
- `verify_ffi_setter_accepts_valid_params`

---

## **CISA/NSA Compliance**

### Secure by Design

This project follows **CISA (Cybersecurity and Infrastructure Security Agency)** and **NSA (National Security Agency)** Secure by Design principles:

| Principle | Implementation |
|-----------|---------------|
| **Memory Safety** | Rust ownership model eliminates buffer overflows, use-after-free, and data races |
| **Formal Verification** | Kani proofs mathematically verify absence of critical bugs |
| **Input Validation** | All CLI and API inputs validated before processing |
| **Defense in Depth** | Multiple layers of safety (language, compiler, runtime checks) |
| **Secure Defaults** | Safe default configurations throughout |
| **Transparency** | Open source with full code visibility |

### Compliance Checklist

- [x] **Memory-safe language** (Rust implementation)
- [x] **Static analysis** (Rust compiler + Clippy)
- [x] **Formal verification** (Kani proof harnesses)
- [x] **Comprehensive testing** (Unit tests + integration tests)
- [x] **Bounds checking** (Verified array access)
- [x] **Input validation** (CLI and Python API argument parsing)
- [x] **No unsafe code in critical paths** (Where possible)
- [x] **Documentation** (Inline docs + README)
- [x] **Version control** (Git)
- [x] **License clarity** (MIT License)

### Attestation

This codebase has been developed following secure software development lifecycle (SSDLC) practices and demonstrates:

- **Comprehensive test suites** across all implementations
- **Zero warnings** compilation across all implementations
- **Consistent API** across all language/backend combinations
- **Production-ready** code quality

---

## **License**

MIT License

Copyright (c) 2025 Matthew Abbott

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

---

## **Author**

**Matthew Abbott**  
Email: mattbachg@gmail.com

---

*Built with precision. Verified with rigor. Secured by design.*
