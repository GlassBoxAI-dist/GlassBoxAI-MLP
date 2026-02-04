# GlassBoxAI-MLP

## **Multi-Layer Perceptron Suite**

### *GPU-Accelerated MLP with Python & Node.js Bindings and Formal Verification*

---

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![CUDA](https://img.shields.io/badge/CUDA-12.0-green.svg)](https://developer.nvidia.com/cuda-toolkit)
[![OpenCL](https://img.shields.io/badge/OpenCL-3.0-blue.svg)](https://www.khronos.org/opencl/)
[![Rust](https://img.shields.io/badge/Rust-1.75+-orange.svg)](https://www.rust-lang.org/)
[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![Node.js](https://img.shields.io/badge/Node.js-16+-339933.svg)](https://nodejs.org/)
[![Kani](https://img.shields.io/badge/Kani-Verified-brightgreen.svg)](https://model-checking.github.io/kani/)
[![CISA Compliant](https://img.shields.io/badge/CISA-Secure%20by%20Design-blue.svg)](https://www.cisa.gov/securebydesign)

---

## **Overview**

GlassBoxAI-MLP is a comprehensive, production-ready Multi-Layer Perceptron implementation suite featuring:

- **Multiple GPU backends**: CUDA and OpenCL acceleration with automatic backend selection
- **Python bindings**: Full-featured Python API via PyO3 and maturin
- **Node.js bindings**: Full-featured Node.js API via napi-rs
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
6. [Python API Reference](#python-api-reference)
7. [Node.js API Reference](#nodejs-api-reference)
8. [CLI Reference](#cli-reference)
9. [Testing](#testing)
10. [Formal Verification with Kani](#formal-verification-with-kani)
11. [CISA/NSA Compliance](#cisansa-compliance)
12. [License](#license)
13. [Author](#author)

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
┌─────────────────────────────────────────────────────────────────┐
│                        GlassBoxAI-MLP                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌────────────────────────────┐  ┌──────────────────────────────┐│
│  │      Python Bindings      │  │      Node.js Bindings        ││
│  │  • facaded_mlp_cuda (PyO3)│  │  • facaded-mlp-cuda (napi-rs)││
│  │  • MLP class with full API│  │  • MLP class with full API   ││
│  │  • PyActivationType enum  │  │  • JsActivationType enum     ││
│  │  • load_csv, normalize    │  │  • loadCsv, normalize        ││
│  └────────────────────────────┘  └──────────────────────────────┘│
│                              │                                  │
│  ┌───────────────────────────┴─────────────────────────────────┐│
│  │                      Rust Core                              ││
│  ├─────────────┬─────────────┬─────────────────────────────────┤│
│  │   CUDA      │   OpenCL    │          CPU                    ││
│  │  Backend    │   Backend   │        Backend                  ││
│  │ (cudarc)    │   (ocl)     │     (Pure Rust)                 ││
│  └─────────────┴─────────────┴─────────────────────────────────┘│
│                                                                 │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │                    Shared Features                          ││
│  │  • Consistent API across all backends                       ││
│  │  • JSON-compatible model format                             ││
│  │  • ONNX import/export                                       ││
│  │  • Comprehensive test suites                                ││
│  └─────────────────────────────────────────────────────────────┘│
│                                                                 │
│  ┌─────────────┐                                                │
│  │    Kani     │                                                │
│  │  Proofs     │                                                │
│  │  (Formal    │                                                │
│  │  Verify)    │                                                │
│  └─────────────┘                                                │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
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
├── kani/                       # Formal verification proofs
│   ├── Cargo.toml
│   ├── lib.rs
│   ├── core_types.rs
│   ├── harnesses.rs
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
| `feature_importance()` | Calculate feature importance scores |

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
| `featureImportance()` | Returns `{ featureIndex, score }[]` |

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

## **CLI Reference**

### Usage

```
facaded_mlp_cuda <command> [options]
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
| `get-bias` | Get bias for a neuron |
| `set-bias` | Set bias for a neuron |
| `layer-info` | Display layer information |
| `histogram` | Display activation/error histogram |

### Options

| Option | Description |
|--------|-------------|
| `-i, --inputs=N` | Number of input neurons |
| `-H, --hidden=N,M,...` | Hidden layer sizes |
| `-o, --outputs=N` | Number of output neurons |
| `-m, --model=FILE` | Model file path |
| `-s, --save=FILE` | Save output to file |
| `-d, --data=FILE` | Training data CSV |
| `--epochs=N` | Training epochs |
| `--lr=VALUE` | Learning rate |
| `--activation=TYPE` | Activation function |
| `--optimizer=TYPE` | Optimizer type |
| `--gpu=BACKEND` | GPU backend: auto, cuda, opencl, cpu |

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
cargo kani

# Run specific proof
cargo kani --harness proof_name

# Run unit tests
cargo test
```

### Why Formal Verification Matters

Traditional testing can only verify specific test cases. Formal verification with Kani:

- **Exhaustively checks all possible inputs** within defined bounds
- **Mathematically proves** absence of panics, buffer overflows, and undefined behavior
- **Catches edge cases** that random testing might miss
- **Provides cryptographic-level assurance** for safety-critical code

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
