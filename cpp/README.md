# FacadedMLP C++ Wrapper

C++ wrapper for the GPU-accelerated Multi-Layer Perceptron library with CUDA/OpenCL/CPU backends.

## Features

- Header-only C++ library (`facaded_mlp.hpp`)
- C API header for interop (`facaded_mlp.h`)
- RAII-based memory management
- Move semantics support
- Exception-based error handling
- CMake integration

## Building

### 1. Build the Rust library

```bash
cd /path/to/GlassBoxAI-MLP
cargo build --release --features julia
```

### 2. Build C++ examples with CMake

```bash
cd cpp
mkdir build && cd build
cmake ..
make
```

### 3. Or compile directly

```bash
g++ -std=c++17 -O2 -I include examples/xor_example.cpp \
    -L ../target/release -lfacaded_mlp_cuda \
    -Wl,-rpath,../target/release \
    -o xor_example
```

## Quick Start

```cpp
#include "facaded_mlp.hpp"
#include <iostream>

int main() {
    using namespace facaded;
    
    // Create a network: 2 inputs, 8 hidden neurons, 1 output
    MLP mlp(2, {8}, 1);
    
    // XOR training data
    std::vector<std::vector<double>> X = {{0,0}, {0,1}, {1,0}, {1,1}};
    std::vector<std::vector<double>> y = {{0}, {1}, {1}, {0}};
    
    // Train
    auto result = mlp.fit(X, y, 1000, true);
    
    // Predict
    for (const auto& x : X) {
        auto output = mlp.predict(x);
        std::cout << x[0] << ", " << x[1] << " => " << output[0] << std::endl;
    }
    
    return 0;
}
```

## API Reference

### Creating Models

```cpp
using namespace facaded;

// Basic creation
MLP mlp(input_size, hidden_sizes, output_size);

// With options
MLPOptions opts;
opts.hidden_activation = Activation::ReLU;
opts.output_activation = Activation::Sigmoid;
opts.backend = "cuda";  // "auto", "cpu", "cuda", "opencl"
opts.learning_rate = 0.001;
opts.optimizer = Optimizer::Adam;
opts.dropout_rate = 0.1;
opts.l2_lambda = 0.0001;
opts.batch_norm = true;

MLP mlp(784, {256, 128}, 10, opts);
```

### Training

```cpp
// Single sample
mlp.train(input, target);

// Full dataset
TrainResult result = mlp.fit(inputs, targets, epochs, verbose);
std::cout << "Final loss: " << result.final_loss << std::endl;
```

### Prediction

```cpp
std::vector<double> output = mlp.predict(input);
std::vector<std::vector<double>> outputs = mlp.predict_batch(inputs);
```

### Properties

```cpp
// Getters
double lr = mlp.learning_rate();
Optimizer opt = mlp.optimizer();
double dr = mlp.dropout_rate();
std::string backend = mlp.backend();
int in_size = mlp.input_size();
int out_size = mlp.output_size();
const auto& hidden = mlp.hidden_sizes();

// Setters
mlp.set_learning_rate(0.001);
mlp.set_optimizer(Optimizer::SGD);
mlp.set_dropout_rate(0.2);
mlp.set_backend("cuda");
```

### Save/Load

```cpp
mlp.save("model.json");
MLP loaded = MLP::load("model.json");
```

### Advanced

```cpp
// Feature importance
std::vector<FeatureImportance> importance = mlp.feature_importance();
for (const auto& fi : importance) {
    std::cout << "Feature " << fi.index << ": " << fi.score << std::endl;
}

// Available backends
std::vector<std::string> backends = MLP::available_backends();

// Neuron access
std::vector<double> weights = mlp.get_neuron_weights(layer, neuron);
double bias = mlp.get_neuron_bias(layer, neuron);
mlp.set_neuron_weight(layer, neuron, weight_idx, value);
mlp.set_neuron_bias(layer, neuron, value);

// Layer outputs (after prediction)
std::vector<double> outputs = mlp.get_layer_outputs(layer);

// String representation
std::cout << mlp << std::endl;
```

## Enums

### Activation

```cpp
enum class Activation {
    Sigmoid,
    Tanh,
    ReLU,
    Softmax
};
```

### Optimizer

```cpp
enum class Optimizer {
    SGD,
    Adam,
    RMSProp
};
```

## Error Handling

The library uses exceptions for error handling:

```cpp
try {
    MLP mlp(2, {8}, 1);
    mlp.train(input, target);
} catch (const facaded::MLPException& e) {
    std::cerr << "Error: " << e.what() << std::endl;
}
```

## C API

For C interop or when you need a C-compatible interface, include `facaded_mlp.h`:

```c
#include "facaded_mlp.h"

mlp_handle_t mlp = mlp_create(2, hidden, 1, 1, 
    MLP_ACTIVATION_SIGMOID, MLP_ACTIVATION_SIGMOID, "auto");

mlp_train(mlp, input, 2, target, 1);

double output[1];
mlp_predict(mlp, input, 2, output, 1);

mlp_destroy(mlp);
```

## CMake Integration

```cmake
# Add as subdirectory
add_subdirectory(path/to/GlassBoxAI-MLP/cpp)
target_link_libraries(your_target PRIVATE facaded_mlp)

# Or after installing
find_package(facaded_mlp REQUIRED)
target_link_libraries(your_target PRIVATE facaded::facaded_mlp)
```

## License

MIT License - Copyright (c) 2025 Matthew Abbott
