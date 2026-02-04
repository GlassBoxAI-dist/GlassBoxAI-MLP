# Building Facaded MLP CUDA/OpenCL

## Prerequisites

### For CUDA support:
- NVIDIA GPU
- CUDA Toolkit 11.0 or higher
- Rust toolchain

### For OpenCL support:
- OpenCL-compatible GPU (NVIDIA, AMD, Intel)
- OpenCL runtime/SDK
- Rust toolchain

### For Python:
- Python 3.8 or higher
- maturin (`pip install maturin`)

## Feature Flags

- `cuda` - Enable CUDA support
- `opencl` - Enable OpenCL support
- `cli` - Build command-line interface
- `python` - Build Python bindings

## Build Commands

### Rust Library
```bash
# CUDA only
cargo build --release --features cuda

# OpenCL only
cargo build --release --features opencl

# Both CUDA and OpenCL
cargo build --release --features cuda,opencl

# CLI with all backends
cargo build --release --features cli,cuda,opencl
```

### Python Package
```bash
# Install maturin
pip install maturin

# Build and install (all backends)
maturin develop --release --features python,cuda,opencl

# Build wheel for distribution
maturin build --release --features python,cuda,opencl

# CPU only (no GPU dependencies)
maturin develop --release --features python
```

### Node.js Package
```bash
# Install napi-rs CLI
npm install -g @napi-rs/cli

# Build and install (all backends)
npm run build

# CPU only (no GPU dependencies)
npm run build:cpu

# Debug build
npm run build:debug
```

## Quick Start

### Using Makefile
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

### Manual Installation
```bash
# Clone repository
git clone <repo-url>
cd facaded_mlp_cuda

# Install Python package
pip install maturin
maturin develop --release --features python,cuda,opencl

# Run example
python examples/xor_example.py
```

## Usage Examples

### CLI
```bash
# Auto-detect best backend
facaded_mlp_cuda create -i 2 -H 8 -o 1 -s model.json

# Force CUDA
facaded_mlp_cuda create -i 2 -H 8 -o 1 -s model.json --gpu=cuda

# Force OpenCL
facaded_mlp_cuda create -i 2 -H 8 -o 1 -s model.json --gpu=opencl

# Force CPU
facaded_mlp_cuda create -i 2 -H 8 -o 1 -s model.json --gpu=cpu

# Train
facaded_mlp_cuda train -m model.json -d data.csv -s trained.json --epochs=1000

# Predict
facaded_mlp_cuda predict -m trained.json -i 1.0,0.0
```

### Python
```python
from facaded_mlp_cuda import MLP, PyActivationType

# Check available backends
print(MLP.available_backends())

# Auto-select backend
mlp = MLP(2, [8], 1, gpu_backend="auto")

# Specific backend
mlp_cuda = MLP(2, [8], 1, gpu_backend="cuda")
mlp_opencl = MLP(2, [8], 1, gpu_backend="opencl")
mlp_cpu = MLP(2, [8], 1, gpu_backend="cpu")

# Train
X = [[0,0], [0,1], [1,0], [1,1]]
y = [[0], [1], [1], [0]]
losses = mlp.fit(X, y, epochs=1000, verbose=True)

# Predict
predictions = mlp.predict_batch(X)

# Save/Load
mlp.save("model.json")
loaded = MLP.load("model.json")
```

### Node.js
```javascript
const { MLP, JsActivationType } = require('facaded-mlp-cuda');

// Check available backends
console.log(MLP.availableBackends());

// Auto-select backend
const mlp = new MLP(2, [8], 1, { gpuBackend: 'auto' });

// Specific backend
const mlpCuda = new MLP(2, [8], 1, { gpuBackend: 'cuda' });
const mlpOpencl = new MLP(2, [8], 1, { gpuBackend: 'opencl' });
const mlpCpu = new MLP(2, [8], 1, { gpuBackend: 'cpu' });

// Train
const X = [[0,0], [0,1], [1,0], [1,1]];
const y = [[0], [1], [1], [0]];
const result = mlp.fit(X, y, 1000, true);

// Predict
const predictions = mlp.predictBatch(X);

// Save/Load
mlp.save('model.json');
const loaded = MLP.load('model.json');
```

### Rust Library
```rust
use facaded_mlp_cuda::{
    TMultiLayerPerceptronCUDA, 
    TActivationType, 
    TGPUBackend,
    select_best_backend
};

fn main() -> Result<(), String> {
    // Auto-select best backend
    let backend = select_best_backend();
    
    // Or choose manually
    let backend = TGPUBackend::CUDA;
    // let backend = TGPUBackend::OpenCL;
    // let backend = TGPUBackend::CPU;
    
    let mut mlp = TMultiLayerPerceptronCUDA::new_with_backend(
        2,
        &vec![8],
        1,
        TActivationType::atSigmoid,
        TActivationType::atSigmoid,
        backend,
    )?;
    
    let input = vec![1.0, 0.0];
    let target = vec![1.0];
    mlp.Train(&input, &target)?;
    
    let output = mlp.Predict(&input)?;
    println!("Output: {:?}", output);
    
    Ok(())
}
```

## Troubleshooting

### OpenCL not found
```bash
# Ubuntu/Debian
sudo apt-get install ocl-icd-opencl-dev

# Fedora/RHEL
sudo dnf install ocl-icd-devel

# macOS
# OpenCL is included with Xcode Command Line Tools
```

### CUDA not found
```bash
# Ensure CUDA_PATH is set
export CUDA_PATH=/usr/local/cuda

# Or install CUDA toolkit from NVIDIA
```

### Python module not found
```bash
# Ensure maturin installed the package
pip list | grep facaded-mlp-cuda

# If not found, reinstall
maturin develop --release --features python,cuda,opencl
```
