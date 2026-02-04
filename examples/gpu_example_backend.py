#!/usr/bin/env python3
"""
Example showing how to use different GPU backends
"""

from facaded_mlp_cuda import MLP, PyActivationType

def main():
    print("=" * 60)
    print("GPU Backend Comparison Example")
    print("=" * 60)
    print()
    
    print("Available backends:", MLP.available_backends())
    print()
    
    # Create with auto-selected backend
    mlp_auto = MLP(2, [8], 1, gpu_backend="auto")
    print(f"✓ Auto-selected backend: {mlp_auto.gpu_backend}")
    print()
    
    # Try CUDA
    try:
        mlp_cuda = MLP(2, [8], 1, gpu_backend="cuda")
        print(f"✓ CUDA backend initialized: {mlp_cuda.gpu_backend}")
    except Exception as e:
        print(f"✗ CUDA not available: {e}")
    print()
    
    # Try OpenCL
    try:
        mlp_opencl = MLP(2, [8], 1, gpu_backend="opencl")
        print(f"✓ OpenCL backend initialized: {mlp_opencl.gpu_backend}")
    except Exception as e:
        print(f"✗ OpenCL not available: {e}")
    print()
    
    # CPU fallback always works
    mlp_cpu = MLP(2, [8], 1, gpu_backend="cpu")
    print(f"✓ CPU backend initialized: {mlp_cpu.gpu_backend}")
    print()
    
    # Switch backends dynamically
    print("-" * 60)
    print("Dynamic Backend Switching:")
    print("-" * 60)
    mlp = MLP(2, [8], 1, gpu_backend="cpu")
    print(f"Started with: {mlp.gpu_backend}")
    
    for backend in ["cuda", "opencl", "cpu"]:
        try:
            mlp.set_backend(backend)
            print(f"✓ Switched to: {mlp.gpu_backend}")
        except Exception as e:
            print(f"✗ Cannot switch to {backend}: {e}")
    print()
    
    # Train with selected backend
    X = [[0,0], [0,1], [1,0], [1,1]]
    y = [[0], [1], [1], [0]]
    
    print("-" * 60)
    print(f"Training XOR with {mlp.gpu_backend} backend...")
    print("-" * 60)
    mlp.learning_rate = 0.5
    losses = mlp.fit(X, y, epochs=1000, verbose=False)
    print(f"Final loss: {losses[-1]:.6f}")
    
    predictions = mlp.predict_batch(X)
    print("\nPredictions:")
    for inp, pred in zip(X, predictions):
        print(f"  {inp} -> {pred[0]:.4f}")

if __name__ == "__main__":
    main()
