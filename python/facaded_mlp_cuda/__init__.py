## @file
## @ingroup MLP_Wrappers
"""
Facaded MLP CUDA/OpenCL - GPU-accelerated Multi-Layer Perceptron

Example usage:
    >>> from facaded_mlp_cuda import MLP, PyActivationType
    >>> 
    >>> # Check available backends
    >>> print("Available:", MLP.available_backends())
    >>> 
    >>> # Create with auto-selected backend
    >>> mlp = MLP(2, [8], 1, gpu_backend="auto")
    >>> print(f"Using: {mlp.gpu_backend}")
    >>> 
    >>> # XOR problem
    >>> X = [[0,0], [0,1], [1,0], [1,1]]
    >>> y = [[0], [1], [1], [0]]
    >>> losses = mlp.fit(X, y, epochs=1000, verbose=True)
    >>> 
    >>> predictions = mlp.predict_batch(X)
    >>> print(predictions)
"""

from .facaded_mlp_cuda import (
    MLP,
    PyActivationType,
    PyOptimizerType,
    load_csv,
    normalize,
)

__all__ = [
    'MLP',
    'PyActivationType',
    'PyOptimizerType',
    'load_csv',
    'normalize',
]

__version__ = '1.0.0'
