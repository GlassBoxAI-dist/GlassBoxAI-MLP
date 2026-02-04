/**
 * Facaded MLP CUDA/OpenCL - GPU-accelerated Multi-Layer Perceptron
 *
 * @example
 * const { MLP, JsActivationType, JsOptimizerType } = require('facaded-mlp-cuda');
 *
 * // Check available backends
 * console.log('Available:', MLP.availableBackends());
 *
 * // Create with auto-selected backend
 * const mlp = new MLP(2, [8], 1, { gpuBackend: 'auto' });
 * console.log(`Using: ${mlp.gpuBackend}`);
 *
 * // XOR problem
 * const X = [[0,0], [0,1], [1,0], [1,1]];
 * const y = [[0], [1], [1], [0]];
 * const result = mlp.fit(X, y, 1000, true);
 *
 * const predictions = mlp.predictBatch(X);
 * console.log(predictions);
 */

const { MLP, JsActivationType, JsOptimizerType, loadCsv, normalize } = require('./facaded-mlp-cuda.node');

module.exports = {
  MLP,
  JsActivationType,
  JsOptimizerType,
  loadCsv,
  normalize,
};
