#!/usr/bin/env node
/**
 * Example showing how to use different GPU backends
 */

const { MLP, JsActivationType } = require('..');

function main() {
  console.log('='.repeat(60));
  console.log('GPU Backend Comparison Example');
  console.log('='.repeat(60));
  console.log();

  console.log('Available backends:', MLP.availableBackends());
  console.log();

  // Create with auto-selected backend
  const mlpAuto = new MLP(2, [8], 1, { gpuBackend: 'auto' });
  console.log(`✓ Auto-selected backend: ${mlpAuto.gpuBackend}`);
  console.log();

  // Try CUDA
  try {
    const mlpCuda = new MLP(2, [8], 1, { gpuBackend: 'cuda' });
    console.log(`✓ CUDA backend initialized: ${mlpCuda.gpuBackend}`);
  } catch (e) {
    console.log(`✗ CUDA not available: ${e.message}`);
  }
  console.log();

  // Try OpenCL
  try {
    const mlpOpencl = new MLP(2, [8], 1, { gpuBackend: 'opencl' });
    console.log(`✓ OpenCL backend initialized: ${mlpOpencl.gpuBackend}`);
  } catch (e) {
    console.log(`✗ OpenCL not available: ${e.message}`);
  }
  console.log();

  // CPU fallback always works
  const mlpCpu = new MLP(2, [8], 1, { gpuBackend: 'cpu' });
  console.log(`✓ CPU backend initialized: ${mlpCpu.gpuBackend}`);
  console.log();

  // Switch backends dynamically
  console.log('-'.repeat(60));
  console.log('Dynamic Backend Switching:');
  console.log('-'.repeat(60));

  const mlp = new MLP(2, [8], 1, { gpuBackend: 'cpu' });
  console.log(`Started with: ${mlp.gpuBackend}`);

  for (const backend of ['cuda', 'opencl', 'cpu']) {
    try {
      mlp.setBackend(backend);
      console.log(`✓ Switched to: ${mlp.gpuBackend}`);
    } catch (e) {
      console.log(`✗ Cannot switch to ${backend}: ${e.message}`);
    }
  }
  console.log();

  // Train with selected backend
  const X = [[0, 0], [0, 1], [1, 0], [1, 1]];
  const y = [[0], [1], [1], [0]];

  console.log('-'.repeat(60));
  console.log(`Training XOR with ${mlp.gpuBackend} backend...`);
  console.log('-'.repeat(60));

  mlp.learningRate = 0.5;
  const result = mlp.fit(X, y, 1000, false);
  console.log(`Final loss: ${result.finalLoss.toFixed(6)}`);

  const predictions = mlp.predictBatch(X);
  console.log('\nPredictions:');
  for (let i = 0; i < X.length; i++) {
    console.log(`  [${X[i]}] -> ${predictions[i][0].toFixed(4)}`);
  }
}

main();
