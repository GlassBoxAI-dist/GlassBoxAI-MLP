/**
 * @file
 * @ingroup MLP_Internal_Logic
 */
#!/usr/bin/env node
/**
 * XOR problem example using facaded-mlp-cuda
 */

const { MLP, JsActivationType, JsOptimizerType } = require('..');

function main() {
  // Show available backends
  console.log('Available GPU backends:', MLP.availableBackends());
  console.log();

  // Create MLP for XOR problem with auto-selected backend
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
  console.log();

  // XOR training data
  const X = [
    [0.0, 0.0],
    [0.0, 1.0],
    [1.0, 0.0],
    [1.0, 1.0],
  ];
  const y = [
    [0.0],
    [1.0],
    [1.0],
    [0.0],
  ];

  // Train
  console.log('Training...');
  const result = mlp.fit(X, y, 2000, true);

  console.log(`\nFinal loss: ${result.finalLoss.toFixed(6)}`);
  console.log();

  // Test predictions
  console.log('Predictions:');
  const predictions = mlp.predictBatch(X);
  for (let i = 0; i < X.length; i++) {
    console.log(
      `  Input: [${X[i]}] -> Target: ${y[i][0].toFixed(1)}, Prediction: ${predictions[i][0].toFixed(4)}`
    );
  }

  // Save model
  mlp.save('xor_model.json');
  console.log('\nModel saved to xor_model.json');

  // Load and test
  const mlp2 = MLP.load('xor_model.json');
  console.log(`Loaded model: ${mlp2.toString()}`);

  const testInput = [1.0, 0.0];
  const output = mlp2.predict(testInput);
  console.log(`\nTest prediction for [${testInput}]: ${output[0].toFixed(4)}`);

  // Feature importance
  const importance = mlp.featureImportance();
  console.log('\nFeature importance:');
  for (const { featureIndex, score } of importance) {
    console.log(`  Feature ${featureIndex}: ${score.toFixed(6)}`);
  }
}

main();
