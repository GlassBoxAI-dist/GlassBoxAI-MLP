## @file
## @ingroup MLP_Internal_Logic
#!/usr/bin/env python3
"""
XOR problem example using facaded_mlp_cuda
"""

from facaded_mlp_cuda import MLP, PyActivationType, PyOptimizerType

def main():
    # Show available backends
    print("Available GPU backends:", MLP.available_backends())
    print()
    
    # Create MLP for XOR problem with auto-selected backend
    mlp = MLP(
        input_size=2,
        hidden_sizes=[8],
        output_size=1,
        hidden_activation=PyActivationType.Sigmoid,
        output_activation=PyActivationType.Sigmoid,
        gpu_backend="auto"  # or "cuda", "opencl", "cpu"
    )
    
    # Set hyperparameters
    mlp.learning_rate = 0.5
    mlp.optimizer = PyOptimizerType.Adam
    
    print(f"Created model: {mlp}")
    print(f"Using backend: {mlp.gpu_backend}")
    print()
    
    # XOR training data
    X = [
        [0.0, 0.0],
        [0.0, 1.0],
        [1.0, 0.0],
        [1.0, 1.0]
    ]
    y = [
        [0.0],
        [1.0],
        [1.0],
        [0.0]
    ]
    
    # Train
    print("Training...")
    losses = mlp.fit(X, y, epochs=2000, verbose=True)
    
    print(f"\nFinal loss: {losses[-1]:.6f}")
    print()
    
    # Test predictions
    print("Predictions:")
    predictions = mlp.predict_batch(X)
    for input_val, target_val, pred_val in zip(X, y, predictions):
        print(f"  Input: {input_val} -> Target: {target_val[0]:.1f}, Prediction: {pred_val[0]:.4f}")
    
    # Save model
    mlp.save("xor_model.json")
    print("\nModel saved to xor_model.json")
    
    # Load and test
    mlp2 = MLP.load("xor_model.json")
    print(f"Loaded model: {mlp2}")
    
    test_input = [1.0, 0.0]
    output = mlp2.predict(test_input)
    print(f"\nTest prediction for {test_input}: {output[0]:.4f}")
    
    # Feature importance
    importance = mlp.feature_importance()
    print("\nFeature importance:")
    for feature_idx, score in importance:
        print(f"  Feature {feature_idx}: {score:.6f}")

if __name__ == "__main__":
    main()
