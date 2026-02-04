//! XOR Example - Demonstrates basic MLP usage with introspection
//!
//! This example shows how to:
//! - Create an MLP
//! - Configure optimizer and learning rate
//! - Train on the XOR problem
//! - Make predictions
//! - Introspect weights, biases, and optimizer state
//! - Save and load models
//!
//! Run with: `cargo run --example xor`

use glassboxai_mlp::{
    MLP, MlpConfig, ActivationType, OptimizerType, BackendType,
    available_backends,
};

fn main() -> Result<(), String> {
    println!("=== GlassBoxAI MLP - XOR Example ===\n");

    // Show available backends
    println!("Available GPU backends: {:?}", available_backends());
    println!();

    // Create an MLP with custom configuration
    let config = MlpConfig {
        hidden_activation: ActivationType::Sigmoid,
        output_activation: ActivationType::Sigmoid,
        learning_rate: 0.5,
        optimizer: OptimizerType::Adam,
        backend: BackendType::CPU, // Use CPU for portability
        ..Default::default()
    };

    let mut mlp = MLP::with_config(2, &[8], 1, config)?;
    println!("Created: {}", mlp);
    println!();

    // XOR training data
    let inputs = vec![
        vec![0.0, 0.0],
        vec![0.0, 1.0],
        vec![1.0, 0.0],
        vec![1.0, 1.0],
    ];
    let targets = vec![
        vec![0.0],
        vec![1.0],
        vec![1.0],
        vec![0.0],
    ];

    // Train
    println!("Training on XOR problem...");
    let result = mlp.fit(&inputs, &targets, 1000, true)?;
    println!("\nFinal loss: {:.6}", result.final_loss);
    println!();

    // Predictions
    println!("Predictions:");
    for (input, target) in inputs.iter().zip(targets.iter()) {
        let output = mlp.predict(input)?;
        let correct = (output[0] - target[0]).abs() < 0.5;
        println!(
            "  [{:.0}, {:.0}] => {:.4} (expected: {:.0}) {}",
            input[0], input[1], output[0], target[0],
            if correct { "✓" } else { "✗" }
        );
    }
    println!();

    // ========== Introspection Demo ==========
    println!("=== Glass Box Introspection ===\n");

    // Layer information
    println!("Network structure:");
    for layer in 0..mlp.num_layers() {
        let info = mlp.layer_info(layer);
        println!(
            "  Layer {}: {} neurons, activation: {}",
            info.index, info.size, info.activation
        );
    }
    println!();

    // Hidden layer neuron details
    println!("Hidden layer neurons (layer 1):");
    for neuron_idx in 0..2 {  // Show first 2 neurons
        let neuron = mlp.neuron_view(1, neuron_idx);
        println!("  Neuron {}:", neuron_idx);
        println!("    Weights: [{:.4}, {:.4}]", neuron.weights[0], neuron.weights[1]);
        println!("    Bias: {:.4}", neuron.bias);
        println!("    Last output: {:.4}", neuron.output);
    }
    println!();

    // Optimizer state (Adam M and V values)
    println!("Adam optimizer state (layer 1, neuron 0):");
    for weight_idx in 0..2 {
        let m = mlp.get_weight_m(1, 0, weight_idx);
        let v = mlp.get_weight_v(1, 0, weight_idx);
        println!("  Weight {}: M={:.6}, V={:.6}", weight_idx, m, v);
    }
    let bias_m = mlp.get_bias_m(1, 0);
    let bias_v = mlp.get_bias_v(1, 0);
    println!("  Bias: M={:.6}, V={:.6}", bias_m, bias_v);
    println!("  Timestep: {}", mlp.timestep());
    println!();

    // Feature importance
    println!("Feature importance:");
    let importance = mlp.feature_importance();
    for fi in &importance {
        println!("  Feature {}: {:.4}", fi.index, fi.score);
    }
    println!();

    // Activation histogram
    println!("Hidden layer activation histogram (10 bins):");
    // Run a prediction first to populate outputs
    mlp.predict(&[1.0, 0.0])?;
    let hist = mlp.activation_histogram(1, 10);
    println!("  {:?}", hist);
    println!();

    // ========== Save and Load ==========
    println!("=== Save/Load Demo ===\n");

    let filename = "xor_model_example.json";
    println!("Saving model to {}...", filename);
    mlp.save(filename)?;

    println!("Loading model...");
    let loaded_mlp = MLP::load(filename)?;
    println!("Loaded: {}", loaded_mlp);

    // Verify loaded model
    println!("\nVerifying loaded model predictions:");
    for (input, target) in inputs.iter().zip(targets.iter()) {
        let mut loaded = MLP::load(filename)?;
        let output = loaded.predict(input)?;
        println!(
            "  [{:.0}, {:.0}] => {:.4} (expected: {:.0})",
            input[0], input[1], output[0], target[0]
        );
    }

    // Clean up
    std::fs::remove_file(filename).ok();

    println!("\n=== Done! ===");
    Ok(())
}
