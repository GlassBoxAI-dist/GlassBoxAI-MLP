// XOR Example - Demonstrates basic MLP usage with introspection
//
// Build:
//   go build -o xor
//
// Run:
//   LD_LIBRARY_PATH=../../target/release ./xor
package main

import (
	"fmt"
	"log"
	"os"

	"github.com/GlassBoxAI-dist/GlassBoxAI-MLP/go/facadedmlp"
)

func main() {
	fmt.Println("=== GlassBoxAI MLP - XOR Example (Go) ===")
	fmt.Println()

	// Show available backends
	backends := facadedmlp.AvailableBackends()
	fmt.Printf("Available GPU backends: %v\n\n", backends)

	// Create an MLP with custom configuration
	config := &facadedmlp.Config{
		HiddenActivation: facadedmlp.Sigmoid,
		OutputActivation: facadedmlp.Sigmoid,
		Backend:          facadedmlp.BackendCPU,
		LearningRate:     0.5,
		Optimizer:        facadedmlp.Adam,
	}

	mlp, err := facadedmlp.New(2, []int{8}, 1, config)
	if err != nil {
		log.Fatalf("Failed to create MLP: %v", err)
	}
	defer mlp.Close()

	fmt.Printf("Created: %s\n\n", mlp)

	// XOR training data
	inputs := [][]float64{
		{0.0, 0.0},
		{0.0, 1.0},
		{1.0, 0.0},
		{1.0, 1.0},
	}
	targets := [][]float64{
		{0.0},
		{1.0},
		{1.0},
		{0.0},
	}

	// Train
	fmt.Println("Training on XOR problem...")
	result, err := mlp.Fit(inputs, targets, 1000, true)
	if err != nil {
		log.Fatalf("Training failed: %v", err)
	}
	fmt.Printf("\nFinal loss: %.6f\n\n", result.FinalLoss)

	// Predictions
	fmt.Println("Predictions:")
	for i, input := range inputs {
		output, err := mlp.Predict(input)
		if err != nil {
			log.Fatalf("Prediction failed: %v", err)
		}
		correct := "✓"
		if (output[0] >= 0.5) != (targets[i][0] >= 0.5) {
			correct = "✗"
		}
		fmt.Printf("  [%.0f, %.0f] => %.4f (expected: %.0f) %s\n",
			input[0], input[1], output[0], targets[i][0], correct)
	}
	fmt.Println()

	// ========== Introspection Demo ==========
	fmt.Println("=== Glass Box Introspection ===")
	fmt.Println()

	// Network structure
	fmt.Println("Network structure:")
	fmt.Printf("  Input layer: %d neurons\n", mlp.InputSize())
	fmt.Printf("  Hidden layers: %v\n", mlp.HiddenSizes())
	fmt.Printf("  Output layer: %d neurons\n", mlp.OutputSize())
	fmt.Printf("  Total layers: %d\n", mlp.NumLayers())
	fmt.Println()

	// Hidden layer neuron details
	fmt.Println("Hidden layer neurons (layer 1):")
	for neuronIdx := 0; neuronIdx < 2; neuronIdx++ {
		view := mlp.NeuronView(1, neuronIdx)
		fmt.Printf("  Neuron %d:\n", neuronIdx)
		fmt.Printf("    Weights: [%.4f, %.4f]\n", view.Weights[0], view.Weights[1])
		fmt.Printf("    Bias: %.4f\n", view.Bias)
	}
	fmt.Println()

	// Feature importance
	fmt.Println("Feature importance:")
	importance := mlp.FeatureImportance()
	for _, fi := range importance {
		fmt.Printf("  Feature %d: %.4f\n", fi.Index, fi.Score)
	}
	fmt.Println()

	// Layer outputs
	mlp.Predict([]float64{1.0, 0.0})
	outputs := mlp.GetLayerOutputs(1)
	fmt.Printf("Hidden layer outputs (after predicting [1,0]): %v\n", outputs[:4])
	fmt.Println()

	// ========== Save and Load ==========
	fmt.Println("=== Save/Load Demo ===")
	fmt.Println()

	filename := "xor_model_go.json"
	fmt.Printf("Saving model to %s...\n", filename)
	if err := mlp.Save(filename); err != nil {
		log.Fatalf("Save failed: %v", err)
	}

	fmt.Println("Loading model...")
	loadedMLP, err := facadedmlp.Load(filename)
	if err != nil {
		log.Fatalf("Load failed: %v", err)
	}
	defer loadedMLP.Close()
	fmt.Printf("Loaded: %s\n", loadedMLP)

	// Verify loaded model
	fmt.Println("\nVerifying loaded model predictions:")
	for i, input := range inputs {
		output, _ := loadedMLP.Predict(input)
		fmt.Printf("  [%.0f, %.0f] => %.4f (expected: %.0f)\n",
			input[0], input[1], output[0], targets[i][0])
	}

	// Clean up
	os.Remove(filename)

	fmt.Println("\n=== Done! ===")
}
