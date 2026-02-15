# GlassBoxAI MLP - C# Wrapper

C# bindings for the GlassBoxAI MLP library. Provides a GPU-accelerated Multi-Layer
Perceptron with full introspection for explainable AI.

## Prerequisites

- .NET 8.0 SDK or later
- Built native library (`cargo build --release --features julia`)

## Quick Start

```csharp
using GlassBoxAI.MLP;

using var mlp = new MLP(2, new[] { 8 }, 1);
mlp.LearningRate = 0.5;
mlp.Optimizer = OptimizerType.Adam;

var inputs = new[] {
    new[] { 0.0, 0.0 }, new[] { 0.0, 1.0 },
    new[] { 1.0, 0.0 }, new[] { 1.0, 1.0 }
};
var targets = new[] {
    new[] { 0.0 }, new[] { 1.0 },
    new[] { 1.0 }, new[] { 0.0 }
};

var result = mlp.Fit(inputs, targets, 1000, verbose: true);
Console.WriteLine($"Final loss: {result.FinalLoss:F6}");

var output = mlp.Predict(new[] { 1.0, 0.0 });
Console.WriteLine($"Prediction: {output[0]:F4}");
```

## Building

```bash
# Build the native library
cargo build --release --features julia

# Build the C# project
cd csharp/examples
dotnet build

# Run the example
LD_LIBRARY_PATH=../../target/release dotnet run
```

## Introspection

```csharp
// Layer info
var info = mlp.GetLayerInfo(1);
Console.WriteLine($"Layer {info.Index}: {info.Size} neurons, {info.Activation}");

// Neuron view
var neuron = mlp.GetNeuronView(1, 0);
Console.WriteLine($"Weights: [{string.Join(", ", neuron.Weights)}]");
Console.WriteLine($"Bias: {neuron.Bias}");

// Adam optimizer state
Console.WriteLine($"Weight M: {mlp.GetWeightM(1, 0, 0)}");
Console.WriteLine($"Weight V: {mlp.GetWeightV(1, 0, 0)}");

// Feature importance
foreach (var fi in mlp.GetFeatureImportance())
    Console.WriteLine($"Feature {fi.Index}: {fi.Score:F4}");

// Histograms
var hist = mlp.GetActivationHistogram(1, 10);
var gradHist = mlp.GetGradientHistogram(1, 10);
```
