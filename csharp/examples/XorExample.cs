/*
 * XOR Example - Demonstrates basic MLP usage in C#
 *
 * Build:
 *   dotnet build
 *
 * Run:
 *   LD_LIBRARY_PATH=../../target/release dotnet run
 */

using System;
using GlassBoxAI.MLP;

class XorExample
{
    static void Main()
    {
        Console.WriteLine("=== XOR Example (C#) ===");
        Console.WriteLine("Available backends: " + string.Join(", ", MLP.AvailableBackends()));
        Console.WriteLine();

        var config = new MLPConfig
        {
            HiddenActivation = ActivationType.Sigmoid,
            OutputActivation = ActivationType.Sigmoid,
            LearningRate = 0.5,
            Optimizer = OptimizerType.Adam,
            Backend = "auto"
        };

        using var mlp = new MLP(2, new[] { 8 }, 1, config);
        Console.WriteLine($"Created: {mlp}");
        Console.WriteLine();

        var inputs = new[]
        {
            new[] { 0.0, 0.0 },
            new[] { 0.0, 1.0 },
            new[] { 1.0, 0.0 },
            new[] { 1.0, 1.0 }
        };

        var targets = new[]
        {
            new[] { 0.0 },
            new[] { 1.0 },
            new[] { 1.0 },
            new[] { 0.0 }
        };

        Console.WriteLine("Training...");
        var result = mlp.Fit(inputs, targets, 1000, verbose: true);
        Console.WriteLine();

        Console.WriteLine("Predictions:");
        for (int i = 0; i < inputs.Length; i++)
        {
            var output = mlp.Predict(inputs[i]);
            Console.WriteLine($"  [{inputs[i][0]}, {inputs[i][1]}] => {output[0]:F4} (expected: {targets[i][0]})");
        }
        Console.WriteLine();

        Console.WriteLine("Saving model to xor_model.json...");
        mlp.Save("xor_model.json");

        Console.WriteLine("Loading model...");
        using var mlp2 = MLP.Load("xor_model.json");
        Console.WriteLine($"Loaded: {mlp2}");

        Console.WriteLine("\nVerifying loaded model:");
        for (int i = 0; i < inputs.Length; i++)
        {
            var output = mlp2.Predict(inputs[i]);
            Console.WriteLine($"  [{inputs[i][0]}, {inputs[i][1]}] => {output[0]:F4}");
        }

        Console.WriteLine("\nFeature importance:");
        var importance = mlp.GetFeatureImportance();
        foreach (var fi in importance)
            Console.WriteLine($"  Feature {fi.Index}: {fi.Score:F4}");

        Console.WriteLine("\nLayer info:");
        for (int l = 0; l < mlp.NumLayers; l++)
        {
            var info = mlp.GetLayerInfo(l);
            Console.WriteLine($"  Layer {info.Index}: {info.Size} neurons, {info.Activation}");
        }

        Console.WriteLine("\nAdam optimizer state (layer 1, neuron 0, weight 0):");
        Console.WriteLine($"  M = {mlp.GetWeightM(1, 0, 0):F6}");
        Console.WriteLine($"  V = {mlp.GetWeightV(1, 0, 0):F6}");
        Console.WriteLine($"  Timestep = {mlp.Timestep}");

        Console.WriteLine("\nDone!");
    }
}
