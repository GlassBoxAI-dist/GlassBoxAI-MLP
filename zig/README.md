# GlassBoxAI MLP - Zig Wrapper

Zig bindings for the GlassBoxAI MLP library. Provides a GPU-accelerated Multi-Layer
Perceptron with full introspection for explainable AI.

## Prerequisites

- Zig 0.13+ 
- Built native library (`cargo build --release --features julia`)

## Quick Start

```zig
const std = @import("std");
const mlp_mod = @import("glassboxai-mlp");
const MLP = mlp_mod.MLP;

pub fn main() !void {
    const allocator = std.heap.page_allocator;

    const hidden = [_]i32{8};
    var net = try MLP.init(allocator, 2, &hidden, 1, .{
        .learning_rate = 0.5,
        .optimizer = .adam,
    });
    defer net.deinit();

    // Train on XOR
    const inputs = [_][]const f64{
        &[_]f64{ 0.0, 0.0 }, &[_]f64{ 0.0, 1.0 },
        &[_]f64{ 1.0, 0.0 }, &[_]f64{ 1.0, 1.0 },
    };
    const targets = [_][]const f64{
        &[_]f64{0.0}, &[_]f64{1.0},
        &[_]f64{1.0}, &[_]f64{0.0},
    };

    var result = try net.fit(allocator, &inputs, &targets, 1000, true);
    defer result.deinit();

    // Predict
    var output: [1]f64 = undefined;
    _ = try net.predict(&[_]f64{ 1.0, 0.0 }, &output);
    std.debug.print("Output: {d:.4}\n", .{output[0]});
}
```

## Building

```bash
# Build the native library
cargo build --release --features julia

# Build the Zig example
cd zig
zig build

# Run the example
LD_LIBRARY_PATH=../target/release zig-out/bin/xor_example
```

## Introspection

```zig
// Layer info
const info = net.getLayerInfo(1);

// Neuron weights
const weights = try net.getNeuronWeights(allocator, 1, 0);
defer allocator.free(weights);

// Adam optimizer state
const m = net.getWeightM(1, 0, 0);
const v = net.getWeightV(1, 0, 0);

// Feature importance
const importance = try net.getFeatureImportance(allocator);
defer allocator.free(importance);

// Histograms
const hist = try net.getActivationHistogram(allocator, 1, 10);
defer allocator.free(hist);
```
