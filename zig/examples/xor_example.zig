/**
 * @file
 * @ingroup MLP_Wrappers
 */
// XOR Example - Demonstrates basic MLP usage in Zig
//
// Build:
//   zig build-exe examples/xor_example.zig -I../cpp/include -L../target/release -lglassboxai_mlp -lc
//
// Run:
//   LD_LIBRARY_PATH=../target/release ./xor_example

const std = @import("std");
const mlp_mod = @import("../mlp.zig");
const MLP = mlp_mod.MLP;
const Config = mlp_mod.Config;

pub fn main() !void {
    const allocator = std.heap.page_allocator;
    const writer = std.io.getStdOut().writer();

    try writer.print("=== XOR Example (Zig) ===\n", .{});

    const backends = try mlp_mod.availableBackends(allocator);
    defer allocator.free(backends);
    try writer.print("Available backends:", .{});
    for (backends) |b| {
        try writer.print(" {s}", .{b});
    }
    try writer.print("\n\n", .{});

    const hidden = [_]i32{8};
    var net = try MLP.init(allocator, 2, &hidden, 1, Config{
        .hidden_activation = .sigmoid,
        .output_activation = .sigmoid,
        .learning_rate = 0.5,
        .optimizer = .adam,
    });
    defer net.deinit();

    try writer.print("Created: MLP(input={d}, hidden={d}, output={d}, lr={d:.4}, optimizer={s}, backend={s})\n\n", .{
        net.input_size,
        net.hidden_sizes.len,
        net.output_size,
        net.getLearningRate(),
        net.getOptimizer().name(),
        net.getBackend(),
    });

    const inputs = [_][]const f64{
        &[_]f64{ 0.0, 0.0 },
        &[_]f64{ 0.0, 1.0 },
        &[_]f64{ 1.0, 0.0 },
        &[_]f64{ 1.0, 1.0 },
    };
    const targets = [_][]const f64{
        &[_]f64{0.0},
        &[_]f64{1.0},
        &[_]f64{1.0},
        &[_]f64{0.0},
    };

    try writer.print("Training...\n", .{});
    var result = try net.fit(allocator, &inputs, &targets, 1000, true);
    defer result.deinit();
    try writer.print("\n", .{});

    try writer.print("Predictions:\n", .{});
    var output_buf: [1]f64 = undefined;
    for (inputs, targets) |input, target| {
        _ = try net.predict(input, &output_buf);
        try writer.print("  [{d:.1}, {d:.1}] => {d:.4} (expected: {d:.1})\n", .{
            input[0], input[1], output_buf[0], target[0],
        });
    }
    try writer.print("\n", .{});

    try writer.print("Saving model to xor_model.json...\n", .{});
    try net.save("xor_model.json");

    try writer.print("Loading model...\n", .{});
    var net2 = try MLP.load(allocator, "xor_model.json");
    defer net2.deinit();

    try writer.print("\nVerifying loaded model:\n", .{});
    for (inputs) |input| {
        _ = try net2.predict(input, &output_buf);
        try writer.print("  [{d:.1}, {d:.1}] => {d:.4}\n", .{ input[0], input[1], output_buf[0] });
    }

    try writer.print("\nFeature importance:\n", .{});
    const importance = try net.getFeatureImportance(allocator);
    defer allocator.free(importance);
    for (importance) |fi| {
        try writer.print("  Feature {d}: {d:.4}\n", .{ fi.index, fi.score });
    }

    try writer.print("\nLayer info:\n", .{});
    const num_layers = net.getNumLayers();
    for (0..num_layers) |l| {
        const info = net.getLayerInfo(l);
        try writer.print("  Layer {d}: {d} neurons, {s}\n", .{ info.index, info.size, info.activation.name() });
    }

    try writer.print("\nAdam optimizer state (layer 1, neuron 0, weight 0):\n", .{});
    try writer.print("  M = {d:.6}\n", .{net.getWeightM(1, 0, 0)});
    try writer.print("  V = {d:.6}\n", .{net.getWeightV(1, 0, 0)});
    try writer.print("  Timestep = {d}\n", .{net.getTimestep()});

    try writer.print("\nDone!\n", .{});
}

