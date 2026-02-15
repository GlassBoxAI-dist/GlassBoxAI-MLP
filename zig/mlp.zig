// MIT License
// Copyright (c) 2025 Matthew Abbott
//
// Zig wrapper for the GlassBoxAI MLP library.

const std = @import("std");
const c = @import("c.zig");

pub const Activation = enum(i32) {
    sigmoid = c.MLP_ACTIVATION_SIGMOID,
    tanh = c.MLP_ACTIVATION_TANH,
    relu = c.MLP_ACTIVATION_RELU,
    softmax = c.MLP_ACTIVATION_SOFTMAX,

    pub fn name(self: Activation) []const u8 {
        return switch (self) {
            .sigmoid => "sigmoid",
            .tanh => "tanh",
            .relu => "relu",
            .softmax => "softmax",
        };
    }
};

pub const Optimizer = enum(i32) {
    sgd = c.MLP_OPTIMIZER_SGD,
    adam = c.MLP_OPTIMIZER_ADAM,
    rmsprop = c.MLP_OPTIMIZER_RMSPROP,

    pub fn name(self: Optimizer) []const u8 {
        return switch (self) {
            .sgd => "SGD",
            .adam => "Adam",
            .rmsprop => "RMSProp",
        };
    }
};

pub const Config = struct {
    hidden_activation: Activation = .sigmoid,
    output_activation: Activation = .sigmoid,
    backend: ?[:0]const u8 = null,
    learning_rate: f64 = 0.01,
    optimizer: Optimizer = .adam,
    dropout_rate: f64 = 0.0,
    l2_lambda: f64 = 0.0,
    batch_norm: bool = false,
};

pub const TrainResult = struct {
    losses: []f64,
    final_loss: f64,
    allocator: std.mem.Allocator,

    pub fn deinit(self: *TrainResult) void {
        self.allocator.free(self.losses);
    }
};

pub const FeatureImportance = struct {
    index: i32,
    score: f64,
};

pub const LayerInfo = struct {
    index: usize,
    size: usize,
    activation: Activation,
    weights_per_neuron: usize,
};

pub const NeuronView = struct {
    layer: usize,
    index: usize,
    weights: []f64,
    bias: f64,
    output: f64,
    @"error": f64,
    allocator: std.mem.Allocator,

    pub fn deinit(self: *NeuronView) void {
        self.allocator.free(self.weights);
    }
};

pub const MlpError = error{
    CreateFailed,
    TrainFailed,
    PredictFailed,
    SaveFailed,
    LoadFailed,
    SetBackendFailed,
    OutOfMemory,
};

pub const MLP = struct {
    handle: c.MlpHandle,
    input_size: usize,
    output_size: usize,
    hidden_sizes: []i32,
    allocator: std.mem.Allocator,

    pub fn init(
        allocator: std.mem.Allocator,
        input_size: usize,
        hidden_sizes: []const i32,
        output_size: usize,
        config: Config,
    ) MlpError!MLP {
        const handle = c.mlp_create(
            @intCast(input_size),
            hidden_sizes.ptr,
            @intCast(hidden_sizes.len),
            @intCast(output_size),
            @intFromEnum(config.hidden_activation),
            @intFromEnum(config.output_activation),
            if (config.backend) |b| b.ptr else null,
        );

        if (handle == null) return MlpError.CreateFailed;

        c.mlp_set_learning_rate(handle, config.learning_rate);
        c.mlp_set_optimizer(handle, @intFromEnum(config.optimizer));
        c.mlp_set_dropout_rate(handle, config.dropout_rate);
        c.mlp_set_l2_lambda(handle, config.l2_lambda);
        c.mlp_set_batch_norm(handle, if (config.batch_norm) 1 else 0);

        const owned = allocator.alloc(i32, hidden_sizes.len) catch return MlpError.OutOfMemory;
        @memcpy(owned, hidden_sizes);

        return MLP{
            .handle = handle,
            .input_size = input_size,
            .output_size = output_size,
            .hidden_sizes = owned,
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *MLP) void {
        if (self.handle != null) {
            c.mlp_destroy(self.handle);
            self.handle = null;
        }
        self.allocator.free(self.hidden_sizes);
    }

    pub fn load(allocator: std.mem.Allocator, filename: [:0]const u8) MlpError!MLP {
        const handle = c.mlp_load(filename.ptr);
        if (handle == null) return MlpError.LoadFailed;

        const input_size: usize = @intCast(c.mlp_get_input_size(handle));
        const output_size: usize = @intCast(c.mlp_get_output_size(handle));

        var sizes_buf: [100]i32 = undefined;
        const count: usize = @intCast(c.mlp_get_hidden_sizes(handle, &sizes_buf, 100));
        const owned = allocator.alloc(i32, count) catch return MlpError.OutOfMemory;
        @memcpy(owned, sizes_buf[0..count]);

        return MLP{
            .handle = handle,
            .input_size = input_size,
            .output_size = output_size,
            .hidden_sizes = owned,
            .allocator = allocator,
        };
    }

    pub fn train(self: *MLP, input: []const f64, target: []const f64) MlpError!void {
        const status = c.mlp_train(
            self.handle,
            input.ptr,
            @intCast(input.len),
            target.ptr,
            @intCast(target.len),
        );
        if (status != 0) return MlpError.TrainFailed;
    }

    pub fn predict(self: *MLP, input: []const f64, output: []f64) MlpError!usize {
        const len = c.mlp_predict(
            self.handle,
            input.ptr,
            @intCast(input.len),
            output.ptr,
            @intCast(output.len),
        );
        if (len < 0) return MlpError.PredictFailed;
        return @intCast(len);
    }

    pub fn predictAlloc(self: *MLP, allocator: std.mem.Allocator, input: []const f64) MlpError![]f64 {
        const output = allocator.alloc(f64, self.output_size) catch return MlpError.OutOfMemory;
        const len = try self.predict(input, output);
        if (len != self.output_size) {
            return output[0..len];
        }
        return output;
    }

    pub fn fit(
        self: *MLP,
        allocator: std.mem.Allocator,
        inputs: []const []const f64,
        targets: []const []const f64,
        epochs: usize,
        verbose: bool,
    ) MlpError!TrainResult {
        const losses = allocator.alloc(f64, epochs) catch return MlpError.OutOfMemory;
        var output_buf = allocator.alloc(f64, self.output_size) catch return MlpError.OutOfMemory;
        defer allocator.free(output_buf);

        const writer = std.io.getStdOut().writer();

        for (0..epochs) |epoch| {
            var epoch_loss: f64 = 0.0;

            for (inputs, targets) |input, target| {
                try self.train(input, target);
                _ = try self.predict(input, output_buf);
                epoch_loss += self.computeLoss(output_buf, target);
            }

            epoch_loss /= @as(f64, @floatFromInt(inputs.len));
            losses[epoch] = epoch_loss;

            if (verbose and (epoch % 100 == 0 or epoch == epochs - 1)) {
                writer.print("Epoch {d}/{d} - Loss: {d:.6}\n", .{ epoch + 1, epochs, epoch_loss }) catch {};
            }
        }

        return TrainResult{
            .losses = losses,
            .final_loss = if (epochs > 0) losses[epochs - 1] else 0.0,
            .allocator = allocator,
        };
    }

    pub fn computeLoss(self: *MLP, output: []const f64, target: []const f64) f64 {
        return c.mlp_compute_loss(
            self.handle,
            output.ptr,
            @intCast(output.len),
            target.ptr,
            @intCast(target.len),
        );
    }

    pub fn save(self: *MLP, filename: [:0]const u8) MlpError!void {
        const status = c.mlp_save(self.handle, filename.ptr);
        if (status != 0) return MlpError.SaveFailed;
    }

    // Properties

    pub fn getLearningRate(self: *const MLP) f64 {
        return c.mlp_get_learning_rate(self.handle);
    }

    pub fn setLearningRate(self: *MLP, value: f64) void {
        c.mlp_set_learning_rate(self.handle, value);
    }

    pub fn getOptimizer(self: *const MLP) Optimizer {
        return @enumFromInt(c.mlp_get_optimizer(self.handle));
    }

    pub fn setOptimizer(self: *MLP, opt: Optimizer) void {
        c.mlp_set_optimizer(self.handle, @intFromEnum(opt));
    }

    pub fn getDropoutRate(self: *const MLP) f64 {
        return c.mlp_get_dropout_rate(self.handle);
    }

    pub fn setDropoutRate(self: *MLP, value: f64) void {
        c.mlp_set_dropout_rate(self.handle, value);
    }

    pub fn getL2Lambda(self: *const MLP) f64 {
        return c.mlp_get_l2_lambda(self.handle);
    }

    pub fn setL2Lambda(self: *MLP, value: f64) void {
        c.mlp_set_l2_lambda(self.handle, value);
    }

    pub fn getBatchNorm(self: *const MLP) bool {
        return c.mlp_get_batch_norm(self.handle) != 0;
    }

    pub fn setBatchNorm(self: *MLP, value: bool) void {
        c.mlp_set_batch_norm(self.handle, if (value) 1 else 0);
    }

    pub fn getNumLayers(self: *const MLP) usize {
        return @intCast(c.mlp_get_num_layers(self.handle));
    }

    pub fn getTimestep(self: *const MLP) i32 {
        return c.mlp_get_timestep(self.handle);
    }

    pub fn getBackend(self: *const MLP) []const u8 {
        const ptr = c.mlp_get_backend(self.handle);
        if (ptr) |p| {
            return std.mem.span(p);
        }
        return "unknown";
    }

    pub fn setBackend(self: *MLP, backend: [:0]const u8) MlpError!void {
        const status = c.mlp_set_backend(self.handle, backend.ptr);
        if (status != 0) return MlpError.SetBackendFailed;
    }

    // Introspection

    pub fn getLayerInfo(self: *const MLP, layer: usize) LayerInfo {
        const li: i32 = @intCast(layer);
        const size: usize = @intCast(c.mlp_get_layer_size(self.handle, li));
        const activation: Activation = @enumFromInt(c.mlp_get_layer_activation(self.handle, li));
        const weights_per_neuron: usize = if (layer == 0) 0 else @intCast(c.mlp_get_layer_size(self.handle, li - 1));

        return LayerInfo{
            .index = layer,
            .size = size,
            .activation = activation,
            .weights_per_neuron = weights_per_neuron,
        };
    }

    pub fn getNeuronWeights(self: *const MLP, allocator: std.mem.Allocator, layer: usize, neuron: usize) MlpError![]f64 {
        const li: i32 = @intCast(layer);
        const prev_size: usize = if (layer == 0) 0 else @intCast(c.mlp_get_layer_size(self.handle, li - 1));
        const buf = allocator.alloc(f64, prev_size) catch return MlpError.OutOfMemory;
        const len: usize = @intCast(c.mlp_get_neuron_weights(self.handle, li, @intCast(neuron), buf.ptr, @intCast(prev_size)));
        if (len != prev_size) {
            return buf[0..len];
        }
        return buf;
    }

    pub fn getNeuronBias(self: *const MLP, layer: usize, neuron: usize) f64 {
        return c.mlp_get_neuron_bias(self.handle, @intCast(layer), @intCast(neuron));
    }

    pub fn setNeuronWeight(self: *MLP, layer: usize, neuron: usize, weight_idx: usize, value: f64) void {
        c.mlp_set_neuron_weight(self.handle, @intCast(layer), @intCast(neuron), @intCast(weight_idx), value);
    }

    pub fn setNeuronBias(self: *MLP, layer: usize, neuron: usize, value: f64) void {
        c.mlp_set_neuron_bias(self.handle, @intCast(layer), @intCast(neuron), value);
    }

    pub fn getNeuronView(self: *const MLP, allocator: std.mem.Allocator, layer: usize, neuron: usize) MlpError!NeuronView {
        const weights = try self.getNeuronWeights(allocator, layer, neuron);
        const bias = self.getNeuronBias(layer, neuron);

        const outputs = try self.getLayerOutputs(allocator, layer);
        defer allocator.free(outputs);
        const errors = try self.getLayerErrors(allocator, layer);
        defer allocator.free(errors);

        return NeuronView{
            .layer = layer,
            .index = neuron,
            .weights = weights,
            .bias = bias,
            .output = if (neuron < outputs.len) outputs[neuron] else 0.0,
            .@"error" = if (neuron < errors.len) errors[neuron] else 0.0,
            .allocator = allocator,
        };
    }

    pub fn getLayerOutputs(self: *const MLP, allocator: std.mem.Allocator, layer: usize) MlpError![]f64 {
        const li: i32 = @intCast(layer);
        const layer_size: usize = @intCast(c.mlp_get_layer_size(self.handle, li));
        const buf = allocator.alloc(f64, layer_size) catch return MlpError.OutOfMemory;
        const len: usize = @intCast(c.mlp_get_layer_outputs(self.handle, li, buf.ptr, @intCast(layer_size)));
        if (len != layer_size) return buf[0..len];
        return buf;
    }

    pub fn getLayerErrors(self: *const MLP, allocator: std.mem.Allocator, layer: usize) MlpError![]f64 {
        const li: i32 = @intCast(layer);
        const layer_size: usize = @intCast(c.mlp_get_layer_size(self.handle, li));
        const buf = allocator.alloc(f64, layer_size) catch return MlpError.OutOfMemory;
        const len: usize = @intCast(c.mlp_get_layer_errors(self.handle, li, buf.ptr, @intCast(layer_size)));
        if (len != layer_size) return buf[0..len];
        return buf;
    }

    pub fn getFeatureImportance(self: *const MLP, allocator: std.mem.Allocator) MlpError![]FeatureImportance {
        const indices = allocator.alloc(i32, self.input_size) catch return MlpError.OutOfMemory;
        defer allocator.free(indices);
        const scores = allocator.alloc(f64, self.input_size) catch return MlpError.OutOfMemory;
        defer allocator.free(scores);

        const len: usize = @intCast(c.mlp_feature_importance(
            self.handle,
            indices.ptr,
            scores.ptr,
            @intCast(self.input_size),
        ));

        const result = allocator.alloc(FeatureImportance, len) catch return MlpError.OutOfMemory;
        for (0..len) |i| {
            result[i] = FeatureImportance{ .index = indices[i], .score = scores[i] };
        }
        return result;
    }

    pub fn getWeightM(self: *const MLP, layer: usize, neuron: usize, weight_idx: usize) f64 {
        return c.mlp_get_weight_m(self.handle, @intCast(layer), @intCast(neuron), @intCast(weight_idx));
    }

    pub fn getWeightV(self: *const MLP, layer: usize, neuron: usize, weight_idx: usize) f64 {
        return c.mlp_get_weight_v(self.handle, @intCast(layer), @intCast(neuron), @intCast(weight_idx));
    }

    pub fn getBiasM(self: *const MLP, layer: usize, neuron: usize) f64 {
        return c.mlp_get_bias_m(self.handle, @intCast(layer), @intCast(neuron));
    }

    pub fn getBiasV(self: *const MLP, layer: usize, neuron: usize) f64 {
        return c.mlp_get_bias_v(self.handle, @intCast(layer), @intCast(neuron));
    }

    pub fn getActivationHistogram(self: *const MLP, allocator: std.mem.Allocator, layer: usize, bins: usize) MlpError![]i32 {
        const buf = allocator.alloc(i32, bins) catch return MlpError.OutOfMemory;
        const len: usize = @intCast(c.mlp_get_activation_histogram(
            self.handle,
            @intCast(layer),
            @intCast(bins),
            buf.ptr,
            @intCast(bins),
        ));
        if (len != bins) return buf[0..len];
        return buf;
    }

    pub fn getGradientHistogram(self: *const MLP, allocator: std.mem.Allocator, layer: usize, bins: usize) MlpError![]i32 {
        const buf = allocator.alloc(i32, bins) catch return MlpError.OutOfMemory;
        const len: usize = @intCast(c.mlp_get_gradient_histogram(
            self.handle,
            @intCast(layer),
            @intCast(bins),
            buf.ptr,
            @intCast(bins),
        ));
        if (len != bins) return buf[0..len];
        return buf;
    }

    // Utility

    pub fn getLastError() []const u8 {
        const ptr = c.mlp_get_last_error();
        if (ptr) |p| {
            return std.mem.span(p);
        }
        return "unknown error";
    }
};

pub fn availableBackends(allocator: std.mem.Allocator) MlpError![][]const u8 {
    const ptr = c.mlp_available_backends();
    if (ptr == null) {
        const result = allocator.alloc([]const u8, 1) catch return MlpError.OutOfMemory;
        result[0] = "cpu";
        return result;
    }

    const str = std.mem.span(ptr.?);
    defer c.mlp_free_string(ptr);

    var count: usize = 1;
    for (str) |ch| {
        if (ch == ',') count += 1;
    }

    const result = allocator.alloc([]const u8, count) catch return MlpError.OutOfMemory;
    var it = std.mem.splitScalar(u8, str, ',');
    var i: usize = 0;
    while (it.next()) |part| : (i += 1) {
        result[i] = part;
    }

    return result[0..i];
}
