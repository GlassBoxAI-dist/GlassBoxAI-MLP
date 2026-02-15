const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    const mlp_mod = b.addModule("glassboxai-mlp", .{
        .root_source_file = b.path("mlp.zig"),
        .target = target,
        .optimize = optimize,
    });

    const xor_example = b.addExecutable(.{
        .name = "xor_example",
        .root_source_file = b.path("examples/xor_example.zig"),
        .target = target,
        .optimize = optimize,
    });

    xor_example.root_module.addImport("glassboxai-mlp", mlp_mod);
    xor_example.addLibraryPath(.{ .cwd_relative = "../target/release" });
    xor_example.linkSystemLibrary("glassboxai_mlp");
    xor_example.linkLibC();

    b.installArtifact(xor_example);

    const run_cmd = b.addRunArtifact(xor_example);
    run_cmd.step.dependOn(b.getInstallStep());

    const run_step = b.step("run", "Run the XOR example");
    run_step.dependOn(&run_cmd.step);
}
