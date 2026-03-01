/**
 * @file
 * @ingroup MLP_Wrappers
 */
/*
 * MIT License
 * Copyright (c) 2025 Matthew Abbott
 *
 * P/Invoke declarations for the GlassBoxAI MLP native library.
 */

using System;
using System.Runtime.InteropServices;

namespace GlassBoxAI.MLP
{
    internal static class NativeMethods
    {
        private const string LibName = "glassboxai_mlp";

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        public static extern IntPtr mlp_get_last_error();

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        public static extern void mlp_free_error(IntPtr ptr);

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        public static extern IntPtr mlp_create(
            int inputSize,
            [In] int[] hiddenSizes,
            int hiddenCount,
            int outputSize,
            int hiddenActivation,
            int outputActivation,
            [MarshalAs(UnmanagedType.LPStr)] string gpuBackend
        );

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        public static extern void mlp_destroy(IntPtr mlp);

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        public static extern int mlp_train(
            IntPtr mlp,
            [In] double[] input, int inputLen,
            [In] double[] target, int targetLen
        );

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        public static extern int mlp_predict(
            IntPtr mlp,
            [In] double[] input, int inputLen,
            [Out] double[] output, int outputCapacity
        );

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        public static extern double mlp_compute_loss(
            IntPtr mlp,
            [In] double[] output, int outputLen,
            [In] double[] target, int targetLen
        );

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        public static extern int mlp_save(
            IntPtr mlp,
            [MarshalAs(UnmanagedType.LPStr)] string filename
        );

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        public static extern IntPtr mlp_load(
            [MarshalAs(UnmanagedType.LPStr)] string filename
        );

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        public static extern double mlp_get_learning_rate(IntPtr mlp);

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        public static extern void mlp_set_learning_rate(IntPtr mlp, double value);

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        public static extern int mlp_get_optimizer(IntPtr mlp);

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        public static extern void mlp_set_optimizer(IntPtr mlp, int value);

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        public static extern double mlp_get_dropout_rate(IntPtr mlp);

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        public static extern void mlp_set_dropout_rate(IntPtr mlp, double value);

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        public static extern double mlp_get_l2_lambda(IntPtr mlp);

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        public static extern void mlp_set_l2_lambda(IntPtr mlp, double value);

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        public static extern int mlp_get_batch_norm(IntPtr mlp);

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        public static extern void mlp_set_batch_norm(IntPtr mlp, int value);

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        public static extern int mlp_get_input_size(IntPtr mlp);

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        public static extern int mlp_get_output_size(IntPtr mlp);

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        public static extern int mlp_get_num_layers(IntPtr mlp);

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        public static extern int mlp_get_hidden_sizes(IntPtr mlp, [Out] int[] output, int capacity);

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        public static extern IntPtr mlp_get_backend(IntPtr mlp);

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        public static extern int mlp_set_backend(
            IntPtr mlp,
            [MarshalAs(UnmanagedType.LPStr)] string backend
        );

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        public static extern IntPtr mlp_available_backends();

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        public static extern void mlp_free_string(IntPtr s);

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        public static extern int mlp_get_neuron_weights(
            IntPtr mlp, int layer, int neuron,
            [Out] double[] output, int capacity
        );

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        public static extern double mlp_get_neuron_bias(IntPtr mlp, int layer, int neuron);

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        public static extern void mlp_set_neuron_weight(
            IntPtr mlp, int layer, int neuron, int weightIdx, double value
        );

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        public static extern void mlp_set_neuron_bias(IntPtr mlp, int layer, int neuron, double value);

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        public static extern int mlp_get_layer_outputs(
            IntPtr mlp, int layer, [Out] double[] output, int capacity
        );

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        public static extern int mlp_get_layer_errors(
            IntPtr mlp, int layer, [Out] double[] output, int capacity
        );

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        public static extern int mlp_get_layer_size(IntPtr mlp, int layer);

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        public static extern int mlp_get_layer_activation(IntPtr mlp, int layer);

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        public static extern int mlp_feature_importance(
            IntPtr mlp, [Out] int[] indices, [Out] double[] scores, int capacity
        );

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        public static extern double mlp_get_weight_m(IntPtr mlp, int layer, int neuron, int weightIdx);

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        public static extern double mlp_get_weight_v(IntPtr mlp, int layer, int neuron, int weightIdx);

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        public static extern double mlp_get_bias_m(IntPtr mlp, int layer, int neuron);

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        public static extern double mlp_get_bias_v(IntPtr mlp, int layer, int neuron);

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        public static extern int mlp_get_activation_histogram(
            IntPtr mlp, int layer, int bins, [Out] int[] output, int capacity
        );

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        public static extern int mlp_get_gradient_histogram(
            IntPtr mlp, int layer, int bins, [Out] int[] output, int capacity
        );

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        public static extern int mlp_get_timestep(IntPtr mlp);

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        public static extern void mlp_set_weight_m(IntPtr mlp, int layer, int neuron, int weightIdx, double value);

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        public static extern void mlp_set_weight_v(IntPtr mlp, int layer, int neuron, int weightIdx, double value);

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        public static extern void mlp_set_bias_m(IntPtr mlp, int layer, int neuron, double value);

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        public static extern void mlp_set_bias_v(IntPtr mlp, int layer, int neuron, double value);

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        public static extern void mlp_set_timestep(IntPtr mlp, int value);

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        public static extern void mlp_set_layer_activation(IntPtr mlp, int layer, int activation);

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        public static extern int mlp_set_neuron_weights(
            IntPtr mlp, int layer, int neuron, [In] double[] weights, int weightsLen
        );
    }
}
