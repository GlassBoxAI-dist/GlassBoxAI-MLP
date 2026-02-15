/*
 * MIT License
 * Copyright (c) 2025 Matthew Abbott
 *
 * C# wrapper for the GlassBoxAI MLP library.
 */

using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;

namespace GlassBoxAI.MLP
{
    public enum ActivationType
    {
        Sigmoid = 0,
        Tanh = 1,
        ReLU = 2,
        Softmax = 3
    }

    public enum OptimizerType
    {
        SGD = 0,
        Adam = 1,
        RMSProp = 2
    }

    public struct TrainResult
    {
        public List<double> Losses;
        public double FinalLoss;
    }

    public struct FeatureImportance
    {
        public int Index;
        public double Score;
    }

    public struct LayerInfo
    {
        public int Index;
        public int Size;
        public ActivationType Activation;
        public int WeightsPerNeuron;
    }

    public struct NeuronView
    {
        public int Layer;
        public int Index;
        public double[] Weights;
        public double Bias;
        public double Output;
        public double Error;
    }

    public class MLPConfig
    {
        public ActivationType HiddenActivation { get; set; } = ActivationType.Sigmoid;
        public ActivationType OutputActivation { get; set; } = ActivationType.Sigmoid;
        public string Backend { get; set; } = "auto";
        public double LearningRate { get; set; } = 0.01;
        public OptimizerType Optimizer { get; set; } = OptimizerType.Adam;
        public double DropoutRate { get; set; } = 0.0;
        public double L2Lambda { get; set; } = 0.0;
        public bool BatchNorm { get; set; } = false;
    }

    public class MLPException : Exception
    {
        public MLPException(string message) : base(message) { }
    }

    public class MLP : IDisposable
    {
        private IntPtr _handle;
        private readonly int _inputSize;
        private readonly int _outputSize;
        private int[] _hiddenSizes;
        private bool _disposed;

        public MLP(int inputSize, int[] hiddenSizes, int outputSize, MLPConfig config = null)
        {
            config ??= new MLPConfig();
            _inputSize = inputSize;
            _outputSize = outputSize;
            _hiddenSizes = (int[])hiddenSizes.Clone();

            _handle = NativeMethods.mlp_create(
                inputSize,
                hiddenSizes,
                hiddenSizes.Length,
                outputSize,
                (int)config.HiddenActivation,
                (int)config.OutputActivation,
                config.Backend
            );

            if (_handle == IntPtr.Zero)
                throw new MLPException(GetLastError());

            LearningRate = config.LearningRate;
            Optimizer = config.Optimizer;
            DropoutRate = config.DropoutRate;
            L2Lambda = config.L2Lambda;
            BatchNorm = config.BatchNorm;
        }

        private MLP(IntPtr handle)
        {
            _handle = handle;
            _inputSize = NativeMethods.mlp_get_input_size(handle);
            _outputSize = NativeMethods.mlp_get_output_size(handle);

            var sizes = new int[100];
            int count = NativeMethods.mlp_get_hidden_sizes(handle, sizes, 100);
            _hiddenSizes = new int[count];
            Array.Copy(sizes, _hiddenSizes, count);
        }

        public static MLP Load(string filename)
        {
            IntPtr handle = NativeMethods.mlp_load(filename);
            if (handle == IntPtr.Zero)
                throw new MLPException(GetLastError());
            return new MLP(handle);
        }

        public void Train(double[] input, double[] target)
        {
            int status = NativeMethods.mlp_train(_handle, input, input.Length, target, target.Length);
            if (status != 0)
                throw new MLPException(GetLastError());
        }

        public double[] Predict(double[] input)
        {
            var output = new double[_outputSize];
            int len = NativeMethods.mlp_predict(_handle, input, input.Length, output, _outputSize);
            if (len < 0)
                throw new MLPException(GetLastError());
            if (len != output.Length)
                Array.Resize(ref output, len);
            return output;
        }

        public TrainResult Fit(double[][] inputs, double[][] targets, int epochs = 100, bool verbose = false)
        {
            if (inputs.Length != targets.Length)
                throw new MLPException("inputs and targets must have same length");

            var result = new TrainResult { Losses = new List<double>(epochs) };

            for (int epoch = 0; epoch < epochs; epoch++)
            {
                double epochLoss = 0.0;

                for (int i = 0; i < inputs.Length; i++)
                {
                    Train(inputs[i], targets[i]);
                    var output = Predict(inputs[i]);
                    epochLoss += ComputeLoss(output, targets[i]);
                }

                epochLoss /= inputs.Length;
                result.Losses.Add(epochLoss);

                if (verbose && (epoch % 100 == 0 || epoch == epochs - 1))
                    Console.WriteLine($"Epoch {epoch + 1}/{epochs} - Loss: {epochLoss:F6}");
            }

            result.FinalLoss = result.Losses.Count > 0 ? result.Losses[^1] : 0.0;
            return result;
        }

        public double[][] PredictBatch(double[][] inputs)
        {
            var outputs = new double[inputs.Length][];
            for (int i = 0; i < inputs.Length; i++)
                outputs[i] = Predict(inputs[i]);
            return outputs;
        }

        public double ComputeLoss(double[] output, double[] target)
        {
            return NativeMethods.mlp_compute_loss(_handle, output, output.Length, target, target.Length);
        }

        public void Save(string filename)
        {
            int status = NativeMethods.mlp_save(_handle, filename);
            if (status != 0)
                throw new MLPException(GetLastError());
        }

        // Properties

        public int InputSize => _inputSize;
        public int OutputSize => _outputSize;
        public int[] HiddenSizes => (int[])_hiddenSizes.Clone();
        public int NumLayers => NativeMethods.mlp_get_num_layers(_handle);
        public int Timestep => NativeMethods.mlp_get_timestep(_handle);

        public double LearningRate
        {
            get => NativeMethods.mlp_get_learning_rate(_handle);
            set => NativeMethods.mlp_set_learning_rate(_handle, value);
        }

        public OptimizerType Optimizer
        {
            get => (OptimizerType)NativeMethods.mlp_get_optimizer(_handle);
            set => NativeMethods.mlp_set_optimizer(_handle, (int)value);
        }

        public double DropoutRate
        {
            get => NativeMethods.mlp_get_dropout_rate(_handle);
            set => NativeMethods.mlp_set_dropout_rate(_handle, value);
        }

        public double L2Lambda
        {
            get => NativeMethods.mlp_get_l2_lambda(_handle);
            set => NativeMethods.mlp_set_l2_lambda(_handle, value);
        }

        public bool BatchNorm
        {
            get => NativeMethods.mlp_get_batch_norm(_handle) != 0;
            set => NativeMethods.mlp_set_batch_norm(_handle, value ? 1 : 0);
        }

        public string Backend
        {
            get
            {
                IntPtr ptr = NativeMethods.mlp_get_backend(_handle);
                return ptr == IntPtr.Zero ? "unknown" : Marshal.PtrToStringAnsi(ptr);
            }
            set
            {
                int status = NativeMethods.mlp_set_backend(_handle, value);
                if (status != 0)
                    throw new MLPException(GetLastError());
            }
        }

        // Introspection

        public LayerInfo GetLayerInfo(int layer)
        {
            int size = NativeMethods.mlp_get_layer_size(_handle, layer);
            var activation = (ActivationType)NativeMethods.mlp_get_layer_activation(_handle, layer);
            int weightsPerNeuron = layer == 0 ? 0 : NativeMethods.mlp_get_layer_size(_handle, layer - 1);

            return new LayerInfo
            {
                Index = layer,
                Size = size,
                Activation = activation,
                WeightsPerNeuron = weightsPerNeuron
            };
        }

        public double[] GetNeuronWeights(int layer, int neuron)
        {
            int prevSize = layer == 0 ? 0 : NativeMethods.mlp_get_layer_size(_handle, layer - 1);
            var weights = new double[prevSize];
            int len = NativeMethods.mlp_get_neuron_weights(_handle, layer, neuron, weights, prevSize);
            if (len != weights.Length)
                Array.Resize(ref weights, len);
            return weights;
        }

        public double GetNeuronBias(int layer, int neuron)
        {
            return NativeMethods.mlp_get_neuron_bias(_handle, layer, neuron);
        }

        public void SetNeuronWeight(int layer, int neuron, int weightIdx, double value)
        {
            NativeMethods.mlp_set_neuron_weight(_handle, layer, neuron, weightIdx, value);
        }

        public void SetNeuronBias(int layer, int neuron, double value)
        {
            NativeMethods.mlp_set_neuron_bias(_handle, layer, neuron, value);
        }

        public NeuronView GetNeuronView(int layer, int neuron)
        {
            var weights = GetNeuronWeights(layer, neuron);
            double bias = GetNeuronBias(layer, neuron);
            var outputs = GetLayerOutputs(layer);
            var errors = GetLayerErrors(layer);

            return new NeuronView
            {
                Layer = layer,
                Index = neuron,
                Weights = weights,
                Bias = bias,
                Output = neuron < outputs.Length ? outputs[neuron] : 0.0,
                Error = neuron < errors.Length ? errors[neuron] : 0.0
            };
        }

        public double[] GetLayerOutputs(int layer)
        {
            int layerSize = NativeMethods.mlp_get_layer_size(_handle, layer);
            var outputs = new double[layerSize];
            int len = NativeMethods.mlp_get_layer_outputs(_handle, layer, outputs, layerSize);
            if (len != outputs.Length)
                Array.Resize(ref outputs, len);
            return outputs;
        }

        public double[] GetLayerErrors(int layer)
        {
            int layerSize = NativeMethods.mlp_get_layer_size(_handle, layer);
            var errors = new double[layerSize];
            int len = NativeMethods.mlp_get_layer_errors(_handle, layer, errors, layerSize);
            if (len != errors.Length)
                Array.Resize(ref errors, len);
            return errors;
        }

        public FeatureImportance[] GetFeatureImportance()
        {
            var indices = new int[_inputSize];
            var scores = new double[_inputSize];
            int len = NativeMethods.mlp_feature_importance(_handle, indices, scores, _inputSize);

            var result = new FeatureImportance[len];
            for (int i = 0; i < len; i++)
                result[i] = new FeatureImportance { Index = indices[i], Score = scores[i] };
            return result;
        }

        public double GetWeightM(int layer, int neuron, int weightIdx)
        {
            return NativeMethods.mlp_get_weight_m(_handle, layer, neuron, weightIdx);
        }

        public double GetWeightV(int layer, int neuron, int weightIdx)
        {
            return NativeMethods.mlp_get_weight_v(_handle, layer, neuron, weightIdx);
        }

        public double GetBiasM(int layer, int neuron)
        {
            return NativeMethods.mlp_get_bias_m(_handle, layer, neuron);
        }

        public double GetBiasV(int layer, int neuron)
        {
            return NativeMethods.mlp_get_bias_v(_handle, layer, neuron);
        }

        public int[] GetActivationHistogram(int layer, int bins)
        {
            var hist = new int[bins];
            int len = NativeMethods.mlp_get_activation_histogram(_handle, layer, bins, hist, bins);
            if (len != hist.Length)
                Array.Resize(ref hist, len);
            return hist;
        }

        public int[] GetGradientHistogram(int layer, int bins)
        {
            var hist = new int[bins];
            int len = NativeMethods.mlp_get_gradient_histogram(_handle, layer, bins, hist, bins);
            if (len != hist.Length)
                Array.Resize(ref hist, len);
            return hist;
        }

        // Utility

        public static string[] AvailableBackends()
        {
            IntPtr ptr = NativeMethods.mlp_available_backends();
            if (ptr == IntPtr.Zero)
                return new[] { "cpu" };

            string result = Marshal.PtrToStringAnsi(ptr);
            NativeMethods.mlp_free_string(ptr);
            return result.Split(',');
        }

        public override string ToString()
        {
            return $"MLP(input={_inputSize}, hidden=[{string.Join(", ", _hiddenSizes)}], " +
                   $"output={_outputSize}, lr={LearningRate:F4}, optimizer={Optimizer}, backend={Backend})";
        }

        // IDisposable

        public void Dispose()
        {
            Dispose(true);
            GC.SuppressFinalize(this);
        }

        protected virtual void Dispose(bool disposing)
        {
            if (!_disposed && _handle != IntPtr.Zero)
            {
                NativeMethods.mlp_destroy(_handle);
                _handle = IntPtr.Zero;
                _disposed = true;
            }
        }

        ~MLP()
        {
            Dispose(false);
        }

        private static string GetLastError()
        {
            IntPtr ptr = NativeMethods.mlp_get_last_error();
            return ptr == IntPtr.Zero ? "Unknown error" : Marshal.PtrToStringAnsi(ptr);
        }
    }
}
