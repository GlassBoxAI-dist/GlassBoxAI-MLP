/**
 * @file
 * @ingroup MLP_Internal_Logic
 */
/**
 * Facaded MLP CUDA/OpenCL - GPU-accelerated Multi-Layer Perceptron
 */

/** Activation function types */
export const enum JsActivationType {
  Sigmoid = 0,
  Tanh = 1,
  ReLU = 2,
  Softmax = 3,
}

/** Optimizer types */
export const enum JsOptimizerType {
  SGD = 0,
  Adam = 1,
  RMSProp = 2,
}

/** Training result containing loss history */
export interface TrainResult {
  losses: number[];
  finalLoss: number;
}

/** Feature importance result */
export interface FeatureImportance {
  featureIndex: number;
  score: number;
}

/** Model information */
export interface ModelInfo {
  inputSize: number;
  outputSize: number;
  hiddenSizes: number[];
  numLayers: number;
  learningRate: number;
  optimizer: string;
  gpuBackend: string;
}

/** MLP creation options */
export interface MlpOptions {
  hiddenActivation?: JsActivationType;
  outputActivation?: JsActivationType;
  gpuBackend?: 'auto' | 'cuda' | 'opencl' | 'cpu';
  learningRate?: number;
  optimizer?: JsOptimizerType;
  dropoutRate?: number;
  l2Lambda?: number;
  batchNorm?: boolean;
}

/** Multi-Layer Perceptron with GPU acceleration */
export class MLP {
  /**
   * Create a new MLP model
   * @param inputSize - Number of input neurons
   * @param hiddenSizes - Array of hidden layer sizes
   * @param outputSize - Number of output neurons
   * @param options - Optional configuration
   */
  constructor(
    inputSize: number,
    hiddenSizes: number[],
    outputSize: number,
    options?: MlpOptions
  );

  /**
   * Train on a single sample
   * @param input - Input values
   * @param target - Target output values
   */
  train(input: number[], target: number[]): void;

  /**
   * Make a prediction
   * @param input - Input values
   * @returns Model output
   */
  predict(input: number[]): number[];

  /**
   * Train on a dataset for multiple epochs
   * @param inputs - Array of input samples
   * @param targets - Array of target outputs
   * @param epochs - Number of training epochs (default: 100)
   * @param verbose - Print progress (default: false)
   * @returns Training result with loss history
   */
  fit(
    inputs: number[][],
    targets: number[][],
    epochs?: number,
    verbose?: boolean
  ): TrainResult;

  /**
   * Predict on multiple samples
   * @param inputs - Array of input samples
   * @returns Array of predictions
   */
  predictBatch(inputs: number[][]): number[][];

  /**
   * Save model to file
   * @param filename - Path to save the model
   */
  save(filename: string): void;

  /**
   * Load model from file
   * @param filename - Path to the model file
   * @returns Loaded model
   */
  static load(filename: string): MLP;

  /**
   * Export model to ONNX format
   * @param filename - Path to save ONNX file
   */
  exportOnnx(filename: string): void;

  /**
   * Import model from ONNX format
   * @param filename - Path to ONNX file
   * @returns Imported model
   */
  static importOnnx(filename: string): MLP;

  /**
   * Compute feature importance
   * @returns Array of feature importance scores
   */
  featureImportance(): FeatureImportance[];

  /** Learning rate */
  learningRate: number;

  /** Optimizer type */
  optimizer: JsOptimizerType;

  /** Dropout rate */
  dropoutRate: number;

  /** L2 regularization lambda */
  l2Lambda: number;

  /** Batch normalization flag */
  batchNorm: boolean;

  /** Current GPU backend (read-only) */
  readonly gpuBackend: string;

  /** Input layer size (read-only) */
  readonly inputSize: number;

  /** Output layer size (read-only) */
  readonly outputSize: number;

  /** Hidden layer sizes (read-only) */
  readonly hiddenSizes: number[];

  /** Total layer count (read-only) */
  readonly numLayers: number;

  /**
   * Set GPU backend
   * @param backend - "cpu", "cuda", or "opencl"
   */
  setBackend(backend: string): void;

  /**
   * Get list of available GPU backends
   * @returns Available backends
   */
  static availableBackends(): string[];

  /**
   * Get weights for a specific neuron
   * @param layer - Layer index
   * @param neuron - Neuron index
   * @returns Weights
   */
  getNeuronWeights(layer: number, neuron: number): number[];

  /**
   * Get bias for a specific neuron
   * @param layer - Layer index
   * @param neuron - Neuron index
   * @returns Bias value
   */
  getNeuronBias(layer: number, neuron: number): number;

  /**
   * Set a specific weight
   * @param layer - Layer index
   * @param neuron - Neuron index
   * @param weightIdx - Weight index
   * @param value - New weight value
   */
  setNeuronWeight(layer: number, neuron: number, weightIdx: number, value: number): void;

  /**
   * Set a neuron's bias
   * @param layer - Layer index
   * @param neuron - Neuron index
   * @param value - New bias value
   */
  setNeuronBias(layer: number, neuron: number, value: number): void;

  /**
   * Get layer outputs
   * @param layer - Layer index
   * @returns Output values for all neurons in the layer
   */
  getLayerOutputs(layer: number): number[];

  /**
   * Get model info
   * @returns Model information object
   */
  info(): ModelInfo;

  /**
   * String representation
   * @returns String describing the model
   */
  toString(): string;
}

/** CSV data result */
export interface CsvData {
  inputs: number[][];
  targets: number[][];
}

/**
 * Load dataset from CSV file
 * @param filename - Path to CSV file
 * @param inputSize - Number of input features
 * @param outputSize - Number of output features
 * @returns Object with inputs and targets arrays
 */
export function loadCsv(
  filename: string,
  inputSize: number,
  outputSize: number
): CsvData;

/**
 * Normalize data (zero mean, unit variance)
 * @param inputs - Input data to normalize
 * @returns Normalized data
 */
export function normalize(inputs: number[][]): number[][];
