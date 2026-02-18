//! @file
//! @ingroup MLP_Internal_Logic
/*
 * MIT License
 * Copyright (c) 2025 Matthew Abbott
 */

pub const OPENCL_KERNEL_SRC: &str = r#"
__kernel void feedForwardLayer(
    __global const double* input,
    __global const double* weights,
    __global const double* biases,
    __global double* output,
    int inputSize,
    int outputSize,
    int activationType)
{
    int neuronIdx = get_global_id(0);
    
    if (neuronIdx < outputSize) {
        double sum = biases[neuronIdx];
        
        for (int i = 0; i < inputSize; i++) {
            sum += weights[neuronIdx * inputSize + i] * input[i];
        }
        
        double result = sum;
        if (activationType == 0) { // Sigmoid
            if (sum < -500.0) result = 0.0;
            else if (sum > 500.0) result = 1.0;
            else result = 1.0 / (1.0 + exp(-sum));
        } else if (activationType == 1) { // Tanh
            result = tanh(sum);
        } else if (activationType == 2) { // ReLU
            result = (sum > 0.0) ? sum : 0.0;
        } else if (activationType == 4) { // Linear
            result = sum;
        }
        
        output[neuronIdx] = result;
    }
}

__kernel void softmaxKernel(
    __global const double* input,
    __global double* output,
    int size,
    double maxVal,
    double sumExp)
{
    int idx = get_global_id(0);
    
    if (idx < size) {
        double val = exp(input[idx] - maxVal) / sumExp;
        if (val < 1e-15) val = 1e-15;
        else if (val > 1.0 - 1e-15) val = 1.0 - 1e-15;
        output[idx] = val;
    }
}

__kernel void batchNormForwardKernel(
    __global const double* input,
    __global const double* gamma,
    __global const double* beta,
    __global const double* runningMean,
    __global const double* runningVar,
    __global double* output,
    int size,
    double epsilon)
{
    int idx = get_global_id(0);
    
    if (idx < size) {
        double normalized = (input[idx] - runningMean[idx]) / sqrt(runningVar[idx] + epsilon);
        output[idx] = gamma[idx] * normalized + beta[idx];
    }
}

__kernel void backPropOutputKernel(
    __global double* errors,
    __global const double* outputs,
    __global const double* target,
    int numNeurons,
    int actType,
    int isSoftmax)
{
    int i = get_global_id(0);
    if (i < numNeurons) {
        if (isSoftmax) {
            errors[i] = target[i] - outputs[i];
        } else {
            double derivative;
            if (actType == 0) { // Sigmoid
                derivative = outputs[i] * (1.0 - outputs[i]);
            } else if (actType == 1) { // Tanh
                derivative = 1.0 - (outputs[i] * outputs[i]);
            } else if (actType == 2) { // ReLU
                derivative = (outputs[i] > 0.0) ? 1.0 : 0.0;
            } else { // Linear
                derivative = 1.0;
            }
            errors[i] = derivative * (target[i] - outputs[i]);
        }
    }
}

__kernel void backPropHiddenKernel(
    __global double* errors,
    __global const double* outputs,
    __global const uchar* dropoutMask,
    __global const double* nextErrors,
    __global const double* nextWeights,
    int numNeurons,
    int nextNumNeurons,
    int nextNumInputs,
    int actType)
{
    int i = get_global_id(0);
    if (i < numNeurons) {
        if (!dropoutMask[i]) {
            errors[i] = 0.0;
            return;
        }
        
        double errorSum = 0.0;
        for (int j = 0; j < nextNumNeurons; j++) {
            errorSum += nextErrors[j] * nextWeights[j * nextNumInputs + i];
        }
        
        double derivative;
        if (actType == 0) { // Sigmoid
            derivative = outputs[i] * (1.0 - outputs[i]);
        } else if (actType == 1) { // Tanh
            derivative = 1.0 - (outputs[i] * outputs[i]);
        } else if (actType == 2) { // ReLU
            derivative = (outputs[i] > 0.0) ? 1.0 : 0.0;
        } else { // Linear
            derivative = 1.0;
        }
        
        errors[i] = derivative * errorSum;
    }
}

__kernel void updateWeightsSGDKernel(
    __global double* weights,
    __global double* biases,
    __global const double* errors,
    __global const double* prevOutputs,
    int numNeurons,
    int numInputs,
    int prevSize,
    double learningRate,
    double l2Lambda)
{
    int i = get_global_id(0);
    if (i < numNeurons) {
        for (int j = 0; j < prevSize; j++) {
            double gradient = errors[i] * prevOutputs[j];
            if (l2Lambda > 0.0) {
                gradient = gradient - l2Lambda * weights[i * numInputs + j];
            }
            weights[i * numInputs + j] += learningRate * gradient;
        }
        biases[i] += learningRate * errors[i];
    }
}

__kernel void updateWeightsAdamKernel(
    __global double* weights,
    __global double* biases,
    __global const double* errors,
    __global const double* prevOutputs,
    __global double* M,
    __global double* V,
    __global double* MBias,
    __global double* VBias,
    int numNeurons,
    int numInputs,
    int prevSize,
    double learningRate,
    double l2Lambda,
    double beta1,
    double beta2,
    int timestep)
{
    int i = get_global_id(0);
    if (i < numNeurons) {
        double eps = 1e-8;
        double beta1_t = pow(beta1, (double)timestep);
        double beta2_t = pow(beta2, (double)timestep);
        
        for (int j = 0; j < prevSize; j++) {
            int idx = i * numInputs + j;
            double gradient = -errors[i] * prevOutputs[j];
            if (l2Lambda > 0.0) {
                gradient += l2Lambda * weights[idx];
            }
            
            M[idx] = beta1 * M[idx] + (1.0 - beta1) * gradient;
            V[idx] = beta2 * V[idx] + (1.0 - beta2) * gradient * gradient;
            
            double mHat = M[idx] / (1.0 - beta1_t);
            double vHat = V[idx] / (1.0 - beta2_t);
            
            weights[idx] -= learningRate * mHat / (sqrt(vHat) + eps);
        }
        
        double gradient = -errors[i];
        MBias[i] = beta1 * MBias[i] + (1.0 - beta1) * gradient;
        VBias[i] = beta2 * VBias[i] + (1.0 - beta2) * gradient * gradient;
        double mHat = MBias[i] / (1.0 - beta1_t);
        double vHat = VBias[i] / (1.0 - beta2_t);
        biases[i] -= learningRate * mHat / (sqrt(vHat) + eps);
    }
}

__kernel void updateWeightsRMSPropKernel(
    __global double* weights,
    __global double* biases,
    __global const double* errors,
    __global const double* prevOutputs,
    __global double* V,
    __global double* VBias,
    int numNeurons,
    int numInputs,
    int prevSize,
    double learningRate,
    double l2Lambda)
{
    int i = get_global_id(0);
    if (i < numNeurons) {
        double eps = 1e-8;
        double decay = 0.9;
        
        for (int j = 0; j < prevSize; j++) {
            int idx = i * numInputs + j;
            double gradient = -errors[i] * prevOutputs[j];
            if (l2Lambda > 0.0) {
                gradient += l2Lambda * weights[idx];
            }
            
            V[idx] = decay * V[idx] + (1.0 - decay) * gradient * gradient;
            weights[idx] -= learningRate * gradient / (sqrt(V[idx]) + eps);
        }
        
        double gradient = -errors[i];
        VBias[i] = decay * VBias[i] + (1.0 - decay) * gradient * gradient;
        biases[i] -= learningRate * gradient / (sqrt(VBias[i]) + eps);
    }
}

__kernel void applyDropoutKernel(
    __global double* outputs,
    __global uchar* dropoutMask,
    int numNeurons,
    double dropoutRate,
    double scale,
    ulong seed)
{
    int i = get_global_id(0);
    if (i < numNeurons) {
        if (dropoutRate <= 0.0) {
            dropoutMask[i] = 1;
            return;
        }
        ulong state = seed + i * 1099087573UL;
        state = state * 1103515245UL + 12345UL;
        float randVal = (float)(state % 10000) / 10000.0f;
        if (randVal > dropoutRate) {
            dropoutMask[i] = 1;
            outputs[i] = outputs[i] * scale;
        } else {
            dropoutMask[i] = 0;
            outputs[i] = 0.0;
        }
    }
}
"#;

