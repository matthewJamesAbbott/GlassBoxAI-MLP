/*
 * MIT License
 *
 * Copyright (c) 2025 Matthew Abbott
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>
#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <ctime>
#include <cstring>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <algorithm>
#include <utility>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(1); \
        } \
    } while(0)

const double EPSILON = 1e-15;
const int BLOCK_SIZE = 256;
const char MODEL_MAGIC[] = "MLPCUDA1";

enum TActivationType { atSigmoid = 0, atTanh = 1, atReLU = 2, atSoftmax = 3 };
enum TOptimizerType { otSGD = 0, otAdam = 1, otRMSProp = 2 };
enum TCommand { cmdNone, cmdCreate, cmdTrain, cmdPredict, cmdInfo, cmdHelp,
                cmdGetWeight, cmdSetWeight, cmdGetBias, cmdSetBias,
                cmdGetOutput, cmdGetError, cmdLayerInfo, cmdHistogram,
                cmdGetOptimizer, cmdGetWeights, cmdGetAllOutputs, cmdBatchPredict,
                cmdExportONNX, cmdImportONNX, cmdFeatureImportance };

// Device functions
__device__ double d_Sigmoid(double x) {
    if (x < -500) return 0;
    else if (x > 500) return 1;
    else return 1.0 / (1.0 + exp(-x));
}

__device__ double d_DSigmoid(double x) {
    return x * (1 - x);
}

__device__ double d_TanhActivation(double x) {
    return tanh(x);
}

__device__ double d_DTanh(double x) {
    return 1 - (x * x);
}

__device__ double d_ReLU(double x) {
    return (x > 0) ? x : 0;
}

__device__ double d_DReLU(double x) {
    return (x > 0) ? 1.0 : 0.0;
}

__device__ double d_ApplyActivation(double x, TActivationType ActType) {
    switch (ActType) {
        case atSigmoid: return d_Sigmoid(x);
        case atTanh: return d_TanhActivation(x);
        case atReLU: return d_ReLU(x);
        default: return d_Sigmoid(x);
    }
}

__device__ double d_ApplyActivationDerivative(double x, TActivationType ActType) {
    switch (ActType) {
        case atSigmoid: return d_DSigmoid(x);
        case atTanh: return d_DTanh(x);
        case atReLU: return d_DReLU(x);
        default: return d_DSigmoid(x);
    }
}

struct LayerData {
    double* Weights;
    double* Biases;
    double* Outputs;
    double* Errors;
    double* M;
    double* V;
    double* MBias;
    double* VBias;
    bool* DropoutMask;
    double* d_Gamma;
    double* d_Beta;
    double* d_RunningMean;
    double* d_RunningVar;
    double* d_BatchMean;
    double* d_BatchVar;
    double* d_XNorm;
    double* d_dGamma;
    double* d_dBeta;
    int NumNeurons;
    int NumInputs;
    TActivationType ActivationType;
};

__global__ void InitRandStates(curandState* states, unsigned long seed, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        curand_init(seed, idx, 0, &states[idx]);
    }
}

__global__ void FeedForwardKernel(LayerData layer, double* prevOutputs, int prevSize) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < layer.NumNeurons) {
        double sum = layer.Biases[i];
        for (int j = 0; j < prevSize; j++) {
            sum += prevOutputs[j] * layer.Weights[i * layer.NumInputs + j];
        }
        layer.Outputs[i] = d_ApplyActivation(sum, layer.ActivationType);
    }
}

__global__ void FeedForwardSoftmaxSumKernel(LayerData layer, double* prevOutputs, int prevSize, double* sums) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < layer.NumNeurons) {
        double sum = layer.Biases[i];
        for (int j = 0; j < prevSize; j++) {
            sum += prevOutputs[j] * layer.Weights[i * layer.NumInputs + j];
        }
        sums[i] = sum;
    }
}

__global__ void SoftmaxKernel(double* sums, double* outputs, int n, double maxVal, double sumExp) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        double val = exp(sums[i] - maxVal) / sumExp;
        if (val < EPSILON) val = EPSILON;
        else if (val > 1 - EPSILON) val = 1 - EPSILON;
        outputs[i] = val;
    }
}

__global__ void ApplyDropoutKernel(LayerData layer, curandState* states, double dropoutRate, double scale) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < layer.NumNeurons) {
        if (dropoutRate <= 0) {
            layer.DropoutMask[i] = true;
            return;
        }
        float randVal = curand_uniform(&states[i]);
        if (randVal > dropoutRate) {
            layer.DropoutMask[i] = true;
            layer.Outputs[i] = layer.Outputs[i] * scale;
        } else {
            layer.DropoutMask[i] = false;
            layer.Outputs[i] = 0;
        }
    }
}

__global__ void BackPropOutputKernel(LayerData layer, double* target, bool isSoftmax) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < layer.NumNeurons) {
        if (isSoftmax) {
            layer.Errors[i] = target[i] - layer.Outputs[i];
        } else {
            layer.Errors[i] = d_ApplyActivationDerivative(layer.Outputs[i], layer.ActivationType) *
                              (target[i] - layer.Outputs[i]);
        }
    }
}

__global__ void BackPropHiddenKernel(LayerData layer, LayerData nextLayer) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < layer.NumNeurons) {
        if (!layer.DropoutMask[i]) {
            layer.Errors[i] = 0;
            return;
        }
        double errorSum = 0;
        for (int j = 0; j < nextLayer.NumNeurons; j++) {
            errorSum += nextLayer.Errors[j] * nextLayer.Weights[j * nextLayer.NumInputs + i];
        }
        layer.Errors[i] = d_ApplyActivationDerivative(layer.Outputs[i], layer.ActivationType) * errorSum;
    }
}

__global__ void UpdateWeightsSGDKernel(LayerData layer, double* prevOutputs, double learningRate, double l2Lambda) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < layer.NumNeurons) {
        for (int j = 0; j < layer.NumInputs; j++) {
            double gradient = layer.Errors[i] * prevOutputs[j];
            if (l2Lambda > 0)
                gradient = gradient - l2Lambda * layer.Weights[i * layer.NumInputs + j];
            layer.Weights[i * layer.NumInputs + j] += learningRate * gradient;
        }
        layer.Biases[i] += learningRate * layer.Errors[i];
    }
}

__global__ void UpdateWeightsAdamKernel(LayerData layer, double* prevOutputs, 
                                         double learningRate, double l2Lambda,
                                         double beta1, double beta2, int timestep) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < layer.NumNeurons) {
        double eps = 1e-8;
        double beta1_t = pow(beta1, timestep);
        double beta2_t = pow(beta2, timestep);

        for (int j = 0; j < layer.NumInputs; j++) {
            int idx = i * layer.NumInputs + j;
            double gradient = -layer.Errors[i] * prevOutputs[j];
            if (l2Lambda > 0)
                gradient += l2Lambda * layer.Weights[idx];

            layer.M[idx] = beta1 * layer.M[idx] + (1 - beta1) * gradient;
            layer.V[idx] = beta2 * layer.V[idx] + (1 - beta2) * gradient * gradient;

            double mHat = layer.M[idx] / (1 - beta1_t);
            double vHat = layer.V[idx] / (1 - beta2_t);

            layer.Weights[idx] -= learningRate * mHat / (sqrt(vHat) + eps);
        }

        double gradient = -layer.Errors[i];
        layer.MBias[i] = beta1 * layer.MBias[i] + (1 - beta1) * gradient;
        layer.VBias[i] = beta2 * layer.VBias[i] + (1 - beta2) * gradient * gradient;
        double mHat = layer.MBias[i] / (1 - beta1_t);
        double vHat = layer.VBias[i] / (1 - beta2_t);
        layer.Biases[i] -= learningRate * mHat / (sqrt(vHat) + eps);
    }
}

__global__ void UpdateWeightsRMSPropKernel(LayerData layer, double* prevOutputs,
                                            double learningRate, double l2Lambda) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < layer.NumNeurons) {
        double eps = 1e-8;
        double decay = 0.9;

        for (int j = 0; j < layer.NumInputs; j++) {
            int idx = i * layer.NumInputs + j;
            double gradient = -layer.Errors[i] * prevOutputs[j];
            if (l2Lambda > 0)
                gradient += l2Lambda * layer.Weights[idx];

            layer.V[idx] = decay * layer.V[idx] + (1 - decay) * gradient * gradient;
            layer.Weights[idx] -= learningRate * gradient / (sqrt(layer.V[idx]) + eps);
        }

        double gradient = -layer.Errors[i];
        layer.VBias[i] = decay * layer.VBias[i] + (1 - decay) * gradient * gradient;
        layer.Biases[i] -= learningRate * gradient / (sqrt(layer.VBias[i]) + eps);
    }
}

// Batch Normalization constants
const double BN_MOMENTUM = 0.1;
const double BN_EPSILON = 1e-5;

// Compute batch mean (single sample = just copy)
__global__ void BatchNormComputeMeanKernel(double* outputs, double* batchMean, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        batchMean[i] = outputs[i];
    }
}

// Compute batch variance (single sample = 0, use running stats)
__global__ void BatchNormComputeVarKernel(double* outputs, double* batchMean, double* batchVar, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        double diff = outputs[i] - batchMean[i];
        batchVar[i] = diff * diff;
    }
}

// Forward pass: normalize and scale
__global__ void BatchNormForwardTrainKernel(LayerData layer, double epsilon) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < layer.NumNeurons) {
        double mean = layer.d_BatchMean[i];
        double var = layer.d_BatchVar[i];
        double xnorm = (layer.Outputs[i] - mean) / sqrt(var + epsilon);
        layer.d_XNorm[i] = xnorm;
        layer.Outputs[i] = layer.d_Gamma[i] * xnorm + layer.d_Beta[i];
    }
}

// Forward pass inference: use running mean/var
__global__ void BatchNormForwardInferenceKernel(LayerData layer, double epsilon) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < layer.NumNeurons) {
        double mean = layer.d_RunningMean[i];
        double var = layer.d_RunningVar[i];
        double xnorm = (layer.Outputs[i] - mean) / sqrt(var + epsilon);
        layer.Outputs[i] = layer.d_Gamma[i] * xnorm + layer.d_Beta[i];
    }
}

// Update running mean/var
__global__ void BatchNormUpdateRunningStatsKernel(LayerData layer, double momentum) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < layer.NumNeurons) {
        layer.d_RunningMean[i] = (1.0 - momentum) * layer.d_RunningMean[i] + momentum * layer.d_BatchMean[i];
        layer.d_RunningVar[i] = (1.0 - momentum) * layer.d_RunningVar[i] + momentum * layer.d_BatchVar[i];
    }
}

// Backward pass for batch norm
__global__ void BatchNormBackwardKernel(LayerData layer, double epsilon) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < layer.NumNeurons) {
        double dout = layer.Errors[i];
        double xnorm = layer.d_XNorm[i];
        double var = layer.d_BatchVar[i];
        double stdInv = 1.0 / sqrt(var + epsilon);
        
        layer.d_dGamma[i] = dout * xnorm;
        layer.d_dBeta[i] = dout;
        
        layer.Errors[i] = dout * layer.d_Gamma[i] * stdInv;
    }
}

// Update gamma and beta
__global__ void BatchNormUpdateParamsKernel(LayerData layer, double learningRate) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < layer.NumNeurons) {
        layer.d_Gamma[i] += learningRate * layer.d_dGamma[i];
        layer.d_Beta[i] += learningRate * layer.d_dBeta[i];
    }
}

class TMultiLayerPerceptronCUDA {
private:
    LayerData* d_Layers;
    LayerData* h_Layers;
    int NumLayers;
    int FInputSize;
    int FOutputSize;
    std::vector<int> FHiddenSizes;
    bool FIsTraining;
    curandState* d_RandStates;
    int MaxNeurons;
    bool FBatchNorm;

    double* d_Target;
    double* d_SoftmaxSums;

    void AllocateLayer(LayerData& layer, int numNeurons, int numInputs, TActivationType actType, bool allocBatchNorm = false) {
        layer.NumNeurons = numNeurons;
        layer.NumInputs = numInputs;
        layer.ActivationType = actType;

        int weightSize = numNeurons * numInputs;
        CUDA_CHECK(cudaMalloc(&layer.Weights, weightSize * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&layer.Biases, numNeurons * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&layer.Outputs, numNeurons * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&layer.Errors, numNeurons * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&layer.M, weightSize * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&layer.V, weightSize * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&layer.MBias, numNeurons * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&layer.VBias, numNeurons * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&layer.DropoutMask, numNeurons * sizeof(bool)));

        CUDA_CHECK(cudaMemset(layer.Biases, 0, numNeurons * sizeof(double)));
        CUDA_CHECK(cudaMemset(layer.M, 0, weightSize * sizeof(double)));
        CUDA_CHECK(cudaMemset(layer.V, 0, weightSize * sizeof(double)));
        CUDA_CHECK(cudaMemset(layer.MBias, 0, numNeurons * sizeof(double)));
        CUDA_CHECK(cudaMemset(layer.VBias, 0, numNeurons * sizeof(double)));

        layer.d_Gamma = nullptr;
        layer.d_Beta = nullptr;
        layer.d_RunningMean = nullptr;
        layer.d_RunningVar = nullptr;
        layer.d_BatchMean = nullptr;
        layer.d_BatchVar = nullptr;
        layer.d_XNorm = nullptr;
        layer.d_dGamma = nullptr;
        layer.d_dBeta = nullptr;

        if (allocBatchNorm) {
            CUDA_CHECK(cudaMalloc(&layer.d_Gamma, numNeurons * sizeof(double)));
            CUDA_CHECK(cudaMalloc(&layer.d_Beta, numNeurons * sizeof(double)));
            CUDA_CHECK(cudaMalloc(&layer.d_RunningMean, numNeurons * sizeof(double)));
            CUDA_CHECK(cudaMalloc(&layer.d_RunningVar, numNeurons * sizeof(double)));
            CUDA_CHECK(cudaMalloc(&layer.d_BatchMean, numNeurons * sizeof(double)));
            CUDA_CHECK(cudaMalloc(&layer.d_BatchVar, numNeurons * sizeof(double)));
            CUDA_CHECK(cudaMalloc(&layer.d_XNorm, numNeurons * sizeof(double)));
            CUDA_CHECK(cudaMalloc(&layer.d_dGamma, numNeurons * sizeof(double)));
            CUDA_CHECK(cudaMalloc(&layer.d_dBeta, numNeurons * sizeof(double)));

            double* h_gamma = new double[numNeurons];
            double* h_beta = new double[numNeurons];
            for (int i = 0; i < numNeurons; i++) {
                h_gamma[i] = 1.0;
                h_beta[i] = 0.0;
            }
            CUDA_CHECK(cudaMemcpy(layer.d_Gamma, h_gamma, numNeurons * sizeof(double), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(layer.d_Beta, h_beta, numNeurons * sizeof(double), cudaMemcpyHostToDevice));
            delete[] h_gamma;
            delete[] h_beta;

            CUDA_CHECK(cudaMemset(layer.d_RunningMean, 0, numNeurons * sizeof(double)));
            double* h_var = new double[numNeurons];
            for (int i = 0; i < numNeurons; i++) h_var[i] = 1.0;
            CUDA_CHECK(cudaMemcpy(layer.d_RunningVar, h_var, numNeurons * sizeof(double), cudaMemcpyHostToDevice));
            delete[] h_var;

            CUDA_CHECK(cudaMemset(layer.d_BatchMean, 0, numNeurons * sizeof(double)));
            CUDA_CHECK(cudaMemset(layer.d_BatchVar, 0, numNeurons * sizeof(double)));
            CUDA_CHECK(cudaMemset(layer.d_XNorm, 0, numNeurons * sizeof(double)));
            CUDA_CHECK(cudaMemset(layer.d_dGamma, 0, numNeurons * sizeof(double)));
            CUDA_CHECK(cudaMemset(layer.d_dBeta, 0, numNeurons * sizeof(double)));
        }

        double limit;
        if (actType == atReLU)
            limit = sqrt(2.0 / numInputs);
        else
            limit = sqrt(6.0 / (numInputs + numNeurons));

        double* h_weights = new double[weightSize];
        for (int i = 0; i < weightSize; i++)
            h_weights[i] = ((double)rand() / RAND_MAX * 2 - 1) * limit;
        CUDA_CHECK(cudaMemcpy(layer.Weights, h_weights, weightSize * sizeof(double), cudaMemcpyHostToDevice));
        delete[] h_weights;

        bool* h_mask = new bool[numNeurons];
        for (int i = 0; i < numNeurons; i++) h_mask[i] = true;
        CUDA_CHECK(cudaMemcpy(layer.DropoutMask, h_mask, numNeurons * sizeof(bool), cudaMemcpyHostToDevice));
        delete[] h_mask;
    }

    void FreeLayer(LayerData& layer) {
        if (layer.Weights) cudaFree(layer.Weights);
        if (layer.Biases) cudaFree(layer.Biases);
        if (layer.Outputs) cudaFree(layer.Outputs);
        if (layer.Errors) cudaFree(layer.Errors);
        if (layer.M) cudaFree(layer.M);
        if (layer.V) cudaFree(layer.V);
        if (layer.MBias) cudaFree(layer.MBias);
        if (layer.VBias) cudaFree(layer.VBias);
        if (layer.DropoutMask) cudaFree(layer.DropoutMask);
        if (layer.d_Gamma) cudaFree(layer.d_Gamma);
        if (layer.d_Beta) cudaFree(layer.d_Beta);
        if (layer.d_RunningMean) cudaFree(layer.d_RunningMean);
        if (layer.d_RunningVar) cudaFree(layer.d_RunningVar);
        if (layer.d_BatchMean) cudaFree(layer.d_BatchMean);
        if (layer.d_BatchVar) cudaFree(layer.d_BatchVar);
        if (layer.d_XNorm) cudaFree(layer.d_XNorm);
        if (layer.d_dGamma) cudaFree(layer.d_dGamma);
        if (layer.d_dBeta) cudaFree(layer.d_dBeta);
    }

public:
    double LearningRate;
    int MaxIterations;
    TOptimizerType Optimizer;
    TActivationType HiddenActivation;
    TActivationType OutputActivation;
    double DropoutRate;
    double L2Lambda;
    double Beta1;
    double Beta2;
    int Timestep;
    bool EnableLRDecay;
    double LRDecayRate;
    int LRDecayEpochs;
    bool EnableEarlyStopping;
    int EarlyStoppingPatience;

    TMultiLayerPerceptronCUDA(int InputSize, const std::vector<int>& HiddenSizes, int OutputSize,
                              TActivationType HiddenAct = atSigmoid, TActivationType OutputAct = atSigmoid,
                              bool batchNorm = false) {
        LearningRate = 0.1;
        MaxIterations = 100;
        Optimizer = otSGD;
        HiddenActivation = HiddenAct;
        OutputActivation = OutputAct;
        DropoutRate = 0;
        L2Lambda = 0;
        Beta1 = 0.9;
        Beta2 = 0.999;
        Timestep = 0;
        EnableLRDecay = false;
        LRDecayRate = 0.95;
        LRDecayEpochs = 10;
        EnableEarlyStopping = false;
        EarlyStoppingPatience = 10;
        FIsTraining = true;
        FBatchNorm = batchNorm;

        FInputSize = InputSize;
        FOutputSize = OutputSize;
        FHiddenSizes = HiddenSizes;

        NumLayers = HiddenSizes.size() + 2;
        h_Layers = new LayerData[NumLayers];
        memset(h_Layers, 0, NumLayers * sizeof(LayerData));
        CUDA_CHECK(cudaMalloc(&d_Layers, NumLayers * sizeof(LayerData)));

        AllocateLayer(h_Layers[0], InputSize + 1, InputSize, atSigmoid, false);

        MaxNeurons = InputSize + 1;
        int numInputs = InputSize;
        for (size_t i = 0; i < HiddenSizes.size(); i++) {
            AllocateLayer(h_Layers[i + 1], HiddenSizes[i] + 1, numInputs + 1, HiddenActivation, batchNorm);
            if (HiddenSizes[i] + 1 > MaxNeurons) MaxNeurons = HiddenSizes[i] + 1;
            numInputs = HiddenSizes[i];
        }

        AllocateLayer(h_Layers[NumLayers - 1], OutputSize, numInputs + 1, OutputActivation, false);
        if (OutputSize > MaxNeurons) MaxNeurons = OutputSize;

        CUDA_CHECK(cudaMemcpy(d_Layers, h_Layers, NumLayers * sizeof(LayerData), cudaMemcpyHostToDevice));

        CUDA_CHECK(cudaMalloc(&d_RandStates, MaxNeurons * sizeof(curandState)));
        int blocks = (MaxNeurons + BLOCK_SIZE - 1) / BLOCK_SIZE;
        InitRandStates<<<blocks, BLOCK_SIZE>>>(d_RandStates, time(nullptr), MaxNeurons);
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaMalloc(&d_Target, OutputSize * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_SoftmaxSums, OutputSize * sizeof(double)));
    }

    ~TMultiLayerPerceptronCUDA() {
        for (int i = 0; i < NumLayers; i++)
            FreeLayer(h_Layers[i]);
        delete[] h_Layers;
        cudaFree(d_Layers);
        cudaFree(d_RandStates);
        cudaFree(d_Target);
        cudaFree(d_SoftmaxSums);
    }

    void FeedForward() {
        for (int k = 1; k < NumLayers - 1; k++) {
            LayerData& layer = h_Layers[k];
            LayerData& prevLayer = h_Layers[k - 1];

            int blocks = (layer.NumNeurons + BLOCK_SIZE - 1) / BLOCK_SIZE;
            FeedForwardKernel<<<blocks, BLOCK_SIZE>>>(layer, prevLayer.Outputs, prevLayer.NumNeurons);
            CUDA_CHECK(cudaDeviceSynchronize());

            if (FBatchNorm && layer.d_Gamma != nullptr) {
                BatchNormComputeMeanKernel<<<blocks, BLOCK_SIZE>>>(layer.Outputs, layer.d_BatchMean, layer.NumNeurons);
                CUDA_CHECK(cudaDeviceSynchronize());
                BatchNormComputeVarKernel<<<blocks, BLOCK_SIZE>>>(layer.Outputs, layer.d_BatchMean, layer.d_BatchVar, layer.NumNeurons);
                CUDA_CHECK(cudaDeviceSynchronize());

                if (FIsTraining) {
                    BatchNormForwardTrainKernel<<<blocks, BLOCK_SIZE>>>(layer, BN_EPSILON);
                    CUDA_CHECK(cudaDeviceSynchronize());
                    BatchNormUpdateRunningStatsKernel<<<blocks, BLOCK_SIZE>>>(layer, BN_MOMENTUM);
                    CUDA_CHECK(cudaDeviceSynchronize());
                } else {
                    BatchNormForwardInferenceKernel<<<blocks, BLOCK_SIZE>>>(layer, BN_EPSILON);
                    CUDA_CHECK(cudaDeviceSynchronize());
                }
            }

            if (FIsTraining && DropoutRate > 0) {
                double scale = 1.0 / (1.0 - DropoutRate);
                ApplyDropoutKernel<<<blocks, BLOCK_SIZE>>>(layer, d_RandStates, DropoutRate, scale);
            }
        }

        LayerData& outputLayer = h_Layers[NumLayers - 1];
        LayerData& lastHidden = h_Layers[NumLayers - 2];
        int blocks = (outputLayer.NumNeurons + BLOCK_SIZE - 1) / BLOCK_SIZE;

        if (OutputActivation == atSoftmax) {
            FeedForwardSoftmaxSumKernel<<<blocks, BLOCK_SIZE>>>(outputLayer, lastHidden.Outputs, 
                                                                 lastHidden.NumNeurons, d_SoftmaxSums);
            CUDA_CHECK(cudaDeviceSynchronize());

            double* h_sums = new double[outputLayer.NumNeurons];
            CUDA_CHECK(cudaMemcpy(h_sums, d_SoftmaxSums, outputLayer.NumNeurons * sizeof(double), cudaMemcpyDeviceToHost));

            double maxVal = h_sums[0];
            for (int i = 1; i < outputLayer.NumNeurons; i++)
                if (h_sums[i] > maxVal) maxVal = h_sums[i];

            double sumExp = 0;
            for (int i = 0; i < outputLayer.NumNeurons; i++)
                sumExp += exp(h_sums[i] - maxVal);

            delete[] h_sums;

            SoftmaxKernel<<<blocks, BLOCK_SIZE>>>(d_SoftmaxSums, outputLayer.Outputs, 
                                                   outputLayer.NumNeurons, maxVal, sumExp);
        } else {
            FeedForwardKernel<<<blocks, BLOCK_SIZE>>>(outputLayer, lastHidden.Outputs, lastHidden.NumNeurons);
        }
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    void BackPropagate() {
        LayerData& outputLayer = h_Layers[NumLayers - 1];
        int blocks = (outputLayer.NumNeurons + BLOCK_SIZE - 1) / BLOCK_SIZE;
        BackPropOutputKernel<<<blocks, BLOCK_SIZE>>>(outputLayer, d_Target, OutputActivation == atSoftmax);
        CUDA_CHECK(cudaDeviceSynchronize());

        for (int k = NumLayers - 2; k >= 1; k--) {
            LayerData& layer = h_Layers[k];
            LayerData& nextLayer = h_Layers[k + 1];
            blocks = (layer.NumNeurons + BLOCK_SIZE - 1) / BLOCK_SIZE;
            BackPropHiddenKernel<<<blocks, BLOCK_SIZE>>>(layer, nextLayer);
            CUDA_CHECK(cudaDeviceSynchronize());

            if (FBatchNorm && layer.d_Gamma != nullptr) {
                BatchNormBackwardKernel<<<blocks, BLOCK_SIZE>>>(layer, BN_EPSILON);
                CUDA_CHECK(cudaDeviceSynchronize());
            }
        }
    }

    void UpdateWeights() {
        Timestep++;

        for (int k = NumLayers - 1; k >= 1; k--) {
            LayerData& layer = h_Layers[k];
            LayerData& prevLayer = h_Layers[k - 1];
            int blocks = (layer.NumNeurons + BLOCK_SIZE - 1) / BLOCK_SIZE;

            switch (Optimizer) {
                case otSGD:
                    UpdateWeightsSGDKernel<<<blocks, BLOCK_SIZE>>>(layer, prevLayer.Outputs, LearningRate, L2Lambda);
                    break;
                case otAdam:
                    UpdateWeightsAdamKernel<<<blocks, BLOCK_SIZE>>>(layer, prevLayer.Outputs, 
                                                                     LearningRate, L2Lambda, Beta1, Beta2, Timestep);
                    break;
                case otRMSProp:
                    UpdateWeightsRMSPropKernel<<<blocks, BLOCK_SIZE>>>(layer, prevLayer.Outputs, LearningRate, L2Lambda);
                    break;
            }

            if (FBatchNorm && layer.d_Gamma != nullptr) {
                BatchNormUpdateParamsKernel<<<blocks, BLOCK_SIZE>>>(layer, LearningRate);
            }
        }
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    void Predict(const double* Input, double* Result) {
        FIsTraining = false;

        double* h_input = new double[FInputSize + 1];
        for (int i = 0; i < FInputSize; i++) h_input[i] = Input[i];
        h_input[FInputSize] = 1.0;
        CUDA_CHECK(cudaMemcpy(h_Layers[0].Outputs, h_input, (FInputSize + 1) * sizeof(double), cudaMemcpyHostToDevice));
        delete[] h_input;

        FeedForward();

        CUDA_CHECK(cudaMemcpy(Result, h_Layers[NumLayers - 1].Outputs, FOutputSize * sizeof(double), cudaMemcpyDeviceToHost));

        FIsTraining = true;
    }

    void Train(const double* Input, const double* Target) {
        FIsTraining = true;

        double* h_input = new double[FInputSize + 1];
        for (int i = 0; i < FInputSize; i++) h_input[i] = Input[i];
        h_input[FInputSize] = 1.0;
        CUDA_CHECK(cudaMemcpy(h_Layers[0].Outputs, h_input, (FInputSize + 1) * sizeof(double), cudaMemcpyHostToDevice));
        delete[] h_input;

        CUDA_CHECK(cudaMemcpy(d_Target, Target, FOutputSize * sizeof(double), cudaMemcpyHostToDevice));

        FeedForward();
        BackPropagate();
        UpdateWeights();
    }

    double ComputeLoss(const double* Predicted, const double* Target) {
        double Result = 0;

        if (OutputActivation == atSoftmax) {
            for (int i = 0; i < FOutputSize; i++) {
                double p = Predicted[i];
                if (p < EPSILON) p = EPSILON;
                if (p > 1 - EPSILON) p = 1 - EPSILON;
                Result -= Target[i] * log(p);
            }
        } else {
            for (int i = 0; i < FOutputSize; i++)
                Result += 0.5 * (Target[i] - Predicted[i]) * (Target[i] - Predicted[i]);
        }

        return Result;
    }

    int GetOutputSize() const { return FOutputSize; }
    int GetInputSize() const { return FInputSize; }
    int GetHiddenLayerCount() const { return FHiddenSizes.size(); }
    const std::vector<int>& GetHiddenSizes() const { return FHiddenSizes; }
    int GetNumLayers() const { return NumLayers; }
    bool GetBatchNorm() const { return FBatchNorm; }
    void SetBatchNorm(bool value) { FBatchNorm = value; }

    int GetLayerSize(int layerIdx) const {
        if (layerIdx < 0 || layerIdx >= NumLayers) return 0;
        return h_Layers[layerIdx].NumNeurons;
    }

    bool Save(const char* filename) {
        std::ofstream f(filename);
        if (!f) return false;
        
        f << "{\n";
        f << "  \"magic\": \"" << MODEL_MAGIC << "\",\n";
        f << "  \"input_size\": " << FInputSize << ",\n";
        f << "  \"output_size\": " << FOutputSize << ",\n";
        f << "  \"hidden_sizes\": [";
        for (size_t i = 0; i < FHiddenSizes.size(); i++) {
            if (i > 0) f << ",";
            f << FHiddenSizes[i];
        }
        f << "],\n";
        f << std::fixed << std::setprecision(6);
        f << "  \"learning_rate\": " << LearningRate << ",\n";
        f << "  \"optimizer\": " << (int)Optimizer << ",\n";
        f << "  \"hidden_activation\": " << (int)HiddenActivation << ",\n";
        f << "  \"output_activation\": " << (int)OutputActivation << ",\n";
        f << std::setprecision(4);
        f << "  \"dropout_rate\": " << DropoutRate << ",\n";
        f << std::setprecision(6);
        f << "  \"l2_lambda\": " << L2Lambda << ",\n";
        f << "  \"beta1\": " << Beta1 << ",\n";
        f << "  \"beta2\": " << Beta2 << ",\n";
        f << "  \"batch_norm\": " << (FBatchNorm ? "true" : "false") << ",\n";
        
        f << "  \"input_layer\": {\n";
        f << "    \"neuron_count\": " << FInputSize << "\n";
        f << "  },\n";
        
        f << "  \"hidden_layers\": [\n";
        for (size_t h = 0; h < FHiddenSizes.size(); h++) {
            LayerData& layer = h_Layers[h + 1];
            int numNeurons = FHiddenSizes[h];
            int numInputs = (h == 0) ? FInputSize : FHiddenSizes[h-1];
            
            double* h_weights = new double[layer.NumNeurons * layer.NumInputs];
            double* h_biases = new double[layer.NumNeurons];
            CUDA_CHECK(cudaMemcpy(h_weights, layer.Weights, layer.NumNeurons * layer.NumInputs * sizeof(double), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(h_biases, layer.Biases, layer.NumNeurons * sizeof(double), cudaMemcpyDeviceToHost));
            
            f << "    {\n";
            f << "      \"neuron_count\": " << numNeurons << ",\n";
            f << "      \"neurons\": [\n";
            for (int j = 0; j < numNeurons; j++) {
                f << "        {\n";
                f << "          \"weights\": [";
                for (int w = 0; w < numInputs; w++) {
                    if (w > 0) f << ",";
                    f << std::fixed << std::setprecision(10) << h_weights[j * layer.NumInputs + w];
                }
                f << "],\n";
                f << std::setprecision(10);
                f << "          \"bias\": " << h_biases[j] << "\n";
                f << "        }";
                if (j < numNeurons - 1) f << ",";
                f << "\n";
            }
            f << "      ],\n";
            f << "      \"biases\": [";
            for (int j = 0; j < numNeurons; j++) {
                if (j > 0) f << ",";
                f << std::fixed << std::setprecision(10) << h_biases[j];
            }
            f << "]";
            
            if (FBatchNorm && layer.d_Gamma != nullptr) {
                double* h_gamma = new double[layer.NumNeurons];
                double* h_beta = new double[layer.NumNeurons];
                double* h_runMean = new double[layer.NumNeurons];
                double* h_runVar = new double[layer.NumNeurons];
                CUDA_CHECK(cudaMemcpy(h_gamma, layer.d_Gamma, layer.NumNeurons * sizeof(double), cudaMemcpyDeviceToHost));
                CUDA_CHECK(cudaMemcpy(h_beta, layer.d_Beta, layer.NumNeurons * sizeof(double), cudaMemcpyDeviceToHost));
                CUDA_CHECK(cudaMemcpy(h_runMean, layer.d_RunningMean, layer.NumNeurons * sizeof(double), cudaMemcpyDeviceToHost));
                CUDA_CHECK(cudaMemcpy(h_runVar, layer.d_RunningVar, layer.NumNeurons * sizeof(double), cudaMemcpyDeviceToHost));
                
                f << ",\n      \"bn_gamma\": [";
                for (int j = 0; j < numNeurons; j++) {
                    if (j > 0) f << ",";
                    f << std::fixed << std::setprecision(10) << h_gamma[j];
                }
                f << "],\n      \"bn_beta\": [";
                for (int j = 0; j < numNeurons; j++) {
                    if (j > 0) f << ",";
                    f << std::fixed << std::setprecision(10) << h_beta[j];
                }
                f << "],\n      \"bn_running_mean\": [";
                for (int j = 0; j < numNeurons; j++) {
                    if (j > 0) f << ",";
                    f << std::fixed << std::setprecision(10) << h_runMean[j];
                }
                f << "],\n      \"bn_running_var\": [";
                for (int j = 0; j < numNeurons; j++) {
                    if (j > 0) f << ",";
                    f << std::fixed << std::setprecision(10) << h_runVar[j];
                }
                f << "]";
                
                delete[] h_gamma;
                delete[] h_beta;
                delete[] h_runMean;
                delete[] h_runVar;
            }
            f << "\n";
            f << "    }";
            if (h < FHiddenSizes.size() - 1) f << ",";
            f << "\n";
            
            delete[] h_weights;
            delete[] h_biases;
        }
        f << "  ],\n";
        
        LayerData& outLayer = h_Layers[NumLayers - 1];
        int outNumInputs = FHiddenSizes.empty() ? FInputSize : FHiddenSizes.back();
        
        double* h_weights = new double[outLayer.NumNeurons * outLayer.NumInputs];
        double* h_biases = new double[outLayer.NumNeurons];
        CUDA_CHECK(cudaMemcpy(h_weights, outLayer.Weights, outLayer.NumNeurons * outLayer.NumInputs * sizeof(double), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_biases, outLayer.Biases, outLayer.NumNeurons * sizeof(double), cudaMemcpyDeviceToHost));
        
        f << "  \"output_layer\": {\n";
        f << "    \"neuron_count\": " << FOutputSize << ",\n";
        f << "    \"neurons\": [\n";
        for (int i = 0; i < FOutputSize; i++) {
            f << "      {\n";
            f << "        \"weights\": [";
            for (int w = 0; w < outNumInputs; w++) {
                if (w > 0) f << ",";
                f << std::fixed << std::setprecision(10) << h_weights[i * outLayer.NumInputs + w];
            }
            f << "],\n";
            f << std::setprecision(10);
            f << "        \"bias\": " << h_biases[i] << "\n";
            f << "      }";
            if (i < FOutputSize - 1) f << ",";
            f << "\n";
        }
        f << "    ],\n";
        f << "    \"biases\": [";
        for (int i = 0; i < FOutputSize; i++) {
            if (i > 0) f << ",";
            f << std::fixed << std::setprecision(10) << h_biases[i];
        }
        f << "]\n";
        f << "  }\n";
        f << "}\n";
        
        delete[] h_weights;
        delete[] h_biases;
        f.close();
        return true;
    }

    static TMultiLayerPerceptronCUDA* Load(const char* filename) {
        std::ifstream f(filename);
        if (!f.is_open()) return nullptr;
        
        std::string content((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
        f.close();
        
        auto getJsonNumber = [&](const std::string& key) -> double {
            size_t pos = content.find("\"" + key + "\"");
            if (pos == std::string::npos) return 0.0;
            size_t colonPos = content.find(":", pos);
            size_t nextComma = content.find(",", colonPos);
            size_t nextBracket = content.find("}", colonPos);
            size_t endPos = (nextComma < nextBracket) ? nextComma : nextBracket;
            std::string value = content.substr(colonPos + 1, endPos - colonPos - 1);
            while (!value.empty() && (value[0] == ' ' || value[0] == '\t' || value[0] == '\n')) value.erase(0, 1);
            while (!value.empty() && (value.back() == ' ' || value.back() == '\t' || value.back() == '\n')) value.pop_back();
            try { return std::stod(value); } catch (...) { return 0.0; }
        };
        
        auto getJsonInt = [&](const std::string& key) -> int {
            return (int)getJsonNumber(key);
        };
        
        auto parseArray = [&](const std::string& jsonStr) -> std::vector<double> {
            std::vector<double> result;
            size_t start = jsonStr.find('[');
            size_t end = jsonStr.find(']');
            if (start == std::string::npos || end == std::string::npos) return result;
            std::string arrayContent = jsonStr.substr(start + 1, end - start - 1);
            std::stringstream ss(arrayContent);
            std::string token;
            while (std::getline(ss, token, ',')) {
                while (!token.empty() && (token[0] == ' ' || token[0] == '\t' || token[0] == '\n')) token.erase(0, 1);
                while (!token.empty() && (token.back() == ' ' || token.back() == '\t' || token.back() == '\n')) token.pop_back();
                if (!token.empty()) {
                    try { result.push_back(std::stod(token)); } catch (...) {}
                }
            }
            return result;
        };
        
        int newInputSize = getJsonInt("input_size");
        int newOutputSize = getJsonInt("output_size");
        TActivationType newHiddenAct = (TActivationType)getJsonInt("hidden_activation");
        TActivationType newOutputAct = (TActivationType)getJsonInt("output_activation");
        double newLR = getJsonNumber("learning_rate");
        TOptimizerType newOpt = (TOptimizerType)getJsonInt("optimizer");
        double newDropout = getJsonNumber("dropout_rate");
        double newL2 = getJsonNumber("l2_lambda");
        double newBeta1 = getJsonNumber("beta1");
        double newBeta2 = getJsonNumber("beta2");
        
        bool newBatchNorm = false;
        size_t bnPos = content.find("\"batch_norm\"");
        if (bnPos != std::string::npos) {
            size_t colonPos = content.find(":", bnPos);
            size_t nextComma = content.find(",", colonPos);
            size_t nextBracket = content.find("}", colonPos);
            size_t endPos = (nextComma < nextBracket) ? nextComma : nextBracket;
            std::string value = content.substr(colonPos + 1, endPos - colonPos - 1);
            while (!value.empty() && (value[0] == ' ' || value[0] == '\t' || value[0] == '\n')) value.erase(0, 1);
            while (!value.empty() && (value.back() == ' ' || value.back() == '\t' || value.back() == '\n')) value.pop_back();
            newBatchNorm = (value == "true");
        }
        
        std::vector<int> newHiddenSizes;
        size_t hiddenArrayPos = content.find("\"hidden_sizes\"");
        if (hiddenArrayPos != std::string::npos) {
            size_t startBracket = content.find("[", hiddenArrayPos);
            size_t endBracket = content.find("]", startBracket);
            std::string arrayContent = content.substr(startBracket + 1, endBracket - startBracket - 1);
            std::stringstream ss(arrayContent);
            std::string token;
            while (std::getline(ss, token, ',')) {
                while (!token.empty() && (token[0] == ' ' || token[0] == '\t' || token[0] == '\n')) token.erase(0, 1);
                while (!token.empty() && (token.back() == ' ' || token.back() == '\t' || token.back() == '\n')) token.pop_back();
                if (!token.empty()) {
                    try { newHiddenSizes.push_back(std::stoi(token)); } catch (...) {}
                }
            }
        }
        
        if (newInputSize <= 0 || newOutputSize <= 0 || newHiddenSizes.empty()) return nullptr;
        
        TMultiLayerPerceptronCUDA* mlp = new TMultiLayerPerceptronCUDA(
            newInputSize, newHiddenSizes, newOutputSize, newHiddenAct, newOutputAct, newBatchNorm);
        mlp->LearningRate = newLR;
        mlp->Optimizer = newOpt;
        mlp->DropoutRate = newDropout;
        mlp->L2Lambda = newL2;
        mlp->Beta1 = newBeta1;
        mlp->Beta2 = newBeta2;
        
        size_t searchPos = 0;
        size_t hiddenStart = content.find("\"hidden_layers\"");
        size_t hiddenEnd = content.find("\"output_layer\"");
        if (hiddenStart != std::string::npos && hiddenEnd != std::string::npos) {
            searchPos = hiddenStart;
            for (size_t h = 0; h < newHiddenSizes.size(); h++) {
                LayerData& layer = mlp->h_Layers[h + 1];
                int layerNeurons = newHiddenSizes[h];
                int layerInputs = (h == 0) ? newInputSize : newHiddenSizes[h-1];
                
                double* h_weights = new double[layer.NumNeurons * layer.NumInputs];
                double* h_biases = new double[layer.NumNeurons];
                memset(h_weights, 0, layer.NumNeurons * layer.NumInputs * sizeof(double));
                memset(h_biases, 0, layer.NumNeurons * sizeof(double));
                
                for (int n = 0; n < layerNeurons; n++) {
                    size_t wPos = content.find("\"weights\": [", searchPos);
                    if (wPos != std::string::npos && wPos < hiddenEnd) {
                        size_t wEnd = content.find("]", wPos);
                        std::string weightsStr = content.substr(wPos, wEnd - wPos + 1);
                        std::vector<double> weights = parseArray(weightsStr);
                        for (size_t w = 0; w < weights.size() && w < (size_t)layerInputs; w++) {
                            h_weights[n * layer.NumInputs + w] = weights[w];
                        }
                        searchPos = wEnd + 1;
                    }
                    size_t bPos = content.find("\"bias\": ", searchPos);
                    if (bPos != std::string::npos && bPos < hiddenEnd) {
                        size_t bEnd = content.find_first_of(",}", bPos + 8);
                        std::string biasStr = content.substr(bPos + 8, bEnd - bPos - 8);
                        while (!biasStr.empty() && (biasStr[0] == ' ' || biasStr[0] == '\t' || biasStr[0] == '\n')) biasStr.erase(0, 1);
                        while (!biasStr.empty() && (biasStr.back() == ' ' || biasStr.back() == '\t' || biasStr.back() == '\n')) biasStr.pop_back();
                        try { h_biases[n] = std::stod(biasStr); } catch (...) {}
                        searchPos = bEnd + 1;
                    }
                }
                CUDA_CHECK(cudaMemcpy(layer.Weights, h_weights, layer.NumNeurons * layer.NumInputs * sizeof(double), cudaMemcpyHostToDevice));
                CUDA_CHECK(cudaMemcpy(layer.Biases, h_biases, layer.NumNeurons * sizeof(double), cudaMemcpyHostToDevice));
                delete[] h_weights;
                delete[] h_biases;
                
                if (newBatchNorm && layer.d_Gamma != nullptr) {
                    size_t layerSearchEnd = (h + 1 < newHiddenSizes.size()) ? 
                        content.find("\"neuron_count\"", searchPos) : hiddenEnd;
                    
                    size_t gammaPos = content.find("\"bn_gamma\": [", searchPos);
                    if (gammaPos != std::string::npos && gammaPos < layerSearchEnd) {
                        size_t gammaEnd = content.find("]", gammaPos);
                        std::vector<double> gamma = parseArray(content.substr(gammaPos, gammaEnd - gammaPos + 1));
                        double* h_gamma = new double[layer.NumNeurons];
                        for (int i = 0; i < layerNeurons && i < (int)gamma.size(); i++) h_gamma[i] = gamma[i];
                        CUDA_CHECK(cudaMemcpy(layer.d_Gamma, h_gamma, layer.NumNeurons * sizeof(double), cudaMemcpyHostToDevice));
                        delete[] h_gamma;
                    }
                    
                    size_t betaPos = content.find("\"bn_beta\": [", searchPos);
                    if (betaPos != std::string::npos && betaPos < layerSearchEnd) {
                        size_t betaEnd = content.find("]", betaPos);
                        std::vector<double> beta = parseArray(content.substr(betaPos, betaEnd - betaPos + 1));
                        double* h_beta = new double[layer.NumNeurons];
                        for (int i = 0; i < layerNeurons && i < (int)beta.size(); i++) h_beta[i] = beta[i];
                        CUDA_CHECK(cudaMemcpy(layer.d_Beta, h_beta, layer.NumNeurons * sizeof(double), cudaMemcpyHostToDevice));
                        delete[] h_beta;
                    }
                    
                    size_t meanPos = content.find("\"bn_running_mean\": [", searchPos);
                    if (meanPos != std::string::npos && meanPos < layerSearchEnd) {
                        size_t meanEnd = content.find("]", meanPos);
                        std::vector<double> mean = parseArray(content.substr(meanPos, meanEnd - meanPos + 1));
                        double* h_mean = new double[layer.NumNeurons];
                        for (int i = 0; i < layerNeurons && i < (int)mean.size(); i++) h_mean[i] = mean[i];
                        CUDA_CHECK(cudaMemcpy(layer.d_RunningMean, h_mean, layer.NumNeurons * sizeof(double), cudaMemcpyHostToDevice));
                        delete[] h_mean;
                    }
                    
                    size_t varPos = content.find("\"bn_running_var\": [", searchPos);
                    if (varPos != std::string::npos && varPos < layerSearchEnd) {
                        size_t varEnd = content.find("]", varPos);
                        std::vector<double> var = parseArray(content.substr(varPos, varEnd - varPos + 1));
                        double* h_var = new double[layer.NumNeurons];
                        for (int i = 0; i < layerNeurons && i < (int)var.size(); i++) h_var[i] = var[i];
                        CUDA_CHECK(cudaMemcpy(layer.d_RunningVar, h_var, layer.NumNeurons * sizeof(double), cudaMemcpyHostToDevice));
                        delete[] h_var;
                    }
                }
            }
        }
        
        size_t outputStart = content.find("\"output_layer\"");
        if (outputStart != std::string::npos) {
            searchPos = outputStart;
            LayerData& layer = mlp->h_Layers[mlp->NumLayers - 1];
            int layerInputs = newHiddenSizes.empty() ? newInputSize : newHiddenSizes.back();
            
            double* h_weights = new double[layer.NumNeurons * layer.NumInputs];
            double* h_biases = new double[layer.NumNeurons];
            memset(h_weights, 0, layer.NumNeurons * layer.NumInputs * sizeof(double));
            memset(h_biases, 0, layer.NumNeurons * sizeof(double));
            
            for (int n = 0; n < newOutputSize; n++) {
                size_t wPos = content.find("\"weights\": [", searchPos);
                if (wPos != std::string::npos) {
                    size_t wEnd = content.find("]", wPos);
                    std::string weightsStr = content.substr(wPos, wEnd - wPos + 1);
                    std::vector<double> weights = parseArray(weightsStr);
                    for (size_t w = 0; w < weights.size() && w < (size_t)layerInputs; w++) {
                        h_weights[n * layer.NumInputs + w] = weights[w];
                    }
                    searchPos = wEnd + 1;
                }
                size_t bPos = content.find("\"bias\": ", searchPos);
                if (bPos != std::string::npos) {
                    size_t bEnd = content.find_first_of(",}", bPos + 8);
                    std::string biasStr = content.substr(bPos + 8, bEnd - bPos - 8);
                    while (!biasStr.empty() && (biasStr[0] == ' ' || biasStr[0] == '\t' || biasStr[0] == '\n')) biasStr.erase(0, 1);
                    while (!biasStr.empty() && (biasStr.back() == ' ' || biasStr.back() == '\t' || biasStr.back() == '\n')) biasStr.pop_back();
                    try { h_biases[n] = std::stod(biasStr); } catch (...) {}
                    searchPos = bEnd + 1;
                }
            }
            CUDA_CHECK(cudaMemcpy(layer.Weights, h_weights, layer.NumNeurons * layer.NumInputs * sizeof(double), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(layer.Biases, h_biases, layer.NumNeurons * sizeof(double), cudaMemcpyHostToDevice));
            delete[] h_weights;
            delete[] h_biases;
        }
        
        return mlp;
    }

    // ===== FACADE METHODS =====

    // Get number of weights for a neuron
    int GetWeightsPerNeuron(int layerIdx, int neuronIdx) const {
        if (layerIdx < 0 || layerIdx >= NumLayers) return 0;
        if (neuronIdx < 0 || neuronIdx >= h_Layers[layerIdx].NumNeurons) return 0;
        return h_Layers[layerIdx].NumInputs;
    }

    // Get a specific weight
    double GetNeuronWeight(int layerIdx, int neuronIdx, int weightIdx) const {
        if (layerIdx < 0 || layerIdx >= NumLayers) return 0;
        LayerData& layer = h_Layers[layerIdx];
        if (neuronIdx < 0 || neuronIdx >= layer.NumNeurons) return 0;
        if (weightIdx < 0 || weightIdx >= layer.NumInputs) return 0;
        
        double value;
        int idx = neuronIdx * layer.NumInputs + weightIdx;
        CUDA_CHECK(cudaMemcpy(&value, layer.Weights + idx, sizeof(double), cudaMemcpyDeviceToHost));
        return value;
    }

    // Set a specific weight
    void SetNeuronWeight(int layerIdx, int neuronIdx, int weightIdx, double value) {
        if (layerIdx < 0 || layerIdx >= NumLayers) return;
        LayerData& layer = h_Layers[layerIdx];
        if (neuronIdx < 0 || neuronIdx >= layer.NumNeurons) return;
        if (weightIdx < 0 || weightIdx >= layer.NumInputs) return;
        
        int idx = neuronIdx * layer.NumInputs + weightIdx;
        CUDA_CHECK(cudaMemcpy(layer.Weights + idx, &value, sizeof(double), cudaMemcpyHostToDevice));
    }

    // Get all weights for a neuron
    std::vector<double> GetNeuronWeights(int layerIdx, int neuronIdx) const {
        std::vector<double> result;
        if (layerIdx < 0 || layerIdx >= NumLayers) return result;
        LayerData& layer = h_Layers[layerIdx];
        if (neuronIdx < 0 || neuronIdx >= layer.NumNeurons) return result;
        
        result.resize(layer.NumInputs);
        int idx = neuronIdx * layer.NumInputs;
        CUDA_CHECK(cudaMemcpy(result.data(), layer.Weights + idx, layer.NumInputs * sizeof(double), cudaMemcpyDeviceToHost));
        return result;
    }

    // Get neuron bias
    double GetNeuronBias(int layerIdx, int neuronIdx) const {
        if (layerIdx < 0 || layerIdx >= NumLayers) return 0;
        LayerData& layer = h_Layers[layerIdx];
        if (neuronIdx < 0 || neuronIdx >= layer.NumNeurons) return 0;
        
        double value;
        CUDA_CHECK(cudaMemcpy(&value, layer.Biases + neuronIdx, sizeof(double), cudaMemcpyDeviceToHost));
        return value;
    }

    // Set neuron bias
    void SetNeuronBias(int layerIdx, int neuronIdx, double value) {
        if (layerIdx < 0 || layerIdx >= NumLayers) return;
        LayerData& layer = h_Layers[layerIdx];
        if (neuronIdx < 0 || neuronIdx >= layer.NumNeurons) return;
        
        CUDA_CHECK(cudaMemcpy(layer.Biases + neuronIdx, &value, sizeof(double), cudaMemcpyHostToDevice));
    }

    // Get neuron output
    double GetNeuronOutput(int layerIdx, int neuronIdx) const {
        if (layerIdx < 0 || layerIdx >= NumLayers) return 0;
        LayerData& layer = h_Layers[layerIdx];
        if (neuronIdx < 0 || neuronIdx >= layer.NumNeurons) return 0;
        
        double value;
        CUDA_CHECK(cudaMemcpy(&value, layer.Outputs + neuronIdx, sizeof(double), cudaMemcpyDeviceToHost));
        return value;
    }

    // Get all outputs for a layer
    std::vector<double> GetLayerOutputs(int layerIdx) const {
        std::vector<double> result;
        if (layerIdx < 0 || layerIdx >= NumLayers) return result;
        LayerData& layer = h_Layers[layerIdx];
        
        result.resize(layer.NumNeurons);
        CUDA_CHECK(cudaMemcpy(result.data(), layer.Outputs, layer.NumNeurons * sizeof(double), cudaMemcpyDeviceToHost));
        return result;
    }

    // Get neuron error
    double GetNeuronError(int layerIdx, int neuronIdx) const {
        if (layerIdx < 0 || layerIdx >= NumLayers) return 0;
        LayerData& layer = h_Layers[layerIdx];
        if (neuronIdx < 0 || neuronIdx >= layer.NumNeurons) return 0;
        
        double value;
        CUDA_CHECK(cudaMemcpy(&value, layer.Errors + neuronIdx, sizeof(double), cudaMemcpyDeviceToHost));
        return value;
    }

    // Get all errors for a layer
    std::vector<double> GetLayerErrors(int layerIdx) const {
        std::vector<double> result;
        if (layerIdx < 0 || layerIdx >= NumLayers) return result;
        LayerData& layer = h_Layers[layerIdx];
        
        result.resize(layer.NumNeurons);
        CUDA_CHECK(cudaMemcpy(result.data(), layer.Errors, layer.NumNeurons * sizeof(double), cudaMemcpyDeviceToHost));
        return result;
    }

    // Get optimizer M value for a weight
    double GetWeightM(int layerIdx, int neuronIdx, int weightIdx) const {
        if (layerIdx < 0 || layerIdx >= NumLayers) return 0;
        LayerData& layer = h_Layers[layerIdx];
        if (neuronIdx < 0 || neuronIdx >= layer.NumNeurons) return 0;
        if (weightIdx < 0 || weightIdx >= layer.NumInputs) return 0;
        
        double value;
        int idx = neuronIdx * layer.NumInputs + weightIdx;
        CUDA_CHECK(cudaMemcpy(&value, layer.M + idx, sizeof(double), cudaMemcpyDeviceToHost));
        return value;
    }

    // Get optimizer V value for a weight
    double GetWeightV(int layerIdx, int neuronIdx, int weightIdx) const {
        if (layerIdx < 0 || layerIdx >= NumLayers) return 0;
        LayerData& layer = h_Layers[layerIdx];
        if (neuronIdx < 0 || neuronIdx >= layer.NumNeurons) return 0;
        if (weightIdx < 0 || weightIdx >= layer.NumInputs) return 0;
        
        double value;
        int idx = neuronIdx * layer.NumInputs + weightIdx;
        CUDA_CHECK(cudaMemcpy(&value, layer.V + idx, sizeof(double), cudaMemcpyDeviceToHost));
        return value;
    }

    // Get bias M value
    double GetBiasM(int layerIdx, int neuronIdx) const {
        if (layerIdx < 0 || layerIdx >= NumLayers) return 0;
        LayerData& layer = h_Layers[layerIdx];
        if (neuronIdx < 0 || neuronIdx >= layer.NumNeurons) return 0;
        
        double value;
        CUDA_CHECK(cudaMemcpy(&value, layer.MBias + neuronIdx, sizeof(double), cudaMemcpyDeviceToHost));
        return value;
    }

    // Get bias V value
    double GetBiasV(int layerIdx, int neuronIdx) const {
        if (layerIdx < 0 || layerIdx >= NumLayers) return 0;
        LayerData& layer = h_Layers[layerIdx];
        if (neuronIdx < 0 || neuronIdx >= layer.NumNeurons) return 0;
        
        double value;
        CUDA_CHECK(cudaMemcpy(&value, layer.VBias + neuronIdx, sizeof(double), cudaMemcpyDeviceToHost));
        return value;
    }

    // Get dropout mask for a neuron
    bool GetDropoutMask(int layerIdx, int neuronIdx) const {
        if (layerIdx < 0 || layerIdx >= NumLayers) return false;
        LayerData& layer = h_Layers[layerIdx];
        if (neuronIdx < 0 || neuronIdx >= layer.NumNeurons) return false;
        
        bool value;
        CUDA_CHECK(cudaMemcpy(&value, layer.DropoutMask + neuronIdx, sizeof(bool), cudaMemcpyDeviceToHost));
        return value;
    }

    // Get layer activation type
    TActivationType GetLayerActivation(int layerIdx) const {
        if (layerIdx < 0 || layerIdx >= NumLayers) return atSigmoid;
        return h_Layers[layerIdx].ActivationType;
    }

    // Compute histogram of activations for a layer
    std::vector<int> GetActivationHistogram(int layerIdx, int numBins = 20) const {
        std::vector<int> histogram(numBins, 0);
        if (layerIdx < 0 || layerIdx >= NumLayers) return histogram;
        
        std::vector<double> outputs = GetLayerOutputs(layerIdx);
        if (outputs.empty()) return histogram;
        
        double minVal = outputs[0], maxVal = outputs[0];
        for (double v : outputs) {
            if (v < minVal) minVal = v;
            if (v > maxVal) maxVal = v;
        }
        
        if (maxVal == minVal) maxVal = minVal + 1;
        double binWidth = (maxVal - minVal) / numBins;
        
        for (double v : outputs) {
            int bin = (int)((v - minVal) / binWidth);
            if (bin >= numBins) bin = numBins - 1;
            if (bin < 0) bin = 0;
            histogram[bin]++;
        }
        return histogram;
    }

    // Compute histogram of gradients/errors for a layer
    std::vector<int> GetGradientHistogram(int layerIdx, int numBins = 20) const {
        std::vector<int> histogram(numBins, 0);
        if (layerIdx < 0 || layerIdx >= NumLayers) return histogram;
        
        std::vector<double> errors = GetLayerErrors(layerIdx);
        if (errors.empty()) return histogram;
        
        double minVal = errors[0], maxVal = errors[0];
        for (double v : errors) {
            if (v < minVal) minVal = v;
            if (v > maxVal) maxVal = v;
        }
        
        if (maxVal == minVal) maxVal = minVal + 1;
        double binWidth = (maxVal - minVal) / numBins;
        
        for (double v : errors) {
            int bin = (int)((v - minVal) / binWidth);
            if (bin >= numBins) bin = numBins - 1;
            if (bin < 0) bin = 0;
            histogram[bin]++;
        }
        return histogram;
    }

    // Calculate feature importance based on first hidden layer weights
    std::vector<std::pair<int, double>> GetFeatureImportance() const {
        std::vector<std::pair<int, double>> importance;
        if (NumLayers < 2) return importance;
        
        LayerData& firstHiddenLayer = h_Layers[1];
        int numInputs = FInputSize;
        int numNeurons = FHiddenSizes.empty() ? FOutputSize : FHiddenSizes[0];
        
        double* h_weights = new double[firstHiddenLayer.NumNeurons * firstHiddenLayer.NumInputs];
        CUDA_CHECK(cudaMemcpy(h_weights, firstHiddenLayer.Weights, 
                              firstHiddenLayer.NumNeurons * firstHiddenLayer.NumInputs * sizeof(double), 
                              cudaMemcpyDeviceToHost));
        
        for (int i = 0; i < numInputs; i++) {
            double sum = 0.0;
            for (int j = 0; j < numNeurons; j++) {
                sum += fabs(h_weights[j * firstHiddenLayer.NumInputs + i]);
            }
            importance.push_back(std::make_pair(i, sum));
        }
        
        delete[] h_weights;
        
        std::sort(importance.begin(), importance.end(), 
                  [](const std::pair<int, double>& a, const std::pair<int, double>& b) {
                      return b.second < a.second;
                  });
        
        return importance;
    }

    // Export to ONNX format
    bool ExportToONNX(const char* filename) const {
        std::ofstream f(filename, std::ios::binary);
        if (!f) return false;
        
        const uint8_t IR_VERSION = 7;
        const int64_t OPSET_VERSION = 13;
        
        auto writeVarint = [&f](uint64_t value) {
            while (value > 0x7F) {
                f.put((uint8_t)((value & 0x7F) | 0x80));
                value >>= 7;
            }
            f.put((uint8_t)value);
        };
        
        auto writeString = [&f, &writeVarint](const std::string& s) {
            writeVarint(s.length());
            f.write(s.data(), s.length());
        };
        
        auto writeFieldTag = [&f](int fieldNumber, int wireType) {
            f.put((uint8_t)((fieldNumber << 3) | wireType));
        };
        
        std::stringstream onnxData;
        
        onnxData.put(0x08); onnxData.put(IR_VERSION);
        
        std::string producerName = "GlassBoxAI-MLP-CUDA";
        onnxData.put(0x12);
        onnxData.put((uint8_t)producerName.length());
        onnxData.write(producerName.data(), producerName.length());
        
        std::string modelVersion = "1.0";
        onnxData.put(0x22);
        onnxData.put((uint8_t)modelVersion.length());
        onnxData.write(modelVersion.data(), modelVersion.length());
        
        std::stringstream graphData;
        
        std::string graphName = "mlp_graph";
        graphData.put(0x0A);
        graphData.put((uint8_t)graphName.length());
        graphData.write(graphName.data(), graphName.length());
        
        std::string inputName = "input";
        std::stringstream inputTensor;
        inputTensor.put(0x0A);
        inputTensor.put((uint8_t)inputName.length());
        inputTensor.write(inputName.data(), inputName.length());
        
        std::stringstream inputType;
        inputType.put(0x0A);
        std::stringstream tensorType;
        tensorType.put(0x08); tensorType.put(0x01);
        std::stringstream shapeData;
        std::stringstream dim1;
        dim1.put(0x08); dim1.put(0x01);
        std::stringstream dim2;
        dim2.put(0x08); dim2.put((uint8_t)FInputSize);
        shapeData.put(0x0A); shapeData.put((uint8_t)dim1.str().size()); shapeData << dim1.str();
        shapeData.put(0x0A); shapeData.put((uint8_t)dim2.str().size()); shapeData << dim2.str();
        tensorType.put(0x12); tensorType.put((uint8_t)shapeData.str().size()); tensorType << shapeData.str();
        inputType.put((uint8_t)tensorType.str().size()); inputType << tensorType.str();
        inputTensor.put(0x12); inputTensor.put((uint8_t)inputType.str().size()); inputTensor << inputType.str();
        
        graphData.put(0x5A);
        graphData.put((uint8_t)inputTensor.str().size());
        graphData << inputTensor.str();
        
        std::string outputName = "output";
        std::stringstream outputTensor;
        outputTensor.put(0x0A);
        outputTensor.put((uint8_t)outputName.length());
        outputTensor.write(outputName.data(), outputName.length());
        
        std::stringstream outputType;
        outputType.put(0x0A);
        std::stringstream outTensorType;
        outTensorType.put(0x08); outTensorType.put(0x01);
        std::stringstream outShapeData;
        std::stringstream outDim1;
        outDim1.put(0x08); outDim1.put(0x01);
        std::stringstream outDim2;
        outDim2.put(0x08); outDim2.put((uint8_t)FOutputSize);
        outShapeData.put(0x0A); outShapeData.put((uint8_t)outDim1.str().size()); outShapeData << outDim1.str();
        outShapeData.put(0x0A); outShapeData.put((uint8_t)outDim2.str().size()); outShapeData << outDim2.str();
        outTensorType.put(0x12); outTensorType.put((uint8_t)outShapeData.str().size()); outTensorType << outShapeData.str();
        outputType.put((uint8_t)outTensorType.str().size()); outputType << outTensorType.str();
        outputTensor.put(0x12); outputTensor.put((uint8_t)outputType.str().size()); outputTensor << outputType.str();
        
        graphData.put(0x62);
        graphData.put((uint8_t)outputTensor.str().size());
        graphData << outputTensor.str();
        
        std::string graphStr = graphData.str();
        onnxData.put(0x3A);
        if (graphStr.size() < 128) {
            onnxData.put((uint8_t)graphStr.size());
        } else {
            onnxData.put((uint8_t)((graphStr.size() & 0x7F) | 0x80));
            onnxData.put((uint8_t)(graphStr.size() >> 7));
        }
        onnxData << graphStr;
        
        std::stringstream opsetData;
        opsetData.put(0x08);
        opsetData.put((uint8_t)OPSET_VERSION);
        onnxData.put(0x42);
        onnxData.put((uint8_t)opsetData.str().size());
        onnxData << opsetData.str();
        
        std::string onnxStr = onnxData.str();
        f.write(onnxStr.data(), onnxStr.size());
        f.close();
        
        printf("Note: ONNX export is simplified. For full ONNX support, use the JSON model with a converter.\n");
        return true;
    }

    // Import from ONNX format (simplified - loads from text representation)
    static TMultiLayerPerceptronCUDA* ImportFromONNX(const char* filename) {
        std::ifstream f(filename, std::ios::binary);
        if (!f.is_open()) {
            printf("Error: Cannot open ONNX file: %s\n", filename);
            printf("Note: ONNX import is limited. Use JSON format for full model loading.\n");
            return nullptr;
        }
        
        f.seekg(0, std::ios::end);
        size_t fileSize = f.tellg();
        f.seekg(0, std::ios::beg);
        
        std::vector<uint8_t> data(fileSize);
        f.read((char*)data.data(), fileSize);
        f.close();
        
        if (fileSize < 10) {
            printf("Error: ONNX file too small\n");
            return nullptr;
        }
        
        printf("ONNX file size: %zu bytes\n", fileSize);
        printf("Note: Full ONNX parsing is complex. This implementation reads basic structure.\n");
        printf("For full ONNX support, convert the ONNX model to JSON format first.\n");
        
        return nullptr;
    }
};

// Data structures
struct DataPoint {
    std::vector<double> Input;
    std::vector<double> Target;
};

// Utility functions
const char* ActivationToStr(TActivationType act) {
    switch (act) {
        case atSigmoid: return "sigmoid";
        case atTanh: return "tanh";
        case atReLU: return "relu";
        case atSoftmax: return "softmax";
    }
    return "sigmoid";
}

const char* OptimizerToStr(TOptimizerType opt) {
    switch (opt) {
        case otSGD: return "sgd";
        case otAdam: return "adam";
        case otRMSProp: return "rmsprop";
    }
    return "sgd";
}

TActivationType ParseActivation(const char* s) {
    if (strcasecmp(s, "tanh") == 0) return atTanh;
    if (strcasecmp(s, "relu") == 0) return atReLU;
    if (strcasecmp(s, "softmax") == 0) return atSoftmax;
    return atSigmoid;
}

TOptimizerType ParseOptimizer(const char* s) {
    if (strcasecmp(s, "adam") == 0) return otAdam;
    if (strcasecmp(s, "rmsprop") == 0) return otRMSProp;
    return otSGD;
}

std::vector<int> ParseIntArray(const char* s) {
    std::vector<int> result;
    std::stringstream ss(s);
    std::string token;
    while (std::getline(ss, token, ',')) {
        result.push_back(atoi(token.c_str()));
    }
    return result;
}

std::vector<double> ParseDoubleArray(const char* s) {
    std::vector<double> result;
    std::stringstream ss(s);
    std::string token;
    while (std::getline(ss, token, ',')) {
        result.push_back(atof(token.c_str()));
    }
    return result;
}

std::vector<DataPoint> LoadDataCSV(const char* filename, int inputSize, int outputSize) {
    std::vector<DataPoint> data;
    std::ifstream file(filename);
    if (!file.is_open()) return data;

    std::string line;
    while (std::getline(file, line)) {
        if (line.empty()) continue;
        std::vector<double> values = ParseDoubleArray(line.c_str());
        if ((int)values.size() < inputSize + outputSize) continue;

        DataPoint dp;
        dp.Input.resize(inputSize);
        dp.Target.resize(outputSize);
        for (int i = 0; i < inputSize; i++) dp.Input[i] = values[i];
        for (int i = 0; i < outputSize; i++) dp.Target[i] = values[inputSize + i];
        data.push_back(dp);
    }
    return data;
}

void ShuffleData(std::vector<DataPoint>& data) {
    for (int i = data.size() - 1; i >= 1; i--) {
        int j = rand() % (i + 1);
        std::swap(data[i], data[j]);
    }
}

void NormalizeData(std::vector<DataPoint>& data) {
    if (data.empty()) return;
    int inputSize = data[0].Input.size();

    std::vector<double> mins(inputSize), maxs(inputSize);
    for (int j = 0; j < inputSize; j++) {
        mins[j] = maxs[j] = data[0].Input[j];
    }
    for (auto& dp : data) {
        for (int j = 0; j < inputSize; j++) {
            if (dp.Input[j] < mins[j]) mins[j] = dp.Input[j];
            if (dp.Input[j] > maxs[j]) maxs[j] = dp.Input[j];
        }
    }
    for (auto& dp : data) {
        for (int j = 0; j < inputSize; j++) {
            double range = maxs[j] - mins[j];
            dp.Input[j] = (range > 0) ? (dp.Input[j] - mins[j]) / range : 0.5;
        }
    }
}

int MaxIndex(const double* arr, int n) {
    int result = 0;
    for (int i = 1; i < n; i++)
        if (arr[i] > arr[result])
            result = i;
    return result;
}

void PrintUsage() {
    printf("Facaded MLP\n");
    printf("\n");
    printf("Usage: facaded_mlp <command> [options]\n");
    printf("\n");
    printf("Commands:\n");
    printf("  create         Create a new MLP model\n");
    printf("  train          Train an existing model with data\n");
    printf("  predict        Make predictions with a trained model\n");
    printf("  batch-predict  Make predictions with a trained model (batch)\n");
    printf("  info           Display model information\n");
    printf("  get-weight     Get a single weight value (FACADE)\n");
    printf("  set-weight     Set a single weight value (FACADE)\n");
    printf("  get-weights    Get all weights for a neuron (FACADE)\n");
    printf("  get-bias       Get bias value for a neuron (FACADE)\n");
    printf("  set-bias       Set bias value for a neuron (FACADE)\n");
    printf("  get-output     Get neuron output value (FACADE)\n");
    printf("  get-error      Get neuron error value (FACADE)\n");
    printf("  layer-info     Display layer information (FACADE)\n");
    printf("  histogram      Display activation or error histogram (FACADE)\n");
    printf("  get-optimizer  Get optimizer state values M, V (FACADE)\n");
    printf("  export-onnx    Export model to ONNX format\n");
    printf("  import-onnx    Import model from ONNX format\n");
    printf("  feature-importance  Calculate and display feature importance\n");
    printf("  help           Show this help message\n");
    printf("\n");
    printf("Create Options:\n");
    printf("  -i, --input=N              Input layer size (required)\n");
    printf("  -H, --hidden=N,N,...       Hidden layer sizes (required)\n");
    printf("  -o, --output=N             Output layer size (required)\n");
    printf("  -s, --save=FILE            Save model to file (required)\n");
    printf("  --lr=VALUE                 Learning rate (default: 0.1)\n");
    printf("  --optimizer=TYPE           sgd|adam|rmsprop (default: sgd)\n");
    printf("  --hidden-act=TYPE          sigmoid|tanh|relu|softmax (default: sigmoid)\n");
    printf("  --output-act=TYPE          sigmoid|tanh|relu|softmax (default: sigmoid)\n");
    printf("  --dropout=VALUE            Dropout rate 0-1 (default: 0)\n");
    printf("  --l2=VALUE                 L2 regularization (default: 0)\n");
    printf("  --beta1=VALUE              Adam beta1 (default: 0.9)\n");
    printf("  --beta2=VALUE              Adam beta2 (default: 0.999)\n");
    printf("  --batch-norm               Enable batch normalization\n");
    printf("\n");
    printf("Train Options:\n");
    printf("  -m, --model=FILE           Load model from file (required)\n");
    printf("  -d, --data=FILE            Training data CSV file (required)\n");
    printf("  -s, --save=FILE            Save trained model to file (required)\n");
    printf("  --epochs=N                 Number of training epochs (default: 100)\n");
    printf("  --batch=N                  Batch size (default: 1)\n");
    printf("  --lr=VALUE                 Override learning rate\n");
    printf("  --lr-decay                 Enable learning rate decay\n");
    printf("  --lr-decay-rate=VALUE      LR decay rate (default: 0.95)\n");
    printf("  --lr-decay-epochs=N        Epochs between decay (default: 10)\n");
    printf("  --early-stop               Enable early stopping\n");
    printf("  --patience=N               Early stopping patience (default: 10)\n");
    printf("  --normalize                Normalize input data\n");
    printf("  --verbose                  Show training progress\n");
    printf("\n");
    printf("Predict Options:\n");
    printf("  -m, --model=FILE           Model file to load (required)\n");
    printf("  -i, --input=v1,v2,...      Input values (required)\n");
    printf("\n");
    printf("Info Options:\n");
    printf("  -m, --model=FILE           Model file to load (required)\n");
    printf("\n");
    printf("Export/Import ONNX Options:\n");
    printf("  -m, --model=FILE           Model file to load (for export-onnx)\n");
    printf("  --onnx=FILE                ONNX file to import (for import-onnx)\n");
    printf("  -s, --save=FILE            Output file (required)\n");
    printf("\n");
    printf("Feature Importance Options:\n");
    printf("  -m, --model=FILE           Model file to load (required)\n");
    printf("\n");
    printf("Facade Options (for get/set commands):\n");
    printf("  -m, --model=FILE           Model file (required)\n");
    printf("  --layer=L                  Layer index (required)\n");
    printf("  --neuron=N                 Neuron index (required)\n");
    printf("  --weight=W                 Weight index within neuron\n");
    printf("  --value=V                  Value to set (required for set-* commands)\n");
    printf("  -s, --save=FILE            Save modified model to file (required for set-* commands)\n");
    printf("  --bins=N                   Number of histogram bins (default: 20)\n");
    printf("  --type=TYPE                Histogram type: activation|error (default: activation)\n");
    printf("  -i, --input=v1,v2,...      Input values for get-output command\n");
    printf("\n");
    printf("Examples:\n");
    printf("  facaded_mlp create -i 2 -H 8 -o 1 -s xor.json\n");
    printf("  facaded_mlp train -m xor.json -d data.csv -s xor_trained.json --epochs=1000\n");
    printf("  facaded_mlp predict -m xor_trained.json -i 1,0\n");
    printf("  facaded_mlp batch-predict -m xor_trained.json -i 1,0\n");
    printf("  facaded_mlp info -m xor_trained.json\n");
    printf("  facaded_mlp get-weight -m xor.json --layer=1 --neuron=0 --weight=0\n");
    printf("  facaded_mlp set-weight -m xor.json --layer=1 --neuron=0 --weight=0 --value=0.5 -s xor_mod.json\n");
    printf("  facaded_mlp layer-info -m xor.json --layer=0\n");
    printf("  facaded_mlp histogram -m xor.json --layer=1 --bins=30 --type=activation\n");
    printf("  facaded_mlp get-output -m xor.json --layer=0 --neuron=3 -i 1,0\n");
    printf("  facaded_mlp get-optimizer -m xor.json --layer=1 --neuron=0\n");
    printf("  facaded_mlp export-onnx -m xor.json --save=xor.onnx\n");
    printf("  facaded_mlp import-onnx --model=xor.onnx --save=xor_imported.json\n");
    printf("  facaded_mlp feature-importance -m xor_trained.json\n");
    printf("  facaded_mlp create --input=4 --hidden=8,4 --output=1 --batch-norm --save=bn_model.json\n");
}

int main(int argc, char** argv) {
    srand((unsigned)time(nullptr));
    
    // Show help if no arguments provided or if --help is the first argument
    if (argc < 2) {
        PrintUsage();
        return 0;
    }

    TCommand command = cmdNone;
    std::string cmdStr = argv[1];
    if (cmdStr == "create") command = cmdCreate;
    else if (cmdStr == "train") command = cmdTrain;
    else if (cmdStr == "predict") command = cmdPredict;
    else if (cmdStr == "batch-predict") command = cmdBatchPredict;
    else if (cmdStr == "info") command = cmdInfo;
    else if (cmdStr == "help" || cmdStr == "--help" || cmdStr == "-h") command = cmdHelp;
    else if (cmdStr == "get-weight") command = cmdGetWeight;
    else if (cmdStr == "set-weight") command = cmdSetWeight;
    else if (cmdStr == "get-weights") command = cmdGetWeights;
    else if (cmdStr == "get-bias") command = cmdGetBias;
    else if (cmdStr == "set-bias") command = cmdSetBias;
    else if (cmdStr == "get-output") command = cmdGetOutput;
    else if (cmdStr == "get-outputs") command = cmdGetAllOutputs;
    else if (cmdStr == "get-error") command = cmdGetError;
    else if (cmdStr == "layer-info") command = cmdLayerInfo;
    else if (cmdStr == "histogram") command = cmdHistogram;
    else if (cmdStr == "get-optimizer") command = cmdGetOptimizer;
    else if (cmdStr == "export-onnx") command = cmdExportONNX;
    else if (cmdStr == "import-onnx") command = cmdImportONNX;
    else if (cmdStr == "feature-importance") command = cmdFeatureImportance;
    else {
        printf("Unknown command: %s\n", argv[1]);
        PrintUsage();
        return 1;
    }

    if (command == cmdHelp) {
        PrintUsage();
        return 0;
    }

    // Parse arguments
    int inputSize = 0, outputSize = 0;
    std::vector<int> hiddenSizes;
    std::vector<double> inputValues;
    std::string modelFile, saveFile, dataFile;
    double learningRate = 0.1;
    TOptimizerType optimizer = otSGD;
    TActivationType hiddenAct = atSigmoid, outputAct = atSigmoid;
    double dropoutRate = 0, l2Lambda = 0, beta1 = 0.9, beta2 = 0.999;
    int epochs = 100, batchSize = 1;
    bool lrDecay = false, earlyStop = false, normalize = false, verbose = false;
    bool batchNorm = false;
    double lrDecayRate = 0.95;
    int lrDecayEpochs = 10, patience = 10;
    bool lrOverride = false;
    std::string onnxFile;
    
    // Facade arguments
    int layerIdx = -1, neuronIdx = -1, weightIdx = -1;
    double setValue = 0;
    bool hasSetValue = false;
    std::string histogramType = "activation";
    int histogramBins = 20;
    std::vector<double> runInput;

    for (int i = 2; i < argc; i++) {
        std::string arg = argv[i];

        if (arg == "--lr-decay") { lrDecay = true; continue; }
        if (arg == "--early-stop") { earlyStop = true; continue; }
        if (arg == "--normalize") { normalize = true; continue; }
        if (arg == "--verbose") { verbose = true; continue; }
        if (arg == "--batch-norm") { batchNorm = true; continue; }

        size_t eq = arg.find('=');
        if (eq == std::string::npos) {
            printf("Invalid argument: %s\n", arg.c_str());
            continue;
        }

        std::string key = arg.substr(0, eq);
        std::string value = arg.substr(eq + 1);

        if (key == "--input") {
            if (command == cmdPredict || command == cmdBatchPredict)
                inputValues = ParseDoubleArray(value.c_str());
            else
                inputSize = atoi(value.c_str());
        }
        else if (key == "--hidden") hiddenSizes = ParseIntArray(value.c_str());
        else if (key == "--output") outputSize = atoi(value.c_str());
        else if (key == "--model") modelFile = value;
        else if (key == "--save") saveFile = value;
        else if (key == "--data") dataFile = value;
        else if (key == "--lr") { learningRate = atof(value.c_str()); lrOverride = true; }
        else if (key == "--optimizer") optimizer = ParseOptimizer(value.c_str());
        else if (key == "--hidden-act") hiddenAct = ParseActivation(value.c_str());
        else if (key == "--output-act") outputAct = ParseActivation(value.c_str());
        else if (key == "--dropout") dropoutRate = atof(value.c_str());
        else if (key == "--l2") l2Lambda = atof(value.c_str());
        else if (key == "--beta1") beta1 = atof(value.c_str());
        else if (key == "--beta2") beta2 = atof(value.c_str());
        else if (key == "--epochs") epochs = atoi(value.c_str());
        else if (key == "--batch") batchSize = atoi(value.c_str());
        else if (key == "--lr-decay-rate") lrDecayRate = atof(value.c_str());
        else if (key == "--lr-decay-epochs") lrDecayEpochs = atoi(value.c_str());
        else if (key == "--patience") patience = atoi(value.c_str());
        // Facade arguments
        else if (key == "--layer") layerIdx = atoi(value.c_str());
        else if (key == "--neuron") neuronIdx = atoi(value.c_str());
        else if (key == "--weight") weightIdx = atoi(value.c_str());
        else if (key == "--value") { setValue = atof(value.c_str()); hasSetValue = true; }
        else if (key == "--type") histogramType = value;
        else if (key == "--bins") histogramBins = atoi(value.c_str());
        else if (key == "--run-input") runInput = ParseDoubleArray(value.c_str());
        else if (key == "--onnx") onnxFile = value;
        else printf("Unknown option: %s\n", key.c_str());
    }

    // Check CUDA
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    if (deviceCount == 0) {
        printf("Error: No CUDA devices found!\n");
        return 1;
    }

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);

    // Execute command
    if (command == cmdCreate) {
        if (inputSize <= 0) { printf("Error: --input is required\n"); return 1; }
        if (hiddenSizes.empty()) { printf("Error: --hidden is required\n"); return 1; }
        if (outputSize <= 0) { printf("Error: --output is required\n"); return 1; }
        if (saveFile.empty()) { printf("Error: --save is required\n"); return 1; }

        TMultiLayerPerceptronCUDA* mlp = new TMultiLayerPerceptronCUDA(
            inputSize, hiddenSizes, outputSize, hiddenAct, outputAct, batchNorm);
        mlp->LearningRate = learningRate;
        mlp->Optimizer = optimizer;
        mlp->DropoutRate = dropoutRate;
        mlp->L2Lambda = l2Lambda;
        mlp->Beta1 = beta1;
        mlp->Beta2 = beta2;

        mlp->Save(saveFile.c_str());

        printf("Created MLP model (GPU: %s):\n", prop.name);
        printf("  Input size: %d\n", inputSize);
        printf("  Hidden sizes: ");
        for (size_t i = 0; i < hiddenSizes.size(); i++)
            printf("%s%d", i > 0 ? "," : "", hiddenSizes[i]);
        printf("\n");
        printf("  Output size: %d\n", outputSize);
        printf("  Hidden activation: %s\n", ActivationToStr(hiddenAct));
        printf("  Output activation: %s\n", ActivationToStr(outputAct));
        printf("  Optimizer: %s\n", OptimizerToStr(optimizer));
        printf("  Learning rate: %.4f\n", learningRate);
        printf("  Batch normalization: %s\n", batchNorm ? "enabled" : "disabled");
        printf("  Saved to: %s\n", saveFile.c_str());

        delete mlp;
    }
    else if (command == cmdTrain) {
        if (modelFile.empty()) { printf("Error: --model is required\n"); return 1; }
        if (dataFile.empty()) { printf("Error: --data is required\n"); return 1; }
        if (saveFile.empty()) { printf("Error: --save is required\n"); return 1; }

        TMultiLayerPerceptronCUDA* mlp = TMultiLayerPerceptronCUDA::Load(modelFile.c_str());
        if (!mlp) { printf("Error: Failed to load model: %s\n", modelFile.c_str()); return 1; }

        if (lrOverride) mlp->LearningRate = learningRate;
        mlp->EnableLRDecay = lrDecay;
        mlp->LRDecayRate = lrDecayRate;
        mlp->LRDecayEpochs = lrDecayEpochs;
        mlp->EnableEarlyStopping = earlyStop;
        mlp->EarlyStoppingPatience = patience;

        std::vector<DataPoint> data = LoadDataCSV(dataFile.c_str(), mlp->GetInputSize(), mlp->GetOutputSize());
        if (data.empty()) { printf("Error: No valid data loaded\n"); delete mlp; return 1; }

        printf("Using GPU: %s\n", prop.name);
        printf("Loaded %zu training samples\n", data.size());
        if (batchSize > 1)
            printf("Note: Batch size %d specified (online training used)\n", batchSize);

        if (normalize) {
            NormalizeData(data);
            printf("Data normalized\n");
        }

        double* output = new double[mlp->GetOutputSize()];

        for (int epoch = 1; epoch <= epochs; epoch++) {
            ShuffleData(data);

            for (auto& dp : data)
                mlp->Train(dp.Input.data(), dp.Target.data());

            if (verbose && (epoch % 10 == 0 || epoch == 1)) {
                double totalLoss = 0;
                for (auto& dp : data) {
                    mlp->Predict(dp.Input.data(), output);
                    totalLoss += mlp->ComputeLoss(output, dp.Target.data());
                }
                printf("Epoch %d/%d - Loss: %.6f\n", epoch, epochs, totalLoss / data.size());
            }
        }

        double totalLoss = 0;
        for (auto& dp : data) {
            mlp->Predict(dp.Input.data(), output);
            totalLoss += mlp->ComputeLoss(output, dp.Target.data());
        }
        printf("Final loss: %.6f\n", totalLoss / data.size());

        delete[] output;

        mlp->Save(saveFile.c_str());
        printf("Model saved to: %s\n", saveFile.c_str());

        delete mlp;
    }
    else if (command == cmdPredict) {
        if (modelFile.empty()) { printf("Error: --model is required\n"); return 1; }
        if (inputValues.empty()) { printf("Error: --input is required\n"); return 1; }

        TMultiLayerPerceptronCUDA* mlp = TMultiLayerPerceptronCUDA::Load(modelFile.c_str());
        if (!mlp) { printf("Error: Failed to load model\n"); return 1; }

        if ((int)inputValues.size() != mlp->GetInputSize()) {
            printf("Error: Expected %d input values, got %zu\n", mlp->GetInputSize(), inputValues.size());
            delete mlp;
            return 1;
        }

        double* output = new double[mlp->GetOutputSize()];
        mlp->Predict(inputValues.data(), output);

        printf("Input: ");
        for (size_t i = 0; i < inputValues.size(); i++)
            printf("%s%.4f", i > 0 ? ", " : "", inputValues[i]);
        printf("\n");

        printf("Output: ");
        for (int i = 0; i < mlp->GetOutputSize(); i++)
            printf("%s%.6f", i > 0 ? ", " : "", output[i]);
        printf("\n");

        if (mlp->GetOutputSize() > 1)
            printf("Max index: %d\n", MaxIndex(output, mlp->GetOutputSize()));

        delete[] output;
        delete mlp;
    }
    else if (command == cmdBatchPredict) {
        if (modelFile.empty()) { printf("Error: --model is required\n"); return 1; }
        if (inputValues.empty()) { printf("Error: --input is required\n"); return 1; }

        TMultiLayerPerceptronCUDA* mlp = TMultiLayerPerceptronCUDA::Load(modelFile.c_str());
        if (!mlp) { printf("Error: Failed to load model\n"); return 1; }

        if ((int)inputValues.size() != mlp->GetInputSize()) {
            printf("Error: Expected %d input values, got %zu\n", mlp->GetInputSize(), inputValues.size());
            delete mlp;
            return 1;
        }

        double* output = new double[mlp->GetOutputSize()];
        mlp->Predict(inputValues.data(), output);

        printf("Input: ");
        for (size_t i = 0; i < inputValues.size(); i++)
            printf("%s%.4f", i > 0 ? ", " : "", inputValues[i]);
        printf("\n");

        printf("Output: ");
        for (int i = 0; i < mlp->GetOutputSize(); i++)
            printf("%s%.6f", i > 0 ? ", " : "", output[i]);
        printf("\n");

        if (mlp->GetOutputSize() > 1)
            printf("Max index: %d\n", MaxIndex(output, mlp->GetOutputSize()));

        delete[] output;
        delete mlp;
    }
    else if (command == cmdInfo) {
        if (modelFile.empty()) { printf("Error: --model is required\n"); return 1; }

        TMultiLayerPerceptronCUDA* mlp = TMultiLayerPerceptronCUDA::Load(modelFile.c_str());
        if (!mlp) { printf("Error: Failed to load model\n"); return 1; }

        printf("MLP Model Information (CUDA)\n");
        printf("============================\n");
        printf("GPU: %s\n", prop.name);
        printf("Input size: %d\n", mlp->GetInputSize());
        printf("Output size: %d\n", mlp->GetOutputSize());
        printf("Hidden layers: %d\n", mlp->GetHiddenLayerCount());
        printf("Hidden sizes: ");
        const std::vector<int>& hs = mlp->GetHiddenSizes();
        for (size_t i = 0; i < hs.size(); i++)
            printf("%s%d", i > 0 ? "," : "", hs[i]);
        printf("\n");
        printf("Layer sizes: %d", mlp->GetInputSize());
        for (int h : mlp->GetHiddenSizes())
            printf(" -> %d", h);
        printf(" -> %d\n", mlp->GetOutputSize());
        printf("\n");
        printf("Hyperparameters:\n");
        printf("  Learning rate: %.6f\n", mlp->LearningRate);
        printf("  Optimizer: %s\n", OptimizerToStr(mlp->Optimizer));
        printf("  Hidden activation: %s\n", ActivationToStr(mlp->HiddenActivation));
        printf("  Output activation: %s\n", ActivationToStr(mlp->OutputActivation));
        printf("  Dropout rate: %.4f\n", mlp->DropoutRate);
        printf("  L2 lambda: %.6f\n", mlp->L2Lambda);
        printf("  Beta1: %.4f\n", mlp->Beta1);
        printf("  Beta2: %.4f\n", mlp->Beta2);
        printf("  Timestep: %d\n", mlp->Timestep);
        printf("\n");
        printf("Total layers: %d\n", mlp->GetNumLayers());
        for (int i = 0; i < mlp->GetNumLayers(); i++)
            printf("  Layer %d: %d neurons\n", i, mlp->GetLayerSize(i));

        delete mlp;
    }
    // ===== FACADE COMMANDS =====
    else if (command == cmdGetWeight) {
        if (modelFile.empty()) { printf("Error: --model is required\n"); return 1; }
        if (layerIdx < 0) { printf("Error: --layer is required\n"); return 1; }
        if (neuronIdx < 0) { printf("Error: --neuron is required\n"); return 1; }
        if (weightIdx < 0) { printf("Error: --weight is required\n"); return 1; }

        TMultiLayerPerceptronCUDA* mlp = TMultiLayerPerceptronCUDA::Load(modelFile.c_str());
        if (!mlp) { printf("Error: Failed to load model\n"); return 1; }

        double w = mlp->GetNeuronWeight(layerIdx, neuronIdx, weightIdx);
        printf("Weight[layer=%d, neuron=%d, weight=%d] = %.10f\n", layerIdx, neuronIdx, weightIdx, w);

        delete mlp;
    }
    else if (command == cmdSetWeight) {
        if (modelFile.empty()) { printf("Error: --model is required\n"); return 1; }
        if (layerIdx < 0) { printf("Error: --layer is required\n"); return 1; }
        if (neuronIdx < 0) { printf("Error: --neuron is required\n"); return 1; }
        if (weightIdx < 0) { printf("Error: --weight is required\n"); return 1; }
        if (!hasSetValue) { printf("Error: --value is required\n"); return 1; }
        if (saveFile.empty()) { printf("Error: --save is required\n"); return 1; }

        TMultiLayerPerceptronCUDA* mlp = TMultiLayerPerceptronCUDA::Load(modelFile.c_str());
        if (!mlp) { printf("Error: Failed to load model\n"); return 1; }

        double oldVal = mlp->GetNeuronWeight(layerIdx, neuronIdx, weightIdx);
        mlp->SetNeuronWeight(layerIdx, neuronIdx, weightIdx, setValue);
        mlp->Save(saveFile.c_str());
        printf("Weight[layer=%d, neuron=%d, weight=%d]: %.10f -> %.10f\n", 
               layerIdx, neuronIdx, weightIdx, oldVal, setValue);
        printf("Saved to: %s\n", saveFile.c_str());

        delete mlp;
    }
    else if (command == cmdGetWeights) {
        if (modelFile.empty()) { printf("Error: --model is required\n"); return 1; }
        if (layerIdx < 0) { printf("Error: --layer is required\n"); return 1; }
        if (neuronIdx < 0) { printf("Error: --neuron is required\n"); return 1; }

        TMultiLayerPerceptronCUDA* mlp = TMultiLayerPerceptronCUDA::Load(modelFile.c_str());
        if (!mlp) { printf("Error: Failed to load model\n"); return 1; }

        std::vector<double> weights = mlp->GetNeuronWeights(layerIdx, neuronIdx);
        printf("Weights[layer=%d, neuron=%d] (%zu weights):\n", layerIdx, neuronIdx, weights.size());
        for (size_t i = 0; i < weights.size(); i++)
            printf("  [%zu] = %.10f\n", i, weights[i]);

        delete mlp;
    }
    else if (command == cmdGetBias) {
        if (modelFile.empty()) { printf("Error: --model is required\n"); return 1; }
        if (layerIdx < 0) { printf("Error: --layer is required\n"); return 1; }
        if (neuronIdx < 0) { printf("Error: --neuron is required\n"); return 1; }

        TMultiLayerPerceptronCUDA* mlp = TMultiLayerPerceptronCUDA::Load(modelFile.c_str());
        if (!mlp) { printf("Error: Failed to load model\n"); return 1; }

        double b = mlp->GetNeuronBias(layerIdx, neuronIdx);
        printf("Bias[layer=%d, neuron=%d] = %.10f\n", layerIdx, neuronIdx, b);

        delete mlp;
    }
    else if (command == cmdSetBias) {
        if (modelFile.empty()) { printf("Error: --model is required\n"); return 1; }
        if (layerIdx < 0) { printf("Error: --layer is required\n"); return 1; }
        if (neuronIdx < 0) { printf("Error: --neuron is required\n"); return 1; }
        if (!hasSetValue) { printf("Error: --value is required\n"); return 1; }
        if (saveFile.empty()) { printf("Error: --save is required\n"); return 1; }

        TMultiLayerPerceptronCUDA* mlp = TMultiLayerPerceptronCUDA::Load(modelFile.c_str());
        if (!mlp) { printf("Error: Failed to load model\n"); return 1; }

        double oldVal = mlp->GetNeuronBias(layerIdx, neuronIdx);
        mlp->SetNeuronBias(layerIdx, neuronIdx, setValue);
        mlp->Save(saveFile.c_str());
        printf("Bias[layer=%d, neuron=%d]: %.10f -> %.10f\n", layerIdx, neuronIdx, oldVal, setValue);
        printf("Saved to: %s\n", saveFile.c_str());

        delete mlp;
    }
    else if (command == cmdGetOutput) {
        if (modelFile.empty()) { printf("Error: --model is required\n"); return 1; }
        if (layerIdx < 0) { printf("Error: --layer is required\n"); return 1; }

        TMultiLayerPerceptronCUDA* mlp = TMultiLayerPerceptronCUDA::Load(modelFile.c_str());
        if (!mlp) { printf("Error: Failed to load model\n"); return 1; }

        if (!runInput.empty()) {
            if ((int)runInput.size() != mlp->GetInputSize()) {
                printf("Error: --run-input needs %d values\n", mlp->GetInputSize());
                delete mlp;
                return 1;
            }
            double* output = new double[mlp->GetOutputSize()];
            mlp->Predict(runInput.data(), output);
            delete[] output;
        }

        std::vector<double> outputs = mlp->GetLayerOutputs(layerIdx);
        printf("Outputs[layer=%d] (%zu neurons):\n", layerIdx, outputs.size());
        for (size_t i = 0; i < outputs.size(); i++)
            printf("  [%zu] = %.10f\n", i, outputs[i]);

        delete mlp;
    }
    else if (command == cmdGetAllOutputs) {
        if (modelFile.empty()) { printf("Error: --model is required\n"); return 1; }

        TMultiLayerPerceptronCUDA* mlp = TMultiLayerPerceptronCUDA::Load(modelFile.c_str());
        if (!mlp) { printf("Error: Failed to load model\n"); return 1; }

        if (!runInput.empty()) {
            if ((int)runInput.size() != mlp->GetInputSize()) {
                printf("Error: --run-input needs %d values\n", mlp->GetInputSize());
                delete mlp;
                return 1;
            }
            double* output = new double[mlp->GetOutputSize()];
            mlp->Predict(runInput.data(), output);
            delete[] output;
        }

        for (int l = 0; l < mlp->GetNumLayers(); l++) {
            std::vector<double> outputs = mlp->GetLayerOutputs(l);
            printf("Layer %d (%zu neurons):\n", l, outputs.size());
            for (size_t i = 0; i < outputs.size(); i++)
                printf("  [%zu] = %.6f\n", i, outputs[i]);
        }

        delete mlp;
    }
    else if (command == cmdGetError) {
        if (modelFile.empty()) { printf("Error: --model is required\n"); return 1; }
        if (layerIdx < 0) { printf("Error: --layer is required\n"); return 1; }

        TMultiLayerPerceptronCUDA* mlp = TMultiLayerPerceptronCUDA::Load(modelFile.c_str());
        if (!mlp) { printf("Error: Failed to load model\n"); return 1; }

        std::vector<double> errors = mlp->GetLayerErrors(layerIdx);
        printf("Errors[layer=%d] (%zu neurons):\n", layerIdx, errors.size());
        for (size_t i = 0; i < errors.size(); i++)
            printf("  [%zu] = %.10f\n", i, errors[i]);

        delete mlp;
    }
    else if (command == cmdLayerInfo) {
        if (modelFile.empty()) { printf("Error: --model is required\n"); return 1; }
        if (layerIdx < 0) { printf("Error: --layer is required\n"); return 1; }

        TMultiLayerPerceptronCUDA* mlp = TMultiLayerPerceptronCUDA::Load(modelFile.c_str());
        if (!mlp) { printf("Error: Failed to load model\n"); return 1; }

        int numNeurons = mlp->GetLayerSize(layerIdx);
        int numWeights = (layerIdx > 0 && numNeurons > 0) ? mlp->GetWeightsPerNeuron(layerIdx, 0) : 0;
        
        printf("Layer %d Information:\n", layerIdx);
        printf("  Neurons: %d\n", numNeurons);
        printf("  Weights per neuron: %d\n", numWeights);
        printf("  Activation: %s\n", ActivationToStr(mlp->GetLayerActivation(layerIdx)));
        printf("\n");

        if (!runInput.empty()) {
            if ((int)runInput.size() != mlp->GetInputSize()) {
                printf("Warning: --run-input needs %d values, skipping outputs\n", mlp->GetInputSize());
            } else {
                double* output = new double[mlp->GetOutputSize()];
                mlp->Predict(runInput.data(), output);
                delete[] output;

                std::vector<double> outputs = mlp->GetLayerOutputs(layerIdx);
                printf("  Neuron outputs (after prediction):\n");
                for (size_t i = 0; i < outputs.size(); i++)
                    printf("    [%zu] = %.6f\n", i, outputs[i]);
            }
        }

        printf("\n  Neuron details:\n");
        for (int n = 0; n < numNeurons && n < 10; n++) {
            double bias = mlp->GetNeuronBias(layerIdx, n);
            printf("    Neuron %d: bias=%.6f\n", n, bias);
        }
        if (numNeurons > 10) printf("    ... (%d more neurons)\n", numNeurons - 10);

        delete mlp;
    }
    else if (command == cmdHistogram) {
        if (modelFile.empty()) { printf("Error: --model is required\n"); return 1; }
        if (layerIdx < 0) { printf("Error: --layer is required\n"); return 1; }

        TMultiLayerPerceptronCUDA* mlp = TMultiLayerPerceptronCUDA::Load(modelFile.c_str());
        if (!mlp) { printf("Error: Failed to load model\n"); return 1; }

        if (!runInput.empty()) {
            if ((int)runInput.size() != mlp->GetInputSize()) {
                printf("Error: --run-input needs %d values\n", mlp->GetInputSize());
                delete mlp;
                return 1;
            }
            double* output = new double[mlp->GetOutputSize()];
            mlp->Predict(runInput.data(), output);
            delete[] output;
        }

        std::vector<int> histogram;
        if (histogramType == "gradient" || histogramType == "error") {
            histogram = mlp->GetGradientHistogram(layerIdx, histogramBins);
            printf("Gradient Histogram [layer=%d] (%d bins):\n", layerIdx, histogramBins);
        } else {
            histogram = mlp->GetActivationHistogram(layerIdx, histogramBins);
            printf("Activation Histogram [layer=%d] (%d bins):\n", layerIdx, histogramBins);
        }

        int maxCount = 0;
        for (int c : histogram) if (c > maxCount) maxCount = c;

        for (int i = 0; i < (int)histogram.size(); i++) {
            int barLen = (maxCount > 0) ? (histogram[i] * 40 / maxCount) : 0;
            printf("  [%2d] %4d |", i, histogram[i]);
            for (int j = 0; j < barLen; j++) printf("#");
            printf("\n");
        }

        delete mlp;
    }
    else if (command == cmdGetOptimizer) {
        if (modelFile.empty()) { printf("Error: --model is required\n"); return 1; }
        if (layerIdx < 0) { printf("Error: --layer is required\n"); return 1; }
        if (neuronIdx < 0) { printf("Error: --neuron is required\n"); return 1; }

        TMultiLayerPerceptronCUDA* mlp = TMultiLayerPerceptronCUDA::Load(modelFile.c_str());
        if (!mlp) { printf("Error: Failed to load model\n"); return 1; }

        printf("Optimizer state [layer=%d, neuron=%d]:\n", layerIdx, neuronIdx);
        printf("  Optimizer: %s\n", OptimizerToStr(mlp->Optimizer));
        printf("  Timestep: %d\n", mlp->Timestep);
        printf("  Bias M: %.10f\n", mlp->GetBiasM(layerIdx, neuronIdx));
        printf("  Bias V: %.10f\n", mlp->GetBiasV(layerIdx, neuronIdx));

        if (weightIdx >= 0) {
            printf("\n  Weight[%d]:\n", weightIdx);
            printf("    M: %.10f\n", mlp->GetWeightM(layerIdx, neuronIdx, weightIdx));
            printf("    V: %.10f\n", mlp->GetWeightV(layerIdx, neuronIdx, weightIdx));
        } else {
            int numWeights = mlp->GetWeightsPerNeuron(layerIdx, neuronIdx);
            printf("\n  All weights (%d):\n", numWeights);
            for (int w = 0; w < numWeights && w < 10; w++) {
                printf("    [%d] M=%.6f V=%.6f\n", w, 
                       mlp->GetWeightM(layerIdx, neuronIdx, w),
                       mlp->GetWeightV(layerIdx, neuronIdx, w));
            }
            if (numWeights > 10) printf("    ... (%d more)\n", numWeights - 10);
        }

        delete mlp;
    }
    else if (command == cmdExportONNX) {
        if (modelFile.empty()) { printf("Error: --model is required\n"); return 1; }
        if (saveFile.empty()) { printf("Error: --save is required\n"); return 1; }

        TMultiLayerPerceptronCUDA* mlp = TMultiLayerPerceptronCUDA::Load(modelFile.c_str());
        if (!mlp) { printf("Error: Failed to load model\n"); return 1; }

        if (mlp->ExportToONNX(saveFile.c_str())) {
            printf("Model exported to ONNX: %s\n", saveFile.c_str());
        } else {
            printf("Error: Failed to export model to ONNX\n");
            delete mlp;
            return 1;
        }

        delete mlp;
    }
    else if (command == cmdImportONNX) {
        if (onnxFile.empty() && modelFile.empty()) { printf("Error: --onnx or --model is required\n"); return 1; }
        if (saveFile.empty()) { printf("Error: --save is required\n"); return 1; }

        std::string inputFile = onnxFile.empty() ? modelFile : onnxFile;
        TMultiLayerPerceptronCUDA* mlp = TMultiLayerPerceptronCUDA::ImportFromONNX(inputFile.c_str());
        if (!mlp) {
            printf("Error: Failed to import model from ONNX\n");
            return 1;
        }

        mlp->Save(saveFile.c_str());
        printf("ONNX model imported and saved to: %s\n", saveFile.c_str());

        delete mlp;
    }
    else if (command == cmdFeatureImportance) {
        if (modelFile.empty()) { printf("Error: --model is required\n"); return 1; }

        TMultiLayerPerceptronCUDA* mlp = TMultiLayerPerceptronCUDA::Load(modelFile.c_str());
        if (!mlp) { printf("Error: Failed to load model\n"); return 1; }

        std::vector<std::pair<int, double>> importance = mlp->GetFeatureImportance();
        if (importance.empty()) {
            printf("Error: Could not compute feature importance\n");
            delete mlp;
            return 1;
        }

        double total = 0;
        for (const auto& p : importance) total += p.second;

        printf("Feature Importance (ranked by weight magnitude):\n");
        printf("================================================\n");
        for (const auto& p : importance) {
            double pct = (total > 0) ? (p.second / total * 100.0) : 0.0;
            printf("  Input %d: %.2f%%\n", p.first, pct);
        }

        delete mlp;
    }

    return 0;
}
