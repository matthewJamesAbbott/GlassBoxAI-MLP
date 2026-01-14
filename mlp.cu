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
#include <iostream>
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

using namespace std;

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << " - " \
                 << cudaGetErrorString(err) << endl; \
            exit(1); \
        } \
    } while(0)

const double EPSILON = 1e-15;
const int BLOCK_SIZE = 256;
const string MODEL_MAGIC = "MLPBKND01";

enum TActivationType { atSigmoid = 0, atTanh = 1, atReLU = 2, atSoftmax = 3, atLinear = 4 };
enum TOptimizerType { otSGD = 0, otAdam = 1, otRMSProp = 2 };
enum TCommand { cmdNone, cmdCreate, cmdTrain, cmdPredict, cmdInfo, cmdHelp };

typedef vector<double> Darray;
typedef vector<double> TDoubleArray;
typedef vector<int> TIntArray;

struct TDataPoint {
    Darray Input;
    Darray Target;
};
typedef vector<TDataPoint> TDataPointArray;

__device__ double d_clip(double v, double maxVal) {
    if (v > maxVal) return maxVal;
    else if (v < -maxVal) return -maxVal;
    else return v;
}

__device__ double d_Sigmoid(double x) {
    double clamped = fmax(-500.0, fmin(500.0, x));
    return 1.0 / (1.0 + exp(-clamped));
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
    return (x > 0) ? 1 : 0;
}

__device__ double d_ApplyActivation(double x, TActivationType ActType) {
    switch (ActType) {
        case atSigmoid: return d_Sigmoid(x);
        case atTanh: return d_TanhActivation(x);
        case atReLU: return d_ReLU(x);
        case atLinear: return x;
        default: return d_Sigmoid(x);
    }
}

__device__ double d_ApplyActivationDerivative(double x, TActivationType ActType) {
    switch (ActType) {
        case atSigmoid: return d_DSigmoid(x);
        case atTanh: return d_DTanh(x);
        case atReLU: return d_DReLU(x);
        case atLinear: return 1.0;
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

__global__ void UpdateWeightsSGDKernel(LayerData layer, double* prevOutputs, double learningRate, double l2Lambda, double clipVal) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < layer.NumNeurons) {
        for (int j = 0; j < layer.NumInputs; j++) {
            double gradient = layer.Errors[i] * prevOutputs[j];
            if (l2Lambda > 0)
                gradient = gradient - l2Lambda * layer.Weights[i * layer.NumInputs + j];
            gradient = d_clip(gradient, clipVal);
            layer.Weights[i * layer.NumInputs + j] += learningRate * gradient;
        }
        double biasGrad = d_clip(layer.Errors[i], clipVal);
        layer.Biases[i] += learningRate * biasGrad;
    }
}

__global__ void UpdateWeightsAdamKernel(LayerData layer, double* prevOutputs, 
                                         double learningRate, double l2Lambda,
                                         double beta1, double beta2, int timestep, double clipVal) {
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
            gradient = d_clip(gradient, clipVal);

            layer.M[idx] = beta1 * layer.M[idx] + (1 - beta1) * gradient;
            layer.V[idx] = beta2 * layer.V[idx] + (1 - beta2) * gradient * gradient;

            double mHat = layer.M[idx] / (1 - beta1_t);
            double vHat = layer.V[idx] / (1 - beta2_t);

            layer.Weights[idx] -= learningRate * mHat / (sqrt(vHat) + eps);
        }

        double gradient = d_clip(-layer.Errors[i], clipVal);
        layer.MBias[i] = beta1 * layer.MBias[i] + (1 - beta1) * gradient;
        layer.VBias[i] = beta2 * layer.VBias[i] + (1 - beta2) * gradient * gradient;
        double mHat = layer.MBias[i] / (1 - beta1_t);
        double vHat = layer.VBias[i] / (1 - beta2_t);
        layer.Biases[i] -= learningRate * mHat / (sqrt(vHat) + eps);
    }
}

__global__ void UpdateWeightsRMSPropKernel(LayerData layer, double* prevOutputs,
                                             double learningRate, double l2Lambda, double clipVal) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < layer.NumNeurons) {
        double eps = 1e-8;
        double decay = 0.9;

        for (int j = 0; j < layer.NumInputs; j++) {
            int idx = i * layer.NumInputs + j;
            double gradient = -layer.Errors[i] * prevOutputs[j];
            if (l2Lambda > 0)
                gradient += l2Lambda * layer.Weights[idx];
            gradient = d_clip(gradient, clipVal);

            layer.V[idx] = decay * layer.V[idx] + (1 - decay) * gradient * gradient;
            layer.Weights[idx] -= learningRate * gradient / (sqrt(layer.V[idx]) + eps);
        }

        double gradient = d_clip(-layer.Errors[i], clipVal);
        layer.VBias[i] = decay * layer.VBias[i] + (1 - decay) * gradient * gradient;
        layer.Biases[i] -= learningRate * gradient / (sqrt(layer.VBias[i]) + eps);
    }
}

string ActivationToStr(TActivationType act) {
    switch (act) {
        case atSigmoid: return "sigmoid";
        case atTanh: return "tanh";
        case atReLU: return "relu";
        case atSoftmax: return "softmax";
        case atLinear: return "linear";
        default: return "unknown";
    }
}

string OptimizerToStr(TOptimizerType opt) {
    switch (opt) {
        case otSGD: return "sgd";
        case otAdam: return "adam";
        case otRMSProp: return "rmsprop";
        default: return "unknown";
    }
}

TActivationType ParseActivation(const string& s) {
    string lower = s;
    transform(lower.begin(), lower.end(), lower.begin(), ::tolower);
    
    if (lower == "tanh") return atTanh;
    if (lower == "relu") return atReLU;
    if (lower == "softmax") return atSoftmax;
    if (lower == "sigmoid") return atSigmoid;
    if (lower == "linear") return atLinear;
    throw invalid_argument("Error: Invalid activation function: " + s);
}

TOptimizerType ParseOptimizer(const string& s) {
    string lower = s;
    transform(lower.begin(), lower.end(), lower.begin(), ::tolower);
    
    if (lower == "adam") return otAdam;
    if (lower == "rmsprop") return otRMSProp;
    return otSGD;
}

void ParseIntArrayHelper(const string& s, TIntArray& result) {
    result.clear();
    stringstream ss(s);
    string item;
    while (getline(ss, item, ',')) {
        result.push_back(stoi(item));
    }
}

void ParseDoubleArrayHelper(const string& s, TDoubleArray& result) {
    result.clear();
    stringstream ss(s);
    string item;
    while (getline(ss, item, ',')) {
        result.push_back(stod(item));
    }
}

int MaxIndex(const Darray& arr) {
    int result = 0;
    for (size_t i = 1; i < arr.size(); i++)
        if (arr[i] > arr[result]) result = i;
    return result;
}

class TMultiLayerPerceptronCUDA {
private:
    LayerData* d_Layers;
    LayerData* h_Layers;
    int NumLayers;
    int FInputSize;
    int FOutputSize;
    vector<int> FHiddenSizes;
    bool FIsTraining;
    curandState* d_RandStates;
    int MaxNeurons;

    double* d_Target;
    double* d_SoftmaxSums;

    void AllocateLayer(LayerData& layer, int numNeurons, int numInputs, TActivationType actType) {
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

    TMultiLayerPerceptronCUDA(int InputSize, const vector<int>& HiddenSizes, int OutputSize,
                               TActivationType HiddenAct = atSigmoid, TActivationType OutputAct = atSigmoid) {
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

        FInputSize = InputSize;
        FOutputSize = OutputSize;
        FHiddenSizes = HiddenSizes;

        NumLayers = HiddenSizes.size() + 2;
        h_Layers = new LayerData[NumLayers];
        memset(h_Layers, 0, NumLayers * sizeof(LayerData));
        CUDA_CHECK(cudaMalloc(&d_Layers, NumLayers * sizeof(LayerData)));

        AllocateLayer(h_Layers[0], InputSize + 1, InputSize, atSigmoid);

        MaxNeurons = InputSize + 1;
        int numInputs = InputSize;
        for (size_t i = 0; i < HiddenSizes.size(); i++) {
            AllocateLayer(h_Layers[i + 1], HiddenSizes[i] + 1, numInputs + 1, HiddenActivation);
            if (HiddenSizes[i] + 1 > MaxNeurons) MaxNeurons = HiddenSizes[i] + 1;
            numInputs = HiddenSizes[i];
        }

        AllocateLayer(h_Layers[NumLayers - 1], OutputSize, numInputs + 1, OutputActivation);
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
                    UpdateWeightsSGDKernel<<<blocks, BLOCK_SIZE>>>(layer, prevLayer.Outputs, LearningRate, L2Lambda, 5.0);
                    break;
                case otAdam:
                    UpdateWeightsAdamKernel<<<blocks, BLOCK_SIZE>>>(layer, prevLayer.Outputs, 
                                                                     LearningRate, L2Lambda, Beta1, Beta2, Timestep, 5.0);
                    break;
                case otRMSProp:
                    UpdateWeightsRMSPropKernel<<<blocks, BLOCK_SIZE>>>(layer, prevLayer.Outputs, LearningRate, L2Lambda, 5.0);
                    break;
            }
        }
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    Darray Predict(const Darray& Input) {
        FIsTraining = false;

        double* h_input = new double[FInputSize + 1];
        for (int i = 0; i < FInputSize; i++) h_input[i] = Input[i];
        h_input[FInputSize] = 1.0;
        CUDA_CHECK(cudaMemcpy(h_Layers[0].Outputs, h_input, (FInputSize + 1) * sizeof(double), cudaMemcpyHostToDevice));
        delete[] h_input;

        FeedForward();

        Darray result(FOutputSize);
        CUDA_CHECK(cudaMemcpy(result.data(), h_Layers[NumLayers - 1].Outputs, FOutputSize * sizeof(double), cudaMemcpyDeviceToHost));

        FIsTraining = true;
        return result;
    }

    void Train(const Darray& Input, const Darray& Target) {
        FIsTraining = true;

        double* h_input = new double[FInputSize + 1];
        for (int i = 0; i < FInputSize; i++) h_input[i] = Input[i];
        h_input[FInputSize] = 1.0;
        CUDA_CHECK(cudaMemcpy(h_Layers[0].Outputs, h_input, (FInputSize + 1) * sizeof(double), cudaMemcpyHostToDevice));
        delete[] h_input;

        CUDA_CHECK(cudaMemcpy(d_Target, Target.data(), FOutputSize * sizeof(double), cudaMemcpyHostToDevice));

        FeedForward();
        BackPropagate();
        UpdateWeights();
    }

    double ComputeLoss(const Darray& Predicted, const Darray& Target) {
        double loss = 0.0;
        for (size_t i = 0; i < Predicted.size(); i++) {
            double diff = Predicted[i] - Target[i];
            loss += diff * diff;
        }
        return loss / Predicted.size();
    }

    int GetOutputSize() const { return FOutputSize; }
    int GetInputSize() const { return FInputSize; }
    int GetHiddenLayerCount() const { return FHiddenSizes.size(); }
    const vector<int>& GetHiddenSizes() const { return FHiddenSizes; }
    int GetNumLayers() const { return NumLayers; }

    int GetLayerSize(int layerIdx) const {
        if (layerIdx < 0 || layerIdx >= NumLayers) return 0;
        return h_Layers[layerIdx].NumNeurons;
    }

    string Array1DToJSON(const Darray& arr) {
        stringstream ss;
        ss << "[";
        for (size_t i = 0; i < arr.size(); i++) {
            if (i > 0) ss << ",";
            ss << fixed << setprecision(10) << arr[i];
        }
        ss << "]";
        return ss.str();
    }

    void SaveModelToJSON(const string& filename) {
        ofstream f(filename);
        
        f << "{" << endl;
        f << "  \"magic\": \"" << MODEL_MAGIC << "\"," << endl;
        f << "  \"input_size\": " << FInputSize << "," << endl;
        f << "  \"output_size\": " << FOutputSize << "," << endl;
        f << "  \"hidden_sizes\": [";
        for (size_t i = 0; i < FHiddenSizes.size(); i++) {
            if (i > 0) f << ",";
            f << FHiddenSizes[i];
        }
        f << "]," << endl;
        f << fixed << setprecision(6);
        f << "  \"learning_rate\": " << LearningRate << "," << endl;
        f << "  \"optimizer\": " << (int)Optimizer << "," << endl;
        f << "  \"hidden_activation\": " << (int)HiddenActivation << "," << endl;
        f << "  \"output_activation\": " << (int)OutputActivation << "," << endl;
        f << setprecision(4);
        f << "  \"dropout_rate\": " << DropoutRate << "," << endl;
        f << setprecision(6);
        f << "  \"l2_lambda\": " << L2Lambda << "," << endl;
        f << "  \"beta1\": " << Beta1 << "," << endl;
        f << "  \"beta2\": " << Beta2 << "," << endl;
        
        f << "  \"input_layer\": {" << endl;
        f << "    \"neuron_count\": " << FInputSize << endl;
        f << "  }," << endl;
        
        f << "  \"hidden_layers\": [" << endl;
        for (size_t h = 0; h < FHiddenSizes.size(); h++) {
            LayerData& layer = h_Layers[h + 1];
            int numNeurons = FHiddenSizes[h];
            int numInputs = (h == 0) ? FInputSize : FHiddenSizes[h-1];
            
            double* h_weights = new double[layer.NumNeurons * layer.NumInputs];
            double* h_biases = new double[layer.NumNeurons];
            CUDA_CHECK(cudaMemcpy(h_weights, layer.Weights, layer.NumNeurons * layer.NumInputs * sizeof(double), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(h_biases, layer.Biases, layer.NumNeurons * sizeof(double), cudaMemcpyDeviceToHost));
            
            f << "    {" << endl;
            f << "      \"neuron_count\": " << numNeurons << "," << endl;
            f << "      \"neurons\": [" << endl;
            for (int j = 0; j < numNeurons; j++) {
                f << "        {" << endl;
                f << "          \"weights\": [";
                for (int w = 0; w < numInputs; w++) {
                    if (w > 0) f << ",";
                    f << fixed << setprecision(10) << h_weights[j * layer.NumInputs + w];
                }
                f << "]," << endl;
                f << setprecision(10);
                f << "          \"bias\": " << h_biases[j] << endl;
                f << "        }";
                if (j < numNeurons - 1) f << ",";
                f << endl;
            }
            f << "      ]," << endl;
            f << "      \"biases\": [";
            for (int j = 0; j < numNeurons; j++) {
                if (j > 0) f << ",";
                f << fixed << setprecision(10) << h_biases[j];
            }
            f << "]" << endl;
            f << "    }";
            if (h < FHiddenSizes.size() - 1) f << ",";
            f << endl;
            
            delete[] h_weights;
            delete[] h_biases;
        }
        f << "  ]," << endl;
        
        LayerData& outLayer = h_Layers[NumLayers - 1];
        int outNumInputs = FHiddenSizes.empty() ? FInputSize : FHiddenSizes.back();
        
        double* h_weights = new double[outLayer.NumNeurons * outLayer.NumInputs];
        double* h_biases = new double[outLayer.NumNeurons];
        CUDA_CHECK(cudaMemcpy(h_weights, outLayer.Weights, outLayer.NumNeurons * outLayer.NumInputs * sizeof(double), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_biases, outLayer.Biases, outLayer.NumNeurons * sizeof(double), cudaMemcpyDeviceToHost));
        
        f << "  \"output_layer\": {" << endl;
        f << "    \"neuron_count\": " << FOutputSize << "," << endl;
        f << "    \"neurons\": [" << endl;
        for (int i = 0; i < FOutputSize; i++) {
            f << "      {" << endl;
            f << "        \"weights\": [";
            for (int w = 0; w < outNumInputs; w++) {
                if (w > 0) f << ",";
                f << fixed << setprecision(10) << h_weights[i * outLayer.NumInputs + w];
            }
            f << "]," << endl;
            f << setprecision(10);
            f << "        \"bias\": " << h_biases[i] << endl;
            f << "      }";
            if (i < FOutputSize - 1) f << ",";
            f << endl;
        }
        f << "    ]," << endl;
        f << "    \"biases\": [";
        for (int i = 0; i < FOutputSize; i++) {
            if (i > 0) f << ",";
            f << fixed << setprecision(10) << h_biases[i];
        }
        f << "]" << endl;
        f << "  }" << endl;
        f << "}" << endl;
        
        delete[] h_weights;
        delete[] h_biases;
        f.close();
    }

    void LoadModelFromJSON(const string& filename) {
        ifstream f(filename);
        if (!f.is_open()) {
            throw runtime_error("Could not open file: " + filename);
        }
        
        string content((istreambuf_iterator<char>(f)), istreambuf_iterator<char>());
        f.close();
        
        auto getJsonNumber = [&](const string& key) -> double {
            size_t pos = content.find("\"" + key + "\"");
            if (pos == string::npos) return 0.0;
            size_t colonPos = content.find(":", pos);
            size_t nextComma = content.find(",", colonPos);
            size_t nextBracket = content.find("}", colonPos);
            size_t endPos = (nextComma < nextBracket) ? nextComma : nextBracket;
            string value = content.substr(colonPos + 1, endPos - colonPos - 1);
            value.erase(0, value.find_first_not_of(" \t\n\r"));
            value.erase(value.find_last_not_of(" \t\n\r") + 1);
            try {
                return stod(value);
            } catch (...) {
                return 0.0;
            }
        };
        
        auto getJsonInt = [&](const string& key) -> int {
            return (int)getJsonNumber(key);
        };
        
        auto parseArray = [&](const string& jsonStr) -> vector<double> {
            vector<double> result;
            size_t start = jsonStr.find('[');
            size_t end = jsonStr.find(']');
            if (start == string::npos || end == string::npos) return result;
            
            string arrayContent = jsonStr.substr(start + 1, end - start - 1);
            stringstream ss(arrayContent);
            string token;
            while (getline(ss, token, ',')) {
                token.erase(0, token.find_first_not_of(" \t\n\r"));
                token.erase(token.find_last_not_of(" \t\n\r") + 1);
                if (!token.empty()) {
                    try {
                        result.push_back(stod(token));
                    } catch (...) {}
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
        
        vector<int> newHiddenSizes;
        size_t hiddenArrayPos = content.find("\"hidden_sizes\": [");
        if (hiddenArrayPos != string::npos) {
            size_t startBracket = hiddenArrayPos + 17;
            size_t endPos = content.find("]", startBracket);
            string arrayContent = content.substr(startBracket, endPos - startBracket);
            
            size_t pos = 0;
            while (pos < arrayContent.length()) {
                size_t numStart = arrayContent.find_first_of("0123456789-", pos);
                if (numStart == string::npos) break;
                
                size_t numEnd = arrayContent.find_first_not_of("0123456789", numStart + (arrayContent[numStart] == '-' ? 1 : 0));
                if (numEnd == string::npos) numEnd = arrayContent.length();
                
                try {
                    newHiddenSizes.push_back(stoi(arrayContent.substr(numStart, numEnd - numStart)));
                } catch (...) {}
                pos = numEnd;
            }
        }
        
        for (int i = 0; i < NumLayers; i++)
            FreeLayer(h_Layers[i]);
        delete[] h_Layers;
        cudaFree(d_Layers);
        cudaFree(d_RandStates);
        cudaFree(d_Target);
        cudaFree(d_SoftmaxSums);
        
        FInputSize = newInputSize;
        FOutputSize = newOutputSize;
        FHiddenSizes = newHiddenSizes;
        HiddenActivation = newHiddenAct;
        OutputActivation = newOutputAct;
        LearningRate = newLR;
        Optimizer = newOpt;
        DropoutRate = newDropout;
        L2Lambda = newL2;
        Beta1 = newBeta1;
        Beta2 = newBeta2;
        
        NumLayers = FHiddenSizes.size() + 2;
        h_Layers = new LayerData[NumLayers];
        memset(h_Layers, 0, NumLayers * sizeof(LayerData));
        CUDA_CHECK(cudaMalloc(&d_Layers, NumLayers * sizeof(LayerData)));

        AllocateLayer(h_Layers[0], FInputSize + 1, FInputSize, atSigmoid);

        MaxNeurons = FInputSize + 1;
        int numInputs = FInputSize;
        for (size_t i = 0; i < FHiddenSizes.size(); i++) {
            AllocateLayer(h_Layers[i + 1], FHiddenSizes[i] + 1, numInputs + 1, HiddenActivation);
            if (FHiddenSizes[i] + 1 > MaxNeurons) MaxNeurons = FHiddenSizes[i] + 1;
            numInputs = FHiddenSizes[i];
        }

        AllocateLayer(h_Layers[NumLayers - 1], FOutputSize, numInputs + 1, OutputActivation);
        if (FOutputSize > MaxNeurons) MaxNeurons = FOutputSize;

        CUDA_CHECK(cudaMemcpy(d_Layers, h_Layers, NumLayers * sizeof(LayerData), cudaMemcpyHostToDevice));

        CUDA_CHECK(cudaMalloc(&d_RandStates, MaxNeurons * sizeof(curandState)));
        int blocks = (MaxNeurons + BLOCK_SIZE - 1) / BLOCK_SIZE;
        InitRandStates<<<blocks, BLOCK_SIZE>>>(d_RandStates, time(nullptr), MaxNeurons);
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaMalloc(&d_Target, FOutputSize * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_SoftmaxSums, FOutputSize * sizeof(double)));
        
        size_t searchPos = 0;
        
        size_t hiddenStart = content.find("\"hidden_layers\": [");
        size_t hiddenEnd = content.find("]", hiddenStart);
        if (hiddenStart != string::npos && hiddenEnd != string::npos) {
            searchPos = hiddenStart;
            for (size_t h = 0; h < FHiddenSizes.size(); h++) {
                LayerData& layer = h_Layers[h + 1];
                int layerNeurons = FHiddenSizes[h];
                int layerInputs = (h == 0) ? FInputSize : FHiddenSizes[h-1];
                
                double* h_weights = new double[layer.NumNeurons * layer.NumInputs];
                double* h_biases = new double[layer.NumNeurons];
                memset(h_weights, 0, layer.NumNeurons * layer.NumInputs * sizeof(double));
                memset(h_biases, 0, layer.NumNeurons * sizeof(double));
                
                for (int n = 0; n < layerNeurons; n++) {
                    size_t wPos = content.find("\"weights\": [", searchPos);
                    if (wPos != string::npos && wPos < hiddenEnd) {
                        size_t wEnd = content.find("]", wPos);
                        string weightsStr = content.substr(wPos, wEnd - wPos + 1);
                        vector<double> weights = parseArray(weightsStr);
                        for (size_t w = 0; w < weights.size() && w < (size_t)layerInputs; w++) {
                            h_weights[n * layer.NumInputs + w] = weights[w];
                        }
                        searchPos = wEnd + 1;
                    }
                    
                    size_t bPos = content.find("\"bias\": ", searchPos);
                    if (bPos != string::npos && bPos < hiddenEnd) {
                        size_t bEnd = content.find_first_of(",}", bPos + 8);
                        string biasStr = content.substr(bPos + 8, bEnd - bPos - 8);
                        biasStr.erase(0, biasStr.find_first_not_of(" \t\n\r"));
                        biasStr.erase(biasStr.find_last_not_of(" \t\n\r") + 1);
                        try {
                            h_biases[n] = stod(biasStr);
                        } catch (...) {}
                        searchPos = bEnd + 1;
                    }
                }
                
                CUDA_CHECK(cudaMemcpy(layer.Weights, h_weights, layer.NumNeurons * layer.NumInputs * sizeof(double), cudaMemcpyHostToDevice));
                CUDA_CHECK(cudaMemcpy(layer.Biases, h_biases, layer.NumNeurons * sizeof(double), cudaMemcpyHostToDevice));
                
                delete[] h_weights;
                delete[] h_biases;
            }
        }
        
        searchPos = 0;
        size_t outputStart = content.find("\"output_layer\": {");
        if (outputStart != string::npos) {
            searchPos = outputStart;
            LayerData& layer = h_Layers[NumLayers - 1];
            int layerInputs = FHiddenSizes.empty() ? FInputSize : FHiddenSizes.back();
            
            double* h_weights = new double[layer.NumNeurons * layer.NumInputs];
            double* h_biases = new double[layer.NumNeurons];
            memset(h_weights, 0, layer.NumNeurons * layer.NumInputs * sizeof(double));
            memset(h_biases, 0, layer.NumNeurons * sizeof(double));
            
            for (int n = 0; n < FOutputSize; n++) {
                size_t wPos = content.find("\"weights\": [", searchPos);
                if (wPos != string::npos) {
                    size_t wEnd = content.find("]", wPos);
                    string weightsStr = content.substr(wPos, wEnd - wPos + 1);
                    vector<double> weights = parseArray(weightsStr);
                    for (size_t w = 0; w < weights.size() && w < (size_t)layerInputs; w++) {
                        h_weights[n * layer.NumInputs + w] = weights[w];
                    }
                    searchPos = wEnd + 1;
                }
                
                size_t bPos = content.find("\"bias\": ", searchPos);
                if (bPos != string::npos) {
                    size_t bEnd = content.find_first_of(",}", bPos + 8);
                    string biasStr = content.substr(bPos + 8, bEnd - bPos - 8);
                    biasStr.erase(0, biasStr.find_first_not_of(" \t\n\r"));
                    biasStr.erase(biasStr.find_last_not_of(" \t\n\r") + 1);
                    try {
                        h_biases[n] = stod(biasStr);
                    } catch (...) {}
                    searchPos = bEnd + 1;
                }
            }
            
            CUDA_CHECK(cudaMemcpy(layer.Weights, h_weights, layer.NumNeurons * layer.NumInputs * sizeof(double), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(layer.Biases, h_biases, layer.NumNeurons * sizeof(double), cudaMemcpyHostToDevice));
            
            delete[] h_weights;
            delete[] h_biases;
        }
    }

    void Save(const string& filename) {
        SaveModelToJSON(filename);
    }

    static TMultiLayerPerceptronCUDA* Load(const string& filename) {
        TMultiLayerPerceptronCUDA* mlp = new TMultiLayerPerceptronCUDA(1, {1}, 1, atSigmoid, atSigmoid);
        mlp->LoadModelFromJSON(filename);
        return mlp;
    }
};

void ShuffleData(TDataPointArray& data) {
    for (int i = data.size() - 1; i >= 1; i--) {
        int j = rand() % (i + 1);
        swap(data[i], data[j]);
    }
}

void NormalizeData(TDataPointArray& data) {
    if (data.empty()) return;
    int inputSize = data[0].Input.size();

    vector<double> mins(inputSize), maxs(inputSize);
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

TDataPointArray LoadDataCSV(const string& filename, int inputSize, int outputSize) {
    TDataPointArray data;
    ifstream file(filename);
    if (!file.is_open()) return data;

    string line;
    while (getline(file, line)) {
        if (line.empty()) continue;
        TDoubleArray values;
        ParseDoubleArrayHelper(line, values);
        if ((int)values.size() < inputSize + outputSize) continue;

        TDataPoint dp;
        dp.Input.resize(inputSize);
        dp.Target.resize(outputSize);
        for (int i = 0; i < inputSize; i++) dp.Input[i] = values[i];
        for (int i = 0; i < outputSize; i++) dp.Target[i] = values[inputSize + i];
        data.push_back(dp);
    }
    return data;
}

void PrintUsage() {
    cout << "MLP - Multi-Layer Perceptron (CUDA)" << endl;
    cout << endl;
    cout << "Usage: mlp <command> [options]" << endl;
    cout << endl;
    cout << "Commands:" << endl;
    cout << "  create   Create a new MLP model" << endl;
    cout << "  train    Train an existing model" << endl;
    cout << "  predict  Make predictions with a model" << endl;
    cout << "  info     Display model information" << endl;
    cout << "  help     Show this help message" << endl;
    cout << endl;
    cout << "Create Options:" << endl;
    cout << "  -i, --input=N              Input layer size (required)" << endl;
    cout << "  -H, --hidden=N,N,...       Hidden layer sizes, comma-separated (required)" << endl;
    cout << "  -o, --output=N             Output layer size (required)" << endl;
    cout << "  -s, --save=FILE            Save model file (required, .json)" << endl;
    cout << "  --lr=VALUE                 Learning rate (default: 0.1)" << endl;
    cout << "  --optimizer=TYPE           sgd|adam|rmsprop (default: sgd)" << endl;
    cout << "  --hidden-act=TYPE          sigmoid|tanh|relu|softmax (default: sigmoid)" << endl;
    cout << "  --output-act=TYPE          sigmoid|tanh|relu|softmax (default: sigmoid)" << endl;
    cout << "  --dropout=VALUE            Dropout rate 0-1 (default: 0)" << endl;
    cout << "  --l2=VALUE                 L2 regularization lambda (default: 0)" << endl;
    cout << "  --beta1=VALUE              Adam beta1 parameter (default: 0.9)" << endl;
    cout << "  --beta2=VALUE              Adam beta2 parameter (default: 0.999)" << endl;
    cout << endl;
    cout << "Train Options:" << endl;
    cout << "  -m, --model=FILE           Load model file (required, .json)" << endl;
    cout << "  -d, --data=FILE            Training data CSV file (required)" << endl;
    cout << "  -s, --save=FILE            Save trained model (required, .json)" << endl;
    cout << "  --epochs=N                 Training epochs (default: 100)" << endl;
    cout << "  --batch=N                  Batch size (default: 1)" << endl;
    cout << "  --lr=VALUE                 Override learning rate" << endl;
    cout << "  --lr-decay                 Enable learning rate decay" << endl;
    cout << "  --lr-decay-rate=VALUE      LR decay rate (default: 0.95)" << endl;
    cout << "  --lr-decay-epochs=N        Decay interval in epochs (default: 10)" << endl;
    cout << "  --early-stop               Enable early stopping" << endl;
    cout << "  --patience=N               Early stopping patience (default: 10)" << endl;
    cout << "  --normalize                Normalize training data" << endl;
    cout << "  --verbose                  Print training progress" << endl;
    cout << endl;
    cout << "Predict Options:" << endl;
    cout << "  -m, --model=FILE           Model file (required, .json)" << endl;
    cout << "  -i, --input=v1,v2,...      Input values, comma-separated (required)" << endl;
    cout << endl;
    cout << "Info Options:" << endl;
    cout << "  -m, --model=FILE           Model file (required, .json)" << endl;
    cout << endl;
    cout << "Examples:" << endl;
    cout << "  mlp create -i 2 -H 8 -o 1 -s xor.json" << endl;
    cout << "  mlp create --input=2 --hidden=8,8 --output=1 --save=xor.json" << endl;
    cout << "  mlp train -m xor.json -d data.csv -s xor_trained.json --epochs=1000" << endl;
    cout << "  mlp train --model=xor.json --data=data.csv --epochs=1000 --save=xor_trained.json --verbose" << endl;
    cout << "  mlp predict -m xor_trained.json -i 1,0" << endl;
    cout << "  mlp info -m xor_trained.json" << endl;
    cout << endl;
    cout << "Exit codes:" << endl;
    cout << "  0 - Success" << endl;
    cout << "  1 - Error" << endl;
    cout << "  2 - Usage error" << endl;
}

int main(int argc, char* argv[]) {
    srand((unsigned)time(nullptr));
    
    if (argc < 2) {
        PrintUsage();
        return 2;
    }
    
    string cmdStr = argv[1];
    TCommand command = cmdNone;
    
    if (cmdStr == "create") command = cmdCreate;
    else if (cmdStr == "train") command = cmdTrain;
    else if (cmdStr == "predict") command = cmdPredict;
    else if (cmdStr == "info") command = cmdInfo;
    else if (cmdStr == "help" || cmdStr == "--help" || cmdStr == "-h") command = cmdHelp;
    else {
        cerr << "Error: Unknown command: " << cmdStr << endl;
        PrintUsage();
        return 2;
    }
    
    if (command == cmdHelp) {
        PrintUsage();
        return 0;
    }
    
    int inputSize = 0;
    int outputSize = 0;
    TIntArray hiddenSizes;
    double learningRate = 0.1;
    double dropoutRate = 0.0;
    double l2Lambda = 0.0;
    double beta1 = 0.9;
    double beta2 = 0.999;
    int epochs = 100;
    int batchSize = 1;
    bool lrDecay = false;
    double lrDecayRate = 0.95;
    int lrDecayEpochs = 10;
    bool earlyStop = false;
    int patience = 10;
    bool normalize = false;
    bool verbose = false;
    TActivationType hiddenAct = atSigmoid;
    TActivationType outputAct = atSigmoid;
    TOptimizerType optimizer = otSGD;
    string modelFile = "";
    string saveFile = "";
    string dataFile = "";
    TDoubleArray inputValues;
    
    int i = 2;
    while (i < argc) {
        string arg = argv[i];
        string key, value;
        size_t eqPos = arg.find('=');
        
        if (arg == "--lr-decay") {
            lrDecay = true;
            i++;
        } else if (arg == "--early-stop") {
            earlyStop = true;
            i++;
        } else if (arg == "--normalize") {
            normalize = true;
            i++;
        } else if (arg == "--verbose") {
            verbose = true;
            i++;
        } else if (arg == "-h") {
            PrintUsage();
            return 0;
        } else {
            if (eqPos != string::npos) {
                key = arg.substr(0, eqPos);
                value = arg.substr(eqPos + 1);
                i++;
            } else if (arg[0] == '-') {
                key = arg;
                if (i + 1 < argc) {
                    i++;
                    value = argv[i];
                    if (value[0] == '-') {
                        i--;
                        value = "";
                    }
                    i++;
                } else {
                    cerr << "Error: Option " << key << " requires a value" << endl;
                    return 2;
                }
            } else {
                cerr << "Error: Invalid argument: " << arg << endl;
                return 2;
            }
            
            if (key == "--input" || key == "-i") {
                if (command == cmdPredict) {
                    if (value == "-") {
                        string line;
                        while (getline(cin, line)) {
                            ParseDoubleArrayHelper(line, inputValues);
                        }
                    } else {
                        ParseDoubleArrayHelper(value, inputValues);
                    }
                } else {
                    inputSize = stoi(value);
                }
            }
            else if (key == "--hidden" || key == "-H") {
                ParseIntArrayHelper(value, hiddenSizes);
            }
            else if (key == "--output" || key == "-o") {
                outputSize = stoi(value);
            }
            else if (key == "--save" || key == "-s") {
                saveFile = value;
            }
            else if (key == "--model" || key == "-m") {
                modelFile = value;
            }
            else if (key == "--data" || key == "-d") {
                dataFile = value;
            }
            else if (key == "--lr") {
                learningRate = stod(value);
            }
            else if (key == "--optimizer") {
                optimizer = ParseOptimizer(value);
            }
            else if (key == "--hidden-act") {
                hiddenAct = ParseActivation(value);
            }
            else if (key == "--output-act") {
                outputAct = ParseActivation(value);
            }
            else if (key == "--dropout") {
                dropoutRate = stod(value);
            }
            else if (key == "--l2") {
                l2Lambda = stod(value);
            }
            else if (key == "--beta1") {
                beta1 = stod(value);
            }
            else if (key == "--beta2") {
                beta2 = stod(value);
            }
            else if (key == "--epochs") {
                epochs = stoi(value);
            }
            else if (key == "--batch") {
                batchSize = stoi(value);
            }
            else if (key == "--lr-decay-rate") {
                lrDecayRate = stod(value);
            }
            else if (key == "--lr-decay-epochs") {
                lrDecayEpochs = stoi(value);
            }
            else if (key == "--patience") {
                patience = stoi(value);
            }
            else if (key != "") {
                cerr << "Error: Unknown option: " << key << endl;
            }
        }
    }
    
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    if (deviceCount == 0) {
        cerr << "Error: No CUDA devices found!" << endl;
        return 1;
    }

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    
    try {
        if (command == cmdCreate) {
            if (inputSize <= 0) { cerr << "Error: --input (-i) is required" << endl; return 1; }
            if (hiddenSizes.empty()) { cerr << "Error: --hidden (-H) is required" << endl; return 1; }
            if (outputSize <= 0) { cerr << "Error: --output (-o) is required" << endl; return 1; }
            if (saveFile.empty()) { cerr << "Error: --save (-s) is required" << endl; return 1; }
            
            TMultiLayerPerceptronCUDA mlp(inputSize, hiddenSizes, outputSize, hiddenAct, outputAct);
            mlp.LearningRate = learningRate;
            mlp.Optimizer = optimizer;
            mlp.DropoutRate = dropoutRate;
            mlp.L2Lambda = l2Lambda;
            mlp.Beta1 = beta1;
            mlp.Beta2 = beta2;
            
            cout << "Created MLP model:" << endl;
            cout << "  Input size: " << inputSize << endl;
            cout << "  Hidden sizes: ";
            for (size_t i = 0; i < hiddenSizes.size(); i++) {
                if (i > 0) cout << ",";
                cout << hiddenSizes[i];
            }
            cout << endl;
            cout << "  Output size: " << outputSize << endl;
            cout << "  Hidden activation: " << ActivationToStr(hiddenAct) << endl;
            cout << "  Output activation: " << ActivationToStr(outputAct) << endl;
            cout << "  Optimizer: " << OptimizerToStr(optimizer) << endl;
            cout << fixed << setprecision(6);
            cout << "  Learning rate: " << learningRate << endl;
            cout << setprecision(4);
            cout << "  Dropout rate: " << dropoutRate << endl;
            cout << setprecision(6);
            cout << "  L2 lambda: " << l2Lambda << endl;
            
            mlp.SaveModelToJSON(saveFile);
            cout << "Model saved to JSON: " << saveFile << endl;
            return 0;
        }
        else if (command == cmdTrain) {
            if (modelFile.empty()) { cerr << "Error: --model (-m) is required" << endl; return 1; }
            if (saveFile.empty()) { cerr << "Error: --save (-s) is required" << endl; return 1; }
            cout << "Model loaded from JSON: " << modelFile << endl;
            TMultiLayerPerceptronCUDA mlp(1, {1}, 1, atSigmoid, atSigmoid);
            mlp.LoadModelFromJSON(modelFile);
            cout << "Model loaded successfully. Training functionality not yet implemented." << endl;
            return 0;
        }
        else if (command == cmdPredict) {
             if (modelFile.empty()) { cerr << "Error: --model (-m) is required" << endl; return 1; }
             if (inputValues.empty()) { cerr << "Error: --input (-i) is required" << endl; return 1; }
             
             TMultiLayerPerceptronCUDA mlp(1, {1}, 1, atSigmoid, atSigmoid);
             mlp.LoadModelFromJSON(modelFile);
             cout << "Model loaded successfully" << endl;
             
             if ((int)inputValues.size() != mlp.GetInputSize()) {
                 cerr << "Error: Expected " << mlp.GetInputSize() << " input values, got " << inputValues.size() << endl;
                 return 1;
             }
             
             Darray output = mlp.Predict(inputValues);
             
             cout << "Input: ";
            cout << fixed << setprecision(4);
            for (size_t i = 0; i < inputValues.size(); i++) {
                if (i > 0) cout << ", ";
                cout << inputValues[i];
            }
            cout << endl;
            
            cout << "Output: ";
            cout << setprecision(6);
            for (size_t i = 0; i < output.size(); i++) {
                if (i > 0) cout << ", ";
                cout << output[i];
            }
            cout << endl;
            
            if (output.size() > 1) {
                int maxIdx = MaxIndex(output);
                cout << "Max index: " << maxIdx << endl;
            }
            
            return 0;
        }
        else if (command == cmdInfo) {
             if (modelFile.empty()) { cerr << "Error: --model (-m) is required" << endl; return 1; }
             
             cout << "Model loaded from JSON: " << modelFile << endl;
             TMultiLayerPerceptronCUDA mlp(1, {1}, 1, atSigmoid, atSigmoid);
             mlp.LoadModelFromJSON(modelFile);
             
             cout << "MLP Model Information" << endl;
             cout << "=====================" << endl;
             cout << "Input size: " << mlp.GetInputSize() << endl;
             cout << "Output size: " << mlp.GetOutputSize() << endl;
             cout << "Hidden layers: " << mlp.GetHiddenLayerCount() << endl;
             cout << "Hidden sizes: " << mlp.GetHiddenLayerCount();
             if (mlp.GetHiddenLayerCount() > 0) {
                 for (size_t i = 0; i < mlp.GetHiddenSizes().size(); i++) {
                     cout << ", " << mlp.GetHiddenSizes()[i];
                 }
             }
             cout << endl;
             cout << "Layer sizes: " << mlp.GetInputSize();
            for (size_t i = 0; i < mlp.GetHiddenSizes().size(); i++)
                cout << " -> " << mlp.GetHiddenSizes()[i];
            cout << " -> " << mlp.GetOutputSize() << endl;
            cout << endl;
            cout << "Hyperparameters:" << endl;
            cout << fixed << setprecision(6);
            cout << "  Learning rate: " << mlp.LearningRate << endl;
            cout << "  Optimizer: " << OptimizerToStr(mlp.Optimizer) << endl;
            cout << "  Hidden activation: " << ActivationToStr(mlp.HiddenActivation) << endl;
            cout << "  Output activation: " << ActivationToStr(mlp.OutputActivation) << endl;
            cout << setprecision(4);
            cout << "  Dropout rate: " << mlp.DropoutRate << endl;
            cout << setprecision(6);
            cout << "  L2 lambda: " << mlp.L2Lambda << endl;
            cout << setprecision(4);
            cout << "  Beta1: " << mlp.Beta1 << endl;
            cout << "  Beta2: " << mlp.Beta2 << endl;
            cout << "  Timestep: " << mlp.Timestep << endl;
            cout << endl;
            cout << "Total layers: " << mlp.GetHiddenLayerCount() + 2 << endl;
            cout << "  Layer 0: " << mlp.GetInputSize() << " neurons (input)" << endl;
            for (size_t i = 0; i < mlp.GetHiddenSizes().size(); i++)
                cout << "  Layer " << (i + 1) << ": " << mlp.GetHiddenSizes()[i] << " neurons" << endl;
            cout << "  Layer " << (mlp.GetHiddenLayerCount() + 1) << ": " << mlp.GetOutputSize() << " neurons (output)" << endl;
            
            return 0;
        }
    } catch (const exception& e) {
        cerr << "Error: " << e.what() << endl;
        return 1;
    }
    
    return 0;
}
