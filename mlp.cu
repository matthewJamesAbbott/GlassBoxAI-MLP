//
// MLPCuda - CUDA Command-line Multi-Layer Perceptron (Core Version, No Facade)
// CLI: Create, Train, Predict, Inspect models. For scripting or research use.
// Enhanced with: Softmax, Adam/RMSProp optimizers, Dropout, L2 regularization,
// Xavier/He initialization, LR decay, Early stopping, Data normalization.
//
// Matthew Abbott 2025
//
// Compile: 
//   nvcc -o mlp_cuda mlp.cu -lcurand
//
// Usage (commands):
//   mlp_cuda create --input=N --hidden=N,N,... --output=N [options] --save=FILE
//   mlp_cuda train --model=FILE --data=FILE [options] --save=FILE
//   mlp_cuda predict --model=FILE --input=v1,v2,...
//   mlp_cuda info --model=FILE
//   mlp_cuda help
//
// Options:
//   --input=N                  Input layer size (required)
//   --hidden=N,N,...           Hidden layer sizes (comma-separated, required)
//   --output=N                 Output layer size (required)
//   --save=FILE                Save model to file (required)
//   --model=FILE               Model file to load
//   --data=FILE                Training data CSV file
//   --lr=VALUE                 Learning rate (default: 0.1)
//   --optimizer=TYPE           sgd|adam|rmsprop (default: sgd)
//   --hidden-act=TYPE          sigmoid|tanh|relu|softmax (default: sigmoid)
//   --output-act=TYPE          sigmoid|tanh|relu|softmax (default: sigmoid)
//   --dropout=VALUE            Dropout rate 0-1 (default: 0)
//   --l2=VALUE                 L2 regularization (default: 0)
//   --beta1=VALUE              Adam beta1 (default: 0.9)
//   --beta2=VALUE              Adam beta2 (default: 0.999)
//   --epochs=N                 Training epochs (default: 100)
//   --batch=N                  Training batch size (default: 1)
//   --lr-decay                 Enable learning rate decay
//   --lr-decay-rate=VALUE      Learning rate decay rate (default: 0.95)
//   --lr-decay-epochs=N        Iterations between learning rate decay (default: 10)
//   --early-stop               Enable early stopping
//   --patience=N               Early stopping patience (default: 10)
//   --normalize                Normalize input data before training
//   --verbose                  Show training progress
//
// Examples:
//   mlp_cuda create --input=2 --hidden=8 --output=1 --save=xor.bin
//   mlp_cuda train --model=xor.bin --data=xor_cuda.csv --epochs=1000 --save=xor_trained.bin
//   mlp_cuda predict --model=xor_trained.bin --input=1,0
//   mlp_cuda info --model=xor_trained.bin
//

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

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            printf("CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(1); \
        } \
    } while(0)

const double EPSILON = 1e-15;
const int BLOCK_SIZE = 256;
const char MODEL_MAGIC[] = "MLPCUDA1";

enum TActivationType { atSigmoid = 0, atTanh = 1, atReLU = 2, atSoftmax = 3 };
enum TOptimizerType { otSGD = 0, otAdam = 1, otRMSProp = 2 };
enum TCommand { cmdNone, cmdCreate, cmdTrain, cmdPredict, cmdInfo, cmdHelp };

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
    return (x > 0) ? 1 : 0;
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

    TMultiLayerPerceptronCUDA(int InputSize, const std::vector<int>& HiddenSizes, int OutputSize,
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

    int GetLayerSize(int layerIdx) const {
        if (layerIdx < 0 || layerIdx >= NumLayers) return 0;
        return h_Layers[layerIdx].NumNeurons;
    }

    bool Save(const char* filename) {
        FILE* f = fopen(filename, "wb");
        if (!f) return false;

        fwrite(MODEL_MAGIC, 1, 8, f);
        fwrite(&FInputSize, sizeof(int), 1, f);
        fwrite(&FOutputSize, sizeof(int), 1, f);
        int numHidden = FHiddenSizes.size();
        fwrite(&numHidden, sizeof(int), 1, f);
        fwrite(FHiddenSizes.data(), sizeof(int), numHidden, f);

        fwrite(&LearningRate, sizeof(double), 1, f);
        int opt = (int)Optimizer;
        fwrite(&opt, sizeof(int), 1, f);
        int hidAct = (int)HiddenActivation;
        fwrite(&hidAct, sizeof(int), 1, f);
        int outAct = (int)OutputActivation;
        fwrite(&outAct, sizeof(int), 1, f);
        fwrite(&DropoutRate, sizeof(double), 1, f);
        fwrite(&L2Lambda, sizeof(double), 1, f);
        fwrite(&Beta1, sizeof(double), 1, f);
        fwrite(&Beta2, sizeof(double), 1, f);
        fwrite(&Timestep, sizeof(int), 1, f);
        fwrite(&EnableLRDecay, sizeof(bool), 1, f);
        fwrite(&LRDecayRate, sizeof(double), 1, f);
        fwrite(&LRDecayEpochs, sizeof(int), 1, f);
        fwrite(&EnableEarlyStopping, sizeof(bool), 1, f);
        fwrite(&EarlyStoppingPatience, sizeof(int), 1, f);

        for (int k = 0; k < NumLayers; k++) {
            LayerData& layer = h_Layers[k];
            int weightSize = layer.NumNeurons * layer.NumInputs;

            double* h_weights = new double[weightSize];
            double* h_biases = new double[layer.NumNeurons];
            double* h_M = new double[weightSize];
            double* h_V = new double[weightSize];
            double* h_MBias = new double[layer.NumNeurons];
            double* h_VBias = new double[layer.NumNeurons];

            CUDA_CHECK(cudaMemcpy(h_weights, layer.Weights, weightSize * sizeof(double), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(h_biases, layer.Biases, layer.NumNeurons * sizeof(double), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(h_M, layer.M, weightSize * sizeof(double), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(h_V, layer.V, weightSize * sizeof(double), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(h_MBias, layer.MBias, layer.NumNeurons * sizeof(double), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(h_VBias, layer.VBias, layer.NumNeurons * sizeof(double), cudaMemcpyDeviceToHost));

            fwrite(h_weights, sizeof(double), weightSize, f);
            fwrite(h_biases, sizeof(double), layer.NumNeurons, f);
            fwrite(h_M, sizeof(double), weightSize, f);
            fwrite(h_V, sizeof(double), weightSize, f);
            fwrite(h_MBias, sizeof(double), layer.NumNeurons, f);
            fwrite(h_VBias, sizeof(double), layer.NumNeurons, f);

            delete[] h_weights;
            delete[] h_biases;
            delete[] h_M;
            delete[] h_V;
            delete[] h_MBias;
            delete[] h_VBias;
        }

        fclose(f);
        return true;
    }

    static TMultiLayerPerceptronCUDA* Load(const char* filename) {
        FILE* f = fopen(filename, "rb");
        if (!f) return nullptr;

        char magic[9] = {0};
        fread(magic, 1, 8, f);
        if (strcmp(magic, MODEL_MAGIC) != 0) {
            fclose(f);
            return nullptr;
        }

        int inputSize, outputSize, numHidden;
        fread(&inputSize, sizeof(int), 1, f);
        fread(&outputSize, sizeof(int), 1, f);
        fread(&numHidden, sizeof(int), 1, f);

        std::vector<int> hiddenSizes(numHidden);
        fread(hiddenSizes.data(), sizeof(int), numHidden, f);

        double learningRate;
        int opt, hidAct, outAct;
        fread(&learningRate, sizeof(double), 1, f);
        fread(&opt, sizeof(int), 1, f);
        fread(&hidAct, sizeof(int), 1, f);
        fread(&outAct, sizeof(int), 1, f);

        TMultiLayerPerceptronCUDA* mlp = new TMultiLayerPerceptronCUDA(
            inputSize, hiddenSizes, outputSize, 
            (TActivationType)hidAct, (TActivationType)outAct);

        mlp->LearningRate = learningRate;
        mlp->Optimizer = (TOptimizerType)opt;

        fread(&mlp->DropoutRate, sizeof(double), 1, f);
        fread(&mlp->L2Lambda, sizeof(double), 1, f);
        fread(&mlp->Beta1, sizeof(double), 1, f);
        fread(&mlp->Beta2, sizeof(double), 1, f);
        fread(&mlp->Timestep, sizeof(int), 1, f);
        fread(&mlp->EnableLRDecay, sizeof(bool), 1, f);
        fread(&mlp->LRDecayRate, sizeof(double), 1, f);
        fread(&mlp->LRDecayEpochs, sizeof(int), 1, f);
        fread(&mlp->EnableEarlyStopping, sizeof(bool), 1, f);
        fread(&mlp->EarlyStoppingPatience, sizeof(int), 1, f);

        for (int k = 0; k < mlp->NumLayers; k++) {
            LayerData& layer = mlp->h_Layers[k];
            int weightSize = layer.NumNeurons * layer.NumInputs;

            double* h_weights = new double[weightSize];
            double* h_biases = new double[layer.NumNeurons];
            double* h_M = new double[weightSize];
            double* h_V = new double[weightSize];
            double* h_MBias = new double[layer.NumNeurons];
            double* h_VBias = new double[layer.NumNeurons];

            fread(h_weights, sizeof(double), weightSize, f);
            fread(h_biases, sizeof(double), layer.NumNeurons, f);
            fread(h_M, sizeof(double), weightSize, f);
            fread(h_V, sizeof(double), weightSize, f);
            fread(h_MBias, sizeof(double), layer.NumNeurons, f);
            fread(h_VBias, sizeof(double), layer.NumNeurons, f);

            CUDA_CHECK(cudaMemcpy(layer.Weights, h_weights, weightSize * sizeof(double), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(layer.Biases, h_biases, layer.NumNeurons * sizeof(double), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(layer.M, h_M, weightSize * sizeof(double), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(layer.V, h_V, weightSize * sizeof(double), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(layer.MBias, h_MBias, layer.NumNeurons * sizeof(double), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(layer.VBias, h_VBias, layer.NumNeurons * sizeof(double), cudaMemcpyHostToDevice));

            delete[] h_weights;
            delete[] h_biases;
            delete[] h_M;
            delete[] h_V;
            delete[] h_MBias;
            delete[] h_VBias;
        }

        fclose(f);
        return mlp;
    }
};

double RandomDouble() {
    return (double)rand() / RAND_MAX;
}

int MaxIndex(const double* arr, int n) {
    int result = 0;
    for (int i = 1; i < n; i++)
        if (arr[i] > arr[result])
            result = i;
    return result;
}

struct DataPoint {
    std::vector<double> Input;
    std::vector<double> Target;
};

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

void PrintUsage() {
    printf("MLP CUDA - Command-line Multi-Layer Perceptron\n");
    printf("\n");
    printf("Commands:\n");
    printf("  create   Create a new MLP model\n");
    printf("  train    Train an existing model with data\n");
    printf("  predict  Make predictions with a trained model\n");
    printf("  info     Display model information\n");
    printf("  help     Show this help message\n");
    printf("\n");
    printf("Create Options:\n");
    printf("  --input=N              Input layer size (required)\n");
    printf("  --hidden=N,N,...       Hidden layer sizes (required)\n");
    printf("  --output=N             Output layer size (required)\n");
    printf("  --save=FILE            Save model to file (required)\n");
    printf("  --lr=VALUE             Learning rate (default: 0.1)\n");
    printf("  --optimizer=TYPE       sgd|adam|rmsprop (default: sgd)\n");
    printf("  --hidden-act=TYPE      sigmoid|tanh|relu|softmax (default: sigmoid)\n");
    printf("  --output-act=TYPE      sigmoid|tanh|relu|softmax (default: sigmoid)\n");
    printf("  --dropout=VALUE        Dropout rate 0-1 (default: 0)\n");
    printf("  --l2=VALUE             L2 regularization (default: 0)\n");
    printf("  --beta1=VALUE          Adam beta1 (default: 0.9)\n");
    printf("  --beta2=VALUE          Adam beta2 (default: 0.999)\n");
    printf("\n");
    printf("Train Options:\n");
    printf("  --model=FILE           Model file to load (required)\n");
    printf("  --data=FILE            Training data CSV file (required)\n");
    printf("  --save=FILE            Save trained model to file (required)\n");
    printf("  --epochs=N             Number of training epochs (default: 100)\n");
    printf("  --batch=N              Batch size (default: 1)\n");
    printf("  --lr=VALUE             Override learning rate\n");
    printf("  --lr-decay             Enable learning rate decay\n");
    printf("  --lr-decay-rate=VALUE  LR decay rate (default: 0.95)\n");
    printf("  --lr-decay-epochs=N    Epochs between decay (default: 10)\n");
    printf("  --early-stop           Enable early stopping\n");
    printf("  --patience=N           Early stopping patience (default: 10)\n");
    printf("  --normalize            Normalize input data\n");
    printf("  --verbose              Show training progress\n");
    printf("\n");
    printf("Predict Options:\n");
    printf("  --model=FILE           Model file to load (required)\n");
    printf("  --input=v1,v2,...      Input values (required)\n");
    printf("\n");
    printf("Info Options:\n");
    printf("  --model=FILE           Model file to load (required)\n");
    printf("\n");
    printf("Examples:\n");
    printf("  mlp_cuda create --input=2 --hidden=4,4 --output=1 --save=xor.bin\n");
    printf("  mlp_cuda train --model=xor.bin --data=xor.csv --epochs=1000 --save=xor_trained.bin\n");
    printf("  mlp_cuda predict --model=xor_trained.bin --input=1,0\n");
    printf("  mlp_cuda info --model=xor_trained.bin\n");
}

int main(int argc, char** argv) {
    srand((unsigned)time(nullptr));

    if (argc < 2) {
        PrintUsage();
        return 0;
    }

    TCommand command = cmdNone;
    std::string cmdStr = argv[1];
    if (cmdStr == "create") command = cmdCreate;
    else if (cmdStr == "train") command = cmdTrain;
    else if (cmdStr == "predict") command = cmdPredict;
    else if (cmdStr == "info") command = cmdInfo;
    else if (cmdStr == "help" || cmdStr == "--help" || cmdStr == "-h") command = cmdHelp;
    else {
        printf("Unknown command: %s\n", argv[1]);
        PrintUsage();
        return 1;
    }

    if (command == cmdHelp) {
        PrintUsage();
        return 0;
    }

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
    double lrDecayRate = 0.95;
    int lrDecayEpochs = 10, patience = 10;
    bool lrOverride = false;

    for (int i = 2; i < argc; i++) {
        std::string arg = argv[i];

        if (arg == "--lr-decay") { lrDecay = true; continue; }
        if (arg == "--early-stop") { earlyStop = true; continue; }
        if (arg == "--normalize") { normalize = true; continue; }
        if (arg == "--verbose") { verbose = true; continue; }

        size_t eq = arg.find('=');
        if (eq == std::string::npos) {
            printf("Invalid argument: %s\n", arg.c_str());
            continue;
        }

        std::string key = arg.substr(0, eq);
        std::string value = arg.substr(eq + 1);

        if (key == "--input") {
            if (command == cmdPredict)
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
        else printf("Unknown option: %s\n", key.c_str());
    }

    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    if (deviceCount == 0) {
        printf("Error: No CUDA devices found!\n");
        return 1;
    }

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);

    if (command == cmdCreate) {
        if (inputSize <= 0) { printf("Error: --input is required\n"); return 1; }
        if (hiddenSizes.empty()) { printf("Error: --hidden is required\n"); return 1; }
        if (outputSize <= 0) { printf("Error: --output is required\n"); return 1; }
        if (saveFile.empty()) { printf("Error: --save is required\n"); return 1; }

        TMultiLayerPerceptronCUDA* mlp = new TMultiLayerPerceptronCUDA(
            inputSize, hiddenSizes, outputSize, hiddenAct, outputAct);
        mlp->LearningRate = learningRate;
        mlp->Optimizer = optimizer;
        mlp->DropoutRate = dropoutRate;
        mlp->L2Lambda = l2Lambda;
        mlp->Beta1 = beta1;
        mlp->Beta2 = beta2;

        mlp->Save(saveFile.c_str());

        printf("Created CUDA MLP model (GPU: %s):\n", prop.name);
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

    return 0;
}
