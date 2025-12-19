//
// Matthew Abbott 19/3/2023
// CUDA Port of MLP
// Enhanced with: Softmax, Adam/RMSProp optimizers, Dropout, L2 regularization,
// Xavier/He initialization, LR decay, Early stopping, Data normalization
//
// Compile: nvcc -o mlp_cuda mlp.cu -lcurand
//

#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>
#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <ctime>
#include <cstring>

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

enum TActivationType { atSigmoid, atTanh, atReLU, atSoftmax };
enum TOptimizerType { otSGD, otAdam, otRMSProp };

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
    int* FHiddenSizes;
    int FHiddenLayerCount;
    bool FIsTraining;
    curandState* d_RandStates;
    int MaxNeurons;

    double* d_Input;
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
        cudaFree(layer.Weights);
        cudaFree(layer.Biases);
        cudaFree(layer.Outputs);
        cudaFree(layer.Errors);
        cudaFree(layer.M);
        cudaFree(layer.V);
        cudaFree(layer.MBias);
        cudaFree(layer.VBias);
        cudaFree(layer.DropoutMask);
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

    TMultiLayerPerceptronCUDA(int InputSize, const int* HiddenSizes, int NumHidden, int OutputSize,
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
        FHiddenLayerCount = NumHidden;

        FHiddenSizes = new int[NumHidden];
        for (int i = 0; i < NumHidden; i++)
            FHiddenSizes[i] = HiddenSizes[i];

        NumLayers = NumHidden + 2;
        h_Layers = new LayerData[NumLayers];
        CUDA_CHECK(cudaMalloc(&d_Layers, NumLayers * sizeof(LayerData)));

        AllocateLayer(h_Layers[0], InputSize + 1, InputSize, atSigmoid);

        MaxNeurons = InputSize + 1;
        int numInputs = InputSize;
        for (int i = 0; i < NumHidden; i++) {
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

        CUDA_CHECK(cudaMalloc(&d_Input, (InputSize + 1) * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_Target, OutputSize * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_SoftmaxSums, OutputSize * sizeof(double)));
    }

    ~TMultiLayerPerceptronCUDA() {
        for (int i = 0; i < NumLayers; i++)
            FreeLayer(h_Layers[i]);
        delete[] h_Layers;
        delete[] FHiddenSizes;
        cudaFree(d_Layers);
        cudaFree(d_RandStates);
        cudaFree(d_Input);
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

        CUDA_CHECK(cudaMemcpy(h_Layers[0].Outputs, Input, (FInputSize + 1) * sizeof(double), cudaMemcpyHostToDevice));

        FeedForward();

        CUDA_CHECK(cudaMemcpy(Result, h_Layers[NumLayers - 1].Outputs, FOutputSize * sizeof(double), cudaMemcpyDeviceToHost));

        FIsTraining = true;
    }

    void Train(const double* Input, const double* Target) {
        FIsTraining = true;

        CUDA_CHECK(cudaMemcpy(h_Layers[0].Outputs, Input, (FInputSize + 1) * sizeof(double), cudaMemcpyHostToDevice));
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

struct TDataPoint {
    double* Input;
    double* Target;
    int InputSize;
    int TargetSize;
};

struct TDataPointArray {
    TDataPoint* Data;
    int Length;

    void SetLength(int n, int inputSize, int targetSize) {
        Data = new TDataPoint[n];
        Length = n;
        for (int i = 0; i < n; i++) {
            Data[i].Input = new double[inputSize];
            Data[i].Target = new double[targetSize];
            Data[i].InputSize = inputSize;
            Data[i].TargetSize = targetSize;
        }
    }

    void Free() {
        for (int i = 0; i < Length; i++) {
            delete[] Data[i].Input;
            delete[] Data[i].Target;
        }
        delete[] Data;
    }

    TDataPoint& operator[](int i) { return Data[i]; }
};

void ShuffleData(TDataPointArray& Data) {
    for (int i = Data.Length - 1; i >= 1; i--) {
        int j = rand() % (i + 1);
        TDataPoint temp = Data[i];
        Data[i] = Data[j];
        Data[j] = temp;
    }
}

int TrainWithEarlyStopping(TMultiLayerPerceptronCUDA* MLP, TDataPointArray& Data, int MaxEpochs, int BatchSize) {
    int ValSize = Data.Length / 10;
    if (ValSize < 1) ValSize = 1;

    double BestLoss = 1e30;
    int EpochsWithoutImprovement = 0;
    double InitialLR = MLP->LearningRate;

    double* Pred = new double[MLP->GetOutputSize()];

    for (int Epoch = 1; Epoch <= MaxEpochs; Epoch++) {
        if (MLP->EnableLRDecay && (Epoch > 1) && (Epoch % MLP->LRDecayEpochs == 0))
            MLP->LearningRate *= MLP->LRDecayRate;

        ShuffleData(Data);

        for (int i = ValSize; i < Data.Length; i++)
            MLP->Train(Data[i].Input, Data[i].Target);

        if (MLP->EnableEarlyStopping) {
            double ValLoss = 0;
            for (int i = 0; i < ValSize; i++) {
                MLP->Predict(Data[i].Input, Pred);
                ValLoss += MLP->ComputeLoss(Pred, Data[i].Target);
            }
            ValLoss /= ValSize;

            if (ValLoss < BestLoss - EPSILON) {
                BestLoss = ValLoss;
                EpochsWithoutImprovement = 0;
            } else {
                EpochsWithoutImprovement++;
            }

            if (EpochsWithoutImprovement >= MLP->EarlyStoppingPatience) {
                printf("Early stopping at epoch %d (validation loss: %.6f)\n", Epoch, ValLoss);
                MLP->LearningRate = InitialLR;
                delete[] Pred;
                return Epoch;
            }
        }
    }

    MLP->LearningRate = InitialLR;
    delete[] Pred;
    return MaxEpochs;
}

double TestAccuracy(TMultiLayerPerceptronCUDA* MLP, TDataPointArray& Data) {
    double* Pred = new double[MLP->GetOutputSize()];
    int correct = 0;

    for (int i = 0; i < Data.Length; i++) {
        MLP->Predict(Data[i].Input, Pred);
        int predClass = MaxIndex(Pred, MLP->GetOutputSize());
        int actualClass = MaxIndex(Data[i].Target, MLP->GetOutputSize());
        if (predClass == actualClass) correct++;
    }

    delete[] Pred;
    return (double)correct / Data.Length;
}

double PrecisionScore(TMultiLayerPerceptronCUDA* MLP, TDataPointArray& Data, int ClassIndex) {
    double* Pred = new double[MLP->GetOutputSize()];
    int TP = 0, FP = 0;

    for (int i = 0; i < Data.Length; i++) {
        MLP->Predict(Data[i].Input, Pred);
        int predClass = MaxIndex(Pred, MLP->GetOutputSize());
        int actualClass = MaxIndex(Data[i].Target, MLP->GetOutputSize());

        if (predClass == ClassIndex) {
            if (actualClass == ClassIndex) TP++;
            else FP++;
        }
    }

    delete[] Pred;
    return (TP + FP == 0) ? 0 : (double)TP / (TP + FP);
}

double RecallScore(TMultiLayerPerceptronCUDA* MLP, TDataPointArray& Data, int ClassIndex) {
    double* Pred = new double[MLP->GetOutputSize()];
    int TP = 0, FN = 0;

    for (int i = 0; i < Data.Length; i++) {
        MLP->Predict(Data[i].Input, Pred);
        int predClass = MaxIndex(Pred, MLP->GetOutputSize());
        int actualClass = MaxIndex(Data[i].Target, MLP->GetOutputSize());

        if (actualClass == ClassIndex) {
            if (predClass == ClassIndex) TP++;
            else FN++;
        }
    }

    delete[] Pred;
    return (TP + FN == 0) ? 0 : (double)TP / (TP + FN);
}

double F1Score(double Precision, double Recall) {
    return (Precision + Recall == 0) ? 0 : 2 * Precision * Recall / (Precision + Recall);
}

int main() {
    srand((unsigned)time(nullptr));

    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    if (deviceCount == 0) {
        printf("No CUDA devices found!\n");
        return 1;
    }

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("Using GPU: %s\n", prop.name);
    printf("Compute capability: %d.%d\n\n", prop.major, prop.minor);

    const int InputSize = 4;
    const int OutputSize = 3;

    TDataPointArray Data;
    Data.SetLength(7500, InputSize + 1, OutputSize);

    for (int i = 0; i < 2500; i++) {
        Data[i].Input[0] = RandomDouble() * 0.5;
        Data[i].Input[1] = RandomDouble() * 0.5;
        Data[i].Input[2] = RandomDouble() * 0.5;
        Data[i].Input[3] = RandomDouble() * 0.5;
        Data[i].Target[0] = 1; Data[i].Target[1] = 0; Data[i].Target[2] = 0;
    }
    for (int i = 2500; i < 5000; i++) {
        Data[i].Input[0] = 0.5 + RandomDouble() * 0.5;
        Data[i].Input[1] = 0.5 + RandomDouble() * 0.5;
        Data[i].Input[2] = 0.5 + RandomDouble() * 0.5;
        Data[i].Input[3] = 0.5 + RandomDouble() * 0.5;
        Data[i].Target[0] = 0; Data[i].Target[1] = 1; Data[i].Target[2] = 0;
    }
    for (int i = 5000; i < 7500; i++) {
        Data[i].Input[0] = RandomDouble() * 0.5;
        Data[i].Input[1] = 0.5 + RandomDouble() * 0.5;
        Data[i].Input[2] = RandomDouble() * 0.5;
        Data[i].Input[3] = 0.5 + RandomDouble() * 0.5;
        Data[i].Target[0] = 0; Data[i].Target[1] = 0; Data[i].Target[2] = 1;
    }

    printf("=== Enhanced MLP CUDA Test ===\n\n");

    int HiddenSizes[] = {8, 8, 8};
    TMultiLayerPerceptronCUDA* MLP = new TMultiLayerPerceptronCUDA(InputSize, HiddenSizes, 3, OutputSize, atSigmoid, atSoftmax);
    MLP->MaxIterations = 30;
    MLP->Optimizer = otAdam;
    MLP->LearningRate = 0.001;
    MLP->DropoutRate = 0.1;
    MLP->L2Lambda = 0.0001;
    MLP->EnableLRDecay = true;
    MLP->LRDecayRate = 0.95;
    MLP->LRDecayEpochs = 10;
    MLP->EnableEarlyStopping = true;
    MLP->EarlyStoppingPatience = 5;

    printf("Configuration:\n");
    printf("  Optimizer: Adam\n");
    printf("  Hidden Activation: Sigmoid\n");
    printf("  Output Activation: Softmax\n");
    printf("  Dropout Rate: %.2f\n", MLP->DropoutRate);
    printf("  L2 Lambda: %.6f\n", MLP->L2Lambda);
    printf("  Learning Rate: %.4f\n", MLP->LearningRate);
    printf("  LR Decay: %.2f every %d epochs\n\n", MLP->LRDecayRate, MLP->LRDecayEpochs);

    printf("Training with early stopping...\n");
    TrainWithEarlyStopping(MLP, Data, 100, 32);
    printf("\n");

    double Accuracy = TestAccuracy(MLP, Data);

    double AvgPrecision = 0, AvgRecall = 0, AvgF1 = 0;

    printf("Per-class metrics:\n");
    for (int c = 0; c < OutputSize; c++) {
        double ClassPrecision = PrecisionScore(MLP, Data, c);
        double ClassRecall = RecallScore(MLP, Data, c);
        double ClassF1 = F1Score(ClassPrecision, ClassRecall);
        printf("  Class %d: Precision=%.3f Recall=%.3f F1=%.3f\n", c, ClassPrecision, ClassRecall, ClassF1);
        AvgPrecision += ClassPrecision;
        AvgRecall += ClassRecall;
        AvgF1 += ClassF1;
    }

    AvgPrecision /= OutputSize;
    AvgRecall /= OutputSize;
    AvgF1 /= OutputSize;

    printf("\nOverall Results:\n");
    printf("  Accuracy: %.3f\n", Accuracy);
    printf("  Avg Precision: %.3f\n", AvgPrecision);
    printf("  Avg Recall: %.3f\n", AvgRecall);
    printf("  Avg F1 Score: %.3f\n", AvgF1);

    delete MLP;
    Data.Free();

    return 0;
}
