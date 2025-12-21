//
// MLPOpenCL - OpenCL Command-line Multi-Layer Perceptron
// CLI: Create, Train, Predict, Inspect models. For scripting or research use.
// Enhanced with: Softmax, Adam/RMSProp optimizers, Dropout, L2 regularization,
// Xavier/He initialization, LR decay, Early stopping, Data normalization.
//
// Matthew Abbott 2025
//
// Compile: 
//   g++ -o mlp_opencl mlp_opencl.cpp -lOpenCL -std=c++11
//
// Usage: Same as CUDA version
//

#define CL_TARGET_OPENCL_VERSION 120
#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <ctime>
#include <cstring>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>

#define CL_CHECK(err) \
    do { \
        if (err != CL_SUCCESS) { \
            printf("OpenCL error at %s:%d: %d\n", __FILE__, __LINE__, err); \
            exit(1); \
        } \
    } while(0)

const double EPSILON = 1e-15;
const int BLOCK_SIZE = 256;
const char MODEL_MAGIC[] = "MLPOCL01";

enum TActivationType { atSigmoid = 0, atTanh = 1, atReLU = 2, atSoftmax = 3 };
enum TOptimizerType { otSGD = 0, otAdam = 1, otRMSProp = 2 };
enum TCommand { cmdNone, cmdCreate, cmdTrain, cmdPredict, cmdInfo, cmdHelp };

// OpenCL kernel source code
const char* kernelSource = R"CLC(

double d_Sigmoid(double x) {
    if (x < -500) return 0.0;
    else if (x > 500) return 1.0;
    else return 1.0 / (1.0 + exp(-x));
}

double d_DSigmoid(double x) {
    return x * (1.0 - x);
}

double d_TanhActivation(double x) {
    return tanh(x);
}

double d_DTanh(double x) {
    return 1.0 - (x * x);
}

double d_ReLU(double x) {
    return (x > 0) ? x : 0.0;
}

double d_DReLU(double x) {
    return (x > 0) ? 1.0 : 0.0;
}

double d_ApplyActivation(double x, int ActType) {
    if (ActType == 0) return d_Sigmoid(x);
    else if (ActType == 1) return d_TanhActivation(x);
    else if (ActType == 2) return d_ReLU(x);
    else return d_Sigmoid(x);
}

double d_ApplyActivationDerivative(double x, int ActType) {
    if (ActType == 0) return d_DSigmoid(x);
    else if (ActType == 1) return d_DTanh(x);
    else if (ActType == 2) return d_DReLU(x);
    else return d_DSigmoid(x);
}

__kernel void FeedForwardKernel(__global double* weights,
                                 __global double* biases,
                                 __global double* outputs,
                                 __global double* prevOutputs,
                                 int numNeurons,
                                 int numInputs,
                                 int prevSize,
                                 int activationType) {
    int i = get_global_id(0);
    if (i < numNeurons) {
        double sum = biases[i];
        for (int j = 0; j < prevSize; j++) {
            sum += prevOutputs[j] * weights[i * numInputs + j];
        }
        outputs[i] = d_ApplyActivation(sum, activationType);
    }
}

__kernel void FeedForwardSoftmaxSumKernel(__global double* weights,
                                           __global double* biases,
                                           __global double* sums,
                                           __global double* prevOutputs,
                                           int numNeurons,
                                           int numInputs,
                                           int prevSize) {
    int i = get_global_id(0);
    if (i < numNeurons) {
        double sum = biases[i];
        for (int j = 0; j < prevSize; j++) {
            sum += prevOutputs[j] * weights[i * numInputs + j];
        }
        sums[i] = sum;
    }
}

__kernel void SoftmaxKernel(__global double* sums,
                            __global double* outputs,
                            int n,
                            double maxVal,
                            double sumExp) {
    int i = get_global_id(0);
    if (i < n) {
        double val = exp(sums[i] - maxVal) / sumExp;
        if (val < 1e-15) val = 1e-15;
        else if (val > 1.0 - 1e-15) val = 1.0 - 1e-15;
        outputs[i] = val;
    }
}

// Simple LCG random number generator for dropout
unsigned int lcg_rand(unsigned int seed) {
    return seed * 1103515245u + 12345u;
}

float lcg_uniform(unsigned int* seed) {
    *seed = lcg_rand(*seed);
    return (float)(*seed) / 4294967296.0f;
}

__kernel void ApplyDropoutKernel(__global double* outputs,
                                  __global uchar* dropoutMask,
                                  int numNeurons,
                                  double dropoutRate,
                                  double scale,
                                  unsigned int seed) {
    int i = get_global_id(0);
    if (i < numNeurons) {
        if (dropoutRate <= 0) {
            dropoutMask[i] = 1;
            return;
        }
        unsigned int localSeed = seed + i * 1103515245u;
        float randVal = lcg_uniform(&localSeed);
        if (randVal > dropoutRate) {
            dropoutMask[i] = 1;
            outputs[i] = outputs[i] * scale;
        } else {
            dropoutMask[i] = 0;
            outputs[i] = 0.0;
        }
    }
}

__kernel void BackPropOutputKernel(__global double* errors,
                                    __global double* outputs,
                                    __global double* target,
                                    int numNeurons,
                                    int activationType,
                                    int isSoftmax) {
    int i = get_global_id(0);
    if (i < numNeurons) {
        if (isSoftmax) {
            errors[i] = target[i] - outputs[i];
        } else {
            errors[i] = d_ApplyActivationDerivative(outputs[i], activationType) *
                        (target[i] - outputs[i]);
        }
    }
}

__kernel void BackPropHiddenKernel(__global double* errors,
                                    __global double* outputs,
                                    __global uchar* dropoutMask,
                                    __global double* nextErrors,
                                    __global double* nextWeights,
                                    int numNeurons,
                                    int activationType,
                                    int nextNumNeurons,
                                    int nextNumInputs) {
    int i = get_global_id(0);
    if (i < numNeurons) {
        if (dropoutMask[i] == 0) {
            errors[i] = 0.0;
            return;
        }
        double errorSum = 0.0;
        for (int j = 0; j < nextNumNeurons; j++) {
            errorSum += nextErrors[j] * nextWeights[j * nextNumInputs + i];
        }
        errors[i] = d_ApplyActivationDerivative(outputs[i], activationType) * errorSum;
    }
}

__kernel void UpdateWeightsSGDKernel(__global double* weights,
                                      __global double* biases,
                                      __global double* errors,
                                      __global double* prevOutputs,
                                      int numNeurons,
                                      int numInputs,
                                      double learningRate,
                                      double l2Lambda) {
    int i = get_global_id(0);
    if (i < numNeurons) {
        for (int j = 0; j < numInputs; j++) {
            double gradient = errors[i] * prevOutputs[j];
            if (l2Lambda > 0)
                gradient = gradient - l2Lambda * weights[i * numInputs + j];
            weights[i * numInputs + j] += learningRate * gradient;
        }
        biases[i] += learningRate * errors[i];
    }
}

__kernel void UpdateWeightsAdamKernel(__global double* weights,
                                       __global double* biases,
                                       __global double* errors,
                                       __global double* prevOutputs,
                                       __global double* M,
                                       __global double* V,
                                       __global double* MBias,
                                       __global double* VBias,
                                       int numNeurons,
                                       int numInputs,
                                       double learningRate,
                                       double l2Lambda,
                                       double beta1,
                                       double beta2,
                                       int timestep) {
    int i = get_global_id(0);
    if (i < numNeurons) {
        double eps = 1e-8;
        double beta1_t = pow(beta1, (double)timestep);
        double beta2_t = pow(beta2, (double)timestep);

        for (int j = 0; j < numInputs; j++) {
            int idx = i * numInputs + j;
            double gradient = -errors[i] * prevOutputs[j];
            if (l2Lambda > 0)
                gradient += l2Lambda * weights[idx];

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

__kernel void UpdateWeightsRMSPropKernel(__global double* weights,
                                          __global double* biases,
                                          __global double* errors,
                                          __global double* prevOutputs,
                                          __global double* V,
                                          __global double* VBias,
                                          int numNeurons,
                                          int numInputs,
                                          double learningRate,
                                          double l2Lambda) {
    int i = get_global_id(0);
    if (i < numNeurons) {
        double eps = 1e-8;
        double decay = 0.9;

        for (int j = 0; j < numInputs; j++) {
            int idx = i * numInputs + j;
            double gradient = -errors[i] * prevOutputs[j];
            if (l2Lambda > 0)
                gradient += l2Lambda * weights[idx];

            V[idx] = decay * V[idx] + (1.0 - decay) * gradient * gradient;
            weights[idx] -= learningRate * gradient / (sqrt(V[idx]) + eps);
        }

        double gradient = -errors[i];
        VBias[i] = decay * VBias[i] + (1.0 - decay) * gradient * gradient;
        biases[i] -= learningRate * gradient / (sqrt(VBias[i]) + eps);
    }
}

)CLC";

struct LayerData {
    cl_mem Weights;
    cl_mem Biases;
    cl_mem Outputs;
    cl_mem Errors;
    cl_mem M;
    cl_mem V;
    cl_mem MBias;
    cl_mem VBias;
    cl_mem DropoutMask;
    int NumNeurons;
    int NumInputs;
    TActivationType ActivationType;
};

class TMultiLayerPerceptronOpenCL {
private:
    cl_context context;
    cl_command_queue queue;
    cl_program program;
    
    cl_kernel feedForwardKernel;
    cl_kernel feedForwardSoftmaxSumKernel;
    cl_kernel softmaxKernel;
    cl_kernel applyDropoutKernel;
    cl_kernel backPropOutputKernel;
    cl_kernel backPropHiddenKernel;
    cl_kernel updateWeightsSGDKernel;
    cl_kernel updateWeightsAdamKernel;
    cl_kernel updateWeightsRMSPropKernel;
    
    LayerData* Layers;
    int NumLayers;
    int FInputSize;
    int FOutputSize;
    std::vector<int> FHiddenSizes;
    bool FIsTraining;
    int MaxNeurons;

    cl_mem d_Target;
    cl_mem d_SoftmaxSums;

    void InitOpenCL() {
        cl_int err;
        cl_platform_id platform;
        cl_device_id device;
        
        err = clGetPlatformIDs(1, &platform, NULL);
        CL_CHECK(err);
        
        err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
        if (err != CL_SUCCESS) {
            err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_CPU, 1, &device, NULL);
        }
        CL_CHECK(err);
        
        context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
        CL_CHECK(err);
        
        queue = clCreateCommandQueue(context, device, 0, &err);
        CL_CHECK(err);
        
        program = clCreateProgramWithSource(context, 1, &kernelSource, NULL, &err);
        CL_CHECK(err);
        
        err = clBuildProgram(program, 1, &device, "-cl-std=CL1.2", NULL, NULL);
        if (err != CL_SUCCESS) {
            size_t len;
            char buffer[4096];
            clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &len);
            printf("Build error:\n%s\n", buffer);
            exit(1);
        }
        
        feedForwardKernel = clCreateKernel(program, "FeedForwardKernel", &err);
        CL_CHECK(err);
        feedForwardSoftmaxSumKernel = clCreateKernel(program, "FeedForwardSoftmaxSumKernel", &err);
        CL_CHECK(err);
        softmaxKernel = clCreateKernel(program, "SoftmaxKernel", &err);
        CL_CHECK(err);
        applyDropoutKernel = clCreateKernel(program, "ApplyDropoutKernel", &err);
        CL_CHECK(err);
        backPropOutputKernel = clCreateKernel(program, "BackPropOutputKernel", &err);
        CL_CHECK(err);
        backPropHiddenKernel = clCreateKernel(program, "BackPropHiddenKernel", &err);
        CL_CHECK(err);
        updateWeightsSGDKernel = clCreateKernel(program, "UpdateWeightsSGDKernel", &err);
        CL_CHECK(err);
        updateWeightsAdamKernel = clCreateKernel(program, "UpdateWeightsAdamKernel", &err);
        CL_CHECK(err);
        updateWeightsRMSPropKernel = clCreateKernel(program, "UpdateWeightsRMSPropKernel", &err);
        CL_CHECK(err);
    }

    void AllocateLayer(LayerData& layer, int numNeurons, int numInputs, TActivationType actType) {
        cl_int err;
        layer.NumNeurons = numNeurons;
        layer.NumInputs = numInputs;
        layer.ActivationType = actType;

        int weightSize = numNeurons * numInputs;
        layer.Weights = clCreateBuffer(context, CL_MEM_READ_WRITE, weightSize * sizeof(double), NULL, &err);
        CL_CHECK(err);
        layer.Biases = clCreateBuffer(context, CL_MEM_READ_WRITE, numNeurons * sizeof(double), NULL, &err);
        CL_CHECK(err);
        layer.Outputs = clCreateBuffer(context, CL_MEM_READ_WRITE, numNeurons * sizeof(double), NULL, &err);
        CL_CHECK(err);
        layer.Errors = clCreateBuffer(context, CL_MEM_READ_WRITE, numNeurons * sizeof(double), NULL, &err);
        CL_CHECK(err);
        layer.M = clCreateBuffer(context, CL_MEM_READ_WRITE, weightSize * sizeof(double), NULL, &err);
        CL_CHECK(err);
        layer.V = clCreateBuffer(context, CL_MEM_READ_WRITE, weightSize * sizeof(double), NULL, &err);
        CL_CHECK(err);
        layer.MBias = clCreateBuffer(context, CL_MEM_READ_WRITE, numNeurons * sizeof(double), NULL, &err);
        CL_CHECK(err);
        layer.VBias = clCreateBuffer(context, CL_MEM_READ_WRITE, numNeurons * sizeof(double), NULL, &err);
        CL_CHECK(err);
        layer.DropoutMask = clCreateBuffer(context, CL_MEM_READ_WRITE, numNeurons * sizeof(unsigned char), NULL, &err);
        CL_CHECK(err);

        double* zeros = new double[std::max(weightSize, numNeurons)]();
        clEnqueueWriteBuffer(queue, layer.Biases, CL_TRUE, 0, numNeurons * sizeof(double), zeros, 0, NULL, NULL);
        clEnqueueWriteBuffer(queue, layer.M, CL_TRUE, 0, weightSize * sizeof(double), zeros, 0, NULL, NULL);
        clEnqueueWriteBuffer(queue, layer.V, CL_TRUE, 0, weightSize * sizeof(double), zeros, 0, NULL, NULL);
        clEnqueueWriteBuffer(queue, layer.MBias, CL_TRUE, 0, numNeurons * sizeof(double), zeros, 0, NULL, NULL);
        clEnqueueWriteBuffer(queue, layer.VBias, CL_TRUE, 0, numNeurons * sizeof(double), zeros, 0, NULL, NULL);
        delete[] zeros;

        double limit;
        if (actType == atReLU)
            limit = sqrt(2.0 / numInputs);
        else
            limit = sqrt(6.0 / (numInputs + numNeurons));

        double* h_weights = new double[weightSize];
        for (int i = 0; i < weightSize; i++)
            h_weights[i] = ((double)rand() / RAND_MAX * 2 - 1) * limit;
        clEnqueueWriteBuffer(queue, layer.Weights, CL_TRUE, 0, weightSize * sizeof(double), h_weights, 0, NULL, NULL);
        delete[] h_weights;

        unsigned char* h_mask = new unsigned char[numNeurons];
        for (int i = 0; i < numNeurons; i++) h_mask[i] = 1;
        clEnqueueWriteBuffer(queue, layer.DropoutMask, CL_TRUE, 0, numNeurons * sizeof(unsigned char), h_mask, 0, NULL, NULL);
        delete[] h_mask;
    }

    void FreeLayer(LayerData& layer) {
        if (layer.Weights) clReleaseMemObject(layer.Weights);
        if (layer.Biases) clReleaseMemObject(layer.Biases);
        if (layer.Outputs) clReleaseMemObject(layer.Outputs);
        if (layer.Errors) clReleaseMemObject(layer.Errors);
        if (layer.M) clReleaseMemObject(layer.M);
        if (layer.V) clReleaseMemObject(layer.V);
        if (layer.MBias) clReleaseMemObject(layer.MBias);
        if (layer.VBias) clReleaseMemObject(layer.VBias);
        if (layer.DropoutMask) clReleaseMemObject(layer.DropoutMask);
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

    TMultiLayerPerceptronOpenCL(int InputSize, const std::vector<int>& HiddenSizes, int OutputSize,
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

        InitOpenCL();

        NumLayers = HiddenSizes.size() + 2;
        Layers = new LayerData[NumLayers];
        memset(Layers, 0, NumLayers * sizeof(LayerData));

        AllocateLayer(Layers[0], InputSize + 1, InputSize, atSigmoid);

        MaxNeurons = InputSize + 1;
        int numInputs = InputSize;
        for (size_t i = 0; i < HiddenSizes.size(); i++) {
            AllocateLayer(Layers[i + 1], HiddenSizes[i] + 1, numInputs + 1, HiddenActivation);
            if (HiddenSizes[i] + 1 > MaxNeurons) MaxNeurons = HiddenSizes[i] + 1;
            numInputs = HiddenSizes[i];
        }

        AllocateLayer(Layers[NumLayers - 1], OutputSize, numInputs + 1, OutputActivation);
        if (OutputSize > MaxNeurons) MaxNeurons = OutputSize;

        cl_int err;
        d_Target = clCreateBuffer(context, CL_MEM_READ_WRITE, OutputSize * sizeof(double), NULL, &err);
        CL_CHECK(err);
        d_SoftmaxSums = clCreateBuffer(context, CL_MEM_READ_WRITE, OutputSize * sizeof(double), NULL, &err);
        CL_CHECK(err);
    }

    ~TMultiLayerPerceptronOpenCL() {
        for (int i = 0; i < NumLayers; i++)
            FreeLayer(Layers[i]);
        delete[] Layers;
        
        clReleaseMemObject(d_Target);
        clReleaseMemObject(d_SoftmaxSums);
        
        clReleaseKernel(feedForwardKernel);
        clReleaseKernel(feedForwardSoftmaxSumKerne
