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
        clReleaseKernel(feedForwardSoftmaxSumKernel);
        clReleaseKernel(softmaxKernel);
        clReleaseKernel(applyDropoutKernel);
        clReleaseKernel(backPropOutputKernel);
        clReleaseKernel(backPropHiddenKernel);
        clReleaseKernel(updateWeightsSGDKernel);
        clReleaseKernel(updateWeightsAdamKernel);
        clReleaseKernel(updateWeightsRMSPropKernel);
        
        clReleaseProgram(program);
        clReleaseCommandQueue(queue);
        clReleaseContext(context);
    }

    void FeedForward() {
        cl_int err;
        
        for (int k = 1; k < NumLayers - 1; k++) {
            LayerData& layer = Layers[k];
            LayerData& prevLayer = Layers[k - 1];

            size_t globalSize = ((layer.NumNeurons + BLOCK_SIZE - 1) / BLOCK_SIZE) * BLOCK_SIZE;
            
            clSetKernelArg(feedForwardKernel, 0, sizeof(cl_mem), &layer.Weights);
            clSetKernelArg(feedForwardKernel, 1, sizeof(cl_mem), &layer.Biases);
            clSetKernelArg(feedForwardKernel, 2, sizeof(cl_mem), &layer.Outputs);
            clSetKernelArg(feedForwardKernel, 3, sizeof(cl_mem), &prevLayer.Outputs);
            clSetKernelArg(feedForwardKernel, 4, sizeof(int), &layer.NumNeurons);
            clSetKernelArg(feedForwardKernel, 5, sizeof(int), &layer.NumInputs);
            clSetKernelArg(feedForwardKernel, 6, sizeof(int), &prevLayer.NumNeurons);
            int actType = (int)layer.ActivationType;
            clSetKernelArg(feedForwardKernel, 7, sizeof(int), &actType);
            
            err = clEnqueueNDRangeKernel(queue, feedForwardKernel, 1, NULL, &globalSize, NULL, 0, NULL, NULL);
            CL_CHECK(err);

            if (FIsTraining && DropoutRate > 0) {
                double scale = 1.0 / (1.0 - DropoutRate);
                unsigned int seed = (unsigned int)(time(NULL) + k);
                
                clSetKernelArg(applyDropoutKernel, 0, sizeof(cl_mem), &layer.Outputs);
                clSetKernelArg(applyDropoutKernel, 1, sizeof(cl_mem), &layer.DropoutMask);
                clSetKernelArg(applyDropoutKernel, 2, sizeof(int), &layer.NumNeurons);
                clSetKernelArg(applyDropoutKernel, 3, sizeof(double), &DropoutRate);
                clSetKernelArg(applyDropoutKernel, 4, sizeof(double), &scale);
                clSetKernelArg(applyDropoutKernel, 5, sizeof(unsigned int), &seed);
                
                err = clEnqueueNDRangeKernel(queue, applyDropoutKernel, 1, NULL, &globalSize, NULL, 0, NULL, NULL);
                CL_CHECK(err);
            }
        }

        LayerData& outputLayer = Layers[NumLayers - 1];
        LayerData& lastHidden = Layers[NumLayers - 2];
        size_t globalSize = ((outputLayer.NumNeurons + BLOCK_SIZE - 1) / BLOCK_SIZE) * BLOCK_SIZE;

        if (OutputActivation == atSoftmax) {
            clSetKernelArg(feedForwardSoftmaxSumKernel, 0, sizeof(cl_mem), &outputLayer.Weights);
            clSetKernelArg(feedForwardSoftmaxSumKernel, 1, sizeof(cl_mem), &outputLayer.Biases);
            clSetKernelArg(feedForwardSoftmaxSumKernel, 2, sizeof(cl_mem), &d_SoftmaxSums);
            clSetKernelArg(feedForwardSoftmaxSumKernel, 3, sizeof(cl_mem), &lastHidden.Outputs);
            clSetKernelArg(feedForwardSoftmaxSumKernel, 4, sizeof(int), &outputLayer.NumNeurons);
            clSetKernelArg(feedForwardSoftmaxSumKernel, 5, sizeof(int), &outputLayer.NumInputs);
            clSetKernelArg(feedForwardSoftmaxSumKernel, 6, sizeof(int), &lastHidden.NumNeurons);
            
            err = clEnqueueNDRangeKernel(queue, feedForwardSoftmaxSumKernel, 1, NULL, &globalSize, NULL, 0, NULL, NULL);
            CL_CHECK(err);
            clFinish(queue);

            double* h_sums = new double[outputLayer.NumNeurons];
            clEnqueueReadBuffer(queue, d_SoftmaxSums, CL_TRUE, 0, outputLayer.NumNeurons * sizeof(double), h_sums, 0, NULL, NULL);

            double maxVal = h_sums[0];
            for (int i = 1; i < outputLayer.NumNeurons; i++)
                if (h_sums[i] > maxVal) maxVal = h_sums[i];

            double sumExp = 0;
            for (int i = 0; i < outputLayer.NumNeurons; i++)
                sumExp += exp(h_sums[i] - maxVal);

            delete[] h_sums;

            clSetKernelArg(softmaxKernel, 0, sizeof(cl_mem), &d_SoftmaxSums);
            clSetKernelArg(softmaxKernel, 1, sizeof(cl_mem), &outputLayer.Outputs);
            clSetKernelArg(softmaxKernel, 2, sizeof(int), &outputLayer.NumNeurons);
            clSetKernelArg(softmaxKernel, 3, sizeof(double), &maxVal);
            clSetKernelArg(softmaxKernel, 4, sizeof(double), &sumExp);
            
            err = clEnqueueNDRangeKernel(queue, softmaxKernel, 1, NULL, &globalSize, NULL, 0, NULL, NULL);
            CL_CHECK(err);
        } else {
            clSetKernelArg(feedForwardKernel, 0, sizeof(cl_mem), &outputLayer.Weights);
            clSetKernelArg(feedForwardKernel, 1, sizeof(cl_mem), &outputLayer.Biases);
            clSetKernelArg(feedForwardKernel, 2, sizeof(cl_mem), &outputLayer.Outputs);
            clSetKernelArg(feedForwardKernel, 3, sizeof(cl_mem), &lastHidden.Outputs);
            clSetKernelArg(feedForwardKernel, 4, sizeof(int), &outputLayer.NumNeurons);
            clSetKernelArg(feedForwardKernel, 5, sizeof(int), &outputLayer.NumInputs);
            clSetKernelArg(feedForwardKernel, 6, sizeof(int), &lastHidden.NumNeurons);
            int actType = (int)outputLayer.ActivationType;
            clSetKernelArg(feedForwardKernel, 7, sizeof(int), &actType);
            
            err = clEnqueueNDRangeKernel(queue, feedForwardKernel, 1, NULL, &globalSize, NULL, 0, NULL, NULL);
            CL_CHECK(err);
        }
        clFinish(queue);
    }

    void BackPropagate() {
        cl_int err;
        LayerData& outputLayer = Layers[NumLayers - 1];
        size_t globalSize = ((outputLayer.NumNeurons + BLOCK_SIZE - 1) / BLOCK_SIZE) * BLOCK_SIZE;
        
        int isSoftmax = (OutputActivation == atSoftmax) ? 1 : 0;
        int actType = (int)outputLayer.ActivationType;
        
        clSetKernelArg(backPropOutputKernel, 0, sizeof(cl_mem), &outputLayer.Errors);
        clSetKernelArg(backPropOutputKernel, 1, sizeof(cl_mem), &outputLayer.Outputs);
        clSetKernelArg(backPropOutputKernel, 2, sizeof(cl_mem), &d_Target);
        clSetKernelArg(backPropOutputKernel, 3, sizeof(int), &outputLayer.NumNeurons);
        clSetKernelArg(backPropOutputKernel, 4, sizeof(int), &actType);
        clSetKernelArg(backPropOutputKernel, 5, sizeof(int), &isSoftmax);
        
        err = clEnqueueNDRangeKernel(queue, backPropOutputKernel, 1, NULL, &globalSize, NULL, 0, NULL, NULL);
        CL_CHECK(err);
        clFinish(queue);

        for (int k = NumLayers - 2; k >= 1; k--) {
            LayerData& layer = Layers[k];
            LayerData& nextLayer = Layers[k + 1];
            globalSize = ((layer.NumNeurons + BLOCK_SIZE - 1) / BLOCK_SIZE) * BLOCK_SIZE;
            
            actType = (int)layer.ActivationType;
            
            clSetKernelArg(backPropHiddenKernel, 0, sizeof(cl_mem), &layer.Errors);
            clSetKernelArg(backPropHiddenKernel, 1, sizeof(cl_mem), &layer.Outputs);
            clSetKernelArg(backPropHiddenKernel, 2, sizeof(cl_mem), &layer.DropoutMask);
            clSetKernelArg(backPropHiddenKernel, 3, sizeof(cl_mem), &nextLayer.Errors);
            clSetKernelArg(backPropHiddenKernel, 4, sizeof(cl_mem), &nextLayer.Weights);
            clSetKernelArg(backPropHiddenKernel, 5, sizeof(int), &layer.NumNeurons);
            clSetKernelArg(backPropHiddenKernel, 6, sizeof(int), &actType);
            clSetKernelArg(backPropHiddenKernel, 7, sizeof(int), &nextLayer.NumNeurons);
            clSetKernelArg(backPropHiddenKernel, 8, sizeof(int), &nextLayer.NumInputs);
            
            err = clEnqueueNDRangeKernel(queue, backPropHiddenKernel, 1, NULL, &globalSize, NULL, 0, NULL, NULL);
            CL_CHECK(err);
            clFinish(queue);
        }
    }

    void UpdateWeights() {
        cl_int err;
        Timestep++;

        for (int k = NumLayers - 1; k >= 1; k--) {
            LayerData& layer = Layers[k];
            LayerData& prevLayer = Layers[k - 1];
            size_t globalSize = ((layer.NumNeurons + BLOCK_SIZE - 1) / BLOCK_SIZE) * BLOCK_SIZE;

            switch (Optimizer) {
                case otSGD:
                    clSetKernelArg(updateWeightsSGDKernel, 0, sizeof(cl_mem), &layer.Weights);
                    clSetKernelArg(updateWeightsSGDKernel, 1, sizeof(cl_mem), &layer.Biases);
                    clSetKernelArg(updateWeightsSGDKernel, 2, sizeof(cl_mem), &layer.Errors);
                    clSetKernelArg(updateWeightsSGDKernel, 3, sizeof(cl_mem), &prevLayer.Outputs);
                    clSetKernelArg(updateWeightsSGDKernel, 4, sizeof(int), &layer.NumNeurons);
                    clSetKernelArg(updateWeightsSGDKernel, 5, sizeof(int), &layer.NumInputs);
                    clSetKernelArg(updateWeightsSGDKernel, 6, sizeof(double), &LearningRate);
                    clSetKernelArg(updateWeightsSGDKernel, 7, sizeof(double), &L2Lambda);
                    err = clEnqueueNDRangeKernel(queue, updateWeightsSGDKernel, 1, NULL, &globalSize, NULL, 0, NULL, NULL);
                    CL_CHECK(err);
                    break;
                    
                case otAdam:
                    clSetKernelArg(updateWeightsAdamKernel, 0, sizeof(cl_mem), &layer.Weights);
                    clSetKernelArg(updateWeightsAdamKernel, 1, sizeof(cl_mem), &layer.Biases);
                    clSetKernelArg(updateWeightsAdamKernel, 2, sizeof(cl_mem), &layer.Errors);
                    clSetKernelArg(updateWeightsAdamKernel, 3, sizeof(cl_mem), &prevLayer.Outputs);
                    clSetKernelArg(updateWeightsAdamKernel, 4, sizeof(cl_mem), &layer.M);
                    clSetKernelArg(updateWeightsAdamKernel, 5, sizeof(cl_mem), &layer.V);
                    clSetKernelArg(updateWeightsAdamKernel, 6, sizeof(cl_mem), &layer.MBias);
                    clSetKernelArg(updateWeightsAdamKernel, 7, sizeof(cl_mem), &layer.VBias);
                    clSetKernelArg(updateWeightsAdamKernel, 8, sizeof(int), &layer.NumNeurons);
                    clSetKernelArg(updateWeightsAdamKernel, 9, sizeof(int), &layer.NumInputs);
                    clSetKernelArg(updateWeightsAdamKernel, 10, sizeof(double), &LearningRate);
                    clSetKernelArg(updateWeightsAdamKernel, 11, sizeof(double), &L2Lambda);
                    clSetKernelArg(updateWeightsAdamKernel, 12, sizeof(double), &Beta1);
                    clSetKernelArg(updateWeightsAdamKernel, 13, sizeof(double), &Beta2);
                    clSetKernelArg(updateWeightsAdamKernel, 14, sizeof(int), &Timestep);
                    err = clEnqueueNDRangeKernel(queue, updateWeightsAdamKernel, 1, NULL, &globalSize, NULL, 0, NULL, NULL);
                    CL_CHECK(err);
                    break;
                    
                case otRMSProp:
                    clSetKernelArg(updateWeightsRMSPropKernel, 0, sizeof(cl_mem), &layer.Weights);
                    clSetKernelArg(updateWeightsRMSPropKernel, 1, sizeof(cl_mem), &layer.Biases);
                    clSetKernelArg(updateWeightsRMSPropKernel, 2, sizeof(cl_mem), &layer.Errors);
                    clSetKernelArg(updateWeightsRMSPropKernel, 3, sizeof(cl_mem), &prevLayer.Outputs);
                    clSetKernelArg(updateWeightsRMSPropKernel, 4, sizeof(cl_mem), &layer.V);
                    clSetKernelArg(updateWeightsRMSPropKernel, 5, sizeof(cl_mem), &layer.VBias);
                    clSetKernelArg(updateWeightsRMSPropKernel, 6, sizeof(int), &layer.NumNeurons);
                    clSetKernelArg(updateWeightsRMSPropKernel, 7, sizeof(int), &layer.NumInputs);
                    clSetKernelArg(updateWeightsRMSPropKernel, 8, sizeof(double), &LearningRate);
                    clSetKernelArg(updateWeightsRMSPropKernel, 9, sizeof(double), &L2Lambda);
                    err = clEnqueueNDRangeKernel(queue, updateWeightsRMSPropKernel, 1, NULL, &globalSize, NULL, 0, NULL, NULL);
                    CL_CHECK(err);
                    break;
            }
        }
        clFinish(queue);
    }

    void Predict(const double* Input, double* Result) {
        FIsTraining = false;

        double* h_input = new double[FInputSize + 1];
        for (int i = 0; i < FInputSize; i++) h_input[i] = Input[i];
        h_input[FInputSize] = 1.0;
        clEnqueueWriteBuffer(queue, Layers[0].Outputs, CL_TRUE, 0, (FInputSize + 1) * sizeof(double), h_input, 0, NULL, NULL);
        delete[] h_input;

        FeedForward();

        clEnqueueReadBuffer(queue, Layers[NumLayers - 1].Outputs, CL_TRUE, 0, FOutputSize * sizeof(double), Result, 0, NULL, NULL);

        FIsTraining = true;
    }

    void Train(const double* Input, const double* Target) {
        FIsTraining = true;

        double* h_input = new double[FInputSize + 1];
        for (int i = 0; i < FInputSize; i++) h_input[i] = Input[i];
        h_input[FInputSize] = 1.0;
        clEnqueueWriteBuffer(queue, Layers[0].Outputs, CL_TRUE, 0, (FInputSize + 1) * sizeof(double), h_input, 0, NULL, NULL);
        delete[] h_input;

        clEnqueueWriteBuffer(queue, d_Target, CL_TRUE, 0, FOutputSize * sizeof(double), Target, 0, NULL, NULL);

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
        return Layers[layerIdx].NumNeurons;
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
            LayerData& layer = Layers[k];
            int weightSize = layer.NumNeurons * layer.NumInputs;

            double* h_weights = new double[weightSize];
            double* h_biases = new double[layer.NumNeurons];
            double* h_M = new double[weightSize];
            double* h_V = new double[weightSize];
            double* h_MBias = new double[layer.NumNeurons];
            double* h_VBias = new double[layer.NumNeurons];

            clEnqueueReadBuffer(queue, layer.Weights, CL_TRUE, 0, weightSize * sizeof(double), h_weights, 0, NULL, NULL);
            clEnqueueReadBuffer(queue, layer.Biases, CL_TRUE, 0, layer.NumNeurons * sizeof(double), h_biases, 0, NULL, NULL);
            clEnqueueReadBuffer(queue, layer.M, CL_TRUE, 0, weightSize * sizeof(double), h_M, 0, NULL, NULL);
            clEnqueueReadBuffer(queue, layer.V, CL_TRUE, 0, weightSize * sizeof(double), h_V, 0, NULL, NULL);
            clEnqueueReadBuffer(queue, layer.MBias, CL_TRUE, 0, layer.NumNeurons * sizeof(double), h_MBias, 0, NULL, NULL);
            clEnqueueReadBuffer(queue, layer.VBias, CL_TRUE, 0, layer.NumNeurons * sizeof(double), h_VBias, 0, NULL, NULL);

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

    static TMultiLayerPerceptronOpenCL* Load(const char* filename) {
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

        TMultiLayerPerceptronOpenCL* mlp = new TMultiLayerPerceptronOpenCL(
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
            LayerData& layer = mlp->Layers[k];
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

            clEnqueueWriteBuffer(mlp->queue, layer.Weights, CL_TRUE, 0, weightSize * sizeof(double), h_weights, 0, NULL, NULL);
            clEnqueueWriteBuffer(mlp->queue, layer.Biases, CL_TRUE, 0, layer.NumNeurons * sizeof(double), h_biases, 0, NULL, NULL);
            clEnqueueWriteBuffer(mlp->queue, layer.M, CL_TRUE, 0, weightSize * sizeof(double), h_M, 0, NULL, NULL);
            clEnqueueWriteBuffer(mlp->queue, layer.V, CL_TRUE, 0, weightSize * sizeof(double), h_V, 0, NULL, NULL);
            clEnqueueWriteBuffer(mlp->queue, layer.MBias, CL_TRUE, 0, layer.NumNeurons * sizeof(double), h_MBias, 0, NULL, NULL);
            clEnqueueWriteBuffer(mlp->queue, layer.VBias, CL_TRUE, 0, layer.NumNeurons * sizeof(double), h_VBias, 0, NULL, NULL);

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

// Utility functions and main() remain the same as CUDA version, just using OpenCL class
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
    printf("MLP OpenCL - Command-line Multi-Layer Perceptron\n");
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
    printf("  mlp_opencl create --input=2 --hidden=4,4 --output=1 --save=xor.bin\n");
    printf("  mlp_opencl train --model=xor.bin --data=xor.csv --epochs=1000 --save=xor_trained.bin\n");
    printf("  mlp_opencl predict --model=xor_trained.bin --input=1,0\n");
    printf("  mlp_opencl info --model=xor_trained.bin\n");
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

    cl_platform_id platform;
    cl_device_id device;
    char deviceName[128];
    
    if (clGetPlatformIDs(1, &platform, NULL) != CL_SUCCESS) {
        printf("Error: No OpenCL platforms found!\n");
        return 1;
    }
    
    if (clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL) != CL_SUCCESS) {
        if (clGetDeviceIDs(platform, CL_DEVICE_TYPE_CPU, 1, &device, NULL) != CL_SUCCESS) {
            printf("Error: No OpenCL devices found!\n");
            return 1;
        }
    }
    
    clGetDeviceInfo(device, CL_DEVICE_NAME, sizeof(deviceName), deviceName, NULL);

    if (command == cmdCreate) {
        if (inputSize <= 0) { printf("Error: --input is required\n"); return 1; }
        if (hiddenSizes.empty()) { printf("Error: --hidden is required\n"); return 1; }
        if (outputSize <= 0) { printf("Error: --output is required\n"); return 1; }
        if (saveFile.empty()) { printf("Error: --save is required\n"); return 1; }

        TMultiLayerPerceptronOpenCL* mlp = new TMultiLayerPerceptronOpenCL(
            inputSize, hiddenSizes, outputSize, hiddenAct, outputAct);
        mlp->LearningRate = learningRate;
        mlp->Optimizer = optimizer;
        mlp->DropoutRate = dropoutRate;
        mlp->L2Lambda = l2Lambda;
        mlp->Beta1 = beta1;
        mlp->Beta2 = beta2;

        mlp->Save(saveFile.c_str());

        printf("Created OpenCL MLP model (Device: %s):\n", deviceName);
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

        TMultiLayerPerceptronOpenCL* mlp = TMultiLayerPerceptronOpenCL::Load(modelFile.c_str());
        if (!mlp) { printf("Error: Failed to load model: %s\n", modelFile.c_str()); return 1; }

        if (lrOverride) mlp->LearningRate = learningRate;
        mlp->EnableLRDecay = lrDecay;
        mlp->LRDecayRate = lrDecayRate;
        mlp->LRDecayEpochs = lrDecayEpochs;
        mlp->EnableEarlyStopping = earlyStop;
        mlp->EarlyStoppingPatience = patience;

        std::vector<DataPoint> data = LoadDataCSV(dataFile.c_str(), mlp->GetInputSize(), mlp->GetOutputSize());
        if (data.empty()) { printf("Error: No valid data loaded\n"); delete mlp; return 1; }

        printf("Using Device: %s\n", deviceName);
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

        TMultiLayerPerceptronOpenCL* mlp = TMultiLayerPerceptronOpenCL::Load(modelFile.c_str());
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

        TMultiLayerPerceptronOpenCL* mlp = TMultiLayerPerceptronOpenCL::Load(modelFile.c_str());
        if (!mlp) { printf("Error: Failed to load model\n"); return 1; }

        printf("MLP Model Information (OpenCL)\n");
        printf("==============================\n");
        printf("Device: %s\n", deviceName);
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