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

#define CL_TARGET_OPENCL_VERSION 200

#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <random>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <iomanip>
#include <CL/cl.h>

using namespace std;

const double EPSILON = 1e-15;
const string MODEL_MAGIC = "MLPBKND01";

// Enums
enum TActivationType { atSigmoid, atTanh, atReLU, atSoftmax, atLinear };
enum TOptimizerType { otSGD, otAdam, otRMSProp };
enum TCommand { cmdNone, cmdCreate, cmdTrain, cmdPredict, cmdInfo, cmdHelp, cmdExportONNX, cmdImportONNX, cmdFeatureImportance };

// Type aliases
typedef vector<double> Darray;
typedef vector<double> TDoubleArray;
typedef vector<int> TIntArray;

// Structs
struct TDataPoint {
    Darray Input;
    Darray Target;
};
typedef vector<TDataPoint> TDataPointArray;

struct TNeuron {
    vector<double> Weights;
    double Bias;
    double Output;
    double Error;
    vector<double> M;
    vector<double> V;
    double MBias;
    double VBias;
};

struct TLayer {
    vector<TNeuron> Neurons;
    TActivationType ActivationType;
    vector<bool> DropoutMask;
    // Batch normalization parameters
    vector<double> Gamma;         // Scale parameter
    vector<double> Beta;          // Shift parameter  
    vector<double> RunningMean;   // Running mean for inference
    vector<double> RunningVar;    // Running variance for inference
};

// OpenCL Kernel source
const char* kernelSource = R"(
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
        }
        
        output[neuronIdx] = result;
    }
}

__kernel void softmaxKernel(
    __global const double* input,
    __global double* output,
    int size)
{
    int idx = get_global_id(0);
    
    if (idx == 0) {
        double maxVal = input[0];
        for (int i = 1; i < size; i++) {
            if (input[i] > maxVal) maxVal = input[i];
        }
        
        double sum = 0.0;
        for (int i = 0; i < size; i++) {
            double exp_val = exp(input[i] - maxVal);
            output[i] = exp_val;
            sum += exp_val;
        }
        
        for (int i = 0; i < size; i++) {
            output[i] = output[i] / sum;
            if (output[i] < 1e-15) output[i] = 1e-15;
            else if (output[i] > 1.0 - 1e-15) output[i] = 1.0 - 1e-15;
        }
    }
}

__kernel void batchNormKernel(
    __global double* data,
    __global const double* gamma,
    __global const double* beta,
    __global const double* running_mean,
    __global const double* running_var,
    int size,
    double epsilon)
{
    int idx = get_global_id(0);
    
    if (idx < size) {
        double x_norm = (data[idx] - running_mean[idx]) / sqrt(running_var[idx] + epsilon);
        data[idx] = gamma[idx] * x_norm + beta[idx];
    }
}
)";

// Activation Functions
double Sigmoid(double x) {
    if (x < -500) return 0;
    if (x > 500) return 1;
    return 1 / (1 + exp(-x));
}

double DSigmoid(double x) {
    return x * (1 - x);
}

double TanhActivation(double x) {
    return tanh(x);
}

double DTanh(double x) {
    return 1 - (x * x);
}

double ReLU(double x) {
    return (x > 0) ? x : 0;
}

double DReLU(double x) {
    return (x > 0) ? 1 : 0;
}

Darray Softmax(const Darray& x) {
    Darray result(x.size());
    Darray expValues(x.size());
    
    double maxVal = x[0];
    for (size_t i = 1; i < x.size(); i++)
        if (x[i] > maxVal) maxVal = x[i];
    
    double sum = 0;
    for (size_t i = 0; i < x.size(); i++) {
        expValues[i] = exp(x[i] - maxVal);
        sum += expValues[i];
    }
    
    for (size_t i = 0; i < x.size(); i++) {
        result[i] = expValues[i] / sum;
        if (result[i] < EPSILON) result[i] = EPSILON;
        else if (result[i] > 1 - EPSILON) result[i] = 1 - EPSILON;
    }
    
    return result;
}

double ApplyActivation(double x, TActivationType actType) {
    switch (actType) {
        case atSigmoid: return Sigmoid(x);
        case atTanh: return TanhActivation(x);
        case atReLU: return ReLU(x);
        case atLinear: return x;
        default: return Sigmoid(x);
    }
}

double ApplyActivationDerivative(double x, TActivationType actType) {
    switch (actType) {
        case atSigmoid: return DSigmoid(x);
        case atTanh: return DTanh(x);
        case atReLU: return DReLU(x);
        case atLinear: return 1.0;
        default: return DSigmoid(x);
    }
}

int MaxIndex(const Darray& arr) {
    int result = 0;
    for (size_t i = 1; i < arr.size(); i++)
        if (arr[i] > arr[result]) result = i;
    return result;
}

// Helper Functions
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

// OpenCL Context Wrapper
class TOpenCLContext {
public:
    cl_platform_id platform;
    cl_device_id device;
    cl_context context;
    cl_command_queue queue;
    cl_program program;
    cl_kernel kernelFeedForward;
    cl_kernel kernelSoftmax;
    cl_kernel kernelBatchNorm;
    
    TOpenCLContext() : platform(nullptr), device(nullptr), context(nullptr), 
                      queue(nullptr), program(nullptr), kernelFeedForward(nullptr), 
                      kernelSoftmax(nullptr), kernelBatchNorm(nullptr) {
        cl_int err;
        cl_uint numPlatforms = 0;
        
        // Try to initialize OpenCL (silently fail if not available)
        try {
            // Get platform
            if (clGetPlatformIDs(1, &platform, &numPlatforms) != CL_SUCCESS || numPlatforms == 0) {
                throw runtime_error("No OpenCL platforms found");
            }
            
            // Get device (try GPU first, then CPU)
            if (clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, nullptr) != CL_SUCCESS) {
                if (clGetDeviceIDs(platform, CL_DEVICE_TYPE_CPU, 1, &device, nullptr) != CL_SUCCESS) {
                    throw runtime_error("No OpenCL devices found");
                }
            }
            
            // Create context
            context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &err);
            if (err != CL_SUCCESS) {
                throw runtime_error("Failed to create OpenCL context");
            }
            
            // Create command queue using OpenCL 2.0 API
            cl_queue_properties properties[] = { 0 };
            queue = clCreateCommandQueueWithProperties(context, device, properties, &err);
            if (err != CL_SUCCESS) {
                throw runtime_error("Failed to create command queue");
            }
            
            // Create program
            program = clCreateProgramWithSource(context, 1, &kernelSource, nullptr, &err);
            if (err != CL_SUCCESS) {
                throw runtime_error("Failed to create OpenCL program");
            }
            
            // Build program
            err = clBuildProgram(program, 1, &device, nullptr, nullptr, nullptr);
            if (err != CL_SUCCESS) {
                throw runtime_error("Failed to build OpenCL program");
            }
            
            // Create kernels
            kernelFeedForward = clCreateKernel(program, "feedForwardLayer", &err);
            if (err != CL_SUCCESS) throw runtime_error("Failed to create feedForwardLayer kernel");
            
            kernelSoftmax = clCreateKernel(program, "softmaxKernel", &err);
            if (err != CL_SUCCESS) throw runtime_error("Failed to create softmaxKernel kernel");
            
            kernelBatchNorm = clCreateKernel(program, "batchNormKernel", &err);
            if (err != CL_SUCCESS) throw runtime_error("Failed to create batchNormKernel kernel");
        } catch (const exception& e) {
            // Silently fail and continue with CPU fallback
            // OpenCL initialization is optional; CPU computation works fine
            // Clean up any partially initialized resources
            if (kernelFeedForward) clReleaseKernel(kernelFeedForward);
            if (kernelSoftmax) clReleaseKernel(kernelSoftmax);
            if (kernelBatchNorm) clReleaseKernel(kernelBatchNorm);
            if (program) clReleaseProgram(program);
            if (queue) clReleaseCommandQueue(queue);
            if (context) clReleaseContext(context);
            // Resources are already nullified by initialization list
        }
    }
    
    ~TOpenCLContext() {
        if (kernelFeedForward) clReleaseKernel(kernelFeedForward);
        if (kernelSoftmax) clReleaseKernel(kernelSoftmax);
        if (kernelBatchNorm) clReleaseKernel(kernelBatchNorm);
        if (program) clReleaseProgram(program);
        if (queue) clReleaseCommandQueue(queue);
        if (context) clReleaseContext(context);
    }
};

// MLP Class with OpenCL
class TMultiLayerPerceptron {
private:
    TLayer FInputLayer;
    vector<TLayer> FHiddenLayers;
    TLayer FOutputLayer;
    TIntArray FHiddenSizes;
    int FInputSize;
    int FOutputSize;
    bool FIsTraining;
    TOpenCLContext* FOpenCL;
    
    void InitializeLayer(TLayer& layer, int numNeurons, int numInputs, TActivationType actType);
    void FeedForward();
    void FeedForwardGPU();
    void BackPropagate(const Darray& target);
    void UpdateWeights();
    void UpdateNeuronWeightsSGD(TNeuron& neuron, const Darray& prevOutputs);
    void UpdateNeuronWeightsAdam(TNeuron& neuron, const Darray& prevOutputs);
    void UpdateNeuronWeightsRMSProp(TNeuron& neuron, const Darray& prevOutputs);
    void ApplyDropout(TLayer& layer);
    Darray InitializeWeights(int numInputs, int numOutputs, TActivationType actType);
    
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
    bool UseGPU;
    bool UseBatchNorm;
    double BNMomentum;
    double BNEpsilon;
    
    TMultiLayerPerceptron(int inputSize, const TIntArray& hiddenSizes, int outputSize,
                          TActivationType hiddenAct = atSigmoid, TActivationType outputAct = atSigmoid,
                          bool useGPU = true, bool useBatchNorm = false);
    ~TMultiLayerPerceptron();
    
    Darray Predict(const Darray& input);
    void Train(const Darray& input, const Darray& target);
    void TrainEpoch(TDataPointArray& data, int batchSize);
    double ComputeLoss(const Darray& predicted, const Darray& target);
    void SaveMLPModel(const string& filename);
    void Save(const string& filename);
    void SaveModelToJSON(const string& filename);
    void LoadModelFromJSON(const string& filename);
    string Array1DToJSON(const Darray& arr);
    
    // Batch normalization methods
    void BatchNormForwardCPU(TLayer& layer, bool training);
    void BatchNormBackwardCPU(TLayer& layer, Darray& dGamma, Darray& dBeta, Darray& dInput);
    void BatchNormForwardGPU(TLayer& layer);
    void InitializeBatchNormParams(TLayer& layer, int size);
    
    // ONNX export/import
    void ExportToONNX(const string& filename);
    void ImportFromONNX(const string& filename);
    
    // Feature importance
    Darray CalculateFeatureImportance();
    
    TLayer GetInputLayer() const { return FInputLayer; }
    TLayer GetOutputLayer() const { return FOutputLayer; }
    TLayer GetHiddenLayer(int index) const;
    int GetHiddenLayerCount() const { return FHiddenLayers.size(); }
    int GetInputSize() const { return FInputSize; }
    int GetOutputSize() const { return FOutputSize; }
};

TMultiLayerPerceptron::TMultiLayerPerceptron(int inputSize, const TIntArray& hiddenSizes, int outputSize,
                                              TActivationType hiddenAct, TActivationType outputAct, bool useGPU, bool useBatchNorm)
     : FInputSize(inputSize), FOutputSize(outputSize), FIsTraining(false), UseGPU(false),  // Force CPU mode - OpenCL support needs more debugging
      LearningRate(0.1), MaxIterations(100), Optimizer(otSGD),
      HiddenActivation(hiddenAct), OutputActivation(outputAct),
      DropoutRate(0.0), L2Lambda(0.0), Beta1(0.9), Beta2(0.999),
      Timestep(0), EnableLRDecay(false), LRDecayRate(0.95), LRDecayEpochs(10),
      EnableEarlyStopping(false), EarlyStoppingPatience(10),
      UseBatchNorm(useBatchNorm), BNMomentum(0.1), BNEpsilon(1e-5) {
    
    FHiddenSizes = hiddenSizes;
    FOpenCL = nullptr;
    
    if (UseGPU) {
        FOpenCL = new TOpenCLContext();
        // Check if OpenCL successfully initialized
        if (!FOpenCL->context) {
            delete FOpenCL;
            FOpenCL = nullptr;
            UseGPU = false;
        }
    }
    
    InitializeLayer(FInputLayer, inputSize, 1, atSigmoid);
    
    FHiddenLayers.resize(hiddenSizes.size());
    for (size_t i = 0; i < hiddenSizes.size(); i++) {
        int numInputs = (i == 0) ? inputSize : hiddenSizes[i-1];
        InitializeLayer(FHiddenLayers[i], hiddenSizes[i], numInputs, hiddenAct);
        if (UseBatchNorm) {
            InitializeBatchNormParams(FHiddenLayers[i], hiddenSizes[i]);
        }
    }
    
    int outputLayerInput = hiddenSizes.empty() ? inputSize : hiddenSizes.back();
    InitializeLayer(FOutputLayer, outputSize, outputLayerInput, outputAct);
}

TMultiLayerPerceptron::~TMultiLayerPerceptron() {
    if (FOpenCL) delete FOpenCL;
}

void TMultiLayerPerceptron::InitializeLayer(TLayer& layer, int numNeurons, int numInputs, TActivationType actType) {
    layer.Neurons.resize(numNeurons);
    layer.ActivationType = actType;
    layer.DropoutMask.resize(numNeurons);
    
    Darray weights = InitializeWeights(numInputs, numNeurons, actType);
    
    for (int i = 0; i < numNeurons; i++) {
        layer.Neurons[i].Weights.resize(numInputs);
        layer.Neurons[i].Bias = 0.0;
        layer.Neurons[i].Output = 0.0;
        layer.Neurons[i].Error = 0.0;
        layer.Neurons[i].M.resize(numInputs, 0.0);
        layer.Neurons[i].V.resize(numInputs, 0.0);
        layer.Neurons[i].MBias = 0.0;
        layer.Neurons[i].VBias = 0.0;
        
        for (int j = 0; j < numInputs; j++) {
            layer.Neurons[i].Weights[j] = weights[i * numInputs + j];
        }
        
        layer.DropoutMask[i] = true;
    }
}

Darray TMultiLayerPerceptron::InitializeWeights(int numInputs, int numOutputs, TActivationType actType) {
    Darray weights(numInputs * numOutputs);
    random_device rd;
    mt19937 gen(rd());
    
    double limit = sqrt(6.0 / (numInputs + numOutputs));
    uniform_real_distribution<> dis(-limit, limit);
    
    for (size_t i = 0; i < weights.size(); i++) {
        weights[i] = dis(gen);
    }
    
    return weights;
}

Darray TMultiLayerPerceptron::Predict(const Darray& input) {
    if (input.size() != (size_t)FInputSize) {
        throw runtime_error("Input size mismatch");
    }
    
    for (int i = 0; i < FInputSize; i++) {
        FInputLayer.Neurons[i].Output = input[i];
    }
    
    if (UseGPU && FOpenCL) {
        FeedForwardGPU();
    } else {
        FeedForward();
    }
    
    Darray result(FOutputSize);
    for (int i = 0; i < FOutputSize; i++) {
        result[i] = FOutputLayer.Neurons[i].Output;
    }
    
    return result;
}

void TMultiLayerPerceptron::FeedForwardGPU() {
     // Fall back to CPU if OpenCL is not available
     if (!FOpenCL || !FOpenCL->context) {
         FeedForward();
         return;
     }
     
     cl_int err;
     
     // Process hidden layers
     for (size_t h = 0; h < FHiddenLayers.size(); h++) {
         TLayer& currentLayer = FHiddenLayers[h];
         
         // Get previous layer output
         Darray prevOutput;
         if (h == 0) {
             prevOutput.resize(FInputLayer.Neurons.size());
             for (size_t i = 0; i < FInputLayer.Neurons.size(); i++) {
                 prevOutput[i] = FInputLayer.Neurons[i].Output;
             }
         } else {
             prevOutput.resize(FHiddenLayers[h-1].Neurons.size());
             for (size_t i = 0; i < FHiddenLayers[h-1].Neurons.size(); i++) {
                 prevOutput[i] = FHiddenLayers[h-1].Neurons[i].Output;
             }
         }
         
         int inputSize = prevOutput.size();
         int outputSize = currentLayer.Neurons.size();
         
         // Flatten weights and biases
         Darray weights(outputSize * inputSize);
         Darray biases(outputSize);
         for (int i = 0; i < outputSize; i++) {
             biases[i] = currentLayer.Neurons[i].Bias;
             for (int j = 0; j < inputSize; j++) {
                 weights[i * inputSize + j] = currentLayer.Neurons[i].Weights[j];
             }
         }
         
         // Create GPU buffers
         cl_mem inputBuffer = clCreateBuffer(FOpenCL->context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                           inputSize * sizeof(double), prevOutput.data(), &err);
        cl_mem weightsBuffer = clCreateBuffer(FOpenCL->context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                             weights.size() * sizeof(double), weights.data(), &err);
        cl_mem biasesBuffer = clCreateBuffer(FOpenCL->context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                            biases.size() * sizeof(double), biases.data(), &err);
        cl_mem outputBuffer = clCreateBuffer(FOpenCL->context, CL_MEM_WRITE_ONLY,
                                            outputSize * sizeof(double), nullptr, &err);
        
        // Set kernel arguments
        clSetKernelArg(FOpenCL->kernelFeedForward, 0, sizeof(cl_mem), &inputBuffer);
        clSetKernelArg(FOpenCL->kernelFeedForward, 1, sizeof(cl_mem), &weightsBuffer);
        clSetKernelArg(FOpenCL->kernelFeedForward, 2, sizeof(cl_mem), &biasesBuffer);
        clSetKernelArg(FOpenCL->kernelFeedForward, 3, sizeof(cl_mem), &outputBuffer);
        clSetKernelArg(FOpenCL->kernelFeedForward, 4, sizeof(int), &inputSize);
        clSetKernelArg(FOpenCL->kernelFeedForward, 5, sizeof(int), &outputSize);
        int actType = (int)currentLayer.ActivationType;
        clSetKernelArg(FOpenCL->kernelFeedForward, 6, sizeof(int), &actType);
        
        // Execute kernel
        size_t globalWorkSize = outputSize;
        clEnqueueNDRangeKernel(FOpenCL->queue, FOpenCL->kernelFeedForward, 1, nullptr,
                               &globalWorkSize, nullptr, 0, nullptr, nullptr);
        
        // Read results back
        Darray output(outputSize);
        clEnqueueReadBuffer(FOpenCL->queue, outputBuffer, CL_TRUE, 0,
                           outputSize * sizeof(double), output.data(), 0, nullptr, nullptr);
        
        // Update layer outputs
        for (int i = 0; i < outputSize; i++) {
            currentLayer.Neurons[i].Output = output[i];
        }
        
        // Clean up
        clReleaseMemObject(inputBuffer);
        clReleaseMemObject(weightsBuffer);
        clReleaseMemObject(biasesBuffer);
        clReleaseMemObject(outputBuffer);
    }
    
    // Process output layer
    Darray prevOutput;
    if (FHiddenLayers.empty()) {
        prevOutput.resize(FInputLayer.Neurons.size());
        for (size_t i = 0; i < FInputLayer.Neurons.size(); i++) {
            prevOutput[i] = FInputLayer.Neurons[i].Output;
        }
    } else {
        prevOutput.resize(FHiddenLayers.back().Neurons.size());
        for (size_t i = 0; i < FHiddenLayers.back().Neurons.size(); i++) {
            prevOutput[i] = FHiddenLayers.back().Neurons[i].Output;
        }
    }
    
    int inputSize = prevOutput.size();
    int outputSize = FOutputLayer.Neurons.size();
    
    // Flatten weights and biases
    Darray weights(outputSize * inputSize);
    Darray biases(outputSize);
    for (int i = 0; i < outputSize; i++) {
        biases[i] = FOutputLayer.Neurons[i].Bias;
        for (int j = 0; j < inputSize; j++) {
            weights[i * inputSize + j] = FOutputLayer.Neurons[i].Weights[j];
        }
    }
    
    // Create GPU buffers
    cl_mem inputBuffer = clCreateBuffer(FOpenCL->context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                       inputSize * sizeof(double), prevOutput.data(), &err);
    cl_mem weightsBuffer = clCreateBuffer(FOpenCL->context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                         weights.size() * sizeof(double), weights.data(), &err);
    cl_mem biasesBuffer = clCreateBuffer(FOpenCL->context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                        biases.size() * sizeof(double), biases.data(), &err);
    cl_mem outputBuffer = clCreateBuffer(FOpenCL->context, CL_MEM_WRITE_ONLY,
                                        outputSize * sizeof(double), nullptr, &err);
    
    // Set kernel arguments
    clSetKernelArg(FOpenCL->kernelFeedForward, 0, sizeof(cl_mem), &inputBuffer);
    clSetKernelArg(FOpenCL->kernelFeedForward, 1, sizeof(cl_mem), &weightsBuffer);
    clSetKernelArg(FOpenCL->kernelFeedForward, 2, sizeof(cl_mem), &biasesBuffer);
    clSetKernelArg(FOpenCL->kernelFeedForward, 3, sizeof(cl_mem), &outputBuffer);
    clSetKernelArg(FOpenCL->kernelFeedForward, 4, sizeof(int), &inputSize);
    clSetKernelArg(FOpenCL->kernelFeedForward, 5, sizeof(int), &outputSize);
    int actType = (int)FOutputLayer.ActivationType;
    clSetKernelArg(FOpenCL->kernelFeedForward, 6, sizeof(int), &actType);
    
    // Execute kernel
    size_t globalWorkSize = outputSize;
    clEnqueueNDRangeKernel(FOpenCL->queue, FOpenCL->kernelFeedForward, 1, nullptr,
                           &globalWorkSize, nullptr, 0, nullptr, nullptr);
    
    // Read results back
    Darray output(outputSize);
    clEnqueueReadBuffer(FOpenCL->queue, outputBuffer, CL_TRUE, 0,
                       outputSize * sizeof(double), output.data(), 0, nullptr, nullptr);
    
    // Update output layer
    for (int i = 0; i < outputSize; i++) {
        FOutputLayer.Neurons[i].Output = output[i];
    }
    
    // Apply softmax if needed
    if (FOutputLayer.ActivationType == atSoftmax) {
        Darray rawOutput(outputSize);
        for (int i = 0; i < outputSize; i++) {
            rawOutput[i] = FOutputLayer.Neurons[i].Output;
        }
        Darray softmaxResult = Softmax(rawOutput);
        for (int i = 0; i < outputSize; i++) {
            FOutputLayer.Neurons[i].Output = softmaxResult[i];
        }
    }
    
    // Clean up
    clReleaseMemObject(inputBuffer);
    clReleaseMemObject(weightsBuffer);
    clReleaseMemObject(biasesBuffer);
    clReleaseMemObject(outputBuffer);
}

void TMultiLayerPerceptron::FeedForward() {
    for (size_t h = 0; h < FHiddenLayers.size(); h++) {
        TLayer& currentLayer = FHiddenLayers[h];
        
        Darray prevOutput;
        if (h == 0) {
            prevOutput.resize(FInputLayer.Neurons.size());
            for (size_t i = 0; i < FInputLayer.Neurons.size(); i++) {
                prevOutput[i] = FInputLayer.Neurons[i].Output;
            }
        } else {
            prevOutput.resize(FHiddenLayers[h-1].Neurons.size());
            for (size_t i = 0; i < FHiddenLayers[h-1].Neurons.size(); i++) {
                prevOutput[i] = FHiddenLayers[h-1].Neurons[i].Output;
            }
        }
        
        for (size_t i = 0; i < currentLayer.Neurons.size(); i++) {
            double sum = currentLayer.Neurons[i].Bias;
            for (size_t j = 0; j < prevOutput.size(); j++) {
                sum += currentLayer.Neurons[i].Weights[j] * prevOutput[j];
            }
            currentLayer.Neurons[i].Output = ApplyActivation(sum, currentLayer.ActivationType);
        }
    }
    
    Darray prevOutput;
    if (FHiddenLayers.empty()) {
        prevOutput.resize(FInputLayer.Neurons.size());
        for (size_t i = 0; i < FInputLayer.Neurons.size(); i++) {
            prevOutput[i] = FInputLayer.Neurons[i].Output;
        }
    } else {
        prevOutput.resize(FHiddenLayers.back().Neurons.size());
        for (size_t i = 0; i < FHiddenLayers.back().Neurons.size(); i++) {
            prevOutput[i] = FHiddenLayers.back().Neurons[i].Output;
        }
    }
    
    for (size_t i = 0; i < FOutputLayer.Neurons.size(); i++) {
        double sum = FOutputLayer.Neurons[i].Bias;
        for (size_t j = 0; j < prevOutput.size(); j++) {
            sum += FOutputLayer.Neurons[i].Weights[j] * prevOutput[j];
        }
        FOutputLayer.Neurons[i].Output = ApplyActivation(sum, FOutputLayer.ActivationType);
    }
    
    if (FOutputLayer.ActivationType == atSoftmax) {
        Darray rawOutput(FOutputSize);
        for (int i = 0; i < FOutputSize; i++) {
            rawOutput[i] = FOutputLayer.Neurons[i].Output;
        }
        Darray softmaxResult = Softmax(rawOutput);
        for (int i = 0; i < FOutputSize; i++) {
            FOutputLayer.Neurons[i].Output = softmaxResult[i];
        }
    }
}

void TMultiLayerPerceptron::BackPropagate(const Darray& target) {
}

void TMultiLayerPerceptron::UpdateWeights() {
}

void TMultiLayerPerceptron::UpdateNeuronWeightsSGD(TNeuron& neuron, const Darray& prevOutputs) {
}

void TMultiLayerPerceptron::UpdateNeuronWeightsAdam(TNeuron& neuron, const Darray& prevOutputs) {
}

void TMultiLayerPerceptron::UpdateNeuronWeightsRMSProp(TNeuron& neuron, const Darray& prevOutputs) {
}

void TMultiLayerPerceptron::ApplyDropout(TLayer& layer) {
}

void TMultiLayerPerceptron::Train(const Darray& input, const Darray& target) {
    if (UseGPU && FOpenCL) {
        FeedForwardGPU();
    } else {
        FeedForward();
    }
    BackPropagate(target);
    UpdateWeights();
}

void TMultiLayerPerceptron::TrainEpoch(TDataPointArray& data, int batchSize) {
}

double TMultiLayerPerceptron::ComputeLoss(const Darray& predicted, const Darray& target) {
    double loss = 0.0;
    for (size_t i = 0; i < predicted.size(); i++) {
        double diff = predicted[i] - target[i];
        loss += diff * diff;
    }
    return loss / predicted.size();
}

void TMultiLayerPerceptron::SaveMLPModel(const string& filename) {
}

void TMultiLayerPerceptron::Save(const string& filename) {
    SaveModelToJSON(filename);
}

string TMultiLayerPerceptron::Array1DToJSON(const Darray& arr) {
    stringstream ss;
    ss << "[";
    for (size_t i = 0; i < arr.size(); i++) {
        if (i > 0) ss << ",";
        ss << fixed << setprecision(10) << arr[i];
    }
    ss << "]";
    return ss.str();
}

void TMultiLayerPerceptron::SaveModelToJSON(const string& filename) {
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
    f << "  \"batch_norm\": " << (UseBatchNorm ? "true" : "false") << "," << endl;
    
    f << "  \"input_layer\": {" << endl;
    f << "    \"neuron_count\": " << FInputLayer.Neurons.size() << endl;
    f << "  }," << endl;
    
    f << "  \"hidden_layers\": [" << endl;
    for (size_t h = 0; h < FHiddenLayers.size(); h++) {
        f << "    {" << endl;
        f << "      \"neuron_count\": " << FHiddenLayers[h].Neurons.size() << "," << endl;
        f << "      \"neurons\": [" << endl;
        for (size_t j = 0; j < FHiddenLayers[h].Neurons.size(); j++) {
            f << "        {" << endl;
            f << "          \"weights\": " << Array1DToJSON(FHiddenLayers[h].Neurons[j].Weights) << "," << endl;
            f << setprecision(10);
            f << "          \"bias\": " << FHiddenLayers[h].Neurons[j].Bias << endl;
            f << "        }";
            if (j < FHiddenLayers[h].Neurons.size() - 1) f << ",";
            f << endl;
        }
        f << "      ]," << endl;
        f << "      \"biases\": [";
        for (size_t j = 0; j < FHiddenLayers[h].Neurons.size(); j++) {
            if (j > 0) f << ",";
            f << fixed << setprecision(10) << FHiddenLayers[h].Neurons[j].Bias;
        }
        f << "]" << endl;
        f << "    }";
        if (h < FHiddenLayers.size() - 1) f << ",";
        f << endl;
    }
    f << "  ]," << endl;
    
    f << "  \"output_layer\": {" << endl;
    f << "    \"neuron_count\": " << FOutputLayer.Neurons.size() << "," << endl;
    f << "    \"neurons\": [" << endl;
    for (size_t i = 0; i < FOutputLayer.Neurons.size(); i++) {
        f << "      {" << endl;
        f << "        \"weights\": " << Array1DToJSON(FOutputLayer.Neurons[i].Weights) << "," << endl;
        f << setprecision(10);
        f << "        \"bias\": " << FOutputLayer.Neurons[i].Bias << endl;
        f << "      }";
        if (i < FOutputLayer.Neurons.size() - 1) f << ",";
        f << endl;
    }
    f << "    ]," << endl;
    f << "    \"biases\": [";
    for (size_t i = 0; i < FOutputLayer.Neurons.size(); i++) {
        if (i > 0) f << ",";
        f << fixed << setprecision(10) << FOutputLayer.Neurons[i].Bias;
    }
    f << "]" << endl;
    f << "  }" << endl;
    f << "}" << endl;
    
    f.close();
}

void TMultiLayerPerceptron::LoadModelFromJSON(const string& filename) {
    ifstream f(filename);
    if (!f.is_open()) {
        throw runtime_error("Could not open file: " + filename);
    }
    
    string content((istreambuf_iterator<char>(f)), istreambuf_iterator<char>());
    f.close();
    
    // Simple JSON parser helpers
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
    
    // Load basic parameters
    FInputSize = getJsonInt("input_size");
    FOutputSize = getJsonInt("output_size");
    HiddenActivation = (TActivationType)getJsonInt("hidden_activation");
    OutputActivation = (TActivationType)getJsonInt("output_activation");
    LearningRate = getJsonNumber("learning_rate");
    Optimizer = (TOptimizerType)getJsonInt("optimizer");
    DropoutRate = getJsonNumber("dropout_rate");
    L2Lambda = getJsonNumber("l2_lambda");
    Beta1 = getJsonNumber("beta1");
    Beta2 = getJsonNumber("beta2");
    
    size_t batchNormPos = content.find("\"batch_norm\":");
    if (batchNormPos != string::npos) {
        size_t colonPos = content.find(":", batchNormPos);
        size_t valueStart = content.find_first_not_of(" \t\n\r", colonPos + 1);
        UseBatchNorm = (content.substr(valueStart, 4) == "true");
    } else {
        UseBatchNorm = false;
    }
    
    // Parse hidden_sizes array
    FHiddenSizes.clear();
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
                FHiddenSizes.push_back(stoi(arrayContent.substr(numStart, numEnd - numStart)));
            } catch (...) {}
            pos = numEnd;
        }
    }
    
    // Re-initialize input layer
     FInputLayer.Neurons.clear();
     FInputLayer.Neurons.resize(FInputSize);
     for (int i = 0; i < FInputSize; i++) {
         FInputLayer.Neurons[i].Output = 0.0;
         FInputLayer.Neurons[i].Bias = 0.0;
         FInputLayer.Neurons[i].Error = 0.0;
         FInputLayer.Neurons[i].Weights.clear();
         FInputLayer.Neurons[i].M.clear();
         FInputLayer.Neurons[i].V.clear();
         FInputLayer.Neurons[i].MBias = 0.0;
         FInputLayer.Neurons[i].VBias = 0.0;
     }
     FInputLayer.ActivationType = atSigmoid;
     FInputLayer.DropoutMask.clear();
     FInputLayer.DropoutMask.resize(FInputSize, true);
    
    // Re-initialize layers from loaded parameters
    FHiddenLayers.clear();
    FHiddenLayers.resize(FHiddenSizes.size());
    for (size_t i = 0; i < FHiddenSizes.size(); i++) {
        int numInputs = (i == 0) ? FInputSize : FHiddenSizes[i-1];
        FHiddenLayers[i].Neurons.clear();
        FHiddenLayers[i].Neurons.resize(FHiddenSizes[i]);
        for (int j = 0; j < FHiddenSizes[i]; j++) {
            FHiddenLayers[i].Neurons[j].Weights.resize(numInputs);
            FHiddenLayers[i].Neurons[j].Bias = 0.0;
            FHiddenLayers[i].Neurons[j].Output = 0.0;
            FHiddenLayers[i].Neurons[j].Error = 0.0;
            FHiddenLayers[i].Neurons[j].M.resize(numInputs, 0.0);
            FHiddenLayers[i].Neurons[j].V.resize(numInputs, 0.0);
            FHiddenLayers[i].Neurons[j].MBias = 0.0;
            FHiddenLayers[i].Neurons[j].VBias = 0.0;
        }
        FHiddenLayers[i].ActivationType = HiddenActivation;
        FHiddenLayers[i].DropoutMask.resize(FHiddenSizes[i]);
    }
    
    int outputLayerInput = FHiddenSizes.empty() ? FInputSize : FHiddenSizes.back();
    FOutputLayer.Neurons.clear();
    FOutputLayer.Neurons.resize(FOutputSize);
    for (int j = 0; j < FOutputSize; j++) {
        FOutputLayer.Neurons[j].Weights.resize(outputLayerInput);
        FOutputLayer.Neurons[j].Bias = 0.0;
        FOutputLayer.Neurons[j].Output = 0.0;
        FOutputLayer.Neurons[j].Error = 0.0;
        FOutputLayer.Neurons[j].M.resize(outputLayerInput, 0.0);
        FOutputLayer.Neurons[j].V.resize(outputLayerInput, 0.0);
        FOutputLayer.Neurons[j].MBias = 0.0;
        FOutputLayer.Neurons[j].VBias = 0.0;
    }
    FOutputLayer.ActivationType = OutputActivation;
    FOutputLayer.DropoutMask.resize(FOutputSize);
    
    // Now load weights and biases from JSON
    size_t searchPos = 0;
    
    // Parse hidden layers
    size_t hiddenStart = content.find("\"hidden_layers\": [");
    size_t hiddenEnd = content.find("]", hiddenStart);
    if (hiddenStart != string::npos && hiddenEnd != string::npos) {
        searchPos = hiddenStart;
        for (size_t h = 0; h < FHiddenLayers.size(); h++) {
            for (size_t n = 0; n < FHiddenLayers[h].Neurons.size(); n++) {
                size_t wPos = content.find("\"weights\": [", searchPos);
                if (wPos != string::npos && wPos < hiddenEnd) {
                    size_t wEnd = content.find("]", wPos);
                    string weightsStr = content.substr(wPos, wEnd - wPos + 1);
                    FHiddenLayers[h].Neurons[n].Weights = parseArray(weightsStr);
                    searchPos = wEnd + 1;
                }
                
                size_t bPos = content.find("\"bias\": ", searchPos);
                if (bPos != string::npos && bPos < hiddenEnd) {
                    size_t bEnd = content.find_first_of(",}", bPos + 8);
                    string biasStr = content.substr(bPos + 8, bEnd - bPos - 8);
                    biasStr.erase(0, biasStr.find_first_not_of(" \t\n\r"));
                    biasStr.erase(biasStr.find_last_not_of(" \t\n\r") + 1);
                    try {
                        FHiddenLayers[h].Neurons[n].Bias = stod(biasStr);
                    } catch (...) {}
                    searchPos = bEnd + 1;
                }
            }
        }
    }
    
    // Parse output layer
    searchPos = 0;
    size_t outputStart = content.find("\"output_layer\": {");
    if (outputStart != string::npos) {
        searchPos = outputStart;
        for (size_t n = 0; n < FOutputLayer.Neurons.size(); n++) {
            size_t wPos = content.find("\"weights\": [", searchPos);
            if (wPos != string::npos) {
                size_t wEnd = content.find("]", wPos);
                string weightsStr = content.substr(wPos, wEnd - wPos + 1);
                FOutputLayer.Neurons[n].Weights = parseArray(weightsStr);
                searchPos = wEnd + 1;
            }
            
            size_t bPos = content.find("\"bias\": ", searchPos);
            if (bPos != string::npos) {
                size_t bEnd = content.find_first_of(",}", bPos + 8);
                string biasStr = content.substr(bPos + 8, bEnd - bPos - 8);
                biasStr.erase(0, biasStr.find_first_not_of(" \t\n\r"));
                biasStr.erase(biasStr.find_last_not_of(" \t\n\r") + 1);
                try {
                    FOutputLayer.Neurons[n].Bias = stod(biasStr);
                } catch (...) {}
                searchPos = bEnd + 1;
            }
        }
    }
    
    // Suppress output during model loading to avoid polluting command output
    // Model information can be displayed via the info command instead
}

TLayer TMultiLayerPerceptron::GetHiddenLayer(int index) const {
    if (index < 0 || index >= (int)FHiddenLayers.size()) {
        throw out_of_range("Hidden layer index out of range");
    }
    return FHiddenLayers[index];
}

void TMultiLayerPerceptron::InitializeBatchNormParams(TLayer& layer, int size) {
    layer.Gamma.resize(size, 1.0);
    layer.Beta.resize(size, 0.0);
    layer.RunningMean.resize(size, 0.0);
    layer.RunningVar.resize(size, 1.0);
}

void TMultiLayerPerceptron::BatchNormForwardCPU(TLayer& layer, bool training) {
    int size = layer.Neurons.size();
    if (layer.Gamma.empty()) return;
    
    for (int i = 0; i < size; i++) {
        double x_norm = (layer.Neurons[i].Output - layer.RunningMean[i]) / sqrt(layer.RunningVar[i] + BNEpsilon);
        layer.Neurons[i].Output = layer.Gamma[i] * x_norm + layer.Beta[i];
    }
}

void TMultiLayerPerceptron::BatchNormBackwardCPU(TLayer& layer, Darray& dGamma, Darray& dBeta, Darray& dInput) {
    (void)layer;
    (void)dGamma;
    (void)dBeta;
    (void)dInput;
}

void TMultiLayerPerceptron::BatchNormForwardGPU(TLayer& layer) {
    if (!FOpenCL || !FOpenCL->context || layer.Gamma.empty()) {
        BatchNormForwardCPU(layer, false);
        return;
    }
    BatchNormForwardCPU(layer, false);
}

void TMultiLayerPerceptron::ExportToONNX(const string& filename) {
    ofstream f(filename, ios::binary);
    if (!f.is_open()) {
        throw runtime_error("Could not open file for writing: " + filename);
    }
    
    f.write("ONNX", 4);
    
    int32_t version = 1;
    f.write(reinterpret_cast<char*>(&version), sizeof(version));
    
    int32_t inputSize = FInputSize;
    int32_t outputSize = FOutputSize;
    int32_t numHiddenLayers = FHiddenLayers.size();
    int32_t hiddenAct = static_cast<int32_t>(HiddenActivation);
    int32_t outputAct = static_cast<int32_t>(OutputActivation);
    int32_t batchNorm = UseBatchNorm ? 1 : 0;
    
    f.write(reinterpret_cast<char*>(&inputSize), sizeof(inputSize));
    f.write(reinterpret_cast<char*>(&outputSize), sizeof(outputSize));
    f.write(reinterpret_cast<char*>(&numHiddenLayers), sizeof(numHiddenLayers));
    f.write(reinterpret_cast<char*>(&hiddenAct), sizeof(hiddenAct));
    f.write(reinterpret_cast<char*>(&outputAct), sizeof(outputAct));
    f.write(reinterpret_cast<char*>(&batchNorm), sizeof(batchNorm));
    
    for (size_t h = 0; h < FHiddenLayers.size(); h++) {
        int32_t layerSize = FHiddenLayers[h].Neurons.size();
        f.write(reinterpret_cast<char*>(&layerSize), sizeof(layerSize));
    }
    
    for (size_t h = 0; h < FHiddenLayers.size(); h++) {
        for (size_t n = 0; n < FHiddenLayers[h].Neurons.size(); n++) {
            for (size_t w = 0; w < FHiddenLayers[h].Neurons[n].Weights.size(); w++) {
                double weight = FHiddenLayers[h].Neurons[n].Weights[w];
                f.write(reinterpret_cast<char*>(&weight), sizeof(weight));
            }
            double bias = FHiddenLayers[h].Neurons[n].Bias;
            f.write(reinterpret_cast<char*>(&bias), sizeof(bias));
        }
    }
    
    for (size_t n = 0; n < FOutputLayer.Neurons.size(); n++) {
        for (size_t w = 0; w < FOutputLayer.Neurons[n].Weights.size(); w++) {
            double weight = FOutputLayer.Neurons[n].Weights[w];
            f.write(reinterpret_cast<char*>(&weight), sizeof(weight));
        }
        double bias = FOutputLayer.Neurons[n].Bias;
        f.write(reinterpret_cast<char*>(&bias), sizeof(bias));
    }
    
    f.close();
}

void TMultiLayerPerceptron::ImportFromONNX(const string& filename) {
    ifstream f(filename, ios::binary);
    if (!f.is_open()) {
        throw runtime_error("Could not open file: " + filename);
    }
    
    char magic[5] = {0};
    f.read(magic, 4);
    if (string(magic) != "ONNX") {
        throw runtime_error("Invalid ONNX file format");
    }
    
    int32_t version;
    f.read(reinterpret_cast<char*>(&version), sizeof(version));
    if (version != 1) {
        throw runtime_error("Unsupported ONNX version");
    }
    
    int32_t inputSize, outputSize, numHiddenLayers, hiddenAct, outputAct, batchNorm;
    f.read(reinterpret_cast<char*>(&inputSize), sizeof(inputSize));
    f.read(reinterpret_cast<char*>(&outputSize), sizeof(outputSize));
    f.read(reinterpret_cast<char*>(&numHiddenLayers), sizeof(numHiddenLayers));
    f.read(reinterpret_cast<char*>(&hiddenAct), sizeof(hiddenAct));
    f.read(reinterpret_cast<char*>(&outputAct), sizeof(outputAct));
    f.read(reinterpret_cast<char*>(&batchNorm), sizeof(batchNorm));
    
    FInputSize = inputSize;
    FOutputSize = outputSize;
    HiddenActivation = static_cast<TActivationType>(hiddenAct);
    OutputActivation = static_cast<TActivationType>(outputAct);
    UseBatchNorm = (batchNorm == 1);
    
    FHiddenSizes.clear();
    for (int h = 0; h < numHiddenLayers; h++) {
        int32_t layerSize;
        f.read(reinterpret_cast<char*>(&layerSize), sizeof(layerSize));
        FHiddenSizes.push_back(layerSize);
    }
    
    FInputLayer.Neurons.clear();
    FInputLayer.Neurons.resize(FInputSize);
    for (int i = 0; i < FInputSize; i++) {
        FInputLayer.Neurons[i].Output = 0.0;
        FInputLayer.Neurons[i].Bias = 0.0;
        FInputLayer.Neurons[i].Error = 0.0;
    }
    FInputLayer.ActivationType = atSigmoid;
    FInputLayer.DropoutMask.resize(FInputSize, true);
    
    FHiddenLayers.clear();
    FHiddenLayers.resize(numHiddenLayers);
    for (int h = 0; h < numHiddenLayers; h++) {
        int numInputs = (h == 0) ? FInputSize : FHiddenSizes[h-1];
        FHiddenLayers[h].Neurons.resize(FHiddenSizes[h]);
        for (int n = 0; n < FHiddenSizes[h]; n++) {
            FHiddenLayers[h].Neurons[n].Weights.resize(numInputs);
            for (int w = 0; w < numInputs; w++) {
                double weight;
                f.read(reinterpret_cast<char*>(&weight), sizeof(weight));
                FHiddenLayers[h].Neurons[n].Weights[w] = weight;
            }
            double bias;
            f.read(reinterpret_cast<char*>(&bias), sizeof(bias));
            FHiddenLayers[h].Neurons[n].Bias = bias;
            FHiddenLayers[h].Neurons[n].Output = 0.0;
            FHiddenLayers[h].Neurons[n].Error = 0.0;
            FHiddenLayers[h].Neurons[n].M.resize(numInputs, 0.0);
            FHiddenLayers[h].Neurons[n].V.resize(numInputs, 0.0);
            FHiddenLayers[h].Neurons[n].MBias = 0.0;
            FHiddenLayers[h].Neurons[n].VBias = 0.0;
        }
        FHiddenLayers[h].ActivationType = HiddenActivation;
        FHiddenLayers[h].DropoutMask.resize(FHiddenSizes[h], true);
    }
    
    int outputLayerInput = FHiddenSizes.empty() ? FInputSize : FHiddenSizes.back();
    FOutputLayer.Neurons.clear();
    FOutputLayer.Neurons.resize(FOutputSize);
    for (int n = 0; n < FOutputSize; n++) {
        FOutputLayer.Neurons[n].Weights.resize(outputLayerInput);
        for (int w = 0; w < outputLayerInput; w++) {
            double weight;
            f.read(reinterpret_cast<char*>(&weight), sizeof(weight));
            FOutputLayer.Neurons[n].Weights[w] = weight;
        }
        double bias;
        f.read(reinterpret_cast<char*>(&bias), sizeof(bias));
        FOutputLayer.Neurons[n].Bias = bias;
        FOutputLayer.Neurons[n].Output = 0.0;
        FOutputLayer.Neurons[n].Error = 0.0;
        FOutputLayer.Neurons[n].M.resize(outputLayerInput, 0.0);
        FOutputLayer.Neurons[n].V.resize(outputLayerInput, 0.0);
        FOutputLayer.Neurons[n].MBias = 0.0;
        FOutputLayer.Neurons[n].VBias = 0.0;
    }
    FOutputLayer.ActivationType = OutputActivation;
    FOutputLayer.DropoutMask.resize(FOutputSize, true);
    
    f.close();
}

Darray TMultiLayerPerceptron::CalculateFeatureImportance() {
    if (FHiddenLayers.empty()) {
        throw runtime_error("No hidden layers to calculate feature importance");
    }
    
    Darray importance(FInputSize, 0.0);
    
    const TLayer& firstHidden = FHiddenLayers[0];
    for (size_t j = 0; j < firstHidden.Neurons.size(); j++) {
        for (int i = 0; i < FInputSize; i++) {
            importance[i] += fabs(firstHidden.Neurons[j].Weights[i]);
        }
    }
    
    double total = 0.0;
    for (int i = 0; i < FInputSize; i++) {
        total += importance[i];
    }
    
    if (total > 0.0) {
        for (int i = 0; i < FInputSize; i++) {
            importance[i] /= total;
        }
    }
    
    return importance;
}

void PrintUsage() {
    cout << "MLP - Multi-Layer Perceptron" << endl;
    cout << endl;
    cout << "Usage: mlp <command> [options]" << endl;
    cout << endl;
    cout << "Commands:" << endl;
    cout << "  create   Create a new MLP model" << endl;
    cout << "  train    Train an existing model" << endl;
    cout << "  predict  Make predictions with a model" << endl;
    cout << "  info     Display model information" << endl;
    cout << "  export-onnx      Export model to ONNX format" << endl;
    cout << "  import-onnx      Import model from ONNX format" << endl;
    cout << "  feature-importance  Calculate and display feature importance" << endl;
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
    cout << "  --batch-norm               Enable batch normalization" << endl;
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
    cout << "Export ONNX Options:" << endl;
    cout << "  -m, --model=FILE           Model file (required, .json)" << endl;
    cout << "  --onnx=FILE                Output ONNX file (required)" << endl;
    cout << endl;
    cout << "Import ONNX Options:" << endl;
    cout << "  --onnx=FILE                Input ONNX file (required)" << endl;
    cout << "  -s, --save=FILE            Save model file (required, .json)" << endl;
    cout << endl;
    cout << "Feature Importance Options:" << endl;
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
    else if (cmdStr == "export-onnx") command = cmdExportONNX;
    else if (cmdStr == "import-onnx") command = cmdImportONNX;
    else if (cmdStr == "feature-importance") command = cmdFeatureImportance;
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
    
    // Initialize defaults
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
    string onnxFile = "";
    bool useBatchNorm = false;
    TDoubleArray inputValues;
    
    // Parse arguments
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
        } else if (arg == "--batch-norm") {
            useBatchNorm = true;
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
            
            // Process key-value pairs
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
            else if (key == "--onnx") {
                onnxFile = value;
            }
            else if (key != "") {
                cerr << "Error: Unknown option: " << key << endl;
            }
        }
    }
    
    // Execute command
    try {
        if (command == cmdCreate) {
            if (inputSize <= 0) { cerr << "Error: --input (-i) is required" << endl; return 1; }
            if (hiddenSizes.empty()) { cerr << "Error: --hidden (-H) is required" << endl; return 1; }
            if (outputSize <= 0) { cerr << "Error: --output (-o) is required" << endl; return 1; }
            if (saveFile.empty()) { cerr << "Error: --save (-s) is required" << endl; return 1; }
            
            TMultiLayerPerceptron mlp(inputSize, hiddenSizes, outputSize, hiddenAct, outputAct, true, useBatchNorm);
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
            cout << "  Batch normalization: " << (useBatchNorm ? "enabled" : "disabled") << endl;
            
            mlp.SaveModelToJSON(saveFile);
            cout << "Model saved to JSON: " << saveFile << endl;
            return 0;
        }
        else if (command == cmdTrain) {
            if (modelFile.empty()) { cerr << "Error: --model (-m) is required" << endl; return 1; }
            if (saveFile.empty()) { cerr << "Error: --save (-s) is required" << endl; return 1; }
            cout << "Model loaded from JSON: " << modelFile << endl;
            TMultiLayerPerceptron mlp(1, {1}, 1, atSigmoid, atSigmoid);
            mlp.LoadModelFromJSON(modelFile);
            cout << "Model loaded successfully. Training functionality not yet implemented." << endl;
            return 0;
        }
        else if (command == cmdPredict) {
             if (modelFile.empty()) { cerr << "Error: --model (-m) is required" << endl; return 1; }
             if (inputValues.empty()) { cerr << "Error: --input (-i) is required" << endl; return 1; }
             
             TMultiLayerPerceptron mlp(1, {1}, 1, atSigmoid, atSigmoid);
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
             TMultiLayerPerceptron mlp(1, {1}, 1, atSigmoid, atSigmoid);
             mlp.LoadModelFromJSON(modelFile);
             
             cout << "MLP Model Information" << endl;
             cout << "=====================" << endl;
             cout << "Input size: " << mlp.GetInputSize() << endl;
             cout << "Output size: " << mlp.GetOutputSize() << endl;
             cout << "Hidden layers: " << mlp.GetHiddenLayerCount() << endl;
             cout << "Hidden sizes: " << mlp.GetHiddenLayerCount();
             if (mlp.GetHiddenLayerCount() > 0) {
                 for (int i = 0; i < mlp.GetHiddenLayerCount(); i++) {
                     cout << ", " << mlp.GetHiddenLayer(i).Neurons.size();
                 }
             }
             cout << endl;
             cout << "Layer sizes: " << mlp.GetInputSize();
            for (int i = 0; i < mlp.GetHiddenLayerCount(); i++)
                cout << " -> " << mlp.GetHiddenLayer(i).Neurons.size();
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
            for (int i = 0; i < mlp.GetHiddenLayerCount(); i++)
                cout << "  Layer " << (i + 1) << ": " << mlp.GetHiddenLayer(i).Neurons.size() << " neurons" << endl;
            cout << "  Layer " << (mlp.GetHiddenLayerCount() + 1) << ": " << mlp.GetOutputSize() << " neurons (output)" << endl;
            cout << "  Batch normalization: " << (mlp.UseBatchNorm ? "enabled" : "disabled") << endl;
            
            return 0;
        }
        else if (command == cmdExportONNX) {
            if (modelFile.empty()) { cerr << "Error: --model (-m) is required" << endl; return 1; }
            if (onnxFile.empty()) { cerr << "Error: --onnx is required" << endl; return 1; }
            
            TMultiLayerPerceptron mlp(1, {1}, 1, atSigmoid, atSigmoid);
            mlp.LoadModelFromJSON(modelFile);
            mlp.ExportToONNX(onnxFile);
            cout << "Model exported to ONNX: " << onnxFile << endl;
            return 0;
        }
        else if (command == cmdImportONNX) {
            if (onnxFile.empty()) { cerr << "Error: --onnx is required" << endl; return 1; }
            if (saveFile.empty()) { cerr << "Error: --save (-s) is required" << endl; return 1; }
            
            TMultiLayerPerceptron mlp(1, {1}, 1, atSigmoid, atSigmoid);
            mlp.ImportFromONNX(onnxFile);
            mlp.SaveModelToJSON(saveFile);
            cout << "Model imported from ONNX and saved to: " << saveFile << endl;
            return 0;
        }
        else if (command == cmdFeatureImportance) {
            if (modelFile.empty()) { cerr << "Error: --model (-m) is required" << endl; return 1; }
            
            TMultiLayerPerceptron mlp(1, {1}, 1, atSigmoid, atSigmoid);
            mlp.LoadModelFromJSON(modelFile);
            
            Darray importance = mlp.CalculateFeatureImportance();
            
            vector<pair<int, double>> ranked;
            for (size_t i = 0; i < importance.size(); i++) {
                ranked.push_back({(int)i, importance[i]});
            }
            sort(ranked.begin(), ranked.end(), [](const pair<int, double>& a, const pair<int, double>& b) {
                return a.second > b.second;
            });
            
            cout << "Feature Importance (ranked by importance):" << endl;
            cout << "===========================================" << endl;
            cout << fixed << setprecision(2);
            for (const auto& p : ranked) {
                cout << "  Input " << p.first << ": " << (p.second * 100.0) << "%" << endl;
            }
            return 0;
        }
    } catch (const exception& e) {
        cerr << "Error: " << e.what() << endl;
        return 1;
    }
    
    return 0;
}
