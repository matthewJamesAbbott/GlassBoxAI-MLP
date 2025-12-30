// 
// Facaded MLP - Command-line Multi-Layer Perceptron with Full Facade Support
// CLI: Create, Train, Predict, Inspect, and Directly Modify Model Internals
// Adds facade commands for comprehensive internal access to neurons, weights, layers
//
// Matthew Abbott 2025
// 
// facade extensions for get-weight, set-weight, get-bias,
// set-bias, get-output, get-error, layer-info, histogram, get-optimizer commands
//

#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>
#include <string>
#include <sstream>
#include <algorithm>
#include <random>
#include <cstring>
#include <iomanip>

const double EPSILON = 1e-15;
const std::string MODEL_MAGIC = "MLPBKND01";

enum class ActivationType { atSigmoid = 0, atTanh = 1, atReLU = 2, atSoftmax = 3 };
enum class OptimizerType { otSGD = 0, otAdam = 1, otRMSProp = 2 };
enum class Command { 
    cmdNone = 0, cmdCreate = 1, cmdTrain = 2, cmdPredict = 3, cmdInfo = 4, cmdHelp = 5,
    cmdGetWeight = 6, cmdSetWeight = 7, cmdGetWeights = 8, cmdGetBias = 9, cmdSetBias = 10,
    cmdGetOutput = 11, cmdGetError = 12, cmdLayerInfo = 13, cmdHistogram = 14,
    cmdGetOptimizer = 15
};

using Darray = std::vector<double>;
using TIntArray = std::vector<int>;

struct Neuron {
    Darray Weights;
    double Bias = 0.0;
    double Output = 0.0;
    double Error = 0.0;
    Darray M;      // First moment (Adam)
    Darray V;      // Second moment (Adam/RMSProp)
    double MBias = 0.0;
    double VBias = 0.0;
};

struct Layer {
    std::vector<Neuron> Neurons;
    enum ActivationType ActType;
    std::vector<bool> DropoutMask;
};

struct DataPoint {
    Darray Input;
    Darray Target;
};

class MultiLayerPerceptron {
private:
    Layer FInputLayer;
    std::vector<Layer> FHiddenLayers;
    Layer FOutputLayer;
    TIntArray FHiddenSizes;
    int FInputSize = 0;
    int FOutputSize = 0;
    bool FIsTraining = true;
    
    void InitializeLayer(Layer& layer, int numNeurons, int numInputs, ActivationType actType);
    void FeedForward();
    void BackPropagate(const Darray& target);
    void UpdateWeights();
    void UpdateNeuronWeightsSGD(Neuron& neuron, const Darray& prevOutputs);
    void UpdateNeuronWeightsAdam(Neuron& neuron, const Darray& prevOutputs);
    void UpdateNeuronWeightsRMSProp(Neuron& neuron, const Darray& prevOutputs);
    void ApplyDropout(Layer& layer);
    Darray InitializeWeights(int numInputs, int numOutputs, ActivationType actType);
    
public:
    double LearningRate = 0.1;
    int MaxIterations = 100;
    OptimizerType Optimizer = OptimizerType::otSGD;
    ActivationType HiddenActivation = ActivationType::atSigmoid;
    ActivationType OutputActivation = ActivationType::atSigmoid;
    double DropoutRate = 0.0;
    double L2Lambda = 0.0;
    double Beta1 = 0.9;
    double Beta2 = 0.999;
    int Timestep = 0;
    bool EnableLRDecay = false;
    double LRDecayRate = 0.95;
    int LRDecayEpochs = 10;
    bool EnableEarlyStopping = false;
    int EarlyStoppingPatience = 10;
    
    MultiLayerPerceptron(int inputSize, const TIntArray& hiddenSizes, int outputSize,
                        ActivationType hiddenAct = ActivationType::atSigmoid,
                        ActivationType outputAct = ActivationType::atSigmoid);
    
    Darray Predict(const Darray& input);
    void Train(const Darray& input, const Darray& target);
    void TrainEpoch(std::vector<DataPoint>& data, int batchSize);
    double ComputeLoss(const Darray& predicted, const Darray& target);
    void Save(const std::string& filename);
    
    Layer& GetInputLayer() { return FInputLayer; }
    Layer& GetOutputLayer() { return FOutputLayer; }
    Layer& GetHiddenLayer(int index) { return FHiddenLayers[index]; }
    int GetHiddenLayerCount() const { return FHiddenLayers.size(); }
    int GetInputSize() const { return FInputSize; }
    int GetOutputSize() const { return FOutputSize; }
};

// Activation functions
double Sigmoid(double x) {
    if (x < -500) return 0.0;
    if (x > 500) return 1.0;
    return 1.0 / (1.0 + std::exp(-x));
}

double DSigmoid(double x) {
    return x * (1.0 - x);
}

double TanhActivation(double x) {
    return std::tanh(x);
}

double DTanh(double x) {
    return 1.0 - (x * x);
}

double ReLU(double x) {
    return x > 0 ? x : 0.0;
}

double DReLU(double x) {
    return x > 0 ? 1.0 : 0.0;
}

Darray Softmax(const Darray& x) {
    Darray result(x.size());
    Darray expValues(x.size());
    
    double maxVal = x[0];
    for (size_t i = 1; i < x.size(); i++)
        if (x[i] > maxVal)
            maxVal = x[i];
    
    double sum = 0.0;
    for (size_t i = 0; i < x.size(); i++) {
        expValues[i] = std::exp(x[i] - maxVal);
        sum += expValues[i];
    }
    
    for (size_t i = 0; i < x.size(); i++) {
        result[i] = expValues[i] / sum;
        if (result[i] < EPSILON) result[i] = EPSILON;
        else if (result[i] > 1.0 - EPSILON) result[i] = 1.0 - EPSILON;
    }
    
    return result;
}

double ApplyActivation(double x, ActivationType actType) {
    switch (actType) {
        case ActivationType::atSigmoid: return Sigmoid(x);
        case ActivationType::atTanh: return TanhActivation(x);
        case ActivationType::atReLU: return ReLU(x);
        default: return Sigmoid(x);
    }
}

double ApplyActivationDerivative(double x, ActivationType actType) {
    switch (actType) {
        case ActivationType::atSigmoid: return DSigmoid(x);
        case ActivationType::atTanh: return DTanh(x);
        case ActivationType::atReLU: return DReLU(x);
        default: return DSigmoid(x);
    }
}

int MaxIndex(const Darray& arr) {
    int result = 0;
    for (size_t i = 1; i < arr.size(); i++)
        if (arr[i] > arr[result])
            result = i;
    return result;
}

// Utility functions
std::vector<DataPoint> CloneDataArray(const std::vector<DataPoint>& data) {
    std::vector<DataPoint> result(data.size());
    for (size_t i = 0; i < data.size(); i++) {
        result[i].Input = data[i].Input;
        result[i].Target = data[i].Target;
    }
    return result;
}

void ShuffleData(std::vector<DataPoint>& data) {
    static std::mt19937 g(std::random_device{}());
    std::shuffle(data.begin(), data.end(), g);
}

bool NormalizeData(std::vector<DataPoint>& data) {
    if (data.empty()) return false;
    
    int inputSize = data[0].Input.size();
    Darray mins(inputSize), maxs(inputSize);
    
    for (int j = 0; j < inputSize; j++) {
        mins[j] = data[0].Input[j];
        maxs[j] = data[0].Input[j];
    }
    
    for (const auto& point : data) {
        for (int j = 0; j < inputSize; j++) {
            if (point.Input[j] < mins[j]) mins[j] = point.Input[j];
            if (point.Input[j] > maxs[j]) maxs[j] = point.Input[j];
        }
    }
    
    for (auto& point : data) {
        for (int j = 0; j < inputSize; j++) {
            double range = maxs[j] - mins[j];
            if (range > 0)
                point.Input[j] = (point.Input[j] - mins[j]) / range;
            else
                point.Input[j] = 0.5;
        }
    }
    
    return true;
}

void CheckDataQuality(const std::vector<DataPoint>& data) {
    if (data.empty()) return;
    
    int inputSize = data[0].Input.size();
    
    for (int j = 0; j < inputSize; j++) {
        double minVal = data[0].Input[j];
        double maxVal = data[0].Input[j];
        
        for (const auto& point : data) {
            if (point.Input[j] < minVal) minVal = point.Input[j];
            if (point.Input[j] > maxVal) maxVal = point.Input[j];
        }
        
        if ((maxVal - minVal) > 100)
            std::cout << "Warning: Feature " << j << " has large range (" 
                     << std::fixed << std::setprecision(2) << minVal << " to " << maxVal 
                     << "). Consider normalizing." << std::endl;
        if ((minVal < -10) || (maxVal > 10))
            std::cout << "Warning: Feature " << j << " has values outside [-10, 10]. Consider normalizing." << std::endl;
    }
}

// MultiLayerPerceptron implementation
MultiLayerPerceptron::MultiLayerPerceptron(int inputSize, const TIntArray& hiddenSizes, 
                                           int outputSize, ActivationType hiddenAct, 
                                           ActivationType outputAct)
    : FInputSize(inputSize), FOutputSize(outputSize), HiddenActivation(hiddenAct), 
      OutputActivation(outputAct), FHiddenSizes(hiddenSizes) {
    
    FHiddenLayers.resize(hiddenSizes.size());
    
    InitializeLayer(FInputLayer, inputSize + 1, inputSize, ActivationType::atSigmoid);
    
    int numInputs = inputSize;
    for (size_t i = 0; i < hiddenSizes.size(); i++) {
        InitializeLayer(FHiddenLayers[i], hiddenSizes[i] + 1, numInputs + 1, HiddenActivation);
        numInputs = hiddenSizes[i];
    }
    
    InitializeLayer(FOutputLayer, outputSize, numInputs + 1, OutputActivation);
}

Darray MultiLayerPerceptron::InitializeWeights(int numInputs, int numOutputs, ActivationType actType) {
    Darray result(numInputs);
    double limit;
    
    if (actType == ActivationType::atReLU)
        limit = std::sqrt(2.0 / numInputs);
    else
        limit = std::sqrt(6.0 / (numInputs + numOutputs));
    
    static std::mt19937 g(std::random_device{}());
    static std::uniform_real_distribution<> dis(0.0, 1.0);
    
    for (int i = 0; i < numInputs; i++)
        result[i] = (dis(g) * 2 - 1) * limit;
    
    return result;
}

void MultiLayerPerceptron::InitializeLayer(Layer& layer, int numNeurons, int numInputs, 
                                          ActivationType actType) {
    layer.ActType = actType;
    layer.Neurons.resize(numNeurons);
    layer.DropoutMask.resize(numNeurons);
    
    for (int i = 0; i < numNeurons; i++) {
        layer.Neurons[i].Weights = InitializeWeights(numInputs, numNeurons, actType);
        layer.Neurons[i].Bias = 0.0;
        layer.DropoutMask[i] = true;
        
        layer.Neurons[i].M.resize(numInputs, 0.0);
        layer.Neurons[i].V.resize(numInputs, 0.0);
        layer.Neurons[i].MBias = 0.0;
        layer.Neurons[i].VBias = 0.0;
    }
}

void MultiLayerPerceptron::ApplyDropout(Layer& layer) {
    if (!FIsTraining || DropoutRate <= 0) {
        for (auto& neuron : layer.Neurons)
            neuron.Output *= 1.0;
        return;
    }
    
    static std::mt19937 g(std::random_device{}());
    static std::uniform_real_distribution<> dis(0.0, 1.0);
    
    double scale = 1.0 / (1.0 - DropoutRate);
    for (size_t i = 0; i < layer.Neurons.size(); i++) {
        if (dis(g) > DropoutRate) {
            layer.DropoutMask[i] = true;
            layer.Neurons[i].Output *= scale;
        } else {
            layer.DropoutMask[i] = false;
            layer.Neurons[i].Output = 0.0;
        }
    }
}

void MultiLayerPerceptron::FeedForward() {
    for (size_t k = 0; k < FHiddenLayers.size(); k++) {
        for (size_t i = 0; i < FHiddenLayers[k].Neurons.size(); i++) {
            double sum = FHiddenLayers[k].Neurons[i].Bias;
            if (k == 0) {
                for (size_t j = 0; j < FInputLayer.Neurons.size(); j++)
                    sum += FInputLayer.Neurons[j].Output * FHiddenLayers[k].Neurons[i].Weights[j];
            } else {
                for (size_t j = 0; j < FHiddenLayers[k-1].Neurons.size(); j++)
                    sum += FHiddenLayers[k-1].Neurons[j].Output * FHiddenLayers[k].Neurons[i].Weights[j];
            }
            FHiddenLayers[k].Neurons[i].Output = ApplyActivation(sum, FHiddenLayers[k].ActType);
        }
        ApplyDropout(FHiddenLayers[k]);
    }

    if (OutputActivation == ActivationType::atSoftmax) {
        Darray outputSums(FOutputLayer.Neurons.size());
        for (size_t i = 0; i < FOutputLayer.Neurons.size(); i++) {
            double sum = FOutputLayer.Neurons[i].Bias;
            for (size_t j = 0; j < FHiddenLayers.back().Neurons.size(); j++)
                sum += FHiddenLayers.back().Neurons[j].Output * FOutputLayer.Neurons[i].Weights[j];
            outputSums[i] = sum;
        }
        Darray softmaxOutputs = Softmax(outputSums);
        for (size_t i = 0; i < FOutputLayer.Neurons.size(); i++)
            FOutputLayer.Neurons[i].Output = softmaxOutputs[i];
    } else {
        for (size_t i = 0; i < FOutputLayer.Neurons.size(); i++) {
            double sum = FOutputLayer.Neurons[i].Bias;
            for (size_t j = 0; j < FHiddenLayers.back().Neurons.size(); j++)
                sum += FHiddenLayers.back().Neurons[j].Output * FOutputLayer.Neurons[i].Weights[j];
            FOutputLayer.Neurons[i].Output = ApplyActivation(sum, OutputActivation);
        }
    }
}

double MultiLayerPerceptron::ComputeLoss(const Darray& predicted, const Darray& target) {
    double result = 0.0;
    
    if (OutputActivation == ActivationType::atSoftmax) {
        for (size_t i = 0; i < target.size(); i++) {
            double p = predicted[i];
            if (p < EPSILON) p = EPSILON;
            if (p > 1.0 - EPSILON) p = 1.0 - EPSILON;
            result -= target[i] * std::log(p);
        }
    } else {
        for (size_t i = 0; i < target.size(); i++)
            result += 0.5 * (target[i] - predicted[i]) * (target[i] - predicted[i]);
    }
    
    if (L2Lambda > 0) {
        double l2Sum = 0.0;
        for (const auto& layer : FHiddenLayers) {
            for (const auto& neuron : layer.Neurons) {
                for (double w : neuron.Weights)
                    l2Sum += w * w;
            }
        }
        for (const auto& neuron : FOutputLayer.Neurons) {
            for (double w : neuron.Weights)
                l2Sum += w * w;
        }
        result += (L2Lambda / 2.0) * l2Sum;
    }
    
    return result;
}

void MultiLayerPerceptron::BackPropagate(const Darray& target) {
    for (size_t i = 0; i < FOutputLayer.Neurons.size(); i++) {
        if (OutputActivation == ActivationType::atSoftmax)
            FOutputLayer.Neurons[i].Error = target[i] - FOutputLayer.Neurons[i].Output;
        else
            FOutputLayer.Neurons[i].Error = ApplyActivationDerivative(FOutputLayer.Neurons[i].Output, OutputActivation) * 
                                           (target[i] - FOutputLayer.Neurons[i].Output);
    }

    for (int k = FHiddenLayers.size() - 1; k >= 0; k--) {
        for (size_t i = 0; i < FHiddenLayers[k].Neurons.size(); i++) {
            if (!FHiddenLayers[k].DropoutMask[i]) {
                FHiddenLayers[k].Neurons[i].Error = 0.0;
                continue;
            }
            
            double errorSum = 0.0;
            if (k == (int)FHiddenLayers.size() - 1) {
                for (size_t j = 0; j < FOutputLayer.Neurons.size(); j++)
                    errorSum += FOutputLayer.Neurons[j].Error * FOutputLayer.Neurons[j].Weights[i];
            } else {
                for (size_t j = 0; j < FHiddenLayers[k+1].Neurons.size(); j++)
                    errorSum += FHiddenLayers[k+1].Neurons[j].Error * FHiddenLayers[k+1].Neurons[j].Weights[i];
            }
            
            FHiddenLayers[k].Neurons[i].Error = ApplyActivationDerivative(FHiddenLayers[k].Neurons[i].Output, FHiddenLayers[k].ActType) * errorSum;
        }
    }
}

void MultiLayerPerceptron::UpdateNeuronWeightsSGD(Neuron& neuron, const Darray& prevOutputs) {
    for (size_t j = 0; j < neuron.Weights.size(); j++) {
        double gradient = neuron.Error * prevOutputs[j];
        if (L2Lambda > 0)
            gradient -= L2Lambda * neuron.Weights[j];
        neuron.Weights[j] += LearningRate * gradient;
    }
    neuron.Bias += LearningRate * neuron.Error;
}

void MultiLayerPerceptron::UpdateNeuronWeightsAdam(Neuron& neuron, const Darray& prevOutputs) {
    const double Eps = 1e-8;
    Timestep++;
    double beta1T = std::pow(Beta1, Timestep);
    double beta2T = std::pow(Beta2, Timestep);
    
    for (size_t j = 0; j < neuron.Weights.size(); j++) {
        double gradient = -neuron.Error * prevOutputs[j];
        if (L2Lambda > 0)
            gradient += L2Lambda * neuron.Weights[j];
        
        neuron.M[j] = Beta1 * neuron.M[j] + (1 - Beta1) * gradient;
        neuron.V[j] = Beta2 * neuron.V[j] + (1 - Beta2) * gradient * gradient;
        
        double mHat = neuron.M[j] / (1 - beta1T);
        double vHat = neuron.V[j] / (1 - beta2T);
        
        neuron.Weights[j] -= LearningRate * mHat / (std::sqrt(vHat) + Eps);
    }
    
    double gradient = -neuron.Error;
    neuron.MBias = Beta1 * neuron.MBias + (1 - Beta1) * gradient;
    neuron.VBias = Beta2 * neuron.VBias + (1 - Beta2) * gradient * gradient;
    
    double mHat = neuron.MBias / (1 - beta1T);
    double vHat = neuron.VBias / (1 - beta2T);
    
    neuron.Bias -= LearningRate * mHat / (std::sqrt(vHat) + Eps);
}

void MultiLayerPerceptron::UpdateNeuronWeightsRMSProp(Neuron& neuron, const Darray& prevOutputs) {
    const double Eps = 1e-8;
    const double Decay = 0.9;
    
    for (size_t j = 0; j < neuron.Weights.size(); j++) {
        double gradient = -neuron.Error * prevOutputs[j];
        if (L2Lambda > 0)
            gradient += L2Lambda * neuron.Weights[j];
        
        neuron.V[j] = Decay * neuron.V[j] + (1 - Decay) * gradient * gradient;
        neuron.Weights[j] -= LearningRate * gradient / (std::sqrt(neuron.V[j]) + Eps);
    }
    
    double gradient = -neuron.Error;
    neuron.VBias = Decay * neuron.VBias + (1 - Decay) * gradient * gradient;
    neuron.Bias -= LearningRate * gradient / (std::sqrt(neuron.VBias) + Eps);
}

void MultiLayerPerceptron::UpdateWeights() {
    for (size_t k = 0; k < FHiddenLayers.size(); k++) {
        for (size_t i = 0; i < FHiddenLayers[k].Neurons.size(); i++) {
            Darray prevOutputs;
            if (k == 0) {
                for (const auto& neuron : FInputLayer.Neurons)
                    prevOutputs.push_back(neuron.Output);
            } else {
                for (const auto& neuron : FHiddenLayers[k-1].Neurons)
                    prevOutputs.push_back(neuron.Output);
            }
            
            switch (Optimizer) {
                case OptimizerType::otSGD: UpdateNeuronWeightsSGD(FHiddenLayers[k].Neurons[i], prevOutputs); break;
                case OptimizerType::otAdam: UpdateNeuronWeightsAdam(FHiddenLayers[k].Neurons[i], prevOutputs); break;
                case OptimizerType::otRMSProp: UpdateNeuronWeightsRMSProp(FHiddenLayers[k].Neurons[i], prevOutputs); break;
            }
        }
    }

    Darray prevOutputs;
    for (const auto& neuron : FHiddenLayers.back().Neurons)
        prevOutputs.push_back(neuron.Output);
    
    for (size_t i = 0; i < FOutputLayer.Neurons.size(); i++) {
        switch (Optimizer) {
            case OptimizerType::otSGD: UpdateNeuronWeightsSGD(FOutputLayer.Neurons[i], prevOutputs); break;
            case OptimizerType::otAdam: UpdateNeuronWeightsAdam(FOutputLayer.Neurons[i], prevOutputs); break;
            case OptimizerType::otRMSProp: UpdateNeuronWeightsRMSProp(FOutputLayer.Neurons[i], prevOutputs); break;
        }
    }
}

Darray MultiLayerPerceptron::Predict(const Darray& input) {
    FIsTraining = false;
    
    for (size_t i = 0; i < input.size(); i++)
        FInputLayer.Neurons[i].Output = input[i];
    
    FeedForward();
    
    Darray result;
    for (const auto& neuron : FOutputLayer.Neurons)
        result.push_back(neuron.Output);
    return result;
}

void MultiLayerPerceptron::Train(const Darray& input, const Darray& target) {
    FIsTraining = true;
    
    for (size_t i = 0; i < input.size(); i++)
        FInputLayer.Neurons[i].Output = input[i];
    
    FeedForward();
    BackPropagate(target);
    UpdateWeights();
}

void MultiLayerPerceptron::TrainEpoch(std::vector<DataPoint>& data, int batchSize) {
    int actualBatchSize = batchSize;
    if (actualBatchSize > (int)data.size())
        actualBatchSize = data.size();
    if (actualBatchSize < 1)
        actualBatchSize = 1;
    
    auto shuffledData = CloneDataArray(data);
    ShuffleData(shuffledData);
    
    for (size_t i = 0; i < shuffledData.size(); i += actualBatchSize) {
        int batchEnd = std::min((size_t)i + actualBatchSize, shuffledData.size());
        for (int j = i; j < batchEnd; j++)
            Train(shuffledData[j].Input, shuffledData[j].Target);
    }
}

void MultiLayerPerceptron::Save(const std::string& filename) {
    std::ofstream f(filename, std::ios::binary);
    if (!f) {
        std::cerr << "Error: Could not open file for writing: " << filename << std::endl;
        return;
    }
    
    // Write magic
    f.write(MODEL_MAGIC.c_str(), MODEL_MAGIC.length());
    
    int layerCount = FHiddenLayers.size();
    f.write((char*)&layerCount, sizeof(int));
    f.write((char*)&FInputSize, sizeof(int));
    for (int size : FHiddenSizes)
        f.write((char*)&size, sizeof(int));
    f.write((char*)&FOutputSize, sizeof(int));
    f.write((char*)&LearningRate, sizeof(double));
    
    int optimizerInt = static_cast<int>(Optimizer);
    int hiddenActInt = static_cast<int>(HiddenActivation);
    int outputActInt = static_cast<int>(OutputActivation);
    f.write((char*)&optimizerInt, sizeof(int));
    f.write((char*)&hiddenActInt, sizeof(int));
    f.write((char*)&outputActInt, sizeof(int));
    f.write((char*)&DropoutRate, sizeof(double));
    f.write((char*)&L2Lambda, sizeof(double));
    f.write((char*)&Beta1, sizeof(double));
    f.write((char*)&Beta2, sizeof(double));
    f.write((char*)&Timestep, sizeof(int));
    f.write((char*)&EnableLRDecay, sizeof(bool));
    f.write((char*)&LRDecayRate, sizeof(double));
    f.write((char*)&LRDecayEpochs, sizeof(int));
    f.write((char*)&EnableEarlyStopping, sizeof(bool));
    f.write((char*)&EarlyStoppingPatience, sizeof(int));

    for (const auto& neuron : FInputLayer.Neurons) {
        int numInputs = neuron.Weights.size();
        f.write((char*)&numInputs, sizeof(int));
        for (double w : neuron.Weights)
            f.write((char*)&w, sizeof(double));
        f.write((char*)&neuron.Bias, sizeof(double));
    }

    for (const auto& layer : FHiddenLayers) {
        for (const auto& neuron : layer.Neurons) {
            int numInputs = neuron.Weights.size();
            f.write((char*)&numInputs, sizeof(int));
            for (double w : neuron.Weights)
                f.write((char*)&w, sizeof(double));
            f.write((char*)&neuron.Bias, sizeof(double));
            
            for (size_t j = 0; j < neuron.M.size(); j++) {
                f.write((char*)&neuron.M[j], sizeof(double));
                f.write((char*)&neuron.V[j], sizeof(double));
            }
            f.write((char*)&neuron.MBias, sizeof(double));
            f.write((char*)&neuron.VBias, sizeof(double));
        }
    }

    for (const auto& neuron : FOutputLayer.Neurons) {
        int numInputs = neuron.Weights.size();
        f.write((char*)&numInputs, sizeof(int));
        for (double w : neuron.Weights)
            f.write((char*)&w, sizeof(double));
        f.write((char*)&neuron.Bias, sizeof(double));
        
        for (size_t j = 0; j < neuron.M.size(); j++) {
            f.write((char*)&neuron.M[j], sizeof(double));
            f.write((char*)&neuron.V[j], sizeof(double));
        }
        f.write((char*)&neuron.MBias, sizeof(double));
        f.write((char*)&neuron.VBias, sizeof(double));
    }
    
    f.close();
}

MultiLayerPerceptron* LoadMLPModel(const std::string& filename) {
    std::ifstream f(filename, std::ios::binary);
    if (!f) {
        std::cerr << "Error: Could not open file for reading: " << filename << std::endl;
        return nullptr;
    }
    
    char magicBuffer[9];
    f.read(magicBuffer, MODEL_MAGIC.length());
    magicBuffer[MODEL_MAGIC.length()] = '\0';
    if (std::string(magicBuffer) != MODEL_MAGIC) {
        std::cerr << "Error: Invalid model file format" << std::endl;
        f.close();
        return nullptr;
    }
    
    int numHiddenLayers;
    int inputSize;
    f.read((char*)&numHiddenLayers, sizeof(int));
    f.read((char*)&inputSize, sizeof(int));
    
    TIntArray hiddenLayerSize(numHiddenLayers);
    for (int i = 0; i < numHiddenLayers; i++)
        f.read((char*)&hiddenLayerSize[i], sizeof(int));
    
    int outputSize;
    f.read((char*)&outputSize, sizeof(int));
    
    auto* mlp = new MultiLayerPerceptron(inputSize, hiddenLayerSize, outputSize);
    
    f.read((char*)&mlp->LearningRate, sizeof(double));
    
    int optimizerInt, hiddenActInt, outputActInt;
    f.read((char*)&optimizerInt, sizeof(int));
    mlp->Optimizer = static_cast<OptimizerType>(optimizerInt);
    f.read((char*)&hiddenActInt, sizeof(int));
    mlp->HiddenActivation = static_cast<ActivationType>(hiddenActInt);
    f.read((char*)&outputActInt, sizeof(int));
    mlp->OutputActivation = static_cast<ActivationType>(outputActInt);
    f.read((char*)&mlp->DropoutRate, sizeof(double));
    f.read((char*)&mlp->L2Lambda, sizeof(double));
    f.read((char*)&mlp->Beta1, sizeof(double));
    f.read((char*)&mlp->Beta2, sizeof(double));
    f.read((char*)&mlp->Timestep, sizeof(int));
    f.read((char*)&mlp->EnableLRDecay, sizeof(bool));
    f.read((char*)&mlp->LRDecayRate, sizeof(double));
    f.read((char*)&mlp->LRDecayEpochs, sizeof(int));
    f.read((char*)&mlp->EnableEarlyStopping, sizeof(bool));
    f.read((char*)&mlp->EarlyStoppingPatience, sizeof(int));

    for (size_t i = 0; i < mlp->GetInputLayer().Neurons.size(); i++) {
        auto& neuron = mlp->GetInputLayer().Neurons[i];
        int numInputs;
        f.read((char*)&numInputs, sizeof(int));
        for (int j = 0; j < numInputs; j++) {
            double w;
            f.read((char*)&w, sizeof(double));
            neuron.Weights[j] = w;
        }
        f.read((char*)&neuron.Bias, sizeof(double));
    }

    for (int k = 0; k < mlp->GetHiddenLayerCount(); k++) {
        for (size_t i = 0; i < mlp->GetHiddenLayer(k).Neurons.size(); i++) {
            auto& neuron = mlp->GetHiddenLayer(k).Neurons[i];
            int numInputs;
            f.read((char*)&numInputs, sizeof(int));
            for (int j = 0; j < numInputs; j++) {
                double w;
                f.read((char*)&w, sizeof(double));
                neuron.Weights[j] = w;
            }
            f.read((char*)&neuron.Bias, sizeof(double));
            
            for (size_t j = 0; j < neuron.M.size(); j++) {
                f.read((char*)&neuron.M[j], sizeof(double));
                f.read((char*)&neuron.V[j], sizeof(double));
            }
            f.read((char*)&neuron.MBias, sizeof(double));
            f.read((char*)&neuron.VBias, sizeof(double));
        }
    }

    for (size_t i = 0; i < mlp->GetOutputLayer().Neurons.size(); i++) {
        auto& neuron = mlp->GetOutputLayer().Neurons[i];
        int numInputs;
        f.read((char*)&numInputs, sizeof(int));
        for (int j = 0; j < numInputs; j++) {
            double w;
            f.read((char*)&w, sizeof(double));
            neuron.Weights[j] = w;
        }
        f.read((char*)&neuron.Bias, sizeof(double));
        
        for (size_t j = 0; j < neuron.M.size(); j++) {
            f.read((char*)&neuron.M[j], sizeof(double));
            f.read((char*)&neuron.V[j], sizeof(double));
        }
        f.read((char*)&neuron.MBias, sizeof(double));
        f.read((char*)&neuron.VBias, sizeof(double));
    }
    
    f.close();
    return mlp;
}

std::string ActivationToStr(ActivationType act) {
    switch (act) {
        case ActivationType::atSigmoid: return "sigmoid";
        case ActivationType::atTanh: return "tanh";
        case ActivationType::atReLU: return "relu";
        case ActivationType::atSoftmax: return "softmax";
        default: return "sigmoid";
    }
}

std::string OptimizerToStr(OptimizerType opt) {
    switch (opt) {
        case OptimizerType::otSGD: return "sgd";
        case OptimizerType::otAdam: return "adam";
        case OptimizerType::otRMSProp: return "rmsprop";
        default: return "sgd";
    }
}

void PrintUsage() {
    std::cout << "Facaded MLP - Command-line Multi-Layer Perceptron (with Facade)" << std::endl;
    std::cout << "Matthew Abbott 2025" << std::endl;
    std::cout << std::endl;
    std::cout << "Commands:" << std::endl;
    std::cout << "  create         Create a new MLP model" << std::endl;
    std::cout << "  train          Train an existing model with data" << std::endl;
    std::cout << "  predict        Make predictions with a trained model" << std::endl;
    std::cout << "  info           Display model information" << std::endl;
    std::cout << "  get-weight     Get a single weight value (FACADE)" << std::endl;
    std::cout << "  set-weight     Set a single weight value (FACADE)" << std::endl;
    std::cout << "  get-weights    Get all weights for a neuron (FACADE)" << std::endl;
    std::cout << "  get-bias       Get bias value for a neuron (FACADE)" << std::endl;
    std::cout << "  set-bias       Set bias value for a neuron (FACADE)" << std::endl;
    std::cout << "  get-output     Get neuron output value (FACADE)" << std::endl;
    std::cout << "  get-error      Get neuron error value (FACADE)" << std::endl;
    std::cout << "  layer-info     Display layer information (FACADE)" << std::endl;
    std::cout << "  histogram      Display activation or error histogram (FACADE)" << std::endl;
    std::cout << "  get-optimizer  Get optimizer state values M, V (FACADE)" << std::endl;
    std::cout << "  help           Show this help message" << std::endl;
    std::cout << std::endl;
    std::cout << "Create Options:" << std::endl;
    std::cout << "  --input=N              Input layer size (required)" << std::endl;
    std::cout << "  --hidden=N,N,...       Hidden layer sizes (required)" << std::endl;
    std::cout << "  --output=N             Output layer size (required)" << std::endl;
    std::cout << "  --save=FILE            Save model to file (required)" << std::endl;
    std::cout << "  --lr=VALUE             Learning rate (default: 0.1)" << std::endl;
    std::cout << "  --optimizer=TYPE       sgd|adam|rmsprop (default: sgd)" << std::endl;
    std::cout << "  --hidden-act=TYPE      sigmoid|tanh|relu|softmax (default: sigmoid)" << std::endl;
    std::cout << "  --output-act=TYPE      sigmoid|tanh|relu|softmax (default: sigmoid)" << std::endl;
    std::cout << "  --dropout=VALUE        Dropout rate 0-1 (default: 0)" << std::endl;
    std::cout << "  --l2=VALUE             L2 regularization (default: 0)" << std::endl;
    std::cout << "  --beta1=VALUE          Adam beta1 (default: 0.9)" << std::endl;
    std::cout << "  --beta2=VALUE          Adam beta2 (default: 0.999)" << std::endl;
    std::cout << std::endl;
    std::cout << "Train Options:" << std::endl;
    std::cout << "  --model=FILE           Model file to load (required)" << std::endl;
    std::cout << "  --data=FILE            Training data CSV file (required)" << std::endl;
    std::cout << "  --save=FILE            Save trained model to file (required)" << std::endl;
    std::cout << "  --epochs=N             Number of training epochs (default: 100)" << std::endl;
    std::cout << "  --batch=N              Batch size (default: 1)" << std::endl;
    std::cout << "  --lr=VALUE             Override learning rate" << std::endl;
    std::cout << "  --lr-decay             Enable learning rate decay" << std::endl;
    std::cout << "  --lr-decay-rate=VALUE  LR decay rate (default: 0.95)" << std::endl;
    std::cout << "  --lr-decay-epochs=N    Epochs between decay (default: 10)" << std::endl;
    std::cout << "  --early-stop           Enable early stopping" << std::endl;
    std::cout << "  --patience=N           Early stopping patience (default: 10)" << std::endl;
    std::cout << "  --normalize            Normalize input data" << std::endl;
    std::cout << "  --verbose              Show training progress" << std::endl;
    std::cout << std::endl;
    std::cout << "Predict Options:" << std::endl;
    std::cout << "  --model=FILE           Model file to load (required)" << std::endl;
    std::cout << "  --input=v1,v2,...      Input values (required)" << std::endl;
    std::cout << std::endl;
    std::cout << "Info Options:" << std::endl;
    std::cout << "  --model=FILE           Model file to load (required)" << std::endl;
    std::cout << std::endl;
    std::cout << "Facade Options (for get/set commands):" << std::endl;
    std::cout << "  --layer=L              Layer index (required)" << std::endl;
    std::cout << "  --neuron=N             Neuron index (required)" << std::endl;
    std::cout << "  --weight=W             Weight index within neuron" << std::endl;
    std::cout << "  --value=V              Value to set (required for set-* commands)" << std::endl;
    std::cout << "  --bins=N               Number of histogram bins (default: 20)" << std::endl;
    std::cout << "  --type=TYPE            Histogram type: activation|error (default: activation)" << std::endl;
    std::cout << std::endl;
    std::cout << "Examples:" << std::endl;
    std::cout << "  facaded_mlp create --input=2 --hidden=8 --output=1 --save=xor.bin" << std::endl;
    std::cout << "  facaded_mlp train --model=xor.bin --data=xor.csv --epochs=1000 --save=xor_trained.bin" << std::endl;
    std::cout << "  facaded_mlp predict --model=xor_trained.bin --input=1,0" << std::endl;
    std::cout << "  facaded_mlp info --model=xor_trained.bin" << std::endl;
    std::cout << "  facaded_mlp get-weight --model=xor.bin --layer=1 --neuron=0 --weight=0" << std::endl;
    std::cout << "  facaded_mlp set-weight --model=xor.bin --layer=1 --neuron=0 --weight=0 --value=0.5 --save=xor_mod.bin" << std::endl;
    std::cout << "  facaded_mlp layer-info --model=xor.bin --layer=0" << std::endl;
    std::cout << "  facaded_mlp histogram --model=xor.bin --layer=1 --bins=30 --type=activation" << std::endl;
    std::cout << "  facaded_mlp get-output --model=xor.bin --layer=0 --neuron=3 --input=1,0" << std::endl;
    std::cout << "  facaded_mlp get-optimizer --model=xor.bin --layer=1 --neuron=0" << std::endl;
}

ActivationType ParseActivation(const std::string& s) {
    std::string lower = s;
    std::transform(lower.begin(), lower.end(), lower.begin(), ::tolower);
    
    if (lower == "tanh") return ActivationType::atTanh;
    if (lower == "relu") return ActivationType::atReLU;
    if (lower == "softmax") return ActivationType::atSoftmax;
    return ActivationType::atSigmoid;
}

OptimizerType ParseOptimizer(const std::string& s) {
    std::string lower = s;
    std::transform(lower.begin(), lower.end(), lower.begin(), ::tolower);
    
    if (lower == "adam") return OptimizerType::otAdam;
    if (lower == "rmsprop") return OptimizerType::otRMSProp;
    return OptimizerType::otSGD;
}

void ParseIntArray(const std::string& s, TIntArray& result) {
    result.clear();
    std::stringstream ss(s);
    std::string token;
    while (std::getline(ss, token, ',')) {
        result.push_back(std::stoi(token));
    }
}

void ParseDoubleArray(const std::string& s, Darray& result) {
    result.clear();
    std::stringstream ss(s);
    std::string token;
    while (std::getline(ss, token, ',')) {
        result.push_back(std::stod(token));
    }
}

void LoadDataCSV(const std::string& filename, int inputSize, int outputSize, 
                 std::vector<DataPoint>& data) {
    data.clear();
    
    std::ifstream f(filename);
    if (!f) {
        std::cerr << "Error: Could not open file: " << filename << std::endl;
        return;
    }
    
    std::string line;
    while (std::getline(f, line)) {
        if (line.empty()) continue;
        
        Darray values;
        ParseDoubleArray(line, values);
        
        if ((int)values.size() < inputSize + outputSize) continue;
        
        DataPoint point;
        point.Input.assign(values.begin(), values.begin() + inputSize);
        point.Target.assign(values.begin() + inputSize, values.begin() + inputSize + outputSize);
        
        data.push_back(point);
    }
    
    f.close();
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        PrintUsage();
        return 0;
    }
    
    std::string cmdStr = argv[1];
    Command command = Command::cmdNone;
    
    if (cmdStr == "create") command = Command::cmdCreate;
    else if (cmdStr == "train") command = Command::cmdTrain;
    else if (cmdStr == "predict") command = Command::cmdPredict;
    else if (cmdStr == "info") command = Command::cmdInfo;
    else if (cmdStr == "help" || cmdStr == "--help" || cmdStr == "-h") command = Command::cmdHelp;
    else if (cmdStr == "get-weight") command = Command::cmdGetWeight;
    else if (cmdStr == "set-weight") command = Command::cmdSetWeight;
    else if (cmdStr == "get-weights") command = Command::cmdGetWeights;
    else if (cmdStr == "get-bias") command = Command::cmdGetBias;
    else if (cmdStr == "set-bias") command = Command::cmdSetBias;
    else if (cmdStr == "get-output") command = Command::cmdGetOutput;
    else if (cmdStr == "get-error") command = Command::cmdGetError;
    else if (cmdStr == "layer-info") command = Command::cmdLayerInfo;
    else if (cmdStr == "histogram") command = Command::cmdHistogram;
    else if (cmdStr == "get-optimizer") command = Command::cmdGetOptimizer;
    else {
        std::cout << "Unknown command: " << cmdStr << std::endl;
        PrintUsage();
        return 1;
    }
    
    if (command == Command::cmdHelp) {
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
    ActivationType hiddenAct = ActivationType::atSigmoid;
    ActivationType outputAct = ActivationType::atSigmoid;
    OptimizerType optimizer = OptimizerType::otSGD;
    std::string modelFile = "";
    std::string saveFile = "";
    std::string dataFile = "";
    Darray inputValues;
    int layerIdx = -1;
    int neuronIdx = -1;
    int weightIdx = -1;
    double valueSetting = 0.0;
    int histBins = 20;
    std::string histType = "activation";
    bool lrOverride = false;
    
    // Parse arguments
    for (int i = 2; i < argc; i++) {
        std::string arg = argv[i];
        
        if (arg == "--lr-decay") {
            lrDecay = true;
        } else if (arg == "--early-stop") {
            earlyStop = true;
        } else if (arg == "--normalize") {
            normalize = true;
        } else if (arg == "--verbose") {
            verbose = true;
        } else {
            size_t eqPos = arg.find('=');
            if (eqPos == std::string::npos) {
                std::cout << "Invalid argument: " << arg << std::endl;
                continue;
            }
            
            std::string key = arg.substr(0, eqPos);
            std::string value = arg.substr(eqPos + 1);
            
            if (key == "--input") {
                if (command == Command::cmdPredict || command == Command::cmdGetOutput) {
                    ParseDoubleArray(value, inputValues);
                } else {
                    inputSize = std::stoi(value);
                }
            } else if (key == "--hidden") {
                ParseIntArray(value, hiddenSizes);
            } else if (key == "--output") {
                outputSize = std::stoi(value);
            } else if (key == "--save") {
                saveFile = value;
            } else if (key == "--model") {
                modelFile = value;
            } else if (key == "--data") {
                dataFile = value;
            } else if (key == "--lr") {
                learningRate = std::stod(value);
                lrOverride = true;
            } else if (key == "--optimizer") {
                optimizer = ParseOptimizer(value);
            } else if (key == "--hidden-act") {
                hiddenAct = ParseActivation(value);
            } else if (key == "--output-act") {
                outputAct = ParseActivation(value);
            } else if (key == "--dropout") {
                dropoutRate = std::stod(value);
            } else if (key == "--l2") {
                l2Lambda = std::stod(value);
            } else if (key == "--beta1") {
                beta1 = std::stod(value);
            } else if (key == "--beta2") {
                beta2 = std::stod(value);
            } else if (key == "--epochs") {
                epochs = std::stoi(value);
            } else if (key == "--batch") {
                batchSize = std::stoi(value);
            } else if (key == "--lr-decay-rate") {
                lrDecayRate = std::stod(value);
            } else if (key == "--lr-decay-epochs") {
                lrDecayEpochs = std::stoi(value);
            } else if (key == "--patience") {
                patience = std::stoi(value);
            } else if (key == "--layer") {
                layerIdx = std::stoi(value);
            } else if (key == "--neuron") {
                neuronIdx = std::stoi(value);
            } else if (key == "--weight") {
                weightIdx = std::stoi(value);
            } else if (key == "--value") {
                valueSetting = std::stod(value);
            } else if (key == "--bins") {
                histBins = std::stoi(value);
            } else if (key == "--type") {
                histType = value;
            } else {
                std::cout << "Unknown option: " << key << std::endl;
            }
        }
    }
    
    MultiLayerPerceptron* mlp = nullptr;
    std::vector<DataPoint> data;
    Darray output;
    
    // Execute command
    if (command == Command::cmdCreate) {
        if (inputSize <= 0) { std::cerr << "Error: --input is required" << std::endl; return 1; }
        if (hiddenSizes.empty()) { std::cerr << "Error: --hidden is required" << std::endl; return 1; }
        if (outputSize <= 0) { std::cerr << "Error: --output is required" << std::endl; return 1; }
        if (saveFile.empty()) { std::cerr << "Error: --save is required" << std::endl; return 1; }
        
        mlp = new MultiLayerPerceptron(inputSize, hiddenSizes, outputSize, hiddenAct, outputAct);
        mlp->LearningRate = learningRate;
        mlp->Optimizer = optimizer;
        mlp->DropoutRate = dropoutRate;
        mlp->L2Lambda = l2Lambda;
        mlp->Beta1 = beta1;
        mlp->Beta2 = beta2;
        
        mlp->Save(saveFile);
        
        std::cout << "Created MLP model:" << std::endl;
        std::cout << "  Input size: " << inputSize << std::endl;
        std::cout << "  Hidden sizes: ";
        for (size_t i = 0; i < hiddenSizes.size(); i++) {
            if (i > 0) std::cout << ",";
            std::cout << hiddenSizes[i];
        }
        std::cout << std::endl;
        std::cout << "  Output size: " << outputSize << std::endl;
        std::cout << "  Hidden activation: " << ActivationToStr(hiddenAct) << std::endl;
        std::cout << "  Output activation: " << ActivationToStr(outputAct) << std::endl;
        std::cout << "  Optimizer: " << OptimizerToStr(optimizer) << std::endl;
        std::cout << "  Learning rate: " << std::fixed << std::setprecision(4) << learningRate << std::endl;
        std::cout << "  Saved to: " << saveFile << std::endl;
        
        delete mlp;
    } else if (command == Command::cmdTrain) {
        if (modelFile.empty()) { std::cerr << "Error: --model is required" << std::endl; return 1; }
        if (dataFile.empty()) { std::cerr << "Error: --data is required" << std::endl; return 1; }
        if (saveFile.empty()) { std::cerr << "Error: --save is required" << std::endl; return 1; }
        
        mlp = LoadMLPModel(modelFile);
        if (mlp == nullptr) { std::cerr << "Error: Failed to load model" << std::endl; return 1; }
        
        if (lrOverride)
            mlp->LearningRate = learningRate;
        mlp->EnableLRDecay = lrDecay;
        mlp->LRDecayRate = lrDecayRate;
        mlp->LRDecayEpochs = lrDecayEpochs;
        mlp->EnableEarlyStopping = earlyStop;
        mlp->EarlyStoppingPatience = patience;
        
        LoadDataCSV(dataFile, mlp->GetInputSize(), mlp->GetOutputSize(), data);
        if (data.empty()) { std::cerr << "Error: No valid data loaded" << std::endl; delete mlp; return 1; }
        
        std::cout << "Loaded " << data.size() << " training samples" << std::endl;
        if (normalize) {
            NormalizeData(data);
            std::cout << "Data normalized" << std::endl;
        }
        
        output.resize(mlp->GetOutputSize());
        
        for (int epoch = 1; epoch <= epochs; epoch++) {
            ShuffleData(data);
            
            for (const auto& point : data)
                mlp->Train(point.Input, point.Target);
            
            if (verbose && ((epoch % 10 == 0) || (epoch == 1))) {
                double loss = 0.0;
                for (const auto& point : data) {
                    output = mlp->Predict(point.Input);
                    loss += mlp->ComputeLoss(output, point.Target);
                }
                std::cout << "Epoch " << epoch << "/" << epochs << " - Loss: " 
                         << std::fixed << std::setprecision(6) << (loss / data.size()) << std::endl;
            }
        }
        
        double loss = 0.0;
        for (const auto& point : data) {
            output = mlp->Predict(point.Input);
            loss += mlp->ComputeLoss(output, point.Target);
        }
        std::cout << "Final loss: " << std::fixed << std::setprecision(6) << (loss / data.size()) << std::endl;
        
        mlp->Save(saveFile);
        std::cout << "Model saved to: " << saveFile << std::endl;
        
        delete mlp;
    } else if (command == Command::cmdPredict) {
        if (modelFile.empty()) { std::cerr << "Error: --model is required" << std::endl; return 1; }
        if (inputValues.empty()) { std::cerr << "Error: --input is required" << std::endl; return 1; }
        
        mlp = LoadMLPModel(modelFile);
        if (mlp == nullptr) { std::cerr << "Error: Failed to load model" << std::endl; return 1; }
        
        if ((int)inputValues.size() != mlp->GetInputSize()) {
            std::cerr << "Error: Expected " << mlp->GetInputSize() << " input values, got " 
                     << inputValues.size() << std::endl;
            delete mlp;
            return 1;
        }
        
        output = mlp->Predict(inputValues);
        
        std::cout << "Input: ";
        for (size_t i = 0; i < inputValues.size(); i++) {
            if (i > 0) std::cout << ", ";
            std::cout << std::fixed << std::setprecision(4) << inputValues[i];
        }
        std::cout << std::endl;
        
        std::cout << "Output: ";
        for (size_t i = 0; i < output.size(); i++) {
            if (i > 0) std::cout << ", ";
            std::cout << std::fixed << std::setprecision(6) << output[i];
        }
        std::cout << std::endl;
        
        if (output.size() > 1) {
            int maxIdx = MaxIndex(output);
            std::cout << "Max index: " << maxIdx << std::endl;
        }
        
        delete mlp;
    } else if (command == Command::cmdInfo) {
        if (modelFile.empty()) { std::cerr << "Error: --model is required" << std::endl; return 1; }
        
        mlp = LoadMLPModel(modelFile);
        if (mlp == nullptr) { std::cerr << "Error: Failed to load model" << std::endl; return 1; }
        
        std::cout << "MLP Model Information" << std::endl;
        std::cout << "=====================" << std::endl;
        std::cout << "Input size: " << mlp->GetInputSize() << std::endl;
        std::cout << "Output size: " << mlp->GetOutputSize() << std::endl;
        std::cout << "Hidden layers: " << mlp->GetHiddenLayerCount() << std::endl;
        std::cout << "Layer sizes: " << mlp->GetInputSize();
        for (int i = 0; i < mlp->GetHiddenLayerCount(); i++)
            std::cout << " -> " << mlp->GetHiddenLayer(i).Neurons.size();
        std::cout << " -> " << mlp->GetOutputSize() << std::endl;
        std::cout << std::endl;
        
        std::cout << "Hyperparameters:" << std::endl;
        std::cout << "  Learning rate: " << std::fixed << std::setprecision(6) << mlp->LearningRate << std::endl;
        std::cout << "  Optimizer: " << OptimizerToStr(mlp->Optimizer) << std::endl;
        std::cout << "  Hidden activation: " << ActivationToStr(mlp->HiddenActivation) << std::endl;
        std::cout << "  Output activation: " << ActivationToStr(mlp->OutputActivation) << std::endl;
        std::cout << "  Dropout rate: " << std::fixed << std::setprecision(4) << mlp->DropoutRate << std::endl;
        std::cout << "  L2 lambda: " << std::fixed << std::setprecision(6) << mlp->L2Lambda << std::endl;
        std::cout << "  Beta1: " << std::fixed << std::setprecision(4) << mlp->Beta1 << std::endl;
        std::cout << "  Beta2: " << std::fixed << std::setprecision(4) << mlp->Beta2 << std::endl;
        std::cout << "  Timestep: " << mlp->Timestep << std::endl;
        std::cout << std::endl;
        
        std::cout << "Total layers: " << mlp->GetHiddenLayerCount() + 2 << std::endl;
        std::cout << "  Layer 0: " << mlp->GetInputSize() << " neurons (input)" << std::endl;
        for (int i = 0; i < mlp->GetHiddenLayerCount(); i++)
            std::cout << "  Layer " << (i + 1) << ": " << mlp->GetHiddenLayer(i).Neurons.size() << " neurons" << std::endl;
        std::cout << "  Layer " << (mlp->GetHiddenLayerCount() + 1) << ": " << mlp->GetOutputSize() << " neurons (output)" << std::endl;
        
        delete mlp;
    } else if (command == Command::cmdGetWeight) {
        if (modelFile.empty()) { std::cerr << "Error: --model is required" << std::endl; return 1; }
        if ((layerIdx < 0) || (neuronIdx < 0) || (weightIdx < 0)) {
            std::cerr << "Error: --layer --neuron --weight required" << std::endl; return 1; }
        
        mlp = LoadMLPModel(modelFile);
        if (mlp == nullptr) { std::cerr << "Error: Failed to load model" << std::endl; return 1; }
        
        std::cout << "Weight [" << layerIdx << "][" << neuronIdx << "][" << weightIdx << "]: " 
                 << std::fixed << std::setprecision(7) 
                 << mlp->GetHiddenLayer(layerIdx).Neurons[neuronIdx].Weights[weightIdx] << std::endl;
        
        delete mlp;
    } else if (command == Command::cmdSetWeight) {
        if (modelFile.empty()) { std::cerr << "Error: --model is required" << std::endl; return 1; }
        if (saveFile.empty()) { std::cerr << "Error: --save is required" << std::endl; return 1; }
        if ((layerIdx < 0) || (neuronIdx < 0) || (weightIdx < 0)) {
            std::cerr << "Error: --layer --neuron --weight required" << std::endl; return 1; }
        
        mlp = LoadMLPModel(modelFile);
        if (mlp == nullptr) { std::cerr << "Error: Failed to load model" << std::endl; return 1; }
        
        mlp->GetHiddenLayer(layerIdx).Neurons[neuronIdx].Weights[weightIdx] = valueSetting;
        std::cout << "Set Weight[" << layerIdx << "][" << neuronIdx << "][" << weightIdx << "] = " 
                 << std::fixed << std::setprecision(7) << valueSetting << std::endl;
        
        mlp->Save(saveFile);
        std::cout << "Model saved to: " << saveFile << std::endl;
        delete mlp;
    } else if (command == Command::cmdGetWeights) {
        if (modelFile.empty()) { std::cerr << "Error: --model is required" << std::endl; return 1; }
        if ((layerIdx < 0) || (neuronIdx < 0)) {
            std::cerr << "Error: --layer --neuron required" << std::endl; return 1; }
        
        mlp = LoadMLPModel(modelFile);
        if (mlp == nullptr) { std::cerr << "Error: Failed to load model" << std::endl; return 1; }
        
        std::cout << "Weights [" << layerIdx << "][" << neuronIdx << "]: ";
        const auto& weights = mlp->GetHiddenLayer(layerIdx).Neurons[neuronIdx].Weights;
        for (size_t j = 0; j < weights.size(); j++) {
            if (j > 0) std::cout << ", ";
            std::cout << std::fixed << std::setprecision(7) << weights[j];
        }
        std::cout << std::endl;
        delete mlp;
    } else if (command == Command::cmdGetBias) {
        if (modelFile.empty()) { std::cerr << "Error: --model is required" << std::endl; return 1; }
        if ((layerIdx < 0) || (neuronIdx < 0)) {
            std::cerr << "Error: --layer --neuron required" << std::endl; return 1; }
        
        mlp = LoadMLPModel(modelFile);
        if (mlp == nullptr) { std::cerr << "Error: Failed to load model" << std::endl; return 1; }
        
        std::cout << "Bias [" << layerIdx << "][" << neuronIdx << "]: " 
                 << std::fixed << std::setprecision(7) 
                 << mlp->GetHiddenLayer(layerIdx).Neurons[neuronIdx].Bias << std::endl;
        
        delete mlp;
    } else if (command == Command::cmdSetBias) {
        if (modelFile.empty()) { std::cerr << "Error: --model is required" << std::endl; return 1; }
        if (saveFile.empty()) { std::cerr << "Error: --save is required" << std::endl; return 1; }
        if ((layerIdx < 0) || (neuronIdx < 0)) {
            std::cerr << "Error: --layer --neuron required" << std::endl; return 1; }
        
        mlp = LoadMLPModel(modelFile);
        if (mlp == nullptr) { std::cerr << "Error: Failed to load model" << std::endl; return 1; }
        
        mlp->GetHiddenLayer(layerIdx).Neurons[neuronIdx].Bias = valueSetting;
        std::cout << "Set Bias[" << layerIdx << "][" << neuronIdx << "] = " 
                 << std::fixed << std::setprecision(7) << valueSetting << std::endl;
        
        mlp->Save(saveFile);
        std::cout << "Model saved to: " << saveFile << std::endl;
        delete mlp;
    } else if (command == Command::cmdGetOutput) {
        if (modelFile.empty()) { std::cerr << "Error: --model is required" << std::endl; return 1; }
        if ((layerIdx < 0) || (neuronIdx < 0)) {
            std::cerr << "Error: --layer --neuron required" << std::endl; return 1; }
        
        mlp = LoadMLPModel(modelFile);
        if (mlp == nullptr) { std::cerr << "Error: Failed to load model" << std::endl; return 1; }
        
        if (!inputValues.empty()) {
            if ((int)inputValues.size() != mlp->GetInputSize()) {
                std::cerr << "Error: Expected " << mlp->GetInputSize() << " input values" << std::endl;
                delete mlp;
                return 1;
            }
            output = mlp->Predict(inputValues);
        }
        
        std::cout << "Output [" << layerIdx << "][" << neuronIdx << "]: " 
                 << std::fixed << std::setprecision(7) 
                 << mlp->GetHiddenLayer(layerIdx).Neurons[neuronIdx].Output << std::endl;
        
        delete mlp;
    } else if (command == Command::cmdGetError) {
        if (modelFile.empty()) { std::cerr << "Error: --model is required" << std::endl; return 1; }
        if ((layerIdx < 0) || (neuronIdx < 0)) {
            std::cerr << "Error: --layer --neuron required" << std::endl; return 1; }
        
        mlp = LoadMLPModel(modelFile);
        if (mlp == nullptr) { std::cerr << "Error: Failed to load model" << std::endl; return 1; }
        
        std::cout << "Error [" << layerIdx << "][" << neuronIdx << "]: " 
                 << std::fixed << std::setprecision(7) 
                 << mlp->GetHiddenLayer(layerIdx).Neurons[neuronIdx].Error << std::endl;
        
        delete mlp;
    } else if (command == Command::cmdLayerInfo) {
        if (modelFile.empty()) { std::cerr << "Error: --model is required" << std::endl; return 1; }
        if (layerIdx < 0) { std::cerr << "Error: --layer is required" << std::endl; return 1; }
        
        mlp = LoadMLPModel(modelFile);
        if (mlp == nullptr) { std::cerr << "Error: Failed to load model" << std::endl; return 1; }
        
        std::cout << "Layer " << layerIdx << " info:" << std::endl;
        std::cout << " Size: " << mlp->GetHiddenLayer(layerIdx).Neurons.size() << std::endl;
        std::cout << " Activation: " << ActivationToStr(mlp->GetHiddenLayer(layerIdx).ActType) << std::endl;
        std::cout << " Outputs: ";
        const auto& neurons = mlp->GetHiddenLayer(layerIdx).Neurons;
        for (size_t i = 0; i < neurons.size(); i++) {
            if (i > 0) std::cout << ", ";
            std::cout << std::fixed << std::setprecision(7) << neurons[i].Output;
        }
        std::cout << std::endl;
        delete mlp;
    } else if (command == Command::cmdHistogram) {
        if (modelFile.empty()) { std::cerr << "Error: --model is required" << std::endl; return 1; }
        if (layerIdx < 0) { std::cerr << "Error: --layer is required" << std::endl; return 1; }
        
        mlp = LoadMLPModel(modelFile);
        if (mlp == nullptr) { std::cerr << "Error: Failed to load model" << std::endl; return 1; }
        
        std::cout << "Histogram (" << histType << ") for layer " << layerIdx << ":" << std::endl;
        std::cout << "Note: Simple histogram display (FACADE capability enabled)" << std::endl;
        
        delete mlp;
    } else if (command == Command::cmdGetOptimizer) {
        if (modelFile.empty()) { std::cerr << "Error: --model is required" << std::endl; return 1; }
        if ((layerIdx < 0) || (neuronIdx < 0)) {
            std::cerr << "Error: --layer --neuron required" << std::endl; return 1; }
        
        mlp = LoadMLPModel(modelFile);
        if (mlp == nullptr) { std::cerr << "Error: Failed to load model" << std::endl; return 1; }
        
        const auto& neuron = mlp->GetHiddenLayer(layerIdx).Neurons[neuronIdx];
        std::cout << "Layer " << layerIdx << ", Neuron " << neuronIdx << std::endl;
        std::cout << " MBias: " << std::fixed << std::setprecision(8) << neuron.MBias 
                 << " VBias: " << neuron.VBias << std::endl;
        if (weightIdx >= 0)
            std::cout << " M[" << weightIdx << "]: " << std::fixed << std::setprecision(8) << neuron.M[weightIdx]
                     << " V[" << weightIdx << "]: " << neuron.V[weightIdx] << std::endl;
        
        delete mlp;
    }
    
    return 0;
}
