// Matthew Abbott 19/3/2023
// Enhanced with: Softmax, Adam/RMSProp optimizers, Dropout, L2 regularization,
// Xavier/He initialization, LR decay, Early stopping, Data normalization
// CLI Support for Create, Train, Predict, Info commands
// Ported to C++ 2025

#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>
#include <sstream>
#include <string>
#include <algorithm>
#include <random>
#include <iomanip>
#include <cstring>
#include <memory>

const double EPSILON = 1e-15;
const char* MODEL_MAGIC = "MLPBKND01";

enum class ActivationType { Sigmoid, Tanh, ReLU, Softmax };
enum class OptimizerType { SGD, Adam, RMSProp };
enum class CommandType { None, Create, Train, Predict, Info, Help };

using DoubleArray = std::vector<double>;
using IntArray = std::vector<int>;
using BoolArray = std::vector<bool>;

struct DataPoint {
    DoubleArray input;
    DoubleArray target;
};

using DataPointArray = std::vector<DataPoint>;

struct Neuron {
    DoubleArray weights;
    double bias = 0.0;
    double output = 0.0;
    double error = 0.0;
    DoubleArray m;      // First moment (Adam)
    DoubleArray v;      // Second moment (Adam/RMSProp)
    double m_bias = 0.0;
    double v_bias = 0.0;
};

struct Layer {
    std::vector<Neuron> neurons;
    ActivationType activation_type;
    BoolArray dropout_mask;
};

class MultiLayerPerceptron {
private:
    Layer input_layer;
    std::vector<Layer> hidden_layers;
    Layer output_layer;
    IntArray hidden_sizes;
    int input_size;
    int output_size;
    bool is_training = true;

    void InitializeLayer(Layer& layer, int num_neurons, int num_inputs, ActivationType act_type);
    void FeedForward();
    void BackPropagate(const DoubleArray& target);
    void UpdateWeights();
    void UpdateNeuronWeightsSGD(Neuron& neuron, const DoubleArray& prev_outputs);
    void UpdateNeuronWeightsAdam(Neuron& neuron, const DoubleArray& prev_outputs);
    void UpdateNeuronWeightsRMSProp(Neuron& neuron, const DoubleArray& prev_outputs);
    void ApplyDropout(Layer& layer);
    DoubleArray InitializeWeights(int num_inputs, int num_outputs, ActivationType act_type);

public:
    double learning_rate = 0.1;
    int max_iterations = 100;
    OptimizerType optimizer = OptimizerType::SGD;
    ActivationType hidden_activation = ActivationType::Sigmoid;
    ActivationType output_activation = ActivationType::Sigmoid;
    double dropout_rate = 0.0;
    double l2_lambda = 0.0;
    double beta1 = 0.9;
    double beta2 = 0.999;
    int timestep = 0;
    bool enable_lr_decay = false;
    double lr_decay_rate = 0.95;
    int lr_decay_epochs = 10;
    bool enable_early_stopping = false;
    int early_stopping_patience = 10;

    MultiLayerPerceptron(int input_sz, const IntArray& hidden_szs, int output_sz,
                         ActivationType hidden_act = ActivationType::Sigmoid,
                         ActivationType output_act = ActivationType::Sigmoid);

    DoubleArray Predict(const DoubleArray& input);
    void Train(const DoubleArray& input, const DoubleArray& target);
    void TrainEpoch(DataPointArray& data, int batch_size);
    double ComputeLoss(const DoubleArray& predicted, const DoubleArray& target);
    void Save(const std::string& filename);
    void Load(const std::string& filename);

    int GetInputSize() const { return input_size; }
    int GetOutputSize() const { return output_size; }
    int GetHiddenLayerCount() const { return hidden_layers.size(); }
    Layer& GetHiddenLayer(int index) { return hidden_layers[index]; }
    const Layer& GetInputLayer() const { return input_layer; }
    const Layer& GetOutputLayer() const { return output_layer; }
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

DoubleArray Softmax(const DoubleArray& x) {
    int n = x.size();
    DoubleArray result(n);
    DoubleArray exp_values(n);

    double max_val = x[0];
    for (int i = 1; i < n; i++)
        if (x[i] > max_val) max_val = x[i];

    double sum = 0.0;
    for (int i = 0; i < n; i++) {
        exp_values[i] = std::exp(x[i] - max_val);
        sum += exp_values[i];
    }

    for (int i = 0; i < n; i++) {
        result[i] = exp_values[i] / sum;
        if (result[i] < EPSILON) result[i] = EPSILON;
        else if (result[i] > 1.0 - EPSILON) result[i] = 1.0 - EPSILON;
    }
    return result;
}

double ApplyActivation(double x, ActivationType act_type) {
    switch (act_type) {
        case ActivationType::Sigmoid: return Sigmoid(x);
        case ActivationType::Tanh: return TanhActivation(x);
        case ActivationType::ReLU: return ReLU(x);
        default: return Sigmoid(x);
    }
}

double ApplyActivationDerivative(double x, ActivationType act_type) {
    switch (act_type) {
        case ActivationType::Sigmoid: return DSigmoid(x);
        case ActivationType::Tanh: return DTanh(x);
        case ActivationType::ReLU: return DReLU(x);
        default: return DSigmoid(x);
    }
}

int MaxIndex(const DoubleArray& arr) {
    int result = 0;
    for (int i = 1; i < arr.size(); i++)
        if (arr[i] > arr[result]) result = i;
    return result;
}

void ShuffleData(DataPointArray& data) {
    static std::mt19937 gen(std::random_device{}());
    std::shuffle(data.begin(), data.end(), gen);
}

bool NormalizeData(DataPointArray& data) {
    if (data.empty()) return false;

    int input_size = data[0].input.size();
    DoubleArray mins(input_size), maxs(input_size);

    for (int j = 0; j < input_size; j++) {
        mins[j] = data[0].input[j];
        maxs[j] = data[0].input[j];
    }

    for (auto& point : data) {
        for (int j = 0; j < input_size; j++) {
            if (point.input[j] < mins[j]) mins[j] = point.input[j];
            if (point.input[j] > maxs[j]) maxs[j] = point.input[j];
        }
    }

    for (auto& point : data) {
        for (int j = 0; j < input_size; j++) {
            double range = maxs[j] - mins[j];
            if (range > 0)
                point.input[j] = (point.input[j] - mins[j]) / range;
            else
                point.input[j] = 0.5;
        }
    }

    return true;
}

void CheckDataQuality(const DataPointArray& data) {
    if (data.empty()) return;

    int input_size = data[0].input.size();

    for (int j = 0; j < input_size; j++) {
        double min_val = data[0].input[j];
        double max_val = data[0].input[j];

        for (const auto& point : data) {
            if (point.input[j] < min_val) min_val = point.input[j];
            if (point.input[j] > max_val) max_val = point.input[j];
        }

        if ((max_val - min_val) > 100) {
            std::cout << "Warning: Feature " << j << " has large range (" 
                      << std::fixed << std::setprecision(2) << min_val << " to " 
                      << max_val << "). Consider normalizing." << std::endl;
        }
        if (min_val < -10 || max_val > 10) {
            std::cout << "Warning: Feature " << j << " has values outside [-10, 10]. Consider normalizing." << std::endl;
        }
    }
}

MultiLayerPerceptron::MultiLayerPerceptron(int input_sz, const IntArray& hidden_szs, int output_sz,
                                           ActivationType hidden_act, ActivationType output_act)
    : input_size(input_sz), output_size(output_sz), hidden_activation(hidden_act), output_activation(output_act) {
    hidden_sizes = hidden_szs;
    hidden_layers.resize(hidden_szs.size());

    InitializeLayer(input_layer, input_size + 1, input_size, ActivationType::Sigmoid);

    int num_inputs = input_size;
    for (int i = 0; i < hidden_szs.size(); i++) {
        InitializeLayer(hidden_layers[i], hidden_szs[i] + 1, num_inputs + 1, hidden_activation);
        num_inputs = hidden_szs[i];
    }

    InitializeLayer(output_layer, output_size, num_inputs + 1, output_activation);
}

DoubleArray MultiLayerPerceptron::InitializeWeights(int num_inputs, int num_outputs, ActivationType act_type) {
    DoubleArray result(num_inputs);
    double limit;
    if (act_type == ActivationType::ReLU)
        limit = std::sqrt(2.0 / num_inputs);
    else
        limit = std::sqrt(6.0 / (num_inputs + num_outputs));

    static std::mt19937 gen(std::random_device{}());
    std::uniform_real_distribution<> dis(-1.0, 1.0);

    for (int i = 0; i < num_inputs; i++)
        result[i] = dis(gen) * limit;

    return result;
}

void MultiLayerPerceptron::InitializeLayer(Layer& layer, int num_neurons, int num_inputs, ActivationType act_type) {
    layer.activation_type = act_type;
    layer.neurons.resize(num_neurons);
    layer.dropout_mask.resize(num_neurons);

    for (int i = 0; i < num_neurons; i++) {
        layer.neurons[i].weights = InitializeWeights(num_inputs, num_neurons, act_type);
        layer.neurons[i].bias = 0.0;
        layer.dropout_mask[i] = true;

        layer.neurons[i].m.resize(num_inputs, 0.0);
        layer.neurons[i].v.resize(num_inputs, 0.0);
        layer.neurons[i].m_bias = 0.0;
        layer.neurons[i].v_bias = 0.0;
    }
}

void MultiLayerPerceptron::ApplyDropout(Layer& layer) {
    if (!is_training || dropout_rate <= 0.0) {
        for (int i = 0; i < layer.neurons.size(); i++)
            layer.dropout_mask[i] = true;
        return;
    }

    static std::mt19937 gen(std::random_device{}());
    std::uniform_real_distribution<> dis(0.0, 1.0);
    double scale = 1.0 / (1.0 - dropout_rate);

    for (int i = 0; i < layer.neurons.size(); i++) {
        if (dis(gen) > dropout_rate) {
            layer.dropout_mask[i] = true;
            layer.neurons[i].output *= scale;
        } else {
            layer.dropout_mask[i] = false;
            layer.neurons[i].output = 0.0;
        }
    }
}

void MultiLayerPerceptron::FeedForward() {
    // Hidden layers
    for (int k = 0; k < hidden_layers.size(); k++) {
        for (int i = 0; i < hidden_layers[k].neurons.size(); i++) {
            double sum = hidden_layers[k].neurons[i].bias;
            
            if (k == 0) {
                for (int j = 0; j < input_layer.neurons.size(); j++)
                    sum += input_layer.neurons[j].output * hidden_layers[k].neurons[i].weights[j];
            } else {
                for (int j = 0; j < hidden_layers[k-1].neurons.size(); j++)
                    sum += hidden_layers[k-1].neurons[j].output * hidden_layers[k].neurons[i].weights[j];
            }
            
            hidden_layers[k].neurons[i].output = ApplyActivation(sum, hidden_layers[k].activation_type);
        }
        ApplyDropout(hidden_layers[k]);
    }

    // Output layer
    if (output_activation == ActivationType::Softmax) {
        int last_hidden = hidden_layers.size() - 1;
        DoubleArray output_sums(output_layer.neurons.size());
        
        for (int i = 0; i < output_layer.neurons.size(); i++) {
            double sum = output_layer.neurons[i].bias;
            for (int j = 0; j < hidden_layers[last_hidden].neurons.size(); j++)
                sum += hidden_layers[last_hidden].neurons[j].output * output_layer.neurons[i].weights[j];
            output_sums[i] = sum;
        }
        
        DoubleArray softmax_outputs = Softmax(output_sums);
        for (int i = 0; i < output_layer.neurons.size(); i++)
            output_layer.neurons[i].output = softmax_outputs[i];
    } else {
        int last_hidden = hidden_layers.size() - 1;
        for (int i = 0; i < output_layer.neurons.size(); i++) {
            double sum = output_layer.neurons[i].bias;
            for (int j = 0; j < hidden_layers[last_hidden].neurons.size(); j++)
                sum += hidden_layers[last_hidden].neurons[j].output * output_layer.neurons[i].weights[j];
            output_layer.neurons[i].output = ApplyActivation(sum, output_activation);
        }
    }
}

double MultiLayerPerceptron::ComputeLoss(const DoubleArray& predicted, const DoubleArray& target) {
    double result = 0.0;

    if (output_activation == ActivationType::Softmax) {
        for (int i = 0; i < target.size(); i++) {
            double p = predicted[i];
            if (p < EPSILON) p = EPSILON;
            if (p > 1.0 - EPSILON) p = 1.0 - EPSILON;
            result -= target[i] * std::log(p);
        }
    } else {
        for (int i = 0; i < target.size(); i++)
            result += 0.5 * (target[i] - predicted[i]) * (target[i] - predicted[i]);
    }

    // L2 regularization
    if (l2_lambda > 0.0) {
        double l2_sum = 0.0;
        for (const auto& layer : hidden_layers) {
            for (const auto& neuron : layer.neurons) {
                for (double w : neuron.weights)
                    l2_sum += w * w;
            }
        }
        for (const auto& neuron : output_layer.neurons) {
            for (double w : neuron.weights)
                l2_sum += w * w;
        }
        result += (l2_lambda / 2.0) * l2_sum;
    }

    return result;
}

void MultiLayerPerceptron::BackPropagate(const DoubleArray& target) {
    // Output layer
    for (int i = 0; i < output_layer.neurons.size(); i++) {
        if (output_activation == ActivationType::Softmax)
            output_layer.neurons[i].error = target[i] - output_layer.neurons[i].output;
        else
            output_layer.neurons[i].error = ApplyActivationDerivative(output_layer.neurons[i].output, output_activation) *
                                            (target[i] - output_layer.neurons[i].output);
    }

    // Hidden layers
    for (int k = (int)hidden_layers.size() - 1; k >= 0; k--) {
        for (int i = 0; i < hidden_layers[k].neurons.size(); i++) {
            if (!hidden_layers[k].dropout_mask[i]) {
                hidden_layers[k].neurons[i].error = 0.0;
                continue;
            }

            double error_sum = 0.0;
            if (k == (int)hidden_layers.size() - 1) {
                for (int j = 0; j < output_layer.neurons.size(); j++)
                    error_sum += output_layer.neurons[j].error * output_layer.neurons[j].weights[i];
            } else {
                for (int j = 0; j < hidden_layers[k+1].neurons.size(); j++)
                    error_sum += hidden_layers[k+1].neurons[j].error * hidden_layers[k+1].neurons[j].weights[i];
            }

            hidden_layers[k].neurons[i].error = ApplyActivationDerivative(hidden_layers[k].neurons[i].output, hidden_layers[k].activation_type) * error_sum;
        }
    }
}

void MultiLayerPerceptron::UpdateNeuronWeightsSGD(Neuron& neuron, const DoubleArray& prev_outputs) {
    for (int j = 0; j < neuron.weights.size(); j++) {
        double gradient = neuron.error * prev_outputs[j];
        if (l2_lambda > 0.0)
            gradient -= l2_lambda * neuron.weights[j];
        neuron.weights[j] += learning_rate * gradient;
    }
    neuron.bias += learning_rate * neuron.error;
}

void MultiLayerPerceptron::UpdateNeuronWeightsAdam(Neuron& neuron, const DoubleArray& prev_outputs) {
    const double eps = 1e-8;
    ++timestep;
    double beta1_t = std::pow(beta1, timestep);
    double beta2_t = std::pow(beta2, timestep);

    for (int j = 0; j < neuron.weights.size(); j++) {
        double gradient = -neuron.error * prev_outputs[j];
        if (l2_lambda > 0.0)
            gradient += l2_lambda * neuron.weights[j];

        neuron.m[j] = beta1 * neuron.m[j] + (1.0 - beta1) * gradient;
        neuron.v[j] = beta2 * neuron.v[j] + (1.0 - beta2) * gradient * gradient;

        double m_hat = neuron.m[j] / (1.0 - beta1_t);
        double v_hat = neuron.v[j] / (1.0 - beta2_t);

        neuron.weights[j] -= learning_rate * m_hat / (std::sqrt(v_hat) + eps);
    }

    double gradient = -neuron.error;
    neuron.m_bias = beta1 * neuron.m_bias + (1.0 - beta1) * gradient;
    neuron.v_bias = beta2 * neuron.v_bias + (1.0 - beta2) * gradient * gradient;

    double m_hat = neuron.m_bias / (1.0 - std::pow(beta1, timestep));
    double v_hat = neuron.v_bias / (1.0 - std::pow(beta2, timestep));

    neuron.bias -= learning_rate * m_hat / (std::sqrt(v_hat) + eps);
}

void MultiLayerPerceptron::UpdateNeuronWeightsRMSProp(Neuron& neuron, const DoubleArray& prev_outputs) {
    const double eps = 1e-8;
    const double decay = 0.9;

    for (int j = 0; j < neuron.weights.size(); j++) {
        double gradient = -neuron.error * prev_outputs[j];
        if (l2_lambda > 0.0)
            gradient += l2_lambda * neuron.weights[j];

        neuron.v[j] = decay * neuron.v[j] + (1.0 - decay) * gradient * gradient;
        neuron.weights[j] -= learning_rate * gradient / (std::sqrt(neuron.v[j]) + eps);
    }

    double gradient = -neuron.error;
    neuron.v_bias = decay * neuron.v_bias + (1.0 - decay) * gradient * gradient;
    neuron.bias -= learning_rate * gradient / (std::sqrt(neuron.v_bias) + eps);
}

void MultiLayerPerceptron::UpdateWeights() {
    for (int k = 0; k < hidden_layers.size(); k++) {
        for (int i = 0; i < hidden_layers[k].neurons.size(); i++) {
            DoubleArray prev_outputs;
            
            if (k == 0) {
                prev_outputs.resize(input_layer.neurons.size());
                for (int j = 0; j < input_layer.neurons.size(); j++)
                    prev_outputs[j] = input_layer.neurons[j].output;
            } else {
                prev_outputs.resize(hidden_layers[k-1].neurons.size());
                for (int j = 0; j < hidden_layers[k-1].neurons.size(); j++)
                    prev_outputs[j] = hidden_layers[k-1].neurons[j].output;
            }

            switch (optimizer) {
                case OptimizerType::SGD:
                    UpdateNeuronWeightsSGD(hidden_layers[k].neurons[i], prev_outputs);
                    break;
                case OptimizerType::Adam:
                    UpdateNeuronWeightsAdam(hidden_layers[k].neurons[i], prev_outputs);
                    break;
                case OptimizerType::RMSProp:
                    UpdateNeuronWeightsRMSProp(hidden_layers[k].neurons[i], prev_outputs);
                    break;
            }
        }
    }

    DoubleArray prev_outputs(hidden_layers.back().neurons.size());
    for (int j = 0; j < hidden_layers.back().neurons.size(); j++)
        prev_outputs[j] = hidden_layers.back().neurons[j].output;

    for (int i = 0; i < output_layer.neurons.size(); i++) {
        switch (optimizer) {
            case OptimizerType::SGD:
                UpdateNeuronWeightsSGD(output_layer.neurons[i], prev_outputs);
                break;
            case OptimizerType::Adam:
                UpdateNeuronWeightsAdam(output_layer.neurons[i], prev_outputs);
                break;
            case OptimizerType::RMSProp:
                UpdateNeuronWeightsRMSProp(output_layer.neurons[i], prev_outputs);
                break;
        }
    }
}

DoubleArray MultiLayerPerceptron::Predict(const DoubleArray& input) {
    is_training = false;

    for (int i = 0; i < input.size(); i++)
        input_layer.neurons[i].output = input[i];

    FeedForward();

    DoubleArray result(output_layer.neurons.size());
    for (int i = 0; i < output_layer.neurons.size(); i++)
        result[i] = output_layer.neurons[i].output;

    return result;
}

void MultiLayerPerceptron::Train(const DoubleArray& input, const DoubleArray& target) {
    is_training = true;

    for (int i = 0; i < input.size(); i++)
        input_layer.neurons[i].output = input[i];

    FeedForward();
    BackPropagate(target);
    UpdateWeights();
}

void MultiLayerPerceptron::TrainEpoch(DataPointArray& data, int batch_size) {
    if (batch_size > data.size()) batch_size = data.size();
    if (batch_size < 1) batch_size = 1;

    ShuffleData(data);

    for (size_t i = 0; i < data.size(); i += batch_size) {
        int batch_end = std::min((size_t)(i + batch_size), data.size());
        for (size_t j = i; j < batch_end; j++)
            Train(data[j].input, data[j].target);
    }
}

void MultiLayerPerceptron::Save(const std::string& filename) {
    std::ofstream f(filename, std::ios::binary);
    if (!f.is_open()) {
        std::cerr << "Error: Could not open file for writing: " << filename << std::endl;
        return;
    }

    // Write magic
    f.write(MODEL_MAGIC, 9);

    // Write structure
    int layer_count = hidden_layers.size();
    f.write(reinterpret_cast<char*>(&layer_count), sizeof(int));
    f.write(reinterpret_cast<char*>(&input_size), sizeof(int));
    for (int sz : hidden_sizes)
        f.write(reinterpret_cast<char*>(&sz), sizeof(int));
    f.write(reinterpret_cast<char*>(&output_size), sizeof(int));

    // Write hyperparameters
    f.write(reinterpret_cast<char*>(&learning_rate), sizeof(double));
    int opt_int = (int)optimizer;
    f.write(reinterpret_cast<char*>(&opt_int), sizeof(int));
    int hidden_act_int = (int)hidden_activation;
    f.write(reinterpret_cast<char*>(&hidden_act_int), sizeof(int));
    int output_act_int = (int)output_activation;
    f.write(reinterpret_cast<char*>(&output_act_int), sizeof(int));
    f.write(reinterpret_cast<char*>(&dropout_rate), sizeof(double));
    f.write(reinterpret_cast<char*>(&l2_lambda), sizeof(double));
    f.write(reinterpret_cast<char*>(&beta1), sizeof(double));
    f.write(reinterpret_cast<char*>(&beta2), sizeof(double));
    f.write(reinterpret_cast<char*>(&timestep), sizeof(int));
    f.write(reinterpret_cast<char*>(&enable_lr_decay), sizeof(bool));
    f.write(reinterpret_cast<char*>(&lr_decay_rate), sizeof(double));
    f.write(reinterpret_cast<char*>(&lr_decay_epochs), sizeof(int));
    f.write(reinterpret_cast<char*>(&enable_early_stopping), sizeof(bool));
    f.write(reinterpret_cast<char*>(&early_stopping_patience), sizeof(int));

    // Write input layer
    for (const auto& neuron : input_layer.neurons) {
        int num_weights = neuron.weights.size();
        f.write(reinterpret_cast<const char*>(&num_weights), sizeof(int));
        for (double w : neuron.weights)
            f.write(reinterpret_cast<const char*>(&w), sizeof(double));
        f.write(reinterpret_cast<const char*>(&neuron.bias), sizeof(double));
    }

    // Write hidden layers
    for (const auto& layer : hidden_layers) {
        for (const auto& neuron : layer.neurons) {
            int num_weights = neuron.weights.size();
            f.write(reinterpret_cast<const char*>(&num_weights), sizeof(int));
            for (double w : neuron.weights)
                f.write(reinterpret_cast<const char*>(&w), sizeof(double));
            f.write(reinterpret_cast<const char*>(&neuron.bias), sizeof(double));

            for (double m : neuron.m)
                f.write(reinterpret_cast<const char*>(&m), sizeof(double));
            for (double v : neuron.v)
                f.write(reinterpret_cast<const char*>(&v), sizeof(double));
            f.write(reinterpret_cast<const char*>(&neuron.m_bias), sizeof(double));
            f.write(reinterpret_cast<const char*>(&neuron.v_bias), sizeof(double));
        }
    }

    // Write output layer
    for (const auto& neuron : output_layer.neurons) {
        int num_weights = neuron.weights.size();
        f.write(reinterpret_cast<const char*>(&num_weights), sizeof(int));
        for (double w : neuron.weights)
            f.write(reinterpret_cast<const char*>(&w), sizeof(double));
        f.write(reinterpret_cast<const char*>(&neuron.bias), sizeof(double));

        for (double m : neuron.m)
            f.write(reinterpret_cast<const char*>(&m), sizeof(double));
        for (double v : neuron.v)
            f.write(reinterpret_cast<const char*>(&v), sizeof(double));
        f.write(reinterpret_cast<const char*>(&neuron.m_bias), sizeof(double));
        f.write(reinterpret_cast<const char*>(&neuron.v_bias), sizeof(double));
    }

    f.close();
}

void MultiLayerPerceptron::Load(const std::string& filename) {
    std::ifstream f(filename, std::ios::binary);
    if (!f.is_open()) {
        std::cerr << "Error: Could not open file for reading: " << filename << std::endl;
        return;
    }

    // Read and verify magic
    char magic[10];
    f.read(magic, 9);
    magic[9] = '\0';

    // Read structure
    int layer_count;
    f.read(reinterpret_cast<char*>(&layer_count), sizeof(int));
    f.read(reinterpret_cast<char*>(&input_size), sizeof(int));
    
    hidden_sizes.resize(layer_count);
    for (int i = 0; i < layer_count; i++)
        f.read(reinterpret_cast<char*>(&hidden_sizes[i]), sizeof(int));
    
    f.read(reinterpret_cast<char*>(&output_size), sizeof(int));

    // Read hyperparameters
    f.read(reinterpret_cast<char*>(&learning_rate), sizeof(double));
    int opt_int;
    f.read(reinterpret_cast<char*>(&opt_int), sizeof(int));
    optimizer = (OptimizerType)opt_int;
    int hidden_act_int;
    f.read(reinterpret_cast<char*>(&hidden_act_int), sizeof(int));
    hidden_activation = (ActivationType)hidden_act_int;
    int output_act_int;
    f.read(reinterpret_cast<char*>(&output_act_int), sizeof(int));
    output_activation = (ActivationType)output_act_int;
    f.read(reinterpret_cast<char*>(&dropout_rate), sizeof(double));
    f.read(reinterpret_cast<char*>(&l2_lambda), sizeof(double));
    f.read(reinterpret_cast<char*>(&beta1), sizeof(double));
    f.read(reinterpret_cast<char*>(&beta2), sizeof(double));
    f.read(reinterpret_cast<char*>(&timestep), sizeof(int));
    f.read(reinterpret_cast<char*>(&enable_lr_decay), sizeof(bool));
    f.read(reinterpret_cast<char*>(&lr_decay_rate), sizeof(double));
    f.read(reinterpret_cast<char*>(&lr_decay_epochs), sizeof(int));
    f.read(reinterpret_cast<char*>(&enable_early_stopping), sizeof(bool));
    f.read(reinterpret_cast<char*>(&early_stopping_patience), sizeof(int));

    // Initialize architecture
    InitializeLayer(input_layer, input_size + 1, input_size, ActivationType::Sigmoid);
    hidden_layers.resize(layer_count);
    int num_inputs = input_size;
    for (int i = 0; i < layer_count; i++) {
        InitializeLayer(hidden_layers[i], hidden_sizes[i] + 1, num_inputs + 1, hidden_activation);
        num_inputs = hidden_sizes[i];
    }
    InitializeLayer(output_layer, output_size, num_inputs + 1, output_activation);

    // Read input layer
    for (auto& neuron : input_layer.neurons) {
        int num_weights;
        f.read(reinterpret_cast<char*>(&num_weights), sizeof(int));
        neuron.weights.resize(num_weights);
        for (int j = 0; j < num_weights; j++)
            f.read(reinterpret_cast<char*>(&neuron.weights[j]), sizeof(double));
        f.read(reinterpret_cast<char*>(&neuron.bias), sizeof(double));
    }

    // Read hidden layers
    for (auto& layer : hidden_layers) {
        for (auto& neuron : layer.neurons) {
            int num_weights;
            f.read(reinterpret_cast<char*>(&num_weights), sizeof(int));
            neuron.weights.resize(num_weights);
            for (int j = 0; j < num_weights; j++)
                f.read(reinterpret_cast<char*>(&neuron.weights[j]), sizeof(double));
            f.read(reinterpret_cast<char*>(&neuron.bias), sizeof(double));

            neuron.m.resize(num_weights);
            neuron.v.resize(num_weights);
            for (int j = 0; j < num_weights; j++) {
                f.read(reinterpret_cast<char*>(&neuron.m[j]), sizeof(double));
                f.read(reinterpret_cast<char*>(&neuron.v[j]), sizeof(double));
            }
            f.read(reinterpret_cast<char*>(&neuron.m_bias), sizeof(double));
            f.read(reinterpret_cast<char*>(&neuron.v_bias), sizeof(double));
        }
    }

    // Read output layer
    for (auto& neuron : output_layer.neurons) {
        int num_weights;
        f.read(reinterpret_cast<char*>(&num_weights), sizeof(int));
        neuron.weights.resize(num_weights);
        for (int j = 0; j < num_weights; j++)
            f.read(reinterpret_cast<char*>(&neuron.weights[j]), sizeof(double));
        f.read(reinterpret_cast<char*>(&neuron.bias), sizeof(double));

        neuron.m.resize(num_weights);
        neuron.v.resize(num_weights);
        for (int j = 0; j < num_weights; j++) {
            f.read(reinterpret_cast<char*>(&neuron.m[j]), sizeof(double));
            f.read(reinterpret_cast<char*>(&neuron.v[j]), sizeof(double));
        }
        f.read(reinterpret_cast<char*>(&neuron.m_bias), sizeof(double));
        f.read(reinterpret_cast<char*>(&neuron.v_bias), sizeof(double));
    }

    f.close();
}

// Utility functions
std::string ActivationToStr(ActivationType act) {
    switch (act) {
        case ActivationType::Sigmoid: return "sigmoid";
        case ActivationType::Tanh: return "tanh";
        case ActivationType::ReLU: return "relu";
        case ActivationType::Softmax: return "softmax";
        default: return "sigmoid";
    }
}

std::string OptimizerToStr(OptimizerType opt) {
    switch (opt) {
        case OptimizerType::SGD: return "sgd";
        case OptimizerType::Adam: return "adam";
        case OptimizerType::RMSProp: return "rmsprop";
        default: return "sgd";
    }
}

void PrintUsage() {
    std::cout << "MLP - Command-line Multi-Layer Perceptron" << std::endl << std::endl;
    std::cout << "Commands:" << std::endl;
    std::cout << "  create   Create a new MLP model" << std::endl;
    std::cout << "  train    Train an existing model with data" << std::endl;
    std::cout << "  predict  Make predictions with a trained model" << std::endl;
    std::cout << "  info     Display model information" << std::endl;
    std::cout << "  help     Show this help message" << std::endl << std::endl;

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
    std::cout << "  --beta2=VALUE          Adam beta2 (default: 0.999)" << std::endl << std::endl;

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
    std::cout << "  --verbose              Show training progress" << std::endl << std::endl;

    std::cout << "Predict Options:" << std::endl;
    std::cout << "  --model=FILE           Model file to load (required)" << std::endl;
    std::cout << "  --input=v1,v2,...      Input values (required)" << std::endl << std::endl;

    std::cout << "Info Options:" << std::endl;
    std::cout << "  --model=FILE           Model file to load (required)" << std::endl << std::endl;

    std::cout << "Examples:" << std::endl;
    std::cout << "  mlp create --input=2 --hidden=8 --output=1 --save=xor.bin" << std::endl;
    std::cout << "  mlp train --model=xor.bin --data=xor.csv --epochs=1000 --save=xor_trained.bin" << std::endl;
    std::cout << "  mlp predict --model=xor_trained.bin --input=1,0" << std::endl;
    std::cout << "  mlp info --model=xor_trained.bin" << std::endl;
}

ActivationType ParseActivation(const std::string& s) {
    std::string lower = s;
    std::transform(lower.begin(), lower.end(), lower.begin(), ::tolower);
    
    if (lower == "tanh") return ActivationType::Tanh;
    if (lower == "relu") return ActivationType::ReLU;
    if (lower == "softmax") return ActivationType::Softmax;
    return ActivationType::Sigmoid;
}

OptimizerType ParseOptimizer(const std::string& s) {
    std::string lower = s;
    std::transform(lower.begin(), lower.end(), lower.begin(), ::tolower);
    
    if (lower == "adam") return OptimizerType::Adam;
    if (lower == "rmsprop") return OptimizerType::RMSProp;
    return OptimizerType::SGD;
}

IntArray ParseIntArray(const std::string& s) {
    IntArray result;
    std::stringstream ss(s);
    std::string token;
    
    while (std::getline(ss, token, ',')) {
        try {
            result.push_back(std::stoi(token));
        } catch (...) {}
    }
    return result;
}

DoubleArray ParseDoubleArray(const std::string& s) {
    DoubleArray result;
    std::stringstream ss(s);
    std::string token;
    
    while (std::getline(ss, token, ',')) {
        try {
            result.push_back(std::stod(token));
        } catch (...) {}
    }
    return result;
}

void LoadDataCSV(const std::string& filename, int input_size, int output_size, DataPointArray& data) {
    data.clear();
    std::ifstream f(filename);
    
    if (!f.is_open()) {
        std::cerr << "Error: Could not open data file: " << filename << std::endl;
        return;
    }

    std::string line;
    while (std::getline(f, line)) {
        if (line.empty()) continue;

        DoubleArray values = ParseDoubleArray(line);
        if ((int)values.size() < input_size + output_size) continue;

        DataPoint point;
        point.input.resize(input_size);
        point.target.resize(output_size);

        for (int i = 0; i < input_size; i++)
            point.input[i] = values[i];
        for (int i = 0; i < output_size; i++)
            point.target[i] = values[input_size + i];

        data.push_back(point);
    }

    f.close();
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        PrintUsage();
        return 0;
    }

    std::string cmd_str = argv[1];
    CommandType command = CommandType::None;

    if (cmd_str == "create") command = CommandType::Create;
    else if (cmd_str == "train") command = CommandType::Train;
    else if (cmd_str == "predict") command = CommandType::Predict;
    else if (cmd_str == "info") command = CommandType::Info;
    else if (cmd_str == "help" || cmd_str == "--help" || cmd_str == "-h") command = CommandType::Help;
    else {
        std::cerr << "Unknown command: " << cmd_str << std::endl;
        PrintUsage();
        return 1;
    }

    if (command == CommandType::Help) {
        PrintUsage();
        return 0;
    }

    // Initialize defaults
    int input_size = 0, output_size = 0, epochs = 100, batch_size = 1;
    int lr_decay_epochs = 10, patience = 10;
    IntArray hidden_sizes;
    double learning_rate = 0.1, dropout_rate = 0.0, l2_lambda = 0.0;
    double beta1 = 0.9, beta2 = 0.999, lr_decay_rate = 0.95;
    bool lr_decay = false, early_stop = false, normalize = false, verbose = false;
    ActivationType hidden_act = ActivationType::Sigmoid, output_act = ActivationType::Sigmoid;
    OptimizerType optimizer = OptimizerType::SGD;
    std::string model_file, save_file, data_file;
    DoubleArray input_values;

    // Parse arguments
    for (int i = 2; i < argc; i++) {
        std::string arg = argv[i];
        
        if (arg == "--lr-decay") {
            lr_decay = true;
        } else if (arg == "--early-stop") {
            early_stop = true;
        } else if (arg == "--normalize") {
            normalize = true;
        } else if (arg == "--verbose") {
            verbose = true;
        } else {
            size_t eq_pos = arg.find('=');
            if (eq_pos == std::string::npos) {
                std::cerr << "Invalid argument: " << arg << std::endl;
                continue;
            }

            std::string key = arg.substr(0, eq_pos);
            std::string value = arg.substr(eq_pos + 1);

            if (key == "--input") {
                if (command == CommandType::Predict)
                    input_values = ParseDoubleArray(value);
                else
                    input_size = std::stoi(value);
            } else if (key == "--hidden") {
                hidden_sizes = ParseIntArray(value);
            } else if (key == "--output") {
                output_size = std::stoi(value);
            } else if (key == "--save") {
                save_file = value;
            } else if (key == "--model") {
                model_file = value;
            } else if (key == "--data") {
                data_file = value;
            } else if (key == "--lr") {
                learning_rate = std::stod(value);
            } else if (key == "--optimizer") {
                optimizer = ParseOptimizer(value);
            } else if (key == "--hidden-act") {
                hidden_act = ParseActivation(value);
            } else if (key == "--output-act") {
                output_act = ParseActivation(value);
            } else if (key == "--dropout") {
                dropout_rate = std::stod(value);
            } else if (key == "--l2") {
                l2_lambda = std::stod(value);
            } else if (key == "--beta1") {
                beta1 = std::stod(value);
            } else if (key == "--beta2") {
                beta2 = std::stod(value);
            } else if (key == "--epochs") {
                epochs = std::stoi(value);
            } else if (key == "--batch") {
                batch_size = std::stoi(value);
            } else if (key == "--lr-decay-rate") {
                lr_decay_rate = std::stod(value);
            } else if (key == "--lr-decay-epochs") {
                lr_decay_epochs = std::stoi(value);
            } else if (key == "--patience") {
                patience = std::stoi(value);
            } else {
                std::cerr << "Unknown option: " << key << std::endl;
            }
        }
    }

    // Execute command
    if (command == CommandType::Create) {
        if (input_size <= 0) { std::cerr << "Error: --input is required" << std::endl; return 1; }
        if (hidden_sizes.empty()) { std::cerr << "Error: --hidden is required" << std::endl; return 1; }
        if (output_size <= 0) { std::cerr << "Error: --output is required" << std::endl; return 1; }
        if (save_file.empty()) { std::cerr << "Error: --save is required" << std::endl; return 1; }

        auto mlp = std::make_unique<MultiLayerPerceptron>(input_size, hidden_sizes, output_size, hidden_act, output_act);
        mlp->learning_rate = learning_rate;
        mlp->optimizer = optimizer;
        mlp->dropout_rate = dropout_rate;
        mlp->l2_lambda = l2_lambda;
        mlp->beta1 = beta1;
        mlp->beta2 = beta2;

        mlp->Save(save_file);

        std::cout << "Created MLP model:" << std::endl;
        std::cout << "  Input size: " << input_size << std::endl;
        std::cout << "  Hidden sizes: ";
        for (size_t i = 0; i < hidden_sizes.size(); i++) {
            if (i > 0) std::cout << ",";
            std::cout << hidden_sizes[i];
        }
        std::cout << std::endl;
        std::cout << "  Output size: " << output_size << std::endl;
        std::cout << "  Hidden activation: " << ActivationToStr(hidden_act) << std::endl;
        std::cout << "  Output activation: " << ActivationToStr(output_act) << std::endl;
        std::cout << "  Optimizer: " << OptimizerToStr(optimizer) << std::endl;
        std::cout << std::fixed << std::setprecision(4);
        std::cout << "  Learning rate: " << learning_rate << std::endl;
        std::cout << "  Saved to: " << save_file << std::endl;
    }
    else if (command == CommandType::Train) {
        if (model_file.empty()) { std::cerr << "Error: --model is required" << std::endl; return 1; }
        if (data_file.empty()) { std::cerr << "Error: --data is required" << std::endl; return 1; }
        if (save_file.empty()) { std::cerr << "Error: --save is required" << std::endl; return 1; }

        auto mlp = std::make_unique<MultiLayerPerceptron>(1, IntArray{1}, 1);
        mlp->Load(model_file);

        mlp->learning_rate = learning_rate;
        mlp->enable_lr_decay = lr_decay;
        mlp->lr_decay_rate = lr_decay_rate;
        mlp->lr_decay_epochs = lr_decay_epochs;
        mlp->enable_early_stopping = early_stop;
        mlp->early_stopping_patience = patience;

        DataPointArray data;
        LoadDataCSV(data_file, mlp->GetInputSize(), mlp->GetOutputSize(), data);
        if (data.empty()) { std::cerr << "Error: No valid data loaded" << std::endl; return 1; }

        std::cout << "Loaded " << data.size() << " training samples" << std::endl;
        if (normalize) {
            NormalizeData(data);
            std::cout << "Data normalized" << std::endl;
        }

        for (int epoch = 1; epoch <= epochs; epoch++) {
            ShuffleData(data);
            
            for (auto& point : data)
                mlp->Train(point.input, point.target);

            if (verbose && (epoch % 10 == 0 || epoch == 1)) {
                double loss = 0.0;
                for (auto& point : data) {
                    auto output = mlp->Predict(point.input);
                    loss += mlp->ComputeLoss(output, point.target);
                }
                std::cout << "Epoch " << epoch << "/" << epochs << " - Loss: "
                          << std::fixed << std::setprecision(6) << (loss / data.size()) << std::endl;
            }
        }

        double loss = 0.0;
        for (auto& point : data) {
            auto output = mlp->Predict(point.input);
            loss += mlp->ComputeLoss(output, point.target);
        }
        std::cout << "Final loss: " << std::fixed << std::setprecision(6) << (loss / data.size()) << std::endl;

        mlp->Save(save_file);
        std::cout << "Model saved to: " << save_file << std::endl;
    }
    else if (command == CommandType::Predict) {
        if (model_file.empty()) { std::cerr << "Error: --model is required" << std::endl; return 1; }
        if (input_values.empty()) { std::cerr << "Error: --input is required" << std::endl; return 1; }

        auto mlp = std::make_unique<MultiLayerPerceptron>(1, IntArray{1}, 1);
        mlp->Load(model_file);

        if ((int)input_values.size() != mlp->GetInputSize()) {
            std::cerr << "Error: Expected " << mlp->GetInputSize() << " input values, got "
                      << input_values.size() << std::endl;
            return 1;
        }

        auto output = mlp->Predict(input_values);

        std::cout << "Input: ";
        for (size_t i = 0; i < input_values.size(); i++) {
            if (i > 0) std::cout << ", ";
            std::cout << std::fixed << std::setprecision(4) << input_values[i];
        }
        std::cout << std::endl;

        std::cout << "Output: ";
        for (size_t i = 0; i < output.size(); i++) {
            if (i > 0) std::cout << ", ";
            std::cout << std::fixed << std::setprecision(6) << output[i];
        }
        std::cout << std::endl;

        if (output.size() > 1) {
            int max_idx = MaxIndex(output);
            std::cout << "Max index: " << max_idx << std::endl;
        }
    }
    else if (command == CommandType::Info) {
        if (model_file.empty()) { std::cerr << "Error: --model is required" << std::endl; return 1; }

        auto mlp = std::make_unique<MultiLayerPerceptron>(1, IntArray{1}, 1);
        mlp->Load(model_file);

        std::cout << "MLP Model Information" << std::endl;
        std::cout << "=====================" << std::endl;
        std::cout << "Input size: " << mlp->GetInputSize() << std::endl;
        std::cout << "Output size: " << mlp->GetOutputSize() << std::endl;
        std::cout << "Hidden layers: " << mlp->GetHiddenLayerCount() << std::endl;
        std::cout << "Layer sizes: " << mlp->GetInputSize();
        for (int i = 0; i < mlp->GetHiddenLayerCount(); i++)
            std::cout << " -> " << mlp->GetHiddenLayer(i).neurons.size();
        std::cout << " -> " << mlp->GetOutputSize() << std::endl << std::endl;

        std::cout << "Hyperparameters:" << std::endl;
        std::cout << std::fixed << std::setprecision(6);
        std::cout << "  Learning rate: " << mlp->learning_rate << std::endl;
        std::cout << "  Optimizer: " << OptimizerToStr(mlp->optimizer) << std::endl;
        std::cout << "  Hidden activation: " << ActivationToStr(mlp->hidden_activation) << std::endl;
        std::cout << "  Output activation: " << ActivationToStr(mlp->output_activation) << std::endl;
        std::cout << std::setprecision(4);
        std::cout << "  Dropout rate: " << mlp->dropout_rate << std::endl;
        std::cout << std::setprecision(6);
        std::cout << "  L2 lambda: " << mlp->l2_lambda << std::endl;
        std::cout << std::setprecision(4);
        std::cout << "  Beta1: " << mlp->beta1 << std::endl;
        std::cout << "  Beta2: " << mlp->beta2 << std::endl;
        std::cout << std::setprecision(0);
        std::cout << "  Timestep: " << mlp->timestep << std::endl << std::endl;

        std::cout << "Total layers: " << (mlp->GetHiddenLayerCount() + 2) << std::endl;
        std::cout << "  Layer 0: " << mlp->GetInputSize() << " neurons (input)" << std::endl;
        for (int i = 0; i < mlp->GetHiddenLayerCount(); i++)
            std::cout << "  Layer " << (i + 1) << ": " << mlp->GetHiddenLayer(i).neurons.size() << " neurons" << std::endl;
        std::cout << "  Layer " << (mlp->GetHiddenLayerCount() + 1) << ": " << mlp->GetOutputSize() << " neurons (output)" << std::endl;
    }

    return 0;
}
