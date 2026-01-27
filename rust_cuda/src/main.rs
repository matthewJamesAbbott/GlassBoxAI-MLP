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

#![allow(non_snake_case)]
#![allow(non_camel_case_types)]
#![allow(dead_code)]
#![allow(unused_variables)]
#![allow(unused_assignments)]

use std::env;
use std::fs::File;
use std::io::{self, BufRead, BufReader, Write};
use std::process;
use std::sync::Arc;

use cudarc::driver::{CudaDevice, CudaSlice, LaunchAsync, LaunchConfig};
use rand::Rng;
use serde::{Deserialize, Serialize};

const EPSILON: f64 = 1e-15;
const BLOCK_SIZE: u32 = 256;
const MODEL_MAGIC: &str = "MLPBKND01";

const CUDA_KERNEL_SRC: &str = r#"
extern "C" {

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
    return x * (1.0 - x);
}

__device__ double d_TanhActivation(double x) {
    return tanh(x);
}

__device__ double d_DTanh(double x) {
    return 1.0 - (x * x);
}

__device__ double d_ReLU(double x) {
    return (x > 0.0) ? x : 0.0;
}

__device__ double d_DReLU(double x) {
    return (x > 0.0) ? 1.0 : 0.0;
}

__device__ double d_ApplyActivation(double x, int actType) {
    switch (actType) {
        case 0: return d_Sigmoid(x);      // atSigmoid
        case 1: return d_TanhActivation(x); // atTanh
        case 2: return d_ReLU(x);          // atReLU
        case 4: return x;                  // atLinear
        default: return d_Sigmoid(x);
    }
}

__device__ double d_ApplyActivationDerivative(double x, int actType) {
    switch (actType) {
        case 0: return d_DSigmoid(x);  // atSigmoid
        case 1: return d_DTanh(x);     // atTanh
        case 2: return d_DReLU(x);     // atReLU
        case 4: return 1.0;            // atLinear
        default: return d_DSigmoid(x);
    }
}

__global__ void FeedForwardKernel(
    double* outputs, double* weights, double* biases,
    double* prevOutputs, int numNeurons, int numInputs, int prevSize, int actType
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < numNeurons) {
        double sum = biases[i];
        for (int j = 0; j < prevSize; j++) {
            sum += prevOutputs[j] * weights[i * numInputs + j];
        }
        outputs[i] = d_ApplyActivation(sum, actType);
    }
}

__global__ void FeedForwardSoftmaxSumKernel(
    double* sums, double* weights, double* biases,
    double* prevOutputs, int numNeurons, int numInputs, int prevSize
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < numNeurons) {
        double sum = biases[i];
        for (int j = 0; j < prevSize; j++) {
            sum += prevOutputs[j] * weights[i * numInputs + j];
        }
        sums[i] = sum;
    }
}

__global__ void SoftmaxKernel(double* sums, double* outputs, int n, double maxVal, double sumExp) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        double val = exp(sums[i] - maxVal) / sumExp;
        if (val < 1e-15) val = 1e-15;
        else if (val > 1.0 - 1e-15) val = 1.0 - 1e-15;
        outputs[i] = val;
    }
}

__global__ void ApplyDropoutKernel(
    double* outputs, bool* dropoutMask, int numNeurons,
    double dropoutRate, double scale, unsigned long seed
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < numNeurons) {
        if (dropoutRate <= 0.0) {
            dropoutMask[i] = true;
            return;
        }
        // Simple LCG random
        unsigned long state = seed + i * 1099087573UL;
        state = state * 1103515245UL + 12345UL;
        float randVal = (float)(state % 10000) / 10000.0f;
        
        if (randVal > dropoutRate) {
            dropoutMask[i] = true;
            outputs[i] = outputs[i] * scale;
        } else {
            dropoutMask[i] = false;
            outputs[i] = 0.0;
        }
    }
}

__global__ void BackPropOutputKernel(
    double* errors, double* outputs, double* target,
    int numNeurons, int actType, bool isSoftmax
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < numNeurons) {
        if (isSoftmax) {
            errors[i] = target[i] - outputs[i];
        } else {
            errors[i] = d_ApplyActivationDerivative(outputs[i], actType) * (target[i] - outputs[i]);
        }
    }
}

__global__ void BackPropHiddenKernel(
    double* errors, double* outputs, bool* dropoutMask,
    double* nextErrors, double* nextWeights,
    int numNeurons, int nextNumNeurons, int nextNumInputs, int actType
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < numNeurons) {
        if (!dropoutMask[i]) {
            errors[i] = 0.0;
            return;
        }
        double errorSum = 0.0;
        for (int j = 0; j < nextNumNeurons; j++) {
            errorSum += nextErrors[j] * nextWeights[j * nextNumInputs + i];
        }
        errors[i] = d_ApplyActivationDerivative(outputs[i], actType) * errorSum;
    }
}

__global__ void UpdateWeightsSGDKernel(
    double* weights, double* biases, double* errors,
    double* prevOutputs, int numNeurons, int numInputs, int prevSize,
    double learningRate, double l2Lambda, double clipVal
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < numNeurons) {
        for (int j = 0; j < prevSize; j++) {
            double gradient = errors[i] * prevOutputs[j];
            if (l2Lambda > 0.0)
                gradient = gradient - l2Lambda * weights[i * numInputs + j];
            gradient = d_clip(gradient, clipVal);
            weights[i * numInputs + j] += learningRate * gradient;
        }
        double biasGrad = d_clip(errors[i], clipVal);
        biases[i] += learningRate * biasGrad;
    }
}

__global__ void UpdateWeightsAdamKernel(
    double* weights, double* biases, double* errors,
    double* prevOutputs, double* M, double* V, double* MBias, double* VBias,
    int numNeurons, int numInputs, int prevSize,
    double* params
) {
    double learningRate = params[0];
    double l2Lambda = params[1];
    double beta1 = params[2];
    double beta2 = params[3];
    int timestep = (int)params[4];
    double clipVal = params[5];
    
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < numNeurons) {
        double eps = 1e-8;
        double beta1_t = pow(beta1, (double)timestep);
        double beta2_t = pow(beta2, (double)timestep);

        for (int j = 0; j < prevSize; j++) {
            int idx = i * numInputs + j;
            double gradient = -errors[i] * prevOutputs[j];
            if (l2Lambda > 0.0)
                gradient += l2Lambda * weights[idx];
            gradient = d_clip(gradient, clipVal);

            M[idx] = beta1 * M[idx] + (1.0 - beta1) * gradient;
            V[idx] = beta2 * V[idx] + (1.0 - beta2) * gradient * gradient;

            double mHat = M[idx] / (1.0 - beta1_t);
            double vHat = V[idx] / (1.0 - beta2_t);

            weights[idx] -= learningRate * mHat / (sqrt(vHat) + eps);
        }

        double gradient = d_clip(-errors[i], clipVal);
        MBias[i] = beta1 * MBias[i] + (1.0 - beta1) * gradient;
        VBias[i] = beta2 * VBias[i] + (1.0 - beta2) * gradient * gradient;
        double mHat = MBias[i] / (1.0 - beta1_t);
        double vHat = VBias[i] / (1.0 - beta2_t);
        biases[i] -= learningRate * mHat / (sqrt(vHat) + eps);
    }
}

__global__ void UpdateWeightsRMSPropKernel(
    double* weights, double* biases, double* errors,
    double* prevOutputs, double* V, double* VBias,
    int numNeurons, int numInputs, int prevSize,
    double learningRate, double l2Lambda, double clipVal
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < numNeurons) {
        double eps = 1e-8;
        double decay = 0.9;

        for (int j = 0; j < prevSize; j++) {
            int idx = i * numInputs + j;
            double gradient = -errors[i] * prevOutputs[j];
            if (l2Lambda > 0.0)
                gradient += l2Lambda * weights[idx];
            gradient = d_clip(gradient, clipVal);

            V[idx] = decay * V[idx] + (1.0 - decay) * gradient * gradient;
            weights[idx] -= learningRate * gradient / (sqrt(V[idx]) + eps);
        }

        double gradient = d_clip(-errors[i], clipVal);
        VBias[i] = decay * VBias[i] + (1.0 - decay) * gradient * gradient;
        biases[i] -= learningRate * gradient / (sqrt(VBias[i]) + eps);
    }
}

__global__ void BatchNormForwardTrainKernel(
    double* output, double* input, double* gamma, double* beta,
    double* runningMean, double* runningVar,
    double* batchMean, double* batchVar,
    int n, double momentum, double epsilon
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        double mean = batchMean[i];
        double var = batchVar[i];
        double xNorm = (input[i] - mean) / sqrt(var + epsilon);
        output[i] = gamma[i] * xNorm + beta[i];
        runningMean[i] = (1.0 - momentum) * runningMean[i] + momentum * mean;
        runningVar[i] = (1.0 - momentum) * runningVar[i] + momentum * var;
    }
}

__global__ void BatchNormForwardInferKernel(
    double* output, double* input, double* gamma, double* beta,
    double* runningMean, double* runningVar,
    int n, double epsilon
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        double xNorm = (input[i] - runningMean[i]) / sqrt(runningVar[i] + epsilon);
        output[i] = gamma[i] * xNorm + beta[i];
    }
}

__global__ void BatchNormBackwardKernel(
    double* dInput, double* dGamma, double* dBeta,
    double* dOutput, double* input, double* gamma,
    double* batchMean, double* batchVar,
    int n, double epsilon
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        double mean = batchMean[i];
        double var = batchVar[i];
        double stdInv = 1.0 / sqrt(var + epsilon);
        double xNorm = (input[i] - mean) * stdInv;
        
        dGamma[i] = dOutput[i] * xNorm;
        dBeta[i] = dOutput[i];
        dInput[i] = dOutput[i] * gamma[i] * stdInv;
    }
}

__global__ void ComputeBatchMeanVarKernel(
    double* batchMean, double* batchVar, double* input, int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        batchMean[i] = input[i];
        batchVar[i] = 0.0;
    }
}

} // extern "C"
"#;

#[derive(Clone, Copy, PartialEq, Eq, Debug, Serialize, Deserialize)]
#[repr(i32)]
pub enum TActivationType {
    atSigmoid = 0,
    atTanh = 1,
    atReLU = 2,
    atSoftmax = 3,
    atLinear = 4,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug, Serialize, Deserialize)]
#[repr(i32)]
pub enum TOptimizerType {
    otSGD = 0,
    otAdam = 1,
    otRMSProp = 2,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum TCommand {
    CmdNone,
    CmdCreate,
    CmdTrain,
    CmdPredict,
    CmdInfo,
    CmdHelp,
    CmdExportONNX,
    CmdFeatureImportance,
}

pub type Darray = Vec<f64>;
pub type TDoubleArray = Vec<f64>;
pub type TIntArray = Vec<i32>;

#[derive(Clone, Debug)]
pub struct TDataPoint {
    pub Input: Darray,
    pub Target: Darray,
}

pub type TDataPointArray = Vec<TDataPoint>;

#[derive(Debug)]
pub struct LayerDataCUDA {
    pub Weights: CudaSlice<f64>,
    pub Biases: CudaSlice<f64>,
    pub Outputs: CudaSlice<f64>,
    pub Errors: CudaSlice<f64>,
    pub M: CudaSlice<f64>,
    pub V: CudaSlice<f64>,
    pub MBias: CudaSlice<f64>,
    pub VBias: CudaSlice<f64>,
    pub DropoutMask: CudaSlice<u8>,
    pub NumNeurons: i32,
    pub NumInputs: i32,
    pub ActivationType: TActivationType,
    pub d_Gamma: CudaSlice<f64>,
    pub d_Beta: CudaSlice<f64>,
    pub d_RunningMean: CudaSlice<f64>,
    pub d_RunningVar: CudaSlice<f64>,
    pub d_BatchMean: CudaSlice<f64>,
    pub d_BatchVar: CudaSlice<f64>,
}

fn ActivationToStr(act: TActivationType) -> &'static str {
    match act {
        TActivationType::atSigmoid => "sigmoid",
        TActivationType::atTanh => "tanh",
        TActivationType::atReLU => "relu",
        TActivationType::atSoftmax => "softmax",
        TActivationType::atLinear => "linear",
    }
}

fn OptimizerToStr(opt: TOptimizerType) -> &'static str {
    match opt {
        TOptimizerType::otSGD => "sgd",
        TOptimizerType::otAdam => "adam",
        TOptimizerType::otRMSProp => "rmsprop",
    }
}

fn ParseActivation(s: &str) -> Result<TActivationType, String> {
    let lower = s.to_lowercase();
    match lower.as_str() {
        "tanh" => Ok(TActivationType::atTanh),
        "relu" => Ok(TActivationType::atReLU),
        "softmax" => Ok(TActivationType::atSoftmax),
        "sigmoid" => Ok(TActivationType::atSigmoid),
        "linear" => Ok(TActivationType::atLinear),
        _ => Err(format!("Error: Invalid activation function: {}", s)),
    }
}

fn ParseOptimizer(s: &str) -> TOptimizerType {
    let lower = s.to_lowercase();
    match lower.as_str() {
        "adam" => TOptimizerType::otAdam,
        "rmsprop" => TOptimizerType::otRMSProp,
        _ => TOptimizerType::otSGD,
    }
}

fn ParseIntArrayHelper(s: &str, result: &mut TIntArray) {
    result.clear();
    for item in s.split(',') {
        if let Ok(v) = item.trim().parse::<i32>() {
            result.push(v);
        }
    }
}

fn ParseDoubleArrayHelper(s: &str, result: &mut TDoubleArray) {
    result.clear();
    for item in s.split(',') {
        if let Ok(v) = item.trim().parse::<f64>() {
            result.push(v);
        }
    }
}

fn MaxIndex(arr: &Darray) -> usize {
    let mut result = 0;
    for i in 1..arr.len() {
        if arr[i] > arr[result] {
            result = i;
        }
    }
    result
}

#[derive(Serialize, Deserialize)]
struct NeuronJSON {
    weights: Vec<f64>,
    bias: f64,
}

#[derive(Serialize, Deserialize)]
struct HiddenLayerJSON {
    neuron_count: i32,
    neurons: Vec<NeuronJSON>,
    biases: Vec<f64>,
}

#[derive(Serialize, Deserialize)]
struct InputLayerJSON {
    neuron_count: i32,
}

#[derive(Serialize, Deserialize)]
struct OutputLayerJSON {
    neuron_count: i32,
    neurons: Vec<NeuronJSON>,
    biases: Vec<f64>,
}

#[derive(Serialize, Deserialize)]
struct BatchNormParamsJSON {
    gamma: Vec<f64>,
    beta: Vec<f64>,
    running_mean: Vec<f64>,
    running_var: Vec<f64>,
}

#[derive(Serialize, Deserialize)]
struct ModelJSON {
    magic: String,
    input_size: i32,
    output_size: i32,
    hidden_sizes: Vec<i32>,
    learning_rate: f64,
    optimizer: i32,
    hidden_activation: i32,
    output_activation: i32,
    dropout_rate: f64,
    l2_lambda: f64,
    beta1: f64,
    beta2: f64,
    #[serde(default)]
    batch_norm: bool,
    input_layer: InputLayerJSON,
    hidden_layers: Vec<HiddenLayerJSON>,
    output_layer: OutputLayerJSON,
    #[serde(default)]
    batch_norm_params: Vec<BatchNormParamsJSON>,
}

pub struct TMultiLayerPerceptronCUDA {
    Dev: Arc<CudaDevice>,
    Layers: Vec<LayerDataCUDA>,
    NumLayers: i32,
    FInputSize: i32,
    FOutputSize: i32,
    FHiddenSizes: Vec<i32>,
    FIsTraining: bool,
    MaxNeurons: i32,

    d_Target: CudaSlice<f64>,
    d_SoftmaxSums: CudaSlice<f64>,
    d_AdamParams: CudaSlice<f64>,

    pub LearningRate: f64,
    pub MaxIterations: i32,
    pub Optimizer: TOptimizerType,
    pub HiddenActivation: TActivationType,
    pub OutputActivation: TActivationType,
    pub DropoutRate: f64,
    pub L2Lambda: f64,
    pub Beta1: f64,
    pub Beta2: f64,
    pub Timestep: i32,
    pub EnableLRDecay: bool,
    pub LRDecayRate: f64,
    pub LRDecayEpochs: i32,
    pub EnableEarlyStopping: bool,
    pub EarlyStoppingPatience: i32,
    pub BatchNormEnabled: bool,
    pub BatchNormMomentum: f64,
    pub BatchNormEpsilon: f64,
}

impl TMultiLayerPerceptronCUDA {
    pub fn new(
        InputSize: i32,
        HiddenSizes: &[i32],
        OutputSize: i32,
        HiddenAct: TActivationType,
        OutputAct: TActivationType,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let dev = CudaDevice::new(0)?;
        
        // Compile CUDA kernels
        let ptx = cudarc::nvrtc::compile_ptx(CUDA_KERNEL_SRC)?;
        dev.load_ptx(ptx, "mlp_kernels", &[
            "FeedForwardKernel",
            "FeedForwardSoftmaxSumKernel",
            "SoftmaxKernel",
            "ApplyDropoutKernel",
            "BackPropOutputKernel",
            "BackPropHiddenKernel",
            "UpdateWeightsSGDKernel",
            "UpdateWeightsAdamKernel",
            "UpdateWeightsRMSPropKernel",
            "BatchNormForwardTrainKernel",
            "BatchNormForwardInferKernel",
            "BatchNormBackwardKernel",
            "ComputeBatchMeanVarKernel",
        ])?;

        let num_layers = (HiddenSizes.len() as i32) + 2;
        let mut layers = Vec::new();
        let mut max_neurons = InputSize + 1;

        // Input layer (layer 0)
        layers.push(Self::AllocateLayerStatic(
            &dev,
            InputSize + 1,
            InputSize,
            TActivationType::atSigmoid,
        )?);

        // Hidden layers
        let mut num_inputs = InputSize;
        for &hidden_size in HiddenSizes.iter() {
            layers.push(Self::AllocateLayerStatic(
                &dev,
                hidden_size + 1,
                num_inputs + 1,
                HiddenAct,
            )?);
            if hidden_size + 1 > max_neurons {
                max_neurons = hidden_size + 1;
            }
            num_inputs = hidden_size;
        }

        // Output layer
        layers.push(Self::AllocateLayerStatic(
            &dev,
            OutputSize,
            num_inputs + 1,
            OutputAct,
        )?);
        if OutputSize > max_neurons {
            max_neurons = OutputSize;
        }

        let d_target = dev.alloc_zeros::<f64>(OutputSize as usize)?;
        let d_softmax_sums = dev.alloc_zeros::<f64>(OutputSize as usize)?;
        let d_adam_params = dev.alloc_zeros::<f64>(6)?;

        Ok(TMultiLayerPerceptronCUDA {
            Dev: dev,
            Layers: layers,
            NumLayers: num_layers,
            FInputSize: InputSize,
            FOutputSize: OutputSize,
            FHiddenSizes: HiddenSizes.to_vec(),
            FIsTraining: true,
            MaxNeurons: max_neurons,

            d_Target: d_target,
            d_SoftmaxSums: d_softmax_sums,
            d_AdamParams: d_adam_params,

            LearningRate: 0.1,
            MaxIterations: 100,
            Optimizer: TOptimizerType::otSGD,
            HiddenActivation: HiddenAct,
            OutputActivation: OutputAct,
            DropoutRate: 0.0,
            L2Lambda: 0.0,
            Beta1: 0.9,
            Beta2: 0.999,
            Timestep: 0,
            EnableLRDecay: false,
            LRDecayRate: 0.95,
            LRDecayEpochs: 10,
            EnableEarlyStopping: false,
            EarlyStoppingPatience: 10,
            BatchNormEnabled: false,
            BatchNormMomentum: 0.1,
            BatchNormEpsilon: 1e-5,
        })
    }

    fn AllocateLayerStatic(
        dev: &Arc<CudaDevice>,
        num_neurons: i32,
        num_inputs: i32,
        act_type: TActivationType,
    ) -> Result<LayerDataCUDA, Box<dyn std::error::Error>> {
        let weight_size = (num_neurons * num_inputs) as usize;
        let neuron_count = num_neurons as usize;

        let limit = if act_type == TActivationType::atReLU {
            (2.0 / num_inputs as f64).sqrt()
        } else {
            (6.0 / (num_inputs + num_neurons) as f64).sqrt()
        };

        let mut rng = rand::thread_rng();
        let weights: Vec<f64> = (0..weight_size)
            .map(|_| (rng.gen::<f64>() * 2.0 - 1.0) * limit)
            .collect();

        let d_weights = dev.htod_copy(weights)?;
        let d_biases = dev.alloc_zeros::<f64>(neuron_count)?;
        let d_outputs = dev.alloc_zeros::<f64>(neuron_count)?;
        let d_errors = dev.alloc_zeros::<f64>(neuron_count)?;
        let d_m = dev.alloc_zeros::<f64>(weight_size)?;
        let d_v = dev.alloc_zeros::<f64>(weight_size)?;
        let d_mbias = dev.alloc_zeros::<f64>(neuron_count)?;
        let d_vbias = dev.alloc_zeros::<f64>(neuron_count)?;
        
        let dropout_mask: Vec<u8> = vec![1u8; neuron_count];
        let d_dropout_mask = dev.htod_copy(dropout_mask)?;

        let gamma_init: Vec<f64> = vec![1.0; neuron_count];
        let d_gamma = dev.htod_copy(gamma_init)?;
        let d_beta = dev.alloc_zeros::<f64>(neuron_count)?;
        let d_running_mean = dev.alloc_zeros::<f64>(neuron_count)?;
        let running_var_init: Vec<f64> = vec![1.0; neuron_count];
        let d_running_var = dev.htod_copy(running_var_init)?;
        let d_batch_mean = dev.alloc_zeros::<f64>(neuron_count)?;
        let d_batch_var = dev.alloc_zeros::<f64>(neuron_count)?;

        Ok(LayerDataCUDA {
            Weights: d_weights,
            Biases: d_biases,
            Outputs: d_outputs,
            Errors: d_errors,
            M: d_m,
            V: d_v,
            MBias: d_mbias,
            VBias: d_vbias,
            DropoutMask: d_dropout_mask,
            NumNeurons: num_neurons,
            NumInputs: num_inputs,
            ActivationType: act_type,
            d_Gamma: d_gamma,
            d_Beta: d_beta,
            d_RunningMean: d_running_mean,
            d_RunningVar: d_running_var,
            d_BatchMean: d_batch_mean,
            d_BatchVar: d_batch_var,
        })
    }

    fn GetBlocks(n: i32) -> u32 {
        ((n as u32) + BLOCK_SIZE - 1) / BLOCK_SIZE
    }

    pub fn FeedForward(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        let num_layers = self.NumLayers as usize;
        let ff_kernel = self.Dev.get_func("mlp_kernels", "FeedForwardKernel").unwrap();
        let dropout_kernel = self.Dev.get_func("mlp_kernels", "ApplyDropoutKernel").unwrap();

        for k in 1..(num_layers - 1) {
            {
                let layer = &self.Layers[k];
                let prev_layer = &self.Layers[k - 1];
                let blocks = Self::GetBlocks(layer.NumNeurons);
                let cfg = LaunchConfig {
                    block_dim: (BLOCK_SIZE, 1, 1),
                    grid_dim: (blocks, 1, 1),
                    shared_mem_bytes: 0,
                };

                unsafe {
                    ff_kernel.clone().launch(cfg, (
                        &layer.Outputs,
                        &layer.Weights,
                        &layer.Biases,
                        &prev_layer.Outputs,
                        layer.NumNeurons,
                        layer.NumInputs,
                        prev_layer.NumNeurons,
                        layer.ActivationType as i32,
                    ))?;
                }
            }

            if self.BatchNormEnabled {
                self.ApplyBatchNorm(k)?;
            }

            if self.FIsTraining && self.DropoutRate > 0.0 {
                let layer = &self.Layers[k];
                let blocks = Self::GetBlocks(layer.NumNeurons);
                let cfg = LaunchConfig {
                    block_dim: (BLOCK_SIZE, 1, 1),
                    grid_dim: (blocks, 1, 1),
                    shared_mem_bytes: 0,
                };
                let scale = 1.0 / (1.0 - self.DropoutRate);
                let seed = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_nanos() as u64;

                unsafe {
                    dropout_kernel.clone().launch(cfg, (
                        &layer.Outputs,
                        &layer.DropoutMask,
                        layer.NumNeurons,
                        self.DropoutRate,
                        scale,
                        seed,
                    ))?;
                }
            }
        }

        // Output layer
        let output_layer = &self.Layers[num_layers - 1];
        let last_hidden = &self.Layers[num_layers - 2];
        let blocks = Self::GetBlocks(output_layer.NumNeurons);
        let cfg = LaunchConfig {
            block_dim: (BLOCK_SIZE, 1, 1),
            grid_dim: (blocks, 1, 1),
            shared_mem_bytes: 0,
        };

        if self.OutputActivation == TActivationType::atSoftmax {
            let softmax_sum_kernel = self.Dev.get_func("mlp_kernels", "FeedForwardSoftmaxSumKernel").unwrap();
            let softmax_kernel = self.Dev.get_func("mlp_kernels", "SoftmaxKernel").unwrap();

            unsafe {
                softmax_sum_kernel.clone().launch(cfg, (
                    &self.d_SoftmaxSums,
                    &output_layer.Weights,
                    &output_layer.Biases,
                    &last_hidden.Outputs,
                    output_layer.NumNeurons,
                    output_layer.NumInputs,
                    last_hidden.NumNeurons,
                ))?;
            }

            self.Dev.synchronize()?;

            let h_sums = self.Dev.dtoh_sync_copy(&self.d_SoftmaxSums)?;
            let max_val = h_sums.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let sum_exp: f64 = h_sums.iter().map(|&s| (s - max_val).exp()).sum();

            unsafe {
                softmax_kernel.clone().launch(cfg, (
                    &self.d_SoftmaxSums,
                    &output_layer.Outputs,
                    output_layer.NumNeurons,
                    max_val,
                    sum_exp,
                ))?;
            }
        } else {
            unsafe {
                ff_kernel.clone().launch(cfg, (
                    &output_layer.Outputs,
                    &output_layer.Weights,
                    &output_layer.Biases,
                    &last_hidden.Outputs,
                    output_layer.NumNeurons,
                    output_layer.NumInputs,
                    last_hidden.NumNeurons,
                    output_layer.ActivationType as i32,
                ))?;
            }
        }

        self.Dev.synchronize()?;
        Ok(())
    }

    pub fn BackPropagate(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        let num_layers = self.NumLayers as usize;

        // Output layer errors
        {
            let output_layer = &self.Layers[num_layers - 1];
            let blocks = Self::GetBlocks(output_layer.NumNeurons);
            let cfg = LaunchConfig {
                block_dim: (BLOCK_SIZE, 1, 1),
                grid_dim: (blocks, 1, 1),
                shared_mem_bytes: 0,
            };

            let bp_output_kernel = self.Dev.get_func("mlp_kernels", "BackPropOutputKernel").unwrap();
            let is_softmax = self.OutputActivation == TActivationType::atSoftmax;

            unsafe {
                bp_output_kernel.clone().launch(cfg, (
                    &output_layer.Errors,
                    &output_layer.Outputs,
                    &self.d_Target,
                    output_layer.NumNeurons,
                    output_layer.ActivationType as i32,
                    is_softmax as i32,
                ))?;
            }
        }

        self.Dev.synchronize()?;

        // Hidden layer errors
        let bp_hidden_kernel = self.Dev.get_func("mlp_kernels", "BackPropHiddenKernel").unwrap();

        for k in (1..(num_layers - 1)).rev() {
            let layer = &self.Layers[k];
            let next_layer = &self.Layers[k + 1];
            let blocks = Self::GetBlocks(layer.NumNeurons);
            let cfg = LaunchConfig {
                block_dim: (BLOCK_SIZE, 1, 1),
                grid_dim: (blocks, 1, 1),
                shared_mem_bytes: 0,
            };

            unsafe {
                bp_hidden_kernel.clone().launch(cfg, (
                    &layer.Errors,
                    &layer.Outputs,
                    &layer.DropoutMask,
                    &next_layer.Errors,
                    &next_layer.Weights,
                    layer.NumNeurons,
                    next_layer.NumNeurons,
                    next_layer.NumInputs,
                    layer.ActivationType as i32,
                ))?;
            }

            self.Dev.synchronize()?;
        }

        Ok(())
    }

    pub fn UpdateWeights(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        self.Timestep += 1;
        let num_layers = self.NumLayers as usize;
        let clip_val = 5.0;

        for k in (1..num_layers).rev() {
            let layer = &self.Layers[k];
            let prev_layer = &self.Layers[k - 1];
            let blocks = Self::GetBlocks(layer.NumNeurons);
            let cfg = LaunchConfig {
                block_dim: (BLOCK_SIZE, 1, 1),
                grid_dim: (blocks, 1, 1),
                shared_mem_bytes: 0,
            };

            match self.Optimizer {
                TOptimizerType::otSGD => {
                    let kernel = self.Dev.get_func("mlp_kernels", "UpdateWeightsSGDKernel").unwrap();
                    unsafe {
                        kernel.clone().launch(cfg, (
                            &layer.Weights,
                            &layer.Biases,
                            &layer.Errors,
                            &prev_layer.Outputs,
                            layer.NumNeurons,
                            layer.NumInputs,
                            prev_layer.NumNeurons,
                            self.LearningRate,
                            self.L2Lambda,
                            clip_val,
                        ))?;
                    }
                }
                TOptimizerType::otAdam => {
                    let kernel = self.Dev.get_func("mlp_kernels", "UpdateWeightsAdamKernel").unwrap();
                    let adam_params = vec![
                        self.LearningRate,
                        self.L2Lambda,
                        self.Beta1,
                        self.Beta2,
                        self.Timestep as f64,
                        clip_val,
                    ];
                    self.Dev.htod_sync_copy_into(&adam_params, &mut self.d_AdamParams)?;
                    unsafe {
                        kernel.clone().launch(cfg, (
                            &layer.Weights,
                            &layer.Biases,
                            &layer.Errors,
                            &prev_layer.Outputs,
                            &layer.M,
                            &layer.V,
                            &layer.MBias,
                            &layer.VBias,
                            layer.NumNeurons,
                            layer.NumInputs,
                            prev_layer.NumNeurons,
                            &self.d_AdamParams,
                        ))?;
                    }
                }
                TOptimizerType::otRMSProp => {
                    let kernel = self.Dev.get_func("mlp_kernels", "UpdateWeightsRMSPropKernel").unwrap();
                    unsafe {
                        kernel.clone().launch(cfg, (
                            &layer.Weights,
                            &layer.Biases,
                            &layer.Errors,
                            &prev_layer.Outputs,
                            &layer.V,
                            &layer.VBias,
                            layer.NumNeurons,
                            layer.NumInputs,
                            prev_layer.NumNeurons,
                            self.LearningRate,
                            self.L2Lambda,
                            clip_val,
                        ))?;
                    }
                }
            }
        }

        self.Dev.synchronize()?;
        Ok(())
    }

    pub fn Predict(&mut self, input: &Darray) -> Result<Darray, Box<dyn std::error::Error>> {
        self.FIsTraining = false;

        // Set input layer outputs
        let input_size = self.FInputSize as usize;
        let mut h_input = vec![0.0f64; input_size + 1];
        for i in 0..input_size {
            h_input[i] = input[i];
        }
        h_input[input_size] = 1.0; // bias

        self.Dev.htod_sync_copy_into(&h_input, &mut self.Layers[0].Outputs)?;

        self.FeedForward()?;

        let num_layers = self.NumLayers as usize;
        let result = self.Dev.dtoh_sync_copy(&self.Layers[num_layers - 1].Outputs)?;

        self.FIsTraining = true;
        Ok(result)
    }

    pub fn Train(&mut self, input: &Darray, target: &Darray) -> Result<(), Box<dyn std::error::Error>> {
        self.FIsTraining = true;

        // Set input layer outputs
        let input_size = self.FInputSize as usize;
        let mut h_input = vec![0.0f64; input_size + 1];
        for i in 0..input_size {
            h_input[i] = input[i];
        }
        h_input[input_size] = 1.0; // bias

        self.Dev.htod_sync_copy_into(&h_input, &mut self.Layers[0].Outputs)?;
        self.Dev.htod_sync_copy_into(target, &mut self.d_Target)?;

        self.FeedForward()?;
        self.BackPropagate()?;
        self.UpdateWeights()?;

        Ok(())
    }

    pub fn ComputeLoss(&self, predicted: &Darray, target: &Darray) -> f64 {
        let mut loss = 0.0;
        for i in 0..predicted.len() {
            let diff = predicted[i] - target[i];
            loss += diff * diff;
        }
        loss / predicted.len() as f64
    }

    pub fn GetOutputSize(&self) -> i32 {
        self.FOutputSize
    }

    pub fn GetInputSize(&self) -> i32 {
        self.FInputSize
    }

    pub fn GetHiddenLayerCount(&self) -> usize {
        self.FHiddenSizes.len()
    }

    pub fn GetHiddenSizes(&self) -> &Vec<i32> {
        &self.FHiddenSizes
    }

    pub fn GetNumLayers(&self) -> i32 {
        self.NumLayers
    }

    pub fn GetLayerSize(&self, layer_idx: usize) -> i32 {
        if layer_idx >= self.Layers.len() {
            return 0;
        }
        self.Layers[layer_idx].NumNeurons
    }

    fn ApplyBatchNorm(&mut self, layer_idx: usize) -> Result<(), Box<dyn std::error::Error>> {
        let layer = &self.Layers[layer_idx];
        let num_neurons = layer.NumNeurons;
        let blocks = Self::GetBlocks(num_neurons);
        let cfg = LaunchConfig {
            block_dim: (BLOCK_SIZE, 1, 1),
            grid_dim: (blocks, 1, 1),
            shared_mem_bytes: 0,
        };

        if self.FIsTraining {
            let compute_kernel = self.Dev.get_func("mlp_kernels", "ComputeBatchMeanVarKernel").unwrap();
            unsafe {
                compute_kernel.clone().launch(cfg, (
                    &layer.d_BatchMean,
                    &layer.d_BatchVar,
                    &layer.Outputs,
                    num_neurons,
                ))?;
            }
            self.Dev.synchronize()?;

            let bn_kernel = self.Dev.get_func("mlp_kernels", "BatchNormForwardTrainKernel").unwrap();
            unsafe {
                bn_kernel.clone().launch(cfg, (
                    &layer.Outputs,
                    &layer.Outputs,
                    &layer.d_Gamma,
                    &layer.d_Beta,
                    &layer.d_RunningMean,
                    &layer.d_RunningVar,
                    &layer.d_BatchMean,
                    &layer.d_BatchVar,
                    num_neurons,
                    self.BatchNormMomentum,
                    self.BatchNormEpsilon,
                ))?;
            }
        } else {
            let bn_kernel = self.Dev.get_func("mlp_kernels", "BatchNormForwardInferKernel").unwrap();
            unsafe {
                bn_kernel.clone().launch(cfg, (
                    &layer.Outputs,
                    &layer.Outputs,
                    &layer.d_Gamma,
                    &layer.d_Beta,
                    &layer.d_RunningMean,
                    &layer.d_RunningVar,
                    num_neurons,
                    self.BatchNormEpsilon,
                ))?;
            }
        }
        self.Dev.synchronize()?;
        Ok(())
    }

    pub fn export_to_onnx(&self, filename: &str) -> io::Result<()> {
        let mut file = File::create(filename)?;

        fn write_varint(buf: &mut Vec<u8>, mut val: u64) {
            while val >= 0x80 {
                buf.push((val as u8) | 0x80);
                val >>= 7;
            }
            buf.push(val as u8);
        }

        fn write_field(buf: &mut Vec<u8>, field_num: u32, wire_type: u8, data: &[u8]) {
            let tag = (field_num << 3) | (wire_type as u32);
            write_varint(buf, tag as u64);
            if wire_type == 2 {
                write_varint(buf, data.len() as u64);
            }
            buf.extend_from_slice(data);
        }

        fn write_string(buf: &mut Vec<u8>, field_num: u32, s: &str) {
            write_field(buf, field_num, 2, s.as_bytes());
        }

        fn write_int64(buf: &mut Vec<u8>, field_num: u32, val: i64) {
            let mut tmp = Vec::new();
            write_varint(&mut tmp, val as u64);
            let tag = (field_num << 3) | 0;
            write_varint(buf, tag as u64);
            buf.extend_from_slice(&tmp);
        }

        fn write_float_array(buf: &mut Vec<u8>, field_num: u32, vals: &[f32]) {
            let mut data = Vec::new();
            for &v in vals {
                data.extend_from_slice(&v.to_le_bytes());
            }
            write_field(buf, field_num, 2, &data);
        }

        fn build_tensor_proto(name: &str, dims: &[i64], data: &[f32]) -> Vec<u8> {
            let mut tensor = Vec::new();
            for &d in dims {
                write_int64(&mut tensor, 1, d);
            }
            write_int64(&mut tensor, 2, 1);
            write_float_array(&mut tensor, 4, data);
            write_string(&mut tensor, 8, name);
            tensor
        }

        fn build_node_proto(op_type: &str, inputs: &[&str], outputs: &[&str], name: &str) -> Vec<u8> {
            let mut node = Vec::new();
            for inp in inputs {
                write_string(&mut node, 1, inp);
            }
            for out in outputs {
                write_string(&mut node, 2, out);
            }
            write_string(&mut node, 3, name);
            write_string(&mut node, 4, op_type);
            node
        }

        fn build_value_info(name: &str, dims: &[i64]) -> Vec<u8> {
            let mut shape = Vec::new();
            for &d in dims {
                let mut dim = Vec::new();
                write_int64(&mut dim, 1, d);
                write_field(&mut shape, 1, 2, &dim);
            }

            let mut tensor_type = Vec::new();
            write_int64(&mut tensor_type, 1, 1);
            write_field(&mut tensor_type, 2, 2, &shape);

            let mut type_proto = Vec::new();
            write_field(&mut type_proto, 1, 2, &tensor_type);

            let mut vi = Vec::new();
            write_string(&mut vi, 1, name);
            write_field(&mut vi, 2, 2, &type_proto);
            vi
        }

        let mut initializers: Vec<Vec<u8>> = Vec::new();
        let mut nodes: Vec<Vec<u8>> = Vec::new();
        let mut prev_output = "input".to_string();

        for h in 0..self.FHiddenSizes.len() {
            let layer = &self.Layers[h + 1];
            let num_neurons = self.FHiddenSizes[h];
            let num_inputs = if h == 0 { self.FInputSize } else { self.FHiddenSizes[h - 1] };

            let h_weights = self.Dev.dtoh_sync_copy(&layer.Weights).unwrap();
            let h_biases = self.Dev.dtoh_sync_copy(&layer.Biases).unwrap();

            let mut weights_f32: Vec<f32> = Vec::new();
            for j in 0..num_neurons as usize {
                for w in 0..num_inputs as usize {
                    weights_f32.push(h_weights[j * layer.NumInputs as usize + w] as f32);
                }
            }
            let biases_f32: Vec<f32> = h_biases[..num_neurons as usize].iter().map(|&x| x as f32).collect();

            let w_name = format!("hidden{}_weights", h);
            let b_name = format!("hidden{}_biases", h);
            let gemm_out = format!("hidden{}_gemm", h);
            let act_out = format!("hidden{}_out", h);

            initializers.push(build_tensor_proto(&w_name, &[num_neurons as i64, num_inputs as i64], &weights_f32));
            initializers.push(build_tensor_proto(&b_name, &[num_neurons as i64], &biases_f32));

            nodes.push(build_node_proto("Gemm", &[&prev_output, &w_name, &b_name], &[&gemm_out], &format!("gemm_{}", h)));

            if self.BatchNormEnabled {
                let h_gamma = self.Dev.dtoh_sync_copy(&layer.d_Gamma).unwrap();
                let h_beta = self.Dev.dtoh_sync_copy(&layer.d_Beta).unwrap();
                let h_mean = self.Dev.dtoh_sync_copy(&layer.d_RunningMean).unwrap();
                let h_var = self.Dev.dtoh_sync_copy(&layer.d_RunningVar).unwrap();

                let gamma_f32: Vec<f32> = h_gamma[..num_neurons as usize].iter().map(|&x| x as f32).collect();
                let beta_f32: Vec<f32> = h_beta[..num_neurons as usize].iter().map(|&x| x as f32).collect();
                let mean_f32: Vec<f32> = h_mean[..num_neurons as usize].iter().map(|&x| x as f32).collect();
                let var_f32: Vec<f32> = h_var[..num_neurons as usize].iter().map(|&x| x as f32).collect();

                let bn_scale = format!("bn{}_scale", h);
                let bn_bias = format!("bn{}_bias", h);
                let bn_mean = format!("bn{}_mean", h);
                let bn_var = format!("bn{}_var", h);
                let bn_out = format!("bn{}_out", h);

                initializers.push(build_tensor_proto(&bn_scale, &[num_neurons as i64], &gamma_f32));
                initializers.push(build_tensor_proto(&bn_bias, &[num_neurons as i64], &beta_f32));
                initializers.push(build_tensor_proto(&bn_mean, &[num_neurons as i64], &mean_f32));
                initializers.push(build_tensor_proto(&bn_var, &[num_neurons as i64], &var_f32));

                nodes.push(build_node_proto("BatchNormalization", 
                    &[&gemm_out, &bn_scale, &bn_bias, &bn_mean, &bn_var], 
                    &[&bn_out], 
                    &format!("bn_{}", h)));

                let act_type = match self.HiddenActivation {
                    TActivationType::atReLU => "Relu",
                    TActivationType::atTanh => "Tanh",
                    TActivationType::atSigmoid => "Sigmoid",
                    _ => "Relu",
                };
                nodes.push(build_node_proto(act_type, &[&bn_out], &[&act_out], &format!("act_{}", h)));
            } else {
                let act_type = match self.HiddenActivation {
                    TActivationType::atReLU => "Relu",
                    TActivationType::atTanh => "Tanh",
                    TActivationType::atSigmoid => "Sigmoid",
                    _ => "Relu",
                };
                nodes.push(build_node_proto(act_type, &[&gemm_out], &[&act_out], &format!("act_{}", h)));
            }

            prev_output = act_out;
        }

        let num_layers = self.NumLayers as usize;
        let out_layer = &self.Layers[num_layers - 1];
        let out_num_inputs = if self.FHiddenSizes.is_empty() { self.FInputSize } else { *self.FHiddenSizes.last().unwrap() };

        let h_weights = self.Dev.dtoh_sync_copy(&out_layer.Weights).unwrap();
        let h_biases = self.Dev.dtoh_sync_copy(&out_layer.Biases).unwrap();

        let mut weights_f32: Vec<f32> = Vec::new();
        for i in 0..self.FOutputSize as usize {
            for w in 0..out_num_inputs as usize {
                weights_f32.push(h_weights[i * out_layer.NumInputs as usize + w] as f32);
            }
        }
        let biases_f32: Vec<f32> = h_biases[..self.FOutputSize as usize].iter().map(|&x| x as f32).collect();

        initializers.push(build_tensor_proto("output_weights", &[self.FOutputSize as i64, out_num_inputs as i64], &weights_f32));
        initializers.push(build_tensor_proto("output_biases", &[self.FOutputSize as i64], &biases_f32));

        nodes.push(build_node_proto("Gemm", &[&prev_output, "output_weights", "output_biases"], &["output_gemm"], "gemm_output"));

        let final_output = if self.OutputActivation == TActivationType::atSoftmax {
            nodes.push(build_node_proto("Softmax", &["output_gemm"], &["output"], "softmax_output"));
            "output"
        } else {
            let act_type = match self.OutputActivation {
                TActivationType::atReLU => "Relu",
                TActivationType::atTanh => "Tanh",
                TActivationType::atSigmoid => "Sigmoid",
                TActivationType::atLinear => { "output_gemm" }
                _ => "Sigmoid",
            };
            if act_type != "output_gemm" {
                nodes.push(build_node_proto(act_type, &["output_gemm"], &["output"], "act_output"));
                "output"
            } else {
                "output_gemm"
            }
        };

        let input_info = build_value_info("input", &[1, self.FInputSize as i64]);
        let output_info = build_value_info(final_output, &[1, self.FOutputSize as i64]);

        let mut graph = Vec::new();
        for node in &nodes {
            write_field(&mut graph, 1, 2, node);
        }
        write_string(&mut graph, 2, "mlp_graph");
        for init in &initializers {
            write_field(&mut graph, 5, 2, init);
        }
        write_field(&mut graph, 11, 2, &input_info);
        write_field(&mut graph, 12, 2, &output_info);

        let mut model = Vec::new();
        write_int64(&mut model, 1, 7);
        write_string(&mut model, 2, "rust_cuda_mlp");
        write_string(&mut model, 3, "1.0");
        write_string(&mut model, 4, "ai.onnx");
        write_int64(&mut model, 5, 13);
        write_field(&mut model, 7, 2, &graph);

        file.write_all(&model)?;
        Ok(())
    }

    pub fn feature_importance(&self) -> Vec<(usize, f64)> {
        if self.Layers.len() < 2 {
            return Vec::new();
        }

        let first_hidden = &self.Layers[1];
        let h_weights = self.Dev.dtoh_sync_copy(&first_hidden.Weights).unwrap();
        
        let num_neurons = first_hidden.NumNeurons as usize;
        let num_inputs = self.FInputSize as usize;
        
        let mut importance: Vec<(usize, f64)> = (0..num_inputs)
            .map(|input_idx| {
                let mut sum = 0.0;
                for neuron_idx in 0..num_neurons {
                    let weight_idx = neuron_idx * first_hidden.NumInputs as usize + input_idx;
                    if weight_idx < h_weights.len() {
                        sum += h_weights[weight_idx].abs();
                    }
                }
                (input_idx, sum)
            })
            .collect();
        
        importance.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        importance
    }

    pub fn SaveModelToJSON(&self, filename: &str) -> io::Result<()> {
        let mut hidden_layers_json = Vec::new();

        for h in 0..self.FHiddenSizes.len() {
            let layer = &self.Layers[h + 1];
            let num_neurons = self.FHiddenSizes[h];
            let num_inputs = if h == 0 {
                self.FInputSize
            } else {
                self.FHiddenSizes[h - 1]
            };

            let h_weights = self.Dev.dtoh_sync_copy(&layer.Weights).unwrap();
            let h_biases = self.Dev.dtoh_sync_copy(&layer.Biases).unwrap();

            let mut neurons = Vec::new();
            for j in 0..num_neurons as usize {
                let mut weights = Vec::new();
                for w in 0..num_inputs as usize {
                    weights.push(h_weights[j * layer.NumInputs as usize + w]);
                }
                neurons.push(NeuronJSON {
                    weights,
                    bias: h_biases[j],
                });
            }

            let biases: Vec<f64> = (0..num_neurons as usize).map(|j| h_biases[j]).collect();

            hidden_layers_json.push(HiddenLayerJSON {
                neuron_count: num_neurons,
                neurons,
                biases,
            });
        }

        let num_layers = self.NumLayers as usize;
        let out_layer = &self.Layers[num_layers - 1];
        let out_num_inputs = if self.FHiddenSizes.is_empty() {
            self.FInputSize
        } else {
            *self.FHiddenSizes.last().unwrap()
        };

        let h_weights = self.Dev.dtoh_sync_copy(&out_layer.Weights).unwrap();
        let h_biases = self.Dev.dtoh_sync_copy(&out_layer.Biases).unwrap();

        let mut out_neurons = Vec::new();
        for i in 0..self.FOutputSize as usize {
            let mut weights = Vec::new();
            for w in 0..out_num_inputs as usize {
                weights.push(h_weights[i * out_layer.NumInputs as usize + w]);
            }
            out_neurons.push(NeuronJSON {
                weights,
                bias: h_biases[i],
            });
        }

        let out_biases: Vec<f64> = (0..self.FOutputSize as usize)
            .map(|i| h_biases[i])
            .collect();

        let mut batch_norm_params = Vec::new();
        if self.BatchNormEnabled {
            for h in 0..self.FHiddenSizes.len() {
                let layer = &self.Layers[h + 1];
                let num_neurons = self.FHiddenSizes[h] as usize;
                let gamma = self.Dev.dtoh_sync_copy(&layer.d_Gamma).unwrap();
                let beta = self.Dev.dtoh_sync_copy(&layer.d_Beta).unwrap();
                let running_mean = self.Dev.dtoh_sync_copy(&layer.d_RunningMean).unwrap();
                let running_var = self.Dev.dtoh_sync_copy(&layer.d_RunningVar).unwrap();
                batch_norm_params.push(BatchNormParamsJSON {
                    gamma: gamma[..num_neurons].to_vec(),
                    beta: beta[..num_neurons].to_vec(),
                    running_mean: running_mean[..num_neurons].to_vec(),
                    running_var: running_var[..num_neurons].to_vec(),
                });
            }
        }

        let model = ModelJSON {
            magic: MODEL_MAGIC.to_string(),
            input_size: self.FInputSize,
            output_size: self.FOutputSize,
            hidden_sizes: self.FHiddenSizes.clone(),
            learning_rate: self.LearningRate,
            optimizer: self.Optimizer as i32,
            hidden_activation: self.HiddenActivation as i32,
            output_activation: self.OutputActivation as i32,
            dropout_rate: self.DropoutRate,
            l2_lambda: self.L2Lambda,
            beta1: self.Beta1,
            beta2: self.Beta2,
            batch_norm: self.BatchNormEnabled,
            input_layer: InputLayerJSON {
                neuron_count: self.FInputSize,
            },
            hidden_layers: hidden_layers_json,
            output_layer: OutputLayerJSON {
                neuron_count: self.FOutputSize,
                neurons: out_neurons,
                biases: out_biases,
            },
            batch_norm_params,
        };

        let json = serde_json::to_string_pretty(&model).map_err(|e| {
            io::Error::new(io::ErrorKind::Other, e.to_string())
        })?;

        let mut file = File::create(filename)?;
        file.write_all(json.as_bytes())?;
        Ok(())
    }

    pub fn LoadModelFromJSON(&mut self, filename: &str) -> Result<(), Box<dyn std::error::Error>> {
        let file = File::open(filename)?;
        let reader = BufReader::new(file);
        let model: ModelJSON = serde_json::from_reader(reader)?;

        // Recreate the MLP with new dimensions
        let new_hidden_sizes: Vec<i32> = model.hidden_sizes;
        let new_hidden_act = match model.hidden_activation {
            1 => TActivationType::atTanh,
            2 => TActivationType::atReLU,
            3 => TActivationType::atSoftmax,
            4 => TActivationType::atLinear,
            _ => TActivationType::atSigmoid,
        };
        let new_output_act = match model.output_activation {
            1 => TActivationType::atTanh,
            2 => TActivationType::atReLU,
            3 => TActivationType::atSoftmax,
            4 => TActivationType::atLinear,
            _ => TActivationType::atSigmoid,
        };

        self.FInputSize = model.input_size;
        self.FOutputSize = model.output_size;
        self.FHiddenSizes = new_hidden_sizes.clone();
        self.LearningRate = model.learning_rate;
        self.Optimizer = match model.optimizer {
            1 => TOptimizerType::otAdam,
            2 => TOptimizerType::otRMSProp,
            _ => TOptimizerType::otSGD,
        };
        self.HiddenActivation = new_hidden_act;
        self.OutputActivation = new_output_act;
        self.DropoutRate = model.dropout_rate;
        self.L2Lambda = model.l2_lambda;
        self.Beta1 = model.beta1;
        self.Beta2 = model.beta2;
        self.BatchNormEnabled = model.batch_norm;

        self.NumLayers = (new_hidden_sizes.len() as i32) + 2;
        self.Layers.clear();

        // Input layer
        self.Layers.push(Self::AllocateLayerStatic(
            &self.Dev,
            self.FInputSize + 1,
            self.FInputSize,
            TActivationType::atSigmoid,
        )?);

        self.MaxNeurons = self.FInputSize + 1;

        // Hidden layers
        let mut num_inputs = self.FInputSize;
        for (h, &hidden_size) in new_hidden_sizes.iter().enumerate() {
            let mut layer = Self::AllocateLayerStatic(
                &self.Dev,
                hidden_size + 1,
                num_inputs + 1,
                new_hidden_act,
            )?;

            if hidden_size + 1 > self.MaxNeurons {
                self.MaxNeurons = hidden_size + 1;
            }

            if h < model.hidden_layers.len() {
                let hl = &model.hidden_layers[h];
                let layer_inputs = if h == 0 {
                    self.FInputSize
                } else {
                    new_hidden_sizes[h - 1]
                };

                let mut h_weights = vec![0.0f64; (layer.NumNeurons * layer.NumInputs) as usize];
                let mut h_biases = vec![0.0f64; layer.NumNeurons as usize];

                for (n, neuron) in hl.neurons.iter().enumerate() {
                    for (w, &weight) in neuron.weights.iter().enumerate() {
                        if w < layer_inputs as usize {
                            h_weights[n * layer.NumInputs as usize + w] = weight;
                        }
                    }
                    h_biases[n] = neuron.bias;
                }

                self.Dev.htod_sync_copy_into(&h_weights, &mut layer.Weights)?;
                self.Dev.htod_sync_copy_into(&h_biases, &mut layer.Biases)?;
            }

            if self.BatchNormEnabled && h < model.batch_norm_params.len() {
                let bn = &model.batch_norm_params[h];
                let num_neurons = hidden_size as usize;
                let mut gamma = vec![1.0f64; layer.NumNeurons as usize];
                let mut beta = vec![0.0f64; layer.NumNeurons as usize];
                let mut running_mean = vec![0.0f64; layer.NumNeurons as usize];
                let mut running_var = vec![1.0f64; layer.NumNeurons as usize];
                for i in 0..num_neurons.min(bn.gamma.len()) {
                    gamma[i] = bn.gamma[i];
                    beta[i] = bn.beta[i];
                    running_mean[i] = bn.running_mean[i];
                    running_var[i] = bn.running_var[i];
                }
                self.Dev.htod_sync_copy_into(&gamma, &mut layer.d_Gamma)?;
                self.Dev.htod_sync_copy_into(&beta, &mut layer.d_Beta)?;
                self.Dev.htod_sync_copy_into(&running_mean, &mut layer.d_RunningMean)?;
                self.Dev.htod_sync_copy_into(&running_var, &mut layer.d_RunningVar)?;
            }

            self.Layers.push(layer);
            num_inputs = hidden_size;
        }

        // Output layer
        let mut out_layer = Self::AllocateLayerStatic(
            &self.Dev,
            self.FOutputSize,
            num_inputs + 1,
            new_output_act,
        )?;

        if self.FOutputSize > self.MaxNeurons {
            self.MaxNeurons = self.FOutputSize;
        }

        let out_num_inputs = if new_hidden_sizes.is_empty() {
            self.FInputSize
        } else {
            *new_hidden_sizes.last().unwrap()
        };

        let mut h_weights = vec![0.0f64; (out_layer.NumNeurons * out_layer.NumInputs) as usize];
        let mut h_biases = vec![0.0f64; out_layer.NumNeurons as usize];

        for (n, neuron) in model.output_layer.neurons.iter().enumerate() {
            for (w, &weight) in neuron.weights.iter().enumerate() {
                if w < out_num_inputs as usize {
                    h_weights[n * out_layer.NumInputs as usize + w] = weight;
                }
            }
            h_biases[n] = neuron.bias;
        }

        self.Dev.htod_sync_copy_into(&h_weights, &mut out_layer.Weights)?;
        self.Dev.htod_sync_copy_into(&h_biases, &mut out_layer.Biases)?;

        self.Layers.push(out_layer);

        // Reallocate target and softmax buffers
        self.d_Target = self.Dev.alloc_zeros::<f64>(self.FOutputSize as usize)?;
        self.d_SoftmaxSums = self.Dev.alloc_zeros::<f64>(self.FOutputSize as usize)?;
        self.d_AdamParams = self.Dev.alloc_zeros::<f64>(6)?;

        Ok(())
    }

    pub fn Save(&self, filename: &str) -> io::Result<()> {
        self.SaveModelToJSON(filename)
    }

    pub fn Load(filename: &str) -> Result<Self, Box<dyn std::error::Error>> {
        let mut mlp = TMultiLayerPerceptronCUDA::new(
            1,
            &[1],
            1,
            TActivationType::atSigmoid,
            TActivationType::atSigmoid,
        )?;
        mlp.LoadModelFromJSON(filename)?;
        Ok(mlp)
    }
}

fn ShuffleData(data: &mut TDataPointArray) {
    let mut rng = rand::thread_rng();
    for i in (1..data.len()).rev() {
        let j = rng.gen_range(0..=i);
        data.swap(i, j);
    }
}

fn NormalizeData(data: &mut TDataPointArray) {
    if data.is_empty() {
        return;
    }
    let input_size = data[0].Input.len();

    let mut mins: Vec<f64> = data[0].Input.clone();
    let mut maxs: Vec<f64> = data[0].Input.clone();

    for dp in data.iter() {
        for j in 0..input_size {
            if dp.Input[j] < mins[j] {
                mins[j] = dp.Input[j];
            }
            if dp.Input[j] > maxs[j] {
                maxs[j] = dp.Input[j];
            }
        }
    }

    for dp in data.iter_mut() {
        for j in 0..input_size {
            let range = maxs[j] - mins[j];
            dp.Input[j] = if range > 0.0 {
                (dp.Input[j] - mins[j]) / range
            } else {
                0.5
            };
        }
    }
}

fn LoadDataCSV(filename: &str, input_size: i32, output_size: i32) -> TDataPointArray {
    let mut data = TDataPointArray::new();
    let file = match File::open(filename) {
        Ok(f) => f,
        Err(_) => return data,
    };

    let reader = BufReader::new(file);
    for line in reader.lines() {
        let line = match line {
            Ok(l) => l,
            Err(_) => continue,
        };
        if line.is_empty() {
            continue;
        }

        let mut values = TDoubleArray::new();
        ParseDoubleArrayHelper(&line, &mut values);

        if (values.len() as i32) < input_size + output_size {
            continue;
        }

        let mut dp = TDataPoint {
            Input: vec![0.0; input_size as usize],
            Target: vec![0.0; output_size as usize],
        };

        for i in 0..input_size as usize {
            dp.Input[i] = values[i];
        }
        for i in 0..output_size as usize {
            dp.Target[i] = values[input_size as usize + i];
        }
        data.push(dp);
    }
    data
}

fn PrintUsage() {
    println!("MLP - Multi-Layer Perceptron (CUDA/Rust)");
    println!();
    println!("Usage: mlp_cuda <command> [options]");
    println!();
    println!("Commands:");
    println!("  create       Create a new MLP model");
    println!("  train        Train an existing model");
    println!("  predict      Make predictions with a model");
    println!("  info         Display model information");
    println!("  export-onnx  Export model to ONNX format");
    println!("  feature-importance  Calculate feature importance");
    println!("  help         Show this help message");
    println!();
    println!("Create Options:");
    println!("  -i, --input=N              Input layer size (required)");
    println!("  -H, --hidden=N,N,...       Hidden layer sizes, comma-separated (required)");
    println!("  -o, --output=N             Output layer size (required)");
    println!("  -s, --save=FILE            Save model file (required, .json)");
    println!("  --lr=VALUE                 Learning rate (default: 0.1)");
    println!("  --optimizer=TYPE           sgd|adam|rmsprop (default: sgd)");
    println!("  --hidden-act=TYPE          sigmoid|tanh|relu|softmax (default: sigmoid)");
    println!("  --output-act=TYPE          sigmoid|tanh|relu|softmax (default: sigmoid)");
    println!("  --dropout=VALUE            Dropout rate 0-1 (default: 0)");
    println!("  --l2=VALUE                 L2 regularization lambda (default: 0)");
    println!("  --beta1=VALUE              Adam beta1 parameter (default: 0.9)");
    println!("  --beta2=VALUE              Adam beta2 parameter (default: 0.999)");
    println!("  --batch-norm               Enable batch normalization");
    println!();
    println!("Train Options:");
    println!("  -m, --model=FILE           Load model file (required, .json)");
    println!("  -d, --data=FILE            Training data CSV file (required)");
    println!("  -s, --save=FILE            Save trained model (required, .json)");
    println!("  --epochs=N                 Training epochs (default: 100)");
    println!("  --batch=N                  Batch size (default: 1)");
    println!("  --lr=VALUE                 Override learning rate");
    println!("  --lr-decay                 Enable learning rate decay");
    println!("  --lr-decay-rate=VALUE      LR decay rate (default: 0.95)");
    println!("  --lr-decay-epochs=N        Decay interval in epochs (default: 10)");
    println!("  --early-stop               Enable early stopping");
    println!("  --patience=N               Early stopping patience (default: 10)");
    println!("  --normalize                Normalize training data");
    println!("  --verbose                  Print training progress");
    println!();
    println!("Predict Options:");
    println!("  -m, --model=FILE           Model file (required, .json)");
    println!("  -i, --input=v1,v2,...      Input values, comma-separated (required)");
    println!();
    println!("Info Options:");
    println!("  -m, --model=FILE           Model file (required, .json)");
    println!();
    println!("Export ONNX Options:");
    println!("  -m, --model=FILE           Model file (required, .json)");
    println!("  -s, --save=FILE            Output ONNX file (required, .onnx)");
    println!();
    println!("Feature Importance Options:");
    println!("  -m, --model=FILE           Model file (required, .json)");
    println!();
    println!("Examples:");
    println!("  mlp_cuda create -i 2 -H 8 -o 1 -s xor.json");
    println!("  mlp_cuda create --input=2 --hidden=8,8 --output=1 --save=xor.json");
    println!("  mlp_cuda train -m xor.json -d data.csv -s xor_trained.json --epochs=1000");
    println!("  mlp_cuda train --model=xor.json --data=data.csv --epochs=1000 --save=xor_trained.json --verbose");
    println!("  mlp_cuda predict -m xor_trained.json -i 1,0");
    println!("  mlp_cuda info -m xor_trained.json");
    println!();
    println!("Exit codes:");
    println!("  0 - Success");
    println!("  1 - Error");
    println!("  2 - Usage error");
}

fn main() {
    let args: Vec<String> = env::args().collect();

    if args.len() < 2 {
        PrintUsage();
        process::exit(2);
    }

    let cmd_str = &args[1];
    let command = match cmd_str.as_str() {
        "create" => TCommand::CmdCreate,
        "train" => TCommand::CmdTrain,
        "predict" => TCommand::CmdPredict,
        "info" => TCommand::CmdInfo,
        "export-onnx" => TCommand::CmdExportONNX,
        "feature-importance" => TCommand::CmdFeatureImportance,
        "help" | "--help" | "-h" => TCommand::CmdHelp,
        _ => {
            eprintln!("Error: Unknown command: {}", cmd_str);
            PrintUsage();
            process::exit(2);
        }
    };

    if command == TCommand::CmdHelp {
        PrintUsage();
        process::exit(0);
    }

    let mut input_size: i32 = 0;
    let mut output_size: i32 = 0;
    let mut hidden_sizes: TIntArray = Vec::new();
    let mut learning_rate: f64 = 0.1;
    let mut dropout_rate: f64 = 0.0;
    let mut l2_lambda: f64 = 0.0;
    let mut beta1: f64 = 0.9;
    let mut beta2: f64 = 0.999;
    let mut epochs: i32 = 100;
    let mut batch_size: i32 = 1;
    let mut lr_decay: bool = false;
    let mut lr_decay_rate: f64 = 0.95;
    let mut lr_decay_epochs: i32 = 10;
    let mut early_stop: bool = false;
    let mut patience: i32 = 10;
    let mut normalize: bool = false;
    let mut verbose: bool = false;
    let mut hidden_act = TActivationType::atSigmoid;
    let mut output_act = TActivationType::atSigmoid;
    let mut optimizer = TOptimizerType::otSGD;
    let mut model_file = String::new();
    let mut save_file = String::new();
    let mut data_file = String::new();
    let mut input_values: TDoubleArray = Vec::new();
    let mut batch_norm: bool = false;

    let mut i = 2;
    while i < args.len() {
        let arg = &args[i];
        let mut key = String::new();
        let mut value = String::new();

        if let Some(eq_pos) = arg.find('=') {
            key = arg[..eq_pos].to_string();
            value = arg[eq_pos + 1..].to_string();
            i += 1;
        } else if arg.starts_with('-') {
            if arg == "--lr-decay" {
                lr_decay = true;
                i += 1;
                continue;
            } else if arg == "--early-stop" {
                early_stop = true;
                i += 1;
                continue;
            } else if arg == "--normalize" {
                normalize = true;
                i += 1;
                continue;
            } else if arg == "--verbose" {
                verbose = true;
                i += 1;
                continue;
            } else if arg == "--batch-norm" {
                batch_norm = true;
                i += 1;
                continue;
            } else if arg == "-h" {
                PrintUsage();
                process::exit(0);
            }

            key = arg.clone();
            if i + 1 < args.len() {
                i += 1;
                value = args[i].clone();
                if value.starts_with('-') && !value.chars().nth(1).map_or(false, |c| c.is_numeric()) {
                    i -= 1;
                    value = String::new();
                } else {
                    i += 1;
                }
            } else {
                eprintln!("Error: Option {} requires a value", key);
                process::exit(2);
            }
        } else {
            eprintln!("Error: Invalid argument: {}", arg);
            process::exit(2);
        }

        match key.as_str() {
            "--input" | "-i" => {
                if command == TCommand::CmdPredict {
                    if value == "-" {
                        let stdin = io::stdin();
                        for line in stdin.lock().lines() {
                            if let Ok(l) = line {
                                ParseDoubleArrayHelper(&l, &mut input_values);
                            }
                        }
                    } else {
                        ParseDoubleArrayHelper(&value, &mut input_values);
                    }
                } else {
                    input_size = value.parse().unwrap_or(0);
                }
            }
            "--hidden" | "-H" => {
                ParseIntArrayHelper(&value, &mut hidden_sizes);
            }
            "--output" | "-o" => {
                output_size = value.parse().unwrap_or(0);
            }
            "--save" | "-s" => {
                save_file = value;
            }
            "--model" | "-m" => {
                model_file = value;
            }
            "--data" | "-d" => {
                data_file = value;
            }
            "--lr" => {
                learning_rate = value.parse().unwrap_or(0.1);
            }
            "--optimizer" => {
                optimizer = ParseOptimizer(&value);
            }
            "--hidden-act" => {
                hidden_act = ParseActivation(&value).unwrap_or_else(|e| {
                    eprintln!("{}", e);
                    process::exit(1);
                });
            }
            "--output-act" => {
                output_act = ParseActivation(&value).unwrap_or_else(|e| {
                    eprintln!("{}", e);
                    process::exit(1);
                });
            }
            "--dropout" => {
                dropout_rate = value.parse().unwrap_or(0.0);
            }
            "--l2" => {
                l2_lambda = value.parse().unwrap_or(0.0);
            }
            "--beta1" => {
                beta1 = value.parse().unwrap_or(0.9);
            }
            "--beta2" => {
                beta2 = value.parse().unwrap_or(0.999);
            }
            "--epochs" => {
                epochs = value.parse().unwrap_or(100);
            }
            "--batch" => {
                batch_size = value.parse().unwrap_or(1);
            }
            "--lr-decay-rate" => {
                lr_decay_rate = value.parse().unwrap_or(0.95);
            }
            "--lr-decay-epochs" => {
                lr_decay_epochs = value.parse().unwrap_or(10);
            }
            "--patience" => {
                patience = value.parse().unwrap_or(10);
            }
            _ => {
                if !key.is_empty() {
                    eprintln!("Error: Unknown option: {}", key);
                }
            }
        }
    }

    match command {
        TCommand::CmdCreate => {
            if input_size <= 0 {
                eprintln!("Error: --input (-i) is required");
                process::exit(1);
            }
            if hidden_sizes.is_empty() {
                eprintln!("Error: --hidden (-H) is required");
                process::exit(1);
            }
            if output_size <= 0 {
                eprintln!("Error: --output (-o) is required");
                process::exit(1);
            }
            if save_file.is_empty() {
                eprintln!("Error: --save (-s) is required");
                process::exit(1);
            }

            let mut mlp = match TMultiLayerPerceptronCUDA::new(
                input_size, &hidden_sizes, output_size, hidden_act, output_act
            ) {
                Ok(m) => m,
                Err(e) => {
                    eprintln!("Error creating MLP: {}", e);
                    process::exit(1);
                }
            };

            mlp.LearningRate = learning_rate;
            mlp.Optimizer = optimizer;
            mlp.DropoutRate = dropout_rate;
            mlp.L2Lambda = l2_lambda;
            mlp.Beta1 = beta1;
            mlp.Beta2 = beta2;
            mlp.BatchNormEnabled = batch_norm;

            println!("Created MLP model:");
            println!("  Input size: {}", input_size);
            print!("  Hidden sizes: ");
            for (idx, &size) in hidden_sizes.iter().enumerate() {
                if idx > 0 {
                    print!(",");
                }
                print!("{}", size);
            }
            println!();
            println!("  Output size: {}", output_size);
            println!("  Hidden activation: {}", ActivationToStr(hidden_act));
            println!("  Output activation: {}", ActivationToStr(output_act));
            println!("  Optimizer: {}", OptimizerToStr(optimizer));
            println!("  Learning rate: {:.6}", learning_rate);
            println!("  Dropout rate: {:.4}", dropout_rate);
            println!("  L2 lambda: {:.6}", l2_lambda);
            println!("  Batch normalization: {}", if batch_norm { "enabled" } else { "disabled" });

            if let Err(e) = mlp.SaveModelToJSON(&save_file) {
                eprintln!("Error saving model: {}", e);
                process::exit(1);
            }
            println!("Model saved to JSON: {}", save_file);
        }
        TCommand::CmdTrain => {
            if model_file.is_empty() {
                eprintln!("Error: --model (-m) is required");
                process::exit(1);
            }
            if save_file.is_empty() {
                eprintln!("Error: --save (-s) is required");
                process::exit(1);
            }

            println!("Model loaded from JSON: {}", model_file);
            let mlp = TMultiLayerPerceptronCUDA::Load(&model_file);
            match mlp {
                Ok(_) => {
                    println!("Model loaded successfully. Training functionality not yet implemented.");
                }
                Err(e) => {
                    eprintln!("Error loading model: {}", e);
                    process::exit(1);
                }
            }
        }
        TCommand::CmdPredict => {
            if model_file.is_empty() {
                eprintln!("Error: --model (-m) is required");
                process::exit(1);
            }
            if input_values.is_empty() {
                eprintln!("Error: --input (-i) is required");
                process::exit(1);
            }

            let mut mlp = match TMultiLayerPerceptronCUDA::Load(&model_file) {
                Ok(m) => m,
                Err(e) => {
                    eprintln!("Error loading model: {}", e);
                    process::exit(1);
                }
            };
            println!("Model loaded successfully");

            if input_values.len() as i32 != mlp.GetInputSize() {
                eprintln!(
                    "Error: Expected {} input values, got {}",
                    mlp.GetInputSize(),
                    input_values.len()
                );
                process::exit(1);
            }

            let output = match mlp.Predict(&input_values) {
                Ok(o) => o,
                Err(e) => {
                    eprintln!("Error during prediction: {}", e);
                    process::exit(1);
                }
            };

            print!("Input: ");
            for (idx, &v) in input_values.iter().enumerate() {
                if idx > 0 {
                    print!(", ");
                }
                print!("{:.4}", v);
            }
            println!();

            print!("Output: ");
            for (idx, &v) in output.iter().enumerate() {
                if idx > 0 {
                    print!(", ");
                }
                print!("{:.6}", v);
            }
            println!();

            if output.len() > 1 {
                let max_idx = MaxIndex(&output);
                println!("Max index: {}", max_idx);
            }
        }
        TCommand::CmdInfo => {
            if model_file.is_empty() {
                eprintln!("Error: --model (-m) is required");
                process::exit(1);
            }

            println!("Model loaded from JSON: {}", model_file);
            let mlp = match TMultiLayerPerceptronCUDA::Load(&model_file) {
                Ok(m) => m,
                Err(e) => {
                    eprintln!("Error loading model: {}", e);
                    process::exit(1);
                }
            };

            println!("MLP Model Information");
            println!("=====================");
            println!("Input size: {}", mlp.GetInputSize());
            println!("Output size: {}", mlp.GetOutputSize());
            println!("Hidden layers: {}", mlp.GetHiddenLayerCount());
            print!("Hidden sizes: {}", mlp.GetHiddenLayerCount());
            if mlp.GetHiddenLayerCount() > 0 {
                for &size in mlp.GetHiddenSizes().iter() {
                    print!(", {}", size);
                }
            }
            println!();
            print!("Layer sizes: {}", mlp.GetInputSize());
            for &size in mlp.GetHiddenSizes().iter() {
                print!(" -> {}", size);
            }
            println!(" -> {}", mlp.GetOutputSize());
            println!();
            println!("Hyperparameters:");
            println!("  Learning rate: {:.6}", mlp.LearningRate);
            println!("  Optimizer: {}", OptimizerToStr(mlp.Optimizer));
            println!("  Hidden activation: {}", ActivationToStr(mlp.HiddenActivation));
            println!("  Output activation: {}", ActivationToStr(mlp.OutputActivation));
            println!("  Dropout rate: {:.4}", mlp.DropoutRate);
            println!("  L2 lambda: {:.6}", mlp.L2Lambda);
            println!("  Beta1: {:.4}", mlp.Beta1);
            println!("  Beta2: {:.4}", mlp.Beta2);
            println!("  Timestep: {}", mlp.Timestep);
            println!("  Batch normalization: {}", if mlp.BatchNormEnabled { "enabled" } else { "disabled" });
            println!();
            println!("Total layers: {}", mlp.GetHiddenLayerCount() + 2);
            println!("  Layer 0: {} neurons (input)", mlp.GetInputSize());
            for (idx, &size) in mlp.GetHiddenSizes().iter().enumerate() {
                println!("  Layer {}: {} neurons", idx + 1, size);
            }
            println!(
                "  Layer {}: {} neurons (output)",
                mlp.GetHiddenLayerCount() + 1,
                mlp.GetOutputSize()
            );
        }
        TCommand::CmdExportONNX => {
            if model_file.is_empty() {
                eprintln!("Error: --model (-m) is required");
                process::exit(1);
            }
            if save_file.is_empty() {
                eprintln!("Error: --save (-s) is required");
                process::exit(1);
            }

            let mlp = match TMultiLayerPerceptronCUDA::Load(&model_file) {
                Ok(m) => m,
                Err(e) => {
                    eprintln!("Error loading model: {}", e);
                    process::exit(1);
                }
            };

            if let Err(e) = mlp.export_to_onnx(&save_file) {
                eprintln!("Error exporting to ONNX: {}", e);
                process::exit(1);
            }
            println!("Model exported to ONNX: {}", save_file);
        }
        TCommand::CmdFeatureImportance => {
            if model_file.is_empty() {
                eprintln!("Error: --model (-m) is required");
                process::exit(1);
            }

            let mlp = match TMultiLayerPerceptronCUDA::Load(&model_file) {
                Ok(m) => m,
                Err(e) => {
                    eprintln!("Error loading model: {}", e);
                    process::exit(1);
                }
            };

            let importance = mlp.feature_importance();
            println!("Feature Importance (ranked by weight magnitude sum):");
            println!("=====================================================");
            for (rank, (idx, score)) in importance.iter().enumerate() {
                println!("  Rank {}: Feature {} - Score: {:.6}", rank + 1, idx, score);
            }
        }
        _ => {}
    }
}
