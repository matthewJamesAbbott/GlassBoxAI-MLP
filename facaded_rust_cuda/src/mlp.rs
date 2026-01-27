use std::fs::File;
use std::io::{self, BufRead, BufReader, Write};
use std::sync::Arc;

use cudarc::driver::{CudaDevice, CudaSlice, LaunchAsync, LaunchConfig};
use rand::Rng;
use serde::{Deserialize, Serialize};

use crate::kernels::{CUDA_KERNEL_SRC, KERNEL_NAMES};
use crate::{BLOCK_SIZE, EPSILON, MODEL_MAGIC};

#[derive(Clone, Copy, PartialEq, Eq, Debug, Serialize, Deserialize)]
#[repr(i32)]
pub enum TActivationType {
    atSigmoid = 0,
    atTanh = 1,
    atReLU = 2,
    atSoftmax = 3,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug, Serialize, Deserialize)]
#[repr(i32)]
pub enum TOptimizerType {
    otSGD = 0,
    otAdam = 1,
    otRMSProp = 2,
}

pub type Darray = Vec<f64>;
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
    pub d_dGamma: CudaSlice<f64>,
    pub d_dBeta: CudaSlice<f64>,
}

pub fn ActivationToStr(act: TActivationType) -> &'static str {
    match act {
        TActivationType::atSigmoid => "sigmoid",
        TActivationType::atTanh => "tanh",
        TActivationType::atReLU => "relu",
        TActivationType::atSoftmax => "softmax",
    }
}

pub fn OptimizerToStr(opt: TOptimizerType) -> &'static str {
    match opt {
        TOptimizerType::otSGD => "sgd",
        TOptimizerType::otAdam => "adam",
        TOptimizerType::otRMSProp => "rmsprop",
    }
}

pub fn ParseActivation(s: &str) -> TActivationType {
    match s.to_lowercase().as_str() {
        "tanh" => TActivationType::atTanh,
        "relu" => TActivationType::atReLU,
        "softmax" => TActivationType::atSoftmax,
        _ => TActivationType::atSigmoid,
    }
}

pub fn ParseOptimizer(s: &str) -> TOptimizerType {
    match s.to_lowercase().as_str() {
        "adam" => TOptimizerType::otAdam,
        "rmsprop" => TOptimizerType::otRMSProp,
        _ => TOptimizerType::otSGD,
    }
}

pub fn ParseIntArray(s: &str) -> TIntArray {
    s.split(',')
        .filter_map(|x| x.trim().parse().ok())
        .collect()
}

pub fn ParseDoubleArray(s: &str) -> Darray {
    s.split(',')
        .filter_map(|x| x.trim().parse().ok())
        .collect()
}

pub fn MaxIndex(arr: &[f64]) -> usize {
    arr.iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(i, _)| i)
        .unwrap_or(0)
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
    input_layer: InputLayerJSON,
    hidden_layers: Vec<HiddenLayerJSON>,
    output_layer: OutputLayerJSON,
    #[serde(default)]
    batch_norm: bool,
}

pub struct TMultiLayerPerceptronCUDA {
    pub Dev: Arc<CudaDevice>,
    pub Layers: Vec<LayerDataCUDA>,
    pub NumLayers: i32,
    pub FInputSize: i32,
    pub FOutputSize: i32,
    pub FHiddenSizes: Vec<i32>,
    pub FIsTraining: bool,
    pub MaxNeurons: i32,

    pub d_Target: CudaSlice<f64>,
    pub d_SoftmaxSums: CudaSlice<f64>,
    pub d_AdamParams: CudaSlice<f64>,

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
    pub UseBatchNorm: bool,
    pub BNMomentum: f64,
    pub BNEpsilon: f64,
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
        let ptx = cudarc::nvrtc::compile_ptx(CUDA_KERNEL_SRC)?;
        dev.load_ptx(ptx, "mlp_kernels", KERNEL_NAMES)?;

        let num_layers = (HiddenSizes.len() as i32) + 2;
        let mut layers = Vec::new();
        let mut max_neurons = InputSize + 1;

        layers.push(Self::AllocateLayerStatic(&dev, InputSize + 1, InputSize, TActivationType::atSigmoid)?);

        let mut num_inputs = InputSize;
        for &hidden_size in HiddenSizes.iter() {
            layers.push(Self::AllocateLayerStatic(&dev, hidden_size + 1, num_inputs + 1, HiddenAct)?);
            if hidden_size + 1 > max_neurons {
                max_neurons = hidden_size + 1;
            }
            num_inputs = hidden_size;
        }

        layers.push(Self::AllocateLayerStatic(&dev, OutputSize, num_inputs + 1, OutputAct)?);
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
            UseBatchNorm: false,
            BNMomentum: 0.1,
            BNEpsilon: 1e-5,
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

        Ok(LayerDataCUDA {
            Weights: dev.htod_copy(weights)?,
            Biases: dev.alloc_zeros::<f64>(neuron_count)?,
            Outputs: dev.alloc_zeros::<f64>(neuron_count)?,
            Errors: dev.alloc_zeros::<f64>(neuron_count)?,
            M: dev.alloc_zeros::<f64>(weight_size)?,
            V: dev.alloc_zeros::<f64>(weight_size)?,
            MBias: dev.alloc_zeros::<f64>(neuron_count)?,
            VBias: dev.alloc_zeros::<f64>(neuron_count)?,
            DropoutMask: dev.htod_copy(vec![1u8; neuron_count])?,
            NumNeurons: num_neurons,
            NumInputs: num_inputs,
            ActivationType: act_type,
            d_Gamma: dev.htod_copy(vec![1.0f64; neuron_count])?,
            d_Beta: dev.alloc_zeros::<f64>(neuron_count)?,
            d_RunningMean: dev.alloc_zeros::<f64>(neuron_count)?,
            d_RunningVar: dev.htod_copy(vec![1.0f64; neuron_count])?,
            d_BatchMean: dev.alloc_zeros::<f64>(neuron_count)?,
            d_BatchVar: dev.alloc_zeros::<f64>(neuron_count)?,
            d_dGamma: dev.alloc_zeros::<f64>(neuron_count)?,
            d_dBeta: dev.alloc_zeros::<f64>(neuron_count)?,
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
            let layer = &self.Layers[k];
            let prev_layer = &self.Layers[k - 1];
            let blocks = Self::GetBlocks(layer.NumNeurons);
            let cfg = LaunchConfig { block_dim: (BLOCK_SIZE, 1, 1), grid_dim: (blocks, 1, 1), shared_mem_bytes: 0 };

            unsafe {
                ff_kernel.clone().launch(cfg, (
                    &layer.Outputs, &layer.Weights, &layer.Biases, &prev_layer.Outputs,
                    layer.NumNeurons, layer.NumInputs, prev_layer.NumNeurons, layer.ActivationType as i32,
                ))?;
            }

            if self.FIsTraining && self.DropoutRate > 0.0 {
                let scale = 1.0 / (1.0 - self.DropoutRate);
                let seed = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos() as u64;
                unsafe {
                    dropout_kernel.clone().launch(cfg, (
                        &layer.Outputs, &layer.DropoutMask, layer.NumNeurons, self.DropoutRate, scale, seed,
                    ))?;
                }
            }
        }

        let output_layer = &self.Layers[num_layers - 1];
        let last_hidden = &self.Layers[num_layers - 2];
        let blocks = Self::GetBlocks(output_layer.NumNeurons);
        let cfg = LaunchConfig { block_dim: (BLOCK_SIZE, 1, 1), grid_dim: (blocks, 1, 1), shared_mem_bytes: 0 };

        if self.OutputActivation == TActivationType::atSoftmax {
            let softmax_sum_kernel = self.Dev.get_func("mlp_kernels", "FeedForwardSoftmaxSumKernel").unwrap();
            let softmax_kernel = self.Dev.get_func("mlp_kernels", "SoftmaxKernel").unwrap();

            unsafe {
                softmax_sum_kernel.clone().launch(cfg, (
                    &self.d_SoftmaxSums, &output_layer.Weights, &output_layer.Biases, &last_hidden.Outputs,
                    output_layer.NumNeurons, output_layer.NumInputs, last_hidden.NumNeurons,
                ))?;
            }
            self.Dev.synchronize()?;

            let h_sums = self.Dev.dtoh_sync_copy(&self.d_SoftmaxSums)?;
            let max_val = h_sums.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let sum_exp: f64 = h_sums.iter().map(|&s| (s - max_val).exp()).sum();

            unsafe {
                softmax_kernel.clone().launch(cfg, (
                    &self.d_SoftmaxSums, &output_layer.Outputs, output_layer.NumNeurons, max_val, sum_exp,
                ))?;
            }
        } else {
            unsafe {
                ff_kernel.clone().launch(cfg, (
                    &output_layer.Outputs, &output_layer.Weights, &output_layer.Biases, &last_hidden.Outputs,
                    output_layer.NumNeurons, output_layer.NumInputs, last_hidden.NumNeurons, output_layer.ActivationType as i32,
                ))?;
            }
        }
        self.Dev.synchronize()?;
        Ok(())
    }

    pub fn BackPropagate(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        let num_layers = self.NumLayers as usize;
        let output_layer = &self.Layers[num_layers - 1];
        let blocks = Self::GetBlocks(output_layer.NumNeurons);
        let cfg = LaunchConfig { block_dim: (BLOCK_SIZE, 1, 1), grid_dim: (blocks, 1, 1), shared_mem_bytes: 0 };

        let bp_output_kernel = self.Dev.get_func("mlp_kernels", "BackPropOutputKernel").unwrap();
        let is_softmax = if self.OutputActivation == TActivationType::atSoftmax { 1i32 } else { 0i32 };

        unsafe {
            bp_output_kernel.clone().launch(cfg, (
                &output_layer.Errors, &output_layer.Outputs, &self.d_Target,
                output_layer.NumNeurons, output_layer.ActivationType as i32, is_softmax,
            ))?;
        }
        self.Dev.synchronize()?;

        let bp_hidden_kernel = self.Dev.get_func("mlp_kernels", "BackPropHiddenKernel").unwrap();
        for k in (1..(num_layers - 1)).rev() {
            let layer = &self.Layers[k];
            let next_layer = &self.Layers[k + 1];
            let blocks = Self::GetBlocks(layer.NumNeurons);
            let cfg = LaunchConfig { block_dim: (BLOCK_SIZE, 1, 1), grid_dim: (blocks, 1, 1), shared_mem_bytes: 0 };

            unsafe {
                bp_hidden_kernel.clone().launch(cfg, (
                    &layer.Errors, &layer.Outputs, &layer.DropoutMask,
                    &next_layer.Errors, &next_layer.Weights,
                    layer.NumNeurons, next_layer.NumNeurons, next_layer.NumInputs, layer.ActivationType as i32,
                ))?;
            }
            self.Dev.synchronize()?;
        }
        Ok(())
    }

    pub fn UpdateWeights(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        self.Timestep += 1;
        let num_layers = self.NumLayers as usize;

        for k in (1..num_layers).rev() {
            let layer = &self.Layers[k];
            let prev_layer = &self.Layers[k - 1];
            let blocks = Self::GetBlocks(layer.NumNeurons);
            let cfg = LaunchConfig { block_dim: (BLOCK_SIZE, 1, 1), grid_dim: (blocks, 1, 1), shared_mem_bytes: 0 };

            match self.Optimizer {
                TOptimizerType::otSGD => {
                    let kernel = self.Dev.get_func("mlp_kernels", "UpdateWeightsSGDKernel").unwrap();
                    unsafe {
                        kernel.clone().launch(cfg, (
                            &layer.Weights, &layer.Biases, &layer.Errors, &prev_layer.Outputs,
                            layer.NumNeurons, layer.NumInputs, prev_layer.NumNeurons,
                            self.LearningRate, self.L2Lambda,
                        ))?;
                    }
                }
                TOptimizerType::otAdam => {
                    let kernel = self.Dev.get_func("mlp_kernels", "UpdateWeightsAdamKernel").unwrap();
                    let adam_params = vec![self.LearningRate, self.L2Lambda, self.Beta1, self.Beta2, self.Timestep as f64];
                    self.Dev.htod_sync_copy_into(&adam_params, &mut self.d_AdamParams)?;
                    unsafe {
                        kernel.clone().launch(cfg, (
                            &layer.Weights, &layer.Biases, &layer.Errors, &prev_layer.Outputs,
                            &layer.M, &layer.V, &layer.MBias, &layer.VBias,
                            layer.NumNeurons, layer.NumInputs, prev_layer.NumNeurons, &self.d_AdamParams,
                        ))?;
                    }
                }
                TOptimizerType::otRMSProp => {
                    let kernel = self.Dev.get_func("mlp_kernels", "UpdateWeightsRMSPropKernel").unwrap();
                    unsafe {
                        kernel.clone().launch(cfg, (
                            &layer.Weights, &layer.Biases, &layer.Errors, &prev_layer.Outputs,
                            &layer.V, &layer.VBias,
                            layer.NumNeurons, layer.NumInputs, prev_layer.NumNeurons,
                            self.LearningRate, self.L2Lambda,
                        ))?;
                    }
                }
            }
        }
        self.Dev.synchronize()?;
        Ok(())
    }

    pub fn Predict(&mut self, input: &[f64]) -> Result<Darray, Box<dyn std::error::Error>> {
        self.FIsTraining = false;
        let input_size = self.FInputSize as usize;
        let mut h_input = vec![0.0f64; input_size + 1];
        for i in 0..input_size { h_input[i] = input[i]; }
        h_input[input_size] = 1.0;
        self.Dev.htod_sync_copy_into(&h_input, &mut self.Layers[0].Outputs)?;
        self.FeedForward()?;
        let num_layers = self.NumLayers as usize;
        let result = self.Dev.dtoh_sync_copy(&self.Layers[num_layers - 1].Outputs)?;
        self.FIsTraining = true;
        Ok(result)
    }

    pub fn Train(&mut self, input: &[f64], target: &[f64]) -> Result<(), Box<dyn std::error::Error>> {
        self.FIsTraining = true;
        let input_size = self.FInputSize as usize;
        let mut h_input = vec![0.0f64; input_size + 1];
        for i in 0..input_size { h_input[i] = input[i]; }
        h_input[input_size] = 1.0;
        self.Dev.htod_sync_copy_into(&h_input, &mut self.Layers[0].Outputs)?;
        self.Dev.htod_sync_copy_into(target, &mut self.d_Target)?;
        self.FeedForward()?;
        self.BackPropagate()?;
        self.UpdateWeights()?;
        Ok(())
    }

    pub fn ComputeLoss(&self, predicted: &[f64], target: &[f64]) -> f64 {
        if self.OutputActivation == TActivationType::atSoftmax {
            let mut result = 0.0;
            for i in 0..predicted.len() {
                let p = predicted[i].max(EPSILON).min(1.0 - EPSILON);
                result -= target[i] * p.ln();
            }
            result
        } else {
            predicted.iter().zip(target).map(|(p, t)| 0.5 * (t - p).powi(2)).sum()
        }
    }

    pub fn GetOutputSize(&self) -> i32 { self.FOutputSize }
    pub fn GetInputSize(&self) -> i32 { self.FInputSize }
    pub fn GetHiddenLayerCount(&self) -> usize { self.FHiddenSizes.len() }
    pub fn GetHiddenSizes(&self) -> &Vec<i32> { &self.FHiddenSizes }
    pub fn GetNumLayers(&self) -> i32 { self.NumLayers }
    pub fn GetLayerSize(&self, layer_idx: usize) -> i32 {
        if layer_idx >= self.Layers.len() { 0 } else { self.Layers[layer_idx].NumNeurons }
    }

    // ===== FACADE METHODS =====
    pub fn GetWeightsPerNeuron(&self, layer_idx: i32, neuron_idx: i32) -> i32 {
        if layer_idx < 0 || layer_idx >= self.NumLayers { return 0; }
        self.Layers[layer_idx as usize].NumInputs
    }

    pub fn GetNeuronWeight(&self, layer_idx: i32, neuron_idx: i32, weight_idx: i32) -> f64 {
        if layer_idx < 0 || layer_idx >= self.NumLayers { return 0.0; }
        let layer = &self.Layers[layer_idx as usize];
        if neuron_idx < 0 || neuron_idx >= layer.NumNeurons { return 0.0; }
        if weight_idx < 0 || weight_idx >= layer.NumInputs { return 0.0; }
        let weights = self.Dev.dtoh_sync_copy(&layer.Weights).unwrap();
        weights[(neuron_idx * layer.NumInputs + weight_idx) as usize]
    }

    pub fn SetNeuronWeight(&mut self, layer_idx: i32, neuron_idx: i32, weight_idx: i32, value: f64) {
        if layer_idx < 0 || layer_idx >= self.NumLayers { return; }
        let layer = &self.Layers[layer_idx as usize];
        if neuron_idx < 0 || neuron_idx >= layer.NumNeurons { return; }
        if weight_idx < 0 || weight_idx >= layer.NumInputs { return; }
        let mut weights = self.Dev.dtoh_sync_copy(&layer.Weights).unwrap();
        weights[(neuron_idx * layer.NumInputs + weight_idx) as usize] = value;
        self.Dev.htod_sync_copy_into(&weights, &mut self.Layers[layer_idx as usize].Weights).unwrap();
    }

    pub fn GetNeuronWeights(&self, layer_idx: i32, neuron_idx: i32) -> Darray {
        if layer_idx < 0 || layer_idx >= self.NumLayers { return vec![]; }
        let layer = &self.Layers[layer_idx as usize];
        if neuron_idx < 0 || neuron_idx >= layer.NumNeurons { return vec![]; }
        let weights = self.Dev.dtoh_sync_copy(&layer.Weights).unwrap();
        let start = (neuron_idx * layer.NumInputs) as usize;
        let end = start + layer.NumInputs as usize;
        weights[start..end].to_vec()
    }

    pub fn GetNeuronBias(&self, layer_idx: i32, neuron_idx: i32) -> f64 {
        if layer_idx < 0 || layer_idx >= self.NumLayers { return 0.0; }
        let layer = &self.Layers[layer_idx as usize];
        if neuron_idx < 0 || neuron_idx >= layer.NumNeurons { return 0.0; }
        let biases = self.Dev.dtoh_sync_copy(&layer.Biases).unwrap();
        biases[neuron_idx as usize]
    }

    pub fn SetNeuronBias(&mut self, layer_idx: i32, neuron_idx: i32, value: f64) {
        if layer_idx < 0 || layer_idx >= self.NumLayers { return; }
        let layer = &self.Layers[layer_idx as usize];
        if neuron_idx < 0 || neuron_idx >= layer.NumNeurons { return; }
        let mut biases = self.Dev.dtoh_sync_copy(&layer.Biases).unwrap();
        biases[neuron_idx as usize] = value;
        self.Dev.htod_sync_copy_into(&biases, &mut self.Layers[layer_idx as usize].Biases).unwrap();
    }

    pub fn GetNeuronOutput(&self, layer_idx: i32, neuron_idx: i32) -> f64 {
        if layer_idx < 0 || layer_idx >= self.NumLayers { return 0.0; }
        let layer = &self.Layers[layer_idx as usize];
        if neuron_idx < 0 || neuron_idx >= layer.NumNeurons { return 0.0; }
        let outputs = self.Dev.dtoh_sync_copy(&layer.Outputs).unwrap();
        outputs[neuron_idx as usize]
    }

    pub fn GetLayerOutputs(&self, layer_idx: i32) -> Darray {
        if layer_idx < 0 || layer_idx >= self.NumLayers { return vec![]; }
        self.Dev.dtoh_sync_copy(&self.Layers[layer_idx as usize].Outputs).unwrap()
    }

    pub fn GetNeuronError(&self, layer_idx: i32, neuron_idx: i32) -> f64 {
        if layer_idx < 0 || layer_idx >= self.NumLayers { return 0.0; }
        let layer = &self.Layers[layer_idx as usize];
        if neuron_idx < 0 || neuron_idx >= layer.NumNeurons { return 0.0; }
        let errors = self.Dev.dtoh_sync_copy(&layer.Errors).unwrap();
        errors[neuron_idx as usize]
    }

    pub fn GetLayerErrors(&self, layer_idx: i32) -> Darray {
        if layer_idx < 0 || layer_idx >= self.NumLayers { return vec![]; }
        self.Dev.dtoh_sync_copy(&self.Layers[layer_idx as usize].Errors).unwrap()
    }

    pub fn GetWeightM(&self, layer_idx: i32, neuron_idx: i32, weight_idx: i32) -> f64 {
        if layer_idx < 0 || layer_idx >= self.NumLayers { return 0.0; }
        let layer = &self.Layers[layer_idx as usize];
        if neuron_idx < 0 || neuron_idx >= layer.NumNeurons { return 0.0; }
        if weight_idx < 0 || weight_idx >= layer.NumInputs { return 0.0; }
        let m = self.Dev.dtoh_sync_copy(&layer.M).unwrap();
        m[(neuron_idx * layer.NumInputs + weight_idx) as usize]
    }

    pub fn GetWeightV(&self, layer_idx: i32, neuron_idx: i32, weight_idx: i32) -> f64 {
        if layer_idx < 0 || layer_idx >= self.NumLayers { return 0.0; }
        let layer = &self.Layers[layer_idx as usize];
        if neuron_idx < 0 || neuron_idx >= layer.NumNeurons { return 0.0; }
        if weight_idx < 0 || weight_idx >= layer.NumInputs { return 0.0; }
        let v = self.Dev.dtoh_sync_copy(&layer.V).unwrap();
        v[(neuron_idx * layer.NumInputs + weight_idx) as usize]
    }

    pub fn GetBiasM(&self, layer_idx: i32, neuron_idx: i32) -> f64 {
        if layer_idx < 0 || layer_idx >= self.NumLayers { return 0.0; }
        let layer = &self.Layers[layer_idx as usize];
        if neuron_idx < 0 || neuron_idx >= layer.NumNeurons { return 0.0; }
        let m = self.Dev.dtoh_sync_copy(&layer.MBias).unwrap();
        m[neuron_idx as usize]
    }

    pub fn GetBiasV(&self, layer_idx: i32, neuron_idx: i32) -> f64 {
        if layer_idx < 0 || layer_idx >= self.NumLayers { return 0.0; }
        let layer = &self.Layers[layer_idx as usize];
        if neuron_idx < 0 || neuron_idx >= layer.NumNeurons { return 0.0; }
        let v = self.Dev.dtoh_sync_copy(&layer.VBias).unwrap();
        v[neuron_idx as usize]
    }

    pub fn GetLayerActivation(&self, layer_idx: i32) -> TActivationType {
        if layer_idx < 0 || layer_idx >= self.NumLayers { return TActivationType::atSigmoid; }
        self.Layers[layer_idx as usize].ActivationType
    }

    pub fn GetActivationHistogram(&self, layer_idx: i32, num_bins: usize) -> Vec<i32> {
        let mut histogram = vec![0i32; num_bins];
        let outputs = self.GetLayerOutputs(layer_idx);
        if outputs.is_empty() { return histogram; }
        let min_val = outputs.iter().cloned().fold(f64::INFINITY, f64::min);
        let max_val = outputs.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let range = if max_val == min_val { 1.0 } else { max_val - min_val };
        for v in outputs {
            let bin = (((v - min_val) / range) * (num_bins - 1) as f64) as usize;
            histogram[bin.min(num_bins - 1)] += 1;
        }
        histogram
    }

    pub fn GetGradientHistogram(&self, layer_idx: i32, num_bins: usize) -> Vec<i32> {
        let mut histogram = vec![0i32; num_bins];
        let errors = self.GetLayerErrors(layer_idx);
        if errors.is_empty() { return histogram; }
        let min_val = errors.iter().cloned().fold(f64::INFINITY, f64::min);
        let max_val = errors.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let range = if max_val == min_val { 1.0 } else { max_val - min_val };
        for v in errors {
            let bin = (((v - min_val) / range) * (num_bins - 1) as f64) as usize;
            histogram[bin.min(num_bins - 1)] += 1;
        }
        histogram
    }

    pub fn Save(&self, filename: &str) -> io::Result<()> {
        let mut hidden_layers_json = Vec::new();
        for h in 0..self.FHiddenSizes.len() {
            let layer = &self.Layers[h + 1];
            let num_neurons = self.FHiddenSizes[h];
            let num_inputs = if h == 0 { self.FInputSize } else { self.FHiddenSizes[h - 1] };
            let h_weights = self.Dev.dtoh_sync_copy(&layer.Weights).unwrap();
            let h_biases = self.Dev.dtoh_sync_copy(&layer.Biases).unwrap();
            let mut neurons = Vec::new();
            for j in 0..num_neurons as usize {
                let mut weights = Vec::new();
                for w in 0..num_inputs as usize {
                    weights.push(h_weights[j * layer.NumInputs as usize + w]);
                }
                neurons.push(NeuronJSON { weights, bias: h_biases[j] });
            }
            let biases: Vec<f64> = (0..num_neurons as usize).map(|j| h_biases[j]).collect();
            hidden_layers_json.push(HiddenLayerJSON { neuron_count: num_neurons, neurons, biases });
        }

        let num_layers = self.NumLayers as usize;
        let out_layer = &self.Layers[num_layers - 1];
        let out_num_inputs = if self.FHiddenSizes.is_empty() { self.FInputSize } else { *self.FHiddenSizes.last().unwrap() };
        let h_weights = self.Dev.dtoh_sync_copy(&out_layer.Weights).unwrap();
        let h_biases = self.Dev.dtoh_sync_copy(&out_layer.Biases).unwrap();
        let mut out_neurons = Vec::new();
        for i in 0..self.FOutputSize as usize {
            let mut weights = Vec::new();
            for w in 0..out_num_inputs as usize {
                weights.push(h_weights[i * out_layer.NumInputs as usize + w]);
            }
            out_neurons.push(NeuronJSON { weights, bias: h_biases[i] });
        }
        let out_biases: Vec<f64> = (0..self.FOutputSize as usize).map(|i| h_biases[i]).collect();

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
            input_layer: InputLayerJSON { neuron_count: self.FInputSize },
            hidden_layers: hidden_layers_json,
            output_layer: OutputLayerJSON { neuron_count: self.FOutputSize, neurons: out_neurons, biases: out_biases },
            batch_norm: self.UseBatchNorm,
        };

        let json = serde_json::to_string_pretty(&model).map_err(|e| io::Error::new(io::ErrorKind::Other, e))?;
        let mut file = File::create(filename)?;
        file.write_all(json.as_bytes())?;
        Ok(())
    }

    pub fn Load(filename: &str) -> Result<Self, Box<dyn std::error::Error>> {
        let file = File::open(filename)?;
        let reader = BufReader::new(file);
        let model: ModelJSON = serde_json::from_reader(reader)?;

        let hidden_act = match model.hidden_activation { 1 => TActivationType::atTanh, 2 => TActivationType::atReLU, 3 => TActivationType::atSoftmax, _ => TActivationType::atSigmoid };
        let output_act = match model.output_activation { 1 => TActivationType::atTanh, 2 => TActivationType::atReLU, 3 => TActivationType::atSoftmax, _ => TActivationType::atSigmoid };

        let mut mlp = Self::new(model.input_size, &model.hidden_sizes, model.output_size, hidden_act, output_act)?;
        mlp.LearningRate = model.learning_rate;
        mlp.Optimizer = match model.optimizer { 1 => TOptimizerType::otAdam, 2 => TOptimizerType::otRMSProp, _ => TOptimizerType::otSGD };
        mlp.DropoutRate = model.dropout_rate;
        mlp.L2Lambda = model.l2_lambda;
        mlp.Beta1 = model.beta1;
        mlp.Beta2 = model.beta2;
        mlp.UseBatchNorm = model.batch_norm;

        for (h, hl) in model.hidden_layers.iter().enumerate() {
            let layer = &mlp.Layers[h + 1];
            let num_inputs = if h == 0 { model.input_size } else { model.hidden_sizes[h - 1] };
            let mut h_weights = vec![0.0f64; (layer.NumNeurons * layer.NumInputs) as usize];
            let mut h_biases = vec![0.0f64; layer.NumNeurons as usize];
            for (n, neuron) in hl.neurons.iter().enumerate() {
                for (w, &weight) in neuron.weights.iter().enumerate() {
                    if w < num_inputs as usize { h_weights[n * layer.NumInputs as usize + w] = weight; }
                }
                h_biases[n] = neuron.bias;
            }
            mlp.Dev.htod_sync_copy_into(&h_weights, &mut mlp.Layers[h + 1].Weights)?;
            mlp.Dev.htod_sync_copy_into(&h_biases, &mut mlp.Layers[h + 1].Biases)?;
        }

        let num_layers = mlp.NumLayers as usize;
        let out_layer = &mlp.Layers[num_layers - 1];
        let out_num_inputs = if model.hidden_sizes.is_empty() { model.input_size } else { *model.hidden_sizes.last().unwrap() };
        let mut h_weights = vec![0.0f64; (out_layer.NumNeurons * out_layer.NumInputs) as usize];
        let mut h_biases = vec![0.0f64; out_layer.NumNeurons as usize];
        for (n, neuron) in model.output_layer.neurons.iter().enumerate() {
            for (w, &weight) in neuron.weights.iter().enumerate() {
                if w < out_num_inputs as usize { h_weights[n * out_layer.NumInputs as usize + w] = weight; }
            }
            h_biases[n] = neuron.bias;
        }
        mlp.Dev.htod_sync_copy_into(&h_weights, &mut mlp.Layers[num_layers - 1].Weights)?;
        mlp.Dev.htod_sync_copy_into(&h_biases, &mut mlp.Layers[num_layers - 1].Biases)?;

        Ok(mlp)
    }

    pub fn export_to_onnx(&self, filename: &str) -> io::Result<()> {
        use crate::onnx::*;

        let mut graph_inputs = Vec::new();
        let mut graph_outputs = Vec::new();
        let mut nodes = Vec::new();
        let mut initializers = Vec::new();

        graph_inputs.push(ValueInfoProto {
            name: "input".to_string(),
            r#type: Some(TypeProto {
                value: Some(type_proto::Value::TensorType(type_proto::Tensor {
                    elem_type: TensorProto::DOUBLE as i32,
                    shape: Some(TensorShapeProto {
                        dim: vec![
                            tensor_shape_proto::Dimension { value: Some(tensor_shape_proto::dimension::Value::DimParam("batch".to_string())), denotation: String::new() },
                            tensor_shape_proto::Dimension { value: Some(tensor_shape_proto::dimension::Value::DimValue(self.FInputSize as i64)), denotation: String::new() },
                        ],
                    }),
                })),
                denotation: String::new(),
            }),
            doc_string: String::new(),
        });

        let num_layers = self.NumLayers as usize;
        let mut prev_output = "input".to_string();

        for k in 1..num_layers {
            let layer = &self.Layers[k];
            let layer_name = format!("layer_{}", k);

            let h_weights = self.Dev.dtoh_sync_copy(&layer.Weights).unwrap();
            let h_biases = self.Dev.dtoh_sync_copy(&layer.Biases).unwrap();

            let internal_num_neurons = layer.NumNeurons as usize;
            let internal_num_inputs = layer.NumInputs as usize;

            let is_hidden = k < num_layers - 1;
            let onnx_num_outputs = if is_hidden {
                self.FHiddenSizes[k - 1] as usize
            } else {
                self.FOutputSize as usize
            };
            let onnx_num_inputs = if k == 1 {
                self.FInputSize as usize
            } else {
                self.FHiddenSizes[k - 2] as usize
            };

            let mut onnx_weights = vec![0.0f64; onnx_num_outputs * onnx_num_inputs];
            for i in 0..onnx_num_outputs.min(internal_num_neurons) {
                for j in 0..onnx_num_inputs.min(internal_num_inputs) {
                    onnx_weights[i * onnx_num_inputs + j] = h_weights[i * internal_num_inputs + j];
                }
            }

            let onnx_biases: Vec<f64> = h_biases.iter().take(onnx_num_outputs).cloned().collect();

            let weight_name = format!("{}_weight", layer_name);
            let bias_name = format!("{}_bias", layer_name);
            let matmul_out = format!("{}_matmul", layer_name);
            let add_out = format!("{}_add", layer_name);
            let act_out = format!("{}_out", layer_name);

            initializers.push(TensorProto {
                dims: vec![onnx_num_outputs as i64, onnx_num_inputs as i64],
                data_type: TensorProto::DOUBLE as i32,
                double_data: onnx_weights,
                name: weight_name.clone(),
                ..Default::default()
            });

            initializers.push(TensorProto {
                dims: vec![onnx_num_outputs as i64],
                data_type: TensorProto::DOUBLE as i32,
                double_data: onnx_biases,
                name: bias_name.clone(),
                ..Default::default()
            });

            nodes.push(NodeProto {
                input: vec![prev_output.clone(), weight_name],
                output: vec![matmul_out.clone()],
                name: format!("{}_gemm", layer_name),
                op_type: "Gemm".to_string(),
                attribute: vec![
                    AttributeProto { name: "transB".to_string(), r#type: AttributeProto::INT as i32, i: 1, ..Default::default() },
                ],
                ..Default::default()
            });

            nodes.push(NodeProto {
                input: vec![matmul_out, bias_name],
                output: vec![add_out.clone()],
                name: format!("{}_add", layer_name),
                op_type: "Add".to_string(),
                ..Default::default()
            });

            let act_type = layer.ActivationType;
            let (op_type, final_out) = if k == num_layers - 1 && act_type == TActivationType::atSoftmax {
                ("Softmax", act_out.clone())
            } else {
                match act_type {
                    TActivationType::atSigmoid => ("Sigmoid", act_out.clone()),
                    TActivationType::atTanh => ("Tanh", act_out.clone()),
                    TActivationType::atReLU => ("Relu", act_out.clone()),
                    TActivationType::atSoftmax => ("Softmax", act_out.clone()),
                }
            };

            nodes.push(NodeProto {
                input: vec![add_out],
                output: vec![final_out.clone()],
                name: format!("{}_{}", layer_name, op_type.to_lowercase()),
                op_type: op_type.to_string(),
                ..Default::default()
            });

            prev_output = final_out;
        }

        graph_outputs.push(ValueInfoProto {
            name: prev_output.clone(),
            r#type: Some(TypeProto {
                value: Some(type_proto::Value::TensorType(type_proto::Tensor {
                    elem_type: TensorProto::DOUBLE as i32,
                    shape: Some(TensorShapeProto {
                        dim: vec![
                            tensor_shape_proto::Dimension { value: Some(tensor_shape_proto::dimension::Value::DimParam("batch".to_string())), denotation: String::new() },
                            tensor_shape_proto::Dimension { value: Some(tensor_shape_proto::dimension::Value::DimValue(self.FOutputSize as i64)), denotation: String::new() },
                        ],
                    }),
                })),
                denotation: String::new(),
            }),
            doc_string: String::new(),
        });

        let graph = GraphProto {
            name: "mlp".to_string(),
            node: nodes,
            input: graph_inputs,
            output: graph_outputs,
            initializer: initializers,
            ..Default::default()
        };

        let model = ModelProto {
            ir_version: 8,
            opset_import: vec![OperatorSetIdProto { domain: String::new(), version: 13 }],
            producer_name: "facaded_mlp_cuda".to_string(),
            producer_version: "1.0.0".to_string(),
            graph: Some(graph),
            ..Default::default()
        };

        use prost::Message;
        let mut buf = Vec::new();
        model.encode(&mut buf).map_err(|e| io::Error::new(io::ErrorKind::Other, e))?;
        let mut file = File::create(filename)?;
        file.write_all(&buf)?;
        Ok(())
    }

    pub fn import_from_onnx(filename: &str) -> Result<Self, Box<dyn std::error::Error>> {
        use crate::onnx::*;
        use prost::Message;

        let data = std::fs::read(filename)?;
        let model = ModelProto::decode(&data[..])?;
        let graph = model.graph.ok_or("No graph in ONNX model")?;

        let mut layer_weights: Vec<(Vec<f64>, Vec<f64>, i32, i32, TActivationType)> = Vec::new();
        let mut initializer_map: std::collections::HashMap<String, &TensorProto> = std::collections::HashMap::new();

        for init in &graph.initializer {
            initializer_map.insert(init.name.clone(), init);
        }

        let mut input_size = 0i32;
        if let Some(input_info) = graph.input.first() {
            if let Some(ref type_proto) = input_info.r#type {
                if let Some(type_proto::Value::TensorType(ref tensor)) = type_proto.value {
                    if let Some(ref shape) = tensor.shape {
                        if shape.dim.len() >= 2 {
                            if let Some(tensor_shape_proto::dimension::Value::DimValue(v)) = &shape.dim[1].value {
                                input_size = *v as i32;
                            }
                        }
                    }
                }
            }
        }

        let mut gemm_nodes: Vec<&NodeProto> = Vec::new();
        let mut act_nodes: Vec<&NodeProto> = Vec::new();

        for node in &graph.node {
            match node.op_type.as_str() {
                "Gemm" | "MatMul" => gemm_nodes.push(node),
                "Sigmoid" | "Tanh" | "Relu" | "Softmax" => act_nodes.push(node),
                _ => {}
            }
        }

        for (i, gemm) in gemm_nodes.iter().enumerate() {
            let weight_name = &gemm.input[1];
            let weight_tensor = initializer_map.get(weight_name).ok_or("Missing weight tensor")?;

            let dims = &weight_tensor.dims;
            let num_outputs = dims[0] as i32;
            let num_inputs = dims[1] as i32;

            let weights = weight_tensor.double_data.clone();
            let bias_name = gemm.input.get(2).cloned().unwrap_or_default();
            let biases = if let Some(bias_tensor) = initializer_map.get(&bias_name) {
                bias_tensor.double_data.clone()
            } else {
                let add_node = graph.node.iter().find(|n| n.op_type == "Add" && n.input.iter().any(|inp| inp.contains(&gemm.output[0])));
                if let Some(add) = add_node {
                    let bias_input = add.input.iter().find(|inp| initializer_map.contains_key(*inp));
                    if let Some(bi) = bias_input {
                        initializer_map.get(bi).map(|t| t.double_data.clone()).unwrap_or_else(|| vec![0.0; num_outputs as usize])
                    } else {
                        vec![0.0; num_outputs as usize]
                    }
                } else {
                    vec![0.0; num_outputs as usize]
                }
            };

            let act_type = if i < act_nodes.len() {
                match act_nodes[i].op_type.as_str() {
                    "Sigmoid" => TActivationType::atSigmoid,
                    "Tanh" => TActivationType::atTanh,
                    "Relu" => TActivationType::atReLU,
                    "Softmax" => TActivationType::atSoftmax,
                    _ => TActivationType::atSigmoid,
                }
            } else {
                TActivationType::atSigmoid
            };

            layer_weights.push((weights, biases, num_inputs, num_outputs, act_type));
        }

        if layer_weights.is_empty() {
            return Err("No layers found in ONNX model".into());
        }

        let output_size = layer_weights.last().unwrap().3;
        let hidden_sizes: Vec<i32> = layer_weights.iter().take(layer_weights.len() - 1).map(|l| l.3).collect();
        let hidden_act = if !layer_weights.is_empty() { layer_weights[0].4 } else { TActivationType::atSigmoid };
        let output_act = layer_weights.last().map(|l| l.4).unwrap_or(TActivationType::atSigmoid);

        let mut mlp = Self::new(input_size, &hidden_sizes, output_size, hidden_act, output_act)?;

        for (i, (weights, biases, onnx_num_inputs, onnx_num_outputs, _act)) in layer_weights.iter().enumerate() {
            let layer_idx = i + 1;
            if layer_idx < mlp.Layers.len() {
                let layer = &mlp.Layers[layer_idx];
                let internal_num_inputs = layer.NumInputs as usize;
                let internal_num_neurons = layer.NumNeurons as usize;
                let onnx_inputs = *onnx_num_inputs as usize;
                let onnx_outputs = *onnx_num_outputs as usize;

                let mut padded_weights = vec![0.0f64; internal_num_neurons * internal_num_inputs];
                for n in 0..onnx_outputs.min(internal_num_neurons) {
                    for w in 0..onnx_inputs.min(internal_num_inputs) {
                        padded_weights[n * internal_num_inputs + w] = weights[n * onnx_inputs + w];
                    }
                }

                let mut padded_biases = vec![0.0f64; internal_num_neurons];
                for n in 0..biases.len().min(internal_num_neurons) {
                    padded_biases[n] = biases[n];
                }

                mlp.Dev.htod_sync_copy_into(&padded_weights, &mut mlp.Layers[layer_idx].Weights)?;
                mlp.Dev.htod_sync_copy_into(&padded_biases, &mut mlp.Layers[layer_idx].Biases)?;
            }
        }

        Ok(mlp)
    }

    pub fn compute_feature_importance(&self) -> Vec<(usize, f64)> {
        if self.NumLayers < 2 {
            return vec![];
        }

        let first_hidden_layer = &self.Layers[1];
        let weights = self.Dev.dtoh_sync_copy(&first_hidden_layer.Weights).unwrap();
        let num_neurons = first_hidden_layer.NumNeurons as usize;
        let num_inputs = first_hidden_layer.NumInputs as usize;

        let mut importance: Vec<(usize, f64)> = Vec::with_capacity(self.FInputSize as usize);

        for input_idx in 0..self.FInputSize as usize {
            let mut sum = 0.0;
            for neuron_idx in 0..num_neurons {
                if input_idx < num_inputs {
                    sum += weights[neuron_idx * num_inputs + input_idx].abs();
                }
            }
            importance.push((input_idx, sum));
        }

        importance.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        importance
    }
}

pub fn ShuffleData(data: &mut TDataPointArray) {
    let mut rng = rand::thread_rng();
    for i in (1..data.len()).rev() {
        let j = rng.gen_range(0..=i);
        data.swap(i, j);
    }
}

pub fn NormalizeData(data: &mut TDataPointArray) {
    if data.is_empty() { return; }
    let input_size = data[0].Input.len();
    let mut mins: Vec<f64> = data[0].Input.clone();
    let mut maxs: Vec<f64> = data[0].Input.clone();
    for dp in data.iter() {
        for j in 0..input_size {
            if dp.Input[j] < mins[j] { mins[j] = dp.Input[j]; }
            if dp.Input[j] > maxs[j] { maxs[j] = dp.Input[j]; }
        }
    }
    for dp in data.iter_mut() {
        for j in 0..input_size {
            let range = maxs[j] - mins[j];
            dp.Input[j] = if range > 0.0 { (dp.Input[j] - mins[j]) / range } else { 0.5 };
        }
    }
}

pub fn LoadDataCSV(filename: &str, input_size: i32, output_size: i32) -> TDataPointArray {
    let mut data = TDataPointArray::new();
    let file = match File::open(filename) { Ok(f) => f, Err(_) => return data };
    let reader = BufReader::new(file);
    for line in reader.lines() {
        let line = match line { Ok(l) => l, Err(_) => continue };
        if line.is_empty() { continue; }
        let values = ParseDoubleArray(&line);
        if (values.len() as i32) < input_size + output_size { continue; }
        let mut dp = TDataPoint { Input: vec![0.0; input_size as usize], Target: vec![0.0; output_size as usize] };
        for i in 0..input_size as usize { dp.Input[i] = values[i]; }
        for i in 0..output_size as usize { dp.Target[i] = values[input_size as usize + i]; }
        data.push(dp);
    }
    data
}
