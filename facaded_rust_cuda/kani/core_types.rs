//! Core types mirrored from main MLP implementation for verification.
//! 
//! These types are standalone to allow Kani verification without CUDA dependencies.

pub const EPSILON: f64 = 1e-15;
pub const MAX_LAYERS: usize = 16;
pub const MAX_NEURONS_PER_LAYER: usize = 1024;
pub const MAX_ARRAY_SIZE: usize = 4096;
pub const MEMORY_BUDGET: usize = 1024 * 1024 * 64; // 64 MB security budget

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[repr(i32)]
pub enum TActivationType {
    AtSigmoid = 0,
    AtTanh = 1,
    AtReLU = 2,
    AtSoftmax = 3,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[repr(i32)]
pub enum TOptimizerType {
    OtSGD = 0,
    OtAdam = 1,
    OtRMSProp = 2,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[repr(i32)]
pub enum TCommand {
    CmdNone = 0,
    CmdCreate = 1,
    CmdTrain = 2,
    CmdPredict = 3,
    CmdInfo = 4,
    CmdHelp = 5,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[repr(u8)]
pub enum PrivilegeLevel {
    Unprivileged = 0,
    User = 1,
    Elevated = 2,
    Admin = 3,
}

#[derive(Clone, Debug)]
pub struct TDataPoint {
    pub input: Vec<f64>,
    pub target: Vec<f64>,
}

#[derive(Clone, Debug)]
pub struct LayerData {
    pub weights: Vec<f64>,
    pub biases: Vec<f64>,
    pub outputs: Vec<f64>,
    pub errors: Vec<f64>,
    pub num_neurons: usize,
    pub num_inputs: usize,
    pub activation_type: TActivationType,
}

impl LayerData {
    pub fn new(num_neurons: usize, num_inputs: usize, activation_type: TActivationType) -> Self {
        LayerData {
            weights: vec![0.0; num_neurons * num_inputs],
            biases: vec![0.0; num_neurons],
            outputs: vec![0.0; num_neurons],
            errors: vec![0.0; num_neurons],
            num_neurons,
            num_inputs,
            activation_type,
        }
    }
    
    pub fn get_weight(&self, neuron_idx: usize, weight_idx: usize) -> Option<f64> {
        if neuron_idx >= self.num_neurons || weight_idx >= self.num_inputs {
            return None;
        }
        Some(self.weights[neuron_idx * self.num_inputs + weight_idx])
    }
    
    pub fn set_weight(&mut self, neuron_idx: usize, weight_idx: usize, value: f64) -> Option<()> {
        if neuron_idx >= self.num_neurons || weight_idx >= self.num_inputs {
            return None;
        }
        self.weights[neuron_idx * self.num_inputs + weight_idx] = value;
        Some(())
    }
    
    pub fn get_bias(&self, neuron_idx: usize) -> Option<f64> {
        if neuron_idx >= self.num_neurons {
            return None;
        }
        Some(self.biases[neuron_idx])
    }
    
    pub fn get_output(&self, neuron_idx: usize) -> Option<f64> {
        if neuron_idx >= self.num_neurons {
            return None;
        }
        Some(self.outputs[neuron_idx])
    }
}

#[derive(Clone, Debug)]
pub struct MLP {
    pub layers: Vec<LayerData>,
    pub num_layers: usize,
    pub input_size: usize,
    pub output_size: usize,
    pub learning_rate: f64,
    pub dropout_rate: f64,
    pub l2_lambda: f64,
    pub beta1: f64,
    pub beta2: f64,
    pub timestep: i32,
    pub optimizer: TOptimizerType,
    pub hidden_activation: TActivationType,
    pub output_activation: TActivationType,
}

impl MLP {
    pub fn new(
        input_size: usize,
        hidden_sizes: &[usize],
        output_size: usize,
    ) -> Option<Self> {
        if input_size == 0 || output_size == 0 {
            return None;
        }
        if hidden_sizes.len() > MAX_LAYERS - 2 {
            return None;
        }
        
        let mut layers = Vec::new();
        layers.push(LayerData::new(input_size + 1, input_size, TActivationType::AtSigmoid));
        
        let mut prev_size = input_size;
        for &hidden_size in hidden_sizes {
            if hidden_size == 0 || hidden_size > MAX_NEURONS_PER_LAYER {
                return None;
            }
            layers.push(LayerData::new(hidden_size + 1, prev_size + 1, TActivationType::AtReLU));
            prev_size = hidden_size;
        }
        
        layers.push(LayerData::new(output_size, prev_size + 1, TActivationType::AtSigmoid));
        
        Some(MLP {
            num_layers: layers.len(),
            layers,
            input_size,
            output_size,
            learning_rate: 0.1,
            dropout_rate: 0.0,
            l2_lambda: 0.0,
            beta1: 0.9,
            beta2: 0.999,
            timestep: 0,
            optimizer: TOptimizerType::OtSGD,
            hidden_activation: TActivationType::AtReLU,
            output_activation: TActivationType::AtSigmoid,
        })
    }
    
    pub fn get_layer(&self, layer_idx: usize) -> Option<&LayerData> {
        self.layers.get(layer_idx)
    }
    
    pub fn get_layer_mut(&mut self, layer_idx: usize) -> Option<&mut LayerData> {
        self.layers.get_mut(layer_idx)
    }
}

pub fn activation_to_str(act: TActivationType) -> &'static str {
    match act {
        TActivationType::AtSigmoid => "sigmoid",
        TActivationType::AtTanh => "tanh",
        TActivationType::AtReLU => "relu",
        TActivationType::AtSoftmax => "softmax",
    }
}

pub fn optimizer_to_str(opt: TOptimizerType) -> &'static str {
    match opt {
        TOptimizerType::OtSGD => "sgd",
        TOptimizerType::OtAdam => "adam",
        TOptimizerType::OtRMSProp => "rmsprop",
    }
}

pub fn parse_activation(s: &str) -> TActivationType {
    match s.to_lowercase().as_str() {
        "tanh" => TActivationType::AtTanh,
        "relu" => TActivationType::AtReLU,
        "softmax" => TActivationType::AtSoftmax,
        _ => TActivationType::AtSigmoid,
    }
}

pub fn parse_optimizer(s: &str) -> TOptimizerType {
    match s.to_lowercase().as_str() {
        "adam" => TOptimizerType::OtAdam,
        "rmsprop" => TOptimizerType::OtRMSProp,
        _ => TOptimizerType::OtSGD,
    }
}

pub fn sigmoid(x: f64) -> f64 {
    if x < -500.0 {
        0.0
    } else if x > 500.0 {
        1.0
    } else {
        1.0 / (1.0 + (-x).exp())
    }
}

pub fn d_sigmoid(x: f64) -> f64 {
    x * (1.0 - x)
}

pub fn tanh_activation(x: f64) -> f64 {
    x.tanh()
}

pub fn d_tanh(x: f64) -> f64 {
    1.0 - (x * x)
}

pub fn relu(x: f64) -> f64 {
    if x > 0.0 { x } else { 0.0 }
}

pub fn d_relu(x: f64) -> f64 {
    if x > 0.0 { 1.0 } else { 0.0 }
}

pub fn safe_add(a: i64, b: i64) -> Option<i64> {
    a.checked_add(b)
}

pub fn safe_sub(a: i64, b: i64) -> Option<i64> {
    a.checked_sub(b)
}

pub fn safe_mul(a: i64, b: i64) -> Option<i64> {
    a.checked_mul(b)
}

pub fn safe_div(a: i64, b: i64) -> Option<i64> {
    if b == 0 {
        None
    } else {
        a.checked_div(b)
    }
}

pub fn max_index(arr: &[f64]) -> Option<usize> {
    if arr.is_empty() {
        return None;
    }
    arr.iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i)
}

pub fn validate_bounds<T>(arr: &[T], index: usize) -> Option<&T> {
    arr.get(index)
}

pub fn validate_bounds_mut<T>(arr: &mut [T], index: usize) -> Option<&mut T> {
    arr.get_mut(index)
}

pub fn compute_loss_checked(predicted: &[f64], target: &[f64], is_softmax: bool) -> Option<f64> {
    if predicted.len() != target.len() || predicted.is_empty() {
        return None;
    }
    
    let mut result = 0.0;
    for i in 0..predicted.len() {
        if is_softmax {
            let p = predicted[i].max(EPSILON).min(1.0 - EPSILON);
            if p.is_nan() || p.is_infinite() {
                return None;
            }
            result -= target[i] * p.ln();
        } else {
            let diff = target[i] - predicted[i];
            result += 0.5 * diff * diff;
        }
        if result.is_nan() || result.is_infinite() {
            return None;
        }
    }
    Some(result)
}

pub fn bounded_loop<F>(max_iterations: usize, mut f: F) -> bool 
where
    F: FnMut(usize) -> bool,
{
    for i in 0..max_iterations {
        if !f(i) {
            return false;
        }
    }
    true
}

pub fn allocate_with_limit(size: usize) -> Option<Vec<f64>> {
    let required_memory = size * std::mem::size_of::<f64>();
    if required_memory > MEMORY_BUDGET {
        return None;
    }
    Some(vec![0.0; size])
}

pub fn check_privilege_transition(current: PrivilegeLevel, target: PrivilegeLevel) -> bool {
    match (current, target) {
        (PrivilegeLevel::Unprivileged, PrivilegeLevel::Unprivileged) => true,
        (PrivilegeLevel::Unprivileged, _) => false,
        (PrivilegeLevel::User, PrivilegeLevel::Unprivileged) => true,
        (PrivilegeLevel::User, PrivilegeLevel::User) => true,
        (PrivilegeLevel::User, _) => false,
        (PrivilegeLevel::Elevated, PrivilegeLevel::Unprivileged) => true,
        (PrivilegeLevel::Elevated, PrivilegeLevel::User) => true,
        (PrivilegeLevel::Elevated, PrivilegeLevel::Elevated) => true,
        (PrivilegeLevel::Elevated, PrivilegeLevel::Admin) => false,
        (PrivilegeLevel::Admin, _) => true,
    }
}

pub fn is_fp_sane(value: f64) -> bool {
    !value.is_nan() && !value.is_infinite()
}

pub fn clamp_fp(value: f64, min: f64, max: f64) -> Option<f64> {
    if value.is_nan() || value.is_infinite() {
        return None;
    }
    Some(value.max(min).min(max))
}
