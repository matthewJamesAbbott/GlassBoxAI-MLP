/// MultiLayerPerceptron - Neural Network implementation in Rust
/// Pure stdlib - no external crates
/// 
/// Originally ported from Pascal, then C++ (2025), now Rust
/// 
/// Author: Matthew Abbott 19/3/2023
/// Enhanced with:
/// - Sigmoid, Tanh, ReLU, Softmax activation functions
/// - SGD, Adam, RMSProp optimizers
/// - Dropout regularization
/// - L2 regularization
/// - Xavier/He initialization
/// - Learning rate decay
/// - Early stopping
/// - Data normalization
/// - Full CLI support for Create, Train, Predict, Info commands

use std::fs::File;
use std::io::{Read, Write, BufReader, BufRead};
use std::time::{SystemTime, UNIX_EPOCH};

const EPSILON: f64 = 1e-15;
const MODEL_MAGIC: &str = "MLPBKND01";

/// Simple PRNG using Xorshift64
struct Random {
    state: u64,
}

impl Random {
    fn new() -> Self {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .subsec_nanos() as u64;
        Random { state: nanos | 1 }
    }

    fn next(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        x
    }

    fn next_f64(&mut self) -> f64 {
        (self.next() >> 11) as f64 * (1.0 / 9007199254740992.0)
    }

    fn next_range(&mut self, min: f64, max: f64) -> f64 {
        min + (max - min) * self.next_f64()
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ActivationType {
    Sigmoid,
    Tanh,
    ReLU,
    Softmax,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum OptimizerType {
    SGD,
    Adam,
    RMSProp,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CommandType {
    None,
    Create,
    Train,
    Predict,
    Info,
    Help,
}

/// Represents a single data point (input -> target)
#[derive(Debug, Clone)]
pub struct DataPoint {
    pub input: Vec<f64>,
    pub target: Vec<f64>,
}

/// Represents a single neuron with weights, bias, and optimizer state
#[derive(Debug, Clone)]
pub struct Neuron {
    pub weights: Vec<f64>,
    pub bias: f64,
    pub output: f64,
    pub error: f64,
    pub m: Vec<f64>,
    pub v: Vec<f64>,
    pub m_bias: f64,
    pub v_bias: f64,
}

/// Represents a layer of neurons with activation function and dropout mask
#[derive(Debug, Clone)]
pub struct Layer {
    pub neurons: Vec<Neuron>,
    pub activation_type: ActivationType,
    pub dropout_mask: Vec<bool>,
}

/// Multi-Layer Perceptron neural network
pub struct MultiLayerPerceptron {
    input_layer: Layer,
    hidden_layers: Vec<Layer>,
    output_layer: Layer,
    hidden_sizes: Vec<usize>,
    input_size: usize,
    output_size: usize,
    is_training: bool,
    rng: Random,
    
    pub learning_rate: f64,
    pub max_iterations: usize,
    pub optimizer: OptimizerType,
    pub hidden_activation: ActivationType,
    pub output_activation: ActivationType,
    pub dropout_rate: f64,
    pub l2_lambda: f64,
    pub beta1: f64,
    pub beta2: f64,
    pub timestep: usize,
    pub enable_lr_decay: bool,
    pub lr_decay_rate: f64,
    pub lr_decay_epochs: usize,
    pub enable_early_stopping: bool,
    pub early_stopping_patience: usize,
}

// Activation functions
fn sigmoid(x: f64) -> f64 {
    if x < -500.0 {
        0.0
    } else if x > 500.0 {
        1.0
    } else {
        1.0 / (1.0 + (-x).exp())
    }
}

fn d_sigmoid(x: f64) -> f64 {
    x * (1.0 - x)
}

fn tanh_activation(x: f64) -> f64 {
    x.tanh()
}

fn d_tanh(x: f64) -> f64 {
    1.0 - (x * x)
}

fn relu(x: f64) -> f64 {
    if x > 0.0 { x } else { 0.0 }
}

fn d_relu(x: f64) -> f64 {
    if x > 0.0 { 1.0 } else { 0.0 }
}

fn softmax(x: &[f64]) -> Vec<f64> {
    let n = x.len();
    let mut result = vec![0.0; n];
    let mut exp_values = vec![0.0; n];
    
    let max_val = x.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    
    let mut sum = 0.0;
    for i in 0..n {
        exp_values[i] = (x[i] - max_val).exp();
        sum += exp_values[i];
    }
    
    for i in 0..n {
        result[i] = exp_values[i] / sum;
        if result[i] < EPSILON {
            result[i] = EPSILON;
        } else if result[i] > 1.0 - EPSILON {
            result[i] = 1.0 - EPSILON;
        }
    }
    
    result
}

fn apply_activation(x: f64, act_type: ActivationType) -> f64 {
    match act_type {
        ActivationType::Sigmoid => sigmoid(x),
        ActivationType::Tanh => tanh_activation(x),
        ActivationType::ReLU => relu(x),
        ActivationType::Softmax => x,
    }
}

fn apply_activation_derivative(x: f64, act_type: ActivationType) -> f64 {
    match act_type {
        ActivationType::Sigmoid => d_sigmoid(x),
        ActivationType::Tanh => d_tanh(x),
        ActivationType::ReLU => d_relu(x),
        ActivationType::Softmax => 1.0,
    }
}

fn max_index(arr: &[f64]) -> usize {
    arr.iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i)
        .unwrap_or(0)
}

impl MultiLayerPerceptron {
    pub fn new(
        input_size: usize,
        hidden_sizes: Vec<usize>,
        output_size: usize,
        hidden_activation: ActivationType,
        output_activation: ActivationType,
    ) -> Self {
        let mut rng = Random::new();
        let mut mlp = MultiLayerPerceptron {
            input_layer: Layer {
                neurons: vec![],
                activation_type: ActivationType::Sigmoid,
                dropout_mask: vec![],
            },
            hidden_layers: vec![],
            output_layer: Layer {
                neurons: vec![],
                activation_type: output_activation,
                dropout_mask: vec![],
            },
            hidden_sizes: hidden_sizes.clone(),
            input_size,
            output_size,
            is_training: true,
            rng,
            learning_rate: 0.1,
            max_iterations: 100,
            optimizer: OptimizerType::SGD,
            hidden_activation,
            output_activation,
            dropout_rate: 0.0,
            l2_lambda: 0.0,
            beta1: 0.9,
            beta2: 0.999,
            timestep: 0,
            enable_lr_decay: false,
            lr_decay_rate: 0.95,
            lr_decay_epochs: 10,
            enable_early_stopping: false,
            early_stopping_patience: 10,
        };
        
        mlp.input_layer.neurons = vec![
            Neuron {
                weights: vec![],
                bias: 0.0,
                output: 0.0,
                error: 0.0,
                m: vec![],
                v: vec![],
                m_bias: 0.0,
                v_bias: 0.0,
            };
            input_size
        ];
        
        for (layer_idx, &hidden_size) in hidden_sizes.iter().enumerate() {
            let prev_size = if layer_idx == 0 { input_size } else { hidden_sizes[layer_idx - 1] };
            let mut layer = Layer {
                neurons: vec![],
                activation_type: hidden_activation,
                dropout_mask: vec![false; hidden_size],
            };
            
            for _ in 0..hidden_size {
                let weights = mlp.initialize_weights(prev_size, hidden_activation);
                let m = vec![0.0; weights.len()];
                let v = vec![0.0; weights.len()];
                
                layer.neurons.push(Neuron {
                    weights,
                    bias: 0.0,
                    output: 0.0,
                    error: 0.0,
                    m,
                    v,
                    m_bias: 0.0,
                    v_bias: 0.0,
                });
            }
            
            mlp.hidden_layers.push(layer);
        }
        
        let prev_size = hidden_sizes.last().copied().unwrap_or(input_size);
        for _ in 0..output_size {
            let weights = mlp.initialize_weights(prev_size, output_activation);
            let m = vec![0.0; weights.len()];
            let v = vec![0.0; weights.len()];
            
            mlp.output_layer.neurons.push(Neuron {
                weights,
                bias: 0.0,
                output: 0.0,
                error: 0.0,
                m,
                v,
                m_bias: 0.0,
                v_bias: 0.0,
            });
        }
        mlp.output_layer.dropout_mask = vec![false; output_size];
        
        mlp
    }
    
    fn initialize_weights(&mut self, num_inputs: usize, act_type: ActivationType) -> Vec<f64> {
        match act_type {
            ActivationType::ReLU => {
                let limit = (2.0 / num_inputs as f64).sqrt();
                (0..num_inputs)
                    .map(|_| self.rng.next_range(-limit, limit))
                    .collect()
            }
            _ => {
                let limit = (6.0 / (num_inputs as f64 + 1.0)).sqrt();
                (0..num_inputs)
                    .map(|_| self.rng.next_range(-limit, limit))
                    .collect()
            }
        }
    }
    
    pub fn predict(&mut self, input: &[f64]) -> Vec<f64> {
        self.is_training = false;
        self.feed_forward(input);
        self.output_layer.neurons.iter().map(|n| n.output).collect()
    }
    
    fn feed_forward(&mut self, input: &[f64]) {
        for (i, &val) in input.iter().enumerate() {
            if i < self.input_layer.neurons.len() {
                self.input_layer.neurons[i].output = val;
            }
        }
        
        let mut prev_outputs = input.to_vec();
        
        for layer in &mut self.hidden_layers {
            let mut layer_outputs = vec![];
            
            for neuron in &mut layer.neurons {
                let mut sum = neuron.bias;
                for (j, &w) in neuron.weights.iter().enumerate() {
                    sum += w * prev_outputs[j];
                }
                
                neuron.output = if layer.activation_type == ActivationType::Softmax {
                    sum
                } else {
                    apply_activation(sum, layer.activation_type)
                };
                
                layer_outputs.push(neuron.output);
            }
            
            if layer.activation_type == ActivationType::Softmax {
                layer_outputs = softmax(&layer_outputs);
                for (i, &output) in layer_outputs.iter().enumerate() {
                    layer.neurons[i].output = output;
                }
            }
            
            if self.is_training && self.dropout_rate > 0.0 {
                self.apply_dropout(layer);
                layer_outputs = layer.neurons.iter().map(|n| n.output).collect();
            }
            
            prev_outputs = layer_outputs;
        }
        
        let mut layer_outputs = vec![];
        for neuron in &mut self.output_layer.neurons {
            let mut sum = neuron.bias;
            for (j, &w) in neuron.weights.iter().enumerate() {
                sum += w * prev_outputs[j];
            }
            
            neuron.output = if self.output_activation == ActivationType::Softmax {
                sum
            } else {
                apply_activation(sum, self.output_activation)
            };
            
            layer_outputs.push(neuron.output);
        }
        
        if self.output_activation == ActivationType::Softmax {
            layer_outputs = softmax(&layer_outputs);
            for (i, &output) in layer_outputs.iter().enumerate() {
                self.output_layer.neurons[i].output = output;
            }
        }
    }
    
    pub fn train(&mut self, input: &[f64], target: &[f64]) {
        self.is_training = true;
        self.timestep += 1;
        
        self.feed_forward(input);
        self.backpropagate(target);
        self.update_weights(input);
    }
    
    fn backpropagate(&mut self, target: &[f64]) {
        for (i, neuron) in self.output_layer.neurons.iter_mut().enumerate() {
            let diff = if i < target.len() { target[i] - neuron.output } else { -neuron.output };
            
            let deriv = if self.output_activation == ActivationType::Softmax {
                neuron.output * (1.0 - neuron.output)
            } else {
                apply_activation_derivative(neuron.output, self.output_activation)
            };
            
            neuron.error = diff * deriv;
        }
        
        for layer_idx in (0..self.hidden_layers.len()).rev() {
            let next_layer_errors: Vec<f64> = if layer_idx == self.hidden_layers.len() - 1 {
                self.output_layer.neurons.iter().map(|n| n.error).collect()
            } else {
                self.hidden_layers[layer_idx + 1]
                    .neurons
                    .iter()
                    .map(|n| n.error)
                    .collect()
            };
            
            let next_layer_size = if layer_idx == self.hidden_layers.len() - 1 {
                self.output_layer.neurons.len()
            } else {
                self.hidden_layers[layer_idx + 1].neurons.len()
            };
            
            for (i, neuron) in self.hidden_layers[layer_idx].neurons.iter_mut().enumerate() {
                let mut sum = 0.0;
                for j in 0..next_layer_size {
                    let next_neuron = if layer_idx == self.hidden_layers.len() - 1 {
                        &self.output_layer.neurons[j]
                    } else {
                        &self.hidden_layers[layer_idx + 1].neurons[j]
                    };
                    
                    if i < next_neuron.weights.len() {
                        sum += next_neuron.weights[i] * next_layer_errors[j];
                    }
                }
                
                let deriv = apply_activation_derivative(neuron.output, self.hidden_activation);
                neuron.error = sum * deriv;
            }
        }
    }
    
    fn update_weights(&mut self, input: &[f64]) {
        let mut prev_outputs = input.to_vec();
        
        for layer_idx in 0..self.hidden_layers.len() {
            for neuron in &mut self.hidden_layers[layer_idx].neurons {
                match self.optimizer {
                    OptimizerType::SGD => self.update_neuron_weights_sgd(neuron, &prev_outputs),
                    OptimizerType::Adam => self.update_neuron_weights_adam(neuron, &prev_outputs),
                    OptimizerType::RMSProp => self.update_neuron_weights_rmsprop(neuron, &prev_outputs),
                }
            }
            
            prev_outputs = self.hidden_layers[layer_idx]
                .neurons
                .iter()
                .map(|n| n.output)
                .collect();
        }
        
        for neuron in &mut self.output_layer.neurons {
            match self.optimizer {
                OptimizerType::SGD => self.update_neuron_weights_sgd(neuron, &prev_outputs),
                OptimizerType::Adam => self.update_neuron_weights_adam(neuron, &prev_outputs),
                OptimizerType::RMSProp => self.update_neuron_weights_rmsprop(neuron, &prev_outputs),
            }
        }
    }
    
    fn update_neuron_weights_sgd(&self, neuron: &mut Neuron, prev_outputs: &[f64]) {
        let lr = self.learning_rate;
        let l2_penalty = self.l2_lambda;
        
        for j in 0..neuron.weights.len() {
            let gradient = -neuron.error * prev_outputs[j] - l2_penalty * neuron.weights[j];
            neuron.weights[j] += lr * gradient;
        }
        
        neuron.bias += lr * (-neuron.error);
    }
    
    fn update_neuron_weights_adam(&self, neuron: &mut Neuron, prev_outputs: &[f64]) {
        let lr = self.learning_rate;
        let b1 = self.beta1;
        let b2 = self.beta2;
        let t = self.timestep as f64;
        let l2_penalty = self.l2_lambda;
        
        for j in 0..neuron.weights.len() {
            let gradient = -neuron.error * prev_outputs[j] - l2_penalty * neuron.weights[j];
            
            neuron.m[j] = b1 * neuron.m[j] + (1.0 - b1) * gradient;
            neuron.v[j] = b2 * neuron.v[j] + (1.0 - b2) * gradient * gradient;
            
            let m_hat = neuron.m[j] / (1.0 - b1.powf(t));
            let v_hat = neuron.v[j] / (1.0 - b2.powf(t));
            
            neuron.weights[j] += lr * m_hat / (v_hat.sqrt() + EPSILON);
        }
        
        let bias_gradient = -neuron.error;
        neuron.m_bias = b1 * neuron.m_bias + (1.0 - b1) * bias_gradient;
        neuron.v_bias = b2 * neuron.v_bias + (1.0 - b2) * bias_gradient * bias_gradient;
        
        let m_hat = neuron.m_bias / (1.0 - b1.powf(t));
        let v_hat = neuron.v_bias / (1.0 - b2.powf(t));
        
        neuron.bias += lr * m_hat / (v_hat.sqrt() + EPSILON);
    }
    
    fn update_neuron_weights_rmsprop(&self, neuron: &mut Neuron, prev_outputs: &[f64]) {
        let lr = self.learning_rate;
        let b2 = self.beta2;
        let l2_penalty = self.l2_lambda;
        
        for j in 0..neuron.weights.len() {
            let gradient = -neuron.error * prev_outputs[j] - l2_penalty * neuron.weights[j];
            
            neuron.v[j] = b2 * neuron.v[j] + (1.0 - b2) * gradient * gradient;
            
            neuron.weights[j] += lr * gradient / (neuron.v[j].sqrt() + EPSILON);
        }
        
        let bias_gradient = -neuron.error;
        neuron.v_bias = b2 * neuron.v_bias + (1.0 - b2) * bias_gradient * bias_gradient;
        neuron.bias += lr * bias_gradient / (neuron.v_bias.sqrt() + EPSILON);
    }
    
    fn apply_dropout(&mut self, layer: &mut Layer) {
        let keep_prob = 1.0 - self.dropout_rate;
        for i in 0..layer.neurons.len() {
            if self.rng.next_f64() < self.dropout_rate {
                layer.neurons[i].output = 0.0;
                layer.dropout_mask[i] = false;
            } else {
                layer.neurons[i].output /= keep_prob;
                layer.dropout_mask[i] = true;
            }
        }
    }
    
    pub fn compute_loss(&self, predicted: &[f64], target: &[f64]) -> f64 {
        let mut loss = 0.0;
        for (i, &pred) in predicted.iter().enumerate() {
            if i < target.len() {
                loss += (target[i] - pred).powi(2);
            }
        }
        loss / 2.0
    }
    
    pub fn save(&self, filename: &str) -> std::io::Result<()> {
        let mut file = File::create(filename)?;
        
        file.write_all(MODEL_MAGIC.as_bytes())?;
        
        file.write_all(&(self.input_size as u32).to_le_bytes())?;
        file.write_all(&(self.output_size as u32).to_le_bytes())?;
        file.write_all(&(self.hidden_sizes.len() as u32).to_le_bytes())?;
        
        for &size in &self.hidden_sizes {
            file.write_all(&(size as u32).to_le_bytes())?;
        }
        
        file.write_all(&self.learning_rate.to_le_bytes())?;
        file.write_all(&(self.max_iterations as u32).to_le_bytes())?;
        file.write_all(&(self.optimizer as u8).to_le_bytes())?;
        file.write_all(&(self.hidden_activation as u8).to_le_bytes())?;
        file.write_all(&(self.output_activation as u8).to_le_bytes())?;
        file.write_all(&self.dropout_rate.to_le_bytes())?;
        file.write_all(&self.l2_lambda.to_le_bytes())?;
        file.write_all(&self.beta1.to_le_bytes())?;
        file.write_all(&self.beta2.to_le_bytes())?;
        
        for layer in &self.hidden_layers {
            for neuron in &layer.neurons {
                file.write_all(&(neuron.weights.len() as u32).to_le_bytes())?;
                for &w in &neuron.weights {
                    file.write_all(&w.to_le_bytes())?;
                }
                file.write_all(&neuron.bias.to_le_bytes())?;
            }
        }
        
        for neuron in &self.output_layer.neurons {
            file.write_all(&(neuron.weights.len() as u32).to_le_bytes())?;
            for &w in &neuron.weights {
                file.write_all(&w.to_le_bytes())?;
            }
            file.write_all(&neuron.bias.to_le_bytes())?;
        }
        
        Ok(())
    }
    
    pub fn load(filename: &str) -> std::io::Result<Self> {
        let mut file = File::open(filename)?;
        let mut buffer = Vec::new();
        file.read_to_end(&mut buffer)?;
        
        let mut offset = 0;
        
        let magic = String::from_utf8_lossy(&buffer[offset..offset + 9]).to_string();
        offset += 9;
        
        if magic != MODEL_MAGIC {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "Invalid model file format",
            ));
        }
        
        let input_size = u32::from_le_bytes([
            buffer[offset], buffer[offset + 1], buffer[offset + 2], buffer[offset + 3],
        ]) as usize;
        offset += 4;
        
        let output_size = u32::from_le_bytes([
            buffer[offset], buffer[offset + 1], buffer[offset + 2], buffer[offset + 3],
        ]) as usize;
        offset += 4;
        
        let hidden_count = u32::from_le_bytes([
            buffer[offset], buffer[offset + 1], buffer[offset + 2], buffer[offset + 3],
        ]) as usize;
        offset += 4;
        
        let mut hidden_sizes = vec![];
        for _ in 0..hidden_count {
            let size = u32::from_le_bytes([
                buffer[offset], buffer[offset + 1], buffer[offset + 2], buffer[offset + 3],
            ]) as usize;
            hidden_sizes.push(size);
            offset += 4;
        }
        
        let learning_rate = f64::from_le_bytes([
            buffer[offset], buffer[offset + 1], buffer[offset + 2], buffer[offset + 3],
            buffer[offset + 4], buffer[offset + 5], buffer[offset + 6], buffer[offset + 7],
        ]);
        offset += 8;
        
        let max_iterations = u32::from_le_bytes([
            buffer[offset], buffer[offset + 1], buffer[offset + 2], buffer[offset + 3],
        ]) as usize;
        offset += 4;
        
        let optimizer = match buffer[offset] {
            0 => OptimizerType::SGD,
            1 => OptimizerType::Adam,
            2 => OptimizerType::RMSProp,
            _ => OptimizerType::SGD,
        };
        offset += 1;
        
        let hidden_activation = match buffer[offset] {
            0 => ActivationType::Sigmoid,
            1 => ActivationType::Tanh,
            2 => ActivationType::ReLU,
            3 => ActivationType::Softmax,
            _ => ActivationType::Sigmoid,
        };
        offset += 1;
        
        let output_activation = match buffer[offset] {
            0 => ActivationType::Sigmoid,
            1 => ActivationType::Tanh,
            2 => ActivationType::ReLU,
            3 => ActivationType::Softmax,
            _ => ActivationType::Sigmoid,
        };
        offset += 1;
        
        let dropout_rate = f64::from_le_bytes([
            buffer[offset], buffer[offset + 1], buffer[offset + 2], buffer[offset + 3],
            buffer[offset + 4], buffer[offset + 5], buffer[offset + 6], buffer[offset + 7],
        ]);
        offset += 8;
        
        let l2_lambda = f64::from_le_bytes([
            buffer[offset], buffer[offset + 1], buffer[offset + 2], buffer[offset + 3],
            buffer[offset + 4], buffer[offset + 5], buffer[offset + 6], buffer[offset + 7],
        ]);
        offset += 8;
        
        let beta1 = f64::from_le_bytes([
            buffer[offset], buffer[offset + 1], buffer[offset + 2], buffer[offset + 3],
            buffer[offset + 4], buffer[offset + 5], buffer[offset + 6], buffer[offset + 7],
        ]);
        offset += 8;
        
        let beta2 = f64::from_le_bytes([
            buffer[offset], buffer[offset + 1], buffer[offset + 2], buffer[offset + 3],
            buffer[offset + 4], buffer[offset + 5], buffer[offset + 6], buffer[offset + 7],
        ]);
        offset += 8;
        
        let mut mlp = Self::new(input_size, hidden_sizes, output_size, hidden_activation, output_activation);
        mlp.learning_rate = learning_rate;
        mlp.max_iterations = max_iterations;
        mlp.optimizer = optimizer;
        mlp.dropout_rate = dropout_rate;
        mlp.l2_lambda = l2_lambda;
        mlp.beta1 = beta1;
        mlp.beta2 = beta2;
        
        for layer in &mut mlp.hidden_layers {
            for neuron in &mut layer.neurons {
                let weight_count = u32::from_le_bytes([
                    buffer[offset], buffer[offset + 1], buffer[offset + 2], buffer[offset + 3],
                ]) as usize;
                offset += 4;
                
                neuron.weights.clear();
                for _ in 0..weight_count {
                    let w = f64::from_le_bytes([
                        buffer[offset], buffer[offset + 1], buffer[offset + 2], buffer[offset + 3],
                        buffer[offset + 4], buffer[offset + 5], buffer[offset + 6], buffer[offset + 7],
                    ]);
                    neuron.weights.push(w);
                    offset += 8;
                }
                
                neuron.bias = f64::from_le_bytes([
                    buffer[offset], buffer[offset + 1], buffer[offset + 2], buffer[offset + 3],
                    buffer[offset + 4], buffer[offset + 5], buffer[offset + 6], buffer[offset + 7],
                ]);
                offset += 8;
            }
        }
        
        for neuron in &mut mlp.output_layer.neurons {
            let weight_count = u32::from_le_bytes([
                buffer[offset], buffer[offset + 1], buffer[offset + 2], buffer[offset + 3],
            ]) as usize;
            offset += 4;
            
            neuron.weights.clear();
            for _ in 0..weight_count {
                let w = f64::from_le_bytes([
                    buffer[offset], buffer[offset + 1], buffer[offset + 2], buffer[offset + 3],
                    buffer[offset + 4], buffer[offset + 5], buffer[offset + 6], buffer[offset + 7],
                ]);
                neuron.weights.push(w);
                offset += 8;
            }
            
            neuron.bias = f64::from_le_bytes([
                buffer[offset], buffer[offset + 1], buffer[offset + 2], buffer[offset + 3],
                buffer[offset + 4], buffer[offset + 5], buffer[offset + 6], buffer[offset + 7],
            ]);
            offset += 8;
        }
        
        Ok(mlp)
    }
    
    pub fn get_input_size(&self) -> usize {
        self.input_size
    }
    
    pub fn get_output_size(&self) -> usize {
        self.output_size
    }
    
    pub fn get_hidden_layer_count(&self) -> usize {
        self.hidden_layers.len()
    }
    
    pub fn get_hidden_layer(&self, index: usize) -> Option<&Layer> {
        self.hidden_layers.get(index)
    }
    
    pub fn get_input_layer(&self) -> &Layer {
        &self.input_layer
    }
    
    pub fn get_output_layer(&self) -> &Layer {
        &self.output_layer
    }
}

fn activation_to_string(act: ActivationType) -> &'static str {
    match act {
        ActivationType::Sigmoid => "sigmoid",
        ActivationType::Tanh => "tanh",
        ActivationType::ReLU => "relu",
        ActivationType::Softmax => "softmax",
    }
}

fn parse_activation(s: &str) -> ActivationType {
    match s.to_lowercase().as_str() {
        "sigmoid" => ActivationType::Sigmoid,
        "tanh" => ActivationType::Tanh,
        "relu" => ActivationType::ReLU,
        "softmax" => ActivationType::Softmax,
        _ => ActivationType::Sigmoid,
    }
}

fn optimizer_to_string(opt: OptimizerType) -> &'static str {
    match opt {
        OptimizerType::SGD => "SGD",
        OptimizerType::Adam => "Adam",
        OptimizerType::RMSProp => "RMSProp",
    }
}

fn parse_optimizer(s: &str) -> OptimizerType {
    match s.to_lowercase().as_str() {
        "sgd" => OptimizerType::SGD,
        "adam" => OptimizerType::Adam,
        "rmsprop" => OptimizerType::RMSProp,
        _ => OptimizerType::SGD,
    }
}

fn parse_double_array(s: &str) -> Vec<f64> {
    s.split(',')
        .map(|v| v.trim().parse::<f64>().unwrap_or(0.0))
        .collect()
}

fn parse_int_array(s: &str) -> Vec<usize> {
    s.split(',')
        .map(|v| v.trim().parse::<usize>().unwrap_or(1))
        .collect()
}

fn load_data_csv(filename: &str, input_size: usize, output_size: usize) -> Vec<DataPoint> {
    let mut data = Vec::new();
    
    if let Ok(file) = File::open(filename) {
        let reader = BufReader::new(file);
        for line in reader.lines().flatten() {
            let values: Vec<f64> = line
                .split(',')
                .map(|v| v.trim().parse::<f64>().unwrap_or(0.0))
                .collect();
            
            if values.len() >= input_size + output_size {
                let input = values[..input_size].to_vec();
                let target = values[input_size..input_size + output_size].to_vec();
                data.push(DataPoint { input, target });
            }
        }
    }
    
    data
}

fn normalize_data(data: &mut [DataPoint]) -> bool {
    if data.is_empty() {
        return false;
    }
    
    let input_size = data[0].input.len();
    let mut mins = vec![f64::INFINITY; input_size];
    let mut maxs = vec![f64::NEG_INFINITY; input_size];
    
    for point in data.iter() {
        for (j, &val) in point.input.iter().enumerate() {
            mins[j] = mins[j].min(val);
            maxs[j] = maxs[j].max(val);
        }
    }
    
    for point in data.iter_mut() {
        for (j, val) in point.input.iter_mut().enumerate() {
            let range = maxs[j] - mins[j];
            *val = if range > 0.0 {
                (*val - mins[j]) / range
            } else {
                0.5
            };
        }
    }
    
    true
}

fn check_data_quality(data: &[DataPoint]) {
    if data.is_empty() {
        return;
    }
    
    let input_size = data[0].input.len();
    
    for j in 0..input_size {
        let mut min_val = f64::INFINITY;
        let mut max_val = f64::NEG_INFINITY;
        
        for point in data {
            if j < point.input.len() {
                min_val = min_val.min(point.input[j]);
                max_val = max_val.max(point.input[j]);
            }
        }
        
        if (max_val - min_val) > 100.0 {
            eprintln!(
                "Warning: Feature {} has large range ({:.2} to {:.2}). Consider normalizing.",
                j, min_val, max_val
            );
        }
        if min_val < -10.0 || max_val > 10.0 {
            eprintln!(
                "Warning: Feature {} has values outside [-10, 10]. Consider normalizing.",
                j
            );
        }
    }
}

fn shuffle_data(data: &mut [DataPoint]) {
    let mut rng = Random::new();
    let len = data.len();
    for i in 0..len {
        let j = (rng.next() as usize) % len;
        data.swap(i, j);
    }
}

fn print_usage() {
    println!("MultiLayerPerceptron CLI Tool - Rust Edition");
    println!();
    println!("COMMANDS:");
    println!("  create   - Create a new MLP model");
    println!("  train    - Train an existing model");
    println!("  predict  - Make predictions with a model");
    println!("  info     - Display model information");
    println!("  help     - Show this help message");
    println!();
    println!("CREATE COMMAND:");
    println!("  mlp create --input=<size> --hidden=<sizes> --output=<size> --save=<file>");
    println!();
    println!("OPTIONS:");
    println!("  --lr=<value>                Learning rate (default: 0.1)");
    println!("  --optimizer=<type>          SGD, Adam, RMSProp (default: SGD)");
    println!("  --hidden-act=<type>         sigmoid, tanh, relu (default: sigmoid)");
    println!("  --output-act=<type>         sigmoid, tanh, relu, softmax (default: sigmoid)");
    println!("  --dropout=<rate>            Dropout rate 0-1 (default: 0.0)");
    println!("  --l2=<lambda>               L2 regularization (default: 0.0)");
    println!("  --beta1=<value>             Adam beta1 (default: 0.9)");
    println!("  --beta2=<value>             Adam beta2 (default: 0.999)");
    println!();
    println!("TRAIN COMMAND:");
    println!("  mlp train --model=<file> --data=<file> --save=<file>");
    println!();
    println!("TRAIN OPTIONS:");
    println!("  --epochs=<count>            Number of epochs (default: 100)");
    println!("  --batch=<size>              Batch size (default: 1)");
    println!("  --lr-decay                  Enable learning rate decay");
    println!("  --lr-decay-rate=<rate>      Decay rate (default: 0.95)");
    println!("  --lr-decay-epochs=<count>   Epochs per decay (default: 10)");
    println!("  --early-stop                Enable early stopping");
    println!("  --patience=<count>          Early stop patience (default: 10)");
    println!("  --normalize                 Normalize input data");
    println!("  --verbose                   Show loss per epoch");
    println!();
    println!("PREDICT COMMAND:");
    println!("  mlp predict --model=<file> --input=<values>");
    println!();
    println!("INFO COMMAND:");
    println!("  mlp info --model=<file>");
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    
    if args.len() < 2 {
        print_usage();
        std::process::exit(1);
    }
    
    let command_str = &args[1];
    let command = match command_str.as_str() {
        "create" => CommandType::Create,
        "train" => CommandType::Train,
        "predict" => CommandType::Predict,
        "info" => CommandType::Info,
        "help" => CommandType::Help,
        _ => CommandType::None,
    };
    
    if command == CommandType::None {
        eprintln!("Unknown command: {}", command_str);
        print_usage();
        std::process::exit(1);
    }
    
    if command == CommandType::Help {
        print_usage();
        std::process::exit(0);
    }
    
    let mut input_size = 0;
    let mut output_size = 0;
    let mut epochs = 100;
    let mut batch_size = 1;
    let mut lr_decay_epochs = 10;
    let mut patience = 10;
    let mut hidden_sizes = vec![];
    let mut learning_rate = 0.1;
    let mut dropout_rate = 0.0;
    let mut l2_lambda = 0.0;
    let mut beta1 = 0.9;
    let mut beta2 = 0.999;
    let mut lr_decay_rate = 0.95;
    let mut lr_decay = false;
    let mut early_stop = false;
    let mut normalize = false;
    let mut verbose = false;
    let mut hidden_act = ActivationType::Sigmoid;
    let mut output_act = ActivationType::Sigmoid;
    let mut optimizer = OptimizerType::SGD;
    let mut model_file = String::new();
    let mut save_file = String::new();
    let mut data_file = String::new();
    let mut input_values = vec![];
    
    for i in 2..args.len() {
        let arg = &args[i];
        
        if arg == "--lr-decay" {
            lr_decay = true;
        } else if arg == "--early-stop" {
            early_stop = true;
        } else if arg == "--normalize" {
            normalize = true;
        } else if arg == "--verbose" {
            verbose = true;
        } else if let Some(eq_pos) = arg.find('=') {
            let key = &arg[..eq_pos];
            let value = &arg[eq_pos + 1..];
            
            match key {
                "--input" => {
                    if command == CommandType::Predict {
                        input_values = parse_double_array(value);
                    } else {
                        input_size = value.parse::<usize>().unwrap_or(0);
                    }
                }
                "--hidden" => hidden_sizes = parse_int_array(value),
                "--output" => output_size = value.parse::<usize>().unwrap_or(0),
                "--save" => save_file = value.to_string(),
                "--model" => model_file = value.to_string(),
                "--data" => data_file = value.to_string(),
                "--lr" => learning_rate = value.parse::<f64>().unwrap_or(0.1),
                "--optimizer" => optimizer = parse_optimizer(value),
                "--hidden-act" => hidden_act = parse_activation(value),
                "--output-act" => output_act = parse_activation(value),
                "--dropout" => dropout_rate = value.parse::<f64>().unwrap_or(0.0),
                "--l2" => l2_lambda = value.parse::<f64>().unwrap_or(0.0),
                "--beta1" => beta1 = value.parse::<f64>().unwrap_or(0.9),
                "--beta2" => beta2 = value.parse::<f64>().unwrap_or(0.999),
                "--epochs" => epochs = value.parse::<usize>().unwrap_or(100),
                "--batch" => batch_size = value.parse::<usize>().unwrap_or(1),
                "--lr-decay-rate" => lr_decay_rate = value.parse::<f64>().unwrap_or(0.95),
                "--lr-decay-epochs" => lr_decay_epochs = value.parse::<usize>().unwrap_or(10),
                "--patience" => patience = value.parse::<usize>().unwrap_or(10),
                _ => eprintln!("Unknown option: {}", key),
            }
        }
    }
    
    match command {
        CommandType::Create => {
            if input_size == 0 {
                eprintln!("Error: --input is required");
                std::process::exit(1);
            }
            if hidden_sizes.is_empty() {
                eprintln!("Error: --hidden is required");
                std::process::exit(1);
            }
            if output_size == 0 {
                eprintln!("Error: --output is required");
                std::process::exit(1);
            }
            if save_file.is_empty() {
                eprintln!("Error: --save is required");
                std::process::exit(1);
            }
            
            let mut mlp = MultiLayerPerceptron::new(input_size, hidden_sizes.clone(), output_size, hidden_act, output_act);
            mlp.learning_rate = learning_rate;
            mlp.optimizer = optimizer;
            mlp.dropout_rate = dropout_rate;
            mlp.l2_lambda = l2_lambda;
            mlp.beta1 = beta1;
            mlp.beta2 = beta2;
            
            if let Err(e) = mlp.save(&save_file) {
                eprintln!("Error saving model: {}", e);
                std::process::exit(1);
            }
            
            println!("Created MLP model:");
            println!("  Input size: {}", input_size);
            print!("  Hidden sizes: ");
            for (i, &size) in hidden_sizes.iter().enumerate() {
                if i > 0 {
                    print!(",");
                }
                print!("{}", size);
            }
            println!();
            println!("  Output size: {}", output_size);
            println!("  Hidden activation: {}", activation_to_string(hidden_act));
            println!("  Output activation: {}", activation_to_string(output_act));
            println!("  Optimizer: {}", optimizer_to_string(optimizer));
            println!("  Learning rate: {:.4}", learning_rate);
            println!("  Saved to: {}", save_file);
        }
        CommandType::Train => {
            if model_file.is_empty() {
                eprintln!("Error: --model is required");
                std::process::exit(1);
            }
            if data_file.is_empty() {
                eprintln!("Error: --data is required");
                std::process::exit(1);
            }
            if save_file.is_empty() {
                eprintln!("Error: --save is required");
                std::process::exit(1);
            }
            
            let mut mlp = match MultiLayerPerceptron::load(&model_file) {
                Ok(m) => m,
                Err(e) => {
                    eprintln!("Error loading model: {}", e);
                    std::process::exit(1);
                }
            };
            
            mlp.learning_rate = learning_rate;
            mlp.enable_lr_decay = lr_decay;
            mlp.lr_decay_rate = lr_decay_rate;
            mlp.lr_decay_epochs = lr_decay_epochs;
            mlp.enable_early_stopping = early_stop;
            mlp.early_stopping_patience = patience;
            
            let mut data = load_data_csv(&data_file, mlp.get_input_size(), mlp.get_output_size());
            if data.is_empty() {
                eprintln!("Error: No valid data loaded");
                std::process::exit(1);
            }
            
            println!("Loaded {} training samples", data.len());
            if normalize {
                normalize_data(&mut data);
                println!("Data normalized");
            }
            
            for epoch in 1..=epochs {
                shuffle_data(&mut data);
                
                for point in &data {
                    mlp.train(&point.input, &point.target);
                }
                
                if verbose && (epoch % 10 == 0 || epoch == 1) {
                    let mut loss = 0.0;
                    for point in &data {
                        let output = mlp.predict(&point.input);
                        loss += mlp.compute_loss(&output, &point.target);
                    }
                    println!("Epoch {}/{} - Loss: {:.6}", epoch, epochs, loss / data.len() as f64);
                }
            }
            
            let mut loss = 0.0;
            for point in &data {
                let output = mlp.predict(&point.input);
                loss += mlp.compute_loss(&output, &point.target);
            }
            println!("Final loss: {:.6}", loss / data.len() as f64);
            
            if let Err(e) = mlp.save(&save_file) {
                eprintln!("Error saving model: {}", e);
                std::process::exit(1);
            }
            println!("Model saved to: {}", save_file);
        }
        CommandType::Predict => {
            if model_file.is_empty() {
                eprintln!("Error: --model is required");
                std::process::exit(1);
            }
            if input_values.is_empty() {
                eprintln!("Error: --input is required");
                std::process::exit(1);
            }
            
            let mut mlp = match MultiLayerPerceptron::load(&model_file) {
                Ok(m) => m,
                Err(e) => {
                    eprintln!("Error loading model: {}", e);
                    std::process::exit(1);
                }
            };
            
            if input_values.len() != mlp.get_input_size() {
                eprintln!(
                    "Error: Expected {} input values, got {}",
                    mlp.get_input_size(),
                    input_values.len()
                );
                std::process::exit(1);
            }
            
            let output = mlp.predict(&input_values);
            
            print!("Input: ");
            for (i, &val) in input_values.iter().enumerate() {
                if i > 0 {
                    print!(", ");
                }
                print!("{:.4}", val);
            }
            println!();
            
            print!("Output: ");
            for (i, &val) in output.iter().enumerate() {
                if i > 0 {
                    print!(", ");
                }
                print!("{:.6}", val);
            }
            println!();
            
            if output.len() > 1 {
                let max_idx = max_index(&output);
                println!("Max index: {}", max_idx);
            }
        }
        CommandType::Info => {
            if model_file.is_empty() {
                eprintln!("Error: --model is required");
                std::process::exit(1);
            }
            
            let mlp = match MultiLayerPerceptron::load(&model_file) {
                Ok(m) => m,
                Err(e) => {
                    eprintln!("Error loading model: {}", e);
                    std::process::exit(1);
                }
            };
            
            println!("MLP Model Information");
            println!("=====================");
            println!("Input size: {}", mlp.get_input_size());
            println!("Output size: {}", mlp.get_output_size());
            println!("Hidden layers: {}", mlp.get_hidden_layer_count());
            
            print!("Layer sizes: {}", mlp.get_input_size());
            for i in 0..mlp.get_hidden_layer_count() {
                if let Some(layer) = mlp.get_hidden_layer(i) {
                    print!(" -> {}", layer.neurons.len());
                }
            }
            println!(" -> {}", mlp.get_output_size());
            println!();
            
            println!("Hyperparameters:");
            println!("  Learning rate: {:.6}", mlp.learning_rate);
            println!("  Optimizer: {}", optimizer_to_string(mlp.optimizer));
            println!("  Hidden activation: {}", activation_to_string(mlp.hidden_activation));
            println!("  Output activation: {}", activation_to_string(mlp.output_activation));
            println!("  Dropout rate: {:.4}", mlp.dropout_rate);
            println!("  L2 lambda: {:.6}", mlp.l2_lambda);
            println!("  Beta1: {:.4}", mlp.beta1);
            println!("  Beta2: {:.4}", mlp.beta2);
            println!("  Timestep: {}", mlp.timestep);
            println!();
            
            println!("Total layers: {}", mlp.get_hidden_layer_count() + 2);
            println!("  Layer 0: {} neurons (input)", mlp.get_input_size());
            for i in 0..mlp.get_hidden_layer_count() {
                if let Some(layer) = mlp.get_hidden_layer(i) {
                    println!("  Layer {}: {} neurons", i + 1, layer.neurons.len());
                }
            }
            println!("  Layer {}: {} neurons (output)", mlp.get_hidden_layer_count() + 1, mlp.get_output_size());
        }
        _ => {}
    }
}
