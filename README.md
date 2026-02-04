# GlassBoxAI-MLP

## **Multi-Layer Perceptron Suite**

### *GPU-Accelerated MLP with Python & Node.js Bindings and Formal Verification*

---

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![CUDA](https://img.shields.io/badge/CUDA-12.0-green.svg)](https://developer.nvidia.com/cuda-toolkit)
[![OpenCL](https://img.shields.io/badge/OpenCL-3.0-blue.svg)](https://www.khronos.org/opencl/)
[![Rust](https://img.shields.io/badge/Rust-1.75+-orange.svg)](https://www.rust-lang.org/)
[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![Node.js](https://img.shields.io/badge/Node.js-16+-339933.svg)](https://nodejs.org/)
[![C++](https://img.shields.io/badge/C++-17+-00599C.svg)](https://isocpp.org/)
[![Julia](https://img.shields.io/badge/Julia-1.9+-9558B2.svg)](https://julialang.org/)
[![Kani](https://img.shields.io/badge/Kani-Verified-brightgreen.svg)](https://model-checking.github.io/kani/)
[![CISA Compliant](https://img.shields.io/badge/CISA-Secure%20by%20Design-blue.svg)](https://www.cisa.gov/securebydesign)

---

## **Overview**

GlassBoxAI-MLP is a comprehensive, production-ready Multi-Layer Perceptron implementation suite featuring:

- **Multiple GPU backends**: CUDA and OpenCL acceleration with automatic backend selection
- **Multi-language support**: Python, Node.js, C++, and Julia bindings
  - **Python bindings**: Full-featured Python API via PyO3 and maturin
  - **Node.js bindings**: Full-featured Node.js API via napi-rs
  - **C++ bindings**: Header-only RAII wrapper with CMake integration
  - **Julia bindings**: Native Julia interface with automatic type conversion
- **Rust implementation**: Memory-safe, high-performance core
- **Facade pattern architecture**: Clean API separation for maintainability and deep introspection
- **Formal verification**: Kani-verified Rust implementation for memory safety guarantees
- **CISA/NSA Secure by Design compliance**: Built following government cybersecurity standards

This project demonstrates enterprise-grade software engineering practices including comprehensive testing, formal verification, cross-platform compatibility, and security-first development.

---

## **Table of Contents**

1. [Features](#features)
2. [Architecture](#architecture)
3. [File Structure](#file-structure)
4. [Prerequisites](#prerequisites)
5. [Installation](#installation)
6. [Python API Reference](#python-api-reference)
7. [Node.js API Reference](#nodejs-api-reference)
8. [C++ API Reference](#c-api-reference)
9. [Julia API Reference](#julia-api-reference)
10. [CLI Reference](#cli-reference)
11. [Testing](#testing)
12. [Formal Verification with Kani](#formal-verification-with-kani)
13. [CISA/NSA Compliance](#cisansa-compliance)
14. [License](#license)
15. [Author](#author)

---

## **Features**

### Core Capabilities

| Feature | Description |
|---------|-------------|
| **Multi-Layer Architecture** | Configurable hidden layers with flexible depth and width |
| **Activation Functions** | Sigmoid, Tanh, ReLU, Softmax, Linear |
| **Optimizers** | SGD, Adam, RMSProp with configurable hyperparameters (beta1, beta2, epsilon) |
| **Regularization** | Dropout and L2 regularization |
| **Batch Normalization** | Stabilize training with learnable scale/shift parameters |
| **Training Features** | Learning rate decay, early stopping, batch training, normalization |
| **Model Persistence** | JSON serialization for model save/load |
| **ONNX Export/Import** | Interoperability with the global AI ecosystem |
| **Feature Importance** | GlassBox interpretability - understand which inputs matter most |
| **Deep Introspection** | Facade pattern enables weight/bias access, layer outputs, error signals, optimizer states |

### GPU Acceleration

| Backend | Implementation | Performance |
|---------|---------------|-------------|
| **CUDA** | Native CUDA via cudarc | Optimal for NVIDIA GPUs |
| **OpenCL** | Cross-platform GPU via ocl | AMD, Intel, NVIDIA support |
| **CPU** | Pure Rust fallback | Always available |
| **Auto** | Automatic backend selection | Best available |

### Safety & Security

| Feature | Technology |
|---------|------------|
| **Memory Safety** | Rust ownership model |
| **Formal Verification** | Kani proof harnesses |
| **Bounds Checking** | Verified array access |
| **Input Validation** | CLI and API argument validation |

---

## **Architecture**

```
┌─────────────────────────────────────────────────────────────────┐
│                        GlassBoxAI-MLP                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌────────────┐  ┌────────────┐  ┌────────────┐  ┌────────────┐│
│  │  Python    │  │  Node.js   │  │    C++     │  │   Julia    ││
│  │  Bindings  │  │  Bindings  │  │  Bindings  │  │  Bindings  ││
│  ├────────────┤  ├────────────┤  ├────────────┤  ├────────────┤│
│  │ facaded_   │  │ facaded-   │  │ facaded_   │  │ FacadedMLP ││
│  │ mlp_cuda   │  │ mlp-cuda   │  │ mlp.hpp    │  │   .jl      ││
│  │  (PyO3)    │  │ (napi-rs)  │  │ (header-   │  │  (ccall)   ││
│  │            │  │            │  │  only)     │  │            ││
│  └────────────┘  └────────────┘  └────────────┘  └────────────┘│
│         │               │               │               │       │
│         └───────────────┴───────────────┴───────────────┘       │
│                              │                                  │
│  ┌───────────────────────────┴─────────────────────────────────┐│
│  │                      Rust Core                              ││
│  │                  (Facade Pattern)                           ││
│  ├─────────────┬─────────────┬─────────────────────────────────┤│
│  │   CUDA      │   OpenCL    │          CPU                    ││
│  │  Backend    │   Backend   │        Backend                  ││
│  │ (cudarc)    │   (ocl)     │     (Pure Rust)                 ││
│  └─────────────┴─────────────┴─────────────────────────────────┘│
│                                                                 │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │                    Shared Features                          ││
│  │  • Consistent API across all language bindings             ││
│  │  • JSON-compatible model format                             ││
│  │  • ONNX import/export                                       ││
│  │  • Comprehensive test suites                                ││
│  │  • Deep introspection via Facade pattern                    ││
│  └─────────────────────────────────────────────────────────────┘│
│                                                                 │
│  ┌─────────────┐                                                │
│  │    Kani     │                                                │
│  │  Proofs     │                                                │
│  │  (Formal    │                                                │
│  │  Verify)    │                                                │
│  └─────────────┘                                                │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## **File Structure**

```
GlassBoxAI-MLP/
│
├── src/                        # Rust source code
│   ├── lib.rs                  # Library entry point
│   ├── main.rs                 # CLI entry point
│   ├── cli.rs                  # Command-line interface
│   ├── mlp.rs                  # Core MLP implementation (Facade)
│   ├── gpu_backend.rs          # GPU backend abstraction
│   ├── kernels.rs              # CUDA kernels
│   ├── opencl_kernels.rs       # OpenCL kernels
│   ├── opencl_mlp.rs           # OpenCL MLP implementation
│   ├── onnx.rs                 # ONNX import/export
│   ├── python.rs               # Python bindings (PyO3)
│   ├── nodejs.rs               # Node.js bindings (napi-rs)
│   ├── c_api.rs                # C FFI for C++/Julia
│   └── julia.rs                # Julia-specific FFI helpers
│
├── python/                     # Python package
│   └── facaded_mlp_cuda/
│       └── __init__.py         # Python module init
│
├── cpp/                        # C++ bindings
│   ├── include/
│   │   ├── facaded_mlp.hpp     # Header-only C++ wrapper (RAII)
│   │   └── facaded_mlp.h       # C API header
│   ├── examples/
│   │   └── xor_example.cpp     # C++ example
│   ├── CMakeLists.txt          # CMake build configuration
│   └── README.md               # C++ documentation
│
├── julia/                      # Julia bindings
│   ├── src/
│   │   └── FacadedMLP.jl       # Julia module
│   ├── test/
│   │   └── runtests.jl         # Julia tests
│   ├── Project.toml            # Julia package metadata
│   └── README.md               # Julia documentation
│
├── kani/                       # Formal verification proofs
│   ├── Cargo.toml
│   ├── lib.rs
│   ├── core_types.rs
│   ├── harnesses.rs
│   └── README.md
│
├── examples/                   # Example scripts
│   ├── xor_example.py          # Python XOR example
│   ├── gpu_example_backend.py  # Python GPU backend demo
│   ├── xor_example.js          # Node.js XOR example
│   └── gpu_backend_example.js  # Node.js GPU backend demo
│
├── tests/                      # Python tests
│   └── test_mlp.py             # pytest test suite
│
├── Cargo.toml                  # Rust dependencies
├── pyproject.toml              # Python build config
├── package.json                # Node.js package config
├── index.js                    # Node.js module entry
├── index.d.ts                  # TypeScript definitions
├── Makefile                    # Build automation
├── BUILD.md                    # Detailed build instructions
└── README.md                   # This file
```

---

## **Prerequisites**

### Required

| Dependency | Version | Purpose |
|------------|---------|---------|
| **Rust** | 1.75+ | Core compilation |
| **Python** | 3.8+ | Python bindings (optional) |
| **Node.js** | 16+ | Node.js bindings (optional) |
| **C++ Compiler** | C++17+ | C++ bindings (optional) |
| **Julia** | 1.9+ | Julia bindings (optional) |

### Build Tools (as needed)

| Dependency | Version | Purpose |
|------------|---------|---------|
| **maturin** | 1.0+ | Python package building |
| **@napi-rs/cli** | 2.0+ | Node.js package building |
| **CMake** | 3.15+ | C++ build system |

### Optional

| Dependency | Version | Purpose |
|------------|---------|---------|
| **CUDA Toolkit** | 11.0+ | CUDA GPU acceleration |
| **OpenCL SDK** | 3.0 | OpenCL GPU acceleration |
| **Kani** | 0.67+ | Formal verification |
| **pytest** | latest | Python testing |

---

## **Installation**

### **Quick Install (Python)**

```bash
# Install maturin
pip install maturin

# Clone and install
git clone <repo-url>
cd GlassBoxAI-MLP

# Build with all GPU backends
maturin develop --release --features python,cuda,opencl

# Or CPU-only (no GPU dependencies)
maturin develop --release --features python
```

### **Using Makefile**

```bash
# See all available commands
make help

# Build and install Python package with all backends
make install

# Run tests
make test

# Run examples
make run-xor
make run-backends
```

### **Quick Install (Node.js)**

```bash
# Install napi-rs CLI
npm install -g @napi-rs/cli

# Clone and install
git clone <repo-url>
cd GlassBoxAI-MLP

# Build with all GPU backends
npm run build

# Or CPU-only (no GPU dependencies)
npm run build:cpu
```

### **Feature Flags**

| Feature | Description |
|---------|-------------|
| `cuda` | Enable CUDA support |
| `opencl` | Enable OpenCL support |
| `cli` | Build command-line interface |
| `python` | Build Python bindings |
| `nodejs` | Build Node.js bindings |
| `julia` | Build Julia FFI (C API) |

```bash
# Python: CUDA only
maturin develop --release --features python,cuda

# Python: OpenCL only
maturin develop --release --features python,opencl

# Python: Both CUDA and OpenCL
maturin develop --release --features python,cuda,opencl

# Node.js: All backends
npm run build

# Node.js: CPU only
npm run build:cpu

# C++/Julia: Build with C API
cargo build --release --features julia,cuda,opencl
```

---

## **Python API Reference**

### **Quick Start**

```python
from facaded_mlp_cuda import MLP, PyActivationType, PyOptimizerType

# Check available backends
print("Available:", MLP.available_backends())

# Create MLP with auto-selected backend
mlp = MLP(2, [8], 1, gpu_backend="auto")
print(f"Using: {mlp.gpu_backend}")

# XOR problem
X = [[0,0], [0,1], [1,0], [1,1]]
y = [[0], [1], [1], [0]]
losses = mlp.fit(X, y, epochs=1000, verbose=True)

predictions = mlp.predict_batch(X)
print(predictions)
```

### **MLP Class**

#### Constructor

```python
MLP(
    input_size: int,
    hidden_sizes: list[int],
    output_size: int,
    hidden_activation: PyActivationType = PyActivationType.Sigmoid,
    output_activation: PyActivationType = PyActivationType.Sigmoid,
    gpu_backend: str = "auto"  # "auto", "cuda", "opencl", "cpu"
)
```

#### Training Methods

| Method | Description |
|--------|-------------|
| `fit(inputs, targets, epochs=100, batch_size=1, verbose=False, lr_decay=False, lr_decay_rate=0.95, lr_decay_epochs=10, early_stop=False, patience=10, normalize=False)` | Train on dataset with advanced options, returns loss history |
| `train(input, target)` | Train on single sample |
| `predict(input)` | Predict single sample |
| `predict_batch(inputs)` | Predict multiple samples |

#### Model I/O

| Method | Description |
|--------|-------------|
| `save(filename)` | Save model to JSON file |
| `MLP.load(filename)` | Load model from JSON file (static method) |
| `export_onnx(filename)` | Export to ONNX format |
| `MLP.import_onnx(filename)` | Import from ONNX format (static method) |

#### Properties

| Property | Type | Description |
|----------|------|-------------|
| `learning_rate` | float | Learning rate (get/set) |
| `optimizer` | PyOptimizerType | Optimizer type (get/set) |
| `dropout_rate` | float | Dropout rate 0-1 (get/set) |
| `l2_lambda` | float | L2 regularization lambda (get/set) |
| `batch_norm` | bool | Batch normalization enabled (get/set) |
| `beta1` | float | Adam beta1 parameter (get/set) |
| `beta2` | float | Adam beta2 parameter (get/set) |
| `gpu_backend` | str | Current GPU backend (get) |
| `input_size` | int | Input layer size (get) |
| `output_size` | int | Output layer size (get) |
| `hidden_sizes` | list[int] | Hidden layer sizes (get) |
| `num_layers` | int | Total layer count (get) |
| `hidden_activation` | PyActivationType | Hidden layer activation (get) |
| `output_activation` | PyActivationType | Output layer activation (get) |

#### Facade Methods (Deep Introspection)

| Method | Description |
|--------|-------------|
| `get_neuron_weights(layer, neuron)` | Get all weights for a specific neuron |
| `get_neuron_weight(layer, neuron, weight_idx)` | Get a single weight value |
| `set_neuron_weight(layer, neuron, weight_idx, value)` | Set a specific weight value |
| `get_neuron_bias(layer, neuron)` | Get bias for a specific neuron |
| `set_neuron_bias(layer, neuron, value)` | Set a neuron's bias |
| `get_layer_outputs(layer)` | Get all neuron outputs for a layer (after forward pass) |
| `get_neuron_output(layer, neuron)` | Get single neuron output (after forward pass) |
| `get_neuron_error(layer, neuron)` | Get neuron error/gradient (after backward pass) |
| `get_layer_info(layer)` | Get layer metadata (size, activation) |
| `get_activation_histogram(layer, bins=20)` | Get histogram of neuron activations |
| `get_error_histogram(layer, bins=20)` | Get histogram of neuron errors/gradients |
| `get_optimizer_state(layer, neuron)` | Get optimizer state (M, V for Adam/RMSProp) |
| `feature_importance()` | Calculate feature importance scores |

#### Backend Methods

| Method | Description |
|--------|-------------|
| `MLP.available_backends()` | List available backends (static method) |
| `set_backend(backend)` | Switch GPU backend dynamically |

### **Enums**

```python
# Activation functions
PyActivationType.Sigmoid
PyActivationType.Tanh
PyActivationType.ReLU
PyActivationType.Softmax
PyActivationType.Linear

# Optimizers
PyOptimizerType.SGD
PyOptimizerType.Adam
PyOptimizerType.RMSProp
```

### **Utility Functions**

```python
from facaded_mlp_cuda import load_csv, normalize

# Load dataset from CSV
inputs, targets = load_csv("data.csv", input_size=4, output_size=1)

# Normalize inputs to [0, 1] range
normalized = normalize(inputs)
```

### **Complete Example**

```python
from facaded_mlp_cuda import MLP, PyActivationType, PyOptimizerType

# Check available backends
print("Available GPU backends:", MLP.available_backends())

# Create MLP for XOR problem
mlp = MLP(
    input_size=2,
    hidden_sizes=[8],
    output_size=1,
    hidden_activation=PyActivationType.Sigmoid,
    output_activation=PyActivationType.Sigmoid,
    gpu_backend="auto"
)

# Set hyperparameters
mlp.learning_rate = 0.5
mlp.optimizer = PyOptimizerType.Adam
mlp.beta1 = 0.9
mlp.beta2 = 0.999
mlp.dropout_rate = 0.0
mlp.l2_lambda = 0.0

print(f"Created model: {mlp}")
print(f"Using backend: {mlp.gpu_backend}")

# XOR training data
X = [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]
y = [[0.0], [1.0], [1.0], [0.0]]

# Train with advanced options
losses = mlp.fit(
    X, y,
    epochs=2000,
    batch_size=4,
    verbose=True,
    lr_decay=True,
    lr_decay_rate=0.95,
    lr_decay_epochs=100,
    early_stop=True,
    patience=50
)
print(f"Final loss: {losses[-1]:.6f}")

# Predictions
predictions = mlp.predict_batch(X)
for inp, tgt, pred in zip(X, y, predictions):
    print(f"Input: {inp} -> Target: {tgt[0]:.1f}, Prediction: {pred[0]:.4f}")

# Deep introspection (Facade pattern)
print("\n--- Deep Introspection ---")

# Get specific weight
weight = mlp.get_neuron_weight(layer=1, neuron=0, weight_idx=0)
print(f"Layer 1, Neuron 0, Weight 0: {weight:.6f}")

# Get all weights for a neuron
weights = mlp.get_neuron_weights(layer=1, neuron=0)
print(f"Layer 1, Neuron 0 all weights: {weights}")

# Get bias
bias = mlp.get_neuron_bias(layer=1, neuron=0)
print(f"Layer 1, Neuron 0 bias: {bias:.6f}")

# Set a weight
mlp.set_neuron_weight(layer=1, neuron=0, weight_idx=0, value=0.5)
print("Set Layer 1, Neuron 0, Weight 0 to 0.5")

# Get layer outputs (requires forward pass first)
output = mlp.predict([1.0, 0.0])  # Run forward pass
layer_outputs = mlp.get_layer_outputs(layer=0)
print(f"Hidden layer outputs: {layer_outputs}")

# Get optimizer state for Adam
optimizer_state = mlp.get_optimizer_state(layer=1, neuron=0)
print(f"Optimizer state (M, V): {optimizer_state}")

# Get activation histogram
histogram = mlp.get_activation_histogram(layer=1, bins=10)
print(f"Activation histogram: {histogram}")

# Save model
mlp.save("xor_model.json")

# Load and test
mlp2 = MLP.load("xor_model.json")
output = mlp2.predict([1.0, 0.0])
print(f"\nLoaded model prediction: {output[0]:.4f}")

# Feature importance (GlassBox interpretability)
importance = mlp.feature_importance()
print("\nFeature importance:")
for feature_idx, score in importance:
    print(f"  Feature {feature_idx}: {score:.6f}")

# Export to ONNX
mlp.export_onnx("xor_model.onnx")
print("\nModel exported to ONNX format")

# Import from ONNX
mlp3 = MLP.import_onnx("xor_model.onnx")
print("Model imported from ONNX format")
```

---

## **Node.js API Reference**

### **Quick Start**

```javascript
const { MLP, JsActivationType, JsOptimizerType } = require('facaded-mlp-cuda');

// Check available backends
console.log('Available:', MLP.availableBackends());

// Create MLP with auto-selected backend
const mlp = new MLP(2, [8], 1, { gpuBackend: 'auto' });
console.log(`Using: ${mlp.gpuBackend}`);

// XOR problem
const X = [[0,0], [0,1], [1,0], [1,1]];
const y = [[0], [1], [1], [0]];
const result = mlp.fit(X, y, { epochs: 1000, verbose: true });

const predictions = mlp.predictBatch(X);
console.log(predictions);
```

### **MLP Class**

#### Constructor

```javascript
new MLP(inputSize, hiddenSizes, outputSize, options?)
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `inputSize` | `number` | Number of input neurons |
| `hiddenSizes` | `number[]` | Array of hidden layer sizes |
| `outputSize` | `number` | Number of output neurons |
| `options` | `MlpOptions` | Optional configuration object |

#### Options Object

```typescript
interface MlpOptions {
  hiddenActivation?: JsActivationType;  // default: Sigmoid
  outputActivation?: JsActivationType;  // default: Sigmoid
  gpuBackend?: 'auto' | 'cuda' | 'opencl' | 'cpu';  // default: 'auto'
  learningRate?: number;
  optimizer?: JsOptimizerType;
  dropoutRate?: number;
  l2Lambda?: number;
  batchNorm?: boolean;
  beta1?: number;  // Adam beta1
  beta2?: number;  // Adam beta2
}
```

#### Training Methods

| Method | Description |
|--------|-------------|
| `fit(inputs, targets, options?)` | Train on dataset with advanced options, returns `{ losses, finalLoss }` |
| `train(input, target)` | Train on single sample |
| `predict(input)` | Predict single sample, returns `number[]` |
| `predictBatch(inputs)` | Predict multiple samples, returns `number[][]` |

#### Fit Options

```typescript
interface FitOptions {
  epochs?: number;          // default: 100
  batchSize?: number;       // default: 1
  verbose?: boolean;        // default: false
  lrDecay?: boolean;        // default: false
  lrDecayRate?: number;     // default: 0.95
  lrDecayEpochs?: number;   // default: 10
  earlyStop?: boolean;      // default: false
  patience?: number;        // default: 10
  normalize?: boolean;      // default: false
}
```

#### Model I/O

| Method | Description |
|--------|-------------|
| `save(filename)` | Save model to JSON file |
| `MLP.load(filename)` | Load model from JSON file (static) |
| `exportOnnx(filename)` | Export to ONNX format |
| `MLP.importOnnx(filename)` | Import from ONNX format (static) |

#### Properties

| Property | Type | Description |
|----------|------|-------------|
| `learningRate` | `number` | Learning rate (get/set) |
| `optimizer` | `JsOptimizerType` | Optimizer type (get/set) |
| `dropoutRate` | `number` | Dropout rate 0-1 (get/set) |
| `l2Lambda` | `number` | L2 regularization lambda (get/set) |
| `batchNorm` | `boolean` | Batch normalization enabled (get/set) |
| `beta1` | `number` | Adam beta1 parameter (get/set) |
| `beta2` | `number` | Adam beta2 parameter (get/set) |
| `gpuBackend` | `string` | Current GPU backend (get) |
| `inputSize` | `number` | Input layer size (get) |
| `outputSize` | `number` | Output layer size (get) |
| `hiddenSizes` | `number[]` | Hidden layer sizes (get) |
| `numLayers` | `number` | Total layer count (get) |
| `hiddenActivation` | `JsActivationType` | Hidden activation (get) |
| `outputActivation` | `JsActivationType` | Output activation (get) |

#### Facade Methods (Deep Introspection)

| Method | Description |
|--------|-------------|
| `getNeuronWeights(layer, neuron)` | Get all weights for a specific neuron |
| `getNeuronWeight(layer, neuron, weightIdx)` | Get a single weight value |
| `setNeuronWeight(layer, neuron, weightIdx, value)` | Set a specific weight value |
| `getNeuronBias(layer, neuron)` | Get bias for a specific neuron |
| `setNeuronBias(layer, neuron, value)` | Set a neuron's bias |
| `getLayerOutputs(layer)` | Get all neuron outputs for a layer (after forward pass) |
| `getNeuronOutput(layer, neuron)` | Get single neuron output (after forward pass) |
| `getNeuronError(layer, neuron)` | Get neuron error/gradient (after backward pass) |
| `getLayerInfo(layer)` | Get layer metadata object `{ size, activation }` |
| `getActivationHistogram(layer, bins?)` | Get histogram of neuron activations |
| `getErrorHistogram(layer, bins?)` | Get histogram of neuron errors/gradients |
| `getOptimizerState(layer, neuron)` | Get optimizer state `{ m, v }` for Adam/RMSProp |
| `featureImportance()` | Returns `{ featureIndex, score }[]` |

#### Backend Methods

| Method | Description |
|--------|-------------|
| `MLP.availableBackends()` | List available backends (static) |
| `setBackend(backend)` | Switch GPU backend dynamically |
| `info()` | Get model info object |
| `toString()` | Get string representation |

### **Enums**

```javascript
// Activation functions
JsActivationType.Sigmoid  // 0
JsActivationType.Tanh     // 1
JsActivationType.ReLU     // 2
JsActivationType.Softmax  // 3
JsActivationType.Linear   // 4

// Optimizers
JsOptimizerType.SGD       // 0
JsOptimizerType.Adam      // 1
JsOptimizerType.RMSProp   // 2
```

### **Utility Functions**

```javascript
const { loadCsv, normalize } = require('facaded-mlp-cuda');

// Load dataset from CSV
const { inputs, targets } = loadCsv('data.csv', 4, 1);

// Normalize inputs to [0, 1] range
const normalized = normalize(inputs);
```

### **Complete Example**

```javascript
const { MLP, JsActivationType, JsOptimizerType } = require('facaded-mlp-cuda');

// Check available backends
console.log('Available GPU backends:', MLP.availableBackends());

// Create MLP for XOR problem
const mlp = new MLP(2, [8], 1, {
  hiddenActivation: JsActivationType.Sigmoid,
  outputActivation: JsActivationType.Sigmoid,
  gpuBackend: 'auto',
});

// Set hyperparameters
mlp.learningRate = 0.5;
mlp.optimizer = JsOptimizerType.Adam;
mlp.beta1 = 0.9;
mlp.beta2 = 0.999;
mlp.dropoutRate = 0.0;
mlp.l2Lambda = 0.0;

console.log(`Created model: ${mlp.toString()}`);
console.log(`Using backend: ${mlp.gpuBackend}`);

// XOR training data
const X = [[0, 0], [0, 1], [1, 0], [1, 1]];
const y = [[0], [1], [1], [0]];

// Train with advanced options
const result = mlp.fit(X, y, {
  epochs: 2000,
  batchSize: 4,
  verbose: true,
  lrDecay: true,
  lrDecayRate: 0.95,
  lrDecayEpochs: 100,
  earlyStop: true,
  patience: 50
});
console.log(`Final loss: ${result.finalLoss.toFixed(6)}`);

// Predictions
const predictions = mlp.predictBatch(X);
for (let i = 0; i < X.length; i++) {
  console.log(`Input: [${X[i]}] -> Target: ${y[i][0]}, Prediction: ${predictions[i][0].toFixed(4)}`);
}

// Deep introspection (Facade pattern)
console.log('\n--- Deep Introspection ---');

// Get specific weight
const weight = mlp.getNeuronWeight(1, 0, 0);
console.log(`Layer 1, Neuron 0, Weight 0: ${weight.toFixed(6)}`);

// Get all weights for a neuron
const weights = mlp.getNeuronWeights(1, 0);
console.log(`Layer 1, Neuron 0 all weights: ${weights}`);

// Get bias
const bias = mlp.getNeuronBias(1, 0);
console.log(`Layer 1, Neuron 0 bias: ${bias.toFixed(6)}`);

// Set a weight
mlp.setNeuronWeight(1, 0, 0, 0.5);
console.log('Set Layer 1, Neuron 0, Weight 0 to 0.5');

// Get layer outputs (requires forward pass first)
const output = mlp.predict([1.0, 0.0]);  // Run forward pass
const layerOutputs = mlp.getLayerOutputs(0);
console.log(`Hidden layer outputs: ${layerOutputs}`);

// Get optimizer state for Adam
const optimizerState = mlp.getOptimizerState(1, 0);
console.log(`Optimizer state (M, V): ${JSON.stringify(optimizerState)}`);

// Get activation histogram
const histogram = mlp.getActivationHistogram(1, 10);
console.log(`Activation histogram: ${histogram}`);

// Get layer info
const layerInfo = mlp.getLayerInfo(1);
console.log(`Layer 1 info: ${JSON.stringify(layerInfo)}`);

// Save model
mlp.save('xor_model.json');

// Load and test
const mlp2 = MLP.load('xor_model.json');
const output2 = mlp2.predict([1.0, 0.0]);
console.log(`\nLoaded model prediction: ${output2[0].toFixed(4)}`);

// Feature importance (GlassBox interpretability)
const importance = mlp.featureImportance();
console.log('\nFeature importance:');
for (const { featureIndex, score } of importance) {
  console.log(`  Feature ${featureIndex}: ${score.toFixed(6)}`);
}

// Export to ONNX
mlp.exportOnnx('xor_model.onnx');
console.log('\nModel exported to ONNX format');

// Import from ONNX
const mlp3 = MLP.importOnnx('xor_model.onnx');
console.log('Model imported from ONNX format');
```

### **TypeScript Support**

The package includes full TypeScript definitions in `index.d.ts`:

```typescript
import {
  MLP,
  JsActivationType,
  JsOptimizerType,
  MlpOptions,
  FitOptions,
  TrainResult,
  LayerInfo,
  OptimizerState
} from 'facaded-mlp-cuda';

const options: MlpOptions = {
  hiddenActivation: JsActivationType.ReLU,
  gpuBackend: 'cuda',
  learningRate: 0.01,
  beta1: 0.9,
  beta2: 0.999
};

const mlp = new MLP(4, [16, 8], 2, options);

const fitOptions: FitOptions = {
  epochs: 1000,
  batchSize: 32,
  verbose: true,
  lrDecay: true,
  earlyStop: true
};

const result: TrainResult = mlp.fit(inputs, targets, fitOptions);

const layerInfo: LayerInfo = mlp.getLayerInfo(0);
const optimizerState: OptimizerState = mlp.getOptimizerState(1, 0);
```

---

## **C++ API Reference**

### **Quick Start**

```cpp
#include "facaded_mlp.hpp"
#include <iostream>

int main() {
    using namespace facaded;
    
    // Create a network: 2 inputs, 8 hidden neurons, 1 output
    MLP mlp(2, {8}, 1);
    
    // XOR training data
    std::vector<std::vector<double>> X = {{0,0}, {0,1}, {1,0}, {1,1}};
    std::vector<std::vector<double>> y = {{0}, {1}, {1}, {0}};
    
    // Train
    auto result = mlp.fit(X, y, 1000, true);
    std::cout << "Final loss: " << result.final_loss << std::endl;
    
    // Predict
    for (const auto& x : X) {
        auto output = mlp.predict(x);
        std::cout << "[" << x[0] << ", " << x[1] << "] => " 
                  << output[0] << std::endl;
    }
    
    return 0;
}
```

### **Installation**

#### 1. Build the Rust library

```bash
cd GlassBoxAI-MLP
cargo build --release --features julia
```

#### 2. Build with CMake

```bash
cd cpp
mkdir build && cd build
cmake ..
make
```

#### 3. Or compile directly

```bash
g++ -std=c++17 -O2 -I include examples/xor_example.cpp \
    -L ../target/release -lfacaded_mlp_cuda \
    -Wl,-rpath,../target/release \
    -o xor_example
```

### **MLP Class**

#### Constructor

```cpp
using namespace facaded;

// Basic creation
MLP mlp(input_size, hidden_sizes, output_size);

// With options
MLPOptions opts;
opts.hidden_activation = Activation::ReLU;
opts.output_activation = Activation::Sigmoid;
opts.backend = "cuda";  // "auto", "cpu", "cuda", "opencl"
opts.learning_rate = 0.001;
opts.optimizer = Optimizer::Adam;
opts.dropout_rate = 0.1;
opts.l2_lambda = 0.0001;
opts.batch_norm = true;
opts.beta1 = 0.9;
opts.beta2 = 0.999;

MLP mlp(784, {256, 128}, 10, opts);
```

#### Training Methods

| Method | Description |
|--------|-------------|
| `train(input, target)` | Train on single sample |
| `fit(inputs, targets, epochs, verbose)` | Train on dataset, returns `TrainResult` |

#### Training Result

```cpp
struct TrainResult {
    std::vector<double> losses;  // Loss per epoch
    double final_loss;           // Final epoch loss
};
```

#### Prediction Methods

| Method | Description |
|--------|-------------|
| `predict(input)` | Predict single sample, returns `std::vector<double>` |
| `predict_batch(inputs)` | Predict multiple samples, returns `std::vector<std::vector<double>>` |

#### Model I/O

| Method | Description |
|--------|-------------|
| `save(filename)` | Save model to JSON file |
| `MLP::load(filename)` | Load model from JSON file (static) |
| `export_onnx(filename)` | Export to ONNX format |
| `MLP::import_onnx(filename)` | Import from ONNX format (static) |

#### Properties (Getters)

| Property | Return Type | Description |
|----------|-------------|-------------|
| `learning_rate()` | `double` | Current learning rate |
| `optimizer()` | `Optimizer` | Optimizer type |
| `dropout_rate()` | `double` | Dropout rate |
| `l2_lambda()` | `double` | L2 regularization |
| `batch_norm()` | `bool` | Batch normalization enabled |
| `beta1()` | `double` | Adam beta1 parameter |
| `beta2()` | `double` | Adam beta2 parameter |
| `backend()` | `std::string` | Current GPU backend |
| `input_size()` | `int` | Input layer size |
| `output_size()` | `int` | Output layer size |
| `hidden_sizes()` | `const std::vector<int>&` | Hidden layer sizes |
| `num_layers()` | `int` | Total layer count |
| `hidden_activation()` | `Activation` | Hidden activation function |
| `output_activation()` | `Activation` | Output activation function |

#### Properties (Setters)

| Method | Description |
|--------|-------------|
| `set_learning_rate(value)` | Set learning rate |
| `set_optimizer(opt)` | Set optimizer type |
| `set_dropout_rate(value)` | Set dropout rate |
| `set_l2_lambda(value)` | Set L2 regularization |
| `set_batch_norm(enabled)` | Enable/disable batch normalization |
| `set_beta1(value)` | Set Adam beta1 |
| `set_beta2(value)` | Set Adam beta2 |
| `set_backend(backend)` | Switch GPU backend |

#### Facade Methods (Deep Introspection)

| Method | Description |
|--------|-------------|
| `get_neuron_weights(layer, neuron)` | Get all weights for a neuron |
| `get_neuron_weight(layer, neuron, weight_idx)` | Get single weight value |
| `set_neuron_weight(layer, neuron, weight_idx, value)` | Set single weight value |
| `get_neuron_bias(layer, neuron)` | Get neuron bias |
| `set_neuron_bias(layer, neuron, value)` | Set neuron bias |
| `get_layer_outputs(layer)` | Get layer outputs (after forward pass) |
| `get_neuron_output(layer, neuron)` | Get single neuron output |
| `get_neuron_error(layer, neuron)` | Get neuron error/gradient |
| `get_layer_info(layer)` | Get layer metadata |
| `get_activation_histogram(layer, bins)` | Get activation histogram |
| `get_error_histogram(layer, bins)` | Get error histogram |
| `get_optimizer_state(layer, neuron)` | Get optimizer state (M, V) |
| `feature_importance()` | Calculate feature importance |

#### Backend Methods

| Method | Description |
|--------|-------------|
| `MLP::available_backends()` | List available backends (static) |

### **Enums**

```cpp
enum class Activation {
    Sigmoid,
    Tanh,
    ReLU,
    Softmax,
    Linear
};

enum class Optimizer {
    SGD,
    Adam,
    RMSProp
};
```

### **Structs**

```cpp
struct FeatureImportance {
    int index;
    double score;
};

struct LayerInfo {
    int size;
    Activation activation;
};

struct OptimizerState {
    std::vector<double> m;  // First moment (Adam/RMSProp)
    std::vector<double> v;  // Second moment (Adam/RMSProp)
};
```

### **Error Handling**

The library uses exceptions for error handling:

```cpp
try {
    MLP mlp(2, {8}, 1);
    mlp.train(input, target);
} catch (const facaded::MLPException& e) {
    std::cerr << "Error: " << e.what() << std::endl;
}
```

### **Complete Example**

```cpp
#include "facaded_mlp.hpp"
#include <iostream>
#include <iomanip>

int main() {
    using namespace facaded;
    
    // Check available backends
    auto backends = MLP::available_backends();
    std::cout << "Available backends: ";
    for (const auto& backend : backends) {
        std::cout << backend << " ";
    }
    std::cout << std::endl;
    
    // Create MLP with custom options
    MLPOptions opts;
    opts.hidden_activation = Activation::Sigmoid;
    opts.output_activation = Activation::Sigmoid;
    opts.backend = "auto";
    opts.learning_rate = 0.5;
    opts.optimizer = Optimizer::Adam;
    opts.beta1 = 0.9;
    opts.beta2 = 0.999;
    
    MLP mlp(2, {8}, 1, opts);
    std::cout << "Created model: " << mlp << std::endl;
    std::cout << "Using backend: " << mlp.backend() << std::endl;
    
    // XOR training data
    std::vector<std::vector<double>> X = {
        {0.0, 0.0}, {0.0, 1.0}, {1.0, 0.0}, {1.0, 1.0}
    };
    std::vector<std::vector<double>> y = {
        {0.0}, {1.0}, {1.0}, {0.0}
    };
    
    // Train
    auto result = mlp.fit(X, y, 2000, true);
    std::cout << std::fixed << std::setprecision(6);
    std::cout << "Final loss: " << result.final_loss << std::endl;
    
    // Predictions
    std::cout << "\nPredictions:" << std::endl;
    for (size_t i = 0; i < X.size(); ++i) {
        auto pred = mlp.predict(X[i]);
        std::cout << "Input: [" << X[i][0] << ", " << X[i][1] << "] -> "
                  << "Target: " << y[i][0] << ", "
                  << "Prediction: " << pred[0] << std::endl;
    }
    
    // Deep introspection (Facade)
    std::cout << "\n--- Deep Introspection ---" << std::endl;
    
    // Get specific weight
    double weight = mlp.get_neuron_weight(1, 0, 0);
    std::cout << "Layer 1, Neuron 0, Weight 0: " << weight << std::endl;
    
    // Get all weights for a neuron
    auto weights = mlp.get_neuron_weights(1, 0);
    std::cout << "Layer 1, Neuron 0 all weights: [";
    for (size_t i = 0; i < weights.size(); ++i) {
        std::cout << weights[i];
        if (i < weights.size() - 1) std::cout << ", ";
    }
    std::cout << "]" << std::endl;
    
    // Get bias
    double bias = mlp.get_neuron_bias(1, 0);
    std::cout << "Layer 1, Neuron 0 bias: " << bias << std::endl;
    
    // Get optimizer state
    auto opt_state = mlp.get_optimizer_state(1, 0);
    std::cout << "Optimizer state (M size, V size): (" 
              << opt_state.m.size() << ", " << opt_state.v.size() << ")" 
              << std::endl;
    
    // Save model
    mlp.save("xor_model.json");
    std::cout << "\nModel saved to xor_model.json" << std::endl;
    
    // Load and test
    auto mlp2 = MLP::load("xor_model.json");
    auto output = mlp2.predict({1.0, 0.0});
    std::cout << "Loaded model prediction: " << output[0] << std::endl;
    
    // Feature importance
    auto importance = mlp.feature_importance();
    std::cout << "\nFeature importance:" << std::endl;
    for (const auto& fi : importance) {
        std::cout << "  Feature " << fi.index << ": " << fi.score << std::endl;
    }
    
    // Export to ONNX
    mlp.export_onnx("xor_model.onnx");
    std::cout << "\nModel exported to ONNX format" << std::endl;
    
    return 0;
}
```

### **CMake Integration**

```cmake
# Find the installed library
find_package(facaded_mlp REQUIRED)

# Link against your target
target_link_libraries(your_target PRIVATE facaded::facaded_mlp)

# Or add as subdirectory
add_subdirectory(path/to/GlassBoxAI-MLP/cpp)
target_link_libraries(your_target PRIVATE facaded_mlp)
```

### **C API**

For C interop or when you need a C-compatible interface, include `facaded_mlp.h`:

```c
#include "facaded_mlp.h"

// Create model
int hidden[] = {8};
mlp_handle_t mlp = mlp_create(2, hidden, 1, 1, 
    MLP_ACTIVATION_SIGMOID, MLP_ACTIVATION_SIGMOID, "auto");

// Train
double input[2] = {1.0, 0.0};
double target[1] = {1.0};
mlp_train(mlp, input, 2, target, 1);

// Predict
double output[1];
mlp_predict(mlp, input, 2, output, 1);

// Cleanup
mlp_destroy(mlp);
```

---

## **Julia API Reference**

### **Quick Start**

```julia
using FacadedMLP

# Create a network: 2 inputs, 8 hidden neurons, 1 output
mlp = MLP(2, [8], 1)

# Train on XOR problem
X = [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]
y = [[0.0], [1.0], [1.0], [0.0]]
losses = fit!(mlp, X, y; epochs=1000, verbose=true)

# Predict
for x in X
    output = predict(mlp, x)
    println("$(x) => $(round(output[1], digits=4))")
end
```

### **Installation**

#### 1. Build the Rust library

```bash
cd GlassBoxAI-MLP
cargo build --release --features julia
```

#### 2. Add the Julia package

```julia
using Pkg
Pkg.develop(path="/path/to/GlassBoxAI-MLP/julia")
```

Or in the Julia REPL:

```julia
] dev /path/to/GlassBoxAI-MLP/julia
```

### **MLP Constructor**

```julia
# Basic creation
mlp = MLP(input_size, hidden_sizes, output_size)

# With options
mlp = MLP(2, [16, 8], 1;
    hidden_activation = ReLU,      # Sigmoid, Tanh, ReLU, Softmax, Linear
    output_activation = Sigmoid,
    backend = "auto",              # "auto", "cpu", "cuda", "opencl"
    learning_rate = 0.01,
    optimizer = Adam,              # SGD, Adam, RMSProp
    dropout_rate = 0.0,
    l2_lambda = 0.0,
    batch_norm = false,
    beta1 = 0.9,
    beta2 = 0.999
)
```

### **Training Functions**

| Function | Description |
|----------|-------------|
| `train!(mlp, input, target)` | Train on single sample |
| `fit!(mlp, inputs, targets; options...)` | Train on dataset, returns loss history |

#### Fit Options

```julia
losses = fit!(mlp, X, y;
    epochs = 100,
    batch_size = 1,
    verbose = false,
    lr_decay = false,
    lr_decay_rate = 0.95,
    lr_decay_epochs = 10,
    early_stop = false,
    patience = 10,
    normalize = false
)
```

### **Prediction Functions**

| Function | Description |
|----------|-------------|
| `predict(mlp, input)` | Predict single sample, returns `Vector{Float64}` |
| `predict_batch(mlp, inputs)` | Predict multiple samples, returns `Vector{Vector{Float64}}` |

### **Model I/O**

| Function | Description |
|----------|-------------|
| `save(mlp, filename)` | Save model to JSON file |
| `load(filename)` | Load model from JSON file |
| `export_onnx(mlp, filename)` | Export to ONNX format |
| `import_onnx(filename)` | Import from ONNX format |

### **Properties**

```julia
# Mutable properties (can be set)
mlp.learning_rate = 0.001
mlp.optimizer = Adam
mlp.dropout_rate = 0.1
mlp.l2_lambda = 0.0001
mlp.batch_norm = true
mlp.beta1 = 0.9
mlp.beta2 = 0.999

# Read-only properties
mlp.input_size
mlp.output_size
mlp.hidden_sizes
mlp.num_layers
mlp.backend
mlp.hidden_activation
mlp.output_activation
```

### **Facade Methods (Deep Introspection)**

| Function | Description |
|----------|-------------|
| `get_neuron_weights(mlp, layer, neuron)` | Get all weights for a neuron |
| `get_neuron_weight(mlp, layer, neuron, weight_idx)` | Get single weight value |
| `set_neuron_weight!(mlp, layer, neuron, weight_idx, value)` | Set single weight value |
| `get_neuron_bias(mlp, layer, neuron)` | Get neuron bias |
| `set_neuron_bias!(mlp, layer, neuron, value)` | Set neuron bias |
| `get_layer_outputs(mlp, layer)` | Get layer outputs (after forward pass) |
| `get_neuron_output(mlp, layer, neuron)` | Get single neuron output |
| `get_neuron_error(mlp, layer, neuron)` | Get neuron error/gradient |
| `get_layer_info(mlp, layer)` | Get layer metadata `(size, activation)` |
| `get_activation_histogram(mlp, layer, bins=20)` | Get activation histogram |
| `get_error_histogram(mlp, layer, bins=20)` | Get error histogram |
| `get_optimizer_state(mlp, layer, neuron)` | Get optimizer state `(m, v)` |
| `feature_importance(mlp)` | Calculate feature importance |

### **Backend Functions**

| Function | Description |
|----------|-------------|
| `available_backends()` | List available GPU backends |
| `set_backend!(mlp, backend)` | Switch GPU backend |

### **Activation Types**

```julia
# Activation enum values
Sigmoid    # Default for hidden and output
Tanh
ReLU
Softmax
Linear
```

### **Optimizer Types**

```julia
# Optimizer enum values
SGD
Adam       # Default
RMSProp
```

### **Complete Example**

```julia
using FacadedMLP

# Check available backends
println("Available GPU backends: ", available_backends())

# Create MLP for XOR problem
mlp = MLP(2, [8], 1;
    hidden_activation = Sigmoid,
    output_activation = Sigmoid,
    backend = "auto",
    learning_rate = 0.5,
    optimizer = Adam,
    beta1 = 0.9,
    beta2 = 0.999
)

println("Created model: ", mlp)
println("Using backend: ", mlp.backend)

# XOR training data
X = [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]
y = [[0.0], [1.0], [1.0], [0.0]]

# Train with options
losses = fit!(mlp, X, y;
    epochs = 2000,
    batch_size = 4,
    verbose = true,
    lr_decay = true,
    lr_decay_rate = 0.95,
    lr_decay_epochs = 100,
    early_stop = true,
    patience = 50
)
println("Final loss: ", round(losses[end], digits=6))

# Predictions
println("\nPredictions:")
for (x, target) in zip(X, y)
    pred = predict(mlp, x)
    println("Input: $x -> Target: $(target[1]), Prediction: $(round(pred[1], digits=4))")
end

# Deep introspection (Facade pattern)
println("\n--- Deep Introspection ---")

# Get specific weight
weight = get_neuron_weight(mlp, 1, 0, 0)
println("Layer 1, Neuron 0, Weight 0: ", round(weight, digits=6))

# Get all weights for a neuron
weights = get_neuron_weights(mlp, 1, 0)
println("Layer 1, Neuron 0 all weights: ", weights)

# Get bias
bias = get_neuron_bias(mlp, 1, 0)
println("Layer 1, Neuron 0 bias: ", round(bias, digits=6))

# Set a weight
set_neuron_weight!(mlp, 1, 0, 0, 0.5)
println("Set Layer 1, Neuron 0, Weight 0 to 0.5")

# Get layer outputs (requires forward pass first)
output = predict(mlp, [1.0, 0.0])  # Run forward pass
layer_outputs = get_layer_outputs(mlp, 0)
println("Hidden layer outputs: ", layer_outputs)

# Get optimizer state for Adam
m, v = get_optimizer_state(mlp, 1, 0)
println("Optimizer state (M size, V size): ($(length(m)), $(length(v)))")

# Get layer info
size, activation = get_layer_info(mlp, 1)
println("Layer 1 info: size=$size, activation=$activation")

# Save model
save(mlp, "xor_model.json")
println("\nModel saved to xor_model.json")

# Load and test
mlp2 = load("xor_model.json")
output = predict(mlp2, [1.0, 0.0])
println("Loaded model prediction: ", round(output[1], digits=4))

# Feature importance (GlassBox interpretability)
importance = feature_importance(mlp)
println("\nFeature importance:")
for (idx, score) in importance
    println("  Feature $idx: ", round(score, digits=6))
end

# Export to ONNX
export_onnx(mlp, "xor_model.onnx")
println("\nModel exported to ONNX format")

# Import from ONNX
mlp3 = import_onnx("xor_model.onnx")
println("Model imported from ONNX format")
```

### **MNIST-like Classification Example**

```julia
using FacadedMLP

# 784 inputs (28x28 images), 2 hidden layers, 10 outputs (digits 0-9)
mlp = MLP(784, [256, 128], 10;
    hidden_activation = ReLU,
    output_activation = Softmax,
    optimizer = Adam,
    learning_rate = 0.001,
    dropout_rate = 0.2,
    batch_norm = true,
    backend = "cuda"
)

# Train
losses = fit!(mlp, train_X, train_y;
    epochs = 50,
    batch_size = 32,
    verbose = true,
    lr_decay = true,
    early_stop = true
)

# Predict class
function predict_class(mlp, x)
    output = predict(mlp, x)
    argmax(output) - 1  # 0-indexed class
end

accuracy = mean(predict_class(mlp, x) == y for (x, y) in zip(test_X, test_y))
println("Test accuracy: ", round(accuracy * 100, digits=2), "%")
```

### **Running Tests**

```bash
cd julia
julia --project=. -e 'using Pkg; Pkg.test()'
```

---

## **CLI Reference**
### Usage

```
facaded_mlp_cuda <command> [options]
```

### Commands

| Command | Description |
|---------|-------------|
| `create` | Create a new model |
| `info` | Display model information |
| `train` | Train on dataset |
| `predict` | Make a single prediction |
| `batch-predict` | Make batch predictions |
| `export-onnx` | Export to ONNX format |
| `import-onnx` | Import from ONNX format |
| `feature-importance` | Calculate feature importance |
| `get-weight` | Get a single weight value (FACADE) |
| `set-weight` | Set a single weight value (FACADE) |
| `get-weights` | Get all weights for a neuron (FACADE) |
| `get-bias` | Get bias for a neuron (FACADE) |
| `set-bias` | Set bias for a neuron (FACADE) |
| `get-output` | Get neuron output value (FACADE) |
| `get-error` | Get neuron error value (FACADE) |
| `layer-info` | Display layer information (FACADE) |
| `histogram` | Display activation or error histogram (FACADE) |
| `get-optimizer` | Get optimizer state values M, V (FACADE) |
| `help` | Show help message |

### Create Options

| Option | Description |
|--------|-------------|
| `-i, --inputs=N` | Number of input neurons (required) |
| `-H, --hidden=N,M,...` | Hidden layer sizes, comma-separated (required) |
| `-o, --outputs=N` | Number of output neurons (required) |
| `-s, --save=FILE` | Save model file (required, .json) |
| `--lr=VALUE` | Learning rate (default: 0.1) |
| `--optimizer=TYPE` | sgd\|adam\|rmsprop (default: sgd) |
| `--hidden-act=TYPE` | sigmoid\|tanh\|relu\|softmax\|linear (default: sigmoid) |
| `--output-act=TYPE` | sigmoid\|tanh\|relu\|softmax\|linear (default: sigmoid) |
| `--dropout=VALUE` | Dropout rate 0-1 (default: 0) |
| `--l2=VALUE` | L2 regularization lambda (default: 0) |
| `--beta1=VALUE` | Adam beta1 parameter (default: 0.9) |
| `--beta2=VALUE` | Adam beta2 parameter (default: 0.999) |
| `--batch-norm` | Enable batch normalization |
| `--gpu=BACKEND` | GPU backend: auto\|cuda\|opencl\|cpu (default: auto) |

### Train Options

| Option | Description |
|--------|-------------|
| `-m, --model=FILE` | Load model file (required, .json) |
| `-d, --data=FILE` | Training data CSV file (required) |
| `-s, --save=FILE` | Save trained model (required, .json) |
| `--epochs=N` | Training epochs (default: 100) |
| `--batch=N` | Batch size (default: 1) |
| `--lr=VALUE` | Override learning rate |
| `--lr-decay` | Enable learning rate decay |
| `--lr-decay-rate=VALUE` | LR decay rate (default: 0.95) |
| `--lr-decay-epochs=N` | Decay interval in epochs (default: 10) |
| `--early-stop` | Enable early stopping |
| `--patience=N` | Early stopping patience (default: 10) |
| `--normalize` | Normalize training data |
| `--verbose` | Print training progress |

### Predict Options

| Option | Description |
|--------|-------------|
| `-m, --model=FILE` | Model file (required, .json) |
| `-i, --input=v1,v2,...` | Input values, comma-separated (required) |

### Batch Predict Options

| Option | Description |
|--------|-------------|
| `-m, --model=FILE` | Model file (required, .json) |
| `-i, --input=v1,v2,...` | Input values, comma-separated (required) |

### Export/Import ONNX Options

| Option | Description |
|--------|-------------|
| `-m, --model=FILE` | Model file to export (required for export-onnx) |
| `-s, --save=FILE` | Output file (.onnx for export, .json for import) |
| `--onnx=FILE` | ONNX file to import (required for import-onnx) |

### Feature Importance Options

| Option | Description |
|--------|-------------|
| `-m, --model=FILE` | Model file to analyze (required) |

### Facade Options (Deep Introspection)

| Option | Description |
|--------|-------------|
| `--layer=L` | Layer index (required for facade commands) |
| `--neuron=N` | Neuron index (required for most facade commands) |
| `--weight=W` | Weight index within neuron (for get/set-weight) |
| `--value=V` | Value to set (required for set-* commands) |
| `--bins=N` | Number of histogram bins (default: 20) |
| `--type=TYPE` | Histogram type: activation\|error (default: activation) |

### CLI Examples

```bash
# Create a new model (auto-detect GPU)
facaded_mlp_cuda create -i 2 -H 8 -o 1 -s model.json

# Create with specific backend and hyperparameters
facaded_mlp_cuda create -i 2 -H 8,8 -o 1 -s model.json \
  --gpu=cuda \
  --optimizer=adam \
  --lr=0.01 \
  --beta1=0.9 \
  --beta2=0.999 \
  --batch-norm

# Train with advanced options
facaded_mlp_cuda train -m model.json -d data.csv -s trained.json \
  --epochs=1000 \
  --batch=32 \
  --lr-decay \
  --lr-decay-rate=0.95 \
  --lr-decay-epochs=50 \
  --early-stop \
  --patience=20 \
  --verbose

# Make a prediction
facaded_mlp_cuda predict -m trained.json -i 1.0,0.0

# Batch prediction
facaded_mlp_cuda batch-predict -m trained.json -i 1.0,0.0

# Export/Import ONNX
facaded_mlp_cuda export-onnx -m trained.json -s model.onnx
facaded_mlp_cuda import-onnx --onnx=model.onnx -s imported.json

# Feature importance
facaded_mlp_cuda feature-importance -m trained.json

# Facade introspection commands
facaded_mlp_cuda get-weight -m model.json --layer=1 --neuron=0 --weight=0
facaded_mlp_cuda set-weight -m model.json --layer=1 --neuron=0 --weight=0 --value=0.5 -s modified.json
facaded_mlp_cuda get-weights -m model.json --layer=1 --neuron=0
facaded_mlp_cuda get-bias -m model.json --layer=1 --neuron=0
facaded_mlp_cuda set-bias -m model.json --layer=1 --neuron=0 --value=0.1 -s modified.json
facaded_mlp_cuda get-output -m model.json --layer=0 --neuron=3 -i 1.0,0.0
facaded_mlp_cuda get-error -m model.json --layer=1 --neuron=0
facaded_mlp_cuda layer-info -m model.json --layer=0
facaded_mlp_cuda histogram -m model.json --layer=1 --bins=30 --type=activation
facaded_mlp_cuda histogram -m model.json --layer=1 --bins=30 --type=error
facaded_mlp_cuda get-optimizer -m model.json --layer=1 --neuron=0
```

---

## **Testing**

### Running Python Tests

```bash
# Install test dependencies
pip install pytest numpy

# Run all tests
pytest tests/ -v

# Run specific test
pytest tests/test_mlp.py::test_xor_training -v

# Run with coverage
pytest tests/ --cov=facaded_mlp_cuda --cov-report=html
```

### Running Node.js Tests

```bash
# Run examples as tests
npm test

# Or run directly
node examples/xor_example.js
node examples/gpu_backend_example.js
```

### Running C++ Tests

```bash
cd cpp/build

# Run example
./xor_example

# Or compile and run directly
cd cpp
g++ -std=c++17 -O2 -I include examples/xor_example.cpp \
    -L ../target/release -lfacaded_mlp_cuda \
    -Wl,-rpath,../target/release \
    -o xor_example
./xor_example
```

### Running Julia Tests

```bash
cd julia

# Run test suite
julia --project=. -e 'using Pkg; Pkg.test()'

# Or run examples interactively
julia --project=.
```

```julia
julia> include("examples/xor_example.jl")
```

### Using Makefile

```bash
# Run all tests
make test

# Run Python tests only
make test-python

# Run Node.js examples
make test-node
```

### Test Categories

Each test suite covers:

| Category | Tests |
|----------|-------|
| **Help & Usage** | Command-line interface verification |
| **Model Creation** | Various architecture configurations, backend selection |
| **Hyperparameters** | Learning rate, activation, optimizer, beta1/beta2 settings |
| **Model Info** | Metadata retrieval |
| **Save & Load** | Model persistence and JSON compatibility |
| **Training** | Forward/backward pass, loss computation, batch training |
| **Advanced Training** | Learning rate decay, early stopping, normalization |
| **Prediction** | Single and batch inference with trained models |
| **ONNX** | Export and import compatibility |
| **Feature Importance** | GlassBox interpretability |
| **Facade Operations** | Weight/bias get/set, layer outputs, errors, histograms, optimizer states |
| **Backend Switching** | Dynamic GPU backend changes |
| **Error Handling** | Invalid input handling |
| **Cross-Implementation** | API compatibility across Python/Node.js/C++/Julia/CLI |

### Test Output Example

```
=========================================
Python MLP Test Suite
=========================================

test_create_mlp ... PASS
test_available_backends ... PASS
test_xor_training ... PASS
test_save_load ... PASS
test_property_access ... PASS
test_facade_weight_access ... PASS
test_facade_bias_access ... PASS
test_feature_importance ... PASS
test_onnx_export_import ... PASS
test_backend_switching ... PASS
...

=========================================
Test Summary
=========================================
Total tests: 42
Passed: 42
Failed: 0

All tests passed!
```

---

## **Formal Verification with Kani**

### Overview

The Rust implementation includes **Kani formal verification proofs** that mathematically prove the absence of certain classes of bugs. This goes beyond traditional testing to provide **mathematical guarantees** about code correctness.

### Running Kani Verification

```bash
cd kani

# Run all proofs
cargo kani

# Run specific proof
cargo kani --harness proof_name

# Run unit tests
cargo test

# Verify specific modules
cargo kani --harness verify_weight_indexing
cargo kani --harness verify_layer_access
cargo kani --harness verify_activation_bounds
```

### Verified Properties

The Kani proofs verify:

| Property | Description |
|----------|-------------|
| **Memory Safety** | No buffer overflows, null pointer dereferences, or use-after-free |
| **Array Bounds** | All array accesses within valid bounds |
| **Integer Overflow** | No arithmetic overflows in critical computations |
| **Layer Indexing** | Valid layer indices for all operations |
| **Weight/Bias Access** | Safe neuron weight and bias indexing |
| **Activation Computation** | Activation functions produce valid outputs |
| **No Panics** | Critical paths proven panic-free |

### Why Formal Verification Matters

Traditional testing can only verify specific test cases. Formal verification with Kani:

- **Exhaustively checks all possible inputs** within defined bounds
- **Mathematically proves** absence of panics, buffer overflows, and undefined behavior
- **Catches edge cases** that random testing might miss (e.g., boundary conditions, integer overflow)
- **Provides cryptographic-level assurance** for safety-critical code
- **Complements traditional testing** by proving properties that tests can only sample

### Verification Results Example

```
Checking harness verify_weight_indexing...
VERIFICATION:- SUCCESSFUL

Checking harness verify_layer_access...
VERIFICATION:- SUCCESSFUL

Checking harness verify_activation_bounds...
VERIFICATION:- SUCCESSFUL

Summary:
Total: 15 harnesses
Successful: 15
Failed: 0
```

---

## **CISA/NSA Compliance**

### Secure by Design

This project follows **CISA (Cybersecurity and Infrastructure Security Agency)** and **NSA (National Security Agency)** Secure by Design principles:

| Principle | Implementation |
|-----------|---------------|
| **Memory Safety** | Rust ownership model eliminates buffer overflows, use-after-free, and data races |
| **Formal Verification** | Kani proofs mathematically verify absence of critical bugs |
| **Input Validation** | All CLI and API inputs validated before processing |
| **Defense in Depth** | Multiple layers of safety (language, compiler, runtime checks) |
| **Secure Defaults** | Safe default configurations throughout (e.g., CPU backend fallback) |
| **Transparency** | Open source with full code visibility |
| **Least Privilege** | Minimal unsafe code blocks, isolated where necessary |
| **Fail-Safe Defaults** | Graceful degradation when GPU unavailable |

### Compliance Checklist

- [x] **Memory-safe language** (Rust core implementation)
- [x] **Static analysis** (Rust compiler + Clippy)
- [x] **Formal verification** (Kani proof harnesses for critical paths)
- [x] **Comprehensive testing** (Unit tests + integration tests + property-based tests)
- [x] **Bounds checking** (Verified array access with proofs)
- [x] **Input validation** (CLI, Python API, Node.js API argument parsing)
- [x] **No unsafe code in critical paths** (Isolated to GPU FFI boundaries)
- [x] **Documentation** (Inline docs + comprehensive README + API docs)
- [x] **Version control** (Git with semantic versioning)
- [x] **License clarity** (MIT License)
- [x] **Dependency management** (Cargo.lock for reproducible builds)
- [x] **Supply chain security** (Audited dependencies)

### Security Features

| Feature | Description |
|---------|-------------|
| **Type Safety** | Strong typing prevents type confusion vulnerabilities |
| **Lifetime Safety** | Rust lifetimes prevent dangling pointers |
| **Thread Safety** | Send/Sync traits ensure safe concurrency |
| **Error Handling** | Result types force explicit error handling |
| **Panic Safety** | Critical sections proven panic-free via Kani |
| **FFI Safety** | Minimal unsafe boundaries with thorough validation |

### Attestation

This codebase has been developed following secure software development lifecycle (SSDLC) practices and demonstrates:

- **Comprehensive test suites** across all implementations (Python, Node.js, CLI, Rust)
- **Zero warnings** compilation across all implementations
- **Formal verification** of critical safety properties
- **Consistent API** across all language/backend combinations
- **Production-ready** code quality with extensive documentation
- **Security-first design** with defense in depth

---

## **License**

MIT License

Copyright (c) 2025 Matthew Abbott

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

---

## **Author**

**Matthew Abbott**  
Email: mattbachg@gmail.com

---

*Built with precision. Verified with rigor. Secured by design.*
