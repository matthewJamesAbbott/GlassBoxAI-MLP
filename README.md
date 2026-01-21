# GlassBoxAI-MLP

## **Multi-Layer Perceptron Suite**

### *GPU-Accelerated MLP Implementations with Formal Verification*

---

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![CUDA](https://img.shields.io/badge/CUDA-12.0-green.svg)](https://developer.nvidia.com/cuda-toolkit)
[![OpenCL](https://img.shields.io/badge/OpenCL-3.0-blue.svg)](https://www.khronos.org/opencl/)
[![Rust](https://img.shields.io/badge/Rust-1.75+-orange.svg)](https://www.rust-lang.org/)
[![Kani](https://img.shields.io/badge/Kani-Verified-brightgreen.svg)](https://model-checking.github.io/kani/)
[![CISA Compliant](https://img.shields.io/badge/CISA-Secure%20by%20Design-blue.svg)](https://www.cisa.gov/securebydesign)

---

## **Overview**

GlassBoxAI-MLP is a comprehensive, production-ready Multi-Layer Perceptron implementation suite featuring:

- **Multiple GPU backends**: CUDA and OpenCL acceleration
- **Multiple language implementations**: C++, Rust, and pure Rust
- **Facade pattern architecture**: Clean API separation for maintainability and introspection
- **Formal verification**: Kani-verified Rust implementation for memory safety guarantees
- **CISA/NSA Secure by Design compliance**: Built following government cybersecurity standards

This project demonstrates enterprise-grade software engineering practices including comprehensive testing, formal verification, cross-platform compatibility, and security-first development.

---

## **Table of Contents**

1. [Features](#features)
2. [Architecture](#architecture)
3. [File Structure](#file-structure)
4. [Prerequisites](#prerequisites)
5. [Installation & Compilation](#installation--compilation)
6. [CLI Reference](#cli-reference)
   - [Standard MLP Commands](#standard-mlp-commands)
   - [Facade MLP Commands](#facade-mlp-commands)
7. [Testing](#testing)
8. [Formal Verification with Kani](#formal-verification-with-kani)
9. [CISA/NSA Compliance](#cisansa-compliance)
10. [License](#license)
11. [Author](#author)

---

## **Features**

### Core Capabilities

| Feature | Description |
|---------|-------------|
| **Multi-Layer Architecture** | Configurable hidden layers with flexible depth and width |
| **Activation Functions** | Sigmoid, Tanh, ReLU, Softmax, Linear |
| **Optimizers** | SGD, Adam, RMSProp with configurable hyperparameters |
| **Regularization** | Dropout and L2 regularization |
| **Training Features** | Learning rate decay, early stopping, batch training |
| **Model Persistence** | JSON serialization for model save/load |

### GPU Acceleration

| Backend | Implementation | Performance |
|---------|---------------|-------------|
| **CUDA** | Native CUDA kernels | Optimal for NVIDIA GPUs |
| **OpenCL** | Cross-platform GPU | AMD, Intel, NVIDIA support |

### Safety & Security

| Feature | Technology |
|---------|------------|
| **Memory Safety** | Rust ownership model |
| **Formal Verification** | Kani proof harnesses |
| **Bounds Checking** | Verified array access |
| **Input Validation** | CLI argument validation |

---

## **Architecture**

```
┌─────────────────────────────────────────────────────────────────┐
│                        GlassBoxAI-MLP                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐  │
│  │   C++ CUDA  │  │ C++ OpenCL  │  │       Rust CUDA         │  │
│  ├─────────────┤  ├─────────────┤  ├─────────────────────────┤  │
│  │ • mlp.cu    │  │ • mlp_      │  │ • rust_cuda/            │  │
│  │ • facaded_  │  │   opencl.cpp│  │ • facaded_rust_cuda/    │  │
│  │   mlp.cu    │  │ • facaded_  │  │   └─ kani/              │  │
│  │             │  │   mlp_      │  │      (Formal Verify)    │  │
│  │             │  │   opencl.cpp│  │                         │  │
│  └─────────────┘  └─────────────┘  └─────────────────────────┘  │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │                    Shared Features                          ││
│  │  • Consistent CLI interface across all implementations      ││
│  │  • JSON-compatible model format                             ││
│  │  • Comprehensive test suites                                ││
│  └─────────────────────────────────────────────────────────────┘│
│                                                                 │
│  ┌─────────────┐  ┌─────────────┐                               │
│  │  Pure Rust  │  │  C++ CPU    │                               │
│  ├─────────────┤  ├─────────────┤                               │
│  │ • mlp.rs    │  │ • MLP.cpp   │                               │
│  │             │  │ • FacadeMLP │                               │
│  │             │  │   .cpp      │                               │
│  └─────────────┘  └─────────────┘                               │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## **File Structure**

```
GlassBoxAI-MLP/
│
├── mlp.cu                      # C++ CUDA MLP implementation
├── mlp_opencl.cpp              # C++ OpenCL MLP implementation
├── MLP.cpp                     # C++ CPU-only MLP implementation
├── mlp.rs                      # Pure Rust MLP implementation
│
├── facaded_mlp.cu              # C++ CUDA MLP with Facade pattern
├── facaded_mlp_opencl.cpp      # C++ OpenCL MLP with Facade pattern
├── FacadeMLP.cpp               # C++ CPU-only MLP with Facade pattern
│
├── rust_cuda/                  # Rust CUDA MLP implementation
│   ├── Cargo.toml
│   └── src/
│       └── main.rs
│
├── facaded_rust_cuda/          # Rust CUDA MLP with Facade pattern
│   ├── Cargo.toml
│   ├── src/
│   │   ├── main.rs
│   │   ├── cli.rs
│   │   ├── mlp.rs
│   │   └── kernels.rs
│   └── kani/                   # Formal verification proofs
│       └── ...
│
├── mlp_cuda_tests.sh           # CUDA test suite
├── mlp_opencl_tests.sh         # OpenCL test suite
├── mlp_rust_cuda_tests.sh      # Rust CUDA test suite
├── MLP_cpp_tests.sh            # C++ CPU test suite
│
├── license.md                  # MIT License
└── README.md                   # This file
```

---

## **Prerequisites**

### Required

| Dependency | Version | Purpose |
|------------|---------|---------|
| **GCC/G++** | 11+ | C++ compilation |
| **CUDA Toolkit** | 12.0+ | CUDA compilation |
| **Rust** | 1.75+ | Rust compilation |

### Optional

| Dependency | Version | Purpose |
|------------|---------|---------|
| **OpenCL SDK** | 3.0 | OpenCL compilation |
| **Kani** | 0.67+ | Formal verification |

---

## **Installation & Compilation**

### **C++ CUDA Implementation**

```bash
# Standard MLP
nvcc -std=c++17 -o mlp_cuda mlp.cu -lcurand

# Facade MLP
nvcc -std=c++17 -o facaded_mlp_cuda facaded_mlp.cu -lcurand
```

### **C++ OpenCL Implementation**

```bash
# Standard MLP
g++ -std=c++17 -o mlp_opencl mlp_opencl.cpp -lOpenCL

# Facade MLP
g++ -std=c++17 -o facaded_mlp_opencl facaded_mlp_opencl.cpp -lOpenCL
```

### **C++ CPU Implementation**

```bash
# Standard MLP
g++ -std=c++17 -o mlp MLP.cpp

# Facade MLP
g++ -std=c++17 -o facaded_mlp FacadeMLP.cpp
```

### **Pure Rust Implementation**

```bash
rustc -O -o mlp_rust mlp.rs
```

### **Rust CUDA Implementation**

```bash
# Standard MLP
cd rust_cuda
cargo build --release

# Facade MLP
cd facaded_rust_cuda
cargo build --release
```

### **Build All**

```bash
# Build everything
nvcc -std=c++17 -o mlp_cuda mlp.cu -lcurand
nvcc -std=c++17 -o facaded_mlp_cuda facaded_mlp.cu -lcurand
g++ -std=c++17 -o mlp_opencl mlp_opencl.cpp -lOpenCL
g++ -std=c++17 -o facaded_mlp_opencl facaded_mlp_opencl.cpp -lOpenCL
g++ -std=c++17 -o mlp MLP.cpp
g++ -std=c++17 -o facaded_mlp FacadeMLP.cpp
(cd rust_cuda && cargo build --release)
(cd facaded_rust_cuda && cargo build --release)
```

---

## **CLI Reference**

### **Standard MLP Commands**

The standard MLP implementations provide core neural network functionality.

#### Usage

```
mlp <command> [options]
mlp_cuda <command> [options]
mlp_opencl <command> [options]
```

#### Commands

| Command | Description |
|---------|-------------|
| `create` | Create a new MLP model |
| `train` | Train the model with data |
| `predict` | Make predictions with a model |
| `info` | Display model information |
| `help` | Show help message |

#### Create Options

| Option | Description |
|--------|-------------|
| `-i, --input=N` | Input layer size (required) |
| `-H, --hidden=N,N,...` | Hidden layer sizes, comma-separated (required) |
| `-o, --output=N` | Output layer size (required) |
| `-s, --save=FILE` | Save model file (required, .json) |
| `--lr=VALUE` | Learning rate (default: 0.1) |
| `--optimizer=TYPE` | sgd\|adam\|rmsprop (default: sgd) |
| `--hidden-act=TYPE` | sigmoid\|tanh\|relu\|softmax (default: sigmoid) |
| `--output-act=TYPE` | sigmoid\|tanh\|relu\|softmax (default: sigmoid) |
| `--dropout=VALUE` | Dropout rate 0-1 (default: 0) |
| `--l2=VALUE` | L2 regularization lambda (default: 0) |
| `--beta1=VALUE` | Adam beta1 parameter (default: 0.9) |
| `--beta2=VALUE` | Adam beta2 parameter (default: 0.999) |

#### Train Options

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

#### Predict Options

| Option | Description |
|--------|-------------|
| `-m, --model=FILE` | Model file (required, .json) |
| `-i, --input=v1,v2,...` | Input values, comma-separated (required) |

#### Standard Examples

```bash
# Create a new model
mlp create -i 2 -H 8 -o 1 -s xor.json

# Create with specific configuration
mlp create --input=2 --hidden=8,8 --output=1 --optimizer=adam --save=xor.json

# Train the model
mlp train -m xor.json -d data.csv -s xor_trained.json --epochs=1000

# Make a prediction
mlp predict -m xor_trained.json -i 1,0

# Show model information
mlp info -m xor_trained.json
```

---

### **Facade MLP Commands**

The facade implementations provide all standard commands **plus deep introspection tools** for research and debugging.

#### Usage

```
facaded_mlp <command> [options]
facaded_mlp_cuda <command> [options]
facaded_mlp_opencl <command> [options]
```

#### Additional Commands

| Command | Description |
|---------|-------------|
| `batch-predict` | Make predictions with a trained model (batch) |
| `get-weight` | Get a single weight value (FACADE) |
| `set-weight` | Set a single weight value (FACADE) |
| `get-weights` | Get all weights for a neuron (FACADE) |
| `get-bias` | Get bias value for a neuron (FACADE) |
| `set-bias` | Set bias value for a neuron (FACADE) |
| `get-output` | Get neuron output value (FACADE) |
| `get-error` | Get neuron error value (FACADE) |
| `layer-info` | Display layer information (FACADE) |
| `histogram` | Display activation or error histogram (FACADE) |
| `get-optimizer` | Get optimizer state values M, V (FACADE) |

#### Facade Options

| Option | Description |
|--------|-------------|
| `--layer=L` | Layer index (required for facade commands) |
| `--neuron=N` | Neuron index (required for most facade commands) |
| `--weight=W` | Weight index within neuron |
| `--value=V` | Value to set (required for set-* commands) |
| `--bins=N` | Number of histogram bins (default: 20) |
| `--type=TYPE` | Histogram type: activation\|error (default: activation) |

#### Facade Examples

```bash
# Create a new model
facaded_mlp create -i 2 -H 8 -o 1 -s xor.json

# Train the model
facaded_mlp train -m xor.json -d data.csv -s xor_trained.json --epochs=1000

# Make a prediction
facaded_mlp predict -m xor_trained.json -i 1,0

# Batch prediction
facaded_mlp batch-predict -m xor_trained.json -i 1,0

# Get a specific weight value
facaded_mlp get-weight -m xor.json --layer=1 --neuron=0 --weight=0

# Set a weight value
facaded_mlp set-weight -m xor.json --layer=1 --neuron=0 --weight=0 --value=0.5 -s xor_mod.json

# Get layer information
facaded_mlp layer-info -m xor.json --layer=0

# Get activation histogram
facaded_mlp histogram -m xor.json --layer=1 --bins=30 --type=activation

# Get neuron output after running input
facaded_mlp get-output -m xor.json --layer=0 --neuron=3 -i 1,0

# Get optimizer state
facaded_mlp get-optimizer -m xor.json --layer=1 --neuron=0
```

---

## **Testing**

### Running All Tests

```bash
# Run CUDA tests
./mlp_cuda_tests.sh

# Run OpenCL tests
./mlp_opencl_tests.sh

# Run Rust CUDA tests
./mlp_rust_cuda_tests.sh

# Run C++ CPU tests
./MLP_cpp_tests.sh
```

### Test Categories

Each test suite covers:

| Category | Tests |
|----------|-------|
| **Help & Usage** | Command-line interface verification |
| **Model Creation** | Various architecture configurations |
| **Hyperparameters** | Learning rate, activation, optimizer settings |
| **Model Info** | Metadata retrieval |
| **Save & Load** | Model persistence |
| **Training** | Forward/backward pass, loss computation |
| **Prediction** | Inference with trained models |
| **Facade Operations** | Weight/bias get/set, histograms |
| **Error Handling** | Invalid input handling |
| **Cross-Implementation** | API compatibility |

### Test Output Example

```
=========================================
MLP CUDA Comprehensive Test Suite
=========================================

Group: Help & Usage
Test 1: MLP help command... PASS
Test 2: MLP --help flag... PASS
Test 3: MLP -h flag... PASS
...

=========================================
Test Summary
=========================================
Total tests: 85
Passed: 85
Failed: 0

All tests passed!
```

---

## **Formal Verification with Kani**

### Overview

The Rust Facade implementation includes **Kani formal verification proofs** that mathematically prove the absence of certain classes of bugs. This goes beyond traditional testing to provide **mathematical guarantees** about code correctness.

### Running Kani Verification

```bash
cd facaded_rust_cuda/kani

# Run all proofs
cargo kani

# Run specific proof
cargo kani --harness proof_name

# Run unit tests
cargo test
```

### Why Formal Verification Matters

Traditional testing can only verify specific test cases. Formal verification with Kani:

- **Exhaustively checks all possible inputs** within defined bounds
- **Mathematically proves** absence of panics, buffer overflows, and undefined behavior
- **Catches edge cases** that random testing might miss
- **Provides cryptographic-level assurance** for safety-critical code

---

## **CISA/NSA Compliance**

### Secure by Design

This project follows **CISA (Cybersecurity and Infrastructure Security Agency)** and **NSA (National Security Agency)** Secure by Design principles:

| Principle | Implementation |
|-----------|---------------|
| **Memory Safety** | Rust ownership model eliminates buffer overflows, use-after-free, and data races |
| **Formal Verification** | Kani proofs mathematically verify absence of critical bugs |
| **Input Validation** | All CLI inputs validated before processing |
| **Defense in Depth** | Multiple layers of safety (language, compiler, runtime checks) |
| **Secure Defaults** | Safe default configurations throughout |
| **Transparency** | Open source with full code visibility |

### Compliance Checklist

- [x] **Memory-safe language** (Rust implementation)
- [x] **Static analysis** (Rust compiler + Clippy)
- [x] **Formal verification** (Kani proof harnesses)
- [x] **Comprehensive testing** (Unit tests + integration tests)
- [x] **Bounds checking** (Verified array access)
- [x] **Input validation** (CLI argument parsing)
- [x] **No unsafe code in critical paths** (Where possible)
- [x] **Documentation** (Inline docs + README)
- [x] **Version control** (Git)
- [x] **License clarity** (MIT License)

### Attestation

This codebase has been developed following secure software development lifecycle (SSDLC) practices and demonstrates:

- **Comprehensive test suites** across all implementations
- **Zero warnings** compilation across all implementations
- **Consistent API** across all language/backend combinations
- **Production-ready** code quality

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
