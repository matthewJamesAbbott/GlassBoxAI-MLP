# GlassBoxAI-MLP

**Author:** Matthew Abbott (2025)  
**MIT License | GPU-Accelerated | Fully Transparent | CLI & Facade**

---

A modern, high-performance CUDA and OpenCL Multi-Layer Perceptron (MLP) library for research, teaching, and advanced scripting. GlassBoxAI-MLP features both minimal/core and introspectable/facade versions in both CUDA and OpenCL. All binaries ship with powerful command-line interfaces for model creation, training, prediction, inspection, and direct access to model internals—on any compatible GPU.

---

## Table of Contents

- [Features](#features)
- [Module Types Overview](#module-types-overview)
- [Requirements](#requirements)
- [Build Instructions](#build-instructions)
- [Usage by Version](#usage-by-version)
  - [1. CUDA Core MLP (`mlp.cu`)](#1-cuda-core-mlp-mlpcu)
  - [2. OpenCL Core MLP (`mlp_opencl.cpp`)](#2-opencl-core-mlp-mlp_openclcpp)
  - [3. CUDA Facade/Introspectable (`facaded_mlp.cu`)](#3-cuda-facadeintrospectable-facaded_mlpcu)
  - [4. OpenCL Facade/Introspectable (`facaded_mlp_opencl.cpp`)](#4-opencl-facadeintrospectable-facaded_mlp_openclcpp)
- [CLI Command Reference](#cli-command-reference)
- [Advanced Facade Usage Examples](#advanced-facade-usage-examples)
- [Help & Examples](#help--examples)
- [License](#license)

---

## Features

- **CUDA or OpenCL MLP**: GPU-accelerated, pure and portable
- **Minimal/core** and **facade/introspectable** MLPs
- Training: SGD, Adam, RMSProp optimizers; dropout; batch; L2; grad clipping; early stopping; LR decay
- All classic and modern activations (`sigmoid`, `tanh`, `relu`, `softmax`, and `linear`)
- `save`/`load` binary model files, reproducibility first
- Powerful CLI for scripting, research, and automation
- **Facade:** CLI deep inspection and direct manipulation of weights, biases, neuron outputs, errors, optimizer state, and layer histograms
- Suitable for classic ML tasks, algorithmic research, and hands-on ML education

---

## Module Types Overview

|              | Core CLI Model         | Facade/Introspectable              |
|--------------|-----------------------|------------------------------------|
| **CUDA**     | `mlp.cu`              | `facaded_mlp.cu`                   |
| **OpenCL**   | `mlp_opencl.cpp`      | `facaded_mlp_opencl.cpp`           |

_All 4 variants are command-line tools with similar arguments and extended facade options for model introspection/hacking._

---

## Requirements

- **CUDA:** NVIDIA GPU, CUDA Toolkit 11+ (`nvcc`)
- **OpenCL:** Any OpenCL 1.2+ device, drivers, and headers (`g++`/`clang++` with `-lOpenCL`)
- **C++11+** compatible compiler (Linux, Windows, WSL, Mac w/ GPU)
- **No external deep learning libraries needed**

---

## Build Instructions

**CUDA:**
```bash
# Core MLP (minimal CLI)
nvcc -O2 -o mlp_cuda mlp.cu -lcurand

# Facade/Introspectable MLP
nvcc -O2 -o facaded_mlp_cuda facaded_mlp.cu -lcurand
```

**OpenCL:**
```bash
# Core MLP
g++ -std=c++14 -O2 -o mlp_opencl mlp_opencl.cpp -lOpenCL

# Facade/Introspectable
g++ -std=c++14 -O2 -o facaded_mlp_opencl facaded_mlp_opencl.cpp -lOpenCL
```
---

## Usage by Version

### 1. CUDA Core MLP (`mlp.cu`)

**Show help:**  
```sh
./mlp_cuda help
```
**Create & train:**
```sh
./mlp_cuda create --input=2 --hidden=8 --output=1 --save=model.bin
./mlp_cuda train --model=model.bin --data=xor.csv --epochs=1000 --save=model.bin
```

**Predict:**
```sh
./mlp_cuda predict --model=model.bin --input=0,1
```

### 2. OpenCL Core MLP (`mlp_opencl.cpp`)

Analogous usage to CUDA above.  
**Show help:**  
```sh
./mlp_opencl help
```

**Train:**
```sh
./mlp_opencl create --input=2 --hidden=8 --output=1 --save=modelcl.bin
./mlp_opencl train --model=modelcl.bin --data=xor_opencl.csv --epochs=1000 --save=modelcl.bin
```

**Predict:**
```sh
./mlp_opencl predict --model=modelcl.bin --input=0,1
```

### 3. CUDA Facade/Introspectable (`facaded_mlp.cu`)

Identical main commands, plus _dozens_ of facade actions:
```sh
./facaded_mlp_cuda get-weight --model=model.bin --layer=1 --neuron=0 --weight=0
./facaded_mlp_cuda set-weight --model=model.bin --layer=1 --neuron=0 --weight=0 --value=0.5 --save=model.bin
./facaded_mlp_cuda get-output --model=model.bin --layer=1 --neuron=0 --run-input=1,0
```

### 4. OpenCL Facade/Introspectable (`facaded_mlp_opencl.cpp`)

All facade commands as in CUDA, e.g.:
```sh
./facaded_mlp_opencl get-bias --model=modelcl.bin --layer=1 --neuron=0
./facaded_mlp_opencl set-bias --model=modelcl.bin --layer=1 --neuron=0 --value=0.1 --save=modelcl.bin
./facaded_mlp_opencl histogram --model=modelcl.bin --layer=1 --type=activation --run-input=1,0
```

---

## CLI Command Reference

_All tools use verb-based commands; options are standard GNU-style:_

| Command        | Meaning / Usage Example |
|----------------|------------------------|
| `create`       | `--input=N --hidden=N,N,... --output=N --save=file` |
| `train`        | `--model=file --data=data.csv --epochs=1000 --save=file` |
| `predict`      | `--model=file --input=val1,val2,...` |
| `info`         | `--model=file` |
| `help`         | Show options/help |

### Facade-only Commands (facaded_mlp_cuda/opencl only)

- `get-weight`, `set-weight`: Inspect/set single weight
- `get-weights`: Print all weights for a neuron
- `get-bias`, `set-bias`: Inspect/set neuron bias
- `get-output`, `get-error`: Print outputs/errors for neuron/layer
- `layer-info`: Print detailed neuron-by-neuron state
- `histogram`: Print activation/gradient histogram
- `get-optimizer`: Show optimizer M/V for any param

#### Facade Option Flags

- `--layer=N`, `--neuron=N`: Indices
- `--weight=N`: Weight index (for neuron)
- `--value=X`: New value for set commands
- `--run-input=...`: Use this input for analysis
- `--type=activation|gradient`: Histogram type

--- 

## Advanced Facade Usage Examples

```sh
# Set and get weights/biases
./facaded_mlp_cuda set-weight --model=model.bin --layer=1 --neuron=3 --weight=2 --value=1.5 --save=model.bin
./facaded_mlp_cuda get-weight --model=model.bin --layer=1 --neuron=3 --weight=2

# Layer activation histogram (using a manual input vector)
./facaded_mlp_cuda histogram --model=model.bin --layer=1 --type=activation --run-input=1,0

# Output for a neuron/layer given specific input
./facaded_mlp_cuda get-output --model=model.bin --layer=1 --neuron=0 --run-input=1,0
```

---

## Help & Examples

Each binary supports `help`:
```sh
./mlp_cuda help
./mlp_opencl help
./facaded_mlp_cuda help
./facaded_mlp_opencl help
```

Examples:
```sh
# Create XOR model, train, evaluate (CUDA)
./mlp_cuda create --input=2 --hidden=8 --output=1 --save=xor.bin
./mlp_cuda train --model=xor.bin --data=xor.csv --epochs=2000 --save=xor_trained.bin
./mlp_cuda predict --model=xor_trained.bin --input=1,0

# All CLI options shown by running "... help"
```

---

## License

MIT License  
© 2025 Matthew Abbott

---
