# GlassBoxAI-MLP

**Author:** Matthew Abbott (2025)

GlassBoxAI-MLP is a transparent, research-grade multi-layer perceptron (MLP) toolkit that offers high-performance CUDA and OpenCL implementations. It includes both minimal core models and advanced introspectable façade versions, with flexible options for creation, training, prediction, and deep model inspection. The repository is designed for reproducibility, extension, and total visibility into neural network internals, ideal for both learning and advanced research.

---

## Table of Contents

- [Features](#features)
- [Module Overview](#module-overview)
- [Requirements](#requirements)
- [Quickstart: Compiling & Running](#quickstart-compiling--running)
- [CLI Usage and Help](#cli-usage-and-help)
  - [1. CUDA Command-Line Model (`mlp.cu`)](#1-cuda-command-line-model-mlpcu)
  - [2. OpenCL Command-Line Model (`mlp_opencl.cpp`)](#2-opencl-command-line-model-mlp_openclcpp)
  - [3. CUDA Facade (`facaded_mlp.cu`)](#3-cuda-facade-facaded_mlpcu)
  - [4. OpenCL Facade (`facaded_mlp_opencl.cpp`)](#4-opencl-facade-facaded_mlp_openclcpp)
    - [OpenCL Facade CLI Example](#opencl-facade-cli-example)
    - [All Facade Introspection Options](#all-facade-introspection-options)
- [Architecture Notes](#architecture-notes)
- [Data Structures & Internals](#data-structures--internals)
- [License](#license)

---

## Features

- Pure, dependency-free CUDA and OpenCL MLPs
- **Two styles:**
  - **Core MLP** (`mlp.cu`, `mlp_opencl.cpp`): Minimal API for scripting/production
  - **Facade MLP** (`facaded_mlp.cu`, `facaded_mlp_opencl.cpp`): Deep CLI for research and teaching
- Support for SGD, Adam, RMSProp, Dropout, L2, lr decay, early stopping, batch training
- Model save/load, flexible hidden depths, thorough argument parsing
- All classic and modern activations (`sigmoid`, `tanh`, `relu`, `softmax`, and `linear`)
- CLI tools for gradient and activation inspection, weight/bias editing, and more
- Designed for maximum hackability and reproducibility—every number can be inspected

---

## Module Overview

There are **2 x 2 = 4 modes**:

| Type      | Core CLI Model      | Facade/Introspectable   |
|-----------|---------------------|-------------------------|
| CUDA      | `mlp.cu`            | `facaded_mlp.cu`        |
| OpenCL    | `mlp_opencl.cpp`    | `facaded_mlp_opencl.cpp`|

**Core** = minimal interface for creation/training/prediction  
**Facade** = rich CLI for inspection, hacking, and research

---

## Requirements

- **CUDA** (`mlp.cu`, `facaded_mlp.cu`): NVIDIA GPU, CUDA Toolkit 11+, C++11
- **OpenCL** (`mlp_opencl.cpp`, `facaded_mlp_opencl.cpp`): OpenCL 1.2+ device, C++11
- C++ build tools: `g++`, `nvcc`, or `clang++`
- *Optional*: CMake for integration

---

## Quickstart: Compiling & Running

**CUDA:**
```bash
# mlp.cu (core, minimal CLI)
nvcc -O2 -o mlp_cuda mlp.cu -lcurand

# facaded_mlp.cu (facade CLI)
nvcc -O2 -o facaded_mlp_cuda facaded_mlp.cu -lcurand
```

**OpenCL:**
```bash
# mlp_opencl.cpp (core)
g++ -O2 -std=c++14 -o mlp_opencl mlp_opencl.cpp -lOpenCL

# facaded_mlp_opencl.cpp (facade CLI)
g++ -O2 -std=c++14 -o facaded_mlp_opencl facaded_mlp_opencl.cpp -lOpenCL
```

---

## CLI Usage and Help

Below are usage templates and built-in help outputs for all modes, following the same pattern as the RNN repo. For each CLI, see `help` output for exhaustive argument info.

---

### 1. CUDA Command-Line Model (`mlp.cu`)

Minimal, scriptable CLI; no facade.  
**Show help:**
```bash
./mlp_cuda help
```

#### Example Help Output (abridged):
```
MLP CUDA - Command-line Multi-Layer Perceptron
Matthew Abbott 2025

Commands:
  create   Create a new MLP model
  train    Train an existing model with data
  predict  Make predictions with a trained model
  info     Display model information
  help     Show this help message

Create Options:
  --input=N              Input layer size (required)
  --hidden=N,N,...       Hidden layer sizes (required)
  --output=N             Output layer size (required)
  --save=FILE            Save model to file (required)
  --lr=VALUE             Learning rate (default: 0.1)
  --optimizer=TYPE       sgd|adam|rmsprop (default: sgd)
  --hidden-act=TYPE      sigmoid|tanh|relu|softmax|linear (default: sigmoid)
  --output-act=TYPE      sigmoid|tanh|relu|softmax|linear (default: sigmoid)
  --dropout=VALUE        Dropout rate 0–1 (default: 0)
  --l2=VALUE             L2 regularization (default: 0)
  --beta1=VALUE          Adam beta1 (default: 0.9)
  --beta2=VALUE          Adam beta2 (default: 0.999)
  --clip=VALUE           Gradient clipping value (default: 5.0)
  --loss=TYPE            mse|crossentropy (default: mse)

Train Options:
  --model=FILE           Model file to load (required)
  --data=FILE            Training data CSV file (required)
  --save=FILE            Save trained model to file (required)
  --epochs=N             Number of epochs (default: 100)
  --batch=N              Batch size (default: 1)
  --lr=VALUE             Override learning rate
  --clip=VALUE           Override gradient clipping
  --lr-decay             Enable learning rate decay
  --lr-decay-rate=VALUE  LR decay rate (default: 0.95)
  --lr-decay-epochs=N    Epochs between decay (default: 10)
  --early-stop           Enable early stopping
  --patience=N           Early stopping patience (default: 10)
  --normalize            Normalize input data
  --verbose              Show training progress

Predict Options:
  --model=FILE           Model file to load (required)
  --input=v1,v2,...      Input values (required)

Info Options:
  --model=FILE           Model file to load (required)

Examples:
  mlp_cuda create --input=2 --hidden=4,4 --output=1 --save=xor.bin
  mlp_cuda train --model=xor.bin --data=xor.csv --epochs=1000 --save=xor_trained.bin
  mlp_cuda predict --model=xor_trained.bin --input=1,0
  mlp_cuda info --model=xor_trained.bin
```

#### Example Usage
```bash
./mlp_cuda create --input=2 --hidden=8 --output=1 --save=model.bin
./mlp_cuda train --model=model.bin --data=xor.csv --epochs=1000 --save=model.bin
./mlp_cuda predict --model=model.bin --input=1,0
./mlp_cuda info --model=model.bin
```

---

### 2. OpenCL Command-Line Model (`mlp_opencl.cpp`)

Identical logic to CUDA core, with OpenCL backend and nearly identical arguments.

**Show help:**
```bash
./mlp_opencl help
# or just run with no arguments
```

#### Example Help Output (abridged):
```
MLP OpenCL - Command-line Multi-Layer Perceptron
Matthew Abbott 2025

Commands:
  create   Create a new MLP model
  train    Train an existing model with data
  predict  Make predictions with a trained model
  info     Display model information
  help     Show this help message

Options and arguments are identical to the CUDA version.
```

---

### 3. CUDA Facade (`facaded_mlp.cu`)

All minimal commands **plus dozens of facade CLI tools for deep model analysis**.

**Show help:**
```bash
./facaded_mlp_cuda help
```

#### Example Help Output (abridged):
```
MLP CUDA Facade - Introspectable Multi-Layer Perceptron
Matthew Abbott 2025

Commands (core):
  create        Create a new MLP model
  train         Train model with data
  predict       Predict with a model
  info          Display model info
  help          Show help

Facade/Introspection Commands:
  get-weight        Query a specific weight value
  set-weight        Set a specific weight value
  get-weights       Query all weights for one neuron
  get-bias          Query specific neuron bias
  set-bias          Set neuron bias
  get-output        Get neuron output for a given input
  get-error         Get neuron error after training
  layer-info        Print layer-wise breakdown
  histogram         Activation/gradient histogram for a layer
  get-optimizer     Show optimizer (Adam/RMSProp) state

General Options:
  --input=N              Input layer size
  --hidden=N,N,...       Hidden layer sizes
  --output=N             Output layer size
  --save=FILE            Save model to file
  --model=FILE           Model file to load
  --data=FILE            Input or training data
  --lr=VALUE             Learning rate
  --optimizer=TYPE       sgd|adam|rmsprop
  --hidden-act=TYPE      sigmoid|tanh|relu|softmax|linear
  --output-act=TYPE      sigmoid|tanh|relu|softmax|linear
  --dropout=VALUE        Dropout rate (0–1)
  --run-input=v1,v2,...  Input for analysis commands

Facade/Advanced Options:
  --layer=N              Layer index
  --neuron=N             Neuron index
  --weight=N             Weight index (for get/set)
  --value=V              Value to set
  --bins=N               Histogram bins (default: 20)
  --type=TYPE            Histogram: activation|gradient

Examples:
  facaded_mlp_cuda get-weight --model=net.bin --layer=1 --neuron=2 --weight=4
  facaded_mlp_cuda set-bias --model=net.bin --layer=2 --neuron=0 --value=0.1 --save=newmodel.bin
  facaded_mlp_cuda histogram --model=net.bin --layer=1 --type=activation --run-input=1,0
```

---

### 4. OpenCL Facade (`facaded_mlp_opencl.cpp`)

Same as CUDA facade, but with OpenCL backend.

**Show help:**
```bash
./facaded_mlp_opencl help
```

#### Example Help Output (abridged):
```
MLP OpenCL Facade - Introspectable Multi-Layer Perceptron
Matthew Abbott 2025

Commands and options identical to CUDA facade. All advanced facade commands available as listed above.
```

---

#### OpenCL Facade CLI Example

```bash
# Create an MLP (OpenCL)
./facaded_mlp_opencl create --input=2 --hidden=8 --output=1 --optimizer=adam --save=modelcl.bin

# Train
./facaded_mlp_opencl train --model=modelcl.bin --data=xor.csv --epochs=2000 --save=modelcl.bin

# Inspect a weight
./facaded_mlp_opencl get-weight --model=modelcl.bin --layer=1 --neuron=0 --weight=0

# Print output for a neuron after running input
./facaded_mlp_opencl get-output --model=modelcl.bin --layer=1 --neuron=0 --run-input=1,0

# Plot an activation histogram for a layer
./facaded_mlp_opencl histogram --model=modelcl.bin --layer=1 --type=activation --run-input=1,0
```

---

#### All Facade Introspection Options

- `get-weight`, `set-weight`: Query/set an individual weight value
- `get-weights`: Get all weights for a neuron
- `get-bias`, `set-bias`: Get/set neuron bias
- `get-output`: Get output of neuron/layer for given input
- `get-error`: Print current error value for neuron/layer
- `layer-info`: Full breakdown for one layer (weights, biases, activations)
- `histogram`: Create and print histogram of activations or gradients/errors
- `get-optimizer`: Print current optimizer state (Adam/RMSProp)
- All regular CLI options for creation, training, prediction, info

---

## License

MIT License  
© 2025 Matthew Abbott

---
