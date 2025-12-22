# GlassBoxAI-MLP

**CUDA and OpenCL Accelerated, Fully Glassbox, Multi-Layer Perceptron Neural Network Toolkit**  
_By [Matthew Abbott](https://github.com/matthewJamesAbbott)_  
MIT Licensed | GPU Powered | CLI & Facade | Open and Hackable

---

## What is GlassBoxAI-MLP?

**A high-performance, fully-transparent, and extendable neural network toolkit in C++ & CUDA or OpenCL, designed for:**
- **Speed** (GPU-accelerated learning and inference)
- **Transparency** (inspect, modify, and audit *everything*)
- **Hackability** (facade pattern, CLI scripting, direct weight/optimizer access)
- **Open Source** (MIT License—yours to use, build on, and monetize!)

Built for audio, tabular data, education, science, and any hackable ML dream.

---

## Build Requirements

- **NVIDIA GPU** with CUDA support (CUDA 11+ recommended)
- **CUDA Toolkit** (`nvcc` in path)
- **C++ Compiler** (`g++`/`clang++`)
- Linux, Windows (WSL), or Mac (NVIDIA only)

### Compile core:
```bash
nvcc -o mlp_cuda mlp.cu -lcurand
```
### Compile facade:
```bash
nvcc -o mlpcuda facaded_mlp.cu -lcurand
```

---

## File Structure

- **mlp.cu**&nbsp;&nbsp;&mdash; Core CUDA MLP (minimal CLI)
- **facaded_mlp.cu**&nbsp;&nbsp;&mdash; CLI + Facade version (full internal inspection/modification)
- **LICENSE**&nbsp;&nbsp;&mdash; MIT License
- **README.md**&nbsp;&nbsp;&mdash; This file

---

## CLI Usage Overview

### Common Commands

```sh
create      # Create a new MLP model
train       # Train a model with a dataset CSV
predict     # Predict output from an input vector
info        # Display model metadata and hyperparameters
help        # Show usage and arguments
```

#### Example:
```sh
mlpcuda create --input=2 --hidden=8 --output=1 --optimizer=adam --save=model.bin
mlpcuda train --model=model.bin --data=xor_cuda.csv --epochs=1000 --save=model.bin
mlpcuda predict --model=model.bin --input=1,0
mlpcuda info --model=model.bin
```

### Advanced ("Facade") Commands

```sh
get-weight    # Query a specific weight value
set-weight    # Set a specific weight value
get-bias      # Query neuron bias
set-bias      # Set neuron bias
get-output    # Get neuron outputs after prediction
get-error     # Get neuron errors after training
layer-info    # Show full layer breakdown
histogram     # Activation/gradient histogram for a layer
get-optimizer # Display optimizer (Adam/RMSProp) state, M/V stats
```

---

## Quick Example: XOR with Minimal MLP (`mlp.cu`)

### Create & Train

```sh
./mlp_cuda create --input=2 --hidden=8 --output=1 --optimizer=adam --lr=0.1 --save=xor.bin
./mlp_cuda train --model=xor.bin --data=xor_cuda.csv --epochs=2000 --verbose --save=xor_trained.bin
```

_Output Snippet:_
```
Created CUDA MLP model (GPU: NVIDIA GeForce RTX 3070 Laptop GPU):
  Input size: 2
  Hidden sizes: 8
  Output size: 1
  Hidden activation: sigmoid
  Output activation: sigmoid
  Optimizer: adam
  Learning rate: 0.1000

Epoch 2000/2000 - Loss: 0.000002
Final loss: 0.000002
Model saved to: xor_trained.bin
```

### Predict & Inspect
```sh
./mlp_cuda predict --model=xor_trained.bin --input=0,1
# Output: 0.998876

./mlp_cuda info --model=xor_trained.bin
```
_Sample Output:_
```
MLP Model Information (CUDA)
============================
GPU: NVIDIA GeForce RTX 3070 Laptop GPU
Input size: 2
Output size: 1
Hidden layers: 1
Layer sizes: 2 -> 8 -> 1
Hyperparameters: [full dump...]
```

---

## Facade Demo: Inspect & Hack Internals (`facaded_mlp.cu`, `mlpcuda`)

### Create & Train
```sh
./mlpcuda create --input=2 --hidden=8 --output=1 --optimizer=adam --save=test.bin
./mlpcuda train --model=test.bin --data=xor_cuda.csv --epochs=1000 --save=test.bin
```
_Output:_
```
Created CUDA MLP model (GPU: NVIDIA GeForce RTX 3070 Laptop GPU):
  Input size: 2
  Hidden sizes: 8
  Output size: 1
  Hidden activation: sigmoid
  Output activation: sigmoid
  Optimizer: adam
  Learning rate: 0.1000
  Saved to: test.bin

Final loss: 0.000007
Model saved to: test.bin
```

### Advanced Facade Actions
```sh
./mlpcuda get-weight --model=test.bin --layer=1 --neuron=0 --weight=0
# Weight[layer=1, neuron=0, weight=0] = -8.6098613769

./mlpcuda get-weights --model=test.bin --layer=1 --neuron=0
# [0] = -8.6098613769
# [1] = 5.8026426096
# [2] = -1.6070945553

./mlpcuda get-bias --model=test.bin --layer=1 --neuron=0
# Bias[layer=1, neuron=0] = -0.9544519568

./mlpcuda get-output --model=test.bin --layer=1 --run-input=1,0
# Outputs[layer=1] (9 neurons): [0] = 0.0000140706 ... [8] = 0.9900115786

./mlpcuda histogram --model=test.bin --layer=1 --type=activation --run-input=1,0
# Activation Histogram [layer=1] (20 bins): [ 0] 4 |######################################## ...

./mlpcuda set-weight --model=test.bin --layer=1 --neuron=0 --weight=0 --value=0.5 --save=test.bin
# Weight[layer=1, neuron=0, weight=0]: -8.6098613769 -> 0.5000000000
# Saved to: test.bin
```

---

## Help Screens

### Core CLI Help
```
MLPCuda - CUDA Command-line Multi-Layer Perceptron

Commands: create, train, predict, info, help
Options: --input, --hidden, --output, --save, --lr, --optimizer, --hidden-act, --output-act, --dropout, --l2, --beta1, --beta2, etc.
Examples:
  mlpcuda create --input=2 --hidden=4,4 --output=1 --save=xor.bin
```

### Facade Help
```
Facade Commands:
  get-weight, set-weight, get-bias, set-bias, get-output, get-error, layer-info, histogram, get-optimizer
Facade Options:
  --layer=N, --neuron=N, --weight=N, --value=V, --type=TYPE, --bins=N, --run-input, --save
Facade Examples:
  mlpcuda get-weight --model=m.bin --layer=1 --neuron=0 --weight=2
  mlpcuda histogram --model=m.bin --layer=1 --type=activation --run-input=1,0
```

---

## Why GlassBoxAI-MLP?

- **Perfect reproducible XOR and classic ML fits (see above)**
- **Built for hacking:** adjust weights, biases, optimizer states, activations, and more at runtime—no black-box barriers!
- **Full CLI for scripting, automation, and reproducibility**
- **MIT license—use in research, commercial, educational, or personal projects**

---

## Example Dataset (XOR)

_Create `xor_cuda.csv`:_
```
0,0,0
0,1,1
1,0,1
1,1,0
```

---

## Contributing & Sponsoring

- Fork, file issues, send PRs!
- [Sponsor @matthewJamesAbbott](https://github.com/sponsors/matthewJamesAbbott)
- Extending interfaces, adding more facades, protocols, new optimizers and layers always welcome!

---

## License

MIT License  
© 2025 Matthew Abbott

---

**For full source code:**  
- [facaded_mlp.cu](facaded_mlp.cu)
- [mlp.cu](mlp.cu)

**[GlassBoxAI.org](https://glassboxai.org) (coming soon for docs/community)**

---

*GlassBoxAI-MLP: Remove the black box—see, control, hack, and scale machine learning your way!*
