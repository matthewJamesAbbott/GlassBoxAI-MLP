# GlassBoxAI-MLP

**CUDA and OpenCL Accelerated Multi-Layer Perceptron Neural Network**  
_By [Matthew Abbott](https://github.com/matthewJamesAbbott)_  
MIT Licensed | GPU Powered | CLI & Facade | Open and Hackable

---

## What is GlassBoxAI-MLP?

A high-performance, fully-transparent, and extendable neural network toolkit in C++, CUDA, and OpenCL, designed for:

- **Speed** (GPU-accelerated learning and inference on CUDA or OpenCL devices)
- **Transparency** (inspect, modify, and audit *everything*)
- **Hackability** (facade pattern, CLI scripting, direct model/optimizer access)
- **Open Source** (MIT License—yours to use, build on, and monetize!)

Built for audio, tabular data, education, science, and hackable ML dreams.

---

## Build Requirements

- **C++ Compiler** (`g++`/`clang++`)
- **CUDA Toolkit** (`nvcc` in path, CUDA 11+ recommended) — for CUDA versions
- **OpenCL SDK/headers & driver** — for OpenCL versions
- Linux, Windows (WSL), or Mac (NVIDIA for CUDA; any OpenCL for OpenCL)

---

## Building GlassBoxAI-MLP

### CUDA Versions

- **Core MLP (CUDA):**
  ```bash
  nvcc -o mlp_cuda mlp.cu -lcurand
  ```
- **Facade/CLI (CUDA):**
  ```bash
  nvcc -o facaded_mlp_cuda facaded_mlp.cu -lcurand
  ```

### OpenCL Versions

- **Core MLP (OpenCL):**
  ```bash
  g++ -std=c++14 -o mlp_opencl mlp_opencl.cpp -lOpenCL
  ```
- **Facade/CLI (OpenCL):**
  ```bash
  g++ -std=c++14 -o facaded_mlp_opencl facaded_mlp_opencl.cpp -lOpenCL
  ```

Adjust compiler flags as needed for your platform.

---

## File Structure

- **mlp.cu:** Core CUDA MLP (minimal CLI)
- **facaded_mlp.cu:** CUDA facade/CLI, advanced operations
- **mlp_opencl.cpp:** Core OpenCL MLP (minimal CLI)
- **facaded_mlp_opencl.cpp:** OpenCL facade/CLI, advanced operations
- **license.md**: MIT License
- **README.md**: This file

---

## CLI Usage Overview

All binaries (CUDA and OpenCL) support similar CLI commands. CLI names may differ (`mlp_cuda`/`mlp_opencl`, `facaded_mlp_cuda`/`facaded_mlp_opencl`). Substitute the filename as appropriate.

### Common Commands

```sh
create      # Create a new MLP model
train       # Train a model with a dataset CSV
predict     # Predict output from an input vector
info        # Display model metadata and hyperparameters
help        # Show usage and arguments
```

#### Example (CUDA):
```sh
./facaded_mlp_cuda create --input=2 --hidden=8 --output=1 --optimizer=adam --save=model.bin
./facaded_mlp_cuda train --model=model.bin --data=xor_cuda.csv --epochs=1000 --save=model.bin
./facaded_mlp_cuda predict --model=model.bin --input=1,0
./facaded_mlp_cuda info --model=model.bin
```

#### Example (OpenCL):
```sh
./facaded_mlp_opencl create --input=2 --hidden=8 --output=1 --optimizer=adam --save=modelcl.bin
./facaded_mlp_opencl train --model=modelcl.bin --data=xor_opencl.csv --epochs=1000 --save=modelcl.bin
./facaded_mlp_opencl predict --model=modelcl.bin --input=1,0
./facaded_mlp_opencl info --model=modelcl.bin
```

### Advanced ("Facade") Commands

Available in `facaded_mlp_cuda` and `facaded_mlp_opencl`:

```sh
get-weight    # Query a specific weight value
set-weight    # Set a specific weight value
get-bias      # Query neuron bias
set-bias      # Set neuron bias
get-output    # Get neuron outputs after prediction
get-error     # Get neuron errors after training
layer-info    # Show full layer breakdown
histogram     # Activation/gradient histogram for a layer
get-optimizer # Display optimizer state, M/V stats
```

---

## Quick Example: XOR with Minimal MLP (CUDA & OpenCL)

### CUDA Example

```sh
./mlp_cuda create --input=2 --hidden=8 --output=1 --optimizer=adam --lr=0.1 --save=xor.bin
./mlp_cuda train --model=xor.bin --data=xor_cuda.csv --epochs=2000 --verbose --save=xor_trained.bin
./mlp_cuda predict --model=xor_trained.bin --input=0,1
```

### OpenCL Example

```sh
./mlp_opencl create --input=2 --hidden=8 --output=1 --optimizer=adam --lr=0.1 --save=xorcl.bin
./mlp_opencl train --model=xorcl.bin --data=xor_opencl.csv --epochs=2000 --verbose --save=xorcl_trained.bin
./mlp_opencl predict --model=xorcl_trained.bin --input=0,1
```

Outputs and options are similar between versions.

---

## Facade Demo: Inspect & Modify Internals

The facade binaries (`facaded_mlp_cuda`, `facaded_mlp_opencl`) expose advanced CLI actions for model research and hacking—on CUDA or OpenCL as you prefer.

#### Advanced Example (CUDA)
```sh
./facaded_mlp_cuda get-weight --model=model.bin --layer=1 --neuron=0 --weight=0
./facaded_mlp_cuda set-weight --model=model.bin --layer=1 --neuron=0 --weight=0 --value=0.5 --save=model.bin
./facaded_mlp_cuda histogram --model=model.bin --layer=1 --type=activation --run-input=1,0
```

#### Advanced Example (OpenCL)
```sh
./facaded_mlp_opencl get-weight --model=modelcl.bin --layer=1 --neuron=0 --weight=0
./facaded_mlp_opencl set-weight --model=modelcl.bin --layer=1 --neuron=0 --weight=0 --value=0.5 --save=modelcl.bin
./facaded_mlp_opencl histogram --model=modelcl.bin --layer=1 --type=activation --run-input=1,0
```

---

## Help Screens

Both CUDA and OpenCL CLIs support `help` commands with full argument info.

---

## Why GlassBoxAI-MLP?

- **Supports multiple backends:** CUDA or OpenCL for maximum compatibility
- **Perfect reproducible XOR and classic ML fits**
- **Full hackability:** adjust weights, biases, activations at runtime
- **Full CLI for scripting, automation, and reproducibility**
- **MIT license—use in research, commercial, educational, or personal projects**

---

## Example Dataset (XOR)

Example for low-dimensional XOR fitting:

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
- Adding new facades, protocols, optimizers, and layers always welcome!

---

## License

MIT License  
© 2025 Matthew Abbott
