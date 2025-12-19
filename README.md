# GlassBoxAI-MLP

CUDA-Accelerated Multi-Layer Perceptron  
**Transparent, Hackable, and Fast — with CLI and Facade**  
**By [Matthew Abbott](https://github.com/matthewJamesAbbott)**

---

## What is GlassBoxAI-MLP?

GlassBoxAI-MLP is an open-source, ultra-transparent neural network toolkit in C++ & CUDA, designed for **full control and total auditability**.  
It's a powerful alternative to black-box frameworks, exposing *every* detail — weights, activations, hyperparameters, persistence, and CLI-driven workflows — with blazing GPU performance.

- **Glassbox ML:** Inspect, hack, extend, and *trust* your models
- **CUDA-accelerated:** Fast training/inference on your GPU, CUDA 11+ compatible
- **Full CLI Facade:** Create, train, predict, and inspect models with no code
- **MIT Licensed:** Yours to use, modify, and commercialize

---

## Features

- Multi-layer perceptron (MLP) with **arbitrary topology**
- **CUDA-accelerated** batch processing and SGD, Adam, RMSProp optimizers
- All major activations: sigmoid, tanh, ReLU, softmax
- **Dropout**, **L2 regularization**, **Xavier/He initialization**
- Learning rate decay, **early stopping**, normalization
- CLI with full **facade**: *create*, *train*, *predict*, *info*
- **Model save/load** to portable binary (with magic/versioning)
- **Glassbox**: Print model structure, hyperparameters, and architecture live
- Example: XOR and arbitrary CSV datasets

---

## File Structure

- [`mlp.cu`](mlp.cu): Core, clean and minimal CLI CUDA MLP (no facade)
- [`facaded_mlp.cu`](facaded_mlp.cu): CLI CUDA MLP *with* full Pascal-style *Facade* pattern & options
- `examples/`: Example CSV datasets for testing (e.g. `xor.csv`)
- `LICENSE`: MIT License

---

## Build Requirements

- NVIDIA GPU with CUDA support
- CUDA toolkit (nvcc in PATH)
- Standard C++ compiler (g++/clang++)
- Linux, Windows (WSL should work), macOS (NVIDIA only)

#### **Compile (facade version):**
```bash
nvcc -o mlpcuda facaded_mlp.cu -lcurand
```

#### **Compile (core version):**
```bash
nvcc -o mlp_cuda mlp.cu -lcurand
```

---

## Usage Examples (with CLI)

**Help:**
```
./mlpcuda help
```

**Create a model (8 hidden, sigmoid, Adam):**
```
./mlpcuda create --input=2 --hidden=8 --output=1 --optimizer=adam --lr=0.1 --save=xor.bin
```

**Train with CSV data, 2000 epochs, verbose:**
```
./mlpcuda train --model=xor.bin --data=xor.csv --epochs=2000 --verbose --save=xor_trained.bin
```

**Predict:**
```
./mlpcuda predict --model=xor_trained.bin --input=1,0
```
Output:
```
Input: 1.0000, 0.0000
Output: 0.998760
```

**Model info:**
```
./mlpcuda info --model=xor_trained.bin
```
Output:
```
MLP Model Information (CUDA)
============================
GPU: NVIDIA GeForce RTX 3070 Laptop GPU
Input size: 2
Output size: 1
Hidden layers: 1
Layer sizes: 2 -> 8 -> 1
Hyperparameters...
```

---

## Facade Pattern: What Makes It Special?

- **Glassbox/interpretable:** Everything is user-inspectable
- **Pluggable IO:** Insert new input/output bridges (audio, MIDI, sensors, etc.)
- **CLI for *all* model operations - ideal for scripting, automation, or education**
- **Code is clean, hackable, inspired by Pascal's MLPFacade** — ready to extend for your own needs

---

## Example Dataset (XOR)

**xor.csv**
```
0,0,0
0,1,1
1,0,1
1,1,0
```

---

## Directory Layout

- `mlp.cu` - Minimal, non-facade CUDA MLP
- `facaded_mlp.cu` - Facade pattern, CLI-driven CUDA MLP (recommended for real CLI/automation use)
- `examples/xor.csv` - Example data
- `README.md` - This file
- `LICENSE` - MIT License

---

## Why GlassBoxAI-MLP?

- Remove the black box: *see and control everything*
- Hack friendly: add new facades, APIs, or domain bridges (audio, sensor, web, etc.)
- Verified: reproduces XOR perfectly, matches/trumps PyTorch or TensorFlow for classic ML tasks
- **MIT license:** commercial, research, and education use encouraged

---

## Contributing & Contact

- Fork, PRs, and issues welcome!
- Ideas and facade/protocol/plugin extensions highly encouraged
- [Sponsor Me](https://github.com/sponsors/matthewJamesAbbott) (optional)
- [GlassBoxAI.ORG](https://glassboxai.org) (coming soon!)

---

## License

**MIT License**  
© 2025 Matthew Abbott

---

*For full source, see [facaded_mlp.cu](facaded_mlp.cu) and [mlp.cu](mlp.cu).*
