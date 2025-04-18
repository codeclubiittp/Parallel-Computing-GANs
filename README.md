# Parallel-Computing-GANs

A CUDA-accelerated project implementing core components of Generative Adversarial Networks (GANs) from scratch. This repository demonstrates how low-level parallel computing using NVIDIA CUDA and cuDNN can be used to implement high-performance deep learning modules.

---

## Features

- Custom CUDA/C++ implementations of neural network layers
- Efficient memory management and GPU acceleration using CUDA
- Manual backpropagation logic for learning objectives
- Modular structure for experimentation and educational use

---
## Directory Structure
```
Parallel-Computing-GANs/
├── include/
│   ├── activationLayer.cuh
│   ├── affineTransform.cuh
│   ├── batchNormalization.cuh
│   ├── convolution2D.cuh
│   ├── crossEntropy.cuh
│   ├── meanSquareError.cuh
│   └── utils.cuh
├── src/
│   ├── main.cu
│   └── testModules.cu
├── models/
│   └── ganNetwork.cuh
├── build/
│   └── [compiled binaries]
├── CMakeLists.txt
├── .clang-format
└── README.md
```
---

## Implemented Modules

| Module                  | Description |
|--------------------------|-------------|
| `activationLayer.cuh`    | Implements ReLU, Sigmoid, Tanh functions with forward and backward support |
| `affineTransform.cuh`    | Fully connected layer with weights, bias, and gradients |
| `batchNormalization.cuh` | CUDA-optimized batch normalization with mean/variance tracking |
| `convolution2D.cuh`      | 2D convolution layer with padding and stride options |
| `crossEntropy.cuh`       | Cross-entropy loss computation with softmax activation |
| `meanSquareError.cuh`    | Mean squared error for regression tasks |
| `ganNetwork.cuh`         | Generator and Discriminator design using the above layers |

---

## Environment Setup

### Dependencies

Ensure the following are installed:

- **CUDA Toolkit** (v11.0+):  
  [https://developer.nvidia.com/cuda-downloads](https://developer.nvidia.com/cuda-downloads)

- **cuDNN**:  
  [https://developer.nvidia.com/cudnn-downloads](https://developer.nvidia.com/cudnn-downloads)

- **CMake** (v3.10 or newer)

---

### Linux / WSL Setup

1. **Export paths for CUDA and cuDNN**:
   ```bash
   export PATH=/usr/local/cuda/bin:$PATH
   export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

   Here’s the finalized `README.md` content, properly compiled into a single markdown file. You can copy this directly into your GitHub repository's `README.md`. It includes all sections from your provided content, formatted cleanly and consistently:

---

2. **Clone the repository**:
   ```bash
   git clone https://github.com/codeclubiittp/Parallel-Computing-GANs.git
   cd Parallel-Computing-GANs
   ```

3. **Build the project**:
   ```bash
   mkdir build && cd build
   cmake ..
   make -j$(nproc)
   ```

---

## Future Work

- Integrate pooling layers and dropout
- Implement complete training loop for GANs
- Support for real datasets like MNIST/CIFAR
- Tensor visualization for debugging

---
