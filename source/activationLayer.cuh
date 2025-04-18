#ifndef ACTIVATION_LAYER_H
#define ACTIVATION_LAYER_H

#include <cuda_runtime.h>

enum class ActivationType { RELU, LEAKY_RELU, SIGMOID, TANH };

void
apply_activation(float* input, int size, ActivationType activation_type, float alpha);
void
apply_activation_backward(float* input,
                          float* grad_output,
                          int size,
                          ActivationType activation_type,
                          float alpha,
                          float* grad_input);

#define BLOCK_SIZE 256  // Fixed block size for all kernels

// Kernel for ReLU activation (completely no if statements)
__global__ void
relu_kernel(float* input, int size) {
    int idx    = blockIdx.x * blockDim.x + threadIdx.x;
    idx        = min(idx, size - 1);       // Prevent out-of-bounds access, no if statement
    input[idx] = fmaxf(input[idx], 0.0f);  // ReLU: max(0, x) (no if statements)
}

// Kernel for Leaky ReLU activation (completely no if statements)
__global__ void
leaky_relu_kernel(float* input, int size, float alpha) {
    int idx    = blockIdx.x * blockDim.x + threadIdx.x;
    idx        = min(idx, size - 1);  // Prevent out-of-bounds access, no if statement
    input[idx] = input[idx] * (1.0f * (input[idx] > 0.0f)) + alpha * input[idx] * (1.0f - (input[idx] > 0.0f));
}

// Kernel for Sigmoid activation (completely no if statements)
__global__ void
sigmoid_kernel(float* input, int size) {
    int idx    = blockIdx.x * blockDim.x + threadIdx.x;
    idx        = min(idx, size - 1);                 // Prevent out-of-bounds access, no if statement
    input[idx] = 1.0f / (1.0f + expf(-input[idx]));  // Sigmoid: 1 / (1 + exp(-x)) (no if statements)
}

// Kernel for Tanh activation (completely no if statements)
__global__ void
tanh_kernel(float* input, int size) {
    int idx    = blockIdx.x * blockDim.x + threadIdx.x;
    idx        = min(idx, size - 1);  // Prevent out-of-bounds access, no if statement
    input[idx] = tanhf(input[idx]);   // Tanh (no if statements)
}

// Kernel for backward pass of ReLU (completely no if statements)
__global__ void
relu_backward_kernel(float* input, float* grad_output, int size, float* grad_input) {
    int idx         = blockIdx.x * blockDim.x + threadIdx.x;
    idx             = min(idx, size - 1);                               // Prevent out-of-bounds access, no if statement
    grad_input[idx] = grad_output[idx] * (1.0f * (input[idx] > 0.0f));  // Grad for ReLU
}

// Kernel for backward pass of Leaky ReLU (completely no if statements)
__global__ void
leaky_relu_backward_kernel(float* input, float* grad_output, int size, float alpha, float* grad_input) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    idx     = min(idx, size - 1);  // Prevent out-of-bounds access, no if statement
    grad_input[idx] =
        grad_output[idx] * (1.0f * (input[idx] > 0.0f) + alpha * (1.0f - (input[idx] > 0.0f)));  // Grad for Leaky ReLU
}

// Kernel for backward pass of Sigmoid (completely no if statements)
__global__ void
sigmoid_backward_kernel(float* input, float* grad_output, int size, float* grad_input) {
    int idx         = blockIdx.x * blockDim.x + threadIdx.x;
    idx             = min(idx, size - 1);  // Prevent out-of-bounds access, no if statement
    float sigmoid   = 1.0f / (1.0f + expf(-input[idx]));
    grad_input[idx] = grad_output[idx] * sigmoid * (1.0f - sigmoid);  // Grad for Sigmoid
}

// Kernel for backward pass of Tanh (completely no if statements)
__global__ void
tanh_backward_kernel(float* input, float* grad_output, int size, float* grad_input) {
    int idx         = blockIdx.x * blockDim.x + threadIdx.x;
    idx             = min(idx, size - 1);  // Prevent out-of-bounds access, no if statement
    float tanh_val  = tanhf(input[idx]);
    grad_input[idx] = grad_output[idx] * (1.0f - tanh_val * tanh_val);  // Grad for Tanh
}

// Apply activation function (Forward Pass)
void
apply_activation(float* input, int size, ActivationType activation_type, float alpha) {
    int gridSize = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;
    switch (activation_type) {
        case ActivationType::RELU:
            relu_kernel<<<gridSize, BLOCK_SIZE>>>(input, size);
            break;
        case ActivationType::LEAKY_RELU:
            leaky_relu_kernel<<<gridSize, BLOCK_SIZE>>>(input, size, alpha);
            break;
        case ActivationType::SIGMOID:
            sigmoid_kernel<<<gridSize, BLOCK_SIZE>>>(input, size);
            break;
        case ActivationType::TANH:
            tanh_kernel<<<gridSize, BLOCK_SIZE>>>(input, size);
            break;
    }
    cudaDeviceSynchronize();
}

// Apply backward pass (gradients computation)
void
apply_activation_backward(float* input,
                          float* grad_output,
                          int size,
                          ActivationType activation_type,
                          float alpha,
                          float* grad_input) {
    int gridSize = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;
    switch (activation_type) {
        case ActivationType::RELU:
            relu_backward_kernel<<<gridSize, BLOCK_SIZE>>>(input, grad_output, size, grad_input);
            break;
        case ActivationType::LEAKY_RELU:
            leaky_relu_backward_kernel<<<gridSize, BLOCK_SIZE>>>(input, grad_output, size, alpha, grad_input);
            break;
        case ActivationType::SIGMOID:
            sigmoid_backward_kernel<<<gridSize, BLOCK_SIZE>>>(input, grad_output, size, grad_input);
            break;
        case ActivationType::TANH:
            tanh_backward_kernel<<<gridSize, BLOCK_SIZE>>>(input, grad_output, size, grad_input);
            break;
    }
    cudaDeviceSynchronize();
}

#endif  // ACTIVATION_LAYER_H
