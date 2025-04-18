#ifndef BATCH_NORMALIZATION_H
#define BATCH_NORMALIZATION_H

#include "utils.h"

void
apply_batch_normalization_forward(float* input,
                                  int size,
                                  float* mean,
                                  float* variance,
                                  float* gamma,
                                  float* beta,
                                  float* output,
                                  float epsilon);
void
apply_batch_normalization_backward(float* input,
                                   float* grad_output,
                                   int size,
                                   float* mean,
                                   float* variance,
                                   float* gamma,
                                   float* grad_input,
                                   float* grad_gamma,
                                   float* grad_beta,
                                   float epsilon);

// Kernel for forward pass of batch normalization
__global__ void
batch_norm_forward_kernel(float* input,
                          int size,
                          float* mean,
                          float* variance,
                          float* gamma,
                          float* beta,
                          float* output,
                          float epsilon) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    idx     = min(idx, size - 1);  // Ensure idx stays within bounds without an if statement

    // Batch normalization formula: y = gamma * ((x - mean) / sqrt(variance + epsilon)) + beta
    float norm_input = (input[idx] - mean[idx]) / sqrtf(variance[idx] + epsilon);
    output[idx]      = gamma[idx] * norm_input + beta[idx];
}

// Kernel for backward pass of batch normalization
__global__ void
batch_norm_backward_kernel(float* input,
                           float* grad_output,
                           int size,
                           float* mean,
                           float* variance,
                           float* gamma,
                           float* grad_input,
                           float* grad_gamma,
                           float* grad_beta,
                           float epsilon) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    idx     = min(idx, size - 1);  // Ensure idx stays within bounds without an if statement

    float norm_input = (input[idx] - mean[idx]) / sqrtf(variance[idx] + epsilon);
    grad_input[idx]  = grad_output[idx] * gamma[idx] / sqrtf(variance[idx] + epsilon);
    grad_gamma[idx]  = grad_output[idx] * norm_input;
    grad_beta[idx]   = grad_output[idx];
}

void
apply_batch_normalization_forward(float* input,
                                  int size,
                                  float* mean,
                                  float* variance,
                                  float* gamma,
                                  float* beta,
                                  float* output,
                                  float epsilon) {
    int gridSize = (size + 255) / 256;
    batch_norm_forward_kernel<<<gridSize, 256>>>(input, size, mean, variance, gamma, beta, output, epsilon);
    cudaDeviceSynchronize();
}

void
apply_batch_normalization_backward(float* input,
                                   float* grad_output,
                                   int size,
                                   float* mean,
                                   float* variance,
                                   float* gamma,
                                   float* grad_input,
                                   float* grad_gamma,
                                   float* grad_beta,
                                   float epsilon) {
    int gridSize = (size + 255) / 256;
    batch_norm_backward_kernel<<<gridSize, 256>>>(
        input, grad_output, size, mean, variance, gamma, grad_input, grad_gamma, grad_beta, epsilon);
    cudaDeviceSynchronize();
}

#endif  // BATCH_NORMALIZATION_H