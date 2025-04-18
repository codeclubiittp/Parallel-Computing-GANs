#include "utils.h"

// Constants to prevent numerical instability
#define EPSILON 1e-7f
#define CLIP_MIN EPSILON
#define CLIP_MAX 1.0f - EPSILON

// CUDA kernel for computing binary cross entropy element-wise
__global__ void
binaryCrossEntropyKernel(const float* predictions, const float* targets, float* losses, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        // Clip predictions to avoid log(0) or log(1)
        float pred   = fmaxf(fminf(predictions[idx], CLIP_MAX), CLIP_MIN);
        float target = targets[idx];

        // Binary cross entropy formula: -[target * log(pred) + (1 - target) * log(1 - pred)]
        losses[idx] = -target * logf(pred) - (1.0f - target) * logf(1.0f - pred);
    }
}

// CUDA kernel for computing the sum (reduction)
__global__ void
sumReductionKernel(float* input, float* output, int size) {
    extern __shared__ float sharedData[];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Load data to shared memory
    if (idx < size) {
        sharedData[tid] = input[idx];
    } else {
        sharedData[tid] = 0.0f;
    }
    __syncthreads();

    // Perform reduction in shared memory
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride && tid + stride < blockDim.x) {
            sharedData[tid] += sharedData[tid + stride];
        }
        __syncthreads();
    }

    // Write result for this block to output
    if (tid == 0) {
        output[blockIdx.x] = sharedData[0];
    }
}

// CUDA kernel for computing gradients w.r.t. predictions for backpropagation
__global__ void
binaryCrossEntropyGradKernel(const float* predictions, const float* targets, float* gradients, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        // Clip predictions to avoid division by zero
        float pred   = fmaxf(fminf(predictions[idx], CLIP_MAX), CLIP_MIN);
        float target = targets[idx];

        // Gradient: -target/pred + (1-target)/(1-pred)
        gradients[idx] = -target / pred + (1.0f - target) / (1.0f - pred);
    }
}

// Host function to compute BCE loss
float
computeBCELoss(const float* h_predictions,
               const float* h_targets,
               int size,
               float* h_gradients     = nullptr,
               bool compute_gradients = false) {
    // Device memory pointers
    float *d_predictions, *d_targets, *d_losses, *d_blockSums, *d_totalSum;
    float* d_gradients = nullptr;

    // Allocate device memory
    cudaMalloc((void**)&d_predictions, size * sizeof(float));
    cudaMalloc((void**)&d_targets, size * sizeof(float));
    cudaMalloc((void**)&d_losses, size * sizeof(float));

    if (compute_gradients) {
        cudaMalloc((void**)&d_gradients, size * sizeof(float));
    }

    // Copy host memory to device
    cudaMemcpy(d_predictions, h_predictions, size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_targets, h_targets, size * sizeof(float), cudaMemcpyHostToDevice);

    // Set up the grid and block dimensions
    int threadsPerBlock = 256;
    int blocksPerGrid   = (size + threadsPerBlock - 1) / threadsPerBlock;

    // Launch kernel to compute BCE loss element-wise
    binaryCrossEntropyKernel<<<blocksPerGrid, threadsPerBlock>>>(d_predictions, d_targets, d_losses, size);

    // Compute gradients if requested
    if (compute_gradients) {
        binaryCrossEntropyGradKernel<<<blocksPerGrid, threadsPerBlock>>>(d_predictions, d_targets, d_gradients, size);

        // Copy gradients back to host
        cudaMemcpy(h_gradients, d_gradients, size * sizeof(float), cudaMemcpyDeviceToHost);
    }

    // Prepare for reduction to compute sum
    cudaMalloc((void**)&d_blockSums, blocksPerGrid * sizeof(float));

    // Launch reduction kernel
    sumReductionKernel<<<blocksPerGrid, threadsPerBlock, threadsPerBlock * sizeof(float)>>>(
        d_losses, d_blockSums, size);

    // Final reduction to get total sum
    float totalLoss;
    if (blocksPerGrid <= threadsPerBlock) {
        cudaMalloc((void**)&d_totalSum, sizeof(float));
        sumReductionKernel<<<1, threadsPerBlock, threadsPerBlock * sizeof(float)>>>(
            d_blockSums, d_totalSum, blocksPerGrid);

        cudaMemcpy(&totalLoss, d_totalSum, sizeof(float), cudaMemcpyDeviceToHost);
        cudaFree(d_totalSum);
    } else {
        // For very large arrays, multi-stage reduction
        float* h_blockSums = new float[blocksPerGrid];
        cudaMemcpy(h_blockSums, d_blockSums, blocksPerGrid * sizeof(float), cudaMemcpyDeviceToHost);

        totalLoss = 0.0f;
        for (int i = 0; i < blocksPerGrid; i++) {
            totalLoss += h_blockSums[i];
        }

        delete[] h_blockSums;
    }

    // Compute mean
    float meanLoss = totalLoss / size;

    // Free device memory
    cudaFree(d_predictions);
    cudaFree(d_targets);
    cudaFree(d_losses);
    cudaFree(d_blockSums);
    if (compute_gradients) {
        cudaFree(d_gradients);
    }

    return meanLoss;
}
