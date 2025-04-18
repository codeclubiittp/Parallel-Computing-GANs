#define __CUDACC__
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <math.h>

__global__ void
squaredDifferenceKernel(const float* predictions, const float* targets, float* differences, int size) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        float diff       = predictions[idx] - targets[idx];
        differences[idx] = diff * diff;
    }
}

__global__ void
sumReductionKernel(float* input, float* output, int size) {
    extern __shared__ float sharedData[];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        sharedData[tid] = input[idx];
    } else {
        sharedData[tid] = 0.0f;
    }
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride && idx + stride < size) {
            sharedData[tid] += sharedData[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        output[blockIdx.x] = sharedData[0];
    }
}

float
computeMSELoss(const float* h_predictions, const float* h_targets, int size) {
    float *d_predictions, *d_targets, *d_differences, *d_blockSums, *d_totalSum;

    cudaMalloc((void**)&d_predictions, size * sizeof(float));
    cudaMalloc((void**)&d_targets, size * sizeof(float));
    cudaMalloc((void**)&d_differences, size * sizeof(float));

    cudaMemcpy(d_predictions, h_predictions, size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_targets, h_targets, size * sizeof(float), cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocksPerGrid   = (size + threadsPerBlock - 1) / threadsPerBlock;

    squaredDifferenceKernel<<<blocksPerGrid, threadsPerBlock>>>(d_predictions, d_targets, d_differences, size);

    cudaMalloc((void**)&d_blockSums, blocksPerGrid * sizeof(float));

    sumReductionKernel<<<blocksPerGrid, threadsPerBlock, threadsPerBlock * sizeof(float)>>>(
        d_differences, d_blockSums, size);

    float finalSum;

    if (blocksPerGrid <= threadsPerBlock) {
        cudaMalloc((void**)&d_totalSum, sizeof(float));
        sumReductionKernel<<<1, threadsPerBlock, threadsPerBlock * sizeof(float)>>>(
            d_blockSums, d_totalSum, blocksPerGrid);

        cudaMemcpy(&finalSum, d_totalSum, sizeof(float), cudaMemcpyDeviceToHost);
        cudaFree(d_totalSum);
    } else {
        float* h_blockSums = new float[blocksPerGrid];
        cudaMemcpy(h_blockSums, d_blockSums, blocksPerGrid * sizeof(float), cudaMemcpyDeviceToHost);

        finalSum = 0.0f;
        for (int i = 0; i < blocksPerGrid; i++) {
            finalSum += h_blockSums[i];
        }

        delete[] h_blockSums;
    }

    float mse = finalSum / size;

    cudaFree(d_predictions);
    cudaFree(d_targets);
    cudaFree(d_differences);
    cudaFree(d_blockSums);

    return mse;
}
