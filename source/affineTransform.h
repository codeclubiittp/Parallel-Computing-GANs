#pragma once

#include "utils.h"

namespace fox {
    /*
    * Performs Affine Transformation y = xW^(T) + B
    */
    void affineTransform(cublasHandle_t& handle, double* x, double* W, std::vector<double>& B, std::vector<double>* y,
        size_t& batchSize, size_t& inputDim, size_t& outputDim) {

        const double alpha = static_cast<double>(1.0);
        const double beta = static_cast<double>(0.0);

        double* dy = nullptr;
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&dy), batchSize * outputDim * sizeof(double)));

        CUBLAS_CHECK(cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, outputDim, batchSize, inputDim, &alpha,
            W, outputDim, x, inputDim, &beta, dy, outputDim));

        CUDA_CHECK(cudaMemcpy((*y).data(), dy, batchSize * outputDim * sizeof(double), cudaMemcpyDeviceToHost));

        #pragma omp parallel for collapse(2)
        for (int i = 0; i < batchSize; i++) {
            for (int j = 0; j < outputDim; j++) {
                (*y)[i * outputDim + j] += B[j];
            }
        }
    }
}