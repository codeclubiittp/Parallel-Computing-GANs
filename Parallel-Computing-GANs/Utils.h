#pragma once

#include <iostream>
#include <vector>
#include <memory>
#include <unordered_map>

#include <cudnn_cnn.h>
#include <cudnn_ops.h>
#include <cudnn_graph.h>

#include <cudnn_frontend.h>

#include <cublas_v2.h>
#include <cuda.h>

#include <omp.h>

#define CUDA_CHECK(CallWrap) if(CallWrap != cudaSuccess){ \
    std::printf("[Cuda Error]: %s\n [File]: %s\n [Line]: %d\n", cudaGetErrorString(CallWrap), __FILE__, __LINE__);\
    std::abort();}

#define CUDNN_CHECK(CallWrap) if(CallWrap != CUDNN_STATUS_SUCCESS){ \
    std::printf("[Cuda Error]: %s\n [File]: %s\n [Line]: %d\n", cudnnGetErrorString(CallWrap), __FILE__, __LINE__);\
    std::abort();}

#define CUBLAS_CHECK(CallWrap) if(CallWrap != CUBLAS_STATUS_SUCCESS){ \
    std::printf("[Cuda Error]: %s\n [File]: %s\n [Line]: %d\n", cublasGetStatusString(CallWrap), __FILE__, __LINE__);\
    std::abort();}

#define CUDNNFE_CHECK(CallWrap) if(CallWrap.is_bad()) { \
    std::cout << "[cuDNN Error]: " << CallWrap.get_message() << "\n [File]: " << __FILE__ << "\n [Line]: " << __LINE__ \
        << std::endl;  \
    std::abort();}

namespace fox {

    struct imageInfo2D {
        int64_t height, width, channels;

        explicit imageInfo2D() : height(0), width(0), channels(3) {};
    };

    struct filtersInfo2D {
        int64_t height, width, n;

        explicit filtersInfo2D() : height(0), width(0), n(1) {};
    };

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