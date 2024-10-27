#include "../affineTransform.h"
#include <cassert>

/*
// Helper function to check results
template <typename T>
void checkResults(T* computed, T* expected, size_t size) {
    for (size_t i = 0; i < size; ++i) {
        assert(computed[i] == expected[i]);  // Use assert for simplicity; consider logging for more complex scenarios
    }
}

int main() {
    size_t batchSize = 2;
    size_t inputDim = 3;
    size_t outputDim = 2;

    // Allocate and initialize host data
    std::vector<double> hx(batchSize * inputDim), hW(inputDim * outputDim), hB(outputDim), hy(batchSize * outputDim, 0.0);

    hx = { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0 };
    hW = { 0.5, 1.0, 0.5, 1.0, 0.5, 1.0 };
    hB = { 1.0, 1.0 };

    // Allocate device memory
    double* d_x, * d_W;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_x), batchSize * inputDim * sizeof(double)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_W), inputDim * outputDim * sizeof(double)));

    // Copy data to device
    CUDA_CHECK(cudaMemcpy(d_x, hx.data(), batchSize * inputDim * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_W, hW.data(), inputDim * outputDim * sizeof(double), cudaMemcpyHostToDevice));

    // Create cuBLAS handle
    cublasHandle_t handle;
    cublasCreate(&handle);

    // Call the affineTransform function
    fox::affineTransform(handle, d_x, d_W, hB, &hy, batchSize, inputDim, outputDim);

    // Clean up
    cudaFree(d_x);
    cudaFree(d_W);
    cublasDestroy(handle);

    return 0;
}
*/