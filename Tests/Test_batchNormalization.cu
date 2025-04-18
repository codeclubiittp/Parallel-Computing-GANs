#include "../source/batchNormalization.cuh"
#define SIZE 5  // Size of the input array

// Helper function to print arrays
void
print_array(float* arr, int size) {
    for (int i = 0; i < size; i++) {
        std::cout << arr[i] << " ";
    }
    std::cout << std::endl;
}

// Test the batch normalization layer
int
main() {
    float input[SIZE]    = {-1.0f, 0.0f, 1.0f, -2.0f, 2.0f};  // Example input
    float mean[SIZE]     = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f};    // Example mean
    float variance[SIZE] = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f};    // Example variance
    float gamma[SIZE]    = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f};    // Example gamma (scale)
    float beta[SIZE]     = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f};    // Example beta (shift)

    // Forward pass
    float *d_input, *d_mean, *d_variance, *d_gamma, *d_beta, *d_output;
    cudaMalloc(&d_input, SIZE * sizeof(float));
    cudaMalloc(&d_mean, SIZE * sizeof(float));
    cudaMalloc(&d_variance, SIZE * sizeof(float));
    cudaMalloc(&d_gamma, SIZE * sizeof(float));
    cudaMalloc(&d_beta, SIZE * sizeof(float));
    cudaMalloc(&d_output, SIZE * sizeof(float));

    cudaMemcpy(d_input, input, SIZE * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_mean, mean, SIZE * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_variance, variance, SIZE * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_gamma, gamma, SIZE * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_beta, beta, SIZE * sizeof(float), cudaMemcpyHostToDevice);

    float epsilon = 1e-5f;

    std::cout << "Testing Batch Normalization Forward Pass:" << std::endl;
    apply_batch_normalization_forward(d_input, SIZE, d_mean, d_variance, d_gamma, d_beta, d_output, epsilon);

    cudaMemcpy(input, d_output, SIZE * sizeof(float), cudaMemcpyDeviceToHost);
    print_array(input, SIZE);  // Expected output: normalized values based on the given mean, variance, gamma, and beta

    // Backward pass
    float grad_output[SIZE] = {0.1f, 0.2f, 0.3f, 0.4f, 0.5f};  // Example gradients
    float grad_input[SIZE], grad_gamma[SIZE], grad_beta[SIZE];
    float *d_grad_output, *d_grad_input, *d_grad_gamma, *d_grad_beta;
    cudaMalloc(&d_grad_output, SIZE * sizeof(float));
    cudaMalloc(&d_grad_input, SIZE * sizeof(float));
    cudaMalloc(&d_grad_gamma, SIZE * sizeof(float));
    cudaMalloc(&d_grad_beta, SIZE * sizeof(float));

    cudaMemcpy(d_grad_output, grad_output, SIZE * sizeof(float), cudaMemcpyHostToDevice);

    std::cout << "\nTesting Batch Normalization Backward Pass:" << std::endl;
    apply_batch_normalization_backward(
        d_input, d_grad_output, SIZE, d_mean, d_variance, d_gamma, d_grad_input, d_grad_gamma, d_grad_beta, epsilon);

    cudaMemcpy(grad_input, d_grad_input, SIZE * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(grad_gamma, d_grad_gamma, SIZE * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(grad_beta, d_grad_beta, SIZE * sizeof(float), cudaMemcpyDeviceToHost);

    std::cout << "Gradient wrt Input:" << std::endl;
    print_array(grad_input, SIZE);  // Expected output: gradients with respect to input

    std::cout << "Gradient wrt Gamma:" << std::endl;
    print_array(grad_gamma, SIZE);  // Expected output: gradients with respect to gamma

    std::cout << "Gradient wrt Beta:" << std::endl;
    print_array(grad_beta, SIZE);  // Expected output: gradients with respect to beta

    cudaFree(d_input);
    cudaFree(d_mean);
    cudaFree(d_variance);
    cudaFree(d_gamma);
    cudaFree(d_beta);
    cudaFree(d_output);
    cudaFree(d_grad_output);
    cudaFree(d_grad_input);
    cudaFree(d_grad_gamma);
    cudaFree(d_grad_beta);

    return 0;
}