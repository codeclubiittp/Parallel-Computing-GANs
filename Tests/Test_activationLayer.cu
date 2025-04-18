#include "../source/activationLayer.cuh"
#define SIZE 5  // Size of the input array

// Helper function to print arrays
void
print_array(float* arr, int size) {
    for (int i = 0; i < size; i++) {
        std::cout << arr[i] << " ";
    }
    std::cout << std::endl;
}

// Test the activation layer
int
main() {
    float original_input[SIZE] = {-1.0f, 0.0f, 1.0f, -2.0f, 2.0f};
    float input[SIZE];

    float* d_input;
    cudaMalloc(&d_input, SIZE * sizeof(float));

    // ---------- ReLU ----------
    std::copy(original_input, original_input + SIZE, input);
    cudaMemcpy(d_input, input, SIZE * sizeof(float), cudaMemcpyHostToDevice);

    std::cout << "Testing ReLU activation:" << std::endl;
    apply_activation(d_input, SIZE, ActivationType::RELU, 0.0f);
    cudaMemcpy(input, d_input, SIZE * sizeof(float), cudaMemcpyDeviceToHost);
    print_array(input, SIZE);  // Expected: 0.0 0.0 1.0 0.0 2.0

    // ---------- Leaky ReLU ----------
    std::copy(original_input, original_input + SIZE, input);
    cudaMemcpy(d_input, input, SIZE * sizeof(float), cudaMemcpyHostToDevice);

    std::cout << "\nTesting Leaky ReLU activation with alpha=0.1:" << std::endl;
    apply_activation(d_input, SIZE, ActivationType::LEAKY_RELU, 0.1f);
    cudaMemcpy(input, d_input, SIZE * sizeof(float), cudaMemcpyDeviceToHost);
    print_array(input, SIZE);  // Expected: -0.1 0.0 1.0 -0.2 2.0

    // ---------- Sigmoid ----------
    std::copy(original_input, original_input + SIZE, input);
    cudaMemcpy(d_input, input, SIZE * sizeof(float), cudaMemcpyHostToDevice);

    std::cout << "\nTesting Sigmoid activation:" << std::endl;
    apply_activation(d_input, SIZE, ActivationType::SIGMOID, 0.0f);
    cudaMemcpy(input, d_input, SIZE * sizeof(float), cudaMemcpyDeviceToHost);
    print_array(input, SIZE);  // Expected: 0.268941 0.5 0.731059 0.119203 0.880797

    // ---------- Tanh ----------
    std::copy(original_input, original_input + SIZE, input);
    cudaMemcpy(d_input, input, SIZE * sizeof(float), cudaMemcpyHostToDevice);

    std::cout << "\nTesting Tanh activation:" << std::endl;
    apply_activation(d_input, SIZE, ActivationType::TANH, 0.0f);
    cudaMemcpy(input, d_input, SIZE * sizeof(float), cudaMemcpyDeviceToHost);
    print_array(input, SIZE);  // Expected: -0.761594 0.0 0.761594 -0.964027 0.964027

    cudaFree(d_input);
    return 0;
}
