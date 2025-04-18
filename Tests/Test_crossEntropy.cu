#include "../source/crossEntropy.cuh"

int
main() {
    const int size       = 1024;
    float* h_predictions = new float[size];
    float* h_targets     = new float[size];
    float* h_gradients   = new float[size];

    // Initialize with some binary classification data
    for (int i = 0; i < size; i++) {
        // Simulated probabilities (would come from sigmoid in real applications)
        h_predictions[i] = 0.5f + 0.4f * sinf(static_cast<float>(i) * 0.1f);
        // Binary targets (0 or 1)
        h_targets[i] = (i % 2 == 0) ? 1.0f : 0.0f;
    }

    // Compute BCE loss and gradients
    float bce = computeBCELoss(h_predictions, h_targets, size, h_gradients, true);

    printf("Binary Cross Entropy Loss: %f\n", bce);
    printf("First few gradients: %f, %f, %f, %f\n", h_gradients[0], h_gradients[1], h_gradients[2], h_gradients[3]);

    // Clean up
    delete[] h_predictions;
    delete[] h_targets;
    delete[] h_gradients;

    return 0;
}