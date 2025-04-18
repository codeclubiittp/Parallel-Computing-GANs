#include "../source/meanSquareError.cuh"

int
main() {
    const int size       = 1024;
    float* h_predictions = new float[size];
    float* h_targets     = new float[size];

    for (int i = 0; i < size; i++) {
        h_predictions[i] = sinf(static_cast<float>(i) * 0.1f);
        h_targets[i]     = sinf(static_cast<float>(i) * 0.1f + 0.2f);
    }

    float mse = computeMSELoss(h_predictions, h_targets, size);

    printf("MSE Loss: %f\n", mse);

    delete[] h_predictions;
    delete[] h_targets;

    return 0;
}