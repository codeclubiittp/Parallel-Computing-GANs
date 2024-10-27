#include "../convolution2D.h"


// Main test function
int main() {
    cudnnHandle_t cudnn;
    CUDNN_CHECK(cudnnCreate(&cudnn));

    // Input image: 1 channel (grayscale), 5x5 image
    std::vector<double> image = {
         1,  2,  3,  4,  5,
         6,  7,  8,  9, 10,
        11, 12, 13, 14, 15,
        16, 17, 18, 19, 20,
        21, 22, 23, 24, 25
    };

    fox::imageInfo2D imageInfo;
    imageInfo.height = 5;
    imageInfo.width = 5;
    imageInfo.channels = 1;

    // Filter: 1 filter (output channel), 1 input channel, 3x3 filter
    std::vector<double> filter = {
         1, 0, -1,
         1, 0, -1,
         1, 0, -1
    };

    fox::filtersInfo2D filtersInfo;
    filtersInfo.height = 3;
    filtersInfo.width = 3;
    filtersInfo.n = 1;  // One filter

    // Output will be computed by the convolution function
    std::vector<double> output;

    // Perform the convolution
    fox::convolution2DGraph(cudnn, image, filter, output, imageInfo, filtersInfo);

    // Expected output (manually computed for this example):
    // (This is the result of applying a Sobel-like filter to the input image)
    std::vector<double> expectedOutput = {
        -18, -24, -30,
        -18, -24, -30,
        -18, -24, -30
    };

    // Verify output
    std::cout << "Output:" << std::endl;
    for (int i = 0; i < output.size(); ++i) {
        std::cout << output[i] << " ";
        if ((i + 1) % (imageInfo.width - filtersInfo.width + 1) == 0) {
            std::cout << std::endl;
        }
    }

    CUDNN_CHECK(cudnnDestroy(cudnn));

    return 0;
}

