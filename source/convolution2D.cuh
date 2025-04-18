
#include "utils.h"

namespace fox {

void
convolution2D(cudnnHandle_t& handle,
              std::vector<double>& image,
              std::vector<double>& filters,
              std::vector<double>& output,
              imageInfo2D& imageInfo,
              filtersInfo2D& filtersInfo) {
    const double alpha = static_cast<double>(1.0);
    const double beta  = static_cast<double>(0.0);

    cudnnTensorDescriptor_t inputDescriptor, outputDescriptor;
    cudnnFilterDescriptor_t filterDescriptor;
    cudnnConvolutionDescriptor_t convolutionDescriptor;

    CUDNN_CHECK(cudnnCreateTensorDescriptor(&inputDescriptor));
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&outputDescriptor));
    CUDNN_CHECK(cudnnCreateFilterDescriptor(&filterDescriptor));
    CUDNN_CHECK(cudnnCreateConvolutionDescriptor(&convolutionDescriptor));

    CUDNN_CHECK(cudnnSetTensor4dDescriptor(inputDescriptor,
                                           CUDNN_TENSOR_NCHW,
                                           CUDNN_DATA_DOUBLE,
                                           1,
                                           imageInfo.channels,
                                           imageInfo.height,
                                           imageInfo.width));
    CUDNN_CHECK(cudnnSetFilter4dDescriptor(filterDescriptor,
                                           CUDNN_DATA_DOUBLE,
                                           CUDNN_TENSOR_NCHW,
                                           filtersInfo.n,
                                           imageInfo.channels,
                                           filtersInfo.height,
                                           filtersInfo.width));
    CUDNN_CHECK(
        cudnnSetConvolution2dDescriptor(convolutionDescriptor, 0, 0, 1, 1, 1, 1, CUDNN_CONVOLUTION, CUDNN_DATA_DOUBLE));

    int outputHeight = (imageInfo.height - filtersInfo.height) + 1;
    int outputWidth  = (imageInfo.width - filtersInfo.width) + 1;

    output = std::vector<double>(outputHeight * outputWidth * filtersInfo.n);

    CUDNN_CHECK(cudnnSetTensor4dDescriptor(
        outputDescriptor, CUDNN_TENSOR_NCHW, CUDNN_DATA_DOUBLE, 1, filtersInfo.n, outputHeight, outputWidth));

    double *dimage = nullptr, *dfilters = nullptr, *doutputImage = nullptr;

    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&dimage), image.size() * sizeof(double)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&dfilters), filters.size() * sizeof(double)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&doutputImage), output.size() * sizeof(double)));

    CUDA_CHECK(cudaMemcpy(dimage, image.data(), image.size() * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dfilters, filters.data(), filters.size() * sizeof(double), cudaMemcpyHostToDevice));

    CUDNN_CHECK(cudnnConvolutionForward(handle,
                                        &alpha,
                                        inputDescriptor,
                                        dimage,
                                        filterDescriptor,
                                        dfilters,
                                        convolutionDescriptor,
                                        CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM,
                                        nullptr,
                                        0,
                                        &beta,
                                        outputDescriptor,
                                        doutputImage));

    CUDA_CHECK(cudaMemcpy(output.data(), doutputImage, output.size() * sizeof(double), cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(dimage));
    CUDA_CHECK(cudaFree(dfilters));
    CUDA_CHECK(cudaFree(doutputImage));
    cudnnDestroyTensorDescriptor(inputDescriptor);
    cudnnDestroyFilterDescriptor(filterDescriptor);
    cudnnDestroyTensorDescriptor(outputDescriptor);
    cudnnDestroyConvolutionDescriptor(convolutionDescriptor);
}

/*
 * Basically there's no Reference for this
 *
 */
void
convolution2DGraphFE(cudnnHandle_t& handle,
                     std::vector<double>& image,
                     std::vector<double>& filters,
                     std::vector<double>& output,
                     imageInfo2D& imageInfo,
                     filtersInfo2D& filtersInfo) {
    cudnn_frontend::isLoggingEnabled() = true;

    std::shared_ptr<cudnn_frontend::graph::Graph> computeGraph = std::make_shared<cudnn_frontend::graph::Graph>();

    computeGraph->set_compute_data_type(cudnn_frontend::DataType_t::DOUBLE);
    computeGraph->set_io_data_type(cudnn_frontend::DataType_t::DOUBLE);

    std::shared_ptr<cudnn_frontend::graph::Tensor_attributes> inputTensorAttributes =
        std::make_shared<cudnn_frontend::graph::Tensor_attributes>();
    inputTensorAttributes->set_name("input");
    inputTensorAttributes->set_dim(
        {1LL, (int64_t)imageInfo.channels, (int64_t)imageInfo.height, (int64_t)imageInfo.width});
    inputTensorAttributes->set_stride(
        {(int64_t)imageInfo.channels * (int64_t)imageInfo.height * (int64_t)imageInfo.width,
         1LL,
         (int64_t)imageInfo.channels * (int64_t)imageInfo.width,
         (int64_t)imageInfo.width});

    std::shared_ptr<cudnn_frontend::graph::Tensor_attributes> filterTensorAttributes =
        std::make_shared<cudnn_frontend::graph::Tensor_attributes>();
    filterTensorAttributes->set_name("filter");
    filterTensorAttributes->set_dim({1LL,
                                     (int64_t)filtersInfo.n,
                                     (int64_t)imageInfo.channels,
                                     (int64_t)filtersInfo.height,
                                     (int64_t)filtersInfo.width});

    filterTensorAttributes->set_stride(
        {(int64_t)imageInfo.channels * (int64_t)filtersInfo.height * (int64_t)filtersInfo.width,
         1LL,
         (int64_t)imageInfo.channels * (int64_t)filtersInfo.width,
         (int64_t)filtersInfo.width});

    cudnn_frontend::graph::Conv_fprop_attributes convOpAttributes;
    convOpAttributes.set_padding({0LL, 0LL});
    convOpAttributes.set_stride({1LL, 1LL});
    convOpAttributes.set_dilation({1LL, 1LL});

    std::shared_ptr<cudnn_frontend::graph::Tensor_attributes> outputAttributes =
        computeGraph->conv_fprop(inputTensorAttributes, filterTensorAttributes, convOpAttributes);

    outputAttributes->set_output(true);

    CUDNNFE_CHECK(computeGraph->validate());
    CUDNNFE_CHECK(computeGraph->build_operation_graph(handle));
    CUDNNFE_CHECK(computeGraph->create_execution_plans({cudnn_frontend::HeurMode_t::A}));
    CUDNNFE_CHECK(computeGraph->check_support(handle));
    CUDNNFE_CHECK(computeGraph->build_plans(handle));

    double *dimage = nullptr, *dfilters = nullptr, *doutputImage = nullptr;

    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&dimage), image.size() * sizeof(double)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&dfilters), filters.size() * sizeof(double)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&doutputImage), output.size() * sizeof(double)));

    CUDA_CHECK(cudaMemcpy(dimage, image.data(), image.size() * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dfilters, filters.data(), filters.size() * sizeof(double), cudaMemcpyHostToDevice));

    std::unordered_map<int64_t, void*> variantPack = {{inputTensorAttributes->get_uid(), dimage},
                                                      {filterTensorAttributes->get_uid(), dfilters},
                                                      {outputAttributes->get_uid(), doutputImage}};

    int64_t workspaceSize;

    computeGraph->get_workspace_size(workspaceSize);

    int8_t* dworkspace;

    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&dworkspace), workspaceSize));

    CUDNNFE_CHECK(computeGraph->execute(handle, variantPack, dworkspace));

    CUDA_CHECK(cudaMemcpy(output.data(), doutputImage, output.size() * sizeof(double), cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(dimage));
    CUDA_CHECK(cudaFree(dfilters));
    CUDA_CHECK(cudaFree(doutputImage));
}

void
convolution2DGraph(cudnnHandle_t& handle,
                   std::vector<double>& image,
                   std::vector<double>& filters,
                   std::vector<double>& output,
                   imageInfo2D& imageInfo,
                   filtersInfo2D& filtersInfo) {
    double *dimage = nullptr, *dfilters = nullptr, *doutputImage = nullptr;

    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&dimage), image.size() * sizeof(double)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&dfilters), filters.size() * sizeof(double)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&doutputImage), output.size() * sizeof(double)));

    CUDA_CHECK(cudaMemcpy(dimage, image.data(), image.size() * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dfilters, filters.data(), filters.size() * sizeof(double), cudaMemcpyHostToDevice));

    cudnnBackendDescriptor_t xDescriptor;
    CUDNN_CHECK(cudnnBackendCreateDescriptor(CUDNN_BACKEND_TENSOR_DESCRIPTOR, &xDescriptor));

    cudnnDataType_t xDataType = CUDNN_DATA_DOUBLE;
    CUDNN_CHECK(
        cudnnBackendSetAttribute(xDescriptor, CUDNN_ATTR_TENSOR_DATA_TYPE, CUDNN_TYPE_DATA_TYPE, 1, &xDataType));

    /*
    int64_t xDimensions[] = { 1, 1, imageInfo.channels, 1, imageInfo.height, imageInfo.width };
    int64_t xStrides[] = {
        1 * imageInfo.channels * 1 * imageInfo.height * imageInfo.width,
        imageInfo.channels * 1 * imageInfo.height * imageInfo.width,
        1 * imageInfo.height * imageInfo.width,
        imageInfo.height * imageInfo.width,
        imageInfo.width,
        1 };
    */

    int64_t xDimensions[] = {1, imageInfo.channels, imageInfo.height, imageInfo.width};
    int64_t xStrides[]    = {imageInfo.channels * imageInfo.height * imageInfo.width,
                             imageInfo.height * imageInfo.width,
                             imageInfo.width,
                             1};
    int64_t xUniqueId     = 'x';
    int64_t xAlignment    = sizeof(double);

    CUDNN_CHECK(cudnnBackendSetAttribute(xDescriptor, CUDNN_ATTR_TENSOR_DIMENSIONS, CUDNN_TYPE_INT64, 4, xDimensions));
    CUDNN_CHECK(cudnnBackendSetAttribute(xDescriptor, CUDNN_ATTR_TENSOR_STRIDES, CUDNN_TYPE_INT64, 4, xStrides));
    CUDNN_CHECK(cudnnBackendSetAttribute(xDescriptor, CUDNN_ATTR_TENSOR_UNIQUE_ID, CUDNN_TYPE_INT64, 1, &xUniqueId));
    CUDNN_CHECK(
        cudnnBackendSetAttribute(xDescriptor, CUDNN_ATTR_TENSOR_BYTE_ALIGNMENT, CUDNN_TYPE_INT64, 1, &xAlignment));
    CUDNN_CHECK(cudnnBackendFinalize(xDescriptor));

    cudnnBackendDescriptor_t filterDescriptor;
    CUDNN_CHECK(cudnnBackendCreateDescriptor(CUDNN_BACKEND_TENSOR_DESCRIPTOR, &filterDescriptor));

    cudnnDataType_t filterDataType = CUDNN_DATA_DOUBLE;
    CUDNN_CHECK(cudnnBackendSetAttribute(
        filterDescriptor, CUDNN_ATTR_TENSOR_DATA_TYPE, CUDNN_TYPE_DATA_TYPE, 1, &filterDataType));

    /*
    int64_t filterDimensions[] = {
        1,
        filtersInfo.n,
        imageInfo.channels,
        1,
        filtersInfo.height,
        filtersInfo.width
    };

    int64_t filterStrides[] = {
        filtersInfo.n * imageInfo.channels * 1 * filtersInfo.height * filtersInfo.width,
        imageInfo.channels * 1 * filtersInfo.height * filtersInfo.width,
        1 * filtersInfo.height * filtersInfo.width,
        filtersInfo.height * filtersInfo.width,
        filtersInfo.width,
        1
    };
    */

    int64_t filterDimensions[] = {filtersInfo.n, imageInfo.channels, filtersInfo.height, filtersInfo.width};

    int64_t filterStrides[] = {imageInfo.channels * filtersInfo.height * filtersInfo.width,
                               filtersInfo.height * filtersInfo.width,
                               filtersInfo.width,
                               1};

    int64_t filterUniqueId  = 'f';
    int64_t filterAlignment = sizeof(double);

    CUDNN_CHECK(cudnnBackendSetAttribute(
        filterDescriptor, CUDNN_ATTR_TENSOR_DIMENSIONS, CUDNN_TYPE_INT64, 4, filterDimensions));
    CUDNN_CHECK(
        cudnnBackendSetAttribute(filterDescriptor, CUDNN_ATTR_TENSOR_STRIDES, CUDNN_TYPE_INT64, 4, filterStrides));
    CUDNN_CHECK(
        cudnnBackendSetAttribute(filterDescriptor, CUDNN_ATTR_TENSOR_UNIQUE_ID, CUDNN_TYPE_INT64, 1, &filterUniqueId));
    CUDNN_CHECK(cudnnBackendSetAttribute(
        filterDescriptor, CUDNN_ATTR_TENSOR_BYTE_ALIGNMENT, CUDNN_TYPE_INT64, 1, &filterAlignment));
    CUDNN_CHECK(cudnnBackendFinalize(filterDescriptor));

    cudnnBackendDescriptor_t outputDescriptor;
    CUDNN_CHECK(cudnnBackendCreateDescriptor(CUDNN_BACKEND_TENSOR_DESCRIPTOR, &outputDescriptor));

    cudnnDataType_t outputDataType = CUDNN_DATA_DOUBLE;
    CUDNN_CHECK(cudnnBackendSetAttribute(
        outputDescriptor, CUDNN_ATTR_TENSOR_DATA_TYPE, CUDNN_TYPE_DATA_TYPE, 1, &outputDataType));

    int64_t outputHeight = imageInfo.height - filtersInfo.height + 1;
    int64_t outputWidth  = imageInfo.width - filtersInfo.width + 1;

    /*
    output = std::vector<double>(outputWidth * outputHeight);

    int64_t outputDimensions[] = {
        1,
        1,
        filtersInfo.n,
        outputHeight,
        outputWidth,
        1
    };

    int64_t outputStrides[] = {
        filtersInfo.n * outputHeight * outputWidth * 1,
        filtersInfo.n * outputHeight * outputWidth * 1,
        outputHeight * outputWidth * 1,
        outputWidth * 1,
        1
    };
    */

    int64_t outputDimensions[] = {1, filtersInfo.n, outputHeight, outputWidth};

    int64_t outputStrides[] = {filtersInfo.n * outputHeight * outputWidth, outputHeight * outputWidth, outputWidth, 1};

    output = std::vector<double>(1 * filtersInfo.n * outputHeight * outputWidth);

    int64_t outputUniqueId  = 'o';
    int64_t outputAlignment = sizeof(double);

    CUDNN_CHECK(cudnnBackendSetAttribute(
        outputDescriptor, CUDNN_ATTR_TENSOR_DIMENSIONS, CUDNN_TYPE_INT64, 4, outputDimensions));
    CUDNN_CHECK(
        cudnnBackendSetAttribute(outputDescriptor, CUDNN_ATTR_TENSOR_STRIDES, CUDNN_TYPE_INT64, 4, outputStrides));
    CUDNN_CHECK(
        cudnnBackendSetAttribute(outputDescriptor, CUDNN_ATTR_TENSOR_UNIQUE_ID, CUDNN_TYPE_INT64, 1, &outputUniqueId));
    CUDNN_CHECK(cudnnBackendSetAttribute(
        outputDescriptor, CUDNN_ATTR_TENSOR_BYTE_ALIGNMENT, CUDNN_TYPE_INT64, 1, &outputAlignment));
    CUDNN_CHECK(cudnnBackendFinalize(outputDescriptor));

    cudnnBackendDescriptor_t convOpDescriptor;
    CUDNN_CHECK(cudnnBackendCreateDescriptor(CUDNN_BACKEND_CONVOLUTION_DESCRIPTOR, &convOpDescriptor));

    cudnnDataType_t convDatatype    = CUDNN_DATA_DOUBLE;
    cudnnConvolutionMode_t convMode = CUDNN_CONVOLUTION;

    int64_t convDimensions = 2;
    int64_t convPadding[]  = {0, 0};
    int64_t convStride[]   = {1, 1};
    int64_t convDilation[] = {1, 1};

    CUDNN_CHECK(cudnnBackendSetAttribute(
        convOpDescriptor, CUDNN_ATTR_CONVOLUTION_COMP_TYPE, CUDNN_TYPE_DATA_TYPE, 1, &convDatatype));
    CUDNN_CHECK(cudnnBackendSetAttribute(
        convOpDescriptor, CUDNN_ATTR_CONVOLUTION_CONV_MODE, CUDNN_TYPE_CONVOLUTION_MODE, 1, &convMode));
    CUDNN_CHECK(cudnnBackendSetAttribute(
        convOpDescriptor, CUDNN_ATTR_CONVOLUTION_SPATIAL_DIMS, CUDNN_TYPE_INT64, 1, &convDimensions));
    CUDNN_CHECK(cudnnBackendSetAttribute(
        convOpDescriptor, CUDNN_ATTR_CONVOLUTION_PRE_PADDINGS, CUDNN_TYPE_INT64, 2, &convPadding));
    CUDNN_CHECK(cudnnBackendSetAttribute(
        convOpDescriptor, CUDNN_ATTR_CONVOLUTION_POST_PADDINGS, CUDNN_TYPE_INT64, 2, &convPadding));
    CUDNN_CHECK(cudnnBackendSetAttribute(
        convOpDescriptor, CUDNN_ATTR_CONVOLUTION_FILTER_STRIDES, CUDNN_TYPE_INT64, 2, &convStride));
    CUDNN_CHECK(cudnnBackendSetAttribute(
        convOpDescriptor, CUDNN_ATTR_CONVOLUTION_DILATIONS, CUDNN_TYPE_INT64, 2, &convDilation));
    CUDNN_CHECK(cudnnBackendFinalize(convOpDescriptor));

    cudnnBackendDescriptor_t convForwardDescriptor;
    CUDNN_CHECK(
        cudnnBackendCreateDescriptor(CUDNN_BACKEND_OPERATION_CONVOLUTION_FORWARD_DESCRIPTOR, &convForwardDescriptor));

    double alpha = 1.0;
    double beta  = 0.0;

    CUDNN_CHECK(cudnnBackendSetAttribute(convForwardDescriptor,
                                         CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_X,
                                         CUDNN_TYPE_BACKEND_DESCRIPTOR,
                                         1,
                                         &xDescriptor));
    CUDNN_CHECK(cudnnBackendSetAttribute(convForwardDescriptor,
                                         CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_W,
                                         CUDNN_TYPE_BACKEND_DESCRIPTOR,
                                         1,
                                         &filterDescriptor));
    CUDNN_CHECK(cudnnBackendSetAttribute(convForwardDescriptor,
                                         CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_Y,
                                         CUDNN_TYPE_BACKEND_DESCRIPTOR,
                                         1,
                                         &outputDescriptor));
    CUDNN_CHECK(cudnnBackendSetAttribute(convForwardDescriptor,
                                         CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_CONV_DESC,
                                         CUDNN_TYPE_BACKEND_DESCRIPTOR,
                                         1,
                                         &convOpDescriptor));
    CUDNN_CHECK(cudnnBackendSetAttribute(
        convForwardDescriptor, CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_ALPHA, CUDNN_TYPE_DOUBLE, 1, &alpha));
    CUDNN_CHECK(cudnnBackendSetAttribute(
        convForwardDescriptor, CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_BETA, CUDNN_TYPE_DOUBLE, 1, &beta));
    CUDNN_CHECK(cudnnBackendFinalize(convForwardDescriptor));

    cudnnBackendDescriptor_t opGraph;

    CUDNN_CHECK(cudnnBackendCreateDescriptor(CUDNN_BACKEND_OPERATIONGRAPH_DESCRIPTOR, &opGraph));
    CUDNN_CHECK(cudnnBackendSetAttribute(
        opGraph, CUDNN_ATTR_OPERATIONGRAPH_OPS, CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &convForwardDescriptor));
    CUDNN_CHECK(cudnnBackendSetAttribute(opGraph, CUDNN_ATTR_OPERATIONGRAPH_HANDLE, CUDNN_TYPE_HANDLE, 1, &handle));
    CUDNN_CHECK(cudnnBackendFinalize(opGraph));

    cudnnBackendDescriptor_t engineDescriptor;
    CUDNN_CHECK(cudnnBackendCreateDescriptor(CUDNN_BACKEND_ENGINE_DESCRIPTOR, &engineDescriptor));

    int64_t globalIndex = 0;

    CUDNN_CHECK(cudnnBackendSetAttribute(
        engineDescriptor, CUDNN_ATTR_ENGINE_OPERATION_GRAPH, CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &opGraph));
    CUDNN_CHECK(
        cudnnBackendSetAttribute(engineDescriptor, CUDNN_ATTR_ENGINE_GLOBAL_INDEX, CUDNN_TYPE_INT64, 1, &globalIndex));
    CUDNN_CHECK(cudnnBackendFinalize(engineDescriptor));

    cudnnBackendDescriptor_t engineConfigDescriptor;
    CUDNN_CHECK(cudnnBackendCreateDescriptor(CUDNN_BACKEND_ENGINECFG_DESCRIPTOR, &engineConfigDescriptor));

    CUDNN_CHECK(cudnnBackendSetAttribute(
        engineConfigDescriptor, CUDNN_ATTR_ENGINECFG_ENGINE, CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &engineDescriptor));
    CUDNN_CHECK(cudnnBackendFinalize(engineConfigDescriptor));

    int64_t engineWorkspaceSize;

    CUDNN_CHECK(cudnnBackendGetAttribute(
        engineConfigDescriptor, CUDNN_ATTR_ENGINECFG_WORKSPACE_SIZE, CUDNN_TYPE_INT64, 1, NULL, &engineWorkspaceSize));

    cudnnBackendDescriptor_t planDescriptor;
    CUDNN_CHECK(cudnnBackendCreateDescriptor(CUDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR, &planDescriptor));
    CUDNN_CHECK(
        cudnnBackendSetAttribute(planDescriptor, CUDNN_ATTR_EXECUTION_PLAN_HANDLE, CUDNN_TYPE_HANDLE, 1, &handle));
    CUDNN_CHECK(cudnnBackendSetAttribute(planDescriptor,
                                         CUDNN_ATTR_EXECUTION_PLAN_ENGINE_CONFIG,
                                         CUDNN_TYPE_BACKEND_DESCRIPTOR,
                                         1,
                                         &engineConfigDescriptor));
    CUDNN_CHECK(cudnnBackendFinalize(planDescriptor));

    int64_t planWorkspaceSize;
    CUDNN_CHECK(cudnnBackendGetAttribute(
        planDescriptor, CUDNN_ATTR_EXECUTION_PLAN_WORKSPACE_SIZE, CUDNN_TYPE_INT64, 1, NULL, &planWorkspaceSize));

    void* pDevice[3] = {dimage, dfilters, doutputImage};
    int64_t uids[3]  = {'x', 'f', 'o'};

    void* workspace;

    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&workspace), planWorkspaceSize));

    cudnnBackendDescriptor_t varpackDescriptor;
    CUDNN_CHECK(cudnnBackendCreateDescriptor(CUDNN_BACKEND_VARIANT_PACK_DESCRIPTOR, &varpackDescriptor));

    CUDNN_CHECK(cudnnBackendSetAttribute(
        varpackDescriptor, CUDNN_ATTR_VARIANT_PACK_DATA_POINTERS, CUDNN_TYPE_VOID_PTR, 3, pDevice));
    CUDNN_CHECK(
        cudnnBackendSetAttribute(varpackDescriptor, CUDNN_ATTR_VARIANT_PACK_UNIQUE_IDS, CUDNN_TYPE_INT64, 3, uids));
    CUDNN_CHECK(cudnnBackendSetAttribute(
        varpackDescriptor, CUDNN_ATTR_VARIANT_PACK_WORKSPACE, CUDNN_TYPE_VOID_PTR, 1, &workspace));
    CUDNN_CHECK(cudnnBackendFinalize(varpackDescriptor));

    CUDNN_CHECK(cudnnBackendExecute(handle, planDescriptor, varpackDescriptor));

    CUDA_CHECK(cudaMemcpy(output.data(), doutputImage, output.size() * sizeof(double), cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(dimage));
    CUDA_CHECK(cudaFree(dfilters));
    CUDA_CHECK(cudaFree(doutputImage));
}
}  // namespace fox