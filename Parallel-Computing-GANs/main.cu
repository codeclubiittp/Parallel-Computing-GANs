
#include "Generator.h"
#include "Discriminator.h"

int main(int argc, char* argv[]) {

	cudnnHandle_t libcuDNNHandle;

	CUDNN_CHECK(cudnnCreate(&libcuDNNHandle));

	std::cout << "Using CUDA Runtime Version: " << cudnnGetCudartVersion() << std::endl;

	int cudnnMajorVersion = 0, cudnnMinorVersion = 0, cudnnPatchVersion = 0;

	CUDNN_CHECK(cudnnGetProperty(libraryPropertyType::MAJOR_VERSION, &cudnnMajorVersion));
	CUDNN_CHECK(cudnnGetProperty(libraryPropertyType::MINOR_VERSION, &cudnnMinorVersion));
	CUDNN_CHECK(cudnnGetProperty(libraryPropertyType::PATCH_LEVEL, &cudnnPatchVersion));

	std::cout << "Using cuDNN Library Version: " << cudnnMajorVersion << "." << cudnnMinorVersion << "." << cudnnPatchVersion
		<< std::endl;





	CUDNN_CHECK(cudnnDestroy(libcuDNNHandle));
	return 0;
}






