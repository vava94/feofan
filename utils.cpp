#include "utils.hpp"

using namespace std;
using namespace nvinfer1;


bool utils::cudaAllocMapped(void** cpuPtr, void** gpuPtr, size_t size) {

    if (!cpuPtr || !gpuPtr || size == 0) {
        if (utils::Log) utils::Log(TAG "Error in cudaAllocMapped function. Null pointer.", 2);
        return false;
    }

    if (cudaHostAlloc(cpuPtr, size, cudaHostAllocMapped) != cudaSuccess ||
        cudaHostGetDevicePointer(gpuPtr, *cpuPtr, 0) != cudaSuccess) {
        if (utils::Log) utils::Log(TAG "Error in cudaAllocMapped function. Failed to allocate " + std::to_string(size) + " bytes.", 2);
        return false;
    }
    memset(*cpuPtr, 0, size);
    if (utils::Log) utils::Log(TAG "Allocated " + std::to_string(size) + " bytes CPU&#12296;-&#12297;GPU.", 0);
    return true;

}

void CudaLogger::log(Severity severity, const char* msg) {
    switch (severity) {
    case nvinfer1::ILogger::Severity::kINFO:
        if (utils::Log) utils::Log(string("CUDA: ") + msg, 0);
        break;
    case nvinfer1::ILogger::Severity::kWARNING:
        if (utils::Log) utils::Log(string("CUDA: ") + msg, 1);
        break;
    case nvinfer1::ILogger::Severity::kERROR:
        if (utils::Log) utils::Log(string("CUDA: ") + msg, 2);
        break;
    case nvinfer1::ILogger::Severity::kINTERNAL_ERROR:
        if (utils::Log) utils::Log(string("CUDA: ") + msg, 2);
        break;
    case Severity::kVERBOSE:
        if (utils::Log) utils::Log(string("CUDA: ") + msg, 0);
        break;
    }
}
