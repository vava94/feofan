#ifndef UTILS_HPP
#define UTILS_HPP
#pragma once

#define TAG "UTILS: "

#include <functional>
#include <NvInfer.h>
#include <string>

using namespace std;
using namespace nvinfer1;



namespace utils {
    static function<void(const string&, int)> log = nullptr;

    inline bool cudaAllocMapped(void** cpuPtr, void** gpuPtr, size_t size) {
        if (!cpuPtr || !gpuPtr || size == 0) {
            if (log) log(TAG "Error in cudaAllocMapped function. Null pointer.", 2);
            return false;
        }

        if (cudaHostAlloc(cpuPtr, size, cudaHostAllocMapped) != cudaSuccess ||
            cudaHostGetDevicePointer(gpuPtr, *cpuPtr, 0) != cudaSuccess) {
            if (log) log(TAG "Error in cudaAllocMapped function. Failed to allocate " + std::to_string(size) + " bytes.", 2);
            return false;
        }
        memset(*cpuPtr, 0, size);
        if (log) log(TAG "Allocated " + std::to_string(size) + " bytes CPU&#12296;-&#12297;GPU.", 0);
        return true;
    }
}


static class CudaLogger : public ILogger
{
    void log(Severity severity, const char* msg) override
    {
        switch (severity) {
        case nvinfer1::ILogger::Severity::kINFO:
            if (utils::log) utils::log(string("CUDA: ") + msg, 0);
            break;
        case nvinfer1::ILogger::Severity::kWARNING:
            if (utils::log) utils::log(string("CUDA: ") + msg, 1);
            break;
        case nvinfer1::ILogger::Severity::kERROR:
            if (utils::log) utils::log(string("CUDA: ") + msg, 2);
            break;
        case nvinfer1::ILogger::Severity::kINTERNAL_ERROR:
            if (utils::log) utils::log(string("CUDA: ") + msg, 2);
            break;
        case Severity::kVERBOSE:
            if (utils::log) utils::log(string("CUDA: ") + msg, 0);
            break;
        }
    }
} cudaLogger;


#endif // !UTILS_HPP