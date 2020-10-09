#ifndef UTILS_HPP
#define UTILS_HPP

#include <cuda_runtime.h>
#include <functional>
#include <NvInfer.h>
#include <string>

#define TAG "UTILS: "

namespace utils {

    /**
    * Log callback function variable.
    */
    static std::function<void(const std::string&, int)> Log;

    /**
    * Alloocate shared between CPU & GPU pointer  
    */
    bool cudaAllocMapped(void** cpuPtr, void** gpuPtr, size_t size); 
    __device__ __host__ unsigned int divUp(unsigned int a, unsigned int b);
    
}

/**
 * Logger for cuda functions.
 */
static class CudaLogger : public nvinfer1::ILogger {
    void log(Severity severity, const char* msg) override;
} cudaLogger;


#endif // !UTILS_HPP