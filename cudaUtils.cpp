//
// Created by vok on 19.08.2020.
//
#include "cudaUtils.h"

#include <cuda_runtime.h>
#include <memory.h>

void log(const std::string&, int level);
#define TAG std::string("cudaUtils: ")


bool cudaAllocMapped(void **cpuPtr, void **gpuPtr, size_t size) {
    if (!cpuPtr || !gpuPtr || size == 0) {
        log(TAG + "Ошибка cudaAllocMapped. Один из передаваемых параметров равен 0.", 2);
        return false;
    }

    if (cudaHostAlloc(cpuPtr, size, cudaHostAllocMapped) != cudaSuccess ||
        cudaHostGetDevicePointer(gpuPtr, *cpuPtr, 0) != cudaSuccess ) {
        log(TAG + "Ошибка cudaAllocMapped. Невозможно выделить " + std::to_string(size) + " байт.", 2);
        return false;
    }
    memset(cpuPtr, 0, size);
    log(TAG + "Выделено " + std::to_string(size) + " байт CPU&#12296;-&#12297;GPU.", 0);
    return true;
}