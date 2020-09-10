//
// Created by vok on 06.08.2020.
//
/// TODO: Вывод ошибок CUDA в общий лог
#include "neuralimage.hpp"
#include "utils.hpp"

#include <memory.h>
#define NI_TAG "NI: "

NeuralImage::NeuralImage(int camIndex, int imageWidth, int imageHeight, int numChannels) {

    initOk = false;
    newImage = false;
    imageChannels = numChannels;
    this->imageHeight = imageHeight;
    imageIndex = camIndex;
    this->imageWidth = imageWidth;
    dataSize = imageWidth * imageHeight * imageChannels;
    neuralSize = imageWidth * imageHeight * sizeof(float4);
    gpuImageData = nullptr;
    imageData = nullptr;

    if (cudaHostAlloc(&imageData, dataSize, cudaHostAllocMapped) != cudaSuccess ||
            cudaHostGetDevicePointer(&gpuImageData, imageData, 0) != cudaSuccess ) {
        if (utils::log) utils::log(NI_TAG + std::string("GPU memory allocation error for image data. Data size: ") + std::to_string(dataSize) + ".", 2);
        return;
    }
    if(cudaMalloc(&neuralData, neuralSize) != cudaSuccess) {
        if (utils::log) log(NI_TAG"GPU memory allocation error for neural data. Data size: " + std::to_string(neuralSize) + ".", 2);
        return;
    }
    outImageData = (uint8_t*)malloc(dataSize);
    font = cudaFont::Create(adaptFontSize(imageWidth));
    initOk = true;

}

uint8_t* NeuralImage::data() {

    if(!initOk) {
        return nullptr;
    }
    return imageData;

}

int NeuralImage::channels() const {
    return imageChannels;
}

uint8_t *NeuralImage::gpuData() const {
    return gpuImageData;
}

int NeuralImage::height() const {
    return imageHeight;
}

int NeuralImage::index() const {
    return imageIndex;
}

void NeuralImage::newData(uint8_t *data) {

    if(newImage) return;
    if(cudaMemcpy(gpuImageData, data, dataSize, cudaMemcpyHostToHost) != cudaSuccess) {
        if (utils::log) utils::log("Error in cudaMemcpy function.", 2);
    }
    //if(cudaRGB8toRGBA32((uchar3*)gpuImageData, (float3*)neuralData, imageWidth, imageHeight) != cudaSuccess) {
    //    log("Ошибка cudaRGB8toRGBA32.", 2);
    //}
    newImage = true;

}

cudaError_t NeuralImage::prepareTensor(cudaStream_t cudaStream, float *inputBinding, size_t bindingWidth,
                                size_t bindingHeight, float mean) {
    if(cudaTensorMeanRGB((void*)gpuImageData, imageWidth, imageHeight, inputBinding, bindingWidth, bindingHeight,
                      mean, cudaStream) != cudaSuccess) {
        if (utils::log) utils::log("Error cudaTensorMeanRGB function.", 2);
        return cudaError::cudaErrorInvalidPtx;
    }
    cudaStreamSynchronize(cudaStream);
    newImage = false;
    return cudaSuccess;
}

int NeuralImage::width() const {
    return imageWidth;
}