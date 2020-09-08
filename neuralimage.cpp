//
// Created by vok on 06.08.2020.
//
/// TODO: Вывод ошибок CUDA в общий лог
#include "neuralimage.hpp"
#include <memory.h>
#define NI_TAG "NI: "

void log(const std::string& str, int level);

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
        log(NI_TAG + std::string("Ошибка выделения памяти на GPU под данные изображения. Размер данных: ") + std::to_string(dataSize) + ".", 2);
        return;
    }
    if(cudaMalloc(&neuralData, neuralSize) != cudaSuccess) {
        log(NI_TAG"Ошибка выделения памяти на GPU под нейроданные. Размер данных: " + std::to_string(neuralSize) + ".", 2);
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
        log("Ошибка cudaMemcpy.", 2);
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
        log("Ошибка cudaTensorMeanRGB.", 2);
        return cudaError::cudaErrorInvalidPtx;
    }
    cudaStreamSynchronize(cudaStream);
    newImage = false;
    return cudaSuccess;
}

int NeuralImage::width() const {
    return imageWidth;
}