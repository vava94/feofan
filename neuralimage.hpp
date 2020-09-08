//
// Created by vok on 06.08.2020.
//
#pragma once

#include "cudaFont.h"

#include <cstdint>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#ifdef __linux__
#include <glob.h>
#endif

typedef unsigned char uchar;



class NeuralImage {

    bool initOk = false, newImage;
    cudaFont *font = nullptr;
    float *neuralData;
    uint imageChannels, imageHeight, imageIndex, imageWidth;
    size_t dataSize, neuralSize;
    uint8_t *imageData, *gpuImageData, *outImageData;

/// Функции CUDA
    cudaError_t cudaRGB8toRGBA32( uchar3* inImage, float3* outImage, uint width, uint height );
    cudaError_t cudaRGBA32toRGB8( float3* inImage, uchar3* outImage, uint width, uint height );
    cudaError_t cudaTensorMeanRGB( void* input, size_t inputWidth, size_t inputHeight, float* output, size_t outputWidth, size_t outputHeight, const float& mean_value, cudaStream_t stream );

public:

    struct detectionData {
        float x1,y1,x2,y2, confidence;
        int type;
    };

    std::vector<detectionData> detections;

    NeuralImage(int camIndex, int imageWidth, int imageHeight, int numChannels);

    uint8_t* data();
    [[nodiscard]] uint8_t *gpuData() const;
    [[]]
    [[nodiscard]] int channels() const;
    [[nodiscard]] int height() const;
    [[nodiscard]] int index() const;
    void newData(uint8_t *data);
    cudaError_t prepareTensor(cudaStream_t cudaStream, float *inputBinding, size_t bindingWidth, size_t bindingHeight, float mean);
    cudaError_t cudaDrawBox(int boxX1, int boxX2, int boxY1, int boxY2, dim3 color, cudaStream_t, uint lineWidth, const std::string& label);

    [[nodiscard]] int width() const;
};
