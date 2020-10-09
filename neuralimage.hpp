//
// Created by vok on 06.08.2020.
//
#ifndef NEURALIMAGE_HPP
#define NEURALIMAGE_HPP
#pragma once


#include <cstdint>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#ifdef __linux__
#include <glob.h>
#endif

typedef unsigned char uchar;
typedef unsigned int uint;


class NeuralImage {
private:
    bool 
        imageLocked = false,
        initOk = false, 
        newImage = false;

    //cudaFont *font = nullptr;
    
    dim3 gridDim = {0};
    float* detectionsCpu, * detectionsGpu;
    float2 mBindingScale = {0};
    int 
        imageIndex = -1,
        numDetections = -1;

    uint
        imageChannels,
        imageHeight,
        imageWidth;


    size_t 
        dataSize = 0, 
        cudaDetectionsSize = 0;

    uchar3
        * imageDataCpu = nullptr,
        * imageDataGpu = nullptr;

    struct __align__(16) GlyphCommand {
        short x;
        short y;
        short u;
        short v;
        short width;
        short height;
    };



    void addText(std::string text, float4 color, short x, short y);
    
    cudaError launchCudaApplyDetections(cudaStream_t stream, int detCount, int lineWidth, float opacity);//, uchar* fontMaps, int fontWidth, int fontHeight , GlyphCommand* commands, size_t numCommands);
public:

    enum APPLY_FLAG {
        NONE        = 0,
        BOX         = 0x001b,
        LABEL        = 0x010b,
        CONFIDENCE  = 0x100b
    };

    struct detectionData {
        float x1,y1,x2,y2, confidence;
        int type;
    };

    

    NeuralImage(int camIndex, int imageWidth, int imageHeight, int numChannels);
    ~NeuralImage();

    void applyDetections(int count, int applyFlags, int lineWidth = 1, float opacity = 0.95, std::vector<std::string> labels = {});
    void cudaDrawBox(int boxX1, int boxY1, int boxX2, int boxY2, dim3 color, uint lineWidth, cudaStream_t);
    

    uint8_t* data();
    float** detectionsDataPtr();
    float2 bindingScale() const;
    [[nodiscard]] uint8_t *gpuData() const;
    [[nodiscard]] int channels() const;
    [[nodiscard]] int height() const;
    [[nodiscard]] int index() const;
    static void loadFont(FILE* fontFIle, uint8_t fontSize);
    void lockImage();
    void newData(uint8_t *data);
    void prepareTensor(cudaStream_t cudaStream, float* inputBinding, size_t bindingWidth, size_t bindingHeight, float mean);
    void setLabels(std::vector<std::string> labelsArray);
    void unlockImage();
    [[nodiscard]] int width() const;
};

#endif