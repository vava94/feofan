//
// Created by vok on 06.08.2020.
//
/// TODO: Function for font loading
/// 
#include "neuralimage.hpp"
#include "neuralimage.cuh"
#define STB_TRUETYPE_IMPLEMENTATION
#include "stb/stb_truetype.h"
#include "utils.hpp"

#include <memory.h>
#include <iostream>
#include <fstream>

#define TAG "NI: "

using namespace std;

struct DrawingString {
    short x;
    short y;
    float4 color;
    std::string text;
};

static uint8_t* fontMapsCpu = nullptr;
static uint8_t* fontMapsGpu = nullptr;
static uint8_t fontSize = 0;

/// Font parameteres ---
static bool fontOk = false;

static void* glyphCommandsCpu;
static void* glyphCommandsGpu;
static void* rectsCpu;
static void* rectsGpu;
static void* commandsCpu;
static void* commandsGpu;

static uint
fontMapHeight,
fontMapWidth;

static const uint
firstGlyph = 32,
lastGlyph = 255,
numGlyphs = lastGlyph - firstGlyph,
maxCommands = 1024;


static struct GlyphInfo
{
    uint16_t x;
    uint16_t y;
    uint16_t width;
    uint16_t height;

    float xAdvance;
    float xOffset;
    float yOffset;
} glyphInfos[numGlyphs];

NeuralImage::NeuralImage(int camIndex, int width, int height, int numChannels) {

    imageChannels = numChannels;
    imageHeight = height;
    imageIndex = camIndex;
    imageWidth = width;
    dataSize = imageWidth * imageHeight * imageChannels;
    //neuralSize = imageWidth * imageHeight * sizeof(float4);

    /// ---
    /// TODO: replace with utils::cudaAllocMApped
    if (cudaHostAlloc(&imageDataCpu, dataSize, cudaHostAllocMapped) != cudaSuccess ||
        cudaHostGetDevicePointer(&imageDataGpu, imageDataCpu, 0) != cudaSuccess) {
        if (utils::Log) utils::Log(TAG + std::string("GPU memory allocation error for image data. Data size: ") + std::to_string(dataSize) + ".", 2);
        return;
    }
    cudaMalloc(&detectionsGpu, sizeof(float) * 600);
    cudaDetectionsSize = 100;
    //if(cudaMalloc(&neuralData, neuralSize) != cudaSuccess) {
    //    if (utils::Log) utils::Log(TAG"GPU memory allocation error for neural data. Data size: " + std::to_string(neuralSize) + ".", 2);
    //    return;
    //}
    ///---
    // font = cudaFont::Create(adaptFontSize(imageWidth));
    initOk = true;

}

void NeuralImage::addText(string text, float4 color, short x, short y) {
    if (text.empty()) return;
    vector<DrawingString> _strings;
    DrawingString _ds;
    _ds.x = x;
    _ds.y = y;
    _ds.color = color;
    _ds.text = std::move(text);
    _strings.emplace_back(std::move(_ds));
}

void NeuralImage::applyDetections(int count, int applyFlags, int lineWidth, float opacity, vector<string> labels) {

    int 
        _numGlyphCommands = 0, 
        _maxFontHeight = 0, 
        _fontWidth = 0;

    std::string _label;
    numDetections = count;
    GlyphCommand* _glyphCmd = nullptr;

    if (applyFlags > 0) {

        float _b, _g, _r;
        int _type, _pixel, _x, _x0, _x1, _y, _y0, _y1;
        size_t _pos;

#pragma omp parallel for 
        for (int _n = 0; _n < count; _n++) {
            /// Отрисовка линий рамок
            _pos = sizeof(float) * 6 * _n;
            if (detectionsCpu[_pos + 5] < 0.6) continue;
            _type = (int)detectionsCpu[_pos + 4];
            _r = abs(sin(_type * 4.5 * 3.14 / 180)) * 255;
            _g = abs(sin(( _type + 26) * 4.5 * 3.14 / 180)) * 255;
            _b = abs(sin((_type + 53) * 4.5 * 3.14 / 180)) * 255;
            _x0 = detectionsCpu[_pos] * mBindingScale.x;
            _y0 = detectionsCpu[_pos + 1] * mBindingScale.y;
            _x1 = detectionsCpu[_pos + 2] * mBindingScale.x;
            _y1 = detectionsCpu[_pos + 3] * mBindingScale.y;

            if (_x0 < 0) _x0 = 0;
            if (_y0 < 0) _y0 = 0;
            if (_x1 > imageWidth) _x1 = imageWidth;
            if (_y1 > imageHeight) _y1 = imageHeight;

            // Отрисовка рамки
            if (applyFlags && BOX) {
                for (int _ix = _x0; _ix < _x1; _ix++) {
                    for (int _lw = 0; _lw < lineWidth; _lw++) {
                        _y = _y0 - _lw;
                        if (_y > -1) {
                            _pixel = _y * imageWidth + _ix;
                            imageDataCpu[_pixel].x = _r;
                            imageDataCpu[_pixel].y = _g;
                            imageDataCpu[_pixel].z = _b;
                        }
                        _y = _y1 + _lw;
                        if (_y < imageHeight) {
                            _pixel = _y * imageWidth + _ix;
                            imageDataCpu[_pixel].x = _r;
                            imageDataCpu[_pixel].y = _g;
                            imageDataCpu[_pixel].z = _b;
                        }
                    }
                }

                for (int _iy = _y0; _iy < _y1; _iy++) {
                    for (int _lw = 0; _lw < lineWidth; _lw++) {
                        _x = _x0 - _lw;
                        if (_x > -1) {
                            _pixel = _iy * imageWidth + _x;
                            imageDataCpu[_pixel].x = _r;
                            imageDataCpu[_pixel].y = _g;
                            imageDataCpu[_pixel].z = _b;
                        }
                        _x = _x1 + _lw;
                        if (_x < imageWidth) {
                            _pixel = _iy * imageWidth + _x;
                            imageDataCpu[_pixel].x = _r;
                            imageDataCpu[_pixel].y = _g;
                            imageDataCpu[_pixel].z = _b;
                        }
                    }
                }
            }
            // Нанесение надписи
            if (applyFlags && LABEL) {
                if (labels.empty()) {
                    if (utils::Log) utils::Log(TAG"The flag 'LABELS' is set, but labels array is empty.", 1);
                    applyFlags -= LABEL;
                    continue;
                }
                char _c;
                int  _textHeight = 0, _textWidth = 0, _yOffset, _tempPoint;
                int2 _textPosition;
                auto _label = labels[detectionsCpu[_pos + 4]];
                for (int _n = 0; _n < _label.size(); _n++) {
                    _c = labels[detectionsCpu[_pos + 4]][_n];
                    if (_c < firstGlyph || _c > lastGlyph)continue;
                    _textHeight = abs(glyphInfos[_c].yOffset) > _textHeight ? 
                        abs(glyphInfos[_c].yOffset) : _textHeight;
                    _textWidth += glyphInfos[_c].width;
                }
                // Координата Х
                _tempPoint = (_x0 + 2);
                if ((_tempPoint + _textWidth) > (imageWidth - 1)) _tempPoint -= (imageWidth - 1 - _tempPoint - _textWidth);
                _textPosition.x = _tempPoint < 0 ? 0 : _tempPoint;
                // Координата Y
                _tempPoint = (_y0 - _textHeight - 2);
                if ((_tempPoint + _textHeight) > (imageWidth - 1)) _tempPoint -= (imageWidth - 1 - _tempPoint - _textHeight);
                _textPosition.y = _tempPoint < 0 ? 0 : _tempPoint;

            }
        }
    }

}

float2 NeuralImage::bindingScale() const {
    return mBindingScale;
}

int NeuralImage::channels() const {
    return imageChannels;
}

uint8_t* NeuralImage::data() {

    if(!initOk) {
        return nullptr;
    }
    return (uint8_t*)imageDataCpu;

}

float** NeuralImage::detectionsDataPtr() {
    return &detectionsCpu;
}



uint8_t *NeuralImage::gpuData() const {
    return (uint8_t*)imageDataGpu;
}

int NeuralImage::height() const {
    return imageHeight;
}

int NeuralImage::index() const {
    return imageIndex;
}

void NeuralImage::lockImage() {
    imageLocked = true;
}

void NeuralImage::loadFont(FILE* fontFile, uint8_t fontSize) {

    fontOk = false;
    if (!fontFile) {
        utils::Log(TAG"Null pointer to font file.", 2);
        return;
    }
    if (fontMapsCpu) {
        cudaFreeHost(fontMapsCpu);
    }
    if (fontMapsGpu) {
        cudaFree(fontMapsGpu);
    }
    fontMapHeight = 256;
    fontMapWidth = 256;
    std::ifstream _ifs(fontFile);
    _ifs.seekg(std::ios::end);
    auto _filesize = _ifs.tellg();
    auto _fontFileBuffer = new uint8_t(_filesize);
    _ifs.close();
    auto _readSize = fread(_fontFileBuffer, 1, _filesize, fontFile);
    fclose(fontFile);
    if (_readSize != _filesize) {
        if (utils::Log) utils::Log(TAG"Error while reading font file.", 2);
        delete _fontFileBuffer;
        return;
    }
    /// Buffer with the coordinates of backed glyphs
    stbtt_bakedchar _backedChars[numGlyphs];
    int _result;
    /// Fitting the size of the glyph bitmap
    size_t _fontMapSize;
    while (true) {
        _fontMapSize = fontMapWidth * fontMapHeight * sizeof(uchar);
        if (!utils::cudaAllocMapped((void**)&fontMapsCpu, (void**)&fontMapsGpu, _fontMapSize)) {
            if (utils::Log) utils::Log(TAG"Failed to allocate " + to_string(_fontMapSize) + " bytes to store " +
                to_string(fontMapWidth) + "x" + to_string(fontMapHeight) + " font map.", 2);
            delete _fontFileBuffer;
            return;
        }
        _result = stbtt_BakeFontBitmap(_fontFileBuffer, 0, _filesize, fontMapsCpu, fontMapWidth, fontMapHeight, firstGlyph, numGlyphs, _backedChars);
        if (_result == 0) {
            if (utils::Log) utils::Log(TAG"Failed to bake font bitmaps.", 2);
            delete _fontFileBuffer;
            return;
        }
        else if (_result < 0) {
            const int _glyphsPacked = _result * -1;
            if (_glyphsPacked == numGlyphs) {
                if (utils::Log) utils::Log(TAG"The font glyphs are packaged successfully.", 0);
                break;
            }
            cudaFreeHost(fontMapsCpu);
            cudaFree(fontMapsGpu);
            fontMapHeight *= 2;
            fontMapWidth *= 2;
        }
        else {
            break;
        }
    }
    delete _fontFileBuffer;
    /// Get texture coordinates
    for (uint _n = 0; _n < numGlyphs; _n++) {
        glyphInfos[_n].x = _backedChars[_n].x0;
        glyphInfos[_n].y = _backedChars[_n].y0;
        glyphInfos[_n].width = _backedChars[_n].x1 - _backedChars[_n].x0;
        glyphInfos[_n].height = _backedChars[_n].y1 - _backedChars[_n].y0;
        glyphInfos[_n].xAdvance = _backedChars[_n].xadvance;
        glyphInfos[_n].xOffset = _backedChars[_n].xoff;
        glyphInfos[_n].yOffset = _backedChars[_n].yoff;
    }
    if (utils::cudaAllocMapped(&commandsCpu, &commandsGpu, sizeof(GlyphCommand) * maxCommands)) {
        if (utils::Log) utils::Log(TAG"Failed to allocate memory for glyph commands.", 2);
        return;
    }
    if (utils::cudaAllocMapped(&rectsCpu, &rectsGpu, sizeof(float4) * maxCommands)) {
        if (utils::Log) utils::Log(TAG"Failed to allocate memory for glyph's rects.", 2);
        return;
    }
    fontOk = true;
}

void NeuralImage::newData(uint8_t *data) {

    if(!initOk || imageLocked) return;
    memcpy(imageDataCpu, data, dataSize);

}

void NeuralImage::prepareTensor(cudaStream_t cudaStream, float *inputBinding, size_t bindingWidth,
                                size_t bindingHeight, float mean) 
{

    if (!inputBinding || bindingWidth == 0 || bindingHeight == 0 || mean == 0) {
        if (utils::Log) utils::Log(TAG"Null value passed to 'prepareTensor' function.", 2);
        return;
    }

    mBindingScale.x = (float)imageWidth / (float)bindingWidth;
    mBindingScale.y = (float)imageHeight / (float)bindingHeight;
    
    if(launchCudaTensorMean(mBindingScale, (uchar3*)imageDataGpu, imageWidth, inputBinding, bindingWidth, bindingHeight, mean, cudaStream) != cudaSuccess) {
        if (utils::Log) utils::Log(TAG"Error cudaTensorMeanRGB function.", 2);
        mBindingScale.x = 0;
        mBindingScale.y = 0;
    }
    else {
        cudaStreamSynchronize(cudaStream);
    }
    
}

void NeuralImage::unlockImage() {
    imageLocked = false;
}

int NeuralImage::width() const {
    return imageWidth;
}

NeuralImage::~NeuralImage(){
    initOk = false;
    newImage = false;
    //delete neuralData;
    cudaFreeHost(imageDataCpu);
    cudaFree(imageDataGpu);
    cudaFreeHost(fontMapsCpu);
    cudaFree(fontMapsGpu);
}