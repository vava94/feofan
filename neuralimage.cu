//
// Created by vok on 06.08.2020.
//
#include "neuralimage.hpp"
#include "neuralimage.cuh"
#include "utils.hpp"
#include <iostream>
typedef unsigned int uint;

__global__ void gpuDrawBox(uchar3* input, int boxX1, int boxY1, int boxX2, int boxY2, int imageWidth, dim3 color, uint lineWidth);

__global__ void cudaApplyDetectionData(
    uchar3* image, int width, int height, float2 scale, float* detetction, int lW, float opacity)//,
    //uchar* font, int fontWidth, int fontHeight, GlyphCommand* commands, size_t numCommands)
{
    /// [x0][y0][x1][y1][type][conf]
    
    int _pos = (blockIdx.z) * 6 * sizeof(float);
    //if (detetction[_pos + 5] < 0.3) return;

    int _x = blockIdx.x * blockDim.x + threadIdx.x;// +x0;
    int _y = blockIdx.y * blockDim.y + threadIdx.y;// +y0;
    if (_x >= width || _y >= height)  return;
    const unsigned int _pixelPosition = _y * width + _x;

    int _x0 = (int)(detetction[_pos] * scale.x);
    int _y0 = (int)(detetction[_pos + 2] * scale.y);
    int _x1    = (int)(detetction[_pos + 1] * scale.x);
    int _y1    = (int)(detetction[_pos + 3] * scale.y);

    const int _type  = (int)detetction[_pos + 4];

    if (_x1 > width) _x1 = width;
    if (_y1 > height) _y1 = height;

    

    /* if (lW > 0) {
        /// draw vertical lines
        if (
            /// Vertical lines
            /// ---
            /// | x0 - lW < x <= x0
            /// | y0 - lW < y < y1 + lW
            /// ---
            /// | x1 <= x < x1 + lW
            /// | y0 - lW < y < y1 + lW
            /// ---
            ((((x0 - lW) < x && x <= x0) || ((x1 <= x && x < (x1 + lW))))
                && (y0 - lW) < y && y < (y1 + lW)) ||
            /// Horizontal lines
            /// ---
            /// | y0 - lW < y <= y0
            /// | x0 - lW < x < x1 + lW
            /// ---
            /// | y1 <= y < y1 + lW
            /// | x0 - lW < x < x1 + lW
            /// ---
            ((((y0 - lW) < y && y <= y0) || ((y1 <= y && y < (y1 + lW))))
                && (x0 - lW) < x && x < (x1 + lW))
            ) 
        {
            image[_pixelPosition].x = r *opacity + image[_pixelPosition].x * (1 - opacity);
            image[_pixelPosition].y = g *opacity + image[_pixelPosition].y * (1 - opacity);
            image[_pixelPosition].z = b *opacity + image[_pixelPosition].z * (1 - opacity);
        } 
    }
    /*if (font && commands && y > y0 && x > x0) {
        /**
         * TODO: Исправить положение надписи 
         */
    /*    const auto& cmd = commands[blockIdx.x];
        const int u = cmd.u + threadIdx.x - 1;
        const int v = cmd.v + threadIdx.y - 1;
        const float pxGlyph = font[v * fontWidth + u];
        const uchar3 pxFont = { pxGlyph * r, pxGlyph * g, pxGlyph * b };
        image[y * width + x] = pxFont;
    }*/
}


__global__ void cudaTensorMean( float2 scale, uchar3* input, int iWidth, float* output, int oWidth, int oHeight, float mean) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    if( x >= oWidth || y >= oHeight )
        return;

    const int n = oWidth * oHeight;
    const int m = y * oWidth + x;

    const int dx = ((float)x * scale.x);
    const int dy = ((float)y * scale.y);

    const uchar3 px = input[dy * iWidth + dx];

    const float3 rgb = make_float3(px.x, px.y, px.z);

    output[n * 0 + m] = rgb.x / mean;
    output[n * 1 + m] = rgb.y / mean;
    output[n * 2 + m] = rgb.z / mean;

}

__global__ void gpuDrawBox(uchar3* input, int boxX1, int boxY1, int boxX2, int boxY2, int imageWidth, dim3 color, uint lineWidth) {

    const unsigned int _x = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int _y = blockIdx.y * blockDim.y + threadIdx.y;
    const unsigned int _pixelPosition = _y * imageWidth + _x;
    const float _a1 = 0.1f, _a2 = 0.90f;
        /// Отрисовка горизонтальныя линий
    if(_x >= boxX1 && _x <= boxX2 && ((_y > (boxY1 - lineWidth) && (_y <= boxY1)) || (_y >= boxY2 && (_y < boxY2 + lineWidth)))) {
        input[_pixelPosition].x = input[_pixelPosition].x * _a1 + color.x * _a2;
        input[_pixelPosition].y = input[_pixelPosition].y * _a1 + color.y * _a2;
        input[_pixelPosition].z = input[_pixelPosition].z * _a1 + color.z * _a2;
    }
    /// Отрисовка вертикальных линий
    if (_y >= boxY1 && _y <= boxY2 && (((_x > boxX1 - lineWidth) && (_x <= boxX1)) || (_x >= boxX2  && (_x < boxX2 + lineWidth)))) {
        input[_pixelPosition].x = input[_pixelPosition].x * _a1 + color.x * _a2;
        input[_pixelPosition].y = input[_pixelPosition].y * _a1 + color.y * _a2;
        input[_pixelPosition].z = input[_pixelPosition].z * _a1 + color.z * _a2;
    }
}


cudaError NeuralImage::launchCudaApplyDetections(cudaStream_t stream, int detCount, int lineWidth, float opacity) {
    const dim3 blockDim = dim3(32, 16, 1);
    const auto gridDim = dim3(utils::divUp(imageWidth, blockDim.x), utils::divUp(imageHeight, blockDim.y), detCount);
   
    cudaApplyDetectionData << <gridDim, blockDim>> > ((uchar3*)imageDataGpu, imageWidth, imageHeight, mBindingScale,
        detectionsGpu, lineWidth, opacity);// , fontMaps, fontWidth, fontHeight, commands, numCommands);
    //cudaStreamSynchronize(stream);
    cudaDeviceSynchronize();
    return cudaGetLastError();
}


cudaError launchCudaTensorMean(float2 scale, uchar3* data, int imageWidth, float* binding, int bindingWidth, int bindingHeight, float mean, cudaStream_t stream) {

    const dim3 blockDim = dim3(32, 8, 1);
    const auto gridDim = dim3(utils::divUp(bindingWidth, blockDim.x), utils::divUp(bindingHeight, blockDim.y), 1);
    cudaTensorMean <<<gridDim, blockDim, 0, stream>>> (scale, (uchar3*)data, imageWidth, binding, bindingWidth, bindingHeight, mean);
    return cudaGetLastError();

}

void NeuralImage::cudaDrawBox(int boxX1, int boxY1, int boxX2, int boxY2, dim3 color, uint lineWidth, cudaStream_t stream) {
    const dim3 blockDim = dim3(32, 8, 1);
    const auto gridDim = dim3(utils::divUp(imageWidth, blockDim.x), utils::divUp(imageHeight, blockDim.y), 1);
    gpuDrawBox<<<gridDim, blockDim, 0, stream>>>((uchar3*)imageDataGpu, boxX1, boxY1, boxX2, boxY2, imageWidth, color, lineWidth);
}