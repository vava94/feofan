//
// Created by vok on 06.08.2020.
//
#include "neuralimage.hpp"
#include "cudaUtils.h"

template<typename T_in, typename T_out>
__global__ void cuConvert(T_in *inImage, T_out *outImage, unsigned int width, unsigned int height) {

    const unsigned int
            _x = (blockIdx.x * blockDim.x) + threadIdx.x,
            _y = (blockIdx.y * blockDim.y) + threadIdx.y;
    if ( _x >= width || _y >= height ) return;

    const unsigned int _pixelPosition = _y * width + _x;
    const T_in _pixelValue = inImage[_pixelPosition];
    outImage[_pixelPosition] = make_vec<T_out>(_pixelValue.x, _pixelValue.y, _pixelValue.z, alpha(_pixelValue));

}

template<typename T, bool isBGR>
__global__ void gpuTensorMean( float2 scale, T* input, int iWidth, float* output, int oWidth, int oHeight, float mean)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    if( x >= oWidth || y >= oHeight )
        return;

    const int n = oWidth * oHeight;
    const int m = y * oWidth + x;

    const int dx = ((float)x * scale.x);
    const int dy = ((float)y * scale.y);

    const T px = input[ dy * iWidth + dx ];

    const float3 rgb = make_float3(px.x, px.y, px.z);

    output[n * 0 + m] = rgb.x / mean;
    output[n * 1 + m] = rgb.y / mean;
    output[n * 2 + m] = rgb.z / mean;

}

__global__ void gpuDrawBox(uchar3 *input, int boxX1, int boxY1, int boxX2, int boxY2, int imageWidth, dim3 color, uint lineWidth){

    const unsigned int _x = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int _y = blockIdx.y * blockDim.y + threadIdx.y;
    const unsigned int _pixelPosition = _y * imageWidth + _x;
    const float _a1 = 0.69f, _a2 = 0.3f;
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

template<bool isBGR>
cudaError_t launchTensorMean( void* input, size_t inputWidth, size_t inputHeight,
                              float* output, size_t outputWidth, size_t outputHeight,
                              const float& mean_value, cudaStream_t stream ) {

    if( !input || !output )
        return cudaErrorInvalidDevicePointer;

    if( inputWidth == 0 || outputWidth == 0 || inputHeight == 0 || outputHeight == 0 )
        return cudaErrorInvalidValue;

    const float2 scale = make_float2( float(inputWidth) / float(outputWidth),
                                      float(inputHeight) / float(outputHeight) );

    // launch kernel
    const dim3 blockDim(8, 8);
    const dim3 gridDim(iDivUp(outputWidth,blockDim.x), iDivUp(outputHeight,blockDim.y));

    gpuTensorMean<uchar3, isBGR><<<gridDim, blockDim, 0, stream>>>(scale, (uchar3 *)input, inputWidth, output, outputWidth, outputHeight, mean_value);

    return cudaGetLastError();

}

cudaError_t NeuralImage::cudaDrawBox(int boxX1, int boxX2, int boxY1, int boxY2, dim3 color, cudaStream_t stream, uint lineWidth, const std::string& label) {

    if(imageWidth == 0 || imageHeight == 0) {
        return cudaErrorInvalidValue;
    }

    const dim3
            _blocKDim3(32, 8, 1),
            _gridDim3(iDivUp(imageWidth, _blocKDim3.x), iDivUp(imageHeight, _blocKDim3.y), 3);

    if(lineWidth) {
        gpuDrawBox<<<_gridDim3, _blocKDim3, 0, stream>>>((uchar3 *) gpuImageData, boxX1, boxY1, boxX2, boxY2,
                                                         imageWidth, color, lineWidth);
    }

    if(label.data()) {
        font->OverlayText((void*)gpuImageData, imageWidth, imageHeight, label.data(), boxX1, boxY1 - font->height(),
                          make_float4(255,255,255,255), make_float4(color.x,color.y,color.z,176), 3);
    }

    return cudaSuccess;
}

cudaError_t NeuralImage::cudaTensorMeanRGB(void *input, size_t inputWidth, size_t inputHeight, float *output,
                                           size_t outputWidth, size_t outputHeight, const float &mean_value,
                                           cudaStream_t stream) {
    auto _a = launchTensorMean<false>(input, inputWidth, inputHeight, output, outputWidth, outputHeight, mean_value, stream);
//    cudaStreamSynchronize(stream);
    return _a;
}

cudaError_t NeuralImage::cudaRGB8toRGBA32(uchar3 *inImage, float3 *outImage, uint width, uint height) {

    if (!inImage || !outImage) {
        return cudaErrorInvalidDevicePointer;
    }
    if (width == 0 || height == 0) {
        return cudaErrorInvalidValue;
    }
    const dim3
            _blocKDim3(32, 8, 1),
            _gridDim3(iDivUp(width, _blocKDim3.x), iDivUp(height, _blocKDim3.y), 1);
    cuConvert<uchar3, float3><<<_gridDim3, _blocKDim3>>>(inImage, outImage, width, height);
    return cudaGetLastError();

}

cudaError_t NeuralImage::cudaRGBA32toRGB8(float3 *inImage, uchar3 *outImage, uint width, uint height) {
    if (!inImage || !outImage) {
        return cudaErrorInvalidDevicePointer;
    }
    if (width == 0 || height == 0) {
        return cudaErrorInvalidValue;
    }
    const dim3
            _blocKDim3(32, 8, 1),
            _gridDim3(iDivUp(width, _blocKDim3.x), iDivUp(height, _blocKDim3.y), 1);
    cuConvert<float3, uchar3><<<_gridDim3, _blocKDim3>>>(inImage, outImage, width, height);
    return cudaGetLastError();
}



