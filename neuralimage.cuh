#ifndef NEURALIMAGE_CUH
#define NEURALIMAGE_CUH
#include <cuda_runtime.h>

typedef unsigned char uchar;
typedef unsigned int uint;

struct __align__(16) GlyphCommand {
    short x;
    short y;
    short u;
    short v;
    short width;
    short height;
};

cudaError launchCudaTensorMean(float2 scale, uchar3* data, int imageWidth, float* binding, int bindingWidth, int bindingHeight, float mean, cudaStream_t stream);


#endif // NEURALIMAGE_CUH
