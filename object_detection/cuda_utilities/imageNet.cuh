#ifndef __IMAGE_NET_PREPROCESSING_H__
#define __IMAGE_NET_PREPROCESSING_H__


#include "cudaUtility.h"


cudaError_t cudaPreImageNetNormBGR( float4* input, size_t inputWidth, size_t inputHeight, float* output, size_t outputWidth, size_t outputHeight, const float2& range, cudaStream_t stream );


#endif

