#ifndef __CUDA_RGB_CONVERT_H
#define __CUDA_RGB_CONVERT_H


#include "cudaUtility.h"
#include <stdint.h>

/**
 * Convert 8-bit fixed-point BGR image to 32-bit floating-point RGBA image
 * @ingroup colorspace
 */
cudaError_t cudaBGR8ToRGBA32( uchar3* input, float4* output, size_t width, size_t height );
	
cudaError_t cudaRGBA32ToBGR8( float4* input, uchar3* output, size_t width, size_t height );

cudaError_t cudaRGBA32ToBGR8( float4* input, uchar3* output, size_t width, size_t height, const float2& inputRange );


#endif