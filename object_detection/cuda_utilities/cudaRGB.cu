#include "cudaRGB.h"

__global__ void RGBToRGBAf(uchar3* srcImage,
                           float4* dstImage,
                           int width, int height)
{
	const int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	const int y = (blockIdx.y * blockDim.y) + threadIdx.y;
	
	const int pixel = y * width + x;

	if( x >= width )
		return; 

	if( y >= height )
		return;

//	printf("cuda thread %i %i  %i %i pixel %i \n", x, y, width, height, pixel);
		
	const float  s  = 1.0f;
	const uchar3 px = srcImage[pixel];
	
	dstImage[pixel] = make_float4(px.z * s, px.y * s, px.x * s, 255.0f * s);
}

cudaError_t cudaBGR8ToRGBA32( uchar3* srcDev, float4* destDev, size_t width, size_t height )
{
	if( !srcDev || !destDev )
		return cudaErrorInvalidDevicePointer;

	const dim3 blockDim(8,8,1);
	const dim3 gridDim(iDivUp(width,blockDim.x), iDivUp(height,blockDim.y), 1);

	RGBToRGBAf<<<gridDim, blockDim>>>( srcDev, destDev, width, height );
	
	return CUDA(cudaGetLastError());
}

__global__ void RGBAToRGB8(float4* srcImage,
                           uchar3* dstImage,
                           int width, int height,
					  float scaling_factor)
{
	const int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	const int y = (blockIdx.y * blockDim.y) + threadIdx.y;
	
	const int pixel = y * width + x;

	if( x >= width )
		return; 

	if( y >= height )
		return;

	const float4 px = srcImage[pixel];

	dstImage[pixel] = make_uchar3(px.z * scaling_factor,px.y * scaling_factor,px.x * scaling_factor);
}

cudaError_t cudaRGBA32ToBGR8( float4* srcDev, uchar3* destDev, size_t width, size_t height, const float2& inputRange )
{
	if( !srcDev || !destDev )
		return cudaErrorInvalidDevicePointer;

	if( width == 0 || height == 0 )
		return cudaErrorInvalidValue;

	const float multiplier = 255.0f / inputRange.y;

	const dim3 blockDim(8,8,1);
	const dim3 gridDim(iDivUp(width,blockDim.x), iDivUp(height,blockDim.y), 1);

	RGBAToRGB8<<<gridDim, blockDim>>>( srcDev, destDev, width, height, multiplier );
	
	return CUDA(cudaGetLastError());
}

cudaError_t cudaRGBA32ToBGR8( float4* srcDev, uchar3* destDev, size_t width, size_t height )
{
	return cudaRGBA32ToBGR8(srcDev, destDev, width, height, make_float2(0.0f, 255.0f));
}
