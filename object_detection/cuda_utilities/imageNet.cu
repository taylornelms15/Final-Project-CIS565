#include "cudaUtility.h"


// gpuPreImageNetNormBGR
__global__ void gpuPreImageNetNormBGR( float2 scale, float4* input, int iWidth, float* output, int oWidth, int oHeight, float multiplier, float min_value )
{
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;

	if( x >= oWidth || y >= oHeight )
		return;

	const int n = oWidth * oHeight;
	const int m = y * oWidth + x;

	const int dx = ((float)x * scale.x);
	const int dy = ((float)y * scale.y);

	const float4 px  = input[ dy * iWidth + dx ];
	const float3 bgr = make_float3(px.z, px.y, px.x);
	
	output[n * 0 + m] = bgr.x * multiplier + min_value;
	output[n * 1 + m] = bgr.y * multiplier + min_value;
	output[n * 2 + m] = bgr.z * multiplier + min_value;
}


// cudaPreImageNetNorm
cudaError_t cudaPreImageNetNormBGR( float4* input, size_t inputWidth, size_t inputHeight,
								 float* output, size_t outputWidth, size_t outputHeight,
								 const float2& range, cudaStream_t stream )
{
	if( !input || !output )
		return cudaErrorInvalidDevicePointer;

	if( inputWidth == 0 || outputWidth == 0 || inputHeight == 0 || outputHeight == 0 )
		return cudaErrorInvalidValue;

	const float2 scale = make_float2( float(inputWidth) / float(outputWidth),
							    float(inputHeight) / float(outputHeight) );

	const float multiplier = (range.y - range.x) / 255.0f;
	
	//printf("cudaPreImageNetNorm([%f, %f])  multiplier=%f\n", range.x, range.y, multiplier);
	
	// launch kernel
	const dim3 blockDim(8, 8);
	const dim3 gridDim(iDivUp(outputWidth,blockDim.x), iDivUp(outputHeight,blockDim.y));

	gpuPreImageNetNormBGR<<<gridDim, blockDim, 0, stream>>>(scale, input, inputWidth, output, outputWidth, outputHeight, multiplier, range.x);

	return CUDA(cudaGetLastError());
}
