
#include "detectNet.h"
#include "cudaUtility.h"



template<typename T>
__global__ void gpuDetectionOverlay( T* input, T* output, int width, int height, ObjectDetection::Detection* detections, int numDetections, float4* colors ) 
{
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;

	if( x >= width || y >= height )
		return;

	const T px_in = input[ y * width + x ];
	T px_out = px_in;
	
	const float fx = x;
	const float fy = y;
	
	for( int n=0; n < numDetections; n++ )
	{
		const detectNet::Detection det = detections[n];

		// check if this pixel is inside the bounding box
		if( fx >= det.Left && fx <= det.Right && fy >= det.Top && fy <= det.Bottom )
		{
			const float4 color = colors[det.ClassID];	

			const float alpha = color.w / 255.0f;
			const float ialph = 1.0f - alpha;

			px_out.x = alpha * color.x + ialph * px_out.x;
			px_out.y = alpha * color.y + ialph * px_out.y;
			px_out.z = alpha * color.z + ialph * px_out.z;
		}
	}
	
	output[y * width + x] = px_out;	 
}


template<typename T>
__global__ void gpuDetectionOverlayBox( T* input, T* output, int imgWidth, int imgHeight, int x0, int y0, int boxWidth, int boxHeight, const float4 color ) 
{
	const int box_x = blockIdx.x * blockDim.x + threadIdx.x;
	const int box_y = blockIdx.y * blockDim.y + threadIdx.y;

	if( box_x >= boxWidth || box_y >= boxHeight )
		return;

	const int x = box_x + x0;
	const int y = box_y + y0;

	if( x >= imgWidth || y >= imgHeight )
		return;

	const T px_in = input[ y * imgWidth + x ];

	const float alpha = color.w / 255.0f;
	const float ialph = 1.0f - alpha;

	output[y * imgWidth + x] = make_float4( alpha * color.x + ialph * px_in.x, 
					    alpha * color.y + ialph * px_in.y,
					    alpha * color.z + ialph * px_in.z,
					    px_in.w );
}

cudaError_t cudaDetectionOverlay( float4* input, float4* output, uint32_t width, uint32_t height, ObjectDetection::Detection* detections, int numDetections, float4* colors )
{
	if( !input || !output || width == 0 || height == 0 || !detections || numDetections == 0 || !colors )
		return cudaErrorInvalidValue;

	// if input and output are the same image, then we can use the faster method
	// which draws 1 box per kernel, but doesn't copy pixels that aren't inside boxes
	if( input == output )
	{
		for( int n=0; n < numDetections; n++ )
		{
			const int boxWidth = (int)detections[n].Width();
			const int boxHeight = (int)detections[n].Height();

			// launch kernel
			const dim3 blockDim(8, 8);
			const dim3 gridDim(iDivUp(boxWidth,blockDim.x), iDivUp(boxHeight,blockDim.y));

			gpuDetectionOverlayBox<float4><<<gridDim, blockDim>>>(input, output, width, height, (int)detections[n].Left, (int)detections[n].Top, boxWidth, boxHeight, colors[detections[n].ClassID]); 
		}
	}
	else
	{
		// launch kernel
		const dim3 blockDim(8, 8);
		const dim3 gridDim(iDivUp(width,blockDim.x), iDivUp(height,blockDim.y));

		gpuDetectionOverlay<float4><<<gridDim, blockDim>>>(input, output, width, height, detections, numDetections, colors); 
	}

	return cudaGetLastError();
}

