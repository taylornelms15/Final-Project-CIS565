#ifndef __DETECTNETCU_H__
#define __DETECTNETCU_H__

// from detectNet.cu
cudaError_t cudaDetectionOverlay( float4* input, float4* output, uint32_t width, uint32_t height, detectNet::Detection* detections, int numDetections, float4* colors );

#endif