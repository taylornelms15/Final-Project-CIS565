#ifndef __DETECTNETCU_H__
#define __DETECTNETCU_H__

#include "../inference/ObjectDetection.h"

// from detectNet.cu
cudaError_t cudaDetectionOverlay( float4* input, float4* output, uint32_t width, uint32_t height, Objectdetection::Detection* detections, int numDetections, float4* colors );

#endif