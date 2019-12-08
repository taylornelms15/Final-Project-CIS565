#ifndef __DETECTNETCU_H__
#define __DETECTNETCU_H__

#include "../inference/ObjectDetection.h"

cudaError_t cudaDetectionOverlay( float4* input, uint32_t width, uint32_t height, ObjectDetection::Detection* detections, int numDetections, float4* colors );

#endif
