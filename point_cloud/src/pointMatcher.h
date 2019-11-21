/**
@file pointMatcher.h
@author Taylor Nelms
*/

#ifndef POINT_MATCHER_H
#define POINT_MATCHER_H


#include <cuda.h>
#include <device_launch_parameters.h>
#include <cuda_runtime.h>



/**
A dumb test function to run a 32-element dot product
Only in here to figure out compiler issues ahead of time
*/
float testCudaFunctionality(float* arrayOne, float* arrayTwo);




#endif
