/**
@file pointMatcher.h
@author Taylor Nelms
*/

#ifndef POINT_MATCHER_H
#define POINT_MATCHER_H


#include <cuda.h>
#include <device_launch_parameters.h>
#include <cuda_runtime.h>

#include <opencv2/core/mat.hpp>
#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <tf2/LinearMath/Vector3.h>
#include <tf2/LinearMath/Transform.h>
#include <glm/glm.hpp>

using namespace cv;
using namespace cv::xfeatures2d;

#define FILENAME (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__)
#define checkCUDAError(msg) checkCUDAErrorFn(msg, FILENAME, __LINE__)

/**
 * Check for CUDA errors; print and exit if there was a problem.
 */
void checkCUDAErrorFn(const char *msg, const char *file = NULL, int line = -1);

/**
Because of difficulties in getting access to PCL functions within the CUDA libraries,
this class allows us to encapsulate some point functionality without it
*/
typedef struct PointSub{
    float x;
    float y;
    float z;
    uint8_t r;
    uint8_t g;
    uint8_t b;
    //uint32_t label;
} PointSub;



void getCameraIntrinsicMatrix(
        Mat                     img1,
        std::vector<KeyPoint>   keypoints1,
        tf2::Transform          xform1,
        Mat                     img2,
        std::vector<KeyPoint>   keypoints2,
        tf2::Transform          xform2,
        std::vector<DMatch>     good_matches,
        Mat*                    output
);

std::vector<PointSub> getMatchingWorldPoints(
        Mat                     img1,
        std::vector<KeyPoint>   keypoints1,
        tf2::Transform          xform1,
        Mat                     img2,
        std::vector<KeyPoint>   keypoints2,
        tf2::Transform          xform2,
        std::vector<DMatch>     good_matches,
        float                   FoV
);

std::vector<PointSub> getMatchingWorldPointsAlt(
        Mat                     img1,
        std::vector<KeyPoint>   keypoints1,
        tf2::Transform          xform1,
        Mat                     img2,
        std::vector<KeyPoint>   keypoints2,
        tf2::Transform          xform2,
        std::vector<DMatch>     good_matches,
        float                   FoV
);

/**
A dumb test function to run a 32-element dot product
Only in here to figure out compiler issues ahead of time
*/
float testCudaFunctionality(float* arrayOne, float* arrayTwo);




#endif
