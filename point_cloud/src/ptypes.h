/**
@file ptypes.h
@author Taylor Nelms
Puts in the relevant typedefs and library includes for them
Should have few dependencies, be common to most functions we use
*/

#ifndef POINT_TYPES_H
#define POINT_TYPES_H

#define CERES_FOUND 1
#include <opencv2/opencv.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
//#include <opencv2/features2d.hpp>
//#include <opencv2/xfeatures2d.hpp>
#include <opencv2/rgbd.hpp>
#include <tf2/LinearMath/Vector3.h>
#include <tf2/LinearMath/Transform.h>
#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>
#include "std_msgs/String.h"
#include <sensor_msgs/Image.h>
#include <sensor_msgs/CameraInfo.h>
#include <sensor_msgs/Imu.h>
#include <sensor_msgs/image_encodings.h>
#include <geometry_msgs/PointStamped.h>
#include <geometry_msgs/TransformStamped.h>
#include <vision_msgs/Detection2DArray.h>
#include <vision_msgs/VisionInfo.h>
#include <drone_mom_msgs/drone_mom.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#include <tf2/LinearMath/Vector3.h>
#include <tf2/LinearMath/Transform.h>



//vector typedefs
typedef cv::KeyPoint KeyPoint;
typedef cv::DMatch DMatch;
typedef cv::Point2f Point2f;
typedef cv::Vec3b Vec3b;
typedef std::vector<Point2f> Point2f_vec;
typedef std::vector<float> float_vec;
typedef std::vector<double> double_vec;
typedef std::vector<int> int_vec;
typedef std::vector<KeyPoint> KeyPoint_vec;
typedef std::vector<DMatch> DMatch_vec;
typedef std::vector<tf2::Vector3> Vector3_vec;
typedef glm::vec3 gvec3;
typedef std::vector<gvec3> gvec3_vec;









#endif
