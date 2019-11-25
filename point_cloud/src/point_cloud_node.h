/**
* @file point_cloud_node.h
* @author Taylor Nelms
*/

#ifndef POINT_CLOUD_NODE_H
#define POINT_CLOUD_NODE_H

#include <ros/ros.h>
#include "std_msgs/String.h"

#include <sensor_msgs/Image.h>
#include <sensor_msgs/Imu.h>
#include <sensor_msgs/image_encodings.h>
#include <geometry_msgs/PointStamped.h>
#include <geometry_msgs/TransformStamped.h>
#include <vision_msgs/Detection2DArray.h>
#include <vision_msgs/VisionInfo.h>
#include <opencv2/core/mat.hpp>
#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <pcl/point_types.h>
#include <pcl/common/projection_matrix.h>
#include <pcl/io/pcd_io.h>
#include <pcl_ros/point_cloud.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#include <tf2/LinearMath/Vector3.h>
#include <tf2/LinearMath/Transform.h>

#include "TrifocalTensor.hpp"
#include "pointMatcher.h"

    typedef pcl::PointXYZRGBL PointT;


    int getEncodingTypeForCV(const std::string& encoding){
        int channels = sensor_msgs::image_encodings::numChannels(encoding);
        int depth = sensor_msgs::image_encodings::bitDepth(encoding);
        switch(channels){
        case 1:
            switch(depth){
            case 8:
                return CV_8UC1;
            case 16:
            default:
                return CV_16UC1;
            }
        case 3:
            switch(depth){
            case 8:
                return CV_8UC3;
            case 16:
            default:
                return CV_16UC3;
            }
        case 4:
        default:
            switch(depth){
            case 8:
                return CV_8UC4;
            case 16:
            default:
                return CV_16UC4;
            }
        }//switch


    }//getEncodingTypeforCV









#endif





