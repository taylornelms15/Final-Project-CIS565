/**
* @file point_cloud_node.h
* @author Taylor Nelms
*/

#ifndef POINT_CLOUD_NODE_H
#define POINT_CLOUD_NODE_H

#include <ros/ros.h>
#include "ptypes.h"
#include <pcl/point_types.h>
#include <pcl/common/projection_matrix.h>
#include <pcl/io/pcd_io.h>
#include <pcl/registration/icp_nl.h>
#include <pcl_ros/point_cloud.h>
#include <tf2_eigen/tf2_eigen.h>

#include "pointMatcher.h"

    typedef pcl::PointXYZRGBL PointT;
    typedef std::vector<PointT, Eigen::aligned_allocator<PointT> > PointT_vec;
    typedef pcl::PointCloud<PointT> PointT_cloud;
    typedef PointT_cloud::Ptr PointT_cloud_ptr;

    PointT_cloud_ptr makeCloudPtr(const PointT_cloud cloud){
        PointT_cloud_ptr pointer(new PointT_cloud);
        *pointer = cloud;
        return pointer;

    }//makeCloudPtr

    void setCloudPoints(PointT_cloud &cloud, PointT_vec &vec){
        cloud.points = vec;
        cloud.height = 1;
        cloud.width = vec.size();

    }//setCloudPoints


    ///Convenience function to convert our transformation matrices
    Eigen::Matrix4f     transformFromTf2(tf2::Transform xform);
    ///Convenience function to convert our transformation matrices
    tf2::Transform      transformFromEigen(Eigen::Matrix4f xform);

    /**
    @param cloud_src        The first-guess transformed src cloud
    @param cloud_tgt        The first-guess transformed dst cloud
    @param srcXform         The given world xform for the src cloud (already applied)
    @param dstXform         The given world xform for the dst cloud (already applied)
    @param output           Output variable: a representation of both stacked together (?)
    @param final_transform  Output variable: the actual transformation to get dst onto src (?)
    */
    void pairAlign(const PointT_cloud& cloud_src,
                   const PointT_cloud& cloud_tgt,
                   tf2::Transform& srcXform,
                   tf2::Transform& dstXform,
                   PointT_cloud& output,
                   tf2::Transform& final_transform);




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





