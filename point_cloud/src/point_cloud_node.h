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
#include <pcl/filters/approximate_voxel_grid.h>
#include <pcl/filters/statistical_outlier_removal.h>
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

    ///Just a dumb little function renaming
    void cloudwrite(const char filename[], const PointT_cloud cloud){
        pcl::io::savePCDFile(filename, cloud);
    }//outputCloud

    /**
    Does some initial filtering to cull outliers, reduce overall point number
    */
    PointT_cloud filterIncoming(PointT_cloud &incoming);

    PointT_cloud filterVoxel(const PointT_cloud cloud, double filterscale);
    PointT_cloud filterOutlier(const PointT_cloud cloud);

    ///Convenience function to convert our transformation matrices
    Eigen::Matrix4f     transformFromTf2(const tf2::Transform xform);
    ///Convenience function to convert our transformation matrices
    tf2::Transform      transformFromEigen(const Eigen::Matrix4f xform);

    /**
    @param cloud_tgt        The first-guess transformed tgt cloud
    @param cloud_src        The first-guess transformed src cloud
    @param dstXform         The given world xform for the tgt cloud (already applied)
    @param srcXform         The given world xform for the src cloud (already applied)

    @return                 Estimated transformation to put the src cloud onto the tgt cloud
    */
    tf2::Transform pairAlign(const PointT_cloud& cloud_tgt,
                   const PointT_cloud& cloud_src,
                   tf2::Transform& tgtXform,
                   tf2::Transform& srcXform);




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





