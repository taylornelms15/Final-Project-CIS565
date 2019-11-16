#include <ros/ros.h>
#include "std_msgs/String.h"
#include <sensor_msgs/Image.h>
#include <sensor_msgs/PointCloud2.h>
#include <vision_msgs/Detection2DArray.h>
#include <vision_msgs/VisionInfo.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/features/normal_3d.h>
#include <pcl/surface/gp3.h>
#include <pcl_conversions/pcl_conversions.h>
#include <string.h>
#include <sstream>
#include <iomanip>
#include <pcl/io/vtk_io.h>

static int iteration = 0;

   std::vector<std::string> class_descriptions;
   std::string key;
  /**
   * This tutorial demonstrates simple receipt of messages over the ROS system.
   * http://wiki.ros.org/ROS/Tutorials/WritingPublisherSubscriber%28c%2B%2B%29
   */
  // this callback is only called once once you subscribe
   void VisionCallback(const vision_msgs::VisionInfo& msg)
   {
    // print stuff to show 
    ROS_INFO("DataBase: %s", msg.database_location.c_str());
    ROS_INFO("method: %s", msg.method.c_str());
     
     // get the key to the classification database and store it in our vector
     key = msg.database_location.c_str();
     ros::NodeHandle nh("~");
     nh.getParam(key, class_descriptions);

   }

   // this is registered every image sent
   void DetectionCallback(const vision_msgs::Detection2DArray& msg)
   {
      // print stuff to show how stuff works
     // Vision messages can be found here http://docs.ros.org/melodic/api/vision_msgs/html/msg/Detection2DArray.html
     int detected_elements = msg.detections.size();
     for(int i = 0; i < detected_elements; i++)
     {
        ROS_INFO("bbox X: %9.6f", msg.detections[i].bbox.size_x);
        ROS_INFO("bbox cx: %9.6f", msg.detections[i].bbox.center.x);
        ROS_INFO("bbox Y: %9.6f", msg.detections[i].bbox.size_y);
        ROS_INFO("bbox cy: %9.6f", msg.detections[i].bbox.center.y);

        // there is only 1 result per detection I am 99% sure
        ROS_INFO("confidece: %1.6f", msg.detections[i].results[0].score);     
     
        int idx = msg.detections[i].results[0].id;


        // since we got the database from setup we can now see what our bounding box contains!
        ROS_INFO("classified: %s", class_descriptions[idx].c_str());

     }
   }

   // this is registered every image sent
   void PointCloud2Callback(const sensor_msgs::PointCloud2& msg)
   {
     ROS_INFO("Got message!");

	// Example code below from http://www.pointclouds.org/documentation/tutorials/greedy_projection.php

     pcl::PCLPointCloud2* conv_cloud = new pcl::PCLPointCloud2;
     // Note, moveToPCL destroys the message structure.
     pcl_conversions::toPCL(msg, *conv_cloud);

     // Now move to totally independent PCL Functions
     // ROS dependence stops here
     pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZRGB>);
     pcl::fromPCLPointCloud2 (*conv_cloud, *cloud);

     // Now cloud is in PCL data type, this can be massaged to create triangle meshes
     // and ultimately GLTF files.

	  // Normal estimation*
	  pcl::NormalEstimation<pcl::PointXYZRGB, pcl::Normal> n;
	  pcl::PointCloud<pcl::Normal>::Ptr normals (new pcl::PointCloud<pcl::Normal>);
	  pcl::search::KdTree<pcl::PointXYZRGB>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZRGB>);
	  tree->setInputCloud (cloud);
	  n.setInputCloud (cloud);
	  n.setSearchMethod (tree);
	  n.setKSearch (20);
	  n.compute (*normals);
	  //* normals should not contain the point normals + surface curvatures

	  // Concatenate the XYZ and normal fields*
	  pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud_with_normals (new pcl::PointCloud<pcl::PointXYZRGBNormal>);
	  pcl::concatenateFields (*cloud, *normals, *cloud_with_normals);
	  //* cloud_with_normals = cloud + normals

	  // Create search tree*
	  pcl::search::KdTree<pcl::PointXYZRGBNormal>::Ptr tree2 (new pcl::search::KdTree<pcl::PointXYZRGBNormal>);
	  tree2->setInputCloud (cloud_with_normals);

	  // Initialize objects
	  pcl::GreedyProjectionTriangulation<pcl::PointXYZRGBNormal> gp3;
	  pcl::PolygonMesh triangles;

	  // Set the maximum distance between connected points (maximum edge length)
	  gp3.setSearchRadius (0.025);

	  // Set typical values for the parameters
	  gp3.setMu (2.5);
	  gp3.setMaximumNearestNeighbors (100);
	  gp3.setMaximumSurfaceAngle(M_PI/4); // 45 degrees
	  gp3.setMinimumAngle(M_PI/18); // 10 degrees
	  gp3.setMaximumAngle(2*M_PI/3); // 120 degrees
	  gp3.setNormalConsistency(false);

	  // Get result
	  gp3.setInputCloud (cloud_with_normals);
	  gp3.setSearchMethod (tree2);
	  gp3.reconstruct (triangles);

	  // Additional vertex information
	  std::vector<int> parts = gp3.getPartIDs();
	  std::vector<int> states = gp3.getPointStates();

	  iteration++;
	  std::string filename = "mesh_" + std::to_string(iteration) + ".vtk";
	  pcl::io::saveVTKFile (filename, triangles);
   }
   
   
   int main(int argc, char **argv)
   {
     /**
      * The ros::init() function needs to see argc and argv so that it can perform
      * any ROS arguments and name remapping that were provided at the command line.
      * For programmatic remappings you can use a different version of init() which takes
      * remappings directly, but for most command-line programs, passing argc and argv is
      * the easiest way to do it.  The third argument to init() is the name of the node.
      *
      * You must call one of the versions of ros::init() before using any other
      * part of the ROS system.
      */
     ros::init(argc, argv, "mesh_construct");
   
     /**
      * NodeHandle is the main access point to communications with the ROS system.
      * The first NodeHandle constructed will fully initialize this node, and the last
      * NodeHandle destructed will close down the node.
      */
     ros::NodeHandle n;
   
     /**
      * The subscribe() call is how you tell ROS that you want to receive messages
      * on a given topic.  This invokes a call to the ROS
      * master node, which keeps a registry of who is publishing and who
      * is subscribing.  Messages are passed to a callback function, here
      * called chatterCallback.  subscribe() returns a Subscriber object that you
      * must hold on to until you want to unsubscribe.  When all copies of the Subscriber
      * object go out of scope, this callback will automatically be unsubscribed from
      * this topic.
      *
      * The second parameter to the subscribe() function is the size of the message
      * queue.  If messages are arriving faster than they are being processed, this
      * is the number of messages that will be buffered up before beginning to throw
      * away the oldest ones.
      * Vision messages can be found here http://docs.ros.org/melodic/api/vision_msgs/html/msg/Detection2DArray.html
      * 
      */
     ros::Subscriber sub = n.subscribe("/detectnet/vision_info", 1000, VisionCallback);

     ros::Subscriber sub2 = n.subscribe("/detectnet/detections", 1000, DetectionCallback);     

     ros::Subscriber sub3 = n.subscribe("/point_cloud_G", 1000, PointCloud2Callback);

     ROS_INFO("point cloud, waiting for messages");
   
     /**
      * ros::spin() will enter a loop, pumping callbacks.  With this version, all
      * callbacks will be called from within this thread (the main one).  ros::spin()
      * will exit when Ctrl-C is pressed, or the node is shutdown by the master.
      */
    ros::spin();
  
  return 0;
  }
