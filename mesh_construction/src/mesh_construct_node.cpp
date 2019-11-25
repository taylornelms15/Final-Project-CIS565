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
#include <pcl/surface/poisson.h>
#include <pcl/io/obj_io.h>
#include <pcl/filters/uniform_sampling.h>
#include <string.h>
#include <sstream>
#include <iomanip>


// Define these only in *one* .cc file.
#define TINYGLTF_IMPLEMENTATION
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
// #define TINYGLTF_NOEXCEPTION // optional. disable exception handling.
#include "tiny_gltf.h"

using namespace tinygltf;

static int iteration = 0;

static pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud_with_normals = 0;
static pcl::PointCloud<pcl::PointXYZRGB>::Ptr incoming_cloud = 0;
static pcl::PointCloud<pcl::Normal>::Ptr normals = 0;
static pcl::PCLPointCloud2* pcl_to_ros_cloud = 0;
static pcl::search::KdTree<pcl::PointXYZRGBNormal>::Ptr kdtree = 0;

std::vector<std::string> class_descriptions;
std::string key;

// Forward Decl.
void ReducePointCloud(pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud);
void WriteMeshToGLTF(pcl::PolygonMesh& mesh);
void PointCloudToMesh(
	pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud,
	pcl::PolygonMesh& mesh);

// this is registered every image sent
void PointCloud2Callback(const sensor_msgs::PointCloud2& msg)
{
	ROS_INFO("Got message!");

	// Init static variables
	static pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
	static pcl::PointCloud<pcl::PointXYZRGB>::Ptr tmp_cloud(new pcl::PointCloud<pcl::PointXYZRGB>);

	//
	// Example code below from http://www.pointclouds.org/documentation/tutorials/greedy_projection.php
	//

	// Conert to proper format. This destroys the contents of the msg object!
	pcl::fromROSMsg(msg, *tmp_cloud);
	*cloud += *tmp_cloud;

	// Write out every 20 cycles to reduce load.
	iteration++;
	if((iteration % 20) == 0) {
		ROS_INFO("generating output!!");
		pcl::PolygonMesh mesh;
		ReducePointCloud(cloud);
		PointCloudToMesh(cloud, mesh);
		WriteMeshToGLTF(mesh);
		cloud->clear(); // Don't reuse points after outputting to model once.
	}
}

// Takes an input mesh and writes it to a file with an incrementing ID
void WriteMeshToGLTF(pcl::PolygonMesh& mesh)
{
	static int mesh_count = 0;
	std::string filename = "mesh_" + std::to_string(mesh_count++) + ".obj";
	pcl::io::saveOBJFile(filename, mesh);
}

// Accepts a point cloud and constructs a surface using
// greedy projection triangles. Still need to play with some other algorithms
// to see what is most performant (plus what is best ported to GPU).
void PointCloudToMesh(
	pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud,
	pcl::PolygonMesh& mesh)
{
	static pcl::search::KdTree<pcl::PointXYZRGB>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZRGB>);
	static pcl::search::KdTree<pcl::PointXYZRGBNormal>::Ptr tree2 (new pcl::search::KdTree<pcl::PointXYZRGBNormal>);
	static pcl::PointCloud<pcl::Normal>::Ptr normals (new pcl::PointCloud<pcl::Normal>);
	static pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud_with_normals (new pcl::PointCloud<pcl::PointXYZRGBNormal>);

	// Normal estimation
	pcl::NormalEstimation<pcl::PointXYZRGB, pcl::Normal> n;
	tree->setInputCloud(cloud);
	n.setInputCloud(cloud);
	n.setSearchMethod(tree);
	n.setKSearch(20);
	n.compute(*normals);

	// Concatenate the XYZ and normal fields
	pcl::concatenateFields(*cloud, *normals, *cloud_with_normals);

	// Create search tree
	tree2->setInputCloud(cloud_with_normals);

	// Initialize Reconstruction Algo
	pcl::GreedyProjectionTriangulation<pcl::PointXYZRGBNormal> gp3;

	// Set typical values for the parameters
	gp3.setSearchRadius(0.025);
	gp3.setMu(2.5);
	gp3.setMaximumNearestNeighbors (100);
	gp3.setMaximumSurfaceAngle(M_PI/4);   // 45 degrees
	gp3.setMinimumAngle(M_PI/18);         // 10 degrees
	gp3.setMaximumAngle(2*M_PI/3);        // 120 degrees
	gp3.setNormalConsistency(false);

	// Get result
	gp3.setInputCloud(cloud_with_normals);
	gp3.setSearchMethod(tree2);
	gp3.reconstruct(mesh);
}

// Filter out unimportant points in the cloud.
void ReducePointCloud(pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud)
{
	// Plan: On GPU, sort by x, y, then z.
	// Then filter out points that are identical within +- some range

	// CPU Impl.
	pcl::UniformSampling<pcl::PointXYZRGB> us;
	us.setRadiusSearch(0.01f); // Experiment or set to 1/1000 of min/max
	us.setInputCloud(cloud);
	us.filter(*cloud);

	// For GPU imple, take UniformSampling code and parallelize by Voxel.
	// Straightforward but should provide big improvement.
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
	// Data from rosbag used for testing
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
