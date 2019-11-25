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
#include <pcl/io/vtk_io.h>
#include <pcl/filters/uniform_sampling.h>
#include <pcl/filters/radius_outlier_removal.h>
#include <pcl/surface/mls.h>
#include <string.h>
#include <sstream>
#include <iomanip>

#define CGLTF_WRITE_IMPLEMENTATION 1
#include "cgltf_write.h"

static int iteration = 0;

//static pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr     cloud_with_normals = 0;
//static pcl::PointCloud<pcl::PointXYZRGB>::Ptr           incoming_cloud = 0;
//static pcl::PointCloud<pcl::Normal>::Ptr                normals = 0;

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

	// Convert to proper format. This destroys the contents of the msg object!
	pcl::fromROSMsg(msg, *tmp_cloud);
	*cloud += *tmp_cloud;

	// Write out every 20 cycles to reduce load.
	iteration++;
	if((iteration % 20) == 0) {
		ROS_INFO("generating output!!");
		pcl::PolygonMesh mesh;
		ROS_INFO("Reducing...");
		ReducePointCloud(cloud);
		ROS_INFO("Generating Mesh...");
		PointCloudToMesh(cloud, mesh);
		ROS_INFO("Writing to output file...");
		WriteMeshToGLTF(mesh);
		cloud->clear(); // Don't reuse points after outputting to model once.
		ROS_INFO("Done!");
	}
}

// Takes an input mesh and writes it to a file with an incrementing ID
void WriteMeshToGLTF(pcl::PolygonMesh& mesh)
{
	static int mesh_count = 0;
	//std::string filename = "mesh_" + std::to_string(mesh_count++) + ".obj";
	//pcl::io::saveOBJFile(filename, mesh);
	// VTK visualizes better.
	std::string filename = "mesh_" + std::to_string(mesh_count) + ".vtk";
	pcl::io::saveVTKFile(filename, mesh);

	// Member back in C when you could just 0 initialize everything?
	cgltf_options options; // Options used for writing the file out or reading one
	cgltf_data data;       // actual data of the gltf file. This will be converted to JSON.

	// Init header data
	data.file_type = cgltf_file_type_gltf;
	data.file_data = NULL;
	data.asset.copyright = "";
	data.asset.generator = "";
	data.asset.version = "";
	data.asset.min_version = "";
	data.asset.extras.start_offset = 0;
	data.asset.extras.end_offset = 0;

	// Init all counts to 0
	data.meshes_count = 0;
	data.materials_count = 0;
	data.accessors_count = 0;
	data.buffer_views_count = 0;
	data.buffers_count = 0;
	data.images_count = 0;
	data.textures_count = 0;
	data.samplers_count = 0;
	data.skins_count = 0;
	data.cameras_count = 0;
	data.lights_count = 0;
	data.nodes_count = 0;
	data.scenes_count = 0;
	data.animations_count = 0;
	data.extensions_used_count = 0;
	data.extensions_required_count = 0;
	data.json_size = 0;
	data.bin_size = 0;
	data.memory_free = NULL;
	data.memory_user_data = NULL;

	// Add Meshes

	// Add verticies
	
	// Populate Vericies buffers as base64 encoded string

	// Add whatever else GLTF will need.

	// Write out to file.
	std::string gltffilename = "mesh_" + std::to_string(mesh_count) + ".gltf";
	cgltf_result result = cgltf_write_file(&options, gltffilename.c_str(), &data);
	if (result != cgltf_result_success)
	{
		ROS_INFO("Failed to write GLTF output, result was %d", result);
	}

	mesh_count++;
}

// Accepts a point cloud and constructs a surface using
// greedy projection triangles. Still need to play with some other algorithms
// to see what is most performant (plus what is best ported to GPU).
void PointCloudToMesh(
	pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud,
	pcl::PolygonMesh& mesh)
{
	static pcl::search::KdTree<pcl::PointXYZRGB>::Ptr       tree (new pcl::search::KdTree<pcl::PointXYZRGB>);
	static pcl::search::KdTree<pcl::PointXYZRGBNormal>::Ptr tree2 (new pcl::search::KdTree<pcl::PointXYZRGBNormal>);
	static pcl::PointCloud<pcl::Normal>::Ptr                normals (new pcl::PointCloud<pcl::Normal>);
	static pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr     cloud_with_normals (new pcl::PointCloud<pcl::PointXYZRGBNormal>);

	// Moving Least Squares. Used for smoothing out incoming data and filling in some holes. 
	pcl::MovingLeastSquares<pcl::PointXYZRGB, pcl::PointXYZRGBNormal> mls;
	mls.setComputeNormals(true);
	mls.setInputCloud (cloud);
	mls.setPolynomialOrder (2);
	mls.setSearchMethod (tree);
	mls.setSearchRadius (0.1);
	mls.process(*cloud_with_normals);

	// Create search tree
	tree2->setInputCloud(cloud_with_normals);

	// Initialize Reconstruction Algo
	pcl::GreedyProjectionTriangulation<pcl::PointXYZRGBNormal> gp3;

	// Set typical values for the parameters
	gp3.setSearchRadius(0.025f);
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
	// Reduce points based on features
	pcl::UniformSampling<pcl::PointXYZRGB> us;
	us.setRadiusSearch(0.01f); // Experiment or set to 1/1000 of min/max
	us.setInputCloud(cloud);
	us.filter(*cloud);

	// Remove points that produce no meaninful geometries
	// Can't remove all of them, but this should help with most.
	pcl::RadiusOutlierRemoval<pcl::PointXYZRGB> ror;
	ror.setRadiusSearch(2 * 0.025f);
	ror.setMinNeighborsInRadius(2);
	ror.setInputCloud(cloud);
	ror.filter(*cloud);

	// For GPU impl, take UniformSampling code and parallelize by Voxel.
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
