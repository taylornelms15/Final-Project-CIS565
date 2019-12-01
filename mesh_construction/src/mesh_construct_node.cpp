#include <ros/ros.h>
#include "std_msgs/String.h"
#include <sensor_msgs/Image.h>
#include <sensor_msgs/PointCloud2.h>
#include <vision_msgs/Detection2DArray.h>
#include <vision_msgs/VisionInfo.h>
#include <pcl/common/common.h>
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
#include <pcl/surface/organized_fast_mesh.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/surface/marching_cubes_hoppe.h>
#include <pcl/surface/mls.h>
#include <string.h>
#include <sstream>
#include <iomanip>
#include <iostream>
#include <fstream>

#define CGLTF_WRITE_IMPLEMENTATION 1
#define CGLTF_IMPLEMENTATION 1
#include "cgltf_write.h"

static int iteration = 0;

std::vector<std::string> class_descriptions;
std::string key;

// Forward Decl.
void ReducePointCloud(pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud);
void WriteMeshToGLTF(pcl::PolygonMesh& mesh, pcl::PointXYZRGBNormal& min, pcl::PointXYZRGBNormal& max);
void PointCloudToMesh(
	pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud,
	pcl::PolygonMesh& mesh, pcl::PointXYZRGBNormal& min, pcl::PointXYZRGBNormal& max);

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
		pcl::PointXYZRGBNormal min;
		pcl::PointXYZRGBNormal max;
		
		ROS_INFO("Reducing...");
		ReducePointCloud(cloud);
		
		ROS_INFO("Generating Mesh...");
		PointCloudToMesh(cloud, mesh, min, max);
		
		ROS_INFO("Writing to output file...");
		WriteMeshToGLTF(mesh, min, max);
		
		ROS_INFO("Done!");
		cloud->clear(); // Don't reuse points after outputting to model once.
	}
}

// Takes an input mesh and writes it to a file with an incrementing ID
void WriteMeshToGLTF(pcl::PolygonMesh& mesh, pcl::PointXYZRGBNormal& min, pcl::PointXYZRGBNormal& max)
{
	static int mesh_count = 0;
	
	std::string basename = "mesh_" + std::to_string(mesh_count);
	
	//std::string filename = "mesh_" + std::to_string(mesh_count++) + ".obj";
	//pcl::io::saveOBJFile(filename, mesh);
	// VTK visualizes better.
	std::string vtkfilename = basename + ".vtk";
	pcl::io::saveVTKFile(vtkfilename, mesh);
	
	// Generate filenames now to populate GLTF data strcutures
	std::string gltfFileName = basename + ".gltf";
	std::string binFileName = basename + ".bin";

	// Member back in C when you could just 0 initialize everything?
	cgltf_options options; // Options used for writing the file out or reading one
	cgltf_data data;       // actual data of the gltf file. This will be converted to JSON.

	// Init header data
	// https://github.com/KhronosGroup/glTF/tree/master/specification/2.0#introduction
	data.file_type = cgltf_file_type_gltf;
	data.file_data = NULL;
	data.asset.copyright = "2019";            // Who knows
	data.asset.generator = "DroneMOM";        // DroneMOM is generating this
	data.asset.version = "2.0";               // Conform to GLTF 2.0 Spec.
	data.asset.min_version = NULL;              // No min version specified, is null enough?
	data.asset.extras.start_offset = 0;       // No extras, so start and end offset is 0.
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

	///////////////////////////////////////////////////
	// Buffers
	// 0 - Indicies
	// 1 - position + normal;s
	cgltf_buffer* buffers = (cgltf_buffer*)malloc(sizeof(cgltf_buffer));
	// mesh.cloud.data -> Vector of unsigned char
	// mesh.polygons -> Vector of Vectors of 3 unsigned int
	uint32_t INDICES_COUNT = mesh.polygons.size() * 3;
	uint32_t VERTICES_COUNT = mesh.cloud.data.size();
	uint32_t INDICES_LEN = (INDICES_COUNT * sizeof(unsigned int));
	uint32_t VERTICES_LEN = (VERTICES_COUNT * sizeof(unsigned char));
	buffers[0].size = INDICES_LEN + VERTICES_LEN;
	buffers[0].uri = &binFileName[0];
	// Store the buffers in our parent
	data.buffers_count = 1;
	data.buffers = buffers;
	
	std::cout << "INDICES_COUNT: " << INDICES_COUNT << std::endl;
	std::cout << "VERTICES_COUNT: " << VERTICES_COUNT << std::endl;

	///////////////////////////////////////////////////
	// Buffer Views
	// view 0 - indices
	// view 1 - position + normals
	cgltf_buffer_view* views = (cgltf_buffer_view*)malloc(sizeof(cgltf_buffer_view) * 2);
	// Indices
	views[0].buffer = &buffers[0];
	views[0].offset = 0;
	views[0].size = INDICES_LEN;
	views[0].stride = 0;  // Must remain undefined
	views[0].type = cgltf_buffer_view_type_indices;
	// Position + Normals
	views[1].buffer = &buffers[0];
	views[1].offset = INDICES_LEN;
	views[1].size = VERTICES_LEN;
	views[1].stride = mesh.cloud.point_step;
	views[1].type = cgltf_buffer_view_type_vertices;
	// Store buffer views in parent
	data.buffer_views_count = 2;
	data.buffer_views = views;

	///////////////////////////////////////////////////
	// Accessors
	// acc 0 - indicies
	// acc 1 - position
	// acc 2 - normals
	cgltf_accessor* accessors = (cgltf_accessor*)malloc(sizeof(cgltf_accessor) * 3);
	// Indicies
	accessors[0].component_type = cgltf_component_type_r_32u;
	accessors[0].normalized = 0;
	accessors[0].type = cgltf_type_scalar;
	accessors[0].offset = 0;
	accessors[0].count = INDICES_COUNT; // TODO: Is this right?
	accessors[0].stride = 0; // Must remain undefined
	accessors[0].buffer_view = &views[0];
	accessors[0].has_min = 0;
	accessors[0].has_max = 0;
	accessors[0].is_sparse = 0;
	// Positions
	accessors[1].component_type = cgltf_component_type_r_32f;
	accessors[1].normalized = 0;
	accessors[1].type = cgltf_type_vec3;
	accessors[1].offset = 0;
	accessors[1].count = mesh.cloud.row_step / mesh.cloud.point_step; // TODO: Is this right?
	accessors[1].stride = mesh.cloud.point_step;
	accessors[1].buffer_view = &views[1];
	accessors[1].has_min = 1;
	accessors[1].min[0] = min.x;
	accessors[1].min[1] = min.y;
	accessors[1].min[2] = min.z;
	accessors[1].has_max = 1;
	accessors[1].max[0] = max.x;
	accessors[1].max[1] = max.y;
	accessors[1].max[2] = max.z;
	accessors[1].is_sparse = 0;
	// Normals
	accessors[2].component_type = cgltf_component_type_r_32f;
	accessors[2].normalized = 0;
	accessors[2].type = cgltf_type_vec3;
	accessors[2].offset = 12;
	accessors[2].count = mesh.cloud.row_step / mesh.cloud.point_step; // TODO: Is this right?
	accessors[2].stride = mesh.cloud.point_step;
	accessors[2].buffer_view = &views[1];
	accessors[2].has_min = 0;
	accessors[2].has_max = 0;
	accessors[2].is_sparse = 0;
	// Sore accessors in parent
	data.accessors_count = 3;
	data.accessors = accessors;
	
	///////////////////////////////////////////////////
	// Attributes
	// 0 - Position
	// 1 - Normals
	cgltf_attribute* attributes = (cgltf_attribute*)malloc(sizeof(cgltf_attribute) * 2);
	// Position
	attributes[0].name = "POSITION";
	attributes[0].type = cgltf_attribute_type_position;
	attributes[0].index = 1;
	attributes[0].data = &accessors[1];
	// Normals
	attributes[1].name = "NORMAL";
	attributes[1].type = cgltf_attribute_type_normal;
	attributes[1].index = 2;
	attributes[1].data = &accessors[2];

	///////////////////////////////////////////////////
	// Primitives
	// 1 - Triangles
	cgltf_primitive* prim = (cgltf_primitive*)malloc(sizeof(cgltf_primitive));
	prim[0].type = cgltf_primitive_type_triangles;
	prim[0].indices = &accessors[0];
	prim[0].material = NULL;
	prim[0].attributes = attributes;
	prim[0].attributes_count = 2;
	prim[0].targets = NULL;
	prim[0].targets_count = 0;

	///////////////////////////////////////////////////
	// Mesh
	cgltf_mesh* gltfmesh = (cgltf_mesh*)malloc(sizeof(cgltf_mesh));
	gltfmesh[0].name = "mesh";
	gltfmesh[0].primitives = prim;
	gltfmesh[0].primitives_count = 1;
	gltfmesh[0].weights = NULL;
	gltfmesh[0].weights_count = 0;
	gltfmesh[0].target_names = NULL;
	gltfmesh[0].target_names_count = 0;
	data.meshes = gltfmesh;
	data.meshes_count = 1;

	///////////////////////////////////////////////////
	// Node
	cgltf_node* node = (cgltf_node*)malloc(sizeof(cgltf_node));
	node[0].name = "node";
	node[0].parent = NULL;
	node[0].children = NULL;
	node[0].children_count = 0;
	node[0].skin = NULL;
	node[0].mesh = gltfmesh;
	node[0].camera = NULL;
	node[0].light = NULL;
	node[0].weights = NULL;
	node[0].weights_count = 0;
	node[0].has_translation = 0;
	// https://github.com/KhronosGroup/glTF/tree/master/specification/2.0#coordinate-system-and-units
	// https://www.ros.org/reps/rep-0103.html
	node[0].has_rotation = 0; // ROS to GLTF rotation.
	node[0].rotation[0] = -1.5708; // x (rads)
	node[0].rotation[1] = -1.5708; // y (rads)
	node[0].rotation[2] =  0.0; // z (rads)
	node[0].rotation[3] =  1.0; // w (scalar) (rads)
	node[0].has_scale = 0;
	node[0].has_matrix = 0;
	data.nodes = node;
	data.nodes_count = 1;
	
	///////////////////////////////////////////////////
	// Scene
	cgltf_scene* scene = (cgltf_scene*)malloc(sizeof(cgltf_scene));
	scene[0].name = "scene";
	scene[0].nodes = &node;
	scene[0].nodes_count = 1;
	data.scenes = scene;
	data.scenes_count = 1;
	data.scene = NULL; 

	///////////////////////////////////////////////////
	// Write out to file.
	// Write Binary Data
	std::ofstream binFile(binFileName);
	for(const auto& e : mesh.polygons) {
		binFile.write((char*)&e.vertices[0], e.vertices.size() * sizeof(std::uint32_t));
	}
	binFile.write((char*)&mesh.cloud.data[0], mesh.cloud.data.size() * sizeof(unsigned char));
	binFile.close();
	
	// Write GLTF File
	cgltf_result result = cgltf_write_file(&options, gltfFileName.c_str(), &data);
	if (result != cgltf_result_success) {
		ROS_INFO("Failed to write GLTF output, result was %d", result);
	}

	mesh_count++;
}

// Accepts a point cloud and constructs a surface using
// greedy projection triangles. Still need to play with some other algorithms
// to see what is most performant (plus what is best ported to GPU).
void PointCloudToMesh(
	pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud,
	pcl::PolygonMesh& mesh,
	pcl::PointXYZRGBNormal& min,
	pcl::PointXYZRGBNormal& max)
{
	static pcl::search::KdTree<pcl::PointXYZRGB>::Ptr       tree (new pcl::search::KdTree<pcl::PointXYZRGB>);
	static pcl::search::KdTree<pcl::PointXYZRGBNormal>::Ptr tree2 (new pcl::search::KdTree<pcl::PointXYZRGBNormal>);
	static pcl::PointCloud<pcl::Normal>::Ptr                normals (new pcl::PointCloud<pcl::Normal>);
	static pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr     cloud_with_normals (new pcl::PointCloud<pcl::PointXYZRGBNormal>);
	static pcl::PointCloud<pcl::PointXYZRGB>::Ptr           temp_cloud (new pcl::PointCloud<pcl::PointXYZRGB>);

	// Moving Least Squares. Used for smoothing out incoming data and filling in some holes. 
	pcl::MovingLeastSquares<pcl::PointXYZRGB, pcl::PointXYZRGB> mls;
	tree->setInputCloud(cloud);
	mls.setComputeNormals(false);
	mls.setInputCloud(cloud);
	mls.setPolynomialOrder(2);
	mls.setSearchMethod(tree);
	mls.setSearchRadius(0.03);
	mls.process(*temp_cloud);
	
	// Normal Estimation
	pcl::NormalEstimation<pcl::PointXYZRGB, pcl::Normal> ne;
	ne.setInputCloud(temp_cloud);
	tree->setInputCloud(temp_cloud);
	ne.setSearchMethod(tree);
	ne.setRadiusSearch(0.03);
	ne.compute(*normals);
	
	// Combine into one cloud
	pcl::concatenateFields(*temp_cloud, *normals, *cloud_with_normals);

	// Initialize Reconstruction Algo
	pcl::GreedyProjectionTriangulation<pcl::PointXYZRGBNormal> gp3;
	gp3.setSearchRadius(0.03);
	gp3.setMu(2.5);
	gp3.setMaximumNearestNeighbors(100);
	gp3.setMaximumSurfaceAngle(M_PI/4);   // 45 degrees
	gp3.setMinimumAngle(M_PI/18);         // 10 degrees
	gp3.setMaximumAngle(2*M_PI/3);        // 120 degrees
	gp3.setNormalConsistency(true);
	gp3.setInputCloud(cloud_with_normals);
	tree2->setInputCloud(cloud_with_normals);
	gp3.setSearchMethod(tree2);

	// Construct Mesh
	gp3.reconstruct(mesh);
	
	// Get min/max, needed for gltf
	pcl::getMinMax3D(*cloud_with_normals, min, max);
}

// Filter out unimportant points in the cloud.
void ReducePointCloud(pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud)
{
	// Remove points that produce no meaninful geometries
	// Can't remove all of them, but this should help with most.
	std::cout << "ROR before: " << cloud->points.size() << std::endl;
	pcl::RadiusOutlierRemoval<pcl::PointXYZRGB> ror;
	ror.setRadiusSearch(0.01f);
	ror.setMinNeighborsInRadius(20);
	ror.setInputCloud(cloud);
	ror.filter(*cloud);
	std::cout << "ROR after: " << cloud->points.size() << std::endl;
	
	// Reduce points based on features
	std::cout << "US before: " << cloud->points.size() << std::endl;
	pcl::UniformSampling<pcl::PointXYZRGB> us;
	us.setRadiusSearch(0.01f); // Experiment or set to 1/1000 of min/max
	us.setInputCloud(cloud);
	us.filter(*cloud);
	std::cout << "US after: " << cloud->points.size() << std::endl;
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
