

#include <ros/ros.h>
#include <stdio.h>
/*
* ROS messages
*/
#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>

#include <sensor_msgs/Imu.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/image_encodings.h>
#include <sensor_msgs/CameraInfo.h>

#include <vision_msgs/Detection2DArray.h>
#include <vision_msgs/VisionInfo.h>

#include <geometry_msgs/TransformStamped.h>
#include <drone_mom_msgs/drone_mom.h>
/*
*
*/ 
#include "../utils/img_write.h"
#include "../inference/ObjectDetection.h"
#include "../cuda_utilities/cudaMappedMemory.h"
#include "../utils/profiler.h"

#include "image_converter.h"

#include <unordered_map>

using namespace message_filters;


// globals
ObjectDetection* 	 net = NULL;
imageConverter* cvt = NULL;

ros::Publisher* detection_pub = NULL;

vision_msgs::VisionInfo info_msg;

//
typedef sync_policies::ApproximateTime<sensor_msgs::Image,sensor_msgs::CameraInfo,sensor_msgs::Image,sensor_msgs::CameraInfo,geometry_msgs::TransformStamped,geometry_msgs::TransformStamped,geometry_msgs::TransformStamped> MySyncPolicy;


// callback triggered when a new subscriber connected to vision_info topic
void info_connect( const ros::SingleSubscriberPublisher& pub )
{
	ROS_INFO("new subscriber '%s' connected to vision_info topic '%s', sending VisionInfo msg", pub.getSubscriberName().c_str(), pub.getTopic().c_str());
	pub.publish(info_msg);
}


// input image subscriber callback
void img_callback( const sensor_msgs::ImageConstPtr& rgb_raw_image,const sensor_msgs::CameraInfo::ConstPtr& rgb_camera_info,const sensor_msgs::ImageConstPtr& depth_raw_image,const sensor_msgs::CameraInfo::ConstPtr& depth_camera_info,const geometry_msgs::TransformStamped::ConstPtr& tic_color,const geometry_msgs::TransformStamped::ConstPtr& tgi,const geometry_msgs::TransformStamped::ConstPtr& tic_depth )
// void img_callback( const sensor_msgs::ImageConstPtr& input)
{
	// convert the image to reside on GPU
	if( !cvt || !cvt->Convert(rgb_raw_image) )
	{
		ROS_INFO("failed to convert %ux%u %s image", rgb_raw_image->width, rgb_raw_image->height, rgb_raw_image->encoding.c_str());
		return;	
	}

	// classify the image
	ObjectDetection::Detection* detections = NULL;

	//
	// profiler_begin(DETECT_BEGIN);

	//
	const int numDetections = net->Detect(cvt->ImageGPU(), cvt->GetWidth(), cvt->GetHeight(), &detections, ObjectDetection::OVERLAY_BOX);

	//
	// profiler_end(DETECT_END);
	
	// verify success	
	if( numDetections < 0 )
	{
		ROS_ERROR("failed to run object detection on %ux%u image", rgb_raw_image->width, rgb_raw_image->height);
		return;
	}

	// if objects were detected, send out message
	if( numDetections >= 0 )
	{
		//ROS_INFO("detected %i objects in %ux%u image", numDetections, input->width, input->height);
		
		//
		drone_mom_msgs::drone_mom msg;

		msg.TIC_color = *tic_color;
		msg.TIC_depth = *tic_depth;
		msg.TGI = *tgi;
		msg.depth_camera_info = *depth_camera_info;
		msg.rgb_camera_info = *rgb_camera_info;
		msg.rgb_image = *rgb_raw_image;
		msg.depth_image = *depth_raw_image;
		

		for( int n=0; n < numDetections; n++ )
		{
			ObjectDetection::Detection* det = detections + n;

			ROS_INFO("object %i class #%u (%s)  confidence=%f", n, det->ClassID, net->GetClassDesc(det->ClassID), det->Confidence);
			ROS_INFO("object %i bounding box (%f, %f)  (%f, %f)  w=%f  h=%f", n, det->Left, det->Top, det->Right, det->Bottom, det->Width(), det->Height()); 
			
			// create a detection sub-message
			vision_msgs::Detection2D detMsg;

			detMsg.bbox.size_x = det->Width();
			detMsg.bbox.size_y = det->Height();
			
			float cx, cy;
			det->Center(&cx, &cy);

			detMsg.bbox.center.x = cx;
			detMsg.bbox.center.y = cy;

            ROS_INFO("object %i centers (%f, %f)", n, cx, cy);

			detMsg.bbox.center.theta = 0.0f;		// TODO optionally output object image

			// create classification hypothesis
			vision_msgs::ObjectHypothesisWithPose hyp;
			
			hyp.id = det->ClassID;
			hyp.score = det->Confidence;
			if(n==0){
			// move this to something else no need to copy the same thing
			cvt->Convert(detMsg.source_img,sensor_msgs::image_encodings::BGR8);
			}
			detMsg.results.push_back(hyp);
			msg.classification.detections.push_back(detMsg);
		}

		detection_pub->publish(msg);
	}

}



// node main loop
int main(int argc, char **argv)
{
	ros::init(argc, argv, "detectnet");
 
	ros::NodeHandle private_nh("~");

	std::string model_name;

	//
	create_events();
		
	// default parameter is to use the ssd mobilenet
	private_nh.param<std::string>("model_name", model_name, "ssd-mobilenet-v2");

	// set mean pixel and threshold defaults
	float mean_pixel = 0.0f;
	float threshold  = 0.5f;
	
	//
	private_nh.param<float>("mean_pixel_value", mean_pixel, mean_pixel);
	private_nh.param<float>("threshold", threshold, threshold);

	//
	ObjectDetection::NetworkType model = ObjectDetection::NetworkTypeFromStr(model_name.c_str());

	// 
	if( model == ObjectDetection::ERROR )
	{
		ROS_ERROR("unknown model\n");
		return 0;
	}

	// create network
	net = ObjectDetection::Create(model);

	if( !net )
	{
		ROS_ERROR("failed to load model exiting!\n");
		return 0;
	}


	/*
	 * create the class labels parameter vector
	 */
	std::hash<std::string> model_hasher;  // hash the model path to avoid collisions on the param server
	std::string model_hash_str = std::string(net->GetModelPath()) + std::string(net->GetClassPath());
	const size_t model_hash = model_hasher(model_hash_str);
	
	ROS_INFO("model hash => %zu", model_hash);
	ROS_INFO("hash string => %s", model_hash_str.c_str());

	// obtain the list of class descriptions
	std::vector<std::string> class_descriptions;
	const uint32_t num_classes = net->GetNumClasses();

	for( uint32_t n=0; n < num_classes; n++ )
		class_descriptions.push_back(net->GetClassDesc(n));

	// create the key on the param server
	std::string class_key = std::string("class_labels_") + std::to_string(model_hash);
	private_nh.setParam(class_key, class_descriptions);
		
	// 
	std::string node_namespace = private_nh.getNamespace();
	ROS_INFO("node namespace => %s", node_namespace.c_str());

	//
	info_msg.database_location = node_namespace + std::string("/") + class_key;
	info_msg.database_version  = 0;
	info_msg.method 		  = net->GetModelPath();
	
	//
	ROS_INFO("class labels => %s", info_msg.database_location.c_str());


	/*
	 * create an image converter object
	 */
	cvt = new imageConverter();
	
	if( !cvt )
	{
		ROS_ERROR("failed to create imageConverter object");
		return 0;
	}


	/*
	 * advertise publisher topics
	 */
	ros::Publisher pub = private_nh.advertise<drone_mom_msgs::drone_mom>("detections", 1000);
	detection_pub = &pub; // we need to publish from the subscriber callback

	// the vision info topic only publishes upon a new connection
	// subscribing to this topic gives access to the classification server
	ros::Publisher info_pub = private_nh.advertise<vision_msgs::VisionInfo>("vision_info", 1, (ros::SubscriberStatusCallback)info_connect);


	/*
	 * subscribe to image topic
	 */
	 // ros::Subscriber img_sub = private_nh.subscribe("/image_publisher/image_raw", 5, img_callback);
	
	//
	message_filters::Subscriber<sensor_msgs::Image> image_rgb_raw_sub(private_nh, "/camera/rgb/image_raw", 1000);

	//
	message_filters::Subscriber<sensor_msgs::CameraInfo> cam_rgb_sub(private_nh, "/camera/rgb/camera_info", 1000);

	//
	message_filters::Subscriber<geometry_msgs::TransformStamped> tic_color_sub(private_nh, "/tango/T_I_C_color", 1000);

	//
	message_filters::Subscriber<geometry_msgs::TransformStamped> tgi_sub(private_nh, "/tango_viwls/T_G_I", 1000);

	//
	message_filters::Subscriber<geometry_msgs::TransformStamped> tic_depth_sub(private_nh, "/tango/T_I_C_depth", 1000);

	//
	message_filters::Subscriber<sensor_msgs::CameraInfo> cam_depth_sub(private_nh, "/camera/depth/camera_info", 1000);	

	//
	message_filters::Subscriber<sensor_msgs::Image> image_depth_raw_sub(private_nh, "/camera/depth/image_raw", 1000);	
  	
  	//
	Synchronizer<MySyncPolicy> sync(MySyncPolicy(1000), image_rgb_raw_sub, cam_rgb_sub,image_depth_raw_sub,cam_depth_sub,tic_color_sub,tgi_sub,tic_depth_sub);
  	
  	//
	sync.registerCallback(boost::bind(&img_callback, _1, _2, _3, _4, _5, _6, _7));

	/*
	 * wait for messages
	 */
	ROS_INFO("detectnet node initialized, waiting for messages");

	ros::spin();

	return 0;
}

