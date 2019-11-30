/*
 * Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

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
//#include <drone_mom_msgs/drone_mom>


/*
*
*/ 
#include "../nvidia_files/img_write.h"
#include "../inference/ObjectDetection.h"
#include "../cuda_utilities/cudaMappedMemory.h"

#include "image_converter.h"

#include <unordered_map>

using namespace message_filters;


// globals
ObjectDetection* 	 net = NULL;
imageConverter* cvt = NULL;

ros::Publisher* detection_pub = NULL;

vision_msgs::VisionInfo info_msg;


// callback triggered when a new subscriber connected to vision_info topic
void info_connect( const ros::SingleSubscriberPublisher& pub )
{
	ROS_INFO("new subscriber '%s' connected to vision_info topic '%s', sending VisionInfo msg", pub.getSubscriberName().c_str(), pub.getTopic().c_str());
	pub.publish(info_msg);
}


// input image subscriber callback
void img_callback( const sensor_msgs::ImageConstPtr& input, const sensor_msgs::Imu::ConstPtr& imu, const sensor_msgs::CameraInfo::ConstPtr& cam,const geometry_msgs::TransformStamped::ConstPtr& tic,const geometry_msgs::TransformStamped::ConstPtr& tgi  )
{
	// convert the image to reside on GPU
	if( !cvt || !cvt->Convert(input) )
	{
		ROS_INFO("failed to convert %ux%u %s image", input->width, input->height, input->encoding.c_str());
		return;	
	}

	// classify the image
	ObjectDetection::Detection* detections = NULL;

	const int numDetections = net->Detect(cvt->ImageGPU(), cvt->GetWidth(), cvt->GetHeight(), &detections, ObjectDetection::OVERLAY_BOX);

	// verify success	
	if( numDetections < 0 )
	{
		ROS_ERROR("failed to run object detection on %ux%u image", input->width, input->height);
		return;
	}

	// if objects were detected, send out message
	if( numDetections > 0 )
	{
		ROS_INFO("detected %i objects in %ux%u image", numDetections, input->width, input->height);
		
		//
		drone_mom_msgs::drone_mom msg;
		
		// create a detection for each bounding box
		//vision_msgs::Detection2DArray msg;

		for( int n=0; n < numDetections; n++ )
		{
			ObjectDetection::Detection* det = detections + n;

			// printf("object %i class #%u (%s)  confidence=%f\n", n, det->ClassID, net->GetClassDesc(det->ClassID), det->Confidence);
			// printf("object %i bounding box (%f, %f)  (%f, %f)  w=%f  h=%f\n", n, det->Left, det->Top, det->Right, det->Bottom, det->Width(), det->Height()); 
			
			// create a detection sub-message
			vision_msgs::Detection2D detMsg;

			detMsg.bbox.size_x = det->Width();
			detMsg.bbox.size_y = det->Height();
			
			float cx, cy;
			det->Center(&cx, &cy);

			detMsg.bbox.center.x = cx;
			detMsg.bbox.center.y = cy;

			detMsg.bbox.center.theta = 0.0f;		// TODO optionally output object image

			// create classification hypothesis
			vision_msgs::ObjectHypothesisWithPose hyp;
			
			hyp.id = det->ClassID;
			hyp.score = det->Confidence;

			// move this to something else no need to copy the same thing
			cvt->Convert(detMsg.source_img,sensor_msgs::image_encodings::BGR8);

			detMsg.results.push_back(hyp);
			msg.classification.detections.push_back(detMsg);
		}

		if( numDetections > 3 )
		{
			if( !saveImageRGBA("file.jpg", (float4*)cvt->ImageGPU(), cvt->GetWidth(), cvt->GetHeight(), 255.0f) )
				printf("failed saving %ix%i image to 'file'\n", cvt->GetWidth(), cvt->GetHeight());
			else	
				printf("successfully wrote %ix%i image to 'file'\n", cvt->GetWidth(), cvt->GetHeight());
		}
		// publish the detection message
		detection_pub->publish(msg);
	}

	int time_secs = imu->header.stamp.sec;
    int time_nsecs = imu->header.stamp.nsec;
    double timeValue = time_secs + (1e-9 * time_nsecs);
	ROS_INFO("===IMU %s===", imu->header.frame_id.c_str());
    ROS_INFO("\ttime: %9.6f", timeValue);
   
	int time2_secs = input->header.stamp.sec;
    int time2_nsecs = input->header.stamp.nsec;
    double timeValue2 = time2_secs + (1e-9 * time2_nsecs);
	ROS_INFO("===IMAGE %s===", input->header.frame_id.c_str());
    ROS_INFO("\ttime: %9.6f", timeValue2);
    
    int time3_secs = cam->header.stamp.sec;
    int time3_nsecs = cam->header.stamp.nsec;
    double timeValue3 = time3_secs + (1e-9 * time3_nsecs);
	ROS_INFO("===CAMERA %s===", cam->header.frame_id.c_str());
    ROS_INFO("\ttime: %9.6f", timeValue3);
	
	int time4_secs = tic->header.stamp.sec;
    int time4_nsecs = tic->header.stamp.nsec;
    double timeValue4 = time4_secs + (1e-9 * time4_nsecs);
	ROS_INFO("===TIC %s===", tic->header.frame_id.c_str());
    ROS_INFO("\ttime: %9.6f", timeValue4);

    int time5_secs = tgi->header.stamp.sec;
    int time5_nsecs = tgi->header.stamp.nsec;
    double timeValue5 = time5_secs + (1e-9 * time5_nsecs);
	ROS_INFO("===TGI %s===", tgi->header.frame_id.c_str());
    ROS_INFO("\ttime: %9.6f", timeValue5);

}



// node main loop
int main(int argc, char **argv)
{
	ros::init(argc, argv, "detectnet");
 
	ros::NodeHandle nh;
	ros::NodeHandle private_nh("~");

	// std::string class_labels_path;
	// std::string prototxt_path;
	// std::string model_path;
	std::string model_name;


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
	ros::Publisher pub = private_nh.advertise<drone_mom_msgs::drone_mom>("detections", 25);
	detection_pub = &pub; // we need to publish from the subscriber callback

	// the vision info topic only publishes upon a new connection
	ros::Publisher info_pub = private_nh.advertise<vision_msgs::VisionInfo>("vision_info", 1, (ros::SubscriberStatusCallback)info_connect);


	/*
	 * subscribe to image topic
	 */
	//ros::Subscriber img_sub = private_nh.subscribe("/cam0/image_raw", 5, img_callback);
//	 ros::Subscriber img_sub = private_nh.subscribe("/image_publisher/image_raw", 5, img_callback);
	// ros::Subscriber imu_sub = private_nh.subscribe("/imu0", 5, imu_callback);
	
	//
	message_filters::Subscriber<sensor_msgs::Image> image_sub(private_nh, "/camera/rgb/image_raw", 50);
  	
  	//
	message_filters::Subscriber<sensor_msgs::Imu> imu_sub(private_nh, "/imu0", 50);

	//
	message_filters::Subscriber<geometry_msgs::TransformStamped> tic_sub(private_nh, "/tango/T_I_C_color", 50);

	//
	message_filters::Subscriber<geometry_msgs::TransformStamped> tgi_sub(private_nh, "/tango_viwls/T_G_I", 50);

	//
	message_filters::Subscriber<sensor_msgs::CameraInfo> cam_sub(private_nh, "/camera/rgb/camera_info", 50);
  	
  	//
	typedef sync_policies::ApproximateTime<sensor_msgs::Image, sensor_msgs::Imu,sensor_msgs::CameraInfo,geometry_msgs::TransformStamped,geometry_msgs::TransformStamped> MySyncPolicy;
  	
  	//
	Synchronizer<MySyncPolicy> sync(MySyncPolicy(10), image_sub, imu_sub,cam_sub,tic_sub,tgi_sub);
  	
  	//
	sync.registerCallback(boost::bind(&img_callback, _1, _2, _3, _4, _5));

	/*
	 * wait for messages
	 */
	ROS_INFO("detectnet node initialized, waiting for messages");

	ros::spin();

	return 0;
}

