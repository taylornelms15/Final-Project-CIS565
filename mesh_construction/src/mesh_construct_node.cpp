#include <ros/ros.h>
#include "std_msgs/String.h"
#include <sensor_msgs/Image.h>
#include <vision_msgs/Detection2DArray.h>
#include <vision_msgs/VisionInfo.h>

#include <cv_bridge/cv_bridge.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>

#include <sensor_msgs/Imu.h>
#include <sensor_msgs/Image.h>
#include <drone_mom_msgs/drone_mom.h>
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

   void DetectionCallback(const drone_mom_msgs::drone_mom::ConstPtr msg1)
   {
      // print stuff to show how stuff works
     // Vision messages can be found here http://docs.ros.org/melodic/api/vision_msgs/html/msg/Detection2DArray.html
     // int detected_elements = msg1->classification.detections.size();
     vision_msgs::Detection2DArray msg = msg1->classification;
 //     for(int i = 0; i < detected_elements; i++)
 //     {
 //        ROS_INFO("bbox X: %9.6f", msg.detections[i].bbox.size_x);
 //        ROS_INFO("bbox cx: %9.6f", msg.detections[i].bbox.center.x);
 //        ROS_INFO("bbox Y: %9.6f", msg.detections[i].bbox.size_y);
 //        ROS_INFO("bbox cy: %9.6f", msg.detections[i].bbox.center.y);

 //        // there is only 1 result per detection I am 99% sure
 //        ROS_INFO("confidece: %1.6f", msg.detections[i].results[0].score);     
     
 //        int idx = msg.detections[i].results[0].id;


 //        // since we got the database from setup we can now see what our bounding box contains!
 //        ROS_INFO("classified: %s", class_descriptions[idx].c_str());
	// }
	cv_bridge::CvImagePtr cv_ptr;
  try
  {
    cv_ptr = cv_bridge::toCvCopy(msg.detections[0].source_img, sensor_msgs::image_encodings::RGB8);
  }
  catch (cv_bridge::Exception& e)
  {
    ROS_ERROR("cv_bridge exception: %s", e.what());
    return;
  }
  cv::imshow("Image window", cv_ptr->image);
  cv::waitKey(1);  

	// ROS_INFO("===height %d===",msg1->cam_info.height);
	// ROS_INFO("===width %d===",msg1->cam_info.width);

	// int time_secs = msg1->imu_msg.header.stamp.sec;
 //    int time_nsecs = msg1->imu_msg.header.stamp.nsec;
 //    double timeValue = time_secs + (1e-9 * time_nsecs);
	// ROS_INFO("===IMU %s===",msg1->imu_msg.header.frame_id.c_str());
 //    ROS_INFO("\ttime: %9.6f", timeValue);
   
	// int time2_secs = msg1->raw_image.header.stamp.sec;
 //    int time2_nsecs =msg1->raw_image.header.stamp.nsec;
 //    double timeValue2 = time2_secs + (1e-9 * time2_nsecs);
	// ROS_INFO("===IMAGE %s===", msg1->raw_image.header.frame_id.c_str());
 //    ROS_INFO("\ttime: %9.6f", timeValue2);
    
 //    int time3_secs = msg1->cam_info.header.stamp.sec;
 //    int time3_nsecs = msg1->cam_info.header.stamp.nsec;
 //    double timeValue3 = time3_secs + (1e-9 * time3_nsecs);
	// ROS_INFO("===CAMERA %s===", msg1->cam_info.header.frame_id.c_str());
 //    ROS_INFO("\ttime: %9.6f", timeValue3);
	
	// int time4_secs = msg1->TIC.header.stamp.sec;
 //    int time4_nsecs = msg1->TIC.header.stamp.nsec;
 //    double timeValue4 = time4_secs + (1e-9 * time4_nsecs);
	// ROS_INFO("===TIC %s===", msg1->TIC.header.frame_id.c_str());
 //    ROS_INFO("\ttime: %9.6f", timeValue4);

 //    int time5_secs = msg1->TGI.header.stamp.sec;
 //    int time5_nsecs = msg1->TGI.header.stamp.nsec;
 //    double timeValue5 = time5_secs + (1e-9 * time5_nsecs);
	// ROS_INFO("===TGI %s===", msg1->TGI.header.frame_id.c_str());
 //    ROS_INFO("\ttime: %9.6f", timeValue5);

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

     ROS_INFO("mesh, waiting for messages");
   
     /**
      * ros::spin() will enter a loop, pumping callbacks.  With this version, all
      * callbacks will be called from within this thread (the main one).  ros::spin()
      * will exit when Ctrl-C is pressed, or the node is shutdown by the master.
      */
    ros::spin();
  
  return 0;
  }
