#include <sensor_msgs/Image.h>
#include <sensor_msgs/Imu.h>
#include <geometry_msgs/PointStamped.h>
#include <geometry_msgs/TransformStamped.h>
#include <vision_msgs/Detection2DArray.h>
#include <vision_msgs/VisionInfo.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
//#include <tf2/LinearMath/Quaternion.h>
#include <tf2/LinearMath/Vector3.h>
#include <tf2/LinearMath/Transform.h>

#include "point_cloud_node.h"

#define DEBUGOUT 1

#define DEBUGIMAGE (DEBUGOUT && 1)
#define DEBUGXFORM (DEBUGOUT && 1)


using namespace cv;

    const static char imageSubPath[] = "color_image";
    const static char xformSubPath1[] = "T_G_C";//tranformation for color camera
    const static char xformSubPath2[] = "T_G_D";//transformation for depth camera
    
    static int imageNumber = 0;
    static double                       xformTime = -1;
    static double                       lastXformTime = -1;
    static tf2::Transform               xform;
    static pcl::PointCloud<PointT> pcloud = pcl::PointCloud<PointT>(); 
    static float                        FoV = 45.0;//making up a field of view?


    void ImageCallback(const sensor_msgs::Image& msg){
        if (lastXformTime == xformTime) return;//don't take a new image without new positional data
        imageNumber++;

        //for at least one dataset, all the data is bgr format, which opencv likes
        int width = msg.width;
        int height = msg.height;


        Mat thisImage = Mat(height, width, getEncodingTypeForCV(msg.encoding), (void*) msg.data.data());

#if DEBUGIMAGE
        ROS_INFO("===IMAGE===");
        ROS_INFO("\ttime of last xform: %f", xformTime);
#endif

        //Every 16 frames, put some more points in the cloud  
        if (imageNumber % 16 == 0){
            float zNDC = 0.5f;//fake depth because we're not working with the real stuff yet
            tf2::Vector3 viewPosition = tf2::Vector3(0.0f, 0.0f, 0.0f);
            tf2::Vector3 transformedPosition = viewPosition + xform.getOrigin();
            for(int j = 0; j < height; j++){
                if (j % 2 == 0) continue;//less-dense output
                float yNDC = (j / (height - 1.0f)) * 2.0f - 1.0f;
                for(int i = 0; i < width; i++){
                    if (i % 2 == 0) continue;//less-dense output
                    //get a 3d point for our pixels (assuming depth = 1)
                    //scale position to between -1 and 1
                    float xNDC = (i / (width - 1.0f)) * 2.0f - 1.0f;
                    tf2::Vector3 viewDirection = tf2::Vector3(xNDC, yNDC, zNDC);
                    tf2::Vector3 transformedDirection = xform(viewDirection);

                    tf2::Vector3 finalPoint = transformedPosition + 1.0 * transformedDirection;

                    Vec3b bgr = thisImage.at<Vec3b>(j, i);
                    int label = 0;
                    PointT point = PointT(bgr[2], bgr[1], bgr[0], label);
                    point.x = finalPoint[0]; point.y = finalPoint[1]; point.z = finalPoint[2];
                    pcloud.push_back(point);


                }//for col
            }//for row
        }//if


        //imwrite("testOutputImage.jpg", thisImage);

        lastXformTime = xformTime;


        //Output point cloud for debugging
        if (imageNumber == 80){
            pcl::io::savePCDFile("testOutput.pcd", pcloud);
        }//if

    }//ImageCallback


    void TransformCallback(const geometry_msgs::TransformStamped& msg){
        int time_secs = msg.header.stamp.sec;
        int time_nsecs = msg.header.stamp.nsec;
        double timeValue = time_secs + (1e-9 * time_nsecs);
        geometry_msgs::Vector3 xlate        = msg.transform.translation;
        geometry_msgs::Quaternion rotate    = msg.transform.rotation;

        xformTime = timeValue;
        tf2::fromMsg(msg.transform, xform);

#if DEBUGXFORM
        ROS_INFO("===XFORM %s===", msg.header.frame_id.c_str());
        ROS_INFO("\ttime: %f", timeValue);
        ROS_INFO("\tTranslation:\t<%.05f, %.05ff, %.05f>", xlate.x, xlate.y, xlate.z);
        ROS_INFO("\tRotation:\t<%.05f, %.05f, %.05f, %.05f>", rotate.x, rotate.y, rotate.z, rotate.w);
#endif


    }//TransformCallback


      // this is registered every image sent
    void ImuCallback(const sensor_msgs::Imu& msg){
      // print stuff to show how stuff works
    //http://docs.ros.org/melodic/api/sensor_msgs/html/msg/Imu.html
    // not sure what the numbers are supposed to look like but they seem a bit weird just FYI
        ROS_INFO("orientation X: %9.6f", msg.orientation_covariance[0]);
        ROS_INFO("orientation y: %9.6f", msg.orientation_covariance[1]);
        ROS_INFO("orientation z: %9.6f", msg.orientation_covariance[2]);
        ROS_INFO("orientation w: %9.6f", msg.orientation_covariance[3]);
      
    }

    int main(int argc, char **argv){

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
        ros::init(argc, argv, "point_cloud");
   
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

        ros::Subscriber cameraSub = n.subscribe(imageSubPath, 1000, ImageCallback);

        ros::Subscriber xformColorSub = n.subscribe(xformSubPath1, 1000, TransformCallback);
        //ros::Subscriber xformDepthSub = n.subscribe(xformSubPath2, 1000, TransformCallback);

        ROS_INFO("point cloud, waiting for messages");
   
     /**
      * ros::spin() will enter a loop, pumping callbacks.  With this version, all
      * callbacks will be called from within this thread (the main one).  ros::spin()
      * will exit when Ctrl-C is pressed, or the node is shutdown by the master.
      */
        ros::spin();

        return 0;
    }//main
