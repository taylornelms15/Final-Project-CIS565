#include "point_cloud_node.h"
#include <Eigen/Dense>

#define DEBUGOUT 1

#define DEBUGCAMINFO (DEBUGOUT && 0)
#define DEBUGIMAGE (DEBUGOUT && 1)
#define DEBUGXFORM (DEBUGOUT && 0)

#define WRITING_PICS 1
#define USING_DRONEMOM_MSG 1

#define ENDFRAMENUM 5
#define SKIPFRAMES  1

//This is what converts depth-space into world space. In reality, USHRT_MAX becomes 4m
#define SCALING (100.0/65535.0)//1/256 also seemed close, so we could be way off the mark
//#define SCALING (1000.0 / 65535)

using namespace cv;
using namespace cv::xfeatures2d;


    const static char dronemomSubPath[] = "detectnet/detections";
    const static char imageSubPath[]    = "camera/rgb/image_raw";
    const static char camInfoSubPath[]  = "camera/rgb/camera_info";
    const static char dimageSubPath[]   = "camera/depth/image_raw";
    const static char dcamInfoSubPath[] = "camera/depth/camera_info";
    const static char xformSubPath1[]   = "tango_viwls/T_G_I";//tranformation for global system
    const static char xformSubPathC[]   = "tango/T_I_C_color";//tranformation for color camera
    const static char xformSubPathD[]   = "tango/T_I_C_depth";//tranformation for depth camera
    
    static sensor_msgs::CameraInfo      caminfo;
    static sensor_msgs::CameraInfo      dcaminfo;

    ///Global point cloud
    static pcl::PointCloud<PointT> pcloud       = pcl::PointCloud<PointT>(); 
    static pcl::PointCloud<PointT> prevcloud    = pcl::PointCloud<PointT>();

    static int                          numMatches  = 0;



    /**
    Gets the relative transform between two transforms in terms of floats
    */
    double_vec relativeRotateAndTranslate(tf2::Transform xform1, tf2::Transform xform2){
        tf2::Vector3 transDiff      = xform2.getOrigin() - xform1.getOrigin();
        tf2::Quaternion rotDiff     = xform2.getRotation() * xform1.getRotation().inverse();
        tf2::Transform xDiff        = tf2::Transform(rotDiff, transDiff);
        tf2Scalar raw[15];
        xDiff.getOpenGLMatrix(raw);//first 12 elements rotation, last 3 elements translation
        double_vec retval(15);
        for (int i = 0; i < 15; i++){
            retval[i] = raw[i];
        }//for
        return retval;

    }//relativeRotateAndTranslate

    /**
    Same as above, but puts the results into a Mat like opencv wants it
    */
    Mat relativeRotateAndTranslateM(tf2::Transform xform1, tf2::Transform xform2){
        double_vec raw = relativeRotateAndTranslate(xform1, xform2);
        Mat retval = Mat(4, 4, CV_64F);
        for(int i = 0; i < 4; i++){
            for(int j = 0; j < 4; j++){
                if (i == 3){
                    if (j == 3) retval.at<double>(i, j) = 1.0;
                    else retval.at<double>(i, j) = 0.0;
                }//last row
                else{
                    if (j == 3) retval.at<double>(i, j) = raw.at(12 + i);
                    else{
                        int index = 4 * i + j;
                        retval.at<double>(i, j) = raw.at(index);
                    }//else
                }//else

            }//for
        }//for
        return retval;

    }//relativeRotateAndTranslate

    /**
    This turns the "camera distortion" part of the message into a format that opencv might like
    We're assuming no translational distortion, gods help us...
    */
    Mat distCoeffsFromCamInfo(sensor_msgs::CameraInfo camInfo){
        Mat retval = Mat(1, 5, CV_64F);
        if (camInfo.D.size() >= 5){
            for (int i = 0; i < 5; i++){
                retval.at<double>(0, i) = camInfo.D[i];
            }//for
            return retval;
        }//if
        //hope for at<float> least 3 components maybe?
        retval.at<double>(0, 0) = camInfo.D[0];
        retval.at<double>(0, 1) = camInfo.D[1];
        retval.at<double>(0, 2) = 0.0f;
        retval.at<double>(0, 3) = 0.0f;
        retval.at<double>(0, 4) = camInfo.D[2];
        return retval;
    }//distCoeffsFromCamInfo

    /**
    Copies the intrinsic matrix to a format that opencv likes
    */
    Mat kFromCamInfo(sensor_msgs::CameraInfo camInfo){
        Mat retval = Mat(3, 3, CV_64F);
        for(int i = 0; i < 3; i++){
            for (int j = 0; j < 3; j++){
                retval.at<double>(i, j) = camInfo.K[3 * i + j];
            }
        }
        return retval;
    }//kFromCamInfo


    Mat undistortColor(const Mat& orig, sensor_msgs::CameraInfo camInfoC){

        Mat retval;
        Mat distcoeffs1         = distCoeffsFromCamInfo(camInfoC); 
        Mat intrinsicK1         = kFromCamInfo(camInfoC);

        undistort(orig, retval, intrinsicK1, distcoeffs1);
        return retval;

    }//undistortColor

    Point2f_vec getValidPoints(Mat imgCU, Mat depthMat){
        //get our list of valid points
        Point2f_vec validPoints = Point2f_vec();
        for(int i = 4; i < imgCU.rows - 4; i++){//removing edges
            for(int j = 4; j < imgCU.cols - 4; j++){//removing edges
                uint16_t depthVal = depthMat.at<uint16_t>(i, j);
                if (depthVal < 400) continue;
                validPoints.push_back(Point2f(j, i));
            }
        }
        return validPoints;

    }//getValidPoints

    void putValidPointsIntoCameraSpace(const Point2f_vec validPoints, const Point2f_vec validPointsU,
                                        gvec3_vec& pointsCamspace, gvec3_vec& colorsCamspace,
                                        const Mat imgCU, const Mat depthMat,
                                        sensor_msgs::CameraInfo camInfoC
                                        ){
        double centerX  = camInfoC.K.at(2);
        double centerY  = camInfoC.K.at(5);
        double focalX   = camInfoC.K.at(0);
        double focalY   = camInfoC.K.at(4);

        for (int i = 0; i < validPoints.size(); i++){
            //put in color calue
            Vec3b colorVal          = imgCU.at<Vec3b>(validPointsU[i]);
            colorsCamspace.push_back(gvec3(colorVal[2], colorVal[1], colorVal[0]));

            //get depth value
            uint16_t depthVal       = depthMat.at<uint16_t>(validPoints[i]);

            //transform our image-space coordinate into camera space
            double Z    = depthVal * SCALING;
            double X    = validPointsU[i].x - centerX;
            double Y    = validPointsU[i].y - centerY;
            X           *= Z / focalX;
            Y           *= Z / focalY;
            gvec3 cameraPoint = gvec3(X, Y, Z);
            pointsCamspace.push_back(cameraPoint);
        }//for


    }//putValidPointsIntoCameraSpace

    gvec3_vec cameraToWorldSpace(gvec3_vec pointsCamspace,
                                 tf2::Transform xformC,
                                 tf2::Transform xformG){

        //Next step: transform into world space
        gvec3_vec pointsWorldspace = gvec3_vec();
        tf2::Transform camToWorld = (xformC * xformG).inverse();
        for(int i = 0; i < pointsCamspace.size(); i++){
            gvec3 camvec            = pointsCamspace.at(i);
            //tf2::Vector3 camvecT    = tf2::Vector3(camvec.x, camvec.z, -camvec.y);//NOTE THE SWITCHED VALUES HERE (x, -z, -y) sorta worked
            tf2::Vector3 camvecT    = tf2::Vector3(-camvec.z, camvec.x, -camvec.y);//NOTE THE SWITCHED VALUES HERE (-z, x, y) working best so far
            tf2::Vector3 worldvecT  = camToWorld(camvecT);
            gvec3 worldvec          = gvec3(worldvecT.x(), worldvecT.y(), worldvecT.z());
            pointsWorldspace.push_back(worldvec);
        }//for

        return pointsWorldspace;
    }//cameraToWorldSpace

    PointT_vec pointsFromRGBD(Mat imgC, Mat imgD,
                        tf2::Transform xformC,//color
                        tf2::Transform xformD,//depth
                        tf2::Transform xformG,//global
                        sensor_msgs::CameraInfo camInfoC,
                        sensor_msgs::CameraInfo camInfoD
                        ){
        if (numMatches % SKIPFRAMES != 0){
            return PointT_vec();
        }//if
        ROS_INFO("===MATCHED IMAGES CALLBACK===");

        Mat rbt = relativeRotateAndTranslateM(xformD, xformC);
        Mat dC  = distCoeffsFromCamInfo(camInfoC);
        Mat kC  = kFromCamInfo(camInfoC);
        Mat kD  = kFromCamInfo(camInfoD);

        //match our d image to our rgb image
        Mat depthMat;
        rgbd::registerDepth(kD, kC, dC, rbt, imgD, imgC.size(), depthMat);

        //Undistort our color and depth map images
        Mat kCinv           = kC.inv();
        Mat imgCU           = undistortColor(imgC, camInfoC);//color image undistorted
        Mat imgDMU          = undistortColor(depthMat, camInfoC);//depth map undistorted
        
        //get our list of valid points
        Point2f_vec validPoints = getValidPoints(imgCU, depthMat);

        //undistort our points: want the projection to be for the most accurate camera projection
        Point2f_vec validPointsU;
        undistortPoints(validPoints, validPointsU, kC, dC, noArray(), kC);

        //Get our camera-space points
        gvec3_vec pointsCamspace = gvec3_vec();  
        gvec3_vec colorsCamspace = gvec3_vec();
        putValidPointsIntoCameraSpace(validPoints, validPointsU,
                                      pointsCamspace, colorsCamspace,
                                      imgCU, depthMat,
                                      camInfoC);

        //Next step: transform into world space
        gvec3_vec pointsWorldspace = cameraToWorldSpace(pointsCamspace, xformC, xformG);
        //gvec3_vec pointsWorldspace = pointsCamspace;

        //create color factor (for debugging)
        gvec3 factorF = gvec3(1.0, 0.99, 0.99); gvec3 factor = gvec3(1.0, 1.0, 1.0);
        for(int i = 0; i < numMatches; i++){
            factor *= factorF;
        }//for

        //Next step: push into our global point cloud
        PointT_vec retval = PointT_vec();
        for (int i = 0; i < colorsCamspace.size(); i++){
            gvec3 color = colorsCamspace.at(i) * factor;
            gvec3 pos   = pointsWorldspace.at(i);
            PointT nextPoint = PointT(color.r, color.g, color.b, 0);
            nextPoint.x = pos.x; nextPoint.y = pos.y; nextPoint.z = pos.z;
            pcloud.push_back(nextPoint);
            retval.push_back(nextPoint);
        }//for


        //breakpoint target so we don't fill our point cloud too much
        if (numMatches % ENDFRAMENUM == 0){
            pcl::io::savePCDFile("testOutput.pcd", pcloud);
            ROS_INFO("Successfully ran %d matches", numMatches);
            ros::shutdown();
        }//if
        return retval;
    }//pointsFromRGBD

    /**
    Helper function to display some points relative to our viewpoint
    */
    void makeDirectionOffsetPoint(tf2::Transform transform, float x, float y, float z, uint8_t r, uint8_t g, uint8_t b){
        transform                   = transform.inverse();
        tf2::Vector3 direction      = tf2::Vector3(x, y, z);
        //tf2::Transform rotation     = tf2::Transform(transform.getRotation());
        //tf2::Vector3 p = transform.getOrigin() + rotation(direction);
        tf2::Vector3 p = transform(direction);
        PointT point = PointT(r, g, b, 2);
        point.x = p[0]; point.y     = p[1]; point.z = p[2];
        pcloud.push_back(point);
    }//makeDirectionOffsetPoint

    //########################
    // ROS CALLBACKS
    //########################

    void DetectionCallback(const drone_mom_msgs::drone_mom::ConstPtr msg){
        numMatches++;
        //unpack the parts of this message that I want
        sensor_msgs::CameraInfo dcaminfo = msg->depth_camera_info;
        sensor_msgs::CameraInfo ccaminfo = msg->rgb_camera_info;
        int dwidth = msg->depth_image.width;
        int dheight = msg->depth_image.height;
        Mat dImage = Mat(dheight, dwidth, getEncodingTypeForCV(msg->depth_image.encoding), 
                         (void*) msg->depth_image.data.data(), msg->depth_image.step);
        int cwidth = msg->rgb_image.width;
        int cheight = msg->rgb_image.height;
        Mat cImage = Mat(cheight, cwidth, getEncodingTypeForCV(msg->rgb_image.encoding), 
                         (void*) msg->rgb_image.data.data(), msg->rgb_image.step);
        tf2::Transform xformD, xformC, xformG;
        tf2::fromMsg(msg->TIC_depth.transform, xformD);
        tf2::fromMsg(msg->TIC_color.transform, xformC);
        tf2::fromMsg(msg->TGI.transform, xformG);
        
        //char filenameC[100]; std::sprintf(filenameC, "color_image%d.png", numMatches);
        //imwrite(filenameC, cImage);

        ROS_INFO("===DRONEMOM MESSAGE==");
        //TODO: deal with classification bullshit


        //make our direction offset point for the "WTF" questions
        tf2::Transform thisXform = xformC * xformG;
        tf2::Transform otherXform = xformG;
        makeDirectionOffsetPoint(thisXform, 0, 0, 0, 255, 0, 255);
        //makeDirectionOffsetPoint(thisXform, .01, 0, 0, 255, 0, 0);
        //makeDirectionOffsetPoint(thisXform, 0, .01, 0, 0, 255, 0);
        //makeDirectionOffsetPoint(thisXform, 0, 0, .01, 0, 0, 255);
        makeDirectionOffsetPoint(thisXform, 0, .01, 0, 255, 0, 0);
        makeDirectionOffsetPoint(thisXform, 0, 0, -.01, 0, 255, 0);
        makeDirectionOffsetPoint(thisXform, -.01, 0, 0, 0, 0, 255);
        //testing other xform direction
        //makeDirectionOffsetPoint(otherXform, 0, 0, 0, 255, 255, 0);
        //makeDirectionOffsetPoint(otherXform, 0, .01, 0, 200, 110, 110);
        //makeDirectionOffsetPoint(otherXform, 0, 0, -.01, 110, 200, 110);
        //makeDirectionOffsetPoint(otherXform, -.01, 0, 0, 110, 110, 200);


        //Pass messages on to our point-cloud-making machine
        PointT_vec bunchOfPoints = pointsFromRGBD(cImage, dImage,
            xformC, xformD, xformG,
            ccaminfo, dcaminfo);


    }//void

    int main(int argc, char **argv){

     /**
      * The ros::init() function needs to see argc and argv so that it can perform
      * any ROS arguments and name remapping that were provided at the command line.
      * For programmatic remappings you can use a different version of init() which takes
      * remappings directly, but for most command-line programs, passing argc and argv is
      * the easiest way to do it.  The third argument to init() is the name of the node.
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
      */

        ros::Subscriber dmomSub         = n.subscribe(dronemomSubPath, 1000, DetectionCallback);

        //small function to verify our link to CUDA
        float array1[32];
        float array2[32];
        for (int i = 0; i < 32; i++){
            //array1[i] = 1.0f;
            //array2[i] = 1.0f;
            array1[i] = (float) ((i * 23) % 4);
            array2[i] = (float) ((i * 25) % 3);
        }//for
        float dotProduct = testCudaFunctionality(array1, array2);
        ROS_INFO("Dot product: %f", dotProduct);


        ROS_INFO("point cloud, waiting for messages");
   
     /**
      * ros::spin() will enter a loop, pumping callbacks.  With this version, all
      * callbacks will be called from within this thread (the main one).  ros::spin()
      * will exit when Ctrl-C is pressed, or the node is shutdown by the master.
      */
        ros::spin();

        return 0;
    }//main

    //########################################################
    // Old/unused functions (for archiving and maybe revival)
    //########################################################

    void getKeyPointMatches(Mat img1, Mat img2, KeyPoint_vec *kp1, KeyPoint_vec *kp2, DMatch_vec *matches){
        //-- Step 1: Detect the keypoints using SURF Detector, compute the descriptors
        int minHessian = 700;
        Ptr<SURF> detector = SURF::create( minHessian );
        KeyPoint_vec keypoints1, keypoints2;
        Mat descriptors1, descriptors2;
        detector->detectAndCompute( img1, noArray(), keypoints1, descriptors1 );
        detector->detectAndCompute( img2, noArray(), keypoints2, descriptors2 );
        //-- Step 2: Matching descriptor vectors with a FLANN based matcher
        // Since SURF is a floating-point descriptor NORM_L2 is used
        Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create(DescriptorMatcher::FLANNBASED);
        std::vector< DMatch_vec > knn_matches;
        matcher->knnMatch( descriptors1, descriptors2, knn_matches, 2 );
        //-- Filter matches using the Lowe's ratio test
        const float ratio_thresh = 0.7f;
        DMatch_vec good_matches;
        for (size_t i = 0; i < knn_matches.size(); i++)
        {
            if (knn_matches[i][0].distance < ratio_thresh * knn_matches[i][1].distance)
            {
                good_matches.push_back(knn_matches[i][0]);
            }
        }
        *kp1 = KeyPoint_vec(keypoints1);
        *kp2 = KeyPoint_vec(keypoints2);
        *matches = DMatch_vec(good_matches);
    }

    std::vector<Point2f_vec > coordsFromMatches(KeyPoint_vec keypoints1,
                                                         KeyPoint_vec keypoints2,
                                                         DMatch_vec good_matches){
        Point2f_vec points1 = Point2f_vec();
        Point2f_vec points2 = Point2f_vec();
        for(int i = 0; i < good_matches.size(); i++){
            DMatch match = good_matches.at(i);
            points1.push_back(keypoints1[match.trainIdx].pt);
            points2.push_back(keypoints2[match.trainIdx].pt);
        }//for

        std::vector<Point2f_vec > retval = std::vector<Point2f_vec >();
        retval.push_back(points1);
        retval.push_back(points2);
        return retval;
        
    }//coordsFromMatches

    /**
    First Mat is a 3x3 rotation, second is a 3x1 translation
    */
    void rotAndTransFromXform(tf2::Transform xform1, Mat &rot, Mat &trans){
        double raw[15];
        xform1.getOpenGLMatrix(raw);
        Mat translation     = Mat(3, 1, CV_64F, raw + 12);
        Mat rotation        = Mat(3, 4, CV_64F, raw);
        rotation            = rotation.colRange(0, 3);

        for (int i = 0; i < 3; i++){
            for(int j = 0; j < 3; j++){
                rot.at<double>(i, j) = rotation.at<double>(i, j);
            }//for
        }//for
        for(int i = 0; i < 3; i++){
            trans.at<double>(i, 1) = translation.at<double>(i, 1);
        }//for
    
    }//rotAndTransFromXform

    /**
    Could definitely be moved into CUDA (look at that sweet, sweet parallelism)
    But... we're not using the fisheye camera
    */
    Mat undistortFishEye(const Mat &distorted, const float w)
    {
        Mat map_x, map_y;
        map_x.create(distorted.size(), CV_32FC1);
        map_y.create(distorted.size(), CV_32FC1);

        double Cx = distorted.cols / 2.0;
        double Cy = distorted.rows / 2.0;

        double halfTan = tan(w / 2.0);
        for (double x = -1.0; x < 1.0; x += 1.0/Cx) {
            for (double y = -1.0; y < 1.0; y += 1.0/Cy) {
                double ru = sqrt(x*x + y*y);
                double rd = (1.0 / w)*atan(2.0*ru*halfTan);

                map_x.at<float>(y*Cy + Cy, x*Cx + Cx) = rd/ru * x*Cx + Cx;
                map_y.at<float>(y*Cy + Cy, x*Cx + Cx) = rd/ru * y*Cy + Cy;
            }
        }

        Mat undistorted;
        remap(distorted, undistorted, map_x, map_y, INTER_LINEAR);
        return undistorted;
    }
