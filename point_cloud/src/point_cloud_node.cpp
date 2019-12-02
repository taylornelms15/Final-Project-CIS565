#include "point_cloud_node.h"
#include <Eigen/Dense>

#define DEBUGOUT 1

#define DEBUGCAMINFO (DEBUGOUT && 0)
#define DEBUGIMAGE (DEBUGOUT && 1)
#define DEBUGXFORM (DEBUGOUT && 0)

#define WRITING_PICS 1


using namespace cv;
using namespace cv::xfeatures2d;


    const static char imageSubPath[]    = "camera/rgb/image_raw";
    const static char camInfoSubPath[]  = "camera/rgb/camera_info";
    const static char dimageSubPath[]   = "camera/depth/image_raw";
    const static char dcamInfoSubPath[] = "camera/depth/camera_info";
    const static char xformSubPath1[]   = "tango_viwls/T_G_I";//tranformation for global system
    const static char xformSubPathC[]   = "tango/T_I_C_color";//tranformation for color camera
    const static char xformSubPathD[]   = "tango/T_I_C_depth";//tranformation for depth camera
    
    static double                       xformTime = -1;
    static bool                         found1stXform       = false;
    static bool                         found1stCaminfo     = false;
    static bool                         found1stDCaminfo    = false;
    static tf2::Transform               xform;
    static tf2::Transform               xformColor;
    static tf2::Transform               xformDepth;
    static sensor_msgs::CameraInfo      caminfo;
    static sensor_msgs::CameraInfo      dcaminfo;
    static pcl::PointCloud<PointT> pcloud = pcl::PointCloud<PointT>(); 

    static Mat imageC;
    static Mat imageD;
    enum Imgprogress{Free, WaitC, WaitD};
    static Imgprogress                  imgprogress = Free;
    static int                          numMatches  = 0;

    static std::deque<Mat> imgqueue                 = std::deque<Mat>();
    static std::deque<tf2::Transform> xformqueue    = std::deque<tf2::Transform>();

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

    Mat undistortColor(const Mat& orig, sensor_msgs::CameraInfo camInfoC){

        Mat retval;
        Mat distcoeffs1         = distCoeffsFromCamInfo(camInfoC); 
        Mat intrinsicK1         = kFromCamInfo(camInfoC);

        undistort(orig, retval, intrinsicK1, distcoeffs1);
        return retval;

    }//undistortColor

    std::vector<glm::vec3> cameraToWorldSpace(std::vector<glm::vec3> pointsCamspace,
                                              tf2::Transform xformC,
                                              tf2::Transform xformG){

        //Next step: transform into world space
        std::vector<glm::vec3> pointsWorldspace = std::vector<glm::vec3>();
        tf2::Transform camToWorld = xformC * xformG;
        for(int i = 0; i < pointsCamspace.size(); i++){
            glm::vec3 camvec        = pointsCamspace.at(i);
            tf2::Vector3 camvecT    = tf2::Vector3(camvec.x, camvec.z, camvec.y);//NOTE THE SWITCHED VALUES HERE
            tf2::Vector3 worldvecT  = camToWorld(camvecT);
            glm::vec3 worldvec      = glm::vec3(worldvecT.x(), worldvecT.y(), worldvecT.z());
            pointsWorldspace.push_back(worldvec);
        }//for

        return pointsWorldspace;
    }//cameraToWorldSpace

    #define SCALING (1.0/1000.0)
    PointT_vec pointsFromRGBD(Mat imgC, Mat imgD,
                        tf2::Transform xformC,//color
                        tf2::Transform xformD,//depth
                        tf2::Transform xformG,//global
                        sensor_msgs::CameraInfo camInfoC,
                        sensor_msgs::CameraInfo camInfoD
                        ){
        ROS_INFO("===MATCHED IMAGES CALLBACK===");
        numMatches++;

        Mat rbt = relativeRotateAndTranslateM(xformD, xformC);
        Mat dC  = distCoeffsFromCamInfo(camInfoC);
        Mat kC  = kFromCamInfo(camInfoC);
        Mat kD  = kFromCamInfo(camInfoD);

        //match our d image to our rgb image
        Mat depthMat;
        rgbd::registerDepth(kD, kC, dC, rbt, imgD, imgC.size(), depthMat);
        Mat depthMatAlt;
        normalize(depthMat, depthMatAlt, 0xffff, 0, NORM_MINMAX);
        #if WRITING_PICS
        imwrite("Depth_map_normalized.png", depthMatAlt);
        #endif

        //Undistort our color and depth map images
        std::vector<glm::vec3> pointsCamspace = std::vector<glm::vec3>();  
        std::vector<glm::vec3> colorsCamspace = std::vector<glm::vec3>();
        double centerX  = camInfoC.K.at(2);
        double centerY  = camInfoC.K.at(5);
        double focalX   = camInfoC.K.at(0);
        double focalY   = camInfoC.K.at(4);
        Mat kCinv   = kC.inv();
        glm::mat3 kCinvG    = glm::make_mat3((double*) kCinv.data);
        Mat imgCU   = undistortColor(imgC, camInfoC);
        Mat imgDMU  = undistortColor(depthMat, camInfoC);
        normalize(imgDMU, depthMatAlt, 0xffff, 0, NORM_MINMAX);
        #if WRITING_PICS
        imwrite("Depth_map_normalized_undistorted.png", depthMatAlt);
        #endif
        
        //get our list of valid points
        Point2f_vec validPoints = Point2f_vec();
        Point2f_vec validPointsU;
        for(int i = 4; i < imgCU.rows - 4; i++){//removing edges
            for(int j = 4; j < imgCU.cols - 4; j++){//removing edges
                uint16_t depthVal = depthMat.at<uint16_t>(i, j);
                if (depthVal < 800) continue;
                validPoints.push_back(Point2f(j, i));
            }
        }

        //undistort our points: want the projection to be for the most accurate camera projection
        undistortPoints(validPoints, validPointsU, kC, dC, noArray(), kC);


        for (int i = 0; i < validPoints.size(); i++){
            //put in color calue
            Vec3b colorVal          = imgCU.at<Vec3b>(validPointsU[i]);
            colorsCamspace.push_back(glm::vec3(colorVal[2], colorVal[1], colorVal[0]));

            //get depth value
            uint16_t depthVal       = depthMat.at<uint16_t>(validPoints[i]);

            //transform our image-space coordinate into camera space
            double Z    = depthVal * SCALING;
            double X    = validPointsU[i].x - centerX;
            double Y    = validPointsU[i].y - centerY;
            X           *= Z / focalX;
            Y           *= Z / focalY;
            glm::vec3 cameraPoint = glm::vec3(X, Y, Z);
            pointsCamspace.push_back(cameraPoint);
        }//for


        //Next step: transform into world space
        std::vector<glm::vec3> pointsWorldspace = cameraToWorldSpace(pointsCamspace,
                                                                     xformC, xformG);


        //Next step: push into our point cloud
        PointT_vec retval = PointT_vec();
        for (int i = 0; i < colorsCamspace.size(); i++){
            glm::vec3 color = colorsCamspace.at(i);
            glm::vec3 pos   = pointsWorldspace.at(i);
            PointT nextPoint = PointT(color.r, color.g, color.b, 0);
            nextPoint.x = pos.x; nextPoint.y = pos.y; nextPoint.z = pos.z;
            pcloud.push_back(nextPoint);
            retval.push_back(nextPoint);
        }//for


        pcl::io::savePCDFile("testOutput.pcd", pcloud);
        //Reset our FSM for "waiting for image data"
        imgprogress = Free;

        //breakpoint target so we don't fill our point cloud too much
        if (numMatches % 12 == 0){
            ROS_INFO("Successfully ran %d matches", numMatches);
        }//if
        return retval;
    }//pointsFromRGBD

    void makeDirectionOffsetPoint(tf2::Transform transform, float x, float y, float z, uint8_t r, uint8_t g, uint8_t b){
        tf2::Vector3 direction = tf2::Vector3(x, y, z);
        tf2::Transform rotation = tf2::Transform(transform.getRotation());
        tf2::Vector3 p = transform.getOrigin() + rotation(direction);
        PointT point = PointT(r, g, b, 2);
        point.x = p[0]; point.y = p[1]; point.z = p[2];
        pcloud.push_back(point);
    }//makeDirectionOffsetPoint

    //########################
    // ROS CALLBACKS
    //########################

    void DCameraInfoCallback(sensor_msgs::CameraInfo msg){
        dcaminfo = sensor_msgs::CameraInfo(msg);
        found1stDCaminfo = true;

        #if DEBUGCAMINFO
        ROS_INFO("===CAMERA INFO D===");
        #endif
    }//CameraInfoCallback

    void CameraInfoCallback(sensor_msgs::CameraInfo msg){
        caminfo = sensor_msgs::CameraInfo(msg);
        found1stCaminfo = true;

        #if DEBUGCAMINFO
        ROS_INFO("===CAMERA INFO C===");
        #endif
    }//CameraInfoCallback

    void DImageCallback(const sensor_msgs::Image& msg){
        if (!found1stDCaminfo) return;
        
        #if DEBUGIMAGE
        ROS_INFO("===IMAGE D===");
        #endif
        int width = msg.width;
        int height = msg.height;
        Mat thisImage = Mat(height, width, getEncodingTypeForCV(msg.encoding), (void*) msg.data.data(), msg.step);

        double minVal, maxVal;
        minMaxLoc(thisImage, &minVal, &maxVal);
        #if WRITING_PICS
        Mat thisImageAlt = Mat(height, width, getEncodingTypeForCV(msg.encoding));
        normalize(thisImage, thisImageAlt, 0xffff, 0, NORM_MINMAX);

        imwrite("depth_image.png", thisImage);
        imwrite("depth_image_normalized.png", thisImageAlt);
        #endif
        #if DEBUGIMAGE
        ROS_INFO("\tdepth mat rows %d, cols %d, min val %f, max val %f", thisImage.rows, thisImage.cols, minVal, maxVal);
        #endif
        imageD = thisImage.clone();

        if (imgprogress == Free){
            imgprogress = WaitC;
        }//if
        else if (imgprogress == WaitD){
            makeDirectionOffsetPoint(xform, 0, 0, 0, 255, 0, 255);
            makeDirectionOffsetPoint(xform, 0.01, 0, 0, 255, 0, 0);
            makeDirectionOffsetPoint(xform, 0, 0.01, 0, 0, 255, 0);
            makeDirectionOffsetPoint(xform, 0, 0, 0.01, 0, 0, 255);
            pointsFromRGBD(imageC, imageD,
            xformColor, xformDepth, xform,
            caminfo, dcaminfo);
        }//else

    }//DImageCallback

    void ImageCallback(const sensor_msgs::Image& msg){
        if (!found1stXform) return;
        if (!found1stCaminfo) return;
        #if DEBUGIMAGE
        ROS_INFO("===IMAGE C===");
        #endif
        int width = msg.width;
        int height = msg.height;
        Mat thisImage = Mat(height, width, getEncodingTypeForCV(msg.encoding), (void*) msg.data.data());

        imwrite("color_image.png", thisImage);

        imageC = thisImage.clone();

        if (imgprogress == Free){
            imgprogress = WaitD;
        }//if
        else if (imgprogress == WaitC){
            makeDirectionOffsetPoint(xform, 0, 0, 0, 255, 0, 255);
            pointsFromRGBD(imageC, imageD,
            xformColor, xformDepth, xform,
            caminfo, dcaminfo);
        }//else


        return;
    }//ImageCallback

    void TransformCallbackG(const geometry_msgs::TransformStamped& msg){
        int time_secs = msg.header.stamp.sec;
        int time_nsecs = msg.header.stamp.nsec;
        double timeValue = time_secs + (1e-9 * time_nsecs);
        geometry_msgs::Vector3 xlate        = msg.transform.translation;
        geometry_msgs::Quaternion rotate    = msg.transform.rotation;

        xformTime = timeValue;
        tf2::fromMsg(msg.transform, xform);
        found1stXform = true;

        #if DEBUGXFORM
        ROS_INFO("===XFORM GLOBAL===");
        ROS_INFO("\ttime: %f", timeValue);
        ROS_INFO("\tTranslation:\t<%.05f, %.05f, %.05f>", xlate.x, xlate.y, xlate.z);
        ROS_INFO("\tRotation:\t<%.05f, %.05f, %.05f, %.05f>", rotate.x, rotate.y, rotate.z, rotate.w);
        #endif
    }//TransformCallback2

    void TransformCallbackC(const geometry_msgs::TransformStamped& msg){
        int time_secs = msg.header.stamp.sec;
        int time_nsecs = msg.header.stamp.nsec;
        double timeValue = time_secs + (1e-9 * time_nsecs);
        geometry_msgs::Vector3 xlate        = msg.transform.translation;
        geometry_msgs::Quaternion rotate    = msg.transform.rotation;

        tf2::fromMsg(msg.transform, xformColor);

        #if DEBUGXFORM
        ROS_INFO("===XFORM COLOR===");
        ROS_INFO("\ttime: %f", timeValue);
        ROS_INFO("\tTranslation:\t<%.05f, %.05f, %.05f>", xlate.x, xlate.y, xlate.z);
        ROS_INFO("\tRotation:\t<%.05f, %.05f, %.05f, %.05f>", rotate.x, rotate.y, rotate.z, rotate.w);
        #endif


    }//TransformCallback

    void TransformCallbackD(const geometry_msgs::TransformStamped& msg){
        int time_secs = msg.header.stamp.sec;
        int time_nsecs = msg.header.stamp.nsec;
        double timeValue = time_secs + (1e-9 * time_nsecs);
        geometry_msgs::Vector3 xlate        = msg.transform.translation;
        geometry_msgs::Quaternion rotate    = msg.transform.rotation;

        tf2::fromMsg(msg.transform, xformDepth);

        #if DEBUGXFORM
        ROS_INFO("===XFORM DEPTH===");
        ROS_INFO("\ttime: %f", timeValue);
        ROS_INFO("\tTranslation:\t<%.05f, %.05f, %.05f>", xlate.x, xlate.y, xlate.z);
        ROS_INFO("\tRotation:\t<%.05f, %.05f, %.05f, %.05f>", rotate.x, rotate.y, rotate.z, rotate.w);
        #endif

    }//TransformCallback

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

        ros::Subscriber cameraSub       = n.subscribe(imageSubPath, 1000, ImageCallback);
        ros::Subscriber dcameraSub      = n.subscribe(dimageSubPath, 1000, DImageCallback);
        
        ros::Subscriber cameraInfoSub   = n.subscribe(camInfoSubPath, 1000, CameraInfoCallback);
        ros::Subscriber dcameraInfoSub  = n.subscribe(dcamInfoSubPath, 1000, DCameraInfoCallback);

        ros::Subscriber xformGlobalSub  = n.subscribe(xformSubPath1, 1000, TransformCallbackG);
        ros::Subscriber xformColorSub   = n.subscribe(xformSubPathC, 1000, TransformCallbackC);
        ros::Subscriber xformDepthSub   = n.subscribe(xformSubPathD, 1000, TransformCallbackD);

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
