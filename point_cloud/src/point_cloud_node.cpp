#include "point_cloud_node.h"
#include <Eigen/Dense>

#define DEBUGOUT 1

#define DEBUGCAMINFO (DEBUGOUT && 1)
#define DEBUGIMAGE (DEBUGOUT && 1)
#define DEBUGXFORM (DEBUGOUT && 0)


using namespace cv;
using namespace cv::xfeatures2d;

    //office dataset: recorded on Google Tango tablet
    //the camera's model number: OV4682 RGB IR
    //const static float    FoV = 131.0;
    const static float    FoV = 120.0;

    const static char imageSubPath[]    = "camera/rgb/image_raw";
    const static char camInfoSubPath[]  = "camera/rgb/camera_info";
    const static char dimageSubPath[]   = "camera/depth/image_raw";
    const static char dcamInfoSubPath[] = "camera/depth/camera_info";
    const static char xformSubPath1[]   = "tango_viwls/T_G_I";//tranformation for global system
    const static char xformSubPathC[]   = "tango/T_I_C_color";//tranformation for color camera
    const static char xformSubPathD[]   = "tango/T_I_C_depth";//tranformation for depth camera
    
    static int imageNumber = 0;
    static double                       xformTime = -1;
    static double                       lastXformTime = -1;
    static bool                         found1stXform = false;
    static tf2::Transform               xform;
    static tf2::Transform               xformColor;
    static tf2::Transform               xformDepth;
    static sensor_msgs::CameraInfo      caminfo;
    static pcl::PointCloud<PointT> pcloud = pcl::PointCloud<PointT>(); 

    static Mat prevImage = Mat();

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

    /**
    This turns the "camera distortion" part of the message into a format that opencv might like
    We're assuming no translational distortion, gods help us...
    */
    Mat distCoeffsFromCamInfo(sensor_msgs::CameraInfo camInfo){
        Mat retval = Mat(1, 5, CV_64F);
        if (camInfo.D.size() > 5){
            for (int i = 0; i < 5; i++){
                retval.at<float>(0, i) = camInfo.D[i];
            }//for
            return retval;
        }//if
        //hope for at<float> least 3 components maybe?
        retval.at<float>(0, 0) = camInfo.D[0];
        retval.at<float>(0, 1) = camInfo.D[1];
        retval.at<float>(0, 2) = 0.0f;
        retval.at<float>(0, 3) = 0.0f;
        retval.at<float>(0, 4) = camInfo.D[2];
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
    Note: unused?
    */
    tf2::Transform switchCoordinateSystems(tf2::Transform xform1){
        tf2::Vector3 originalOrigin     = xform1.getOrigin();
        tf2::Quaternion originalRotate  = xform1.getRotation();
        tf2::Vector3 newOrigin          = tf2::Vector3(-originalOrigin.y(), -originalOrigin.z(), originalOrigin.x());
        tf2::Quaternion newRotate       = originalRotate;

        return tf2::Transform(newRotate, newOrigin);

    }//switchCoordinateSystems

    std::vector<PointT> pointsFromDuo(Mat img1, Mat img2, 
                                      tf2::Transform xform1, 
                                      tf2::Transform xform2, 
                                      sensor_msgs::CameraInfo camInfo){
        std::vector<PointT> retval = std::vector<PointT>();

        //TODO: re-evaluate this
        //xform1 = xform1.inverse();
        //xform2 = xform2.inverse();

        //undistort our images
        Mat img1better, img2better;
        Mat distcoeffs1         = distCoeffsFromCamInfo(camInfo); 
        Mat distcoeffs2         = distcoeffs1.clone();
        Mat intrinsicK1         = kFromCamInfo(camInfo);
        Mat intrinsicK2         = intrinsicK1.clone();

        undistort(img1, img1better, intrinsicK1, distcoeffs1);
        undistort(img2, img2better, intrinsicK2, distcoeffs2);

        //get keypoints
        KeyPoint_vec keypoints1, keypoints2;
        DMatch_vec good_matches;
        getKeyPointMatches(img1better, img2better, &keypoints1, &keypoints2, &good_matches);

        //get our feature points
        Point2f_vec img1points, img2points;
        std::vector<Point2f_vec> coordVec = coordsFromMatches(keypoints1, keypoints2, good_matches);
        img1points = coordVec[0];
        img2points = coordVec[1];

        //match our feature points
        Mat img_matches;
        drawMatches( img1better, keypoints1, img2better, keypoints2, good_matches, img_matches, Scalar::all(-1),
                     Scalar::all(-1), std::vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );
        imwrite("matches.jpg", img_matches);


        //testing: sfm stuff
        Mat rot1, rot2, trn1, trn2;
        double xform1raw[15]; double xform2raw[15];
        xform1.getOpenGLMatrix(xform1raw);
        xform2.getOpenGLMatrix(xform2raw);
        rot1 = Mat(3, 4, CV_64F, xform1raw); rot2 = Mat(3, 4, CV_64F, xform2raw); trn1 = Mat(3, 1, CV_64F, xform1raw + 12); trn2 = Mat(3, 1, CV_64F, xform2raw + 12);
        rot1 = rot1.colRange(0, 3); rot2 = rot2.colRange(0, 3);
        Mat Pguess1, Pguess2;
        sfm::projectionFromKRt(intrinsicK1, rot1, trn1, Pguess1);
        sfm::projectionFromKRt(intrinsicK2, rot2, trn2, Pguess2);

        //triangulate our points
        Mat triout2;

        Mat points1Mat = Mat(2, img1points.size(), CV_64F);
        Mat points2Mat = Mat(2, img2points.size(), CV_64F);
        for(int i = 0; i < img1points.size(); i++){
            points1Mat.at<double>(0, i) = img1points[i].x;
            points1Mat.at<double>(1, i) = img1points[i].y;
            points2Mat.at<double>(0, i) = img2points[i].x;
            points2Mat.at<double>(1, i) = img2points[i].y;
        }//for
        std::vector<Mat> inputPoints = std::vector<Mat>(); inputPoints.push_back(points1Mat); inputPoints.push_back(points2Mat);
        std::vector<Mat> projMatrices = std::vector<Mat>(); projMatrices.push_back(Pguess1); projMatrices.push_back(Pguess2);

        //sfm triangulation
        sfm::triangulatePoints(inputPoints, projMatrices, triout2);

        //sfm reconstruction
        Mat Routs, Touts, triout;
        sfm::reconstruct(inputPoints, Routs, Touts, intrinsicK1, triout, true);

        std::vector<glm::vec4> results4d = std::vector<glm::vec4>();
        //note: can take out this homogenous stuff, not used anymore
        for(int i = 0; i < triout2.cols; i++){
            glm::vec4 next = glm::vec4(triout2.at<double>(0, i),
                                       triout2.at<double>(1, i),
                                       triout2.at<double>(2, i),
                                       1);
            results4d.push_back(next);
        }//for

        for (int i = 0; i < results4d.size(); i++){
        //for (int i = 0; i < 4; i++){
            Point2f loc = img1points[i];
            Vec3b color = img1better.at<Vec3b>(loc);
            PointT point = PointT(color[2], color[1], color[0], 0);
            //PointT point = PointT(255, 255, 255, 0);
            glm::vec4 point4d = results4d[i];
            point.x = point4d.x / point4d.w;
            point.y = point4d.y / point4d.w;
            point.z = point4d.z / point4d.w;
            retval.push_back(point);
        }//for

        return retval;
    }//pointsFromDuo



    void makeDirectionOffsetPoint(tf2::Transform transform, float x, float y, float z, uint8_t r, uint8_t g, uint8_t b){
        tf2::Vector3 direction = tf2::Vector3(x, y, z);
        tf2::Transform rotation = tf2::Transform(transform.getRotation());
        tf2::Vector3 p = transform.getOrigin() + rotation(direction);
        PointT point = PointT(r, g, b, 2);
        point.x = p[0]; point.y = p[1]; point.z = p[2];
        pcloud.push_back(point);
    }//makeDirectionOffsetPoint

    void CameraInfoCallback(sensor_msgs::CameraInfo msg){
        caminfo = sensor_msgs::CameraInfo(msg);

#if DEBUGCAMINFO
        ROS_INFO("===CAMERA INFO===");
        ROS_INFO("\tw: %d\th: %d", msg.width, msg.height);
#endif
    }//CameraInfoCallback

    void ImageCallback(const sensor_msgs::Image& msg){
        if (!found1stXform) return;
#if DEBUGIMAGE
        ROS_INFO("===IMAGE===");
#endif
        int width = msg.width;
        int height = msg.height;
        Mat thisImage = Mat(height, width, getEncodingTypeForCV(msg.encoding), (void*) msg.data.data());

        tf2::Transform thisTransform = xformColor * xform;

        imgqueue.push_back(thisImage.clone());
        xformqueue.push_back(thisTransform);
        if (imgqueue.size() < 10) return;//if first images, ignore
        else if (imgqueue.size() > 10){
            imgqueue.pop_front();//discard the last image back
            xformqueue.pop_front();
        }//else

        tf2::Transform xform1 = xformqueue.at(0);
        tf2::Transform xform2 = xformqueue.at(9);
        std::vector<PointT> resultPoints = pointsFromDuo(imgqueue[0], imgqueue[9], xform1, xform2, caminfo);


        makeDirectionOffsetPoint(xform1, 0.0, 0.0, 0.0, 255, 0, 255);//c magenta
        makeDirectionOffsetPoint(xform2, 0.0, 0.0, 0.0, 0, 255, 255);//c cyan
        makeDirectionOffsetPoint(xform1, 0.015, 0.0, 0.0, 255, 0, 0);//x red
        makeDirectionOffsetPoint(xform1, 0.0, 0.015, 0.0, 0, 255, 0);//y green
        makeDirectionOffsetPoint(xform1, 0.0, 0.0, 0.015, 0, 0, 255);//z blue
        makeDirectionOffsetPoint(xform2, 0.015, 0.0, 0.0, 255, 0, 0);//x red
        makeDirectionOffsetPoint(xform2, 0.0, 0.015, 0.0, 0, 255, 0);//y green
        makeDirectionOffsetPoint(xform2, 0.0, 0.0, 0.015, 0, 0, 255);//z blue
        for(int i = 0; i < resultPoints.size(); i += 4){
            PointT point = resultPoints[i];
            pcloud.push_back(point);
        }//for
        pcl::io::savePCDFile("testOutput.pcd", pcloud);
        return;
    }//ImageCallback

    void TransformCallback1(const geometry_msgs::TransformStamped& msg){
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
        ROS_INFO("===XFORM COLOR===");
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
        
        ros::Subscriber cameraInfoSub   = n.subscribe(camInfoSubPath, 1000, CameraInfoCallback);

        ros::Subscriber xformGlobalSub  = n.subscribe(xformSubPath1, 1000, TransformCallback1);
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
