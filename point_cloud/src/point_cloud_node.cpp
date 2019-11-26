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
    const static char xformSubPath1[]   = "tango_viwls/T_G_I";//tranformation for global system
    const static char xformSubPath2[]   = "tango/T_I_C_color";//tranformation for color camera
    //const static char imageSubPath[]  = "color_image";
    //const static char xformSubPath1[] = "T_G_C";//tranformation for color camera
    //const static char xformSubPath2[]   = "T_G_D";//transformation for depth camera
    
    static int imageNumber = 0;
    static double                       xformTime = -1;
    static double                       lastXformTime = -1;
    static bool                         found1stXform = false;
    static tf2::Transform               xform;
    static sensor_msgs::CameraInfo      caminfo;
    static pcl::PointCloud<PointT> pcloud = pcl::PointCloud<PointT>(); 

    static Mat prevImage = Mat();
    static tf2::Transform prevxform;

    static std::deque<Mat> imgqueue = std::deque<Mat>();
    static std::deque<tf2::Transform> xformqueue = std::deque<tf2::Transform>();

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

    void getKeyPointMatches(Mat img1, Mat img2, Mat img3, 
            KeyPoint_vec *kp1, KeyPoint_vec *kp2, KeyPoint_vec *kp3, 
            DMatch_vec *match12, DMatch_vec *match23, DMatch_vec *match31){
        int minHessian = 700;
        Ptr<SURF> detector = SURF::create( minHessian );
        KeyPoint_vec keypoints1, keypoints2, keypoints3;
        Mat descriptors1, descriptors2, descriptors3;
        detector->detectAndCompute( img1, noArray(), keypoints1, descriptors1 );
        detector->detectAndCompute( img2, noArray(), keypoints2, descriptors2 );
        detector->detectAndCompute( img3, noArray(), keypoints3, descriptors3 );
        Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create(DescriptorMatcher::FLANNBASED);
        std::vector< DMatch_vec > knn_matches1, knn_matches2, knn_matches3;
        const float ratio_thresh = 0.7f;
        DMatch_vec good_matches1, good_matches2, good_matches3;
        for (size_t i = 0; i < knn_matches1.size(); i++)
        {
            if (knn_matches1[i][0].distance < ratio_thresh * knn_matches1[i][1].distance)
            {
                good_matches1.push_back(knn_matches1[i][0]);
            }
        }
        for (size_t i = 0; i < knn_matches2.size(); i++)
        {
            if (knn_matches2[i][0].distance < ratio_thresh * knn_matches2[i][1].distance)
            {
                good_matches2.push_back(knn_matches2[i][0]);
            }
        }
        for (size_t i = 0; i < knn_matches3.size(); i++)
        {
            if (knn_matches3[i][0].distance < ratio_thresh * knn_matches3[i][1].distance)
            {
                good_matches3.push_back(knn_matches3[i][0]);
            }
        }

        *kp1 = KeyPoint_vec(keypoints1);
        *kp2 = KeyPoint_vec(keypoints2);
        *kp3 = KeyPoint_vec(keypoints3);
        *match12 = DMatch_vec(good_matches1);
        *match23 = DMatch_vec(good_matches2);
        *match31 = DMatch_vec(good_matches3);
    }//getKeypointMatches

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

    std::vector<PointT> pointsFromDuo(Mat img1, Mat img2, 
                                      tf2::Transform xform1, 
                                      tf2::Transform xform2, 
                                      sensor_msgs::CameraInfo camInfo){
        std::vector<PointT> retval = std::vector<PointT>();


        KeyPoint_vec keypoints1, keypoints2;
        DMatch_vec good_matches;
        getKeyPointMatches(img1, img2, &keypoints1, &keypoints2, &good_matches);

        //get our heature points
        Point2f_vec img1points, img2points;
        std::vector<Point2f_vec> coordVec = coordsFromMatches(keypoints1, keypoints2, good_matches);
        img1points = coordVec[0];
        img2points = coordVec[1];

        //match our feature points
        Mat img_matches;
        drawMatches( img1, keypoints1, img2, keypoints2, good_matches, img_matches, Scalar::all(-1),
                     Scalar::all(-1), std::vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );
        imwrite("matches.jpg", img_matches);

        //rectify our "stereo" camera
        double_vec rotAndTrans   = relativeRotateAndTranslate(xform1, xform2);
        Mat rotMatrix           = Mat(3, 4, CV_64F, rotAndTrans.data());
        Mat transMatrix         = Mat(3, 1, CV_64F, rotAndTrans.data() + 12);
        rotMatrix               = rotMatrix.colRange(0, rotMatrix.cols - 1);//cut out last col of 0s
        Size2i imgSize          = Size2i(img1.rows, img1.cols);
        Mat distcoeffs1         = distCoeffsFromCamInfo(camInfo); 
        Mat distcoeffs2         = distcoeffs1.clone();
        Mat intrinsicK1         = kFromCamInfo(camInfo);
        Mat intrinsicK2         = intrinsicK1.clone();
        Mat R1, R2, P1, P2, Q;
        stereoRectify(intrinsicK1, distcoeffs1,//cam 1
                      intrinsicK2, distcoeffs2,//cam 2
                      imgSize,
                      rotMatrix,
                      transMatrix,
                      R1, R2, P1, P2, Q);//out matrixes

        //triangulate our points
        Mat triout;
        triangulatePoints(P1, P2, img1points, img2points, triout);

        std::vector<glm::vec4> results4d = std::vector<glm::vec4>();
        for(int i = 0; i < triout.cols; i++){
            glm::vec4 next = glm::vec4(triout.at<double>(0, i),
                                       triout.at<double>(1, i),
                                       triout.at<double>(2, i),
                                       triout.at<double>(3, i));
            results4d.push_back(next);
        }//for



        return retval;
    }//pointsFromDuo

    std::vector<Vec3i> getGoodIndexSets(KeyPoint_vec keypoints1, KeyPoint_vec keypoints2, KeyPoint_vec keypoints3,
                                        DMatch_vec match12, DMatch_vec match23, DMatch_vec match31){
        std::vector<Vec3i> goodIndexSets = std::vector<Vec3i>();//for a matching-triangle, indices into the features for (img1, img2, img3)
        //log the matches that are good for all three images 
        for(int i = 0; i < match12.size(); i++){
            DMatch match1 = match12.at(i);
            int img1idx = match1.trainIdx;
            int img2idx = match1.queryIdx;
            for(int j = 0; j < match23.size(); j++){
                DMatch match2 = match23.at(j);
                if (match2.trainIdx = img2idx){
                    int img3idx = match2.queryIdx;
                    for(int k = 0; k < match31.size(); k++){
                        DMatch match3 = match31.at(k);
                        if (match3.trainIdx == img3idx && match3.queryIdx == img1idx){
                            goodIndexSets.push_back(Vec3i(img1idx, img2idx, img3idx));
                            break;//no need to keep searching innermost
                        }//found a triangle-match
                    }//for
                }//if
            }//for
        }//for
        return goodIndexSets;
    }

    std::vector<PointT> pointsFromTrio(Mat img1, Mat img2, Mat img3, tf2::Transform xform1, tf2::Transform xform2, tf2::Transform xform3){

        KeyPoint_vec keypoints1, keypoints2, keypoints3;
        DMatch_vec match12, match23, match31;
        //later development: only calculate key points for the new img, cache the old ones
        getKeyPointMatches(img1, img2, img3, &keypoints1, &keypoints2, &keypoints3, &match12, &match23, &match31);
        Point2f_vec img1Points = Point2f_vec();
        Point2f_vec img2Points = Point2f_vec();
        Point2f_vec img3Points = Point2f_vec();
        std::vector<Vec3i> goodIndexSets = getGoodIndexSets(keypoints1, keypoints2, keypoints3, match12, match23, match31);
        for(int i = 0; i < goodIndexSets.size(); i++){
            img1Points.push_back(keypoints1[goodIndexSets.at(i)[0]].pt);
            img2Points.push_back(keypoints2[goodIndexSets.at(i)[1]].pt);
            img3Points.push_back(keypoints3[goodIndexSets.at(i)[2]].pt);
        }//for
        Mat fundamental21 = findFundamentalMat(img1Points, img2Points, FM_RANSAC);
        Mat fundamental32 = findFundamentalMat(img2Points, img3Points, FM_RANSAC);
        Mat fundamental31 = findFundamentalMat(img1Points, img3Points, FM_RANSAC);

        //make the Eigen matrices to help contructo our Trifocal Tensor
        int numListRows = img1Points.size();
        int numListCols = 3;
        Eigen::MatrixXd img1List(numListRows, numListCols); 
        Eigen::MatrixXd img2List(numListRows, numListCols); 
        Eigen::MatrixXd img3List(numListRows, numListCols); 
        //x'' = (F31 x) * (F32 x') (this is the fundamental matrix mapping, probably don't use) 

        TrifocalTensor tensor = TrifocalTensor();
        tensor.computeTensor(img1List, img2List, img3List);
       
        return std::vector<PointT>();

    }//pointsFromTrio


    std::vector<PointSub> showMatches(Mat img1, Mat img2, tf2::Transform xform1, tf2::Transform xform2){
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
        //-- Draw matches
        Mat img_matches;
        drawMatches( img1, keypoints1, img2, keypoints2, good_matches, img_matches, Scalar::all(-1),
                     Scalar::all(-1), std::vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );
        //-- Show detected matches
        //imshow("Good Matches", img_matches );
        //waitKey(0);
        imwrite("matches.jpg", img_matches);
        std::vector<PointSub> retval = getMatchingWorldPointsAlt(img1, keypoints1, xform1,
                               img2, keypoints2, xform2,
                               good_matches,
                               FoV);
        return retval;
    }//showMatches

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

    void ImageCallbackAlt(const sensor_msgs::Image& msg){
        if (!found1stXform) return;
#if DEBUGIMAGE
        ROS_INFO("===IMAGE===");
#endif
        int width = msg.width;
        int height = msg.height;
        Mat thisImage = Mat(height, width, getEncodingTypeForCV(msg.encoding), (void*) msg.data.data());

        imgqueue.push_back(thisImage.clone());
        xformqueue.push_back(xform);
        if (imgqueue.size() < 6) return;//if first five images, ignore
        else if (imgqueue.size() > 6){
            imgqueue.pop_front();//discard the 7th image back
            xformqueue.pop_front();
        }//else

        tf2::Transform nullxform = tf2::Transform();
        std::vector<PointT> resultPoints = pointsFromDuo(imgqueue[0], imgqueue[5], xformqueue[0], xformqueue[5], caminfo);
        //std::vector<PointT> resultPoints = pointsFromTrio(imgqueue[0], imgqueue[1], imgqueue[2], 
        //                                                  xformqueue[0], xformqueue[1], xformqueue[2]);
    }

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

        if (imageNumber % 6 != 0){
            return;//only process one image every 4 frames
        }//if

        if (prevImage.empty()){
            prevImage = thisImage.clone();
            prevxform = tf2::Transform(xform);
        }
        else{

            std::vector<PointSub> matchedPoints = showMatches(prevImage, thisImage, prevxform, xform);
            for(int i = 0; i < matchedPoints.size(); i++){
                PointSub thisVal = matchedPoints.at(i);
                PointT point = PointT(thisVal.r, thisVal.g, thisVal.b, 0);
                point.x = thisVal.x; point.y = thisVal.y; point.z = thisVal.z;
                pcloud.push_back(point);
            }//for
            //put in fake points for camera positions
            PointT point = PointT(255, 0, 255, 1);//magenta
            tf2::Vector3 viewPosition;
            viewPosition = prevxform.getOrigin();
            point.x = viewPosition[0]; point.y = viewPosition[1]; point.z = viewPosition[2];
            pcloud.push_back(point);
            PointT point2 = PointT(0, 255, 255, 1);//cyan
            viewPosition = xform.getOrigin();
            point2.x = viewPosition[0]; point2.y = viewPosition[1]; point2.z = viewPosition[2];
            pcloud.push_back(point2);

            //put our "view" direction points in
            makeDirectionOffsetPoint(prevxform, 0.015, 0.0, 0.0, 255, 0, 0);//x red
            makeDirectionOffsetPoint(prevxform, 0.0, 0.015, 0.0, 0, 255, 0);//y green
            makeDirectionOffsetPoint(prevxform, 0.0, 0.0, 0.015, 0, 0, 255);//z blue
            makeDirectionOffsetPoint(xform, 0.015, 0.0, 0.0, 255, 0, 0);//x red
            makeDirectionOffsetPoint(xform, 0.0, 0.015, 0.0, 0, 255, 0);//y green
            makeDirectionOffsetPoint(xform, 0.0, 0.0, 0.015, 0, 0, 255);//z blue

            pcl::io::savePCDFile("testOutput.pcd", pcloud);

            prevImage = thisImage.clone();
            prevxform = tf2::Transform(xform);
        }//else



        //imwrite("testOutputImage.jpg", thisImage);

        lastXformTime = xformTime;


        //Output point cloud for debugging
        /*
        if (imageNumber == 80){
            pcl::io::savePCDFile("testOutput.pcd", pcloud);
        }//if
        */

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
        ROS_INFO("\tTranslation:\t<%.05f, %.05ff, %.05f>", xlate.x, xlate.y, xlate.z);
        ROS_INFO("\tRotation:\t<%.05f, %.05f, %.05f, %.05f>", rotate.x, rotate.y, rotate.z, rotate.w);
#endif
    }//TransformCallback2

    void TransformCallback2(const geometry_msgs::TransformStamped& msg){
        int time_secs = msg.header.stamp.sec;
        int time_nsecs = msg.header.stamp.nsec;
        double timeValue = time_secs + (1e-9 * time_nsecs);
        geometry_msgs::Vector3 xlate        = msg.transform.translation;
        geometry_msgs::Quaternion rotate    = msg.transform.rotation;

        tf2::Transform thisxform;

        tf2::fromMsg(msg.transform, thisxform);

#if DEBUGXFORM
        ROS_INFO("===XFORM COLOR===");
        ROS_INFO("\ttime: %f", timeValue);
        ROS_INFO("\tTranslation:\t<%.05f, %.05ff, %.05f>", xlate.x, xlate.y, xlate.z);
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

        //ros::Subscriber cameraSub       = n.subscribe(imageSubPath, 1000, ImageCallback);
        ros::Subscriber cameraSub       = n.subscribe(imageSubPath, 1000, ImageCallbackAlt);
        
        ros::Subscriber cameraInfoSub   = n.subscribe(camInfoSubPath, 1000, CameraInfoCallback);

        ros::Subscriber xformGlobalSub  = n.subscribe(xformSubPath1, 1000, TransformCallback1);
        ros::Subscriber xformColorSub   = n.subscribe(xformSubPath2, 1000, TransformCallback2);

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
