#include "point_cloud_node.h"
#include <Eigen/Dense>

#define DEBUGOUT 1

#define DEBUGCAMINFO (DEBUGOUT && 0)
#define DEBUGIMAGE (DEBUGOUT && 1)
#define DEBUGXFORM (DEBUGOUT && 0)
#define DEBUGCLOUD (DEBUGOUT && 1)

#define WRITING_PICS 1
#define USING_DRONEMOM_MSG 1

#define ENDFRAMENUM 2
#define SKIPFRAMES  1

#define ALIGNMENT_ITERATIONS 10
#define ALIGNMENT_EPSILON 1e-6
#define ALIGNMENT_DISTMAX 0.1
#define ALIGNMENT_FILTER_SCALE 0.035

//This is what converts depth-space into world space. In reality, USHRT_MAX becomes 4m
#define SCALING (100.0/65535.0)//1/256 also seemed close, so we could be way off the mark
//#define SCALING (1000.0 / 65535)

//using namespace cv;
//using namespace cv::xfeatures2d;


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
    static PointT_cloud pcloud       = PointT_cloud(); 
    static PointT_cloud prevcloud    = PointT_cloud();
    static tf2::Transform          prevxform;

    static int                          numMatches  = 0;


    /**
    Gets the relative transform between two transforms in terms of a tf2::Transform
    */
    tf2::Transform relativeRotateAndTranslateT(tf2::Transform xform1, tf2::Transform xform2){
        tf2::Vector3 transDiff      = xform2.getOrigin() - xform1.getOrigin();
        tf2::Quaternion rotDiff     = xform2.getRotation() * xform1.getRotation().inverse();
        tf2::Transform xDiff        = tf2::Transform(rotDiff, transDiff);
        return xDiff;
    }//relativeRotateAndTranslateT

    /**
    Gets the relative transform between two transforms in terms of floats
    */
    double_vec relativeRotateAndTranslate(tf2::Transform xform1, tf2::Transform xform2){
        tf2::Transform xDiff        = relativeRotateAndTranslateT(xform1, xform2);
        tf2Scalar raw[15];
        xDiff.getOpenGLMatrix(raw);//first 12 elements rotation, last 3 elements translation
        double_vec retval(15);
        for (int i = 0; i < 15; i++){
            retval[i] = raw[i];
        }//for
        return retval;

    }//relativeRotateAndTranslate

    /**
    Same as above, but puts the results into a cv::Mat like opencv wants it
    */
    cv::Mat relativeRotateAndTranslateM(tf2::Transform xform1, tf2::Transform xform2){
        double_vec raw = relativeRotateAndTranslate(xform1, xform2);
        cv::Mat retval = cv::Mat(4, 4, CV_64F);
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
    cv::Mat distCoeffsFromCamInfo(sensor_msgs::CameraInfo camInfo){
        cv::Mat retval = cv::Mat(1, 5, CV_64F);
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
    cv::Mat kFromCamInfo(sensor_msgs::CameraInfo camInfo){
        cv::Mat retval = cv::Mat(3, 3, CV_64F);
        for(int i = 0; i < 3; i++){
            for (int j = 0; j < 3; j++){
                retval.at<double>(i, j) = camInfo.K[3 * i + j];
            }
        }
        return retval;
    }//kFromCamInfo


    cv::Mat undistortColor(const cv::Mat& orig, sensor_msgs::CameraInfo camInfoC){

        cv::Mat retval;
        cv::Mat distcoeffs1         = distCoeffsFromCamInfo(camInfoC); 
        cv::Mat intrinsicK1         = kFromCamInfo(camInfoC);

        undistort(orig, retval, intrinsicK1, distcoeffs1);
        return retval;

    }//undistortColor

    Point2f_vec getValidPoints(cv::Mat imgCU, cv::Mat depthMat){
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
                                        const cv::Mat imgCU, const cv::Mat depthMat,
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

    /**
    //TODO: put the classification info in here
    @param validPoints      Vector of cv::Point2f points that describe the locations on the image we're projecting into space
    @param imgC             The cv::Mat representing the input color image
    @param colorsCamspace   The colors from the image for each point (same size as validPoints; already sampled)
    @param pointsWorldspace The points' coordinates in world space (same size as validPoints)
    @param classification   ??????
    @param factor           Color-scaling factor (was useful for debugging temporal information)

    @return                 Vector of PointT objects representing the points' positions, colors, and classification labels
    */
    PointT_vec createPcloudPoints(Point2f_vec validPoints, cv::Mat imgC,
                                  gvec3_vec colorsCamspace, gvec3_vec pointsWorldspace,
                                  vision_msgs::Detection2DArray classification,
                                  gvec3 factor = gvec3(1.0f, 1.0f, 1.0f)){
        PointT_vec retval = PointT_vec();
        for (int i = 0; i < colorsCamspace.size(); i++){
            gvec3 color = colorsCamspace.at(i) * factor;
            gvec3 pos   = pointsWorldspace.at(i);
            //TODO: put the right classification index into the back end of this constructor
            PointT nextPoint = PointT(color.r, color.g, color.b, 0);
            nextPoint.x = pos.x; nextPoint.y = pos.y; nextPoint.z = pos.z;
            //pcloud.push_back(nextPoint);//this would put the new points into the global point cloud
            retval.push_back(nextPoint);
        }//for

        return retval;
    }

    PointT_vec pointsFromRGBD(cv::Mat imgC, cv::Mat imgD,
                        tf2::Transform xformC,//color
                        tf2::Transform xformD,//depth
                        tf2::Transform xformG,//global
                        sensor_msgs::CameraInfo camInfoC,
                        sensor_msgs::CameraInfo camInfoD,
                        vision_msgs::Detection2DArray classification
                        ){
        if (numMatches % SKIPFRAMES != 0){
            return PointT_vec();
        }//if
        ROS_INFO("===MATCHED IMAGES CALLBACK===");

        cv::Mat rbt = relativeRotateAndTranslateM(xformD, xformC);
        cv::Mat dC  = distCoeffsFromCamInfo(camInfoC);
        cv::Mat kC  = kFromCamInfo(camInfoC);
        cv::Mat kD  = kFromCamInfo(camInfoD);

        //match our d image to our rgb image
        cv::Mat depthMat;
        cv::rgbd::registerDepth(kD, kC, dC, rbt, imgD, imgC.size(), depthMat);

        //Undistort our color and depth map images
        cv::Mat kCinv           = kC.inv();
        cv::Mat imgCU           = undistortColor(imgC, camInfoC);//color image undistorted
        cv::Mat imgDMU          = undistortColor(depthMat, camInfoC);//depth map undistorted
        
        //get our list of valid points
        Point2f_vec validPoints = getValidPoints(imgCU, depthMat);

        //undistort our points: want the projection to be for the most accurate camera projection
        Point2f_vec validPointsU;
        undistortPoints(validPoints, validPointsU, kC, dC, cv::noArray(), kC);

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

        //TODO: consider classification stuff here
        PointT_vec retval = createPcloudPoints(validPoints, imgC,
                                               colorsCamspace, pointsWorldspace,
                                               classification,
                                               factor);

        //breakpoint target so we don't fill our point cloud too much
        if (numMatches % ENDFRAMENUM == 0){
            pcl::io::savePCDFile("testOutput.pcd", pcloud);
            ROS_INFO("Successfully ran %d matches", numMatches);
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
        cv::Mat dImage = cv::Mat(dheight, dwidth, getEncodingTypeForCV(msg->depth_image.encoding), 
                         (void*) msg->depth_image.data.data(), msg->depth_image.step);
        int cwidth = msg->rgb_image.width;
        int cheight = msg->rgb_image.height;
        cv::Mat cImage = cv::Mat(cheight, cwidth, getEncodingTypeForCV(msg->rgb_image.encoding), 
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
            ccaminfo, dcaminfo,
            msg->classification);

        if (numMatches == 1){
            //move our retval to our point cloud
            setCloudPoints(prevcloud, bunchOfPoints);
            prevxform = thisXform;
        }//if first frame
        else{
            //TODO: try some point cloud alignment
            PointT_cloud cloudIn = PointT_cloud();
            setCloudPoints(cloudIn, bunchOfPoints);
            tf2::Transform requiredXform  = pairAlign(prevcloud, cloudIn,
                                                      prevxform, thisXform);
            ros::shutdown();
        }//else, align some stuff


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
    // Point Cloud alignment
    //########################################################

    /**
    Filter an incoming cloud down for alignment
    */
    PointT_cloud filterVoxel(const PointT_cloud cloud){
        PointT_cloud_ptr cloud_ptr = makeCloudPtr(cloud);
        pcl::ApproximateVoxelGrid<PointT> sor;
        sor.setInputCloud(cloud_ptr);
        sor.setLeafSize(ALIGNMENT_FILTER_SCALE, ALIGNMENT_FILTER_SCALE, ALIGNMENT_FILTER_SCALE);
        PointT_cloud retval = PointT_cloud();
        sor.filter(retval);
        ROS_INFO("Beginning points: %zu\tEnding points:%zu", cloud.size(), retval.size());
        return retval;

    }//filterVoxel

    ///Just a dumb little function renaming
    void cloudwrite(const char filename[], const PointT_cloud cloud){
        pcl::io::savePCDFile(filename, cloud);
    }//outputCloud

    Eigen::Matrix4f     transformFromTf2(const tf2::Transform xform){
        tf2::Stamped<tf2::Transform> xformstamped;
        xformstamped.setData(xform);
        geometry_msgs::TransformStamped intermediate = tf2::toMsg(xformstamped);
        Eigen::Matrix4f retval = tf2::transformToEigen(intermediate).matrix().cast<float>();
        return retval;
    }//transformFromTf2


    tf2::Transform      transformFromEigen(const Eigen::Matrix4f xform){
        Eigen::Affine3d xformAffine = Eigen::Affine3d(xform.cast<double>());
        geometry_msgs::TransformStamped intermediate = tf2::eigenToTransform(xformAffine);
        tf2::Transform retval;
        tf2::fromMsg(intermediate.transform, retval);
        return retval;
    }//transformFromEigen

    tf2::Transform pairAlign(const PointT_cloud& cloud_tgt,
                   const PointT_cloud& cloud_src,
                   tf2::Transform& tgtXform,
                   tf2::Transform& srcXform){
        ROS_INFO("BEGINNING ALIGNMENT");//TODO: allow for down-filtering to speed up computation

        tf2::Transform estimatedT = relativeRotateAndTranslateT(srcXform, tgtXform);
        Eigen::Matrix4f estimated = transformFromTf2(estimatedT);

        //filter input clouds
        PointT_cloud cloud_tgt1 = filterVoxel(cloud_tgt);
        PointT_cloud cloud_src1 = filterVoxel(cloud_src);

        PointT_cloud combined = PointT_cloud();
        #if DEBUGCLOUD
        combined += cloud_src1;
        combined += cloud_tgt1;
        cloudwrite("cloud_src.pcd", cloud_src1);
        cloudwrite("cloud_tgt.pcd", cloud_tgt1);
        cloudwrite("cloud_comb_1.pcd", combined);
        #endif

        //make pointers of our clouds
        PointT_cloud_ptr tgtptr = makeCloudPtr(cloud_tgt1);
        PointT_cloud_ptr srcptr = makeCloudPtr(cloud_src1);
        
        
        //Alignment
        pcl::IterativeClosestPointNonLinear<PointT, PointT> reg;
        reg.setTransformationEpsilon(ALIGNMENT_EPSILON);
        reg.setMaxCorrespondenceDistance(ALIGNMENT_DISTMAX);
        boost::shared_ptr<const pcl::DefaultPointRepresentation<PointT> > pointRepPtr;
        pointRepPtr = boost::make_shared<const pcl::DefaultPointRepresentation<PointT> >(pcl::DefaultPointRepresentation<PointT>() );
        reg.setPointRepresentation(pointRepPtr);

        reg.setInputTarget(tgtptr);
        reg.setInputSource(srcptr);

        //give initial xform estimate (already doing this)

        //run optimization
        Eigen::Matrix4f Ti;
        PointT_cloud_ptr reg_result = srcptr;
        reg.setMaximumIterations(ALIGNMENT_ITERATIONS);//???
        reg.align(*reg_result);
        Ti = reg.getFinalTransformation();
        tf2::Transform retval = transformFromEigen(Ti); 

        combined.resize(0);
        combined += cloud_tgt1;
        combined += *reg_result;
        #if DEBUGCLOUD
        cloudwrite("cloud_comb_2.pcd", combined);
        #endif

        ROS_INFO("ENDING ALIGNMENT");
        return retval;

    }




    //########################################################
    // Old/unused functions (for archiving and maybe revival)
    //########################################################

