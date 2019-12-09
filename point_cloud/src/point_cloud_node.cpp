#include "point_cloud_node.h"
#include <Eigen/Dense>

#define DEBUGOUT 1

#define DEBUGCAMINFO (DEBUGOUT && 0)
#define DEBUGIMAGE (DEBUGOUT && 0)
#define DEBUGXFORM (DEBUGOUT && 0)
#define DEBUGCLOUD (DEBUGOUT && 0)

#define WRITING_PICS 1
#define USING_DRONEMOM_MSG 1

#define ENDFRAMENUM     32
#define SKIPFRAMES      1
#define SYNCFRAMES      16
#define OUTFRAMES       4

#define ALIGNMENT_ITERATIONS 16
#define ALIGNMENT_EPSILON 1e-6
#define ALIGNMENT_DISTMAX 0.12
#define ALIGNMENT_FILTER_SCALE 0.04

//This is what converts depth-space into world space. In reality, USHRT_MAX becomes 4m...maybe
#define SCALING (100.0/65535.0)//this should definitely stay here now, other scaling factors are based off it

//using namespace cv;
//using namespace cv::xfeatures2d;


    const static char dronemomSubPath[] = "detectnet/detections";
    //other paths unused
    const static char imageSubPath[]    = "camera/rgb/image_raw";
    const static char camInfoSubPath[]  = "camera/rgb/camera_info";
    const static char dimageSubPath[]   = "camera/depth/image_raw";
    const static char dcamInfoSubPath[] = "camera/depth/camera_info";
    const static char xformSubPath1[]   = "tango_viwls/T_G_I";//tranformation for global system
    const static char xformSubPathC[]   = "tango/T_I_C_color";//tranformation for color camera
    const static char xformSubPathD[]   = "tango/T_I_C_depth";//tranformation for depth camera

    ros::Publisher* pcloud_pub = NULL;
    
    static sensor_msgs::CameraInfo      caminfo;
    static sensor_msgs::CameraInfo      dcaminfo;

    ///Global point cloud
    static PointT_cloud pcloud       = PointT_cloud(); 
    static PointT_cloud prevcloud    = PointT_cloud();
    static tf2::Transform              firstxform;
    static tf2::Transform              prevxform;
    static tf2::Transform              cumulativeError = tf2::Transform::getIdentity();

    static int                         numMatches  = 0;

    static bool Overlaps(float x, float y, float Right, float Left, float Bottom, float Top){
        return ( (x < Right) && (x > Left) && (y < Top) && (y > Bottom));
    }//Overlaps

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
        for(int i = 50; i < imgCU.rows - 50; i++){//removing edges
            for(int j = 50; j < imgCU.cols - 50; j++){//removing edges
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
        int size = classification.detections.size();



        for (int i = 0; i < colorsCamspace.size(); i++){
            gvec3 color = colorsCamspace.at(i) * factor;
            gvec3 pos   = pointsWorldspace.at(i);
            Point2f imgpt = validPoints[i];
            int label = 0x00;

            for(int j = 0; j < size; j++){
                float center_x = classification.detections[j].bbox.center.x;
                float center_y = classification.detections[j].bbox.center.y;
                float x = classification.detections[j].bbox.size_x;
                float y = classification.detections[j].bbox.size_y;
                float Right = center_x + x * 0.5;
                float Left = center_x - x * 0.5;
                float Bottom = center_y - y * 0.5;
                float Top = center_y + y * 0.5;

                bool rval = Overlaps(imgpt.x, imgpt.y, Right, Left, Bottom, Top);
                if(rval){
                    label = classification.detections[j].results[0].id;
                }//if


            }//for


            PointT nextPoint = PointT(color.r, color.g, color.b, label);
            nextPoint.x = pos.x; nextPoint.y = pos.y; nextPoint.z = pos.z;
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
        //gvec3 factorF = gvec3(1.0, 0.97, 0.97); gvec3 factor = gvec3(1.0, 1.0, 1.0);
        gvec3 factorF = gvec3(1.0, 1.0, 1.0); gvec3 factor = gvec3(1.0, 1.0, 1.0);
        for(int i = 0; i < numMatches; i++){
            factor *= factorF;
        }//for

        //TODO: consider classification stuff here
        PointT_vec retval = createPcloudPoints(validPoints, imgC,
                                               colorsCamspace, pointsWorldspace,
                                               classification,
                                               factor);

        return retval;
    }//pointsFromRGBD


    /**
    Does alignment from the previous frame to the current frame
    Then, updates our cumulative error, applies it to our current frame, and
    adds the incoming (current) frame into the global static pcloud.
    Note: may not actually care about the transformation guesses

    Does not return anything; updates all static variables according to what it good and proper

    @param  incoming            Point cloud from the most recent frame
    @param  bagGuess            The rosbag's guess at camera position for the most recent frame
    @param  previous            Point cloud from the frame before this one
    @param  prevGuess           The rosbag's guess at camera position for the frame before this one
    */
    void accumulatePointCloud(PointT_cloud incoming, tf2::Transform bagGuess,
                              PointT_cloud previous, tf2::Transform prevGuess){
        ROS_INFO("===Accumulating Point cloud for match number %d===", numMatches);

        //align current point cloud to the previous frame
        tf2::Transform curToPrevXform = pairAlign(previous, incoming, prevGuess, bagGuess);

        //update the cumulative error
        cumulativeError = cumulativeError * curToPrevXform;

        //apply the cumulative error to our current frame
        Eigen::Matrix4f errorEigen = transformFromTf2(cumulativeError);
        PointT_cloud transformedCurrent = PointT_cloud();
        pcl::transformPointCloud(incoming, transformedCurrent, errorEigen, true);

        //update our global point cloud with the new frame's points THIS UPDATES THE STATIC CLOUD
        pcloud += transformedCurrent;


        //Make our current frame the "previous" for the next iteration
        prevcloud = incoming;
        prevxform = bagGuess;

    }//accumulatePointCloud

    void syncPointCloud(PointT_cloud incoming, tf2::Transform bagGuess,
                        PointT_cloud cumulative, tf2::Transform totalError){
        ROS_INFO("===Syncing Point cloud for match number %d===", numMatches);

        //Transform our incoming point cloud by our total error so far (for the previous frame)
        Eigen::Matrix4f errorEigen = transformFromTf2(totalError);
        PointT_cloud transformedCurrent = PointT_cloud();
        pcl::transformPointCloud(incoming, transformedCurrent, errorEigen, true);

        //Get the transformation needed to align the fresh cloud to the total cloud
        tf2::Transform curToPrevXform = pairAlign(cumulative, transformedCurrent, totalError, bagGuess);//the transform params don't mean shit

        //transform our incoming frame again to add it to the total
        errorEigen = transformFromTf2(curToPrevXform);
        PointT_cloud extratransformedCurrent = PointT_cloud();
        pcl::transformPointCloud(transformedCurrent, extratransformedCurrent, errorEigen, true);

        //update our global point cloud with the new frame's points THIS UPDATES THE STATIC CLOUD
        pcloud += extratransformedCurrent;

        //make our cumulative error reflect this series of transforms
        cumulativeError = curToPrevXform * totalError;//error is switched because we figured it out the reverse way (I HOPE)
        //cumulativeError = totalError * curToPrevXform;//error is switched because we figured it out the reverse way (I HOPE)

        //Make our current frame the "previous" for the next iteration
        prevcloud = incoming;
        prevxform = bagGuess;


 
    }//syncPointCloud

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
        
        //these lines helped make a "video" to track how things actually moved
        //char filenameC[100]; std::sprintf(filenameC, "color_image%d.png", numMatches);
        //imwrite(filenameC, cImage);

        tf2::Transform thisXform = xformC * xformG;

        //make our direction offset point to track "camera path"
        //makeDirectionOffsetPoint(thisXform, 0, 0, 0, 255, 0, 255);
        //makeDirectionOffsetPoint(thisXform, .01, 0, 0, 255, 0, 0);
        //makeDirectionOffsetPoint(thisXform, 0, .01, 0, 0, 255, 0);
        //makeDirectionOffsetPoint(thisXform, 0, 0, .01, 0, 0, 255);


        //Pass messages on to our point-cloud-making machine
        PointT_vec bunchOfPoints = pointsFromRGBD(cImage, dImage,
            xformC, xformD, xformG,
            ccaminfo, dcaminfo,
            msg->classification);

        //move our retval to our point cloud
        PointT_cloud thisFrameCloud;
        setCloudPoints(thisFrameCloud, bunchOfPoints);
        thisFrameCloud = filterIncoming(thisFrameCloud);

        if (numMatches == 1){
            pcloud      += thisFrameCloud;
            prevcloud   = thisFrameCloud;
            prevxform   = thisXform;
            firstxform  = thisXform;
        }//if
        else if (numMatches % SYNCFRAMES == 0){

            syncPointCloud(thisFrameCloud, thisXform, pcloud, cumulativeError);

            //filter our global cloud at this point to keep point numbers down
            pcloud = filterVoxel(pcloud, ALIGNMENT_FILTER_SCALE / 2.0);

        }//else if syncing our input back to the static cloud
        else{

            accumulatePointCloud(thisFrameCloud, thisXform, prevcloud, prevxform);

        }//else

        #if DEBUGCLOUD
        if (numMatches % OUTFRAMES == 0){
            char filenameC[100]; std::sprintf(filenameC, "cloudsync_%d.pcd", numMatches);
            cloudwrite(filenameC, pcloud);
        }
        #endif
        
        //breakpoint target so we don't fill our point cloud too much
        if (numMatches % ENDFRAMENUM == 0){
            ROS_INFO("Successfully ran %d matches", numMatches);
            publishPointCloud(); 
            //ros::shutdown();
        }//if

        return;

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

        ros::Subscriber dmomSub         = n.subscribe(dronemomSubPath, 200, DetectionCallback);


        //Advertise that we will totally publish something
        ros::Publisher pub = n.advertise<PointT_cloud>("dronemom_pointcloud", 100);
        pcloud_pub = &pub;



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
    Filter an incoming cloud down for alignment (or anything)
    */
    PointT_cloud filterVoxel(const PointT_cloud cloud, double filterscale){
        PointT_cloud_ptr cloud_ptr = makeCloudPtr(cloud);
        pcl::ApproximateVoxelGrid<PointT> sor;
        sor.setInputCloud(cloud_ptr);
        sor.setLeafSize(filterscale, filterscale, filterscale);
        PointT_cloud retval = PointT_cloud();
        sor.filter(retval);
        #if DEBUGCLOUD
        //ROS_INFO("Scale %f\tBeginning points: %zu Ending points:%zu", filterscale, cloud.size(), retval.size());
        #endif
        return retval;

    }//filterVoxel

    PointT_cloud filterOutlier(const PointT_cloud cloud){
        PointT_cloud_ptr cloud_ptr = makeCloudPtr(cloud);
        pcl::StatisticalOutlierRemoval<PointT> sor;
        sor.setInputCloud(cloud_ptr);
        sor.setMeanK(3);
        PointT_cloud retval = PointT_cloud();
        sor.filter(retval);
        #if DEBUGCLOUD
        //ROS_INFO("MeankThresh %.3f\tBeginning points: %zu Ending points:%zu", sor.getStddevMulThresh(), cloud.size(), retval.size());
        #endif
        return retval;

    }//filterShadow

    PointT_cloud filterIncoming(PointT_cloud &incoming){
        PointT_cloud interim = filterVoxel(incoming, ALIGNMENT_FILTER_SCALE / 2.0);//cull some noise out of the system, but keep most of the points 

        //future expansion: try to cull outliers
        PointT_cloud retval = filterOutlier(interim);


        return retval;
    }//filterIncoming


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

    //TODO: get rid of the transform parameters, they're actually irrelevant here
    tf2::Transform pairAlign(const PointT_cloud& cloud_tgt,
                   const PointT_cloud& cloud_src,
                   tf2::Transform& tgtXform,
                   tf2::Transform& srcXform){
        #if DEBUGCLOUD
        //ROS_INFO("BEGINNING ALIGNMENT");
        #endif
        ros::Time begin = ros::Time::now();

        tf2::Transform estimatedT = relativeRotateAndTranslateT(srcXform, tgtXform);
        Eigen::Matrix4f estimated = transformFromTf2(estimatedT);

        //filter input clouds
        PointT_cloud cloud_tgt1 = filterVoxel(cloud_tgt, ALIGNMENT_FILTER_SCALE);
        PointT_cloud cloud_src1 = filterVoxel(cloud_src, ALIGNMENT_FILTER_SCALE);

        PointT_cloud combined = PointT_cloud();
        #if DEBUGCLOUD
        //combined += cloud_src1;
        //combined += cloud_tgt1;
        //cloudwrite("cloud_src.pcd", cloud_src1);
        //cloudwrite("cloud_tgt.pcd", cloud_tgt1);
        //cloudwrite("cloud_comb_1.pcd", combined);
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
        reg.setMaximumIterations(ALIGNMENT_ITERATIONS);
        reg.align(*reg_result);
        Ti = reg.getFinalTransformation();
        tf2::Transform retval = transformFromEigen(Ti); 

        combined.resize(0);
        combined += cloud_tgt1;
        combined += *reg_result;
        #if DEBUGCLOUD
        //cloudwrite("cloud_comb_2.pcd", combined);
        #endif

        ros::Time end = ros::Time::now();
        #if DEBUGCLOUD
        ros::Duration diffTime = end - begin;
        double diffSecs = diffTime.toSec();
        ROS_INFO("\tALIGNMENT FROM %zu TO %zu POINTS TOOK %.3f SECONDS", cloud_src1.size(), cloud_tgt1.size(), diffSecs);
        #endif
        return retval;

    }//pairAlign

    void publishPointCloud(){
        PointT_cloud_ptr msg = makeCloudPtr(pcloud); 
        msg->header.frame_id = "some_point_tf_frame";
        msg->height = 1;
        msg->width = pcloud.size();
        pcl_conversions::toPCL(ros::Time::now(), msg->header.stamp);
        pcloud_pub->publish(msg);

        return;

    }//publishPointCloud


