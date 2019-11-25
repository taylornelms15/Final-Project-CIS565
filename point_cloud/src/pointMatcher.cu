/**
@file pointMatcher.cu
@author Taylor Nelms
*/





#include "pointMatcher.h"


void checkCUDAErrorFn(const char *msg, const char *file, int line) {
    cudaError_t err = cudaGetLastError();
    if (cudaSuccess == err) {
        return;
    }

    fprintf(stderr, "CUDA error");
    if (file) {
        fprintf(stderr, " (%s:%d)", file, line);
    }
    fprintf(stderr, ": line %d: %s: %s\n", line, msg, cudaGetErrorString(err));
    exit(EXIT_FAILURE);
}


#define MINTIME 0.001f
#define DIST_THRESH 0.02f

float* d_A1;
float* d_A2;
float* d_O;

#define DOTP(a,b) (a.x * b.x + a.y * b.y + a.z * b.z)

/**
Ray 1: origin a, direction b
Ray 2: origin c, direction d
@return[0:2]: midpoint of where they come closest
@return[3]: their closest distance from each other
*/
__host__ __device__ glm::vec4 closestRayIntersect(glm::vec3 a, glm::vec3 b, glm::vec3 c, glm::vec3 d){
    //float s = DOTP(b, d) * (DOTP(a, d) - DOTP(b, c)) - DOTP(a, d) * DOTP(c, d) / (DOTP(b, d) * DOTP(b, d) - 1);
    //float t = DOTP(b, d) * (DOTP(c, d) - DOTP(a, d)) - DOTP(b, c) * DOTP(a, b) / (DOTP(b, d) * DOTP(b, d) - 1);
    glm::vec3 ab = c - a;
    float s = (-DOTP(b, d) * DOTP(d, ab) + DOTP(b, ab) * DOTP(d, d)) / (DOTP(b, b) * DOTP(d, d) - DOTP(b, d) * DOTP(b, d));
    float t = (DOTP(b, d) * DOTP(b, ab) - DOTP(d, ab) * DOTP(b, b)) / (DOTP(b, b) * DOTP(d, d) - DOTP(b, d) * DOTP(b, d));

    //glm::vec3 closest1 = a + b * t;
    //glm::vec3 closest2 = c + d * s;
    glm::vec3 closest1 = a + b * s;
    glm::vec3 closest2 = c + d * t;
    float dist = glm::distance(closest1, closest2);
    if (s < MINTIME || t < MINTIME)
        dist = 1000000.0f;//don't accept negative points on the ray!
    glm::vec3 midpoint = 0.5f * (closest1 + closest2);

    return glm::vec4(midpoint, dist);

}//closestRayIntersect

__global__ void multiplyNumbers(float* A1, float* A2, float* O){
    int index = (blockIdx.x * blockDim.x) + threadIdx.x;

    float intermediary = A1[index] * A2[index];
    O[index] = intermediary;


}//multiplyNumbers

/**
This converts from a space where a positive Z value is "forward from the camera"
into the space that the rotation wants for our view vector
*/
tf2::Vector3 zForwardToOrientation(float X, float Y, float Z){
    //tf2::Vector3 Da1 = tf2::Vector3(X1, Y1, Z1);
    //camera moving right, we look towards the -x direction as our Z increases
    //So, for our construction space, o1=(0,0,0), o2=(1,0,0)
    //and "forward" for both is around (0,0,1), with "up" at (0,1,0) and "right" at (1,0,0)
    //we rotate to a world where o1=(0,0,0), o2=(0,0,1)
    //and "forward" for both is (-1,0,0), with "up" at (0,-1,0) and "right" at (0,0,1)
    //-Z, X, Y
    //return tf2::Vector3(-Z, X, Y);
    //return tf2::Vector3(Z, X, Y);
    return tf2::Vector3(X, Y, Z);

}//zForwardToOrientation

PointSub matchTwoPoints(PointSub pt1, 
                        tf2::Transform xform1, 
                        PointSub pt2, 
                        tf2::Transform xform2,
                        float fov,
                        float* distance,//out param
                        int width, int height){
    //make our rotation-only transforms                    
    tf2::Transform rotate1 = tf2::Transform(xform1.getRotation());
    tf2::Transform rotate2 = tf2::Transform(xform2.getRotation());
    //make return value
    PointSub retval = {};
    //get NDC ranging [0:1]
    float percentX1 = pt1.x / width; float percentY1 = pt1.y / height;
    float percentX2 = pt2.x / width; float percentY2 = pt2.y / height;
    //make the range go from -1 to 1
    percentX1 = (percentX1 - 0.5f) * 2.0f; percentY1 = (percentY1 - 0.5f) * 2.0f;
    percentX2 = (percentX2 - 0.5f) * 2.0f; percentY2 = (percentY2 - 0.5f) * 2.0f;
    float aspectRatio = (width * 1.0f) / height;
    percentX1 *= aspectRatio; percentX2 *= aspectRatio;
    //turn these into ray pieces
    float halftan = glm::tan(glm::radians(fov / 2.0f));
    float Xa1 = percentX1 * halftan; float Ya1 = percentY1 * halftan;
    float Xa2 = percentX2 * halftan; float Ya2 = percentY2 * halftan;
    //floats to go into vectors
    float X1 = Xa1; float Y1 = Ya1; float Z1 = 1.0f;
    float X2 = Xa2; float Y2 = Ya2; float Z2 = 1.0f;
    //make direction vector and transform
    tf2::Vector3 Da1 = zForwardToOrientation(X1, Y1, Z1);
    tf2::Vector3 Da2 = zForwardToOrientation(X2, Y2, Z2);
    tf2::Vector3 D1 = rotate1(Da1);
    tf2::Vector3 D2 = rotate2(Da2);
    //make glm vectors for position, direction
    tf2::Vector3 o1 = xform1.getOrigin();
    tf2::Vector3 o2 = xform2.getOrigin();
    glm::vec3 p1 = glm::vec3(o1[0], o1[1], o1[2]);
    glm::vec3 p2 = glm::vec3(o2[0], o2[1], o2[2]);
    glm::vec3 d1 = glm::normalize(glm::vec3(D1[0], D1[1], D1[2]));
    glm::vec3 d2 = glm::normalize(glm::vec3(D2[0], D2[1], D2[2]));
    //get matching point and distance
    glm::vec4 match = closestRayIntersect(p1, d1, p2, d2);
    //fill in our return values
    *distance = match.w;
    retval.x = match.x; retval.y = match.y; retval.z = match.z;
    //average the colors
    retval.r = pt1.r + pt2.r / 2;
    retval.g = pt1.g + pt2.g / 2;
    retval.b = pt1.b + pt2.b / 2;

    //FAKE: return a "view ray direction" position
    //tf2::Vector3 viewDirec = D1 * 0.3f;
    //tf2::Vector3 viewDireca = o1 + viewDirec;
    //retval.x = viewDireca[0]; retval.y = viewDireca[1]; retval.z = viewDireca[2];
    //retval.r = pt1.r; retval.g = pt1.g; retval.b = pt1.b;


    return retval;
}//matchTwoPoints


float testCudaFunctionality(float* arrayOne, float* arrayTwo){

    float O[32];

    cudaMalloc(&d_A1, 32 * sizeof(float));
    cudaMalloc(&d_A2, 32 * sizeof(float));
    cudaMalloc(&d_O, 32 * sizeof(float));
    checkCUDAError("cudaMallocs error");

    dim3 threadsPerBlock(32);
    dim3 blocksPerGrid(1);


    cudaMemcpy(d_A1, arrayOne, 32 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_A2, arrayTwo, 32 * sizeof(float), cudaMemcpyHostToDevice);
    checkCUDAError("cudaMemcpy error");

    multiplyNumbers<<< blocksPerGrid, threadsPerBlock >>>(d_A1, d_A2, d_O);

    cudaMemcpy(O, d_O, 32 * sizeof(float), cudaMemcpyDeviceToHost);

    cudaDeviceSynchronize();

    float result = 0.0f;
    for (int i = 0; i < 32; i++){
        result += O[i]; 
    }//for

    cudaFree(d_A1);
    cudaFree(d_A1);
    cudaFree(d_O);

    return result;
    


}//testCudaFunctionality

/**
Converts an (x,y) coordinate to the relevant RGB value in the image
*/
__host__ Vec3b coordToColor(Mat img, Point2f coord){
    return img.at<Vec3b>(coord);//is it this easy?
    float col = coord.x;
    float row = coord.y;//img.rows - coord.y;//top-down vs bottom-up conversion
    return img.at<Vec3b>((int)(row + 0.5f), (int)(col + 0.5));

}//coordToColor

void getCameraIntrinsicMatrix(
        Mat                     img1,
        std::vector<KeyPoint>   keypoints1,
        tf2::Transform          xform1,
        Mat                     img2,
        std::vector<KeyPoint>   keypoints2,
        tf2::Transform          xform2,
        std::vector<DMatch>     good_matches,
        Mat*                    output
){
    Mat retval = Mat();
    tf2::Vector3 o1 = xform1.getOrigin();
    tf2::Vector3 o2 = xform2.getOrigin();
    tf2::Quaternion r1 = xform1.getRotation();
    tf2::Quaternion r2 = xform2.getRotation();
    std::vector<Point2f> img1Points = std::vector<Point2f>();
    std::vector<Point2f> img2Points = std::vector<Point2f>();
    printf("==MATCHES==\n");
    for (int i = 0; i < good_matches.size(); i++){
        KeyPoint img1pt = keypoints1[good_matches[i].trainIdx];
        KeyPoint img2pt = keypoints2[good_matches[i].queryIdx];
        img1Points.push_back(img1pt.pt);
        img2Points.push_back(img2pt.pt);
    }
    Mat fundamental = findFundamentalMat(img1Points, img2Points, FM_RANSAC);
    Mat H1, H2;
    stereoRectifyUncalibrated(img1Points, img2Points, fundamental, Size(img1.cols, img1.rows), H1, H2);


    *output = Mat(retval);
}

std::vector<PointSub> getMatchingWorldPointsAlt(
        Mat                     img1,
        std::vector<KeyPoint>   keypoints1,
        tf2::Transform          xform1,
        Mat                     img2,
        std::vector<KeyPoint>   keypoints2,
        tf2::Transform          xform2,
        std::vector<DMatch>     good_matches,
        float                   FoV)
{
    std::vector<PointSub> retval = std::vector<PointSub>();
    std::vector<Point2f> img1Points = std::vector<Point2f>();
    std::vector<Point2f> img2Points = std::vector<Point2f>();
    printf("==MATCHES==\n");
    for (int i = 0; i < good_matches.size(); i++){
        KeyPoint img1pt = keypoints1[good_matches[i].trainIdx];
        KeyPoint img2pt = keypoints2[good_matches[i].queryIdx];
        img1Points.push_back(img1pt.pt);
        img2Points.push_back(img2pt.pt);
    }
    Mat fundamental = findFundamentalMat(img1Points, img2Points, FM_RANSAC);
    Mat H1, H2;
    stereoRectifyUncalibrated(img1Points, img2Points, fundamental, Size(img1.cols, img1.rows), H1, H2);
    //std::vector<Vec3f> lines1, lines2;
    //computeCorrespondEpilines(img1Points, 1, fundamental, lines1);
    //computeCorrespondEpilines(img2Points, 2, fundamental, lines2);


    return retval;
}//getMathingWorldPointsAlt

std::vector<PointSub> getMatchingWorldPoints(
        Mat                     img1,
        std::vector<KeyPoint>   keypoints1,
        tf2::Transform          xform1,
        Mat                     img2,
        std::vector<KeyPoint>   keypoints2,
        tf2::Transform          xform2,
        std::vector<DMatch>     good_matches,
        float                   FoV)
{
    std::vector<PointSub> retval = std::vector<PointSub>();
    //Keypoints: given in (x, y) coordinates (scaled as pixels, origin bottom left (likely)
    //good_matches: given in (query, train) pairs: indices of the keypoints1 and keypoints2 entries

    std::vector<PointSub> img1Points = std::vector<PointSub>();//not the actual world points
    std::vector<PointSub> img2Points = std::vector<PointSub>();//not the actual world points
    printf("==MATCHES==\n");
    for (int i = 0; i < good_matches.size(); i++){
        KeyPoint img1pt = keypoints1[good_matches[i].trainIdx];
        KeyPoint img2pt = keypoints2[good_matches[i].queryIdx];
        PointSub pt1, pt2;
        pt1.x = img1pt.pt.x; pt1.y = /*img1.rows -*/ img1pt.pt.y;
        pt2.x = img2pt.pt.x; pt2.y = /*img1.rows -*/ img2pt.pt.y;
        Vec3b col1 = coordToColor(img1, img1pt.pt);
        Vec3b col2 = coordToColor(img2, img2pt.pt);
        pt1.b = col1[0]; pt1.g = col1[1]; pt1.r = col1[2];
        pt2.b = col2[0]; pt2.g = col2[1]; pt2.r = col2[2];
        img1Points.push_back(pt1);
        img2Points.push_back(pt2);
    }

    int width = img1.cols;
    int height = img1.rows;

    float totalDist = 0.0f;

    for(int i = 0; i < img1Points.size(); i++){
        float distance;
        PointSub resultMatch = matchTwoPoints(img1Points.at(i), xform1,
                                              img2Points.at(i), xform2,
                                              FoV,
                                              &distance,
                                              width, height);
        if (distance < DIST_THRESH){
            retval.push_back(resultMatch); 
            totalDist += distance;
        }
    }//for

    printf("\tAvg distance %f\n", totalDist / retval.size());



    return retval;
}//getMatchingWorldPoints














