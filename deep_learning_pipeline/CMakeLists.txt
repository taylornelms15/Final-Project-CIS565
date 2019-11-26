
cmake_minimum_required(VERSION 2.8.3)
project(ros_deep_learning)

# locate dependencies
find_package(catkin REQUIRED COMPONENTS
  image_transport
  roscpp
  sensor_msgs
  cv_bridge
  vision_msgs
  std_msgs
)

find_package(OpenCV REQUIRED)
find_package(CUDA REQUIRED)
find_package(jetson-utils REQUIRED)
find_package(jetson-inference REQUIRED)

# define catkin package
catkin_package(
    LIBRARIES ros_deep_learning_nodelets
    CATKIN_DEPENDS nodelet roscpp image_transport sensor_msgs
)

# include paths
include_directories(
  ${catkin_INCLUDE_DIRS}
  ${CUDA_INCLUDE_DIRS}
)

set(nvidia_files
	nvidia_files/filesystem.cpp
	nvidia_files/img_write.cpp
	nvidia_files/randInt8Calibrator.cpp
)

set(cudasources
	cuda_utilities/imageNet.cu 
	cuda_utilities/detectNet.cu
  	cuda_utilities/cudaRGB.cu
)

set(detectsources
	src/node_detectnet.cpp
	src/image_converter.cpp 
	inference/ModelImporter.cpp
	inference/ObjectDetection.cpp
)

set(cudaheaders
	cuda_utilites/imageNet.cuh
	cuda_utilities/detectNet.cuh
)

list(SORT cudasources)
list(SORT detectsources)
list(SORT nvidia_files)

#source_group(detector FILES ${cudasources} ${detectsources})

# enable c++11 (TensorRT requirement)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

list(APPEND CUDA_NVCC_FLAGS ${CUDA_GENERATE_CODE})
# sm was 53
set(CUDA_NVCC_FLAGS "-arch=sm_53" CACHE STRING "nvcc flags" FORCE)
set(CUDA_VERBOSE_BUILD ON)
set(LIB_TYPE STATIC)
cuda_add_library(cuda_util ${LIB_TYPE} ${cudasources})

#include_directories(/usr/include/gstreamer-1.0 /usr/lib/aarch64-linux-gnu/gstreamer-1.0/include /usr/include/glib-2.0 /usr/include/libxml2 /usr/lib/aarch64-linux-gnu/glib-2.0/include/)

# inference nodes
#add_executable(imagenet src/node_imagenet.cpp src/image_converter.cpp)
#target_link_libraries(imagenet ${catkin_LIBRARIES} jetson-inference)


add_executable(detectnet ${detectsources} ${nvidia_files})
add_dependencies(detectnet cuda_util)
#add_executable(detectnet ${detectsources})
target_link_libraries(detectnet ${catkin_LIBRARIES} ${OpenCV_LIBS} cuda_util jetson-inference)

#target_link_libraries(detectnet ${catkin_LIBRARIES} ${OpenCV_LIBS} cuda_util nvinfer nvinfer_plugin nvonnxparser nvcaffe_parser jetson-inference)

#opencv_core opencv_calib3d
#add_executable(segnet src/node_segnet.cpp src/image_converter.cpp)
#target_link_libraries(segnet ${catkin_LIBRARIES} jetson-inference)


# the library we are going to compile (has all of the nodelets in it)
#add_library(ros_deep_learning_nodelets src/nodelet_imagenet.cpp src/image_converter.cpp)
#target_link_libraries(ros_deep_learning_nodelets ${catkin_LIBRARIES} jetson-inference)


#if(catkin_EXPORTED_LIBRARIES)
#    add_dependencies(ros_deep_learning_nodelets ${catkin_EXPORTED_LIBRARIES})
#endif()

