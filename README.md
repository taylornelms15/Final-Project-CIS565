# Drone Mom
## CIS565 Final Project for John Marcao, Eric Micallef, and Taylor Nelms

- [Problem Statement](#Problem-Statement)
- [Repo Structure](#Repo-Structure) 
- [Project Overview](#Project-Overview)
- [Design](#Design)
- [Results](#Results)
- [Build Instructions](#Build-Instructions)


## Problem Statement

Collecting 3D object datasets involves a large amount of manual work and is time consuming. Can we build a system that can automate this?

## Repo Structure

This repository is laid out in the following manner. The top level README lays out high level functionality of the system. The separate ROS nodes each have a README that contains more information and performance analysis of the individual components.

## Project Overview (add pictures)

ROS is heavily used in research. We utilized the ROS architecture for our design. This allows us to use ROS bags to replay back our data and refine our algorithms. This also lets other developers interchange ROS components easily. For example, If someone wanted to create their own point cloud node in our system they can easily swap out the point cloud node for theirs as long as they publish the same infromation then nothing in theory should break.



## Design

add picture of pipeline 

The first step in our pipeline is to classify the important objects in a scene. This is done for two reasons. Reason 1 is that we need to be able to give a description of what objects are in a scene. The second reason is that we can give boundingboxes to the point cloud on where to focus in on. Point cloud computation is expensive so this is one way of optimizing. Only generate a point cloud in regions where we are interested in. after the point cloud is generated we can then render a 3d mesh!

## Results

Further information and alysis can be found in the actual folder of the ROS node. Because we are utilizing ROS each subcomponent has its own readme. This is because users usign ROS can drag and drop nodes

## Performance Analysis

Data was collected of running inference on the CPU and GPU FP16 is an optimzation that tensorRT makes where it can turn your FP32 to FP16 types without losing any precision in inference. TensorRT also supports Int8 but this is not supported on the Jetson Nano and as such is not shown. 

![](images/trt_graph.png)


## Build Instructions

### Do not clone this repo until the instructions tell you to.

you need to build a ROS workspace first...

### Installation

First, install the latest [JetPack](https://developer.nvidia.com/embedded/jetpack) on your Jetson (JetPack 4.2.2 for ROS Melodic or JetPack 3.3 for ROS Kinetic on TX1/TX2).

Once you are logged onto your jetson continue.

### ROS Core

Luckily tensorRT comes pre installed on the jetpack. We just need to insteall a few extra plugins for streaming 

```bash
cd ~
sudo apt-get install git cmake
sudo apt-get update
sudo apt-get install -y dialog
sudo apt-get install -y libglew-dev glew-utils libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev libglib2.0-dev
sudo apt-get install -y libopencv-calib3d-dev libopencv-dev 
sudo apt-get update

```

Install the `ros-melodic-ros-base`package on your Jetson following [these](
https://www.stereolabs.com/blog/ros-and-nvidia-jetson-nano/) directions:

or, here are the necessary commands ( information on the commands is in link above )

```bash
sudo sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list'
sudo apt-key adv --keyserver 'hkp://keyserver.ubuntu.com:80' --recv-key C1CF6E31E6BADE8868B172B4F42ED6FBAB17C654
sudo apt update
sudo apt install ros-melodic-desktop
sudo rosdep init 
rosdep update
echo "source /opt/ros/melodic/setup.bash" >> ~/.bashrc 
source ~/.bashrc
```

For our project we will need some additional nodes. Install the necessary dependencies by running the commands below. This assumees you are running ROS melodic.

```bash
sudo apt-get install -y ros-melodic-image-transport ros-melodic-image-publisher ros-melodic-vision-msgs ros-melodic-tf2
```

### PCL

Execute the following commands to install the PCL libraries:

```bash
sudo apt-get install -y libpcl-dev ros-melodic-pcl-ros
```

### OpenCV Advanced features

We are using some of the non-standard features from OpenCV (specifically, SURF feature detection). As such, we need to compile and build OpenCV from source. [This Link](https://linuxize.com/post/how-to-install-opencv-on-ubuntu-18-04/) shows how to do it; alternatively, follow these steps:

```bash
sudo apt-get install -y build-essential cmake git pkg-config libgtk-3-dev libglm-dev
sudo apt-get install -y libavcodec-dev libavformat-dev libswscale-dev libv4l-dev libxvidcore-dev libx264-dev
sudo apt-get install -y libjpeg-dev libpng-dev libtiff-dev gfortran openexr libatlas-base-dev libtbb2 libtbb-dev libdc1394-22-dev
sudo apt-get install -y libeigen3-dev libgflags-dev libgoogle-glog-dev libsuitesparse-dev libatlas-base-dev
```
We need to install the Ceres solver as well before we get openCV up and running:

```bash
git clone https://ceres-solver.googlesource.com/ceres-solver
cd ceres-solver
mkdir build && cd build
cmake ..
make -j4
sudo make install
```
Now we can get all the openCV parts together:

```bash
mkdir ~/opencv_build && cd ~/opencv_build
git clone https://github.com/opencv/opencv.git
git clone https://github.com/opencv/opencv_contrib.git
cd ~/opencv_build/opencv
mkdir build && cd build
cmake -D CMAKE_BUILD_TYPE=RELEASE \
    -D CMAKE_INSTALL_PREFIX=/usr/local \
    -D WITH_CUDA=ON \
    -D ENABLE_FAST_MATH=1 \
    -D CUDA_FAST_MATH=1 \
    -D WITH_CUBLAS=1 \
    -D OPENCV_GENERATE_PKGCONFIG=ON \
    -D OPENCV_EXTRA_MODULES_PATH=~/opencv_build/opencv_contrib/modules \
    -D OPENCV_ENABLE_NONFREE=ON ..
make -j4 && sudo make install
```
This process may take a little bit to complete.

### Create Workspace

Now you must make the catkin workspace or your DroneMoM workspace. How ever you like to think about it.

Instructions can be found [here](http://wiki.ros.org/ROS/Tutorials/InstallingandConfiguringROSEnvironment#Create_a_ROS_Workspace):

Or follow these commands. The workspace can be created where ever you are most comfortable below is an example of mine. Please note that catkin looks for the `_ws` so a workspace names `drone_mom_ws` will fail to build.

```bash
mkdir -p ~/CIS565/droneMoM_ws/src
cd ~/CIS565/droneMoM_ws/
catkin_make
source devel/setup.bash
echo $ROS_PACKAGE_PATH
```

Ensure that the path from the echo output matches your path. Assuming you are running ROS melodic it will look something like this

`/home/youruser/CIS565/droneMoM_ws/src:/opt/ros/melodic/share`

### Clone

Now you can clone this repo into the src folder of your newly crated ROS workspace!

### Build 

navigate to your workspace so `~/CIS565/droneMoM_ws`

and type `catkin_make` This will build everything. Ensure there are no errors. Report to me if there are.

That is it! Now you have ROS running and can make your ROS nodes.

### Test 

open 4 terminals.

This is our roscore terminal it is like a master node ROS can only run with roscore

```bash
source devel/setup.bash
roscore
```

run this last

```bash
source devel/setup.bash
rosrun image_publisher image_publisher __name:=image_publisher ~/CIS565/jetson-inference/data/images/peds_0.jpg 
```

This will take about 5 minutes the first time as it needs to load the neural network
once this node is ready the image publisher can be started

```bash
source devel/setup.bash
rosrun ros_deep_learning detectnet _model_name:=ssd-mobilenet-v2
```

This is a sample app that gets messages fro mthe detectnet

```bash
source devel/setup.bash
rosrun point_cloud point_cloud
```
### Running ROS bag

you will first need to download a ros bag. Download the machine hall 01 ros bag [here](https://projects.asl.ethz.ch/datasets/doku.php?id=iros2018incrementalobjectdatabase)

After downloading the ros bag open 3 terminals and run these commands

```bash
source devel/setup.bash
roscore
```

```bash
cd ~/Downloads
rosbag play <bag you downloaded>
```


```bash
source devel/setup.bash
rosrun object_detection object_detection model_name:=mobilenet_v2
```

If all goes well the first stageo of the pipeline is running!
To see what topics to subscribe to or what is in the bag type in.

```bash
rosbag info <your bag>
```

# References

* [MIT Blackbird Dataset](https://github.com/mit-fast/Blackbird-Dataset)
  * Huge dump of drone data for processing
* [The UZH-FPV Drone Racing Dataset](http://rpg.ifi.uzh.ch/uzh-fpv.html)
  * Another drone data dump; this one focuses on high-speed drone operation in particular
* [EuRoc MAV Dataset](https://projects.asl.ethz.ch/datasets/doku.php?id=kmavvisualinertialdatasets)
  * More Drone footage datasets
* [Github Link with lists of further datasets](https://github.com/youngguncho/awesome-slam-datasets#unmanned-aerial-vehicle)
* [Equation for Ray Closest Intersections](http://morroworks.palitri.com/Content/Docs/Rays%20closest%20point.pdf)
* [Trifocal Tensor Code](https://github.com/cchampet/TrifocalTensor)

# Overview

This ros node takes in input from a ros image message and publishes a custom drone mom message which contains bounding boxes, and depth and color maps for generating a point cloud. For more information on the message published you can see .msg file in the drone_mom_msg folder.

# TensorRT Issues

TensorRT is used to acclerate inference on the GPU. To use TensorRT one must first build the engine. This can be done in may ways using a UFF, Caffe Model or ONNX file. Because ONNX is so new and rapidly evolving I had a huge amount of version issues when attmepting to build a network from ONNX file. 

The easiest and least painful way is to freeze a tensorflow graph and then use nvidias uff converter to converter your tensorflow pb file to a uff file. If you can do this succesfully one huge hurdle is out of the way... Seriously it is painful...

The next step is to build the acclerated structure. One must parse the file and then set parameters accordingly.

# Performance Analysis

Data was collected of running inference on the CPU and GPU FP16 is an optimzation that tensorRT makes where it can turn your FP32 to FP16 types without losing any precision in inference. TensorRT also supports Int8 but this is not supported on the Jetson Nano and as such is not shown. 

![](images/trt_graph.png)

# Credits 

RandInt8Calibrator.cpp and plugin.cpp are from nvidia and are credited as such. They were needed to perform proper calibration and building of tensorRT engines






