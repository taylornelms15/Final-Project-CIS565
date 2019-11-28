# Drone Mom
## CIS565 Final Project for John Marcao, Eric Micallef, and Taylor Nelms

- [Problem Statement](#Problem-Statement)
- [Repo Structure](#Repo-Structure) 
- [Build Instructions](#Build-Instructions)


## Problem Statement

## Build Instructions

### Do not clone this repo until the instructions tell you to.

you need to build a ROS workspace first...

### Installation

First, install the latest [JetPack](https://developer.nvidia.com/embedded/jetpack) on your Jetson (JetPack 4.2.2 for ROS Melodic or JetPack 3.3 for ROS Kinetic on TX1/TX2).

Once you are logged onto your jetson continue.

### ROS Core

These ROS nodes use the DNN objects from the jetson-inference project (aka Hello AI World). To build and install it, see this page or run the commands below:

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
sudo apt-get install -y build-essential cmake git pkg-config libgtk-3-dev
sudo apt-get install -y libavcodec-dev libavformat-dev libswscale-dev libv4l-dev libxvidcore-dev libx264-dev
sudo apt-get install -y libjpeg-dev libpng-dev libtiff-dev gfortran openexr libatlas-base-dev libtbb2 libtbb-dev libdc1394-22-dev

mkdir ~/opencv_build && cd ~/opencv_build
git clone https://github.com/opencv/opencv.git
git clone https://github.com/opencv/opencv_contrib.git
cd ~/opencv_build/opencv
mkdir build && cd build
cmake -D CMAKE_BUILD_TYPE=RELEASE \
    -D CMAKE_INSTALL_PREFIX=/usr/local \
    -D OPENCV_GENERATE_PKGCONFIG=ON \
    -D OPENCV_EXTRA_MODULES_PATH=~/opencv_build/opencv_contrib/modules \
    -D OPENCV_ENABLE_NONFREE=ON ..
make -j4
sudo make install
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

you will first need to download a ros bag. Download the machine hall 01 ros bag [here](https://projects.asl.ethz.ch/datasets/doku.php?id=kmavvisualinertialdatasets)

another good one could be this [one](https://projects.asl.ethz.ch/datasets/doku.php?id=iros2018incrementalobjectdatabase)

but lets worry about one thing at a time...

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
rosrun point_cloud point_cloud
```

you should see the point could ros node print data as well as the bag. to see what topics to subscribe to or what is in the bag type in.

```bash
rosbag info <your bag>
```

### Downloading the MIT Dataset 

I've set up a python script for downloading a subsection of the mit dataset. You may need to run the following commands to get it to run:
```
sudo apt-get install pip3
pip3 install wget
```

To use it, navigate to a subfolder of the [MIT Blackbird Dataset Download Site](http://blackbird-dataset.mit.edu/BlackbirdDatasetData/) that you wish to install, and copy the URL of that folder. Then you can run:

```
python3 downloadMitSubfolder.py [-o OutputdirectoryName] URL_Path_To_Recursively_Download
```

This will download all the relevant files to a given directory root. Example usage:

```
python3 downloadMitSubfolder.py http://blackbird-dataset.mit.edu/BlackbirdDatasetData/clover/yawConstant/maxSpeed2p0/ -o../Datasets/Clover/maxSpeed2p0/
```

# References

* [MIT Blackbird Dataset](https://github.com/mit-fast/Blackbird-Dataset)
  * Huge dump of drone data for processing
* [The UZH-FPV Drone Racing Dataset](http://rpg.ifi.uzh.ch/uzh-fpv.html)
  * Another drone data dump; this one focuses on high-speed drone operation in particular
* [EuRoc MAV Dataset](https://projects.asl.ethz.ch/datasets/doku.php?id=kmavvisualinertialdatasets)
  * More Drone footage datasets
* [Github Link with lists of further datasets](https://github.com/youngguncho/awesome-slam-datasets#unmanned-aerial-vehicle)






