# Drone Mom
## CIS565 Final Project for John Marcao, Eric Micallef, and Taylor Nelms

## Build Instructions

## Do not clone this repo until the instructions tell you to.

you need to build a ROS workspace first...

## Installation

First, install the latest [JetPack](https://developer.nvidia.com/embedded/jetpack) on your Jetson (JetPack 4.2.2 for ROS Melodic or JetPack 3.3 for ROS Kinetic on TX1/TX2).

Once you are logged onto your jetson continue.

### ROS Core

These ROS nodes use the DNN objects from the jetson-inference project (aka Hello AI World). To build and install it, see this page or run the commands below:

```
$ cd ~
$ sudo apt-get install git cmake
$ git clone --recursive https://github.com/dusty-nv/jetson-inference
$ cd jetson-inference
$ mkdir build
$ cd build
$ cmake ../
$ make
$ sudo make install
```

Install the `ros-melodic-ros-base`package on your Jetson following [these](
https://www.stereolabs.com/blog/ros-and-nvidia-jetson-nano/) directions:

or, here are the necessary commands ( information on the commands is in link above )

```bash
$ sudo sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list'
$ sudo apt-key adv --keyserver 'hkp://keyserver.ubuntu.com:80' --recv-key C1CF6E31E6BADE8868B172B4F42ED6FBAB17C654
$ sudo apt update
$ sudo apt install ros-melodic-desktop
$ sudo rosdep init 
$ rosdep update
$ echo "source /opt/ros/melodic/setup.bash" >> ~/.bashrc 
$ source ~/.bashrc
```

For our project we will need some additional nodes. Install the necessary dependencies by running the commands below. This assumees you are running ROS melodic.

```bash
$ sudo apt-get install -y ros-melodic-image-transport
$ sudo apt-get install -y ros-melodic-image-publisher
$ sudo apt-get install -y ros-melodic-vision-msgs
$ sudo apt-get install -y ros-melodic-tf2
$ sudo apt-get install -y ros-melodic-vision-opencv
```

### PCL

Execute the following commands to install the PCL libraries:
```
$ sudo apt-get install -y libpcl-dev
$ sudo apt-get install -y ros-melodic-pcl-ros
```

### OpenCV Advanced features

We are using some of the non-standard features from OpenCV (specifically, SURF feature detection). As such, we need to compile and build OpenCV from source. [This Link](https://linuxize.com/post/how-to-install-opencv-on-ubuntu-18-04/) shows how to do it; alternatively, follow these steps:

```
mkdir ~/opencv_build && cd ~/opencv_build
git clone https://github.com/opencv/opencv.git
git clone https://github.com/opencv/opencv_contrib.git
cd ~/opencv_build/opencv
mkdir build && cd build
cmake -D CMAKE_BUILD_TYPE=RELEASE \
    -D CMAKE_INSTALL_PREFIX=/usr/local \
    -D INSTALL_C_EXAMPLES=ON \
    -D INSTALL_PYTHON_EXAMPLES=ON \
    -D OPENCV_GENERATE_PKGCONFIG=ON \
    -D OPENCV_EXTRA_MODULES_PATH=~/opencv_build/opencv_contrib/modules \
    -D OPENCV_ENABLE_NONFREE=ON ..
make -j4
sudo make install
```
This process may take a little bit to complete.

#### Create Workspace

Now you must make the catkin workspace or your DroneMoM workspace. How ever you like to think about it.

Instructions can be found [here](http://wiki.ros.org/ROS/Tutorials/InstallingandConfiguringROSEnvironment#Create_a_ROS_Workspace):

Or follow these commands. The workspace can be created where ever you are most comfortable below is an example of mine. Please note that catkin looks for the `_ws` so a workspace names `drone_mom_ws` will fail to build.

```bash
$ mkdir -p ~/CIS565/droneMoM_ws/src
$ cd ~/CIS565/droneMoM_ws/
$ catkin_make
$ source devel/setup.bash
$ echo $ROS_PACKAGE_PATH
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

### Application/Framework Resources CURRENTLY UNUSED IGNORE THIS

#### NVIDIA SDK Manager
* [NVidia SDK Manager](https://developer.nvidia.com/embedded/downloads)

This is the platform I plan to use to develop on the Jetson platform

##### Installation Instructions

**NOTE**: Native linux is required for using the NVidia SDK Manager. I tried a workaround with Windows Subsystem Linux, but in the end, this was pretty much infeasible.

1. Initial OS flash on Jetson Nano

    For this, you will need a MicroSD card with the [Jetson Nano Boot Image](https://developer.nvidia.com/jetson-nano-sd-card-image-r3221) installed. Instructions can be had from [here](https://developer.nvidia.com/embedded/learn/get-started-jetson-nano-devkit#write).
    
    You will need to connect it to a monitor, mouse, and keyboard to do the initial Ubuntu setup, once the disk is flashed and inserted into the device

2. Install SDK Manager

    Start up your linux machine, and navigate to the `Downloads` folder.

    Then, execute:
    ```
    sudo apt-get update
    sudo apt-get -y install libgtk-3-0
    sudo apt-get -y install libx11-xcb-dev
    sudo apt-get -y install libxss1
    sudo apt install ./sdkmanager_0.9.14-4964_amd64.deb
    ```

    Then, with the X-server up and running, you can execute:
    ```
    sdkmanager
    ```
    This will start up the NVIDIA SDK Manager

3. Install Drivers and Components

    [This link](https://docs.nvidia.com/sdk-manager/install-with-sdkm-jetson/index.html) describes how to configure the SDK. The notable changes you'll want to make from the default are:

    1. Choose the Jetson Nano hardware platform
    2. Choose to install the Deepstream SDK option, just in case.

    Note that this will take something like 20GB on your computer, so be aware before you start this process.

4. Flash the chip onto the Nano
    
    For this, you'll need to:
    
    1. Have your Jetson Nano connected to your computer with a MicroUSB connector
    2. Have your Jetson Nano connected to your local area network, so the host computer can SSH into it
    3. Know the IP address of your Jetson Nano (`ip addr show`)
    4. Have your Jetson Nano up and running
    
    The SDK Manager will guide you through the process, but it's pretty much a matter of inputting the IP Address, username, and password for your Nano, and then letting the SDK Manager handle the rest.
    
5. Install SDK Components onto Jetson Nano
    
    At some point in the flashing process, the SDK Manager will effectively wipe the contents of your Jetson Nano, and you'll need to go through the initial setup again.
    
    Once you set your username and password again, log in, and find your IP address again, you can select "Install" on the SDK Manager to put all the relevant development drivers onto the Nano.
    
6. Start Developing

    The SDK Manager will install [NSight Eclipse Edition](https://developer.nvidia.com/nsight-eclipse-edition) automatically.
    
    This is the tool we can use to develop on our host machine and run the software on the Jetson device itself. I'm still figuring out the next steps on how to do this. ([This](https://devblogs.nvidia.com/cuda-jetson-nvidia-nsight-eclipse-edition/) seems like a good starting point, though!)

#### Deepstream SDK

* [DeepStream SDK](https://developer.nvidia.com/deepstream-sdk)
  * Potential structure by which to take in video feed input; can do direct camera feed or video file input
  * Possibility of just making use of the very front end of one of their sample pipelines, then implementing our own after that point

I recommend installing this through the [NVIDIA SDK Manager](#NVIDIA-SDK-Manager); the steps in that section should detail how to get it up and running.

#### Downloading the MIT Dataset

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

### References

* [MIT Blackbird Dataset](https://github.com/mit-fast/Blackbird-Dataset)
  * Huge dump of drone data for processing
* [The UZH-FPV Drone Racing Dataset](http://rpg.ifi.uzh.ch/uzh-fpv.html)
  * Another drone data dump; this one focuses on high-speed drone operation in particular
* [EuRoc MAV Dataset](https://projects.asl.ethz.ch/datasets/doku.php?id=kmavvisualinertialdatasets)
  * More Drone footage datasets
* [Github Link with lists of further datasets](https://github.com/youngguncho/awesome-slam-datasets#unmanned-aerial-vehicle)






