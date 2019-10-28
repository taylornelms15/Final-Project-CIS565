# Onboard UAV Photogrammetry on NVIDIA Jetson Platform
## CIS565 Final Project for John Marcao, Eric Micallef, and Taylor Nelms

### Application/Framework Resources

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
    
    This is the tool we can use to develop on our host machine and run the software on the Jetson device itself. I'm still figuring out the next steps on how to do this.

#### Deepstream SDK

* [DeepStream SDK](https://developer.nvidia.com/deepstream-sdk)
  * Potential structure by which to take in video feed input; can do direct camera feed or video file input
  * Possibility of just making use of the very front end of one of their sample pipelines, then implementing our own after that point

I recommend installing this through the [NVIDIA SDK Manager](#NVIDIA-SDK-Manager); the steps in that section should detail how to get it up and running.

### References

* [MIT Blackbird Dataset](https://github.com/mit-fast/Blackbird-Dataset)
  * Huge dump of drone data for processing
* [The UZH-FPV Drone Racing Dataset](http://rpg.ifi.uzh.ch/uzh-fpv.html)
  * Another drone data dump; this one focuses on high-speed drone operation in particular
