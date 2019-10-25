# Onboard UAV Photogrammetry on NVIDIA Jetson Platform
## CIS565 Final Project for John Marcao, Eric Micallef, and Taylor Nelms

### Application/Framework Resources

#### NVIDIA SDK Manager
* [NVidia SDK Manager](https://developer.nvidia.com/embedded/downloads)

This is the platform I plan to use to develop on the Jetson platform

##### Installation Instructions

1. (Windows) Set up Ubuntu Subsystem

    It only runs on Linux, so if you're on Windows, I recommend downloading Ubuntu from the Microsoft Store.

    There is a lovely guide for how to set up the relevane X server [here](https://www.howtogeek.com/261575/how-to-run-graphical-linux-desktop-applications-from-windows-10s-bash-shell/). This is necessary on Windows; otherwise, there is no way for the Ubuntu subsystem to spin up a display on your computer.

2. Install SDK Manager

    Start up Ubuntu, and navigate to the `Downloads` folder.

    Then, execute:
    ```
    sudo apt-get update
    sudo apt-get install libgtk-3-0
    sudo apt-get install libx11-xcb-dev
    sudo apt-get install libxss1
    sudo apt-get install libxss1
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

4. (Windows) Flash Image to Board Separately
    I haven't gotten the stuff together to make this work, but the next step (after install fails) may be to run the following commands:
    ```
    cd ~/nvidia/nvidia_sdk/JetPack_4.2.2_Linux_GA_P3448/Linux_for_Tegra
    sudo ./flash.sh -S 29318MiB jetson-nano-qspi mmcblk0p1
    ```
    This document is definitely a work in progress.

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
