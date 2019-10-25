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

#### Deepstream SDK

* [DeepStream SDK](https://developer.nvidia.com/deepstream-sdk)
  * Potential structure by which to take in video feed input; can do direct camera feed or video file input
  * Possibility of just making use of the very front end of one of their sample pipelines, then implementing our own after that point

##### Installation Instructions

This can be downloaded and installed through the [NVIDIA SDK Manager](#NVIDIA-SDK-Manager). In your initial download/setup of

### References

* [MIT Blackbird Dataset](https://github.com/mit-fast/Blackbird-Dataset)
  * Huge dump of drone data for processing
* [The UZH-FPV Drone Racing Dataset](http://rpg.ifi.uzh.ch/uzh-fpv.html)
  * Another drone data dump; this one focuses on high-speed drone operation in particular
