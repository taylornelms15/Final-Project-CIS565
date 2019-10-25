# Onboard UAV Photogrammetry on NVIDIA Jetson Platform
## CIS565 Final Project for John Marcao, Eric Micallef, and Taylor Nelms

### Application/Framework Resources

#### NVIDIA SDK Manager
* [NVidia SDK Manager](https://developer.nvidia.com/embedded/downloads)

This is the platform I plan to use to develop on the Jetson platform

##### Installation Instructions

It only runs on Linux, so if you're on Windows, I recommend downloading Ubuntu from the Microsoft Store.

Start up Ubuntu, and navigate to the `Downloads` folder.

Then, execute:
```
sudo apt-get update
sudo apt install libgtk-3-0
sudo apt install ./sdkmanager_0.9.14-4964_amd64.deb
```

#### Deepstream SDK

* [DeepStream SDK](https://developer.nvidia.com/deepstream-sdk)
  * Potential structure by which to take in video feed input; can do direct camera feed or video file input
  * Possibility of just making use of the very front end of one of their sample pipelines, then implementing our own after that point

### References

