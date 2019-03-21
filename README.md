# Installation Instructions
## Installation of NVIDIA Drivers for RTX 2060
- Download and install NVIDIA Graphics Driver version 418.43 (or newer)
    - best experience with installation via .run file
- Download and install CUDA 10.0. Officially, the new RTX cards require CUDA
  10.1, but this is not supported by tensorflow 1.13
    - download the .run file (do not use the deb package!)
    - install CUDA 10.0 only (!)
    - cuda 10.0 comes with graphics driver 410.xx and wants to install this one, 
      RTX GPUs do not support this! We have already driver 418.43 installed.
      Avoid the installation of driver 410.xx.
    - Do not create any Symlinks as proposed by the installer
- Download cudnn 7.5.0 for CUDA 10.0. Use the tarball (i do not recommend to
  use the deb package). Extract and copy the files as described in the NVIDIA
  manual

## Installation of Anaconda
 - Install the latest Anaconda release
 - Create a new environment using Python 3.6
 - activate this environment
 - install required packages
 
```
 pip install tensorflow-gpu
 pip install keras
 pip install matplotlib
```

 ## Installation of TensorRT 5.0 GA
 - 5.0 is the version that is supported by JetPack for the Jetson TX2
 - 5.1 has no support for CUDA 10.0
 - Installation via local repo packages without any problems
 - For installation of the tensorrt python utils it is recommended to 
   download the tar package and extract the required .whl files. Install
   these .whl files with pip in your anaconda environment.

