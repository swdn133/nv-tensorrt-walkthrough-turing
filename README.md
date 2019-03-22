This repo provides a simple example to manage the first steps of Machine Learning
on embedded devices like Nvidia's Jetson modules. This tutorial foremost describes the
toolchain to work with Nvidia's latest Turing devices. 
Typically the workflow contains several steps:
- Training the Artificial Neural Network on a Host-PC/Server with much power
- Freeze and export the trained model
- Optimize the exported model for the inference on the target architecture
- Transfer the optimized model on the target device
- Perform inference on the target device (Jetson)

This repository describes the required installation process (to setup the toolchain
on a new PC) and provides a sipmle project (simple mnist handwritten character
recognition) to get familiar with the workflow (and test your installation).
# Installation Instructions
This first section describes the installation of all the tools required to run
the process. The process is tested for the target OS Ubuntu 18.04
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
