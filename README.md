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
## Installation of NVIDIA Drivers for RTX series GPUs
### Graphics Driver
- Download and install NVIDIA Graphics Driver version 418.43 (or newer)
    - best experience with installation via .run file

### CUDA
- Download and install CUDA 10.0. Officially, the new RTX cards require CUDA
  10.1, but this is not supported by tensorflow 1.13
    - download the .run file (do not use the deb package!)
    - install CUDA 10.0 only (!)
    - cuda 10.0 comes with graphics driver 410.xx and wants to install this one, 
      RTX GPUs do not support this! We have already driver 418.43 installed.
      Avoid the installation of driver 410.xx.
    - Do not create any Symlinks as proposed by the installer
    
### cuDNN
- Download cuDNN 7.5.0 for CUDA 10.0. Use the tarball (i do not recommend to
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

# Run the Example Project
### Training and subsequent model export
```
cd ./src
python3 ./model_training.py
```
This script loads the mnist dataset (from keras) and trains the network using tensorflow-gpu.
After the training and the evaluation (you should get an accuracy of about 99.30%), the model
is saved. You should find the following files in the src folder:
- modeldir/checkpoint
- modeldir/mnistModel.data-xxx
- modeldir/mnistModel.index
- modeldir/mnistMode.lmeta
- modeldir/node_names.txt: this file contains the name of the input- and output nodes.
  This information will be important in the subsequent steps

### Generate the frozen graph and save as .pb file
To run the *export_frozen_graph_pb.py* script, you will need to specify the following items:
- model_dir: the path with the exported model
- output_node_names: the name of the output nodes of your network (as saved in node_names.txt)
```
python3 ./export_frozen_graph_pb.py --model_dir ./modeldir --output_node_names output_softmax/Softmax
```
Now you should find this file:
- modeldir/frozen_model.pb

### Load the frozen model for inference
```
python3 ./inference_frozen_model.py --model_path ./modeldir/frozen_graph.pb --output_node_names output_softmax/Softmax \
--input_node_name input_conv_input --picture ../resources/number.jpg
```

If everything went good, you should see the following message:

```
###############################
your number was recognized as 5
###############################
```

### Create the optimized graph model with TensorRT
```
python3 ./create_optimized_trt_graph.py --frozen_path ./modeldir/frozen_model.pb --output_node_names output_softmax/Softmax --saving_path ./modeldir --precision FP16
```
now there sould be a file
- modeldir/newFrozenModel_TRT_FP16.pb

### Test inference with optimized graph
```
python3 ./inference_frozen_model.py --model_path ./modeldir/newFrozenModel_TRT_FP16.pb --output_node_names output_softmax/Softmax \
--input_node_name input_conv_input --picture ../resources/number.jpg
```

If everything went good, you should see the following message:

```
###############################
your number was recognized as 5
###############################
```
   
