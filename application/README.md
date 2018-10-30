# Reference Implementation: Store Traffic Monitor

| Details           |              |
|-----------------------|---------------|
| Target OS:            |  Ubuntu\* 16.04 LTS   |
| Programming Language: |  C++ |
| Time to Complete:    |  50-70min     |

![store-traffic-monitor](../docs/images/store-traffic-monitor-image.png)

An application capable of detecting objects on any number of screens.

## What it Does
This application is one of a series of IoT reference implementations aimed at instructing users on how to develop a working solution for a particular problem. It demonstrates how to create a smart video IoT solution using Intel® hardware and software tools. This reference implementation monitors people activity inside and outside a facility as well as counts product inventory.

## How it Works
The counter uses the Inference Engine included in the Intel® Distribution of OpenVINO™ toolkit. A trained neural network detects objects within a designated area by displaying a green bounding box over them. This reference implementation identifies multiple intruding objects entering the frame and identifies their class, count, and time entered.

## Requirements
### Hardware
* 6th Generation Intel® Core™ processor with Intel® Iris® Pro graphics or Intel® HD Graphics

### Software
* [Ubuntu\* 16.04 LTS](http://releases.ubuntu.com/16.04/)
*Note*: We recommend using a 4.14+ Linux* kernel with this software. Run the following command to determine your kernel version:

```
uname -a
```
* OpenCL™ Runtime Package
* Intel® Distribution of OpenVINO™ toolkit

## Setup

### Install Intel® Distribution of OpenVINO™ toolkit
Refer to https://software.intel.com/en-us/articles/OpenVINO-Install-Linux on how to install and setup the Intel® Distribution of OpenVINO™ toolkit.

You will need the OpenCL™ Runtime Package if you plan to run inference on the GPU as shown by the instructions below. It is not mandatory for CPU inference.

### ffmpeg
ffmpeg is installed separately from the Ubuntu repositories:
```
sudo apt update
sudo apt install ffmpeg
```

## Build the application

To build, open a terminal in the `application` folder and run the following commands:

```
source env.sh
mkdir -p build && cd build
cmake ..
make
```


## Configure the application

### What model to use
The application works with any object-detection model, provided it has the same input and output format of the SSD model.  
The model can be any object detection model:
* that is provided by Intel® Distribution of OpenVINO™ toolkit.  
   These can be found in the `deployment_tools/intel_models` folder.
* downloaded using the **model downloader**, provided by Intel® Distribution of OpenVINO™ toolkit.   
   These can be found in the `deployment_tools/model_downloader/object_detection` folder.
* created by the user

By default this application uses the mobilenet-ssd model, that can be accesed using the **model downloader**.
The **model downloader** downloads the model as Caffe* model files. These need to be passed through the **model optimizer** to generate the IR (the __.xml__ and __.bin__ files), that will be used by the application.

   
#### Downloading the mobilenet-ssd Intel® Model
To download the mobilenet-ssd model you first need to go to the _deployment_tools/model_downloader/_ folder inside your Intel® Distribution of OpenVINO™ toolkit install folder.   
If the Intel® Distribution of OpenVINO™ toolkit was installed using the default installation paths, you can go to the **model downloader** folder by running:
```
cd /opt/intel/computer_vision_sdk/deployment_tools/model_downloader/
```

You can specify which model to download with `--name` and the output path with `-o` (otherwise it will be downloaded to the current folder).
Run the model downloader with the following command:
```
./downloader.py --name mobilenet-ssd
```
**Note:** You may need to run the `downloader.py` command with sudo
   

You will find the model inside the _object_detection/common_ folder. To make it work with the Intel® Distribution of OpenVINO™ toolkit, the model needs to be passed through the **model optimizer** to generate the IR (the __.xml__ and __.bin__ files).   
   

If you haven't configured the **model optimizer** yet, use this link to do so https://software.intel.com/en-us/articles/OpenVINO-ModelOptimizer.   
After configuration go to the _deployment_tools/model_optimizer/_ folder inside your Intel® Distribution of OpenVINO™ toolkit install folder.   
If the Intel® Distribution of OpenVINO™ toolkit was installed using the default installation paths, you can go to the **model optimizer** folder by running:
```
cd /opt/intel/computer_vision_sdk/deployment_tools/model_optimizer/
```
   
Assuming you've downloaded the reference-implementation to your home directory, run this command to optimize mobilenet-ssd:
```
./mo_caffe.py --input_model /opt/intel/computer_vision_sdk/deployment_tools/model_downloader/object_detection/common/mobilenet-ssd/caffe/mobilenet-ssd.caffemodel -o $HOME/store-traffic-monitor/application/resources
```
**Note:** Replace `$HOME` in the command with the path to the reference-implementation's folder
   

If you want to optimize the model for FP16:
```
./mo_caffe.py --input_model /opt/intel/computer_vision_sdk/deployment_tools/model_downloader/object_detection/common/mobilenet-ssd/caffe/mobilenet-ssd.caffemodel -o $HOME/store-traffic-monitor/application/resources --data_type FP16
```
   

### The labels file
In order to work, this application requires a _labels_ file associated with the model being used for detection.  
All detection models work with integer labels and not string labels (e.g. for the ssd300 and mobilenet-ssd models, the number 15 represents the class "person"), that is why each model must have a _labels_ file, which associates an integer (the label the algorithm detects) with a string (denoting the human-readable label).   


The _labels_ file is a text file containing all the classes/labels that the model can recognize, in the order that it was trained to recognize them (one class per line). 
For the ssd300 and mobilenet-ssd models, we provide the class file _labels.txt_ in the resources folder.


### The config file
The _resources/conf.txt_ contains the videos that will be used by the application, one video per line.   
Each of the lines in the file is of the form `path/to/video label`, e.g.:
```
videos/video1.mp4 person
```
The `path/to/video` is the path, on the local system, to a video to use as input. Followed by the `label` of the class (person, bottle, etc.) to be detected on that video. The labels used in the _conf.txt_ file must coincide with the labels from the _labels_ file.

The application can use any number of videos for detection (i.e. the _conf.txt_ file can have any number of lines), but the more videos the application uses in parallel, the more the frame rate of each video scales down. This can be solved by adding more computation power to the machine the application is running on.


### What input video to use
The application works with any input video.
Sample videos for object detection are provided [here](https://github.com/intel-iot-devkit/sample-videos/).  

For first-use, we recommend using the [people-detection](https://github.com/intel-iot-devkit/sample-videos/blob/master/people-detection.mp4), [one-by-one-person-detection](https://github.com/intel-iot-devkit/sample-videos/blob/master/one-by-one-person-detection.mp4), [bottle-detection](https://github.com/intel-iot-devkit/sample-videos/blob/master/bottle-detection.mp4) videos.   
E.g.:
```
sample-videos/people-detection.mp4 person
sample-videos/one-by-one-person-detection.mp4 person
sample-videos/bottle-detection.mp4 bottle
```
These videos can be downloaded directly, via the `video_downloader` python script provided. The script works with both python2 and python3. Run the following command: 
```
python video_downloader.py
```
The videos are automatically downloaded to the `application/resources/` folder.


### Using camera stream instead of video file
Replace `path/to/video` with the camera ID, where the ID is taken from yout video device (the number X in /dev/videoX).
On Ubuntu, to list all available video devices use the following command:
```
ls /dev/video*
```
   

## Run the application

If not in build folder, go there by using:

```
cd build/
```

To run the application with the needed models:
```
./store-traffic-monitor -m resources/mobilenet-ssd.xml -l ../resources/labels.txt
```
   
### Having the input video loop
By default, the application reads the input videos only once, and ends when the videos ends.
In order to not have the sample videos end, thereby ending the application, the option to continously loop the videos is provided.    
This is done by running the application with the `-lp true` command-line argument:

```
./store-traffic-monitor -lp -m ../resources/mobilenet-ssd.xml -l resources/labels.txt
```

This looping does not affect live camera streams, as camera video streams are continuous and do not end.
   

## Running on different hardware
A user can specify what target device to run on by using the device command-line argument `-d` followed by one of the values `CPU`, `GPU`, or `MYRIAD`.   
If no target device is specified the application will default to running on the CPU.

### Running on the CPU
Although the application runs on the CPU by default, this can also be explicitly specified through the `-d CPU` command-line argument:
```
./store-traffic-monitor -d CPU -m ../resources/mobilenet-ssd.xml -l ../resources/labels.txt
```

### Running on the integrated GPU
To run on the integrated Intel® GPU, use the `-d GPU` command-line argument:
```
./store-traffic-monitor -d GPU -m ../resources/mobilenet-ssd.xml -l ../resources/labels.txt
```
### Running on the Intel® Neural Compute Stick
To run on the Intel® Neural Compute Stick, use the `-d MYRIAD` command-line argument:
```
./store-traffic-monitor -d MYRIAD -m ../resources/mobilenet-ssd.xml -l ../resources/labels.txt
```
**Note:** The Intel® Neural Compute Stick can only run FP16 models. The model that is passed to the application, through the `-m <path_to_model>` command-line argument, must be of data type FP16.

## Using the browser UI

The default application uses a simple user interface created with OpenCV.
A web based UI, with more features is also provided [here](../UI).
In order for the application to work with the browser UI, the output format must be slightly changed. This is done, by running the `make` command with the `UI_OUTPUT` variable set.

````
make CXX_DEFINES=-DUI_OUTPUT
````
