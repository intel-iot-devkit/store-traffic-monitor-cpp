# Store Traffic Monitor

| Details               |                    |
| --------------------- | ------------------ |
| Target OS:            | Ubuntu\* 16.04 LTS |
| Programming Language: | C++                |
| Time to Complete:     | 50-70min           |

![store-traffic-monitor](docs/images/store-traffic-monitor-image.png)


## What It Does

The store traffic monitor reference implementation gives the total number of people currently present and total number of people visited the facility. It also counts the product inventory. The application is capable of processing the inputs from multiple cameras and video files.

## Requirements

### Hardware

- 6th to 8th Generation Intel® Core™ processors with Iris® Pro graphics or Intel® HD Graphics

### Software

- [Ubuntu\* 16.04 LTS](http://releases.ubuntu.com/16.04/)<br>
  **Note**: We recommend using a 4.14+ Linux* kernel with this software. Run the following command to determine the kernel version:<br>
     ```
     uname -a
     ```

- OpenCL™ Runtime Package

- Intel® Distribution of OpenVINO™ toolkit 2019 R2 Release

## How It Works

The application uses the Inference Engine included in the Intel® Distribution of OpenVINO™ toolkit. A trained neural network detects objects by displaying a green bounding box over them. This reference implementation identifies multiple objects entering the frame and records the class of each object, the count, and the time the object entered the frame.
![Architectural Diagram](docs/images/arch.png)

**Figure 2:** Architectural Diagram.



## Setup
### Get the code
Clone the reference implementation
```
sudo apt-get update && sudo apt-get install git
git clone https://github.com/intel-iot-devkit/store-traffic-monitor-cpp.git
```

### Install OpenVINO

Refer to [Install Intel® Distribution of OpenVINO™ toolkit for Linux*](https://software.intel.com/en-us/articles/OpenVINO-Install-Linux) to learn how to install and configure the toolkit.

Install the OpenCL™ Runtime Package to run inference on the GPU, as shown in the instructions below. It is not mandatory for CPU inference.


### Other Dependencies
**FFmpeg***<br>
FFmpeg is a free and open-source project capable of recording, converting and streaming digital audio and video in various formats. It can be used to do most of our multimedia tasks quickly and easily say, audio compression, audio/video format conversion, extract images from a video and a lot more.




### Which Model to Use

This application uses the [mobilenet-ssd](https://github.com/chuanqi305/MobileNet-SSD) model, that can be accessed using the **model downloader**. The **model downloader** downloads the model as Caffe* model files. These need to be passed through the **model optimizer** to generate the IR (the __.xml__ and __.bin__ files) that will be used by the application.

The application also works with any object-detection model, provided it has the same input and output format of the SSD model.
The model can be any object detection model:
- Downloaded using the **model downloader**, provided by Intel® Distribution of OpenVINO™ toolkit.

- Built by the user.<br>

To download the models and install the dependencies of the application, run the below command in the `store-traffic-monitor-cpp` directory:
```
./setup.sh
```

### The Labels File

The application requires a _labels_ file associated with the model used for detection. 

 All detection models work with integer labels, not string labels (e.g., For the ssd300 and mobilenet-ssd models, the number 15 represents the class "person".). Each model must have a _labels_ file, which associates an integer, the label the algorithm detects, with a string denoting the human-readable label.

The _labels_ file is a text file containing all the classes/labels that the model can recognize, in the order that it was trained to recognize them, one class per line.<br>

For mobilenet-ssd model, _labels.txt_ file is provided in the _resources_ directory.

### The Config File

The _resources/config.json_ contains the path of videos and label that will be used by the application as input. Each block represents one video file and its corresponding label for detection.

For example:
   ```
   {
       "inputs": [
          {
              "video":"path_to_video/video1.mp4",
              "label":"person"
          }
       ]
   }
   ```

The `path/to/video` is the path to an input video file and the `label` of the class (e.g., person, bottle) to be detected on that video. The labels used in the _config.json_ file must coincide with the labels from the _labels_ file.

The application can use any number of videos for detection (i.e., the _config.json_ file can have any number of blocks), but the more videos the application uses in parallel, the more the frame rate of each video scales down. This can be solved by adding more computation power to the machine on which the application is running.

### Which Input Video to use

The application works with any input video. Sample videos are provided [here](https://github.com/intel-iot-devkit/sample-videos/).

For first-use, we recommend using the [people-detection](https://github.com/intel-iot-devkit/sample-videos/blob/master/people-detection.mp4), [one-by-one-person-detection](https://github.com/intel-iot-devkit/sample-videos/blob/master/one-by-one-person-detection.mp4), [bottle-detection](https://github.com/intel-iot-devkit/sample-videos/blob/master/bottle-detection.mp4) videos.
For example:

```
{
   "inputs":[
      {
         "video":"sample-videos/people-detection.mp4",
         "label":"person"
      },
      {
         "video":"sample-videos/one-by-one-person-detection.mp4",
         "label":"person"
      },
      {
         "video":"sample-videos/bottle-detection.mp4",
         "label":"bottle"
      }
   ]
}
```

If the user wants to use any other video, it can be used by providing the path in the config.json file.


### Using the Camera Stream instead of video

Replace `path/to/video` with the camera ID in the config.json file, where the ID is taken from the video device (the number X in /dev/videoX).

On Ubuntu, to list all available video devices use the following command:

```
ls /dev/video*
```

For example, if the output of above command is __/dev/video0__, then config.json would be:

```
  {
     "inputs": [
        {
           "video":"0",
           "label":"person"
        }
     ]
   }
```

### Setup the Environment

Configure the environment to use the Intel® Distribution of OpenVINO™ toolkit by exporting environment variables:

```
source /opt/intel/openvino/bin/setupvars.sh
```

__Note__: This command needs to be executed only once in the terminal where the application will be executed. If the terminal is closed, the command needs to be executed again.

### Build the Application

To build, go to the `store-traffic-monitor-cpp` and run the following commands:

```
mkdir -p build && cd build
cmake -DUI_OUTPUT=OFF ..
make
```

## Run the Application

To see a list of the various options:

```
./store-traffic-monitor -h
```

A user can specify what target device to run on by using the device command-line argument `-d` followed by one of the devices _CPU_, _GPU_, _HDDL_ or _MYRIAD_. If no target device is specified the application will run on the CPU by default.
To run with multiple devices use _-d MULTI:device1,device2_. For example: _-d MULTI:CPU,GPU,MYRIAD_

### Run on the CPU

Although the application runs on the CPU by default, this can also be explicitly specified through the `-d CPU` command-line argument:

```
./store-traffic-monitor -d CPU -m ../resources/FP32/mobilenet-ssd.xml -l ../resources/labels.txt
```
**Note:** By default, the application runs on async mode. To run the application on sync mode, use `-f sync` as command-line argument.

### Run on the Integrated GPU
- To run on the integrated Intel® GPU with floating point precision 32 (FP32), use the `-d GPU` command-line argument:

    ```
    ./store-traffic-monitor -d GPU -m ../resources/FP32/mobilenet-ssd.xml -l ../resources/labels.txt
    ```
    **FP32**: FP32 is single-precision floating-point arithmetic uses 32 bits to represent numbers. 8 bits for the magnitude and 23 bits for the precision. For more information, [click here](https://en.wikipedia.org/wiki/Single-precision_floating-point_format)<br>

- To run on the integrated Intel® GPU with floating point precision 16 (FP16), use the following command:
    ```
    ./store-traffic-monitor -d GPU -m ../resources/FP16/mobilenet-ssd.xml -l ../resources/labels.txt
    ```
    **FP16**: FP16 is half-precision floating-point arithmetic uses 16 bits. 5 bits for the magnitude and 10 bits for the precision. For more information, [click here](https://en.wikipedia.org/wiki/Half-precision_floating-point_format)

### Run on the Intel® Neural Compute Stick

To run on the Intel® Neural Compute Stick, use the `-d MYRIAD` command-line argument.

```
./store-traffic-monitor -d MYRIAD -m ../resources/FP16/mobilenet-ssd.xml -l ../resources/labels.txt
```

**Note:** The Intel® Neural Compute Stick can only run FP16 models. The model that is passed to the application, through the `-m <path_to_model>` command-line argument, must be of data type FP16.

### Run on the Intel® Movidius™ VPU
To run on the Intel® Movidius™ VPU, use the `-d HDDL ` command-line argument:

```
./store-traffic-monitor -d HDDL -m ../resources/FP16/mobilenet-ssd.xml -l ../resources/labels.txt
```
**Note:** The HDDL-R can only run FP16 models. The model that is passed to the application, through the `-m <path_to_model>` command-line argument, must be of data type FP16.

<!--
### Run on the FPGA

Before running the application on the FPGA, program the AOCX (bitstream) file.<br>
Use the setup_env.sh script from [fpga_support_files.tgz](http://registrationcenter-download.intel.com/akdlm/irc_nas/12954/fpga_support_files.tgz) to set the environment variables.<br>


```
source /home/<user>/Downloads/fpga_support_files/setup_env.sh
```

The bitstreams for HDDL-F can be found under the `/opt/intel/openvino/bitstreams/a10_vision_design_bitstreams` folder. To program the bitstream use the following command:<br>


```
aocl program acl0 /opt/intel/openvino/bitstreams/a10_vision_design_bitstreams/2019R1_PL1_FP11_MobileNet_Clamp.aocx
```

For more information on programming the bitstreams, please refer https://software.intel.com/en-us/articles/OpenVINO-Install-Linux-FPGA#inpage-nav-11

To run the application on the FPGA , use the `-d HETERO:FPGA,CPU` command-line argument:

```
./store-traffic-monitor -d HETERO:FPGA,CPU -m ../resources/FP16/mobilenet-ssd.xml -l ../resources/labels.txt
```
-->
### Loop the Input Video

By default, the application reads the input videos only once and ends when the videos end.

The reference implementation provides an option to loop the video so that the input videos and application run continuously.

To loop the sample video, run the application with the `-lp true` command-line argument:

```
./store-traffic-monitor -lp true -d CPU -m ../resources/FP32/mobilenet-ssd.xml -l ../resources/labels.txt
```

This looping does not affect live camera streams, as camera video streams are continuous and do not end.

## Use the Browser UI

The default application uses a simple user interface created with OpenCV. A web based UI with more features is also provided with this application.

For the application to work with the browser UI, the output format must be slightly changed. This is done by compiling the application with `UI_OUTPUT` variable set:

```
cmake -DUI_OUTPUT=ON ..
make
```

Follow the readme provided [here](./UI) to run the web based UI. 
