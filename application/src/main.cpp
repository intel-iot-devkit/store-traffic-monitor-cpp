/*
 * Copyright (c) 2018 Intel Corporation.
 *
 * Permission is hereby granted, free of charge, to any person obtaining
 * a copy of this software and associated documentation files (the
 * "Software"), to deal in the Software without restriction, including
 * without limitation the rights to use, copy, modify, merge, publish,
 * distribute, sublicense, and/or sell copies of the Software, and to
 * permit persons to whom the Software is furnished to do so, subject to
 * the following conditions:
 *
 * The above copyright notice and this permission notice shall be
 * included in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 * NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
 * LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
 * OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
 * WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */

#include <iostream>
#include <fstream>
#include <algorithm>
#include "opencv2/opencv.hpp"
#include "opencv2/photo/photo.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/video/video.hpp"
#include <ie_icnn_net_reader.h>
#include <ie_device.hpp>
#include <ie_plugin_config.hpp>
#include <ie_plugin_dispatcher.hpp>
#include <ie_plugin_ptr.hpp>
#include <inference_engine.hpp>
#include <ie_extension.h>
#include <ext_list.hpp>
#include <samples/ocv_common.hpp>
#include <samples/slog.hpp>
#include <nlohmann/json.hpp>

#include <videocap.hpp>
using namespace std;
using namespace cv;
using namespace InferenceEngine::details;
using namespace InferenceEngine;
bool isAsyncMode = true;
using json = nlohmann::json;
json jsonobj;

// Parse the environmental variables
void parseEnv()
{
	if (const char* env_d = std::getenv("DEVICE"))
	{
		conf_targetDevice = std::string(env_d);
	}

	if (const char* env_d = std::getenv("LOOP"))
	{
		if(std::string(env_d) == "true")
		{
			loopVideos = true;
		}
	}
}

// Parse the command line argument
void parseArgs (int argc, char **argv)
{
	if (argc == 1)
	{
		std::cout << "You must specify the model, device and label arguments. Use " << argv[0] << " --help for info\n";
		exit(5);
	}
	if ("-h" == std::string(argv[1]) || "--help" == std::string(argv[1]))
	{
		std::cout << argv[0] << " -m MODEL -l LABELS [OPTIONS]\n\n"
					"-m, --model	Path to .xml file containing model layers\n"
					"-l, --labels	Path to labels file\n"
					"-d, --device	Device to run the inference (CPU, GPU, MYRIAD, FPGA or HDDL only)."
					                "Default option is CPU."
							" To run on multiple devices, use MULTI:<device1>,<device2>,<device3>\n"
					"-f, --flag	Execution on SYNC or ASYNC mode. Default option is ASYNC mode\n"
					"-lp, --loop	Loop video to mimic continuous input\n";
		exit(0);
	}
	for (int i = 1; i < argc; i += 2)
	{
		if ("-m" == std::string(argv[i]) || "--model" == std::string(argv[i]))
		{
			conf_modelPath = std::string(argv[i + 1]);
			int pos = conf_modelPath.rfind(".");
			conf_binFilePath = conf_modelPath.substr(0 , pos) + ".bin";
		}
		else if ("-l" == std::string(argv[i]) || "--labels" == std::string(argv[i]))
		{
			conf_labelsFilePath = std::string(argv[i + 1]);
		}
		else if ("-d" == std::string(argv[i]) || "--device" == std::string(argv[i]))
		{
			conf_targetDevice = std::string(argv[i + 1]);
		}
		else if ("-lp" == std::string(argv[i]) || "--loop" == std::string(argv[i]))
		{
			if(std::string(argv[i + 1]) == "true")
			{
				loopVideos = true;
			}
			if(std::string(argv[i + 1]) == "false")
			{
				loopVideos = false;
			}
		}
		else if ("-f" == std::string(argv[i]) || "--flag" == std::string(argv[i]))
		{
			if (std::string(argv[i + 1]) == "sync")
				isAsyncMode = false;
			else
				isAsyncMode = true;
		}
		else
		{
			std::cout << "Unrecognised option "  << argv[i] << std::endl;
			i--;
		}
	}
}

// Validate the command line arguments
void checkArgs()
{
	if (conf_modelPath.empty())
	{
		std::cout << "You need to specify the path to the .xml file\n";
		std::cout << "Use -m MODEL or --model MODEL or set the MODEL environment variable\n";
		exit(11);
	}

	if (conf_labelsFilePath.empty())
	{
		std::cout << "You need to specify the path to the labels file\n";
		std::cout << "Use -l LABELS or --labels LABELS or set the LABELS environment variable\n";
		exit(12);
	}

	if (conf_targetDevice.empty())
	{
		conf_targetDevice = "CPU";

	}
	else if (!conf_targetDevice.find("MULTI"))
	{
		conf_targetDevice = conf_targetDevice;
	}
	else if (!(std::find(acceptedDevices.begin(), acceptedDevices.end(), conf_targetDevice) != acceptedDevices.end()))
	{
		std::cout << "Unsupported device " << conf_targetDevice << std::endl;
		exit(13);
	}
}

static void configureNetwork(InferenceEngine::CNNNetReader &network) {
	try {
		network.ReadNetwork(conf_modelPath);
	} catch (InferenceEngineException ex) {
		std::cerr << "Failed to load network: " << std::endl;
	}

	network.ReadWeights(conf_binFilePath);

	// Set batch size
	network.getNetwork().setBatchSize(conf_batchSize);
}

// Read the model's label file and get the position of labels required by the application
static std::vector<bool> getUsedLabels(std::vector<VideoCap> &vidCaps, std::vector<string> *reqLabels) {
	std::vector<bool> usedLabels;

	std::ifstream labelsFile(conf_labelsFilePath);

	if (!labelsFile.is_open()) {
		std::cout << "Could not open labels file" << std::endl;
		return usedLabels;
	}

	std::string label;
	int i = 0;
	while (getline(labelsFile, label)) {
		if (std::find((*reqLabels).begin(), (*reqLabels).end(), label) != (*reqLabels).end()) {
			usedLabels.push_back(true);
			for (auto &v : vidCaps) {
				if (v.labelName == label) {
					v.label = i;
				}
			}
		} else {
			usedLabels.push_back(false);
		}
		++i;
	}

	labelsFile.close();

	return usedLabels;
}

// Parse the configuration file conf.txt and get the videos to be processed
std::vector<VideoCap> getVideos (std::ifstream *file, size_t width, size_t height, vector<string> *reqLabels)
{
	std::vector<VideoCap> videos;
	std::string str;
	char camName[20];
	std::string word, AdTemp, label, video_path;
	std::vector<std::string> words;
	unsigned int jsonInstanceCount = 0;
	*file>>jsonobj;
	auto obj = jsonobj["inputs"];
	for(int i=0;i<obj.size();i++)
	{
		label = obj[i]["label"];
		video_path = obj[i]["video"];
		sprintf(camName, "Video %d", i+1);
		if (video_path.size() == 1 && *(video_path.c_str()) >= '0' && *(video_path.c_str()) <= '9')
		{
			videos.push_back(VideoCap(width, height, std::stoi(video_path), camName, label ));
		}
		else
		{
			videos.push_back(VideoCap(width, height, video_path, camName, label ));
		}
		(*reqLabels).push_back(obj[i]["label"]);
	}
	return videos;
}

// Get the minimum fps of the videos
int get_minFPS(std::vector<VideoCap> &vidCaps)
{
	int minFPS = 240;

	for(auto&& i : vidCaps)
	{
		minFPS = std::min(minFPS, (int)round(i.vc.get(CAP_PROP_FPS)));
	}

	return minFPS;
}

// Write the video results to json files
#ifdef UI_OUTPUT
int saveJSON (vector<VideoCap> &vidCaps, vector<string> frameNames)
{

	// This JSON contains info about current and total object count
	ofstream dataJSON(conf_dataJSON_file);
	if(!dataJSON.is_open())
	{
		cout << "Could not open dataJSON file" << endl;
		return 5;
	}

	// This JSON contains the next frames to be processed by the UI
	ofstream videoJSON(conf_videJSON_file);
	if(!videoJSON.is_open())
	{
		cout << "Could not open videoJSON file" << endl;
		return 5;
	}

	int i = 0;
	int j;
	char str[50];
	dataJSON << "{\n";
	videoJSON << "{\n";
	int vsz = static_cast<int>(vidCaps.size());
	int fsz;
	for (; i < vsz; ++i)
	{
		if (vidCaps[i].countAtFrame.size() > 0)
		{
			j = 0;
			dataJSON << "\t\"Video_" << i + 1 << "\": {\n";
			fsz = static_cast<int>(vidCaps[i].countAtFrame.size()) - 1;
			for (; j < fsz; ++j)
			{
				sprintf(str, "\t\t\"%d\": {\n\t\t\t\"count\":\"%d\",\n\t\t\t\"time\":\"%s\"\n\t\t},\n",
								vidCaps[i].countAtFrame[j].frameNo, vidCaps[i].countAtFrame[j].count,
								vidCaps[i].countAtFrame[j].timestamp);
				dataJSON << str;
			}
			sprintf(str, "\t\t\"%d\": {\n\t\t\t\"count\":\"%d\",\n\t\t\t\"time\":\"%s\"\n\t\t}\n",
								vidCaps[i].countAtFrame[j].frameNo, vidCaps[i].countAtFrame[j].count,
								vidCaps[i].countAtFrame[j].timestamp);
			dataJSON << str;
			dataJSON << "\t},\n";
		}
	}

	dataJSON << "\t\"totals\": {\n";
	for (i = 0; i < vsz - 1; ++i)
	{
		dataJSON << "\t\t\"Video_" << i + 1 << "\": \"" << vidCaps[i].totalCount << "\",\n";
	}
	dataJSON << "\t\t\"Video_" << i + 1 << "\": \"" << vidCaps[i].totalCount << "\"\n";
	dataJSON << "\t}\n";

	dataJSON << "}";

	dataJSON.close();

	int sz = static_cast<int>(frameNames.size()) - 1;
	for (i = 0; i < sz; ++i)
	{
		videoJSON << "\t\"" << i + 1 << "\":\"" << frameNames[i] << "\",\n";
	}
	videoJSON << "\t\"" << i + 1 << "\":\"" << frameNames[i] << "\"\n";
	videoJSON << "}";
	videoJSON.close();
	return 0;
}
#else

// Arranges the windows so that they are not overlapping
void arrangeWindows(vector<VideoCap> *vidCaps, size_t width, size_t height)
{
	int spacer = 25;
	int cols = 0;
	int rows = 0;

	// Arrange video windows
	for (int i = 0; i < (*vidCaps).size(); ++i)
	{
		if (cols == conf_windowColumns)
		{
			cols = 0;
			++rows;
			moveWindow((*vidCaps)[i].camName, (spacer + width) * cols, (spacer + height) * rows);
			++cols;
		}
		else
		{
			moveWindow((*vidCaps)[i].camName, (spacer + width) * cols, (spacer + height) * rows);
			++cols;
		}
	}

	// Arrange statistics window
	if (cols == conf_windowColumns)
	{
		cols = 0;
		++rows;
		moveWindow("Statistics", (spacer + width) * cols, (spacer + height) * rows);
	}
	else
	{
		moveWindow("Statistics", (spacer + width) * cols, (spacer + height) * rows);
	}
}

// Write the video results to json files at the end of the application
int saveJSON (vector<VideoCap> &vidCaps)
{
	// This JSON contains info about current and total object count
	// It is saved at the end of the program
	ofstream dataJSON(conf_dataJSON_file);
	if(!dataJSON.is_open())
	{
		cout << "Could not open JSON file" << endl;
		return 5;
	}

	int i = 0;
	int j;
	char str[10];
	dataJSON << "{\n";
	int vsz = static_cast<int>(vidCaps.size());
	int fsz;
	for (; i < vsz; ++i)
	{
		if (vidCaps[i].countAtFrame.size() > 0)
		{
			j = 0;
			dataJSON << "\t\"Video_" << i + 1 << "\": {\n";
			fsz = static_cast<int>(vidCaps[i].countAtFrame.size()) - 1;
			for (; j < fsz; ++j)
			{
				sprintf(str, "\t\t\"%.2f\" : \"%d\",\n", (float)vidCaps[i].countAtFrame[j].first /
				                vidCaps[i].vc.get(cv::CAP_PROP_FPS),	vidCaps[i].countAtFrame[j].second);
				dataJSON << str;
			}
			sprintf(str, "\t\t\"%.2f\" : \"%d\"\n",	(float)vidCaps[i].countAtFrame[j].first /
			                vidCaps[i].vc.get(cv::CAP_PROP_FPS), vidCaps[i].countAtFrame[j].second);
			dataJSON << str;
			dataJSON << "\t},\n";
		}
	}
	dataJSON << "\t\"totals\": {\n";
	for (i = 0; i < vsz - 1; ++i)
	{
		dataJSON << "\t\t\"Video_" << i + 1 << "\": \"" << vidCaps[i].totalCount << "\",\n";
	}
	dataJSON << "\t\t\"Video_" << i + 1 << "\": \"" << vidCaps[i].totalCount << "\"\n";
	dataJSON << "\t}\n";
	dataJSON << "}";
	dataJSON.close();

	return 0;
}
#endif


int main(int argc, char **argv)
{

	vector<bool> noMoreData;
	int index = 0;
	parseEnv();
	parseArgs(argc, argv);
	checkArgs();

	std::ifstream confFile(conf_file);
	if (!confFile.is_open())
	{
		cout << "Could not open config file" << endl;
		return 2;
	}

	// Load the IE plugin for the target device
	Core ie;
	InferenceEngine::CNNNetReader network;

	// Configure the network
	configureNetwork(network);
	if ((conf_targetDevice.find("CPU") != std::string::npos))
	{
		// Required for support of certain layers in CPU
		ie.AddExtension(std::make_shared<Extensions::Cpu::CpuExtensions>(), "CPU");
	}

	InputsDataMap inputInfo(network.getNetwork().getInputsInfo());

	std::string imageInputName, imageInfoInputName;
	size_t netInputHeight, netInputWidth, netInputChannel;

	for (const auto &inputInfoItem : inputInfo)
	{
		if (inputInfoItem.second->getTensorDesc().getDims().size() ==4)
		{ // first input contains images
			imageInputName = inputInfoItem.first;
			inputInfoItem.second->setPrecision(Precision::U8);
			inputInfoItem.second->getInputData()->setLayout(Layout::NCHW);
			const TensorDesc &inputDesc = inputInfoItem.second->getTensorDesc();
			netInputHeight = getTensorHeight(inputDesc);
			netInputWidth = getTensorWidth(inputDesc);
			netInputChannel = getTensorChannels(inputDesc);
		}
		else if (inputInfoItem.second->getTensorDesc().getDims().size() == 2) { // second input contains image info
			imageInfoInputName = inputInfoItem.first;
			inputInfoItem.second->setPrecision(Precision::FP32);
		}
		else
		{
			throw std::logic_error(
				"Unsupported " +
				std::to_string(
				inputInfoItem.second->getTensorDesc().getDims().size()) +
				"D "
				"input layer '" +
				inputInfoItem.first + "'. "
				"Only 2D and 4D input layers are supported");
		}
	}

	OutputsDataMap outputInfo(network.getNetwork().getOutputsInfo());
	if (outputInfo.size() != 1) {
		throw std::logic_error("This demo accepts networks having only one output");
	}
	DataPtr &output = outputInfo.begin()->second;
	auto outputName = outputInfo.begin()->first;
	const int num_classes = network.getNetwork().getLayerByName(outputName.c_str())->GetParamAsInt("num_classes");
	const SizeVector outputDims = output->getTensorDesc().getDims();
	const int maxProposalCount = outputDims[2];

	const int objectSize = outputDims[3];
	if (objectSize != 7) {
		throw std::logic_error("Output should have 7 as a last dimension");
	}
	if (outputDims.size() != 4) {
		throw std::logic_error("Incorrect output dimensions for SSD");
	}
	output->setPrecision(Precision::FP32);
	output->setLayout(Layout::NCHW);

	// -----------------------------------------------------------------------------------------------------

	// --------------------------- 4. Loading model to the device
	// -----------------------------------------------------------------------------------------------------
	slog::info << "Loading model to the device" << slog::endl;
	ExecutableNetwork net =	ie.LoadNetwork(network.getNetwork(), conf_targetDevice);
	// -----------------------------------------------------------------------------------------------------

	// --------------------------- 5. Create infer request
	// -----------------------------------------------------------------------------------------------------
	InferenceEngine::InferRequest::Ptr currInfReq = net.CreateInferRequestPtr();
	InferenceEngine::InferRequest::Ptr nextInfReq = net.CreateInferRequestPtr();

	// ----------------------
	// get output dimensions
	// ----------------------
	/* it's enough just to set image info input (if used in the model) only once
	*/
	if (!imageInfoInputName.empty()) {
		auto setImgInfoBlob = [&](const InferRequest::Ptr &inferReq) {
			auto blob = inferReq->GetBlob(imageInfoInputName);
			auto data =	blob->buffer().as<PrecisionTrait<Precision::FP32>::value_type *>();
			data[0] = static_cast<float>(netInputHeight); // height
			data[1] = static_cast<float>(netInputWidth);  // width
			data[2] = 1;
		};
		setImgInfoBlob(currInfReq);
		setImgInfoBlob(nextInfReq);
	}

	// Create VideoCap objects for all cams
	std::vector<VideoCap> vidCaps;

	// Requested label for each video
	std::vector<string> reqLabels;

	vidCaps = getVideos(&confFile, netInputHeight, netInputWidth, &reqLabels);
	const size_t input_width = vidCaps[0].vc.get(CAP_PROP_FRAME_WIDTH);
	const size_t input_height = vidCaps[0].vc.get(CAP_PROP_FRAME_HEIGHT);
	const size_t output_width = netInputWidth;
	const size_t output_height = netInputHeight;

	int minFPS = get_minFPS(vidCaps);
	int waitTime = (int)(round(1000 / minFPS / vidCaps.size()));

#ifndef UI_OUTPUT
	// Create video writer for every input source
	for (auto &vidCapObj : vidCaps)
	{
		if(!vidCapObj.initVW(output_height, output_width, minFPS))
		{
			cout << "Could not open " << vidCapObj.videoName << " for writing\n";
			return 4;
		}
	}

	namedWindow("Statistics", WINDOW_AUTOSIZE);

	arrangeWindows(&vidCaps, output_width, output_height + 4);
	Mat stats;
#endif
	Mat frameInfer;
	Mat frame, prev_frame;
	Mat *output_frames = new Mat[conf_batchSize];

	auto input_channels = netInputChannel; // Channels for color format, RGB=4
	auto channel_size = output_width * output_height;
	auto input_size = channel_size * input_channels;
	bool no_more_data = false;

	// Read class names
	std::vector<bool> usedLabels = getUsedLabels(vidCaps, &reqLabels);
	if (usedLabels.empty()) {
		std::cout << "Error: No labels currently in use. Please check your path."
		<< std::endl;
		return 1;
	}

#ifdef UI_OUTPUT
	vector<string> frameNames;
#else
	list<string> logList;
	int rollingLogSize = (output_height - 15) / 20;
#endif

	for (auto &vidCapObj : vidCaps)
	{
		vidCapObj.t1 = std::chrono::high_resolution_clock::now();
		noMoreData.push_back(false);
	}
	if (isAsyncMode)
		std::cout << "Application running in Async Mode" << std::endl;
	else
		std::cout << "Application running in sync Mode" << std::endl;

	VideoCap *prevVideoCap;
	typedef std::chrono::duration<double,std::ratio<1, 1000>> ms;

	// Main loop starts here
	for (;;) {
		index = 0;
		for (auto &vidCapObj : vidCaps) {
			// Get a new frame
			int vfps = (int)round(vidCapObj.vc.get(CAP_PROP_FPS));
			for (int i = 0; i < round(vfps / minFPS); ++i)
			{
				vidCapObj.vc.read(frame);
				vidCapObj.loopFrames++;
			}

			if (!frame.data) {
				noMoreData[index] = true;
			}

			// Store as planar BGR for Inference Engine
			// imgIdx -> image pixel counter
			// channel_size -> size of a channel, computed as image size in bytes divided by number of channels,
			//   or image width * image height
			// input_channels -> 3 for RGB image
			// inputPtr -> a pointer to pre-allocated inout buffer
			if (noMoreData[index]){
				++index;
#ifndef UI_OUTPUT
				Mat messageWindow = Mat(output_height, output_width, CV_8UC1, Scalar(0));
				std::string message = "Video stream from " + vidCapObj.camName + " has ended!";
				cv::putText(messageWindow, message, Point(15, output_height / 2),
						cv::FONT_HERSHEY_COMPLEX, 0.4, (255, 255, 255), 1, 8 , false);
				imshow(vidCapObj.camName, messageWindow);
#endif
				continue;
			}
			Blob::Ptr inputBlob;
			vidCapObj.frame = frame;
			vidCapObj.inputWidth = frame.cols;
			vidCapObj.inputHeight = frame.rows;

			//----------------------------------------------
			// Resize to expected size (in model .xml file)
			//----------------------------------------------

			// Input frame is resized to infer resolution
			resize(vidCapObj.frame, frameInfer, Size(output_width, output_height));
			if (isAsyncMode)
			{
				inputBlob = nextInfReq->GetBlob(imageInputName);
			}
			else
			{
				inputBlob = currInfReq->GetBlob(imageInputName);
				prevVideoCap = &vidCapObj;
				prev_frame = vidCapObj.frame;
			}
			matU8ToBlob<uint8_t>(frameInfer, inputBlob);

			//------------------------------------------------------
			// PREPROCESS STAGE:
			// Convert image to format expected by inference engine
			// IE expects planar, convert from packed
			//------------------------------------------------------
			size_t framesize = frameInfer.rows * frameInfer.step1();

			if (framesize != input_size)
			{
				std::cout << "input pixels mismatch, expecting " << input_size
					<< " bytes, got: " << framesize << endl;
				return 1;
			}

			//---------------------------
			// INFER STAGE
			//---------------------------
			std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();
			if (isAsyncMode)
				nextInfReq->StartAsync();
			else
				currInfReq->StartAsync();

			std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
			ms infer_time = std::chrono::duration_cast<ms>(t2 - t1);

			//---------------------------
			// POSTPROCESS STAGE:
			// Parse output
			//---------------------------

#ifdef UI_OUTPUT
			int frames = vidCapObj.frames;
#endif

			if (currInfReq->Wait(IInferRequest::WaitMode::RESULT_READY) == OK) {

				float *box = currInfReq->GetBlob(outputName)->buffer().as<InferenceEngine::PrecisionTrait<
					InferenceEngine::Precision::FP32>::value_type *>();
				prevVideoCap->currentCount = 0;
				prevVideoCap->changedCount = false;

				//---------------------------
				// Parse SSD output
				//---------------------------
				for (int c = 0; c < maxProposalCount; c++) {
					float *localbox = &box[c * 7];
					float image_id = localbox[0];
					float label = localbox[1] - 1;
					float confidence = localbox[2];

					int labelnum = (int)label;
					if ((confidence > conf_thresholdValue) && usedLabels[labelnum] &&
					(prevVideoCap->label == labelnum)) {
						prevVideoCap->currentCount++;
						float xmin = localbox[3] * prevVideoCap->inputWidth;
						float ymin = localbox[4] * prevVideoCap->inputHeight;
						float xmax = localbox[5] * prevVideoCap->inputWidth;
						float ymax = localbox[6] * prevVideoCap->inputHeight;
						rectangle(prev_frame, Point((int)xmin, (int)ymin), Point((int)xmax, (int)ymax),
							Scalar(0, 255, 0), 4, LINE_AA, 0);
					}
				}

				if (prevVideoCap->candidateCount == prevVideoCap->currentCount)
					prevVideoCap->candidateConfidence++;
				else {
					prevVideoCap->candidateConfidence = 0;
					prevVideoCap->candidateCount = prevVideoCap->currentCount;
				}
				if (prevVideoCap->candidateConfidence == conf_candidateConfidence) {
					prevVideoCap->candidateConfidence = 0;
					prevVideoCap->changedCount = true;

#ifdef UI_OUTPUT
					frames++;
#else
					vidCapObj.frames++;
#endif
					if (prevVideoCap->currentCount > prevVideoCap->lastCorrectCount) {
						prevVideoCap->totalCount += prevVideoCap->currentCount - prevVideoCap->lastCorrectCount;
					}

					if (prevVideoCap->currentCount != prevVideoCap->lastCorrectCount) {
						time_t t = time(nullptr);
						tm *currTime = localtime(&t);
#ifdef UI_OUTPUT
						frameInfo fr;
						fr.frameNo = frames;
						fr.count = prevVideoCap->currentCount;
						sprintf(fr.timestamp, "%02d:%02d:%02d", currTime->tm_hour,
							currTime->tm_min, currTime->tm_sec);
						prevVideoCap->countAtFrame.push_back(fr);
#else
						prevVideoCap->countAtFrame.emplace_back(prevVideoCap->frames, prevVideoCap->currentCount);
						int detObj = prevVideoCap->currentCount - prevVideoCap->lastCorrectCount;
						char str[50];
						for (int j = 0; j < detObj; ++j) {
							sprintf(str, "%02d:%02d:%02d - %s detected on %s", currTime->tm_hour,
								currTime->tm_min, currTime->tm_sec, prevVideoCap->labelName.c_str(),
								prevVideoCap->camName.c_str());
							logList.emplace_back(str);
							sprintf(str, "%s\n", str);
							if (logList.size() > rollingLogSize) {
								logList.pop_front();
							}
						}
#endif
					}
#ifdef UI_OUTPUT
					frames++;
#else
					prevVideoCap->frames++;
#endif
					prevVideoCap->lastCorrectCount = prevVideoCap->currentCount;
				}


				resize(prev_frame, prev_frame, Size(output_width, output_height));
				//-------------------------------------------
				//  Display the vidCapObj result and log window
				//-------------------------------------------


#ifdef UI_OUTPUT
				// Saving frames for real-time UI
				string imgName(prevVideoCap->camName);
				replace(imgName.begin(), imgName.end(), ' ', '_');
				prevVideoCap->frames++;
				imgName += '_' + to_string(prevVideoCap->frames);
				frameNames.emplace_back(imgName);
				imgName = conf_videoDir + imgName + ".jpg";
				imwrite(imgName, prev_frame);

				int a;
				if (a = saveJSON(vidCaps, frameNames)) // Save JSONs for Live UI
				{
					return a;
				}
#else
				prevVideoCap->vw.write(prev_frame);

				/* Add log text to each frame */
				std::ostringstream s;
				s << "Total " << prevVideoCap->labelName << " count: " << prevVideoCap->totalCount;
				cv::putText(prev_frame, s.str(), cv::Point(10, output_height - 10),	FONT_HERSHEY_SIMPLEX,
					0.5, cv::Scalar(255, 255, 255), 1, 8, false);
				s.str("");
				s.clear();
				s << "Current " << prevVideoCap->labelName	<< " count: " << prevVideoCap->lastCorrectCount;
				cv::putText(prev_frame, s.str(), cv::Point(10, output_height - 30),	FONT_HERSHEY_SIMPLEX,
					0.5, cv::Scalar(255, 255, 255), 1, 8, false);

				// Get app FPS
				prevVideoCap->t2 = std::chrono::high_resolution_clock::now();
				std::chrono::duration<float> time_span = std::chrono::duration_cast<std::chrono::duration<float>>(
					prevVideoCap->t2 - prevVideoCap->t1);
				char vid_fps[20];
				sprintf(vid_fps, "FPS: %.2f", 1 / time_span.count());
				cv::putText(prev_frame, string(vid_fps), cv::Point(10, output_height - 50),
					FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1, 8, false);

				// Print infer time
				char infTm[20];
				if (!isAsyncMode) {
					sprintf(infTm, "Infer time: %.3f", infer_time.count());
				} else {
					sprintf(infTm, "Infer time: N/A for Async mode");
				}
				cv::putText(prev_frame, string(infTm), cv::Point(10, output_height - 70),
					FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1, 8, false);

				// Show current frame and update statistics window
				cv::imshow(prevVideoCap->camName, prev_frame);

				prevVideoCap->t1 = std::chrono::high_resolution_clock::now();

				stats =	Mat(output_height > (vidCaps.size() * 20 + 15) ? output_height :
					(vidCaps.size() * 20 + 15),	output_width > 345 ? output_width : 345, CV_8UC1, Scalar(0));

				int i = 0;
				for (list<string>::iterator it = logList.begin(); it != logList.end(); ++it)
				{
					putText(stats, *it, Point(10, 15 + 20 * i), FONT_HERSHEY_SIMPLEX, 0.5,
						Scalar(255, 255, 255), 1, 8, false);
					++i;
				}

				cv::imshow("Statistics", stats);

				/**
				* Show frame as soon as possible and exit if ESC key is
				pressed and
				* window is active.
				* waitKey takes miliseconds as argument.
				* waitKey(1) is recommended for camera input. If
				processing is faster
				* than input
				* the application will wait for next frame on capture.
				* You can use vidCaps[0].vc.get(cv::CAP_PROP_FPS) to use
				the FPS of
				* the 1st vidCapObj.
				*/
				if (waitKey(1) == 27) {
					saveJSON(vidCaps);
					delete[] output_frames;
					cout << "Finished\n";
					return 0;
				}
#endif
				if (loopVideos && !vidCapObj.isCam)
				{
					int vfps = (int)round(vidCapObj.vc.get(CAP_PROP_FPS));
					if (vidCapObj.loopFrames > vidCapObj.vc.get(CAP_PROP_FRAME_COUNT) - round(vfps / minFPS))
					{
						vidCapObj.loopFrames = 0;
						vidCapObj.vc.set(CAP_PROP_POS_FRAMES, 0);
					}

				}
			}

			++index;
			if (isAsyncMode)
			{
				currInfReq.swap(nextInfReq);
				prev_frame = vidCapObj.frame.clone();
				prevVideoCap = &vidCapObj;
			}

		}

		// Check if all the videos have ended
		if (find(noMoreData.begin(), noMoreData.end(), false) == noMoreData.end())
			break;
	}
	delete[] output_frames;
	cout << "Finished\n";
	return 0;
}

