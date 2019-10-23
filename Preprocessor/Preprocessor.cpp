#include <fstream>
#include <iostream>
#include <sstream>
#include <iomanip>
#include <algorithm>
#include <iterator>
#include <conio.h>
#include <vector>
#include <stdlib.h>

#include <tclap/CmdLine.h>
#include <Json\Json.h>
#include <cudaCompress/Instance.h>
#include <cudaUtil.h>

#include "Utils.h"
#include "SysTools.h"
#include "LargeArray3D.h"
#include "Statistics.h"

#include "TimeVolumeIO.h"

#include "CompressVolume.h"
#include "GPUResources.h"

using namespace std;


void e()
{
	system("pause");
}


void WriteStatsHeaderCSV(std::ostream& stream)
{
	stream << "ElemCount;OriginalSize;CompressedSize;BitsPerElem;CompressionRate;"
		<< "Min;Max;Range;Average;Variance;"
		<< "ReconstMin;ReconstMax;ReconstRange;ReconstAverage;ReconstVariance;"
		<< "QuantStep;AvgAbsError;MaxAbsError;AvgRelError;MaxRelError;RelErrorCount;"
		<< "RMSError;SNR;PSNR" << std::endl;
}

void WriteStatsCSV(std::ostream& stream, const Statistics::Stats& stats, float quantStep)
{
	stream << stats.ElemCount << ";" << stats.OriginalSize << ";" << stats.CompressedSize << ";" << stats.BitsPerElem << ";" << stats.CompressionRate << ";"
		<< stats.Min << ";" << stats.Max << ";" << stats.Range << ";" << stats.Average << ";" << stats.Variance << ";"
		<< stats.ReconstMin << ";" << stats.ReconstMax << ";" << stats.ReconstRange << ";" << stats.ReconstAverage << ";" << stats.ReconstVariance << ";"
		<< quantStep << ";" << stats.AvgAbsError << ";" << stats.MaxAbsError << ";" << stats.AvgRelError << ";" << stats.MaxRelError << ";" << stats.RelErrorCount << ";"
		<< stats.RMSError << ";" << stats.SNR << ";" << stats.PSNR << std::endl;
}


string getFullPath(string relName)
{
	char* fp = _fullpath(NULL, relName.c_str(), 0);
	string s(fp);
	delete[] fp;
	return s;
}

bool isRelative(string name)
{
	return (name.find(":") == string::npos && name.find("//") == string::npos);
}

// https://stackoverflow.com/a/236803
template<typename Out>
void split(const std::string &s, char delim, Out result) {
	std::stringstream ss;
	ss.str(s);
	std::string item;
	while (std::getline(ss, item, delim)) {
		*(result++) = item;
	}
}
std::vector<std::string> split(const std::string &s, char delim) {
	std::vector<std::string> elems;
	split(s, delim, std::back_inserter(elems));
	return elems;
}


int main(int argc, char* argv[])
{
	#if defined(DEBUG) | defined(_DEBUG)
		_CrtSetDbgFlag( _CRTDBG_ALLOC_MEM_DF | _CRTDBG_LEAK_CHECK_DF );
	#endif

	struct FileListItem
	{
		string fileMask;
		int channels;
		LA3D::LargeArray3D<float>* tmpArray;
	};

	//atexit(&e);

	// **** Parse command line ****

	vector<FileListItem> fileList;
	string outFile;
	string inPath;
	string inRootPath = "";
	string outPath;
	string tmpPath;
	bool overwrite;
	int32 tMin, tMax, tStep, tOffset;
	Vec3i volumeSize;
	int channels = 0;
	bool periodic;
	

	// @Behdad
	bool tempExist;


	tum3D::Vec3f gridSpacing;
	float timeSpacing;
	int32 brickSize;
	int32 overlap;

	

	Json::Object jsonObject;
	bool jsonObjectAvailable;

	eCompressionType compression = COMPRESSION_NONE; // = 0
	string quantStepsString;
	vector<float> quantSteps;
	bool lessRLE;
	uint huffmanBits;

	bool autoBuild;
	bool keepLA3Ds;



	try
	{
		TCLAP::CmdLine cmd("", ' ');

		TCLAP::UnlabeledValueArg<string> outFileArg("outfile", "Output file name without extension (index file will be <filename>.timevol)", true, "", "String", cmd);
		TCLAP::UnlabeledMultiArg<string> inMaskArg("inmask", "Any number of input files in the format <filename>:<channels>, where <filename>"
			" must contain a decimal-formator like %d that will be replaced by the timestep index. For example, \"testdata_256_%d.raw:3\" adds"
			" a 3-channel-file."
			"\nThis argument is optional: Either pass the file here or with --filenames and --channels",
			false, "String", cmd);

		TCLAP::ValueArg<string> jsonFileArg("i", "json", "The path to a json file with key-value pairs for settings the arguments.\n"
			"Command line argumenst always have priority", false, "", "String", cmd);

		TCLAP::ValueArg<string> outPathArg("O", "outpath", "Output path", false, "", "String", cmd);
		TCLAP::ValueArg<string> inPathArg("I", "inpath", "Input path", false, "", "String", cmd);
		TCLAP::ValueArg<string> tmpPathArg("", "tmp", "Path for temporary la3d files; default outpath", false, "", "String", cmd);
		TCLAP::SwitchArg keepLA3DsArg("", "keepla3ds", "Don't delete temporary .la3d files after finishing", cmd);
		TCLAP::SwitchArg overwriteArg("", "overwrite", "Force overwriting of existing output files (default is trying to append)", cmd);

		TCLAP::ValueArg<int32> tMinArg("", "tmin", "First (inclusive) timestep to include", false, 0, "Integer", cmd);
		TCLAP::ValueArg<int32> tMaxArg("", "tmax", "Last (inclusive) timestep to include", false, 0, "Integer", cmd);
		TCLAP::ValueArg<int32> tStepArg("", "tstep", "Timestep increment", false, 1, "Integer", cmd);
		TCLAP::ValueArg<int32> tOffsetArg("", "toffset", "Index of the first timestep to write (for appending!)", false, 0, "Integer", cmd);


		TCLAP::ValueArg<int32> volumeSizeXArg("x", "volumesizex", "X dimension of the volume", false, 1024, "Integer", cmd);
		TCLAP::ValueArg<int32> volumeSizeYArg("y", "volumesizey", "Y dimension of the volume", false, 1024, "Integer", cmd);
		TCLAP::ValueArg<int32> volumeSizeZArg("z", "volumesizez", "Z dimension of the volume", false, 1024, "Integer", cmd);

		TCLAP::SwitchArg periodicArg("", "periodic", "Use periodic boundary, i.e. wrap (default is clamp)", cmd);

		// @Behdad
		TCLAP::SwitchArg tempExistArg("", "tempExist", "Dataset contains temperatures (otherwise a dummy channel for temperature is added to the dataset)", cmd);

		TCLAP::ValueArg<float> gridSpacingArg("g", "gridspacing", "Distance between grid points (for cubic cells)", false, 1.0f, "Float", cmd);
		TCLAP::ValueArg<float> gridSpacingXArg("", "gridspacingX", "Distance between grid points in x-direction (for non-cubic cells)", false, 1.0f, "Float", cmd);
		TCLAP::ValueArg<float> gridSpacingYArg("", "gridspacingY", "Distance between grid points in y-direction (for non-cubic cells)", false, 1.0f, "Float", cmd);
		TCLAP::ValueArg<float> gridSpacingZArg("", "gridspacingZ", "Distance between grid points in z-direction (for non-cubic cells)", false, 1.0f, "Float", cmd);
		TCLAP::ValueArg<float> domainSizeXArg("", "domainX", "The domain size in x-direction", false, 1.0f, "Float", cmd);
		TCLAP::ValueArg<float> domainSizeYArg("", "domainY", "The domain size in y-direction", false, 1.0f, "Float", cmd);
		TCLAP::ValueArg<float> domainSizeZArg("", "domainZ", "The domain size in z-direction", false, 1.0f, "Float", cmd);
		TCLAP::ValueArg<float> timeSpacingArg("t", "timespacing", "Distance between time steps", false, 1.0f, "Float", cmd);
		
		TCLAP::ValueArg<int32> brickSizeArg("b", "bricksize", "Brick size including overlap", true, 64, "Integer", cmd); /* \param req changed to "true" entering bricksize
																														 in command line 28.12.2018 - Behdad Ghaffari */
		//TCLAP::ValueArg<int32> brickSizeArg("b", "bricksize", "Brick size including overlap", false, 64, "Integer", cmd); 


		TCLAP::ValueArg<int32> overlapArg("v", "overlap", "Brick overlap (size of halo)", false, 4, "Integer", cmd);

		TCLAP::ValueArg<string> quantStepsArg("q", "quantstep", "Enable compression with fixed quantization steps [%f %f ...]", false, "[5e-4 5e-4 5e-4 5e-4]", "Float Array", cmd);
		TCLAP::SwitchArg quantFirstArg("", "quantfirst", "Perform quantization before DWT", cmd);
		TCLAP::SwitchArg lessRLEArg("", "lessrle", "Do RLE only on finest level (improves compression with fine to moderate quant steps)", cmd);
		TCLAP::ValueArg<int32> huffmanBitsArg("", "huffmanbits", "Max number of bits per symbol for Huffman coding", false, 0, "Integer", cmd);

		TCLAP::SwitchArg autoBuildArg("a", "autobuild", "Suppress all confirmation messages", cmd, false);

		TCLAP::ValueArg<string> filenamesArg("f", "filenames", "Input filename template: e.g. %s.32.dat_%03d. %s will be filled by the channels-argument, %d by the timestep", 
			false, "", "String", cmd);
		TCLAP::ValueArg<string> channelsArg("c", "channels",
			"If the input files are given as templates by --filenames. This channel argument then fills the values. Format: <name>:<channels>,<name>:<channels>...",
			false, "", "String Array", cmd);


		cmd.parse(argc, argv);

		//Load Json-File
		if (jsonFileArg.isSet()) {
			jsonObject = Json::ParseFile(jsonFileArg.getValue());
			if (jsonObject.Size() == 0) {
				cout << "Invalid or empty file passed as --json argument" << endl;
				return -1;
			}
			jsonObjectAvailable = true;
			cout << "Json file with extra arguments loaded" << endl;
			string absFilePath = getFullPath(jsonFileArg.getValue());
			auto lastDel = absFilePath.find_last_of("/");
			if (lastDel == string::npos)
				lastDel = absFilePath.find_last_of("\\");
			if (lastDel == string::npos) {
				cout << "Unable to extract the folder from the json file: " << absFilePath << endl;
				return -1;
			}
			inRootPath = absFilePath.substr(0, lastDel);
			cout << "Resolve input files relative to " << inRootPath << endl;
		}

		outFile = outFileArg.getValue();

		vector<string> inMask = inMaskArg.getValue();
		if (!inMaskArg.isSet() && jsonObjectAvailable) { //try to set inMask from the json
			Json::Value v = jsonObject["inMask"];
			if (v.Type() == Json::ARRAY) {
				Json::Array va = v.AsArray();
				for (auto it = va.Begin(); it != va.End(); ++it) {
					inMask.push_back(it->AsString());
				}
			}
		}

		string filenames = filenamesArg.getValue();
		string channelsStr = channelsArg.getValue();
		bool filenamesSet = filenamesArg.isSet();
		bool channelsSet = channelsArg.isSet();
		bool filenameTemplatesSet = false;

		if (!filenamesSet && jsonObjectAvailable) { //try to load filenames from the json file
			Json::Value v = jsonObject["filenames"];
			if (v.Type() == Json::STRING) {
				filenames = v.AsString();
				filenamesSet = true;
			}
		}
		if (!channelsSet && jsonObjectAvailable) { //try to load channels from the json file
			Json::Value v = jsonObject["channels"];
			if (v.Type() == Json::STRING) {
				channelsStr = v.AsString();
				channelsSet = true;
			}
		}
		if (filenamesSet || channelsSet) {
			if (!filenamesSet) {
				cout << "If you specify --channels, you must also use --filenames" << endl;
				return -1;
			}
			if (!channelsSet) {
				cout << "If you specify --filenames, you must also use --channels" << endl;
				return -1;
			}
			filenameTemplatesSet = true;
		}
		if (filenameTemplatesSet) {
			if (!inMask.empty()) {
				cout << "WARNING: --filenames and --channels specified, but input file mask is not empty, overwrite them" << endl;
			}
			inMask = vector<string>();
			auto it = filenames.find("%s");
			if (it == string::npos) {
				cout << "--filenames does not contain %s for the channel name!" << endl;
				return -1;
			}
			for (string channel : split(channelsStr, ',')) {
				vector<string> parts = split(channel, ':');
				if (parts.size() != 2) {
					cout << "Each entry in --channels must have the form <name>:<channels>, seperated by ','. But it was: " << channel << endl;
					return -1;
				}
				string name = filenames;
				name.replace(it, 2, parts[0]);
				inMask.push_back(name + ":" + parts[1]);
			}
		}

#define LOAD_ARG_FROM_JSON(variable, cmdArg, jsonArg, jsonType, jsonGetter) \
	variable = (cmdArg).getValue(); \
	if (!(cmdArg).isSet() && jsonObjectAvailable) { \
		Json::Value v = jsonObject[jsonArg]; \
		if (v.Type() == jsonType) { \
			variable = v. jsonGetter; \
		} \
	}

		LOAD_ARG_FROM_JSON(outPath, outPathArg, "outPath", Json::STRING, AsString());
		LOAD_ARG_FROM_JSON(inPath, inPathArg, "inPath", Json::STRING, AsString());
		LOAD_ARG_FROM_JSON(tmpPath, tmpPathArg, "tmpPath", Json::STRING, AsString());
		LOAD_ARG_FROM_JSON(keepLA3Ds, keepLA3DsArg, "keepLA3Ds", Json::BOOL, AsBool());
		LOAD_ARG_FROM_JSON(overwrite, overwriteArg, "overwrite", Json::BOOL, AsBool());

		

		if (isRelative(inPath) //not an absolute path
			&& !inRootPath.empty()) { //json file defines the root
			inPath = inRootPath + "\\" + inPath;
			std::cout << "Resolve input path relative to the json file" << std::endl;
		}
		if (isRelative(outPath) //not an absolute path
			&& !inRootPath.empty()) { //json file defines the root
			outPath = inRootPath + "\\" + outPath;
			std::cout << "Resolve output path relative to the json file" << std::endl;
		}

		LOAD_ARG_FROM_JSON(tMin, tMinArg, "tmin", Json::INT, AsInt32());
		LOAD_ARG_FROM_JSON(tMax, tMaxArg, "tmax", Json::INT, AsInt32());
		LOAD_ARG_FROM_JSON(tStep, tStepArg, "tstep", Json::INT, AsInt32());
		LOAD_ARG_FROM_JSON(tOffset, tOffsetArg, "toffset", Json::INT, AsInt32());


		LOAD_ARG_FROM_JSON(volumeSize[2], volumeSizeXArg, "griddims", Json::ARRAY, AsArray()[0].AsInt32());
		LOAD_ARG_FROM_JSON(volumeSize[1], volumeSizeYArg, "griddims", Json::ARRAY, AsArray()[1].AsInt32());
		LOAD_ARG_FROM_JSON(volumeSize[0], volumeSizeZArg, "griddims", Json::ARRAY, AsArray()[2].AsInt32());

		//volumeSize[0] = volumeSizeXArg.getValue();
		//volumeSize[1] = volumeSizeYArg.getValue();
		//volumeSize[2] = volumeSizeZArg.getValue();
		periodic = periodicArg.getValue();
		
		//@Behdad
		tempExist = tempExistArg.getValue();


		if (gridSpacingArg.isSet()) {
			if (gridSpacingXArg.isSet() || gridSpacingYArg.isSet() || gridSpacingZArg.isSet()
				|| domainSizeXArg.isSet() || domainSizeYArg.isSet() || domainSizeZArg.isSet()) {
				cout << "If you use -gridspacing, you must not use the arguments for non-cubic cells like -gridspacingX or specify the domain boundary like -domainX" << endl;
				return -1;
			}
			float spacing = gridSpacingArg.getValue();
			gridSpacing = tum3D::Vec3f(spacing);
		}
		else if (gridSpacingXArg.isSet() || gridSpacingYArg.isSet() || gridSpacingZArg.isSet()) {
			if (!(gridSpacingXArg.isSet() && gridSpacingYArg.isSet() && gridSpacingZArg.isSet())) {
				cout << "If you specify the grid spacing for non-cubic cells, you must use all three gridSpacing[XYZ] arguments" << endl;
				return -1;
			}
			if (domainSizeXArg.isSet() || domainSizeYArg.isSet() || domainSizeZArg.isSet()) {
				cout << "If you use -gridSpacing[XYZ] then you must not use the arguments to specify the domain size" << endl;
				return -1;
			}
			gridSpacing = tum3D::Vec3f(
				gridSpacingXArg.getValue(),
				gridSpacingYArg.getValue(),
				gridSpacingZArg.getValue()
			);
		}
		else if (domainSizeXArg.isSet() || domainSizeYArg.isSet() || domainSizeZArg.isSet()) {
			if (!(domainSizeXArg.isSet() && domainSizeYArg.isSet() && domainSizeZArg.isSet())) {
				cout << "If you specify the domain size, you must use all three domainSize[XYZ] arguments" << endl;
				return -1;
			}
			gridSpacing = tum3D::Vec3f(
				domainSizeXArg.getValue() / float(volumeSize.x()),
				domainSizeYArg.getValue() / float(volumeSize.y()),
				domainSizeZArg.getValue() / float(volumeSize.z())
			);
		}
		else {
			//try JSON
			bool spacingDefined = false;
			if (jsonObjectAvailable) {
				Json::Value v = jsonObject["aspectratio"];
				if (v.Type() == Json::ARRAY) {
					gridSpacing = tum3D::Vec3f(
						v.AsArray()[2].AsFloat() / float(volumeSize.x() - 1),
						v.AsArray()[1].AsFloat() / float(volumeSize.y() - 1),
						v.AsArray()[0].AsFloat() / float(volumeSize.z() - 1));
					//normalize size to one
					float maxGridSize = (gridSpacing * tum3D::Vec3f(volumeSize)).maximum();
					gridSpacing /= maxGridSize;
					//gridSpacing *= 2.0f;
					spacingDefined = true;
				}
			}
			if (!spacingDefined) {
				//cubic cells with default spacing
				gridSpacing = tum3D::Vec3f(2.0f / float(volumeSize.maximum()));
			}
		}

		timeSpacing = timeSpacingArg.getValue();
		brickSize = brickSizeArg.getValue();
		overlap = overlapArg.getValue();

		if(quantStepsArg.isSet()) // @Behdad: Remove quantization 
		{
			compression = quantFirstArg.isSet() ? COMPRESSION_FIXEDQUANT_QF : COMPRESSION_FIXEDQUANT;
			quantStepsString = quantStepsArg.getValue();
		}
		lessRLE = lessRLEArg.getValue();
		huffmanBits = huffmanBitsArg.getValue();

		autoBuild = autoBuildArg.getValue();


		// build input file descriptors from input file mask
		E_ASSERT("No input files given", inMask.size() > 0);
		for (auto it = inMask.cbegin(); it != inMask.cend(); ++it)
		{
			char tmp[1024];
			FileListItem desc;
			E_ASSERT("Invalid input file desc: " << *it, sscanf_s(it->c_str(), "%[^:]:%d", tmp, 1024, &desc.channels) == 2);
			desc.fileMask = tmp;
			desc.tmpArray = new LA3D::LargeArray3D<float>(desc.channels);

			fileList.push_back(desc);
			channels += desc.channels;
		}
	}
	catch (TCLAP::ArgException &e)
	{
		cout << "*** ERROR: " << e.error() << " for arg " << e.argId() << "\n";
		return -1;
	}

	// Sanity checks
	if (outPath != "")
	{
		outPath += "\\";
	}
	if (inPath != "")
	{
		inPath += "\\";
	}
	if (tmpPath == "")
	{
		tmpPath = outPath;
	}
	else
	{
		tmpPath += "\\";
	}

	if(compression == COMPRESSION_FIXEDQUANT || compression == COMPRESSION_FIXEDQUANT_QF)
	{
		// Parse quantization steps
		quantSteps.resize(channels);
		for (int32 i = 0; i < channels; ++i)
		{
			quantSteps[i] = 1.0f / 256.0f;
		}

		// Tokenize quant steps
		if (quantStepsString.length() > 0)
		{
			E_ASSERT("Invalid quantization steps parameter", (quantStepsString[0] == '[')
				&& (quantStepsString[quantStepsString.length() - 1] == ']'));
			std::istringstream iss(quantStepsString.substr(1, quantStepsString.length() - 2));
			vector<string> quantStepsTokens;
	
			copy(istream_iterator<string>(iss),
				istream_iterator<string>(),
				back_inserter<vector<string>>(quantStepsTokens));

			E_ASSERT("Invalid number of quantization steps: " << quantStepsTokens.size(), quantStepsTokens.size() == channels);
			for (int32 i = 0; i < quantStepsTokens.size(); ++i)
			{
				E_ASSERT("Invalid quantization step token: "/* << s */, sscanf_s(quantStepsTokens[i].c_str(), "%f", &quantSteps[i]) == 1);
			}
		}
	}


	E_ASSERT("BrickSize " << brickSize << " is not a power of 2", IsPow2(brickSize));
	E_ASSERT("No channels found", channels > 0);
	int32 brickDataSize = brickSize - 2 * overlap;

	cout << std::endl;
	cout << "Preprocessing started with the following parameters: \n" <<
		"Inpath: " << inPath << "\n" <<
		"Outpath: " << outPath << "\n" <<
		"Tmp path: " << tmpPath << "\n" <<
		"Outfile: " << outFile << "\n" <<
		"Volume size: [" << volumeSize[0] << ", " << volumeSize[1] << ", " << volumeSize[2] << "]\n"
		"Brick size: " << brickSize << "\n" <<
		"Grid spacing: " << gridSpacing.x() << ", " << gridSpacing.y() << ", " << gridSpacing.z() << "\n"
		"Overlap: " << overlap << "\n" <<
		"Compression: " << GetCompressionTypeName(compression) << "\n" <<
		"tMin: " << tMin << ", tMax: " << tMax << "\n";
	if(compression == COMPRESSION_FIXEDQUANT || compression == COMPRESSION_FIXEDQUANT_QF)
	{
		cout << "Quantization Steps: [";
		for_each(quantSteps.cbegin(), quantSteps.cend() - 1, [](float f)
		{
			std::cout << f << ", ";
		});
		cout << *(quantSteps.cend() - 1) << "]\n";
		cout << "LessRLE: " << lessRLE << "\n";
	}

	cout << "\n\n";

	cout << "A total of " << channels << " channels was found. Channel files are: \n\n";
	for (auto it = fileList.cbegin(); it != fileList.cend(); ++it)
	{
		cout << "\t" << it->fileMask << " with " << it->channels << " channels\n";
	}

	cout << "\n\n";

	if (!autoBuild)
	{
		cout << "Press \"y\" to continue or any other key to abort: ";
		char c = _getch();

		if (c != 'y')
		{
			exit(0);
		}
	}


	const float REL_ERROR_EPSILON = 0.01f;


	// start global timer
	LARGE_INTEGER timestampGlobalStart;
	QueryPerformanceCounter(&timestampGlobalStart);

	LARGE_INTEGER perfFreq;
	QueryPerformanceFrequency(&perfFreq);

	// common quant step over all channels? (only for statistics output)
	float quantStepCommon = 0.0f;
	if(compression == COMPRESSION_FIXEDQUANT || compression == COMPRESSION_FIXEDQUANT_QF)
	{
		bool haveCommonQuantStep = true;
		for(size_t i = 1; i < quantSteps.size(); i++)
		{
			if(quantSteps[i] != quantSteps[i-1])
			{
				haveCommonQuantStep = false;
			}
		}
		quantStepCommon = haveCommonQuantStep ? quantSteps[0] : -1.0f;
	}

	// accumulated timings, in seconds
	float timeCreateLA3D = 0.0f;
	float timeBricking = 0.0f;
	float timeCompressGPU = 0.0f;
	float timeDecompressGPU = 0.0f;

	cudaEvent_t eventStart, eventEnd;
	cudaSafeCall(cudaEventCreate(&eventStart));
	cudaSafeCall(cudaEventCreate(&eventEnd));


	std::string statsPath = outPath + "stats\\";

	// Create directories
	cout << "\n\n";
	system((string("mkdir \"") + outPath + "\"").c_str());
	system((string("mkdir \"") + tmpPath + "\"").c_str());
	system((string("mkdir \"") + statsPath + "\"").c_str());



	// **** Build each timestep ****
	GPUResources compressShared;
	CompressVolumeResources compressVolume;
	if (compression)
	{
		compressShared.create(CompressVolumeResources::getRequiredResources(brickSize, brickSize, brickSize, 1, huffmanBits));
		compressVolume.create(compressShared.getConfig());
	}


	const size_t PAGE_SIZE = 64;
	const size_t numPagesX = (volumeSize[0] + PAGE_SIZE - 1) / PAGE_SIZE;
	const size_t numPagesY = (volumeSize[0] + PAGE_SIZE - 1) / PAGE_SIZE;
	const size_t numPagesZ = (volumeSize[0] + PAGE_SIZE - 1) / PAGE_SIZE;

	float* srcSlice = new float[volumeSize[0] * PAGE_SIZE * 4];		// 4 = max number of channels
	std::vector<std::vector<float>> rawBrickChannelData(channels);
	std::vector<std::vector<float>> rawBrickChannelDataReconst(channels);
	std::vector<std::vector<uint32>> compressedBrickChannelData(channels);
	std::vector<float*> deviceBrickChannelData(channels);

	for (int i = 0; i < channels; ++i)
	{
		cudaSafeCall(cudaMalloc(&deviceBrickChannelData[i], Cube(brickSize) * sizeof(float)));
	}

	TimeVolumeIO out;

	if (tum3d::FileExists(outPath + outFile + ".timevol") && !overwrite)
	{
		out.Open(outPath + outFile + ".timevol", true);
		E_ASSERT("Cannot append to existing file: VolumeSize mismatch", out.GetVolumeSize() == volumeSize);
		E_ASSERT("Cannot append to existing file: BrickSize mismatch", out.GetBrickSizeWithOverlap() == brickSize);
		E_ASSERT("Cannot append to existing file: Overlap mismatch", out.GetBrickOverlap() == overlap);
		E_ASSERT("Cannot append to existing file: Channels mismatch", out.GetChannelCount() == channels);
		E_ASSERT("Cannot append to existing file: Compression mismatch", out.GetCompressionType() == compression);

		if(compression == COMPRESSION_FIXEDQUANT || compression == COMPRESSION_FIXEDQUANT_QF)
		{
			for (int32 i = 0; i < channels; ++i)
			{
				E_ASSERT("Cannot append to existing file: Channel " << i << " quantization mismatch", out.GetQuantStep(i) == quantSteps[i]);
			}
		}
		if(compression != COMPRESSION_NONE)
		{
			E_ASSERT("Cannot append to existing file: LessRLE mismatch", out.GetUseLessRLE() == lessRLE);
			E_ASSERT("Cannot append to existing file: Huffman bits mismatch", out.GetHuffmanBitsMax() == huffmanBits);
		}
	}
	else
	{
		out.Create(outPath + outFile + ".timevol", volumeSize, periodic, brickSize, overlap, channels, outFile);

		out.SetCompressionType(compression);
		if(compression == COMPRESSION_FIXEDQUANT || compression == COMPRESSION_FIXEDQUANT_QF)
		{
			for (int32 i = 0; i < channels; ++i)
			{
				out.SetQuantStep(i, quantSteps[i]);
			}
		}
		out.SetUseLessRLE(lessRLE);
		out.SetHuffmanBitsMax(huffmanBits);
	}

	out.SetGridSpacing(gridSpacing);
	out.SetTimeSpacing(timeSpacing);

	Vec3i brickCount = (volumeSize + brickDataSize - 1) / brickDataSize;



	char fn[1024];

	uint64 laMem = uint64(0.8f * float(GetAvailableMemory()) / float(fileList.size()));
	// don't use more than 4 GB total
	uint64 laMemMax = 4 * 1024ull * 1024ull * 1024ull / fileList.size();
	laMem = min(laMem, laMemMax);


	Statistics statsGlobal;
	std::vector<Statistics> statsGlobalChannels(channels);

	std::ofstream fileStatsGlobal(statsPath + outFile + "_stats_global.csv");
	std::ofstream fileStatsTimestep(statsPath + outFile + "_stats_timestep.csv");
	std::ofstream fileStatsBrick(statsPath + outFile + "_stats_brick.csv");

	WriteStatsHeaderCSV(fileStatsGlobal);
	fileStatsTimestep << "Timestep;";
	WriteStatsHeaderCSV(fileStatsTimestep);
	fileStatsBrick << "Timestep;BrickX;BrickY;BrickZ;";
	WriteStatsHeaderCSV(fileStatsBrick);

	std::vector<std::ofstream> fileStatsGlobalChannel(channels);
	std::vector<std::ofstream> fileStatsTimestepChannel(channels);
	std::vector<std::ofstream> fileStatsBrickChannel(channels);
	for(int c = 0; c < channels; c++) {
		std::stringstream chan; chan << c;
		fileStatsGlobalChannel[c].open(statsPath + outFile + "_stats_global_channel" + chan.str() + ".csv");
		fileStatsTimestepChannel[c].open(statsPath + outFile + "_stats_timestep_channel" + chan.str() + ".csv");
		fileStatsBrickChannel[c].open(statsPath + outFile + "_stats_brick_channel" + chan.str() + ".csv");

		WriteStatsHeaderCSV(fileStatsGlobalChannel[c]);
		fileStatsTimestepChannel[c] << "Timestep;";
		WriteStatsHeaderCSV(fileStatsTimestepChannel[c]);
		fileStatsBrickChannel[c] << "Timestep;BrickX;BrickY;BrickZ;";
		WriteStatsHeaderCSV(fileStatsBrickChannel[c]);
	}


	// Run through all timesteps
	for (int32 timestep = tMin; timestep <= tMax; timestep += tStep)
	{
		cout << "\n\n\n>>>> Processing timestep #" << timestep << " <<<<<\n\n";

		std::vector<Statistics> statsTimestepChannels(channels);

		LARGE_INTEGER timestampLA3DStart;
		QueryPerformanceCounter(&timestampLA3DStart);

		bool createdLA3Ds = true;

		for (auto fdesc = fileList.begin(); fdesc != fileList.end(); ++fdesc)
		{
			cout << "\n\n";

			sprintf_s(fn, fdesc->fileMask.c_str(), timestep);
			string fileName(fn);

			string tmpFilePath = tmpPath + "tmp_" + fileName + ".la3d";
			wstring wstrTmpFilePath(tmpFilePath.begin(), tmpFilePath.end());

			if (tum3d::FileExists(tmpFilePath))
			{
				//TODO check size etc?
				cout << "Using existing LA3D file " << tmpFilePath << "\n";
				fdesc->tmpArray->Open(wstrTmpFilePath.c_str(), true, laMem);
				createdLA3Ds = false;
				continue;
			}

			cout << "Loading channels from " << fileName << "...\n";

			// Check if file exists
			string filePath = inPath + fileName;
			E_ASSERT("File " << filePath << " does not exist", tum3d::FileExists(filePath));

			// Open file
			FILE* file;
			fopen_s(&file, filePath.c_str(), "rb");
			E_ASSERT("Could not open file \"" << filePath << "\"", file != nullptr);


			// Create output array
			fdesc->tmpArray->Create(volumeSize[0], volumeSize[1], volumeSize[2], 64, 64, 64, wstrTmpFilePath.c_str(), laMem);

			// Read slice by slice and write to target array
			cout << "Creating LA3D for " << fileName << "\n\n";

			
			for (int32 pageSliceZ = 0; pageSliceZ < numPagesZ; ++pageSliceZ)
			{
				SimpleProgress(pageSliceZ, numPagesZ);

				size_t pageZStart = pageSliceZ * PAGE_SIZE;
				size_t pageDepth = PAGE_SIZE;

				if (pageSliceZ == numPagesZ - 1 && volumeSize[2] % PAGE_SIZE != 0) {
					pageDepth = volumeSize[2] % PAGE_SIZE;
				}

				for (int32 pageSliceY = 0; pageSliceY < numPagesY; ++pageSliceY)
				{
					size_t pageYStart = pageSliceY * PAGE_SIZE;
					size_t pageHeight = PAGE_SIZE;

					if (pageYStart == numPagesY - 1 && volumeSize[1] % PAGE_SIZE != 0) {
						pageHeight = volumeSize[1] % PAGE_SIZE;
					}

					for (int32 sliceZ = pageZStart; sliceZ < pageZStart + pageDepth; ++sliceZ)
					{
						int64_t filePos = (static_cast<uint64_t>(sliceZ) * volumeSize[1] * volumeSize[0] + static_cast<uint64_t>(pageYStart) * volumeSize[0])
							* sizeof(float) * fdesc->channels;
						_fseeki64(file, filePos, SEEK_SET);
						fread(srcSlice, sizeof(float) * fdesc->channels, volumeSize[0] * pageHeight, file);
						fdesc->tmpArray->CopyFrom(srcSlice, 0, pageYStart, sliceZ, volumeSize[0], pageHeight, 1, volumeSize[0], pageHeight);
					}
				}
			}
		}

		LARGE_INTEGER timestampLA3DEnd;
		QueryPerformanceCounter(&timestampLA3DEnd);
		timeCreateLA3D += float(timestampLA3DEnd.QuadPart - timestampLA3DStart.QuadPart) / float(perfFreq.QuadPart);

		cout << "\n\nBricking...\n";

		LARGE_INTEGER timestampBrickingStart;
		QueryPerformanceCounter(&timestampBrickingStart);

		// Brick and write
		for (int32 bz = 0; bz < brickCount[2]; ++bz)
		{
			for (int32 by = 0; by < brickCount[1]; ++by)
			{
				for (int32 bx = 0; bx < brickCount[0]; ++bx)
				{
					SimpleProgress(bx + by * brickCount[0] + bz * brickCount[0] * brickCount[1], brickCount[0] * brickCount[1] * brickCount[2] - 1);

					Vec3i spatialIndex(bx, by, bz);
					Vec3ui size(
						bx < brickCount[0] - 1 ? brickSize : volumeSize[0] - bx * brickDataSize + 2 * overlap,
						by < brickCount[1] - 1 ? brickSize : volumeSize[1] - by * brickDataSize + 2 * overlap,
						bz < brickCount[2] - 1 ? brickSize : volumeSize[2] - bz * brickDataSize + 2 * overlap);

					Statistics statsBrick;

					int32 absChannel = 0;
					for (auto fdesc = fileList.begin(); fdesc != fileList.end(); ++fdesc)
					{
						vector<float*> dstPtr(fdesc->channels);
						for (int32 localChannel = 0; localChannel < fdesc->channels; ++localChannel)
						{
							int32 channel = absChannel + localChannel;
							// Create target channel data
							rawBrickChannelData[channel].resize(size.volume());
							dstPtr[localChannel] = rawBrickChannelData[channel].data();
							rawBrickChannelDataReconst[channel].resize(size.volume());
						}

						for (uint32 z = 0; z < size.z(); ++z)
						{
							for (uint32 y = 0; y < size.y(); ++y)
							{
								for (uint32 x = 0; x < size.x(); ++x)
								{
									int32 volPos[3];
									volPos[0] = bx * brickDataSize - overlap + x;
									volPos[1] = by * brickDataSize - overlap + y;
									volPos[2] = bz * brickDataSize - overlap + z;

									if(periodic)
									{
										volPos[0] = (volumeSize[0] + volPos[0]) % volumeSize[0];
										volPos[1] = (volumeSize[1] + volPos[1]) % volumeSize[1];
										volPos[2] = (volumeSize[2] + volPos[2]) % volumeSize[2];
									}
									else
									{
										volPos[0] = clamp(volPos[0], 0, volumeSize[0] - 1);
										volPos[1] = clamp(volPos[1], 0, volumeSize[1] - 1);
										volPos[2] = clamp(volPos[2], 0, volumeSize[2] - 1);
									}

									for (int32 localChannel = 0; localChannel < fdesc->channels; ++localChannel)
									{
										*dstPtr[localChannel]++ = fdesc->tmpArray->Get(volPos[0], volPos[1], volPos[2])[localChannel];
									}
								}
							}
						}

						for (int32 localChannel = 0; localChannel < fdesc->channels; ++localChannel)
						{
							int32 channel = absChannel + localChannel;
							Statistics statsBrickChannel;
							float quantStep = (compression == COMPRESSION_FIXEDQUANT || compression == COMPRESSION_FIXEDQUANT_QF) ? quantSteps[channel] : 0.0f;
							if (compression != COMPRESSION_NONE)
							{
								float* pOrig = rawBrickChannelData[channel].data();
								float* pDevice = deviceBrickChannelData[channel];
								float* pReconst = rawBrickChannelDataReconst[channel].data();
								size_t sizeRaw = rawBrickChannelData[channel].size() * sizeof(float);

								std::vector<uint>& bitStream = compressedBrickChannelData[channel];

								// upload raw brick data
								cudaSafeCall(cudaMemcpy(pDevice, pOrig, sizeRaw, cudaMemcpyHostToDevice));

								// compress
								cudaSafeCall(cudaEventRecord(eventStart));
								if(compression == COMPRESSION_FIXEDQUANT) {
									compressVolumeFloat(compressShared, compressVolume, pDevice, size[0], size[1], size[2], 2, bitStream, quantStep, lessRLE);
								} else if(compression == COMPRESSION_FIXEDQUANT_QF) {
									compressVolumeFloatQuantFirst(compressShared, compressVolume, pDevice, size[0], size[1], size[2], 2, bitStream, quantStep, lessRLE);
								} else {
									printf("Unknown/unsupported compression mode %u\n", uint(compression));
									exit(42);
								}
								cudaSafeCall(cudaEventRecord(eventEnd));

								float t;
								cudaSafeCall(cudaEventSynchronize(eventEnd));
								cudaSafeCall(cudaEventElapsedTime(&t, eventStart, eventEnd));
								timeCompressGPU += t / 1000.0f;

								// decompress again
								cudaSafeCall(cudaEventRecord(eventStart));
								if(compression == COMPRESSION_FIXEDQUANT) {
									decompressVolumeFloat(compressShared, compressVolume, pDevice, size[0], size[1], size[2], 2, bitStream, quantStep, lessRLE);
								} else if(compression == COMPRESSION_FIXEDQUANT_QF) {
									decompressVolumeFloatQuantFirst(compressShared, compressVolume, pDevice, size[0], size[1], size[2], 2, bitStream, quantStep, lessRLE);
								} else {
									printf("Unknown/unsupported compression mode %u\n", uint(compression));
									exit(42);
								}
								cudaSafeCall(cudaEventRecord(eventEnd));

								cudaSafeCall(cudaEventSynchronize(eventEnd));
								cudaSafeCall(cudaEventElapsedTime(&t, eventStart, eventEnd));
								timeDecompressGPU += t / 1000.0f;

								// download reconstructed data
								cudaSafeCall(cudaMemcpy(pReconst, pDevice, sizeRaw, cudaMemcpyDeviceToHost));

								// compute statistics
								statsBrickChannel.AddData(pOrig, pReconst, size.volume(), bitStream.size() * sizeof(uint), REL_ERROR_EPSILON);
							}
							else
							{
								float* pData = rawBrickChannelData[channel].data();
								// compute statistics
								statsBrickChannel.AddData(pData, (const float*)nullptr, size.volume(), 0, REL_ERROR_EPSILON);
							}

							// write statsBrickChannel to file
							fileStatsBrickChannel[channel] << timestep << ";" << bx << ";" << by << ";" << bz << ";";
							WriteStatsCSV(fileStatsBrickChannel[channel], statsBrickChannel.GetStats(), quantStep);

							// update accumulated statistics
							statsBrick += statsBrickChannel;
							statsTimestepChannels[channel] += statsBrickChannel;
						}

						absChannel += fdesc->channels;
					}

					// write statsBrick to file
					fileStatsBrick << timestep << ";" << bx << ";" << by << ";" << bz << ";";
					WriteStatsCSV(fileStatsBrick, statsBrick.GetStats(), quantStepCommon);


					// Add brick to output
					if (compression != COMPRESSION_NONE)
					{
						out.AddBrick((timestep - tMin) / tStep + tOffset, spatialIndex, size, compressedBrickChannelData);
					}
					else
					{
						out.AddBrick((timestep - tMin) / tStep + tOffset, spatialIndex, size, rawBrickChannelData);
					}

				}	// For x
			}	// For y
		}	// For z

		LARGE_INTEGER timestampBrickingEnd;
		QueryPerformanceCounter(&timestampBrickingEnd);
		timeBricking += float(timestampBrickingEnd.QuadPart - timestampBrickingStart.QuadPart) / float(perfFreq.QuadPart);


		// write per-channel timestep stats to file, and accumulate stats over all channels
		Statistics statsTimestep;
		for(int c = 0; c < channels; c++) {
			float quantStep = (compression == COMPRESSION_FIXEDQUANT || compression == COMPRESSION_FIXEDQUANT_QF) ? quantSteps[c] : 0.0f;
			fileStatsTimestepChannel[c] << timestep << ";";
			WriteStatsCSV(fileStatsTimestepChannel[c], statsTimestepChannels[c].GetStats(), quantStep);

			statsTimestep += statsTimestepChannels[c];
			statsGlobalChannels[c] += statsTimestepChannels[c];
		}

		// write accumulated timestep stats to file
		fileStatsTimestep << timestep << ";";
		WriteStatsCSV(fileStatsTimestep, statsTimestep.GetStats(), quantStepCommon);


		// close and (optionally) delete temp files
		for (auto fdesc = fileList.begin(); fdesc != fileList.end(); ++fdesc)
		{
			// delete file only if we created it ourselves
			bool discard = createdLA3Ds && !keepLA3Ds;
			// don't bother saving anything if file will be deleted anyway
			fdesc->tmpArray->Close(discard);
			if(discard)
			{
				// delete the file
				sprintf_s(fn, fdesc->fileMask.c_str(), timestep);
				string fileName(fn);
				string tmpFilePath = tmpPath + "tmp_" + fileName + ".la3d";
				DeleteFileA(tmpFilePath.c_str());
			}
		}

	}	// For timestep

	// write accumulated global stats to file
	Statistics statsTimestep;
	for(int c = 0; c < channels; c++) {
		float quantStep = (compression == COMPRESSION_FIXEDQUANT || compression == COMPRESSION_FIXEDQUANT_QF) ? quantSteps[c] : 0.0f;
		WriteStatsCSV(fileStatsGlobalChannel[c], statsGlobalChannels[c].GetStats(), quantStep);

		statsGlobal += statsGlobalChannels[c];
	}
	WriteStatsCSV(fileStatsGlobal, statsGlobal.GetStats(), quantStepCommon);


	for (auto fdesc = fileList.begin(); fdesc != fileList.end(); ++fdesc)
	{
		delete fdesc->tmpArray;
	}

	delete[] srcSlice;

	for (int i = 0; i < channels; ++i)
	{
		cudaSafeCall(cudaFree(deviceBrickChannelData[i]));
	}

	if (compression)
	{
		compressVolume.destroy();
		compressShared.destroy();
	}

	cudaSafeCall(cudaEventDestroy(eventEnd));
	cudaSafeCall(cudaEventDestroy(eventStart));


	// stop global timer
	LARGE_INTEGER timestampGlobalEnd;
	QueryPerformanceCounter(&timestampGlobalEnd);
	float timeGlobal = float(timestampGlobalEnd.QuadPart - timestampGlobalStart.QuadPart) / float(perfFreq.QuadPart);

	// write timings to file
	std::ofstream fileTimings(outPath + outFile + "_timings.txt");
	fileTimings;
	fileTimings << "Total time:       " << fixed << setw(10) << setprecision(3) << timeGlobal << " s\n";
	fileTimings << "Create LA3Ds:     " << fixed << setw(10) << setprecision(3) << timeCreateLA3D << " s\n";
	fileTimings << "Bricking:         " << fixed << setw(10) << setprecision(3) << timeBricking << " s\n";
	fileTimings << "Compress (GPU):   " << fixed << setw(10) << setprecision(3) << timeCompressGPU << " s\n";
	fileTimings << "Decompress (GPU): " << fixed << setw(10) << setprecision(3) << timeDecompressGPU << " s\n";
	fileTimings.close();

	return 0;
}

