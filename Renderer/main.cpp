#include <global.h>

#include <cstdio>
#include <ctime>
#include <algorithm>
#include <fstream>
#include <sstream>
#include <vector>
#include <iostream>

#include <omp.h>

#include <cuda_runtime.h>
#include <cuda_d3d11_interop.h>
#include <cuda_profiler_api.h>

#include <AntTweakBar/AntTweakBar.h>
#include <D3DX11Effect/d3dx11effect.h>

#include <cudaCompress/Instance.h>

#include <SysTools.h>
#include <CSysTools.h>
#include <cudaUtil.h>
#include <Vec.h>
using namespace tum3D;

#include "TransferFunctionEditor/TransferFunctionEditor.h"

#include "DXUT.h"
#include <WICTextureLoader.h>

#include "stb_image_write.h"

#include "ProgressBarEffect.h"
#include "ScreenEffect.h"

#include "Range.h"

#include "TimeVolume.h"

#include "FilteringManager.h"
#include "TracingManager.h"
#include "RenderingManager.h"
#include "HeatMapManager.h"

#include "CompressVolume.h"
#include "GPUResources.h"

#include "BatchTraceParams.h"
#include "FilterParams.h"
#include "ProjectionParams.h"
#include "RaycastParams.h"
#include "StereoParams.h"
#include "ParticleTraceParams.h"
#include "ParticleRenderParams.h"
#include "ViewParams.h"
#include "HeatMapParams.h"

#include "WorkerThread.h"


#include "FlowGraph.h"
//#include "TracingBenchmark.h"

#if 0
#include <vld.h>
#endif



// ============================================================
// Variables and stuff
// ============================================================

Vec2i            g_windowSize(0, 0);
float            g_renderBufferSizeFactor = 2.0f;
Vec4f            g_rotationX = Vec4f(1, 0, 0, 0);
Vec4f            g_rotationY = Vec4f(1, 0, 0, 0);
Vec4f            g_rotation = Vec4f(1, 0, 0, 0);
bool             g_keyboardShiftPressed = false;

ProjectionParams g_projParams;
StereoParams     g_stereoParams;
ViewParams       g_viewParams;
Vec2f            g_mouseScreenPosition;

FilterParams         g_filterParams;
RaycastParams        g_raycastParams;
ParticleTraceParams  g_particleTraceParams;
bool                 g_particleTracingPaused = false;
ParticleRenderParams g_particleRenderParams;
HeatMapParams        g_heatMapParams;

BatchTraceParams g_batchTraceParams;


TimeVolume       g_volume(0.8f);
FlowGraph        g_flowGraph;



struct MyCudaDevice
{
	MyCudaDevice(int device, float computePower, float memoryPower)
		: device(device), computePower(computePower), memoryPower(memoryPower), pThread(nullptr) {}

	int device;
	float computePower;
	float memoryPower;

	Range1D range;

	WorkerThread* pThread;
};

// the thread at g_primaryCudaDeviceIndex will not be started!
std::vector<MyCudaDevice> g_cudaDevices;
int                       g_primaryCudaDeviceIndex = -1;
bool                      g_useAllGPUs = false;


// resources on primary GPU
FilteringManager g_filteringManager;
TracingManager   g_tracingManager;
RenderingManager g_renderingManager;
HeatMapManager   g_heatMapManager;

GPUResources            g_compressShared;
CompressVolumeResources g_compressVolume;

std::vector<LineBuffers*> g_lineBuffers;
std::vector<BallBuffers*> g_ballBuffers;
float                     g_ballRadius = 0.011718750051f;

int                       g_lineIDOverride = -1;


// texture to hold results from other GPUs
ID3D11Texture2D*			g_pRenderBufferTempTex = nullptr;
ID3D11ShaderResourceView*	g_pRenderBufferTempSRV = nullptr;
ID3D11RenderTargetView*		g_pRenderBufferTempRTV = nullptr;

// render target to hold last finished image
ID3D11Texture2D*        g_pRaycastFinishedTex = nullptr;
ID3D11RenderTargetView* g_pRaycastFinishedRTV = nullptr;

// staging textures for screenshots
ID3D11Texture2D*        g_pStagingTex = nullptr;
ID3D11Texture2D*		g_pRenderBufferStagingTex = nullptr;

ScreenEffect			g_screenEffect;
ProgressBarEffect		g_progressBarEffect;



Vec4f		g_backgroundColor(1.0f, 1.0f, 1.0f, 1.0f);

bool		g_showPreview = true;

bool		g_bRenderDomainBox = true;
bool		g_bRenderBrickBoxes = false;
bool		g_bRenderClipBox = true;
bool		g_bRenderSeedBox = true;


bool		g_redraw = true;
bool		g_retrace = true;



TimerCPU	g_timerTracing;
TimerCPU	g_timerRendering;



// flag to indicate to the render callback to save a screenshot next time it is called
bool g_saveScreenshot = false;
bool g_saveRenderBufferScreenshot = false;


// image sequence settings/state
struct ImageSequence
{
	ImageSequence()
		: FrameCount(100)
		, AngleInc(0.0f), ViewDistInc(0.0f), FramesPerTimestep(1)
		, Record(false), FromRenderBuffer(false)
		, BaseRotationQuat(0.0f, 0.0f, 0.0f, 0.0f), BaseTimestep(0)
		, Running(false), FrameCur(0) {}

	//TODO move into params struct:
	int32 FrameCount;
	float AngleInc;
	float ViewDistInc;
	int32 FramesPerTimestep;

	bool  Record;
	bool  FromRenderBuffer;

	Vec4f BaseRotationQuat;
	int32 BaseTimestep;
	// ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

	bool  Running;
	int32 FrameCur;
} g_imageSequence;


// batch tracing settings/state
struct BatchTrace
{
	BatchTrace()
		: WriteLinebufs(false), Running(false), FileCur(0), StepCur(0), ExitAfterFinishing(false) {}

	std::vector<std::string> VolumeFiles;

	std::string OutPath;
	bool WriteLinebufs;

	bool Running;
	uint FileCur;
	uint StepCur;

	std::ofstream FileStats;
	std::ofstream FileTimings;

	std::vector<float> Timings;

	bool ExitAfterFinishing;
} g_batchTrace;



// GUI
bool   g_bRenderUI = true;
TwBar* g_pTwBarMain = nullptr;
TwBar* g_pTwBarImageSequence = nullptr;
TwBar* g_pTwBarBatchTrace = nullptr;

clock_t	g_lastRenderParamsUpdate = 0;
clock_t	g_lastTraceParamsUpdate = 0;
float	g_startWorkingDelay = 0.1f;

// cached gui state - apparently values can't be queried from a TwBar?!
uint g_guiFilterRadius[3] = { 0, 0, 0 };


//Transfer function editor
const int				TF_LINE_MEASURES = 0;
const int				TF_HEAT_MAP = 1;
const int				TF_RAYTRACE = 2;
TransferFunctionEditor  g_tfEdt(400, 250, std::vector<std::string>(
	{"Measures", "HeatMap", "Raytrace"}));
int                     g_tfTimestamp = -1;
cudaGraphicsResource*   g_pTfEdtSRVCuda = nullptr;


// OpenMP thread count (because omp_get_num_threads is stupid)
uint g_threadCount = 0;





// ============================================================
// Utility
// ============================================================

std::string toString(float val)
{
	std::ostringstream str;
	str << val;
	return str.str();
}


void DrawProgressBar(ID3D11DeviceContext* context, const Vec2f& pos, const Vec2f& size, const Vec4f& color, float progress)
{
	g_progressBarEffect.m_pvPositionVariable->SetFloatVector(pos);
	g_progressBarEffect.m_pvSizeVariable->SetFloatVector(size);
	g_progressBarEffect.m_pvColorVariable->SetFloatVector(color);
	g_progressBarEffect.m_pfProgressVariable->SetFloat(progress);

	ID3D11Buffer* pNull = nullptr;
	UINT stride = 0;
	UINT offset = 0;
	context->IASetVertexBuffers(0, 0, &pNull, &stride, &offset);
	context->IASetIndexBuffer(nullptr, DXGI_FORMAT_R16_UINT, 0);

	context->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLESTRIP);
	g_progressBarEffect.m_pTechnique->GetPassByIndex(0)->Apply(0, context);
	context->Draw(4, 0);

	context->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_LINESTRIP);
	g_progressBarEffect.m_pTechnique->GetPassByIndex(1)->Apply(0, context);
	context->Draw(5, 0);
}


bool LoadRenderingParams(const std::string& filename)
{
	ConfigFile config;
	if(!config.Read(filename))
		return false;

	Vec2i windowSizeNew = g_windowSize;
	for(size_t s = 0; s < config.GetSections().size(); s++) {
		const ConfigSection& section = config.GetSections()[s];

		std::string sectionName = section.GetName();

		if(section.GetName() == "Main") {

			// this is our section - parse entries
			for(size_t e = 0; e < section.GetEntries().size(); e++) {
				const ConfigEntry& entry = section.GetEntries()[e];

				std::string entryName = entry.GetName();

				if(entryName == "WindowSize") {
					entry.GetAsVec2i(windowSizeNew);
				} else if(entryName == "RenderBufferSizeFactor") {
					entry.GetAsFloat(g_renderBufferSizeFactor);
				} else if(entryName == "RenderDomainBox" || entryName == "RenderBBox") {
					entry.GetAsBool(g_bRenderDomainBox);
				} else if(entryName == "RenderBrickBoxes") {
					entry.GetAsBool(g_bRenderBrickBoxes);
				} else if(entryName == "RenderClipBox" || entryName == "RenderClipBBox") {
					entry.GetAsBool(g_bRenderClipBox);
				} else if(entryName == "RenderSeedBox") {
					entry.GetAsBool(g_bRenderSeedBox);
				} else if(entryName == "RenderUI") {
					entry.GetAsBool(g_bRenderUI);
				} else if(entryName == "BackgroundColor") {
					entry.GetAsVec4f(g_backgroundColor);
				} else {
					printf("WARNING: LoadRenderingParams: unknown config entry \"%s\" ignored\n", entryName.c_str());
				}
			}
		}
	}

	g_viewParams.ApplyConfig(config);
	g_stereoParams.ApplyConfig(config);
	g_filterParams.ApplyConfig(config);
	g_raycastParams.ApplyConfig(config);
	g_particleTraceParams.ApplyConfig(config);
	g_particleRenderParams.ApplyConfig(config);
	g_batchTraceParams.ApplyConfig(config);

	// update filter gui state
	for(uint i = 0; i < 3; i++)
	{
		g_guiFilterRadius[i] = g_filterParams.m_radius.size() > i ? g_filterParams.m_radius[i] : 0;
	}

	// forward params to tf editor
	g_tfEdt.setAlphaScale(TF_RAYTRACE, g_raycastParams.m_alphaScale);
	g_tfEdt.setTfRangeMin(TF_RAYTRACE, g_raycastParams.m_transferFunctionRangeMin);
	g_tfEdt.setTfRangeMax(TF_RAYTRACE, g_raycastParams.m_transferFunctionRangeMax);

	// load transfer function from separate binary file
	std::ifstream fileTF(filename + ".tf", std::ios_base::binary);
	if(fileTF.good()) {
		g_tfEdt.loadTransferFunction(TF_RAYTRACE, &fileTF);
		fileTF.close();
	}

	// resize window if necessary
	if(windowSizeNew != g_windowSize)
	{
		g_windowSize = windowSizeNew;

		// SetWindowPos wants the total size including borders, so determine how large they are...
		RECT oldWindowRect; GetWindowRect(DXUTGetHWND(), &oldWindowRect);
		RECT oldClientRect; GetClientRect(DXUTGetHWND(), &oldClientRect);
		uint borderX = (oldWindowRect.right - oldWindowRect.left) - (oldClientRect.right - oldClientRect.left);
		uint borderY = (oldWindowRect.bottom - oldWindowRect.top) - (oldClientRect.bottom - oldClientRect.top);
		SetWindowPos(DXUTGetHWND(), HWND_TOP, 0, 0, g_windowSize.x() + borderX, g_windowSize.y() + borderY, SWP_NOMOVE);
	}

	return true;
}

bool SaveRenderingParams(const std::string& filename)
{
	ConfigFile config;

	ConfigSection section("Main");
	section.AddEntry(ConfigEntry("WindowSize", g_windowSize));
	section.AddEntry(ConfigEntry("RenderBufferSizeFactor", g_renderBufferSizeFactor));
	section.AddEntry(ConfigEntry("RenderDomainBox", g_bRenderDomainBox));
	section.AddEntry(ConfigEntry("RenderBrickBoxes", g_bRenderBrickBoxes));
	section.AddEntry(ConfigEntry("RenderClipBox", g_bRenderClipBox));
	section.AddEntry(ConfigEntry("RenderSeedBox", g_bRenderSeedBox));
	section.AddEntry(ConfigEntry("RenderUI", g_bRenderUI));
	section.AddEntry(ConfigEntry("BackgroundColor", g_backgroundColor));

	config.AddSection(section);

	g_viewParams.WriteConfig(config);
	g_stereoParams.WriteConfig(config);
	g_filterParams.WriteConfig(config);
	g_raycastParams.WriteConfig(config);
	g_particleTraceParams.WriteConfig(config);
	g_particleRenderParams.WriteConfig(config);

	config.Write(filename);

	std::ofstream fileTF(filename + ".tf", std::ios_base::binary);
	g_tfEdt.saveTransferFunction(TF_RAYTRACE, &fileTF);
	fileTF.close();

	return true;
}


void BuildFlowGraph(const std::string& filenameTxt = "")
{
	//TODO might need to cancel tracing first?
	uint advectStepsPerRoundBak = g_particleTraceParams.m_advectStepsPerRound;
	g_particleTraceParams.m_advectStepsPerRound = 256;
	g_flowGraph.Build(&g_compressShared, &g_compressVolume, 1024, g_particleTraceParams, filenameTxt);
	g_particleTraceParams.m_advectStepsPerRound = advectStepsPerRoundBak;
}

bool SaveFlowGraph()
{
	std::string filename = tum3d::RemoveExt(g_volume.GetFilename()) + ".fg";
	return g_flowGraph.SaveToFile(filename);
}

bool LoadFlowGraph()
{
	std::string filename = tum3d::RemoveExt(g_volume.GetFilename()) + ".fg";
	return g_flowGraph.LoadFromFile(filename);
}

void LoadOrBuildFlowGraph()
{
	if(!LoadFlowGraph()) {
		BuildFlowGraph();
		SaveFlowGraph();
	}
}


void ReleaseLineBuffers()
{
	for(size_t i = 0; i < g_lineBuffers.size(); i++)
	{
		delete g_lineBuffers[i];
	}
	g_lineBuffers.clear();
}

void ReleaseBallBuffers()
{
	for(size_t i = 0; i < g_ballBuffers.size(); i++)
	{
		delete g_ballBuffers[i];
	}
	g_ballBuffers.clear();
}


void ReleaseVolumeDependentResources()
{
	g_flowGraph.Shutdown();

	if(g_useAllGPUs)
	{
		for(size_t i = 0; i < g_cudaDevices.size(); i++)
		{
			if(g_cudaDevices[i].pThread)
			{
				g_cudaDevices[i].pThread->CancelCurrentTask();
			}
		}
		for(size_t i = 0; i < g_cudaDevices.size(); i++)
		{
			if(g_cudaDevices[i].pThread)
			{
				g_cudaDevices[i].pThread->ReleaseVolumeDependentResources();
			}
		}
	}

	ReleaseBallBuffers();
	ReleaseLineBuffers();

	g_renderingManager.Release();

	g_heatMapManager.Release();

	g_tracingManager.ClearResult();
	g_tracingManager.Release();

	g_filteringManager.ClearResult();
	g_filteringManager.Release();

	g_compressVolume.destroy();
	g_compressShared.destroy();
}

HRESULT CreateVolumeDependentResources(ID3D11Device* pDevice)
{
	ReleaseVolumeDependentResources();

	if(g_volume.IsCompressed()) {
		uint brickSize = g_volume.GetBrickSizeWithOverlap();
		// do multi-channel decoding only for small bricks; for large bricks, mem usage gets too high
		uint channelCount = (brickSize <= 128) ? g_volume.GetChannelCount() : 1;
		uint huffmanBits = g_volume.GetHuffmanBitsMax();
		g_compressShared.create(CompressVolumeResources::getRequiredResources(brickSize, brickSize, brickSize, channelCount, huffmanBits));
		g_compressVolume.create(g_compressShared.getConfig());
	}

	HRESULT hr;

	if(FAILED(hr = g_filteringManager.Create(&g_compressShared, &g_compressVolume))) {
		return hr;
	}

	if(FAILED(hr = g_tracingManager.Create(&g_compressShared, &g_compressVolume, pDevice))) {
		return hr;
	}

	if(FAILED(hr = g_renderingManager.Create(&g_compressShared, &g_compressVolume, pDevice))) {
		return hr;
	}

	if (FAILED(hr = g_heatMapManager.Create(&g_compressShared, &g_compressVolume, pDevice))) {
		return hr;
	}


	if(g_useAllGPUs)
	{
		for(size_t i = 0; i < g_cudaDevices.size(); i++)
		{
			if(g_cudaDevices[i].pThread)
			{
				g_cudaDevices[i].pThread->CreateVolumeDependentResources();
			}
		}
	}

	g_flowGraph.Init(g_volume);

	return S_OK;
}


void CloseVolumeFile()
{
	ReleaseVolumeDependentResources();
	g_volume.Close();
}

bool OpenVolumeFile(const std::string& filename, ID3D11Device* pDevice)
{
	CloseVolumeFile();


	if(!g_volume.Open(filename))
	{
		return false;
	}


	// recreate brick slots and re-init cudaCompress - brick size may have changed
	CreateVolumeDependentResources(pDevice);


	int32 timestepMax = g_volume.GetTimestepCount() - 1;
	float timeSpacing = g_volume.GetTimeSpacing();
	float timeMax = timeSpacing * float(timestepMax);
	TwSetParam(g_pTwBarMain, "Time", "max", TW_PARAM_FLOAT, 1, &timeMax);
	TwSetParam(g_pTwBarMain, "Time", "step", TW_PARAM_FLOAT, 1, &timeSpacing);
	TwSetParam(g_pTwBarMain, "Timestep", "max", TW_PARAM_INT32, 1, &timestepMax);


	g_volume.SetCurTime(0.0f);
	g_redraw = true;
	g_retrace = true;

	g_raycastParams.m_clipBoxMin = -g_volume.GetVolumeHalfSizeWorld();
	g_raycastParams.m_clipBoxMax =  g_volume.GetVolumeHalfSizeWorld();


	g_imageSequence.FrameCount = g_volume.GetTimestepCount();


	LoadFlowGraph();


	return true;
}


HRESULT ResizeRenderBuffer(ID3D11Device* pd3dDevice)
{
	g_projParams.m_imageWidth = uint(g_windowSize.x() * g_renderBufferSizeFactor);
	g_projParams.m_imageHeight = uint(g_windowSize.y() * g_renderBufferSizeFactor);
	g_projParams.m_aspectRatio = float(g_projParams.m_imageWidth) / float(g_projParams.m_imageHeight);


	SAFE_RELEASE(g_pRenderBufferTempRTV);
	SAFE_RELEASE(g_pRenderBufferTempSRV);
	SAFE_RELEASE(g_pRenderBufferTempTex);
	SAFE_RELEASE(g_pRenderBufferStagingTex);


	if(g_projParams.m_imageWidth * g_projParams.m_imageHeight > 0 && pd3dDevice)
	{
		HRESULT hr;

		D3D11_TEXTURE2D_DESC desc;
		desc.ArraySize = 1;
		desc.BindFlags = 0;
		desc.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
		desc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
		desc.MipLevels = 1;
		desc.MiscFlags = 0;
		desc.SampleDesc.Count = 1;
		desc.SampleDesc.Quality = 0;
		desc.Usage = D3D11_USAGE_STAGING;
		desc.Width = g_projParams.m_imageWidth;
		desc.Height = g_projParams.m_imageHeight;
		hr = pd3dDevice->CreateTexture2D(&desc, nullptr, &g_pRenderBufferStagingTex);
		if(FAILED(hr)) return hr;

		// create texture to blend together opaque objects with raycasted stuff
		// (also used to upload results from other GPUs)
		desc.ArraySize = 1;
		desc.BindFlags = D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_RENDER_TARGET;
		desc.CPUAccessFlags = 0;
		desc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
		desc.MipLevels = 1;
		desc.MiscFlags = 0;
		desc.SampleDesc.Count = 1;
		desc.SampleDesc.Quality = 0;
		desc.Usage = D3D11_USAGE_DEFAULT;
		desc.Width = g_projParams.m_imageWidth;
		desc.Height = g_projParams.m_imageHeight;
		hr = pd3dDevice->CreateTexture2D(&desc, nullptr, &g_pRenderBufferTempTex);
		if(FAILED(hr)) return hr;
		hr = pd3dDevice->CreateShaderResourceView(g_pRenderBufferTempTex, nullptr, &g_pRenderBufferTempSRV);
		if(FAILED(hr)) return hr;
		hr = pd3dDevice->CreateRenderTargetView(g_pRenderBufferTempTex, nullptr, &g_pRenderBufferTempRTV);
		if(FAILED(hr)) return hr;
	}

	return S_OK;
}


uint GetShadersPerMP(int major, int minor)
{
	switch(major)
	{
		case 1: return 8;
		case 2: return (minor == 0) ? 32 : 48;
		case 3: return 192;
		case 5: return 128;
		case 6: return (minor == 0) ? 64 : 128;
		default: assert(false); return 1;
	}
}

float GetDeviceComputePower(const cudaDeviceProp& prop)
{
	uint shaders = prop.multiProcessorCount * GetShadersPerMP(prop.major, prop.minor);
	return float(shaders * prop.clockRate) / (1024.0f * 1024.0f);
}

float GetDeviceMemoryPower(const cudaDeviceProp& prop)
{
	return (prop.memoryBusWidth * prop.memoryClockRate) / (8.0f * 1024.0f * 1024.0f);
}

bool GetCudaDevices()
{
	assert(g_cudaDevices.empty());

	// loop over all cuda devices, collect those that we can use
	int deviceCount = 0;
	cudaSafeCallNoSync(cudaGetDeviceCount(&deviceCount));

	for(int device = 0; device < deviceCount; device++)
	{
		cudaDeviceProp prop;
		cudaSafeCallNoSync(cudaGetDeviceProperties(&prop, device));

		bool deviceOK = (prop.major >= 2);
		if(deviceOK)
		{
			float compute = GetDeviceComputePower(prop);
			float memory = GetDeviceMemoryPower(prop);
			printf("Found CUDA device %i: %s, compute %i.%i %.1f G/s, memory %.1f G/s\n", device, prop.name, prop.major, prop.minor, compute, memory);
			g_cudaDevices.push_back(MyCudaDevice(device, compute, memory));
		}
		else
		{
			printf("Skipping CUDA device %i: %s\n", device, prop.name);
		}
	}

	return true;
}

void ClearCudaDevices()
{
	g_cudaDevices.clear();
}


bool InitCudaDevices()
{
	// current device has been set to the primary one by cudaD3D11SetDirect3DDevice
	int deviceCur = -1;
	cudaSafeCallNoSync(cudaGetDevice(&deviceCur));

	// find primary device index, and sum up total compute/memory power
	g_primaryCudaDeviceIndex = -1;
	float totalComputePower = 0.0f;
	float totalMemoryPower = 0.0f;
	for(size_t index = 0; index < g_cudaDevices.size(); index++)
	{
		totalComputePower += g_cudaDevices[index].computePower;
		totalMemoryPower += g_cudaDevices[index].memoryPower;
		if(g_cudaDevices[index].device == deviceCur)
		{
			g_primaryCudaDeviceIndex = (int)index;
		}
	}
	if(g_primaryCudaDeviceIndex == -1)
	{
		printf("ERROR: Did not find primary CUDA device %i among all eligible CUDA devices\n", deviceCur);
		return false;
	}

	if(g_useAllGPUs)
	{
		printf("Using all CUDA devices; %i is primary\n", deviceCur);

		// create and start rendering threads
		float rangeMin = 0.0f, rangeMax = 0.0f;
		for(size_t index = 0; index < g_cudaDevices.size(); index++)
		{
			float myPower = g_cudaDevices[index].computePower / totalComputePower;
			//TODO also/only use memory power?

			rangeMin = rangeMax;
			rangeMax = rangeMin + myPower;
			if(index == g_cudaDevices.size() - 1) rangeMax = 1.0f;
			//rangeMax = float(index + 1) / float(g_cudaDevices.size());

			g_cudaDevices[index].range.Set(rangeMin, rangeMax);

			// don't create extra thread for main GPU
			if(index == g_primaryCudaDeviceIndex) continue;

			g_cudaDevices[index].pThread = new WorkerThread(g_cudaDevices[index].device, g_volume);
			g_cudaDevices[index].pThread->Start();

			// set initial projection params
			g_cudaDevices[index].pThread->SetProjectionParams(g_projParams, g_cudaDevices[index].range);
		}
	}
	else
	{
		// single-GPU case
		printf("Using only CUDA device %i\n", deviceCur);
		g_cudaDevices[g_primaryCudaDeviceIndex].range.Set(0.0f, 1.0f);
	}

	return true;
}

void ShutdownCudaDevices()
{
	for(size_t index = 0; index < g_cudaDevices.size(); index++)
	{
		if(g_cudaDevices[index].pThread == nullptr) continue;

		g_cudaDevices[index].pThread->CancelCurrentTask();
	}
	for(size_t index = 0; index < g_cudaDevices.size(); index++)
	{
		if(g_cudaDevices[index].pThread == nullptr) continue;

		g_cudaDevices[index].pThread->Stop();
		delete g_cudaDevices[index].pThread;
	}

	g_primaryCudaDeviceIndex = -1;
}



void GetMajorWorldPlane(const Vec3f& vecViewX, const Vec3f& vecViewY, const Mat4f& matInv, Vec3f& vecWorldX, Vec3f& vecWorldY)
{
	Vec4f transformedX = matInv * Vec4f(vecViewX, 0.0f);
	Vec4f transformedY = matInv * Vec4f(vecViewY, 0.0f);
	vecWorldX = transformedX.xyz();
	vecWorldY = transformedY.xyz();

	Vec3f normal; crossProd(vecWorldX, vecWorldY, normal);

	// find major axis of normal
	int normalMajorAxis;
	if (fabsf(normal.x()) >= fabsf(normal.y()) && fabsf(normal.x()) >= fabsf(normal.z())) {
		// x axis is maximum
		normalMajorAxis = 0;
	} else if (fabsf(normal.y()) >= fabsf(normal.x()) && fabsf(normal.y()) >= fabsf(normal.z())) {
		// y axis is maximum
		normalMajorAxis = 1;
	} else {
		// z axis is maximum
		normalMajorAxis = 2;
	}

	// make x and y orthogonal to normal
	vecWorldX[normalMajorAxis] = 0.0f;
	vecWorldY[normalMajorAxis] = 0.0f;

	normalize(vecWorldX);
	normalize(vecWorldY);
}

// ============================================================
// GUI
// ============================================================


void TW_CALL Redraw(void *clientData)
{
	g_redraw = true;
}

void TW_CALL Retrace(void *clientData)
{
	g_retrace = true;
}

void TW_CALL SetBoundingBoxToDomainSize(void *clientData)
{
	g_particleTraceParams.m_seedBoxMin = -g_volume.GetVolumeHalfSizeWorld();
	g_particleTraceParams.m_seedBoxSize = 2 * g_volume.GetVolumeHalfSizeWorld();
}


void TW_CALL SetTime(const void *value, void *clientData)
{
	float timeNew = *reinterpret_cast<const float*>(value);
	g_volume.SetCurTime(timeNew);
}

void TW_CALL GetTime(void *value, void *clientData)
{
	*reinterpret_cast<float*>(value) = g_volume.GetCurTime();
}

void TW_CALL SetTimestepIndex(const void *value, void *clientData)
{
	int32 timestepNew = *reinterpret_cast<const int32*>(value);
	float timeNew = float(timestepNew) * g_volume.GetTimeSpacing();
	g_volume.SetCurTime(timeNew);
}

void TW_CALL GetTimestepIndex(void *value, void *clientData)
{
	*reinterpret_cast<uint*>(value) = g_volume.GetCurNearestTimestepIndex();
}

void TW_CALL SetTimeSpacing(const void *value, void *clientData)
{
	float timeSpacingNew = *reinterpret_cast<const float*>(value);
	g_volume.SetTimeSpacing(timeSpacingNew);
}

void TW_CALL GetTimeSpacing(void *value, void *clientData)
{
	*reinterpret_cast<float*>(value) = g_volume.GetTimeSpacing();
}


void TW_CALL SelectFile(void *clientData)
{
	std::string filename;
	if (tum3d::GetFilenameDialog("Select TimeVolume file", "TimeVolume (*.timevol)\0*.timevol\0", filename, false))
	{
		CloseVolumeFile();
		OpenVolumeFile(filename, DXUTGetD3D11Device());
	}
}

void TW_CALL LoadTimestepCallback(void *clientData)
{
	printf("Loading timestep...");
	TimerCPU timer;
	timer.Start();
	g_volume.LoadNearestTimestep();
	timer.Stop();
	printf(" done in %.2f s\n", timer.GetElapsedTimeMS() / 1000.0f);
}

void TW_CALL SelectOutputRawFile(void *clientData)
{
	std::string filename;
	if (tum3d::GetFilenameDialog("Select output file", "Raw (*.raw)\0*.raw\0", filename, true))
	{
		// remove extension
		if(filename.substr(filename.size() - 4) == ".raw")
		{
			filename = filename.substr(0, filename.size() - 4);
		}
		std::vector<std::string> filenames;
		for(int c = 0; c < g_volume.GetChannelCount(); c++)
		{
			std::ostringstream str;
			str << filename << char('X' + c) << ".raw";
			filenames.push_back(str.str());
		}
		g_renderingManager.WriteCurTimestepToRaws(g_volume, filenames);
	}
}

void TW_CALL SelectOutputLA3DFile(void *clientData)
{
	std::string filename;
	if (tum3d::GetFilenameDialog("Select output file", "LargeArray3D (*.la3d)\0*.la3d\0", filename, true))
	{
		// remove extension
		if(filename.substr(filename.size() - 5) == ".la3d")
		{
			filename = filename.substr(0, filename.size() - 5);
		}
		std::vector<std::string> filenames;
		for(int c = 0; c < g_volume.GetChannelCount(); c++)
		{
			std::ostringstream str;
			str << filename << char('X' + c) << ".la3d";
			filenames.push_back(str.str());
		}
		g_renderingManager.WriteCurTimestepToLA3Ds(g_volume, filenames);
	}
}


void TW_CALL ResetView(void *clientData)
{
	g_viewParams.Reset();
	g_rotationX = Vec4f(1, 0, 0, 0);
	g_rotationY = Vec4f(1, 0, 0, 0);
	g_rotation = Vec4f(1, 0, 0, 0);
}


void TW_CALL SetMeasure1(const void *value, void *clientData)
{
	g_raycastParams.m_measure1 = *reinterpret_cast<const eMeasure*>(value);
	g_raycastParams.m_measureScale1 = GetDefaultMeasureScale(g_raycastParams.m_measure1);
}

void TW_CALL GetMeasure1(void *value, void *clientData)
{
	*reinterpret_cast<eMeasure*>(value) = g_raycastParams.m_measure1;
}


void TW_CALL SetMeasure2(const void *value, void *clientData)
{
	g_raycastParams.m_measure2 = *reinterpret_cast<const eMeasure*>(value);
	g_raycastParams.m_measureScale2 = GetDefaultMeasureScale(g_raycastParams.m_measure2);
}

void TW_CALL GetMeasure2(void *value, void *clientData)
{
	*reinterpret_cast<eMeasure*>(value) = g_raycastParams.m_measure2;
}


void TW_CALL SetTwColorMode(const void *value, void *clientData)
{
	g_raycastParams.m_colorMode = *reinterpret_cast<const eColorMode*>(value);
}


void TW_CALL GetTwColorMode(void *value, void *clientData)
{
	*reinterpret_cast<eColorMode*>(value) = g_raycastParams.m_colorMode;
}


void UpdateFilterRadii()
{
	g_filterParams.m_radius.clear();
	for(uint i = 0; i < 3; i++)
	{
		g_filterParams.m_radius.push_back(g_guiFilterRadius[i]);
	}
}

void TW_CALL SetFilterRadius1(const void *value, void *clientData)
{
	g_guiFilterRadius[0] = *reinterpret_cast<const uint*>(value);
	UpdateFilterRadii();
}

void TW_CALL GetFilterRadius1(void *value, void *clientData)
{
	*reinterpret_cast<uint*>(value) = g_guiFilterRadius[0];
}

void TW_CALL SetFilterRadius2(const void *value, void *clientData)
{
	g_guiFilterRadius[1] = *reinterpret_cast<const uint*>(value);
	UpdateFilterRadii();
}

void TW_CALL GetFilterRadius2(void *value, void *clientData)
{
	*reinterpret_cast<uint*>(value) = g_guiFilterRadius[1];
}

void TW_CALL SetFilterRadius3(const void *value, void *clientData)
{
	g_guiFilterRadius[2] = *reinterpret_cast<const uint*>(value);
	UpdateFilterRadii();
}

void TW_CALL GetFilterRadius3(void *value, void *clientData)
{
	*reinterpret_cast<uint*>(value) = g_guiFilterRadius[2];
}


void TW_CALL SaveLinesDialog(void *clientData)
{
	bool bFullscreen = (!DXUTIsWindowed());

	if( bFullscreen ) DXUTToggleFullScreen();

	std::string filename;
	if (tum3d::GetFilenameDialog("Save Lines", "*.linebuf\0*.linebuf", filename, true)) 
	{
		filename = tum3d::RemoveExt(filename) + ".linebuf";
		float posOffset = 0.0f;
		if(g_particleTraceParams.m_upsampledVolumeHack)
		{
			// upsampled volume is offset by half a grid spacing...
			float gridSpacingWorld = 2.0f / float(g_volume.GetVolumeSize().maximum());
			posOffset = 0.5f * gridSpacingWorld;
		}
		if(!g_tracingManager.GetResult()->Write(filename, posOffset))
		{
			printf("Saving lines to file %s failed!\n", filename.c_str());
		}
	}

	if( bFullscreen ) DXUTToggleFullScreen();
}

void TW_CALL LoadLinesDialog(void *clientData)
{
	bool bFullscreen = (!DXUTIsWindowed());

	if( bFullscreen ) DXUTToggleFullScreen();

	std::string filename;
	if (tum3d::GetFilenameDialog("Load Lines", "*.linebuf\0*.linebuf", filename, false))
	{
		LineBuffers* pBuffers = new LineBuffers(DXUTGetD3D11Device());
		if(!pBuffers->Read(filename, g_lineIDOverride))
		{
			printf("Loading lines from file %s failed!\n", filename.c_str());
			delete pBuffers;
		}
		else
		{
			g_lineBuffers.push_back(pBuffers);
		}
	}

	if( bFullscreen ) DXUTToggleFullScreen();

	g_redraw = true;
}

void TW_CALL ClearLinesCallback(void *clientData)
{
	ReleaseLineBuffers();
	g_redraw = true;
}

void TW_CALL LoadBallsDialog(void *clientData)
{
	bool bFullscreen = (!DXUTIsWindowed());

	if( bFullscreen ) DXUTToggleFullScreen();

	std::string filename;
	if (tum3d::GetFilenameDialog("Load Balls", "*.*\0*.*", filename, false))
	{
		BallBuffers* pBuffers = new BallBuffers(DXUTGetD3D11Device());
		if(!pBuffers->Read(filename))
		{
			printf("Loading balls from file %s failed!\n", filename.c_str());
			delete pBuffers;
		}
		else
		{
			g_ballBuffers.push_back(pBuffers);
		}
	}

	if( bFullscreen ) DXUTToggleFullScreen();

	g_redraw = true;
}

void TW_CALL ClearBallsCallback(void *clientData)
{
	ReleaseBallBuffers();
	g_redraw = true;
}


void TW_CALL SetRotation(const void *value, void *clientData)
{
	const float* pRotationNew = reinterpret_cast<const float*>(value);
	g_viewParams.m_rotationQuat.set(pRotationNew[3], pRotationNew[0], pRotationNew[1], pRotationNew[2]);
}

void TW_CALL GetRotation(void *value, void *clientData)
{
	reinterpret_cast<float*>(value)[0] = g_viewParams.m_rotationQuat.y();
	reinterpret_cast<float*>(value)[1] = g_viewParams.m_rotationQuat.z();
	reinterpret_cast<float*>(value)[2] = g_viewParams.m_rotationQuat.w();
	reinterpret_cast<float*>(value)[3] = g_viewParams.m_rotationQuat.x();
}


void TW_CALL SaveScreenshot(void *clientData)
{
	g_saveScreenshot = true;
}

void TW_CALL SaveRenderBufferScreenshot(void *clientData)
{
	g_saveRenderBufferScreenshot = true;
}


void TW_CALL SaveRenderingParamsDialog(void *clientData)
{
	bool bFullscreen = (!DXUTIsWindowed());

	if( bFullscreen ) DXUTToggleFullScreen();

	std::string filename;
	if ( tum3d::GetFilenameDialog("Save Settings", "*.cfg\0*.cfg", filename, true) ) 
	{
		filename = tum3d::RemoveExt(filename) + ".cfg";
		SaveRenderingParams( filename );
	}

	if( bFullscreen ) DXUTToggleFullScreen();
}

void TW_CALL LoadRenderingParamsDialog(void *clientData)
{
	bool bFullscreen = (!DXUTIsWindowed());

	if( bFullscreen ) DXUTToggleFullScreen();

	std::string filename;
	if ( tum3d::GetFilenameDialog("Load Settings", "*.cfg\0*.cfg", filename, false) ) 
		LoadRenderingParams( filename );

	if( bFullscreen ) DXUTToggleFullScreen();	
}


void TW_CALL SetMemLimitCallback(const void *value, void *clientData)
{
	g_volume.GetSystemMemoryUsage().SetSystemMemoryLimitMBytes(*reinterpret_cast<const float*>(value));
}

void TW_CALL GetMemLimitCallback(void *value, void *clientData)
{
	*reinterpret_cast<float*>(value) = g_volume.GetSystemMemoryUsage().GetSystemMemoryLimitMBytes();
}


void TW_CALL BuildFlowGraphCallback(void *clientData)
{
	BuildFlowGraph("flowgraph.txt");
}

void TW_CALL SaveFlowGraphCallback(void *clientData)
{
	SaveFlowGraph();
}

void TW_CALL LoadFlowGraphCallback(void *clientData)
{
	LoadFlowGraph();
}


void TW_CALL SetUseAllGPUs(const void *value, void *clientData)
{
	ReleaseVolumeDependentResources();
	ShutdownCudaDevices();

	g_useAllGPUs = *reinterpret_cast<const bool*>(value);

	InitCudaDevices();
	CreateVolumeDependentResources(DXUTGetD3D11Device());
}

void TW_CALL GetUseAllGPUs(void *value, void *clientData)
{
	*reinterpret_cast<bool*>(value) = g_useAllGPUs;
}


void TW_CALL StartImageSequence(void *clientData)
{
	g_imageSequence.BaseRotationQuat = g_viewParams.m_rotationQuat;
	g_imageSequence.BaseTimestep = g_volume.GetCurNearestTimestepIndex();

	g_imageSequence.FrameCur = 0;
	g_imageSequence.Running = true;
}

void TW_CALL StopImageSequence(void *clientData)
{
	g_imageSequence.Running = false;
	g_imageSequence.FrameCur = 0;
}


void TW_CALL StartProfiler(void *clientData)
{
	cudaProfilerStart();
	cudaGetLastError();
}

void TW_CALL StopProfiler(void *clientData)
{
	cudaProfilerStop();
	cudaGetLastError();
}


void TW_CALL SetEnableBatchAdvectMode(const void *value, void *clientData)
{
	bool enable = *reinterpret_cast<const bool*>(value);
	eAdvectMode mode = eAdvectMode(uint(clientData));

	if(enable) {
		g_batchTraceParams.m_advectModes.insert(mode);
	} else {
		g_batchTraceParams.m_advectModes.erase(mode);
	}
}

void TW_CALL GetEnableBatchAdvectMode(void *value, void *clientData)
{
	eAdvectMode mode = eAdvectMode(uint(clientData));

	bool enabled = (g_batchTraceParams.m_advectModes.count(mode) > 0);

	*reinterpret_cast<bool*>(value) = enabled;
}

void TW_CALL SetEnableBatchFilterMode(const void *value, void *clientData)
{
	bool enable = *reinterpret_cast<const bool*>(value);
	eTextureFilterMode mode = eTextureFilterMode(uint(clientData));

	if(enable) {
		g_batchTraceParams.m_filterModes.insert(mode);
	} else {
		g_batchTraceParams.m_filterModes.erase(mode);
	}
}

void TW_CALL GetEnableBatchFilterMode(void *value, void *clientData)
{
	eTextureFilterMode mode = eTextureFilterMode(uint(clientData));

	bool enabled = (g_batchTraceParams.m_filterModes.count(mode) > 0);

	*reinterpret_cast<bool*>(value) = enabled;
}

void StartBatchTracing()
{
	if(g_batchTrace.VolumeFiles.empty()) {
		return;
	}

	g_tracingManager.CancelTracing();
	CloseVolumeFile();

	// semi-HACK: disable rendering preview
	//g_showPreview = false;

	// create outpath if it doesn't exist yet
	system((string("mkdir \"") + tum3d::GetPath(g_batchTrace.OutPath) + "\"").c_str());

	// save config
	ConfigFile config;
	g_particleTraceParams.WriteConfig(config, true);
	g_batchTraceParams.WriteConfig(config);
	config.Write(g_batchTrace.OutPath + "BatchParams.cfg");

	g_batchTrace.FileCur = 0;
	g_batchTrace.StepCur = 0;
	g_batchTrace.Running = true;

	std::string filenameStats = g_batchTrace.OutPath + "BatchStats.csv";
	g_batchTrace.FileStats.open(filenameStats);
	if(!g_batchTrace.FileStats.good()) {
		printf("WARNING: Failed opening output file %s\n", filenameStats.c_str());
	}
	std::string filenameTimings = g_batchTrace.OutPath + "BatchTimings.csv";
	g_batchTrace.FileTimings.open(filenameTimings);
	if(!g_batchTrace.FileStats.good()) {
		printf("WARNING: Failed opening output file %s\n", filenameTimings.c_str());
	}

	std::vector<std::string> extraColumns;
	extraColumns.push_back("Name");
	extraColumns.push_back("Bonus");
	extraColumns.push_back("Penalty");
	TracingManager::Stats::WriteCSVHeader(g_batchTrace.FileStats, extraColumns);
	TracingManager::Timings::WriteCSVHeader(g_batchTrace.FileTimings, extraColumns);

	printf("\n------ Batch trace started.  File count: %Iu  Step count: %u\n\n", g_batchTrace.VolumeFiles.size(), g_batchTraceParams.GetTotalStepCount());

	OpenVolumeFile(g_batchTrace.VolumeFiles[g_batchTrace.FileCur], DXUTGetD3D11Device());
	if(!LineModeIsTimeDependent(g_particleTraceParams.m_lineMode) && g_volume.IsCompressed()) {
		// semi-HACK: fully load the whole timestep, so we get results consistently without IO
		printf("Loading file %u...", g_batchTrace.FileCur);
		g_volume.LoadNearestTimestep();
		printf("\n");
	}
	if(g_particleTraceParams.HeuristicUseFlowGraph()) {
		LoadOrBuildFlowGraph();
	}
	g_batchTraceParams.ApplyToTraceParams(g_particleTraceParams, g_batchTrace.StepCur);
}

void TW_CALL StartBatchTracingChooseFiles(void *clientData = nullptr)
{
	if(!tum3d::GetMultiFilenameDialog("Choose volume files", "TimeVolume (*.timevol)\0*.timevol\0", g_batchTrace.VolumeFiles))
		return;
	assert(!g_batchTrace.VolumeFiles.empty());

	g_batchTrace.OutPath = tum3d::GetPath(g_batchTrace.VolumeFiles[0]);
	//if(!tum3d::GetPathDialog("Choose output folder", g_batchTrace.OutPath))
	//	return;

	StartBatchTracing();
}

void TW_CALL StopBatchTracing(void *clientData = nullptr)
{
	if(g_batchTrace.Running) {
		printf("\n------ Batch trace stopped.\n\n");
	}

	g_batchTrace.Running = false;
	g_batchTrace.FileCur = 0;
	g_batchTrace.StepCur = 0;
}


void TW_CALL GetNumOMPThreads(void *value, void *clientData)
{
	*reinterpret_cast<uint*>(value) = g_threadCount;
}

void TW_CALL SetNumOMPThreads(const void *value, void *clientData)
{
	g_threadCount = *reinterpret_cast<const uint*>(value);
	omp_set_num_threads(g_threadCount);
}

void TW_CALL LoadSliceTexture(void *clientData)
{
	std::string filename;
	if (tum3d::GetFilenameDialog("Load Texture", "Images (jpg, png, bmp)\0*.png;*.jpg;*.jpeg;*.bmp\0", filename, false)) {
		//release old texture
		SAFE_RELEASE(g_particleRenderParams.m_pSliceTexture);
		//create new texture
		ID3D11Device* pd3dDevice = (ID3D11Device*)clientData;
		std::wstring wfilename(filename.begin(), filename.end());
		ID3D11Resource* tmp = NULL;
		if (!FAILED(DirectX::CreateWICTextureFromFile(pd3dDevice, wfilename.c_str(), &tmp, &g_particleRenderParams.m_pSliceTexture))) {
			std::cout << "Slice texture " << filename << " loaded" << std::endl;
			g_particleRenderParams.m_showSlice = true;
			g_redraw = true;
		}
		else {
			std::cerr << "Failed to load slice texture" << std::endl;
		}
		SAFE_RELEASE(tmp);
	}
}
void TW_CALL LoadColorTexture(void *clientData)
{
	std::string filename;
	if (tum3d::GetFilenameDialog("Load Texture", "Images (jpg, png, bmp)\0*.png;*.jpg;*.jpeg;*.bmp\0", filename, false)) {
		//release old texture
		SAFE_RELEASE(g_particleRenderParams.m_pColorTexture);
		//create new texture
		ID3D11Device* pd3dDevice = (ID3D11Device*)clientData;
		std::wstring wfilename(filename.begin(), filename.end());
		ID3D11Resource* tmp = NULL;
		if (!FAILED(DirectX::CreateWICTextureFromFile(pd3dDevice, wfilename.c_str(), &tmp, &g_particleRenderParams.m_pColorTexture))) {
			std::cout << "Color texture " << filename << " loaded" << std::endl;
			g_particleRenderParams.m_lineColorMode = eLineColorMode::TEXTURE;
			g_redraw = true;
		}
		else {
			std::cerr << "Failed to load color texture" << std::endl;
		}
		SAFE_RELEASE(tmp);
	}
}

void TW_CALL LoadSeedTexture(void *clientData)
{
	std::string filename;
	if (tum3d::GetFilenameDialog("Load Texture", "Images (jpg, png, bmp)\0*.png;*.jpg;*.jpeg;*.bmp\0", filename, false)) {
		//create new texture
		ID3D11Device* pd3dDevice = (ID3D11Device*)clientData;
		std::wstring wfilename(filename.begin(), filename.end());
		ID3D11Resource* res = NULL;
		//ID3D11ShaderResourceView* srv = NULL;
		if (!FAILED(DirectX::CreateWICTextureFromFileEx(pd3dDevice, wfilename.c_str(), 0Ui64, D3D11_USAGE_STAGING, 0, D3D11_CPU_ACCESS_READ, 0, false, &res, NULL))) {
			std::cout << "Seed texture " << filename << " loaded" << std::endl;
			//delete old data
			delete[] g_particleTraceParams.m_seedTexture.m_colors;
			g_particleTraceParams.m_seedTexture.m_colors = NULL;
			//Copy to cpu memory
			ID3D11Texture2D* tex = NULL;
			res->QueryInterface(&tex);
			D3D11_TEXTURE2D_DESC desc;
			tex->GetDesc(&desc);
			g_particleTraceParams.m_seedTexture.m_width = desc.Width;
			g_particleTraceParams.m_seedTexture.m_height = desc.Height;
			g_particleTraceParams.m_seedTexture.m_colors = new unsigned int[desc.Width * desc.Height];
			D3D11_MAPPED_SUBRESOURCE mappedResource;
			ID3D11DeviceContext* context = NULL;
			pd3dDevice->GetImmediateContext(&context);
			if (!FAILED(context->Map(tex, 0, D3D11_MAP_READ, 0, &mappedResource))) {
				for (int y = 0; y < desc.Width; ++y) {
					memcpy(&g_particleTraceParams.m_seedTexture.m_colors[y*desc.Width], ((char*)mappedResource.pData) + (y*mappedResource.RowPitch), sizeof(unsigned int) * desc.Width);
				}
				context->Unmap(tex, 0);
			}
			SAFE_RELEASE(context);
			SAFE_RELEASE(tex);
			//reset color
			g_particleTraceParams.m_seedTexture.m_picked.clear();
			//set seed box to domain
			SetBoundingBoxToDomainSize(clientData);
			std::cout << "Seed texture copied to cpu memory" << std::endl;
		}
		else {
			std::cerr << "Failed to load seed texture" << std::endl;
		}
		//SAFE_RELEASE(srv);
		SAFE_RELEASE(res);
	}
}

void InitTwBars(ID3D11Device* pDevice, UINT uiBBHeight)
{
	TwInit(TW_DIRECT3D11, pDevice);

	TwDefine("GLOBAL iconpos=bottomleft iconalign=horizontal");

	std::ostringstream strFilterModes;
	strFilterModes << GetTextureFilterModeName(eTextureFilterMode(0));
	for(uint i = 1; i < TEXTURE_FILTER_MODE_COUNT; i++) 
		strFilterModes << "," << GetTextureFilterModeName(eTextureFilterMode(i));
	TwType twFilterMode = TwDefineEnumFromString("EnumTextureFilterMode", strFilterModes.str().c_str());

	//Raycaster only supports the first two modes, linear and cubic
	std::ostringstream strFilterModes2;
	strFilterModes2 << GetTextureFilterModeName(eTextureFilterMode(0));
	for (uint i = 1; i < 2; i++)
		strFilterModes2 << "," << GetTextureFilterModeName(eTextureFilterMode(i));
	TwType twFilterMode2 = TwDefineEnumFromString("EnumTextureFilterMode2", strFilterModes2.str().c_str());

	std::ostringstream strRaycastModes;
	strRaycastModes << GetRaycastModeName(eRaycastMode(0));
	for(uint i = 1; i < RAYCAST_MODE_COUNT; i++)
		strRaycastModes << "," << GetRaycastModeName(eRaycastMode(i));
	TwType twRaycastMode = TwDefineEnumFromString("EnumRaycastMode", strRaycastModes.str().c_str());

	std::ostringstream strMeasureModes;
	strMeasureModes << GetMeasureName(eMeasure(0));
	for(uint i = 1; i < MEASURE_COUNT; i++)
		strMeasureModes << "," << GetMeasureName(eMeasure(i));
	TwType twMeasureMode = TwDefineEnumFromString("EnumMeasureMode", strMeasureModes.str().c_str());

	std::ostringstream strColorModes;
	strColorModes << GetColorModeName(eColorMode(0));
	for(uint i = 1; i < COLOR_MODE_COUNT; i++)
		strColorModes << "," << GetColorModeName(eColorMode(i));
	TwType twColorMode = TwDefineEnumFromString("EnumColorMode", strColorModes.str().c_str());

	std::ostringstream strMeasureComputeModes;
	strMeasureComputeModes << GetMeasureComputeModeName(eMeasureComputeMode(0));
	for(uint i = 1; i < MEASURE_COMPUTE_COUNT; i++)
		strMeasureComputeModes << "," << GetMeasureComputeModeName(eMeasureComputeMode(i));
	TwType twMeasureComputeMode = TwDefineEnumFromString("EnumMeasureComputeMode", strMeasureComputeModes.str().c_str());

	std::ostringstream strLineRenderModes;
	strLineRenderModes << GetLineRenderModeName(eLineRenderMode(0));
	for(uint i = 1; i < LINE_RENDER_MODE_COUNT; i++)
		strLineRenderModes << "," << GetLineRenderModeName(eLineRenderMode(i));
	TwType twLineRenderMode = TwDefineEnumFromString("EnumLineRenderMode", strLineRenderModes.str().c_str());

	std::ostringstream strLineColorModes;
	strLineColorModes << GetLineColorModeName(eLineColorMode(0));
	for (uint i = 1; i < LINE_COLOR_MODE_COUNT; i++)
		strLineColorModes << "," << GetLineColorModeName(eLineColorMode(i));
	TwType twLineColorMode = TwDefineEnumFromString("EnumLineColorMode", strLineColorModes.str().c_str());

	std::ostringstream strParticleRenderModes;
	strParticleRenderModes << GetParticleRenderModeName(eParticleRenderMode(0));
	for (uint i = 1; i < PARTICLE_RENDER_MODE_COUNT; i++)
		strParticleRenderModes << "," << GetParticleRenderModeName(eParticleRenderMode(i));
	TwType twParticleRenderMode = TwDefineEnumFromString("EnumParticleRenderMode", strParticleRenderModes.str().c_str());

	std::ostringstream strAdvectModes;
	strAdvectModes << GetAdvectModeName(eAdvectMode(0));
	for(uint i = 1; i < ADVECT_MODE_COUNT; i++) 
		strAdvectModes << "," << GetAdvectModeName(eAdvectMode(i));
	TwType twAdvectMode = TwDefineEnumFromString("EnumAdvectMode", strAdvectModes.str().c_str());

	std::ostringstream strLineModes;
	strLineModes << GetLineModeName(eLineMode(0));
	for(uint i = 1; i < LINE_MODE_COUNT; i++) 
		strLineModes << "," << GetLineModeName(eLineMode(i));
	TwType twLineMode = TwDefineEnumFromString("EnumLineMode", strLineModes.str().c_str());

	std::ostringstream strHeatMapNormalizationModes;
	strHeatMapNormalizationModes << GetHeatMapNormalizationModeName(eHeatMapNormalizationMode(0));
	for (uint i = 1; i < HEAT_MAP_NORMALIZATION_MODE_COUNT; i++)
		strHeatMapNormalizationModes << "," << GetHeatMapNormalizationModeName(eHeatMapNormalizationMode(i));
	TwType twHeatMapNormalizationMode = TwDefineEnumFromString("EnumHeatMapNormalizationMode", strHeatMapNormalizationModes.str().c_str());

	// MAIN BAR
	g_pTwBarMain = TwNewBar("Main");
	std::ostringstream ss;
	int iHeight = max(static_cast<int>(uiBBHeight)-40, 200);
	ss << "Main label='Main' size='260 " << iHeight << "' position='10 10' text=light";
	TwDefine(ss.str().c_str());

	// "global" params: data set and timestep
	TwAddButton(g_pTwBarMain, "SelectFile", SelectFile, nullptr, "label='Select file'");
	TwAddVarCB(g_pTwBarMain, "Time", TW_TYPE_FLOAT, SetTime, GetTime, nullptr, "label='Time' min=0 max=0 precision=4");
	TwAddVarCB(g_pTwBarMain, "Timestep", TW_TYPE_UINT32, SetTimestepIndex, GetTimestepIndex, nullptr, "label='Timestep' min=0 max=0");
	int32 c = g_volume.GetTimestepCount() - 1;
	TwSetParam(g_pTwBarMain, "Timestep", "max", TW_PARAM_INT32, 1, &c);
	TwAddVarCB(g_pTwBarMain, "TimeSpacing", TW_TYPE_FLOAT, SetTimeSpacing, GetTimeSpacing, nullptr, "label='Time Spacing' min=0.001 step=0.001 precision=3");
	TwAddButton(g_pTwBarMain, "LoadTimestep", LoadTimestepCallback, nullptr, "label='Preload Timestep' group=MiscSettings");
	TwAddButton(g_pTwBarMain, "WriteTimestepRaw", SelectOutputRawFile, nullptr, "label='Write as .raw' group=MiscSettings");
	TwAddButton(g_pTwBarMain, "WriteTimestepLA3D", SelectOutputLA3DFile, nullptr, "label='Write as .la3d' group=MiscSettings");

	TwAddSeparator(g_pTwBarMain, "", " group=MiscSettings");
	TwAddButton(g_pTwBarMain, "SaveSettings", SaveRenderingParamsDialog, nullptr, "label='Save Settings' group=MiscSettings");
	TwAddButton(g_pTwBarMain, "LoadSettings", LoadRenderingParamsDialog, nullptr, "label='Load Settings' group=MiscSettings");

	TwAddSeparator(g_pTwBarMain, "", " group=MiscSettings");
	TwAddVarCB(g_pTwBarMain, "MemLimit", TW_TYPE_FLOAT, SetMemLimitCallback, GetMemLimitCallback, nullptr, "label='Mem Usage Limit (MB)' precision=1 group=MiscSettings");

	TwDefine("Main/MiscSettings label='Misc. Settings' opened=false");

	// ray casting params
	TwAddSeparator(g_pTwBarMain, "", "");

	TwAddVarRW(g_pTwBarMain, "RaycastEnabled",	TW_TYPE_BOOLCPP,	&g_raycastParams.m_raycastingEnabled,		"label='Enable' group=RayCast");

	TwAddVarRW(g_pTwBarMain, "Source",			twMeasureComputeMode,&g_raycastParams.m_measureComputeMode,		"label='Measure Computation' group=RayCast");

	TwAddVarRW(g_pTwBarMain, "Interpolation",	twFilterMode2,		&g_raycastParams.m_textureFilterMode,		"label='Interpolation' group=RayCast");
	TwAddVarRW(g_pTwBarMain, "VisMode",			twRaycastMode,		&g_raycastParams.m_raycastMode,				"label='Raycast Mode' group=RayCast");
	TwAddVarCB(g_pTwBarMain, "Measure1",		twMeasureMode,		SetMeasure1, GetMeasure1, nullptr,			"label='Measure 1' group=RayCast");
	TwAddVarRW(g_pTwBarMain, "MeasureScale1",	TW_TYPE_FLOAT,		&g_raycastParams.m_measureScale1,			"label='Measure 1 Scale' step=0.01 precision=6 group=RayCast");
	TwAddVarCB(g_pTwBarMain, "Measure2",		twMeasureMode,		SetMeasure2, GetMeasure2, nullptr,			"label='Measure 2' group=RayCast");
	TwAddVarRW(g_pTwBarMain, "MeasureScale2",	TW_TYPE_FLOAT,		&g_raycastParams.m_measureScale2,			"label='Measure 2 Scale' step=0.01 precision=6 group=RayCast");

	TwAddVarRW(g_pTwBarMain, "SampleRate",		TW_TYPE_FLOAT,		&g_raycastParams.m_sampleRate,				"label='Sample Rate' min=0.01 max=20.0 step=0.01 group=RayCast");
	TwAddVarRW(g_pTwBarMain, "Density",			TW_TYPE_FLOAT,		&g_raycastParams.m_density,					"label='Density' min=0.1 max=1000.0 step=0.1 group=RayCast");
	TwAddVarRW(g_pTwBarMain, "Iso1",			TW_TYPE_FLOAT,		&g_raycastParams.m_isoValue1,				"label='IsoValue1' step=0.01 precision=4 group=IsoSettings");
	TwAddVarRW(g_pTwBarMain, "IsoColor1",		TW_TYPE_COLOR4F,	&g_raycastParams.m_isoColor1,				"label='IsoColor1' group=IsoSettings");
	TwAddVarRW(g_pTwBarMain, "Iso2",			TW_TYPE_FLOAT,		&g_raycastParams.m_isoValue2,				"label='IsoValue2' step=0.01 precision=4 group=IsoSettings");
	TwAddVarRW(g_pTwBarMain, "IsoColor2",		TW_TYPE_COLOR4F,	&g_raycastParams.m_isoColor2,				"label='IsoColor2' group=IsoSettings");
	TwAddVarRW(g_pTwBarMain, "Iso3",			TW_TYPE_FLOAT,		&g_raycastParams.m_isoValue3,				"label='IsoValue3' step=0.01 precision=4 group=IsoSettings");
	TwAddVarRW(g_pTwBarMain, "IsoColor3",		TW_TYPE_COLOR4F,	&g_raycastParams.m_isoColor3,				"label='IsoColor3' group=IsoSettings");
	TwAddVarCB(g_pTwBarMain, "Color Mode",		twColorMode,		SetTwColorMode, GetTwColorMode, nullptr,	"label='Color Mode' group=IsoSettings");
	TwDefine("Main/IsoSettings label='Iso Settings' group=RayCast");

	TwAddVarCB(g_pTwBarMain, "FilterRadius1",	TW_TYPE_UINT32,		SetFilterRadius1, GetFilterRadius1, nullptr,"label='Radius 1' min=0 max=247 step=1 group=Filter");
	TwAddVarCB(g_pTwBarMain, "FilterRadius2",	TW_TYPE_UINT32,		SetFilterRadius2, GetFilterRadius2, nullptr,"label='Radius 2' min=0 max=247 step=1 group=Filter");
	TwAddVarCB(g_pTwBarMain, "FilterRadius3",	TW_TYPE_UINT32,		SetFilterRadius3, GetFilterRadius3, nullptr,"label='Radius 3' min=0 max=247 step=1 group=Filter");
	TwAddVarRW(g_pTwBarMain, "FilterOffset",	TW_TYPE_UINT32,		&g_raycastParams.m_filterOffset,			"label='Filter (Scale) Offset' max=2 group=Filter");
	TwDefine("Main/Filter group=RayCast opened=false");

	TwAddVarRW(g_pTwBarMain, "ShowClipBox",		TW_TYPE_BOOLCPP,	&g_bRenderClipBox,							"label='Show Clip Box (Red)' group=RayCast");
	TwAddVarRW(g_pTwBarMain, "ClipBoxMinX",		TW_TYPE_FLOAT,		&g_raycastParams.m_clipBoxMin.x(),			"label='X' min=-1 max=1 step=0.01 group=ClipBoxMin");
	TwAddVarRW(g_pTwBarMain, "ClipBoxMinY",		TW_TYPE_FLOAT,		&g_raycastParams.m_clipBoxMin.y(),			"label='Y' min=-1 max=1 step=0.01 group=ClipBoxMin");
	TwAddVarRW(g_pTwBarMain, "ClipBoxMinZ",		TW_TYPE_FLOAT,		&g_raycastParams.m_clipBoxMin.z(),			"label='Z' min=-1 max=1 step=0.01 group=ClipBoxMin");
	TwDefine("Main/ClipBoxMin label='Clip Box Min' group=RayCast opened=false");
	TwAddVarRW(g_pTwBarMain, "ClipBoxMaxX",		TW_TYPE_FLOAT,		&g_raycastParams.m_clipBoxMax.x(),			"label='X' min=-1 max=1 step=0.01 group=ClipBoxMax");
	TwAddVarRW(g_pTwBarMain, "ClipBoxMaxY",		TW_TYPE_FLOAT,		&g_raycastParams.m_clipBoxMax.y(),			"label='Y' min=-1 max=1 step=0.01 group=ClipBoxMax");
	TwAddVarRW(g_pTwBarMain, "ClipBoxMaxZ",		TW_TYPE_FLOAT,		&g_raycastParams.m_clipBoxMax.z(),			"label='Z' min=-1 max=1 step=0.01 group=ClipBoxMax");
	TwDefine("Main/ClipBoxMax label='Clip Box Max' group=RayCast opened=false");

	TwDefine("Main/RayCast label='Ray Casting' opened=false");


	// particle params
	TwAddVarRW(g_pTwBarMain, "Verbose",			TW_TYPE_BOOLCPP,	&g_tracingManager.GetVerbose(),				"label='Verbose' group=ParticleTrace");
	TwAddVarRW(g_pTwBarMain, "ShowSeedBox",		TW_TYPE_BOOLCPP,	&g_bRenderSeedBox,							"label='Show Seed Box (Green)' group=ParticleTrace");
	TwAddVarRW(g_pTwBarMain, "CPUTrace",		TW_TYPE_BOOLCPP,	&g_particleTraceParams.m_cpuTracing,		"label='CPU Tracing' group=ParticleTraceAdvanced");
	TwAddVarCB(g_pTwBarMain, "CPUThreads",		TW_TYPE_UINT32,		SetNumOMPThreads, GetNumOMPThreads, nullptr, "label='# CPU Threads' group=ParticleTraceAdvanced");
	TwAddVarRW(g_pTwBarMain, "BrickSlotsMax",	TW_TYPE_UINT32,		&g_tracingManager.GetBrickSlotCountMax(),	"label='Max Brick Slot Count' group=ParticleTraceAdvanced");
	TwAddVarRW(g_pTwBarMain, "TimeSlotsMax",	TW_TYPE_UINT32,		&g_tracingManager.GetTimeSlotCountMax(),	"label='Max Time Slot Count' group=ParticleTraceAdvanced");
	TwAddSeparator(g_pTwBarMain, "", "group=ParticleTrace");
	TwAddButton(g_pTwBarMain,"SeedToDomain", SetBoundingBoxToDomainSize, NULL,                                  "label='Set Seed Box to Domain' group=ParticleTrace");
	TwAddButton(g_pTwBarMain, "LoadSeedTexture", LoadSeedTexture, pDevice, "label='Load Seed Texture' group=ParticleTrace");
	TwAddButton(g_pTwBarMain, "SeedTextureInfo", NULL, NULL, "label='Press P to pick the color under the mouse' group=ParticleTrace");
	TwAddVarRW(g_pTwBarMain, "SeedBoxMinX",		TW_TYPE_FLOAT,		&g_particleTraceParams.m_seedBoxMin.x(),	"label='X' min=-1 max=1 step=0.01 precision=3 group=SeedBoxMin");
	TwAddVarRW(g_pTwBarMain, "SeedBoxMinY",		TW_TYPE_FLOAT,		&g_particleTraceParams.m_seedBoxMin.y(),	"label='Y' min=-1 max=1 step=0.01 precision=3 group=SeedBoxMin");
	TwAddVarRW(g_pTwBarMain, "SeedBoxMinZ",		TW_TYPE_FLOAT,		&g_particleTraceParams.m_seedBoxMin.z(),	"label='Z' min=-1 max=1 step=0.01 precision=3 group=SeedBoxMin");
	TwDefine("Main/SeedBoxMin label='Seed Box Min' group=ParticleTrace opened=false");
	TwAddVarRW(g_pTwBarMain, "SeedBoxSizeX",	TW_TYPE_FLOAT,		&g_particleTraceParams.m_seedBoxSize.x(),	"label='X' min=0 max=2 step=0.01 precision=3 group=SeedBoxSize");
	TwAddVarRW(g_pTwBarMain, "SeedBoxSizeY",	TW_TYPE_FLOAT,		&g_particleTraceParams.m_seedBoxSize.y(),	"label='Y' min=0 max=2 step=0.01 precision=3 group=SeedBoxSize");
	TwAddVarRW(g_pTwBarMain, "SeedBoxSizeZ",	TW_TYPE_FLOAT,		&g_particleTraceParams.m_seedBoxSize.z(),	"label='Z' min=0 max=2 step=0.01 precision=3 group=SeedBoxSize");
	TwDefine("Main/SeedBoxSize label='Seed Box Size' group=ParticleTrace opened=false");
	TwAddSeparator(g_pTwBarMain, "", "group=ParticleTrace");
	TwAddVarRW(g_pTwBarMain, "AdvectMode",		twAdvectMode,		&g_particleTraceParams.m_advectMode,		"label='Advection' group=ParticleTrace");
	TwAddVarRW(g_pTwBarMain, "DenseOutput",		TW_TYPE_BOOLCPP,	&g_particleTraceParams.m_enableDenseOutput,	"label='Dense Output' group=ParticleTrace");
	TwAddVarRW(g_pTwBarMain, "TraceInterpol",	twFilterMode,		&g_particleTraceParams.m_filterMode,		"label='Interpolation' group=ParticleTrace");
	//TwAddVarRW(g_pTwBarMain, "LineMode",		twLineMode,			&g_particleTraceParams.m_lineMode,			"label='Line Mode' group=ParticleTrace");
	TwAddVarCB(g_pTwBarMain, "LineMode",		twLineMode,
		[](const void* valueToSet, void* clientData) {
			g_particleTraceParams.m_lineMode = *((eLineMode*)valueToSet);
			if (LineModeIsIterative(g_particleTraceParams.m_lineMode)) {
				//force render mode to 'particles' and preview to true
				g_particleRenderParams.m_lineRenderMode = eLineRenderMode::LINE_RENDER_PARTICLES;
				g_showPreview = true;
			}
			if (LineModeGenerateAlwaysNewSeeds(g_particleTraceParams.m_lineMode)) {
				//new seeds are always generated -> color-by-line does not make sense, switch to color by age
				if (g_particleRenderParams.m_lineColorMode == eLineColorMode::LINE_ID)
					g_particleRenderParams.m_lineColorMode = eLineColorMode::AGE;
			}
		},
		[](void* value, void* clientData) {
			*((eLineMode*) value) = g_particleTraceParams.m_lineMode;
		}, 
		NULL, "label='Line Mode' group=ParticleTrace");
	TwAddVarRW(g_pTwBarMain, "LineCount",		TW_TYPE_UINT32,		&g_particleTraceParams.m_lineCount,			"label='Line Count' group=ParticleTrace");
	TwAddVarRW(g_pTwBarMain, "LineLengthMax",	TW_TYPE_UINT32,		&g_particleTraceParams.m_lineLengthMax,		"label='Max Line Length' min=2 group=ParticleTrace");
	TwAddVarRW(g_pTwBarMain, "LineAgeMax",		TW_TYPE_FLOAT,		&g_particleTraceParams.m_lineAgeMax,		"label='Max Line Age' min=0 precision=2 step=0.1 group=ParticleTrace");
	TwAddVarRW(g_pTwBarMain, "MinVelocity",		TW_TYPE_FLOAT,		&g_particleTraceParams.m_minVelocity,		"label='Min Velocity' min=0 precision=2 step=0.01 group=ParticleTrace");
	TwAddVarRW(g_pTwBarMain, "ParticlesPerSecond",TW_TYPE_FLOAT,	&g_particleTraceParams.m_particlesPerSecond,"label='Particles per second' min=0 step=0.01 group=ParticleTrace");
	TwAddVarRW(g_pTwBarMain, "AdvectDeltaT",	TW_TYPE_FLOAT,		&g_particleTraceParams.m_advectDeltaT,		"label='Advection Delta T' min=0 precision=5 step=0.001 group=ParticleTrace");
	TwAddButton(g_pTwBarMain, "SeedManyParticles", [](void* data) {
		g_tracingManager.SeedManyParticles();
	}, NULL, "label='Seed many particles' group=ParticleTrace");
	TwAddVarRW(g_pTwBarMain, "CellChangeThreshold", TW_TYPE_FLOAT,  &g_particleTraceParams.m_cellChangeThreshold, "label='Cell Change Time Threshold' min=0 precision=5 step=0.001 group=ParticleTrace");
	TwAddVarRW(g_pTwBarMain, "AdvectErrorTol",	TW_TYPE_FLOAT,		&g_particleTraceParams.m_advectErrorTolerance,"label='Advection Error Tolerance (Voxels)' min=0 precision=5 step=0.001 group=ParticleTraceAdvanced");
	TwAddVarRW(g_pTwBarMain, "AdvectDeltaTMin",	TW_TYPE_FLOAT,		&g_particleTraceParams.m_advectDeltaTMin,	"label='Advection Delta T Min' min=0 precision=5 step=0.001 group=ParticleTraceAdvanced");
	TwAddVarRW(g_pTwBarMain, "AdvectDeltaTMax",	TW_TYPE_FLOAT,		&g_particleTraceParams.m_advectDeltaTMax,	"label='Advection Delta T Max' min=0 precision=5 step=0.001 group=ParticleTraceAdvanced");
	TwAddVarRW(g_pTwBarMain, "AdvectStepsMax",	TW_TYPE_UINT32,		&g_particleTraceParams.m_advectStepsPerRound,"label='Advect Steps per Round' min=0 group=ParticleTraceAdvanced");
	TwAddVarRW(g_pTwBarMain, "PurgeTimeout",	TW_TYPE_UINT32,		&g_particleTraceParams.m_purgeTimeoutInRounds,"label='Brick Purge Timeout' min=0 group=ParticleTraceAdvanced");
	TwAddVarRW(g_pTwBarMain, "HeuristicBonus",	TW_TYPE_FLOAT,		&g_particleTraceParams.m_heuristicBonusFactor,"label='Heuristic: Bonus Factor' min=0 step=0.01 precision=3 group=ParticleTraceAdvanced");
	TwAddVarRW(g_pTwBarMain, "HeuristicPenalty",TW_TYPE_FLOAT,		&g_particleTraceParams.m_heuristicPenaltyFactor,"label='Heuristic: Penalty Factor' min=0 step=0.01 precision=3 group=ParticleTraceAdvanced");
	TwAddVarRW(g_pTwBarMain, "HeuristicFlags",	TW_TYPE_UINT32,		&g_particleTraceParams.m_heuristicFlags,	"label='Heuristic: Flags' group=ParticleTraceAdvanced");
	TwAddVarRW(g_pTwBarMain, "OutputPosDiff",	TW_TYPE_FLOAT,		&g_particleTraceParams.m_outputPosDiff,		"label='Output Pos Diff (Voxels)' min=0 precision=5 step=0.001 group=ParticleTraceAdvanced");
	TwAddVarRW(g_pTwBarMain, "OutputTimeDiff",	TW_TYPE_FLOAT,		&g_particleTraceParams.m_outputTimeDiff,	"label='Output Time Diff' min=0 precision=5 step=0.001 group=ParticleTraceAdvanced");
	TwAddVarRW(g_pTwBarMain, "WaitForDisk",		TW_TYPE_BOOLCPP,	&g_particleTraceParams.m_waitForDisk,		"label='Wait for Disk' group=ParticleTraceAdvanced");
	TwAddVarRW(g_pTwBarMain, "Prefetching",		TW_TYPE_BOOLCPP,	&g_particleTraceParams.m_enablePrefetching,	"label='Prefetching' group=ParticleTraceAdvanced");
	TwAddVarRW(g_pTwBarMain, "UpsampledVolume",	TW_TYPE_BOOLCPP,	&g_particleTraceParams.m_upsampledVolumeHack,"label='Upsampled Volume Hack' group=ParticleTraceAdvanced");
	TwDefine("Main/ParticleTraceAdvanced label='Advanced Settings' group=ParticleTrace opened=false");
	TwAddSeparator(g_pTwBarMain, "", "group=ParticleTrace");
	TwAddButton(g_pTwBarMain, "Retrace", Retrace, nullptr, "label='Retrace' group=ParticleTrace");
	TwAddVarRW(g_pTwBarMain, "TracingPaused", TW_TYPE_BOOLCPP, &g_particleTracingPaused, "label='Paused (SPACE)' group=ParticleTrace key=SPACE");
	TwAddSeparator(g_pTwBarMain, "", "group=ParticleTrace");

	// IO
	TwAddButton(g_pTwBarMain, "SaveLines", SaveLinesDialog, nullptr, "label='Save Traced Lines' group=ParticleTraceIO");
	TwAddButton(g_pTwBarMain, "LoadLines", LoadLinesDialog, nullptr, "label='Load Lines' group=ParticleTraceIO");
	TwAddVarRW(g_pTwBarMain,  "LineID", TW_TYPE_INT32, &g_lineIDOverride,"label='Line ID Override' min=-1 group=ParticleTraceIO");
	TwAddButton(g_pTwBarMain, "ClearLines", ClearLinesCallback, nullptr, "label='Clear Loaded Lines' group=ParticleTraceIO");
	TwAddButton(g_pTwBarMain, "LoadBalls", LoadBallsDialog, nullptr, "label='Load Balls' group=ParticleTraceIO");
	TwAddVarRW(g_pTwBarMain, "BallsRadius", TW_TYPE_FLOAT, &g_ballRadius, "label='Ball Radius' group=ParticleTraceIO");
	TwAddButton(g_pTwBarMain, "ClearBalls", ClearBallsCallback, nullptr, "label='Clear Loaded Balls' group=ParticleTraceIO");
	TwAddSeparator(g_pTwBarMain, "", "group=ParticleTraceIO");
	TwAddButton(g_pTwBarMain, "FlowGraph", BuildFlowGraphCallback, nullptr, "label='Build Flow Graph' group=ParticleTraceIO");
	TwAddButton(g_pTwBarMain, "SaveFlowGraph", SaveFlowGraphCallback, nullptr, "label='Save Flow Graph' group=ParticleTraceIO");
	TwAddButton(g_pTwBarMain, "LoadFlowGraph", LoadFlowGraphCallback, nullptr, "label='Load Flow Graph' group=ParticleTraceIO");
	TwDefine("Main/ParticleTraceIO label='Extra IO' group=ParticleTrace opened=false");

	TwDefine("Main/ParticleTrace label='Particle Tracing'");

	//Rendering

	TwAddVarRW(g_pTwBarMain, "ParticleEnabled",	TW_TYPE_BOOLCPP,	&g_particleRenderParams.m_linesEnabled,		"label='Enable' group=ParticleRender");
	//TwAddVarRW(g_pTwBarMain, "LightDirView",	TW_TYPE_DIR3F,		&g_particleRenderParams.m_lightDirView,		"label='Light Dir (View Space)' group=ParticleRender");
	TwAddVarRW(g_pTwBarMain, "LineRenderMode",	twLineRenderMode,	&g_particleRenderParams.m_lineRenderMode,	"label='Line Render Mode' group=ParticleRender");
	TwAddVarRW(g_pTwBarMain, "RibbonWidth",		TW_TYPE_FLOAT,		&g_particleRenderParams.m_ribbonWidth,		"label='Ribbon Width' step=0.01 group=ParticleRender");
	TwAddVarRW(g_pTwBarMain, "TubeRadius",		TW_TYPE_FLOAT,		&g_particleRenderParams.m_tubeRadius,		"label='Tube Radius' min=0 step=0.01 group=ParticleRender");
	TwAddVarRW(g_pTwBarMain, "ParticleSize",    TW_TYPE_FLOAT,      &g_particleRenderParams.m_particleSize,     "label='Particle Size' group=ParticleRender min=0 step=0.01");
	TwAddVarRW(g_pTwBarMain, "RadiusFromVel",	TW_TYPE_BOOLCPP,	&g_particleRenderParams.m_tubeRadiusFromVelocity, "label='Display Velocity' group=ParticleRender"); //"label='Tube Radius from Velocity' group=ParticleRender");
	TwAddVarRW(g_pTwBarMain, "ReferenceVel",	TW_TYPE_FLOAT,		&g_particleRenderParams.m_referenceVelocity,"label='Reference Velocity' min=0.001 step=0.01 precision=3 group=ParticleRender");
	TwAddSeparator(g_pTwBarMain, "", "group=ParticleRender");
	TwAddVarRW(g_pTwBarMain, "ParticleRenderMode", twParticleRenderMode, &g_particleRenderParams.m_particleRenderMode, "label='Particle Render Mode' group=ParticleRender");
	TwAddVarRW(g_pTwBarMain, "ParticleTransparency", TW_TYPE_FLOAT, &g_particleRenderParams.m_particleTransparency, "label='Particle Transparency' group=ParticleRender min=0 max=1 step=0.01");
	TwAddVarRW(g_pTwBarMain, "SortParticles",	TW_TYPE_BOOLCPP,	&g_particleRenderParams.m_sortParticles,	"label='Sort Particles' group=ParticleRender");
	TwAddSeparator(g_pTwBarMain, "", "group=ParticleRender");
	TwAddVarRW(g_pTwBarMain, "ColorMode",		twLineColorMode,	&g_particleRenderParams.m_lineColorMode,	"label='Color Mode' group=ParticleRender");
	TwAddVarRW(g_pTwBarMain, "Color0",			TW_TYPE_COLOR3F,	&g_particleRenderParams.m_color0,			"label='Color 0' group=ParticleRender");
	TwAddVarRW(g_pTwBarMain, "Color1",			TW_TYPE_COLOR3F,	&g_particleRenderParams.m_color1,			"label='Color 1' group=ParticleRender");
	TwAddButton(g_pTwBarMain, "LoadColorTexture", LoadColorTexture, pDevice,                                    "label='Load Color Texture' group=ParticleRender");
	TwAddVarRW(g_pTwBarMain, "RenderMeasure",   twMeasureMode,      &g_particleRenderParams.m_measure,          "label='Measure' group=ParticleRender");
	TwAddVarRW(g_pTwBarMain, "RenderMeasureScale", TW_TYPE_FLOAT,	&g_particleRenderParams.m_measureScale,		"label='Measure Scale' step=0.01 min=0 precision=6 group=ParticleRender");
	TwAddSeparator(g_pTwBarMain, "", "group=ParticleRender");
	TwAddVarRW(g_pTwBarMain, "TimeStripes",		TW_TYPE_BOOLCPP,	&g_particleRenderParams.m_timeStripes,		"label='Time Stripes' group=ParticleRender");
	TwAddVarRW(g_pTwBarMain, "TimeStripeLength",TW_TYPE_FLOAT,		&g_particleRenderParams.m_timeStripeLength,	"label='Time Stripe Length' min=0.001 step=0.001 group=ParticleRender");
	TwAddSeparator(g_pTwBarMain, "", "group=ParticleRender");
	TwAddButton(g_pTwBarMain, "LoadSliceTexture", LoadSliceTexture, pDevice, "label='Load Slice Texture' group=ParticleRender");
	TwAddVarRW(g_pTwBarMain, "ShowSlices",      TW_TYPE_BOOLCPP,    &g_particleRenderParams.m_showSlice,        "label='Show Slice' group=ParticleRender");
	TwAddVarRW(g_pTwBarMain, "SlicePosition",   TW_TYPE_FLOAT,      &g_particleRenderParams.m_slicePosition,    "label='Slice Position' step=0.001 group=ParticleRender");
	TwAddVarRW(g_pTwBarMain, "SliceAlpha",      TW_TYPE_FLOAT,      &g_particleRenderParams.m_sliceAlpha,       "label='Slice Transparency' step=0.01 min=0 max=1 group=ParticleRender");
	TwDefine("Main/ParticleRender label='Rendering'");

	// Heat Map
	TwAddVarRW(g_pTwBarMain, "HeatMap_EnableRecording", TW_TYPE_BOOLCPP, &g_heatMapParams.m_enableRecording,
		"label='Enable Recording' group='Heat Map'");
	TwAddVarRW(g_pTwBarMain, "HeatMap_EnableRendering", TW_TYPE_BOOLCPP, &g_heatMapParams.m_enableRendering,
		"label='Enable Rendering' group='Heat Map'");
	TwAddVarRW(g_pTwBarMain, "HeatMap_AutoReset", TW_TYPE_BOOLCPP, &g_heatMapParams.m_autoReset,
		"label='Auto Reset' group='Heat Map'");
	TwAddButton(g_pTwBarMain, "HeatMap_Reset", [](void* data) {
			g_heatMapManager.ClearChannels();
			g_redraw = true;
		}, 
		NULL, "label='Reset (C)' group='Heat Map' key=c");
	TwAddVarRW(g_pTwBarMain, "HeatMap_Normalize", twHeatMapNormalizationMode, &g_heatMapParams.m_normalizationMode,
		"label='Normalization' group='Heat Map'");
	TwAddVarRW(g_pTwBarMain, "HeatMap_StepSize", TW_TYPE_FLOAT, &g_heatMapParams.m_stepSize,
		"label='Step Size' min=0.001 step=0.001 group='Heat Map'");
	TwAddVarRW(g_pTwBarMain, "HeatMap_DensityScale", TW_TYPE_FLOAT, &g_heatMapParams.m_densityScale,
		"label='Density Scale' min=0 step=0.01 group='Heat Map'");
	TwAddVarRO(g_pTwBarMain, "HeatMap_Channel1", TW_TYPE_COLOR32, &g_heatMapParams.m_renderedChannels[0],
		"label='First displayed channel (1)' group='Heat Map'");
	TwAddVarRO(g_pTwBarMain, "HeatMap_Channel2", TW_TYPE_COLOR32, &g_heatMapParams.m_renderedChannels[1],
		"label='Second displayed channel (2)' group='Heat Map'");
	TwAddVarRW(g_pTwBarMain, "HeatMap_EnableIsosurface", TW_TYPE_BOOLCPP, &g_heatMapParams.m_isosurface,
		"label='Isosurface Rendering' group='Heat Map'");
	TwAddVarRW(g_pTwBarMain, "HeatMap_Isovalue", TW_TYPE_FLOAT, &g_heatMapParams.m_isovalue,
		"label='Isovalue' min=0 max=1 step=0.001 group='Heat Map'");

	// bounding boxes
	TwAddVarRW(g_pTwBarMain, "ShowDomainBox",	TW_TYPE_BOOLCPP,	&g_bRenderDomainBox,						"label='Show Domain Box (Blue)' group=MiscRendering");
	TwAddVarRW(g_pTwBarMain, "ShowBrickBoxes",	TW_TYPE_BOOLCPP,	&g_bRenderBrickBoxes,						"label='Show Brick Boxes (Light Blue)' group=MiscRendering");


	// view params
	TwAddSeparator(g_pTwBarMain, "", "");
	TwAddVarRW(g_pTwBarMain, "Supersample",		TW_TYPE_FLOAT,		&g_renderBufferSizeFactor,		"label='SuperSample Factor' min=0.5 max=8 step=0.5 group=MiscRendering");
	TwAddVarRW(g_pTwBarMain, "LookAtX",			TW_TYPE_FLOAT,		&g_viewParams.m_lookAt.x(),		"label='X' group=LookAt");
	TwAddVarRW(g_pTwBarMain, "LookAtY",			TW_TYPE_FLOAT,		&g_viewParams.m_lookAt.y(),		"label='Y' group=LookAt");
	TwAddVarRW(g_pTwBarMain, "LookAtZ",			TW_TYPE_FLOAT,		&g_viewParams.m_lookAt.z(),		"label='Z' group=LookAt");
	TwAddVarRW(g_pTwBarMain, "ViewDistance",	TW_TYPE_FLOAT,		&g_viewParams.m_viewDistance,	"label='View Distance' min=0 step=0.01 group=MiscRendering");
	TwDefine("Main/LookAt label='LookAt' opened=false group=MiscRendering");
	TwAddVarCB(g_pTwBarMain, "Rotation",		TW_TYPE_QUAT4F,	SetRotation, GetRotation, nullptr,	"label='Rotation' opened=false group=MiscRendering");
	TwAddButton(g_pTwBarMain, "ResetView", ResetView, nullptr, "label='Reset View' group=MiscRendering");
	TwAddVarRW(g_pTwBarMain, "BackColor",		TW_TYPE_COLOR3F,	&g_backgroundColor,	"label='Background Color' opened=false group=MiscRendering");
	TwAddVarRW(g_pTwBarMain, "StereoEnable",	TW_TYPE_BOOLCPP,	&g_stereoParams.m_stereoEnabled,"label='Enable' group=Stereo");
	TwAddVarRW(g_pTwBarMain, "StereoEyeDist",	TW_TYPE_FLOAT,		&g_stereoParams.m_eyeDistance,	"label='Eye Distance' min=0 step=0.001 group=Stereo");
	TwDefine("Main/Stereo label='Stereo 3D' opened=false group=MiscRendering");
	TwDefine("Main/MiscRendering label='Misc Rendering Settings' opened=false");

	// screenshot
	TwAddButton(g_pTwBarMain, "SaveScreenshot", SaveScreenshot, nullptr, "label='Save Screenshot' key=s");
	TwAddButton(g_pTwBarMain, "SaveRBScreenshot", SaveRenderBufferScreenshot, nullptr, "label='Save RenderBuffer' key=r");


	// other stuff
	TwAddSeparator(g_pTwBarMain, "", "");
	TwAddVarRW(g_pTwBarMain, "Preview", TW_TYPE_BOOLCPP, &g_showPreview, "label='Rendering Preview'");
	//TwAddVarCB(g_pTwBarMain, "UseAllGPUs", TW_TYPE_BOOLCPP, SetUseAllGPUs, GetUseAllGPUs, nullptr, "label='Use all GPUs'");

	TwAddSeparator(g_pTwBarMain, "", "");
	TwAddButton(g_pTwBarMain, "Redraw", Redraw, nullptr, "label='Redraw'");

	TwAddSeparator(g_pTwBarMain, "", "");
	TwAddButton(g_pTwBarMain, "StartProfiler", StartProfiler, nullptr, "label='Start CUDA Profiler'");
	TwAddButton(g_pTwBarMain, "StopProfiler", StopProfiler, nullptr, "label='Stop CUDA Profiler'");


	// IMAGE SEQUENCE RECORDER BAR
	g_pTwBarImageSequence = TwNewBar("ImageSequence");
	TwDefine("ImageSequence label='Image Sequence' size='260 180' text=light iconified=true");

	TwAddVarRW(g_pTwBarImageSequence, "FrameCount", TW_TYPE_INT32, &g_imageSequence.FrameCount, "label='Frame Count' min=1");
	TwAddVarRW(g_pTwBarImageSequence, "AngleInc", TW_TYPE_FLOAT, &g_imageSequence.AngleInc, "label='Rotation per Frame' step=0.1");
	TwAddVarRW(g_pTwBarImageSequence, "ViewDistInc", TW_TYPE_FLOAT, &g_imageSequence.ViewDistInc, "label='Distance offset per Frame' step=0.001");
	TwAddVarRW(g_pTwBarImageSequence, "FramesPerTimestep", TW_TYPE_INT32, &g_imageSequence.FramesPerTimestep, "label='Frames per Timestep' min=1");
	TwAddVarRW(g_pTwBarImageSequence, "Record", TW_TYPE_BOOLCPP, &g_imageSequence.Record, "label='Record'");
	TwAddVarRW(g_pTwBarImageSequence, "FromRenderbuffer", TW_TYPE_BOOLCPP, &g_imageSequence.FromRenderBuffer, "label='Grab RenderBuffer'");
	TwAddButton(g_pTwBarImageSequence, "StartSequence", StartImageSequence, nullptr, "label='Start'");
	TwAddButton(g_pTwBarImageSequence, "StopSequence", StopImageSequence, nullptr, "label='Stop'");
	TwAddVarRO(g_pTwBarImageSequence, "FrameCur", TW_TYPE_INT32, &g_imageSequence.FrameCur, "label='Current Frame'");


	// BATCH TRACING BAR
	g_pTwBarBatchTrace = TwNewBar("BatchTracing");
	TwDefine("BatchTracing label='Batch Tracing' size='260 480' text=light iconified=true");

	for(uint i = 0; i < ADVECT_MODE_COUNT; i++)
	{
		std::ostringstream strName;
		strName << "AdvectMode" << i;
		std::ostringstream strDef;
		strDef << "label='" << GetAdvectModeName(eAdvectMode(i)) << "' group=Advection";
		TwAddVarCB(g_pTwBarBatchTrace, strName.str().c_str(), TW_TYPE_BOOLCPP, SetEnableBatchAdvectMode, GetEnableBatchAdvectMode, (void*)(i), strDef.str().c_str());
	}
	for(uint i = 0; i < TEXTURE_FILTER_MODE_COUNT; i++)
	{
		std::ostringstream strName;
		strName << "FilterMode" << i;
		std::ostringstream strDef;
		strDef << "label='" << GetTextureFilterModeName(eTextureFilterMode(i)) << "' group=Filtering";
		TwAddVarCB(g_pTwBarBatchTrace, strName.str().c_str(), TW_TYPE_BOOLCPP, SetEnableBatchFilterMode, GetEnableBatchFilterMode, (void*)(i), strDef.str().c_str());
	}
	TwAddVarRW(g_pTwBarBatchTrace, "DeltaTMin", TW_TYPE_FLOAT, &g_batchTraceParams.m_deltaTMin, "label='Delta T Min' min=0 precision=5 step=0.001");
	TwAddVarRW(g_pTwBarBatchTrace, "DeltaTMax", TW_TYPE_FLOAT, &g_batchTraceParams.m_deltaTMax, "label='Delta T Max' min=0 precision=5 step=0.001");
	TwAddVarRW(g_pTwBarBatchTrace, "ErrorToleranceMin", TW_TYPE_FLOAT, &g_batchTraceParams.m_errorToleranceMin, "label='Error Tolerance Min' min=0 precision=5 step=0.001");
	TwAddVarRW(g_pTwBarBatchTrace, "ErrorToleranceMax", TW_TYPE_FLOAT, &g_batchTraceParams.m_errorToleranceMax, "label='Error Tolerance Max' min=0 precision=5 step=0.001");
	TwAddVarRW(g_pTwBarBatchTrace, "QualitySteps", TW_TYPE_UINT32, &g_batchTraceParams.m_qualityStepCount, "label='Quality Steps'");
	TwAddVarRW(g_pTwBarBatchTrace, "HeuristicFactorMin", TW_TYPE_FLOAT, &g_batchTraceParams.m_heuristicFactorMin, "label='Heuristic Factor Min' precision=1 step=1");
	TwAddVarRW(g_pTwBarBatchTrace, "HeuristicFactorMax", TW_TYPE_FLOAT, &g_batchTraceParams.m_heuristicFactorMax, "label='Heuristic Factor Max' precision=1 step=1");
	TwAddVarRW(g_pTwBarBatchTrace, "HeuristicSteps", TW_TYPE_UINT32, &g_batchTraceParams.m_heuristicStepCount, "label='Heuristic Steps'");
	TwAddVarRW(g_pTwBarBatchTrace, "HeuristicBPSeparate", TW_TYPE_BOOLCPP, &g_batchTraceParams.m_heuristicBPSeparate, "label='Heuristic B/P Separate'");
	TwAddVarRW(g_pTwBarBatchTrace, "WriteLinebufs", TW_TYPE_BOOLCPP, &g_batchTrace.WriteLinebufs, "label='Write .linebuf files'");
	TwAddButton(g_pTwBarBatchTrace, "StartBatch", StartBatchTracingChooseFiles, nullptr, "label='Start'");
	TwAddButton(g_pTwBarBatchTrace, "StopBatch", StopBatchTracing, nullptr, "label='Stop'");
}



// ============================================================
// DXUT Callbacks
// ============================================================


//--------------------------------------------------------------------------------------
// Reject any D3D11 devices that aren't acceptable by returning false
//--------------------------------------------------------------------------------------
bool CALLBACK IsD3D11DeviceAcceptable( const CD3D11EnumAdapterInfo *AdapterInfo, UINT Output, const CD3D11EnumDeviceInfo *DeviceInfo,
                                       DXGI_FORMAT BackBufferFormat, bool bWindowed, void* pUserContext )
{
	bool deviceAcceptable = false;

	IDXGIFactory *pFactory;
	if(FAILED(CreateDXGIFactory(__uuidof(IDXGIFactory), (void**)(&pFactory) )))
		return false;

	// get a candidate DXGI adapter
	IDXGIAdapter* pAdapter = 0;
	if(FAILED(pFactory->EnumAdapters(AdapterInfo->AdapterOrdinal, &pAdapter))) {
		pFactory->Release();
		return false;
	}

	// Check if adapter is CUDA capable
	// query to see if there exists a corresponding compute device
	cudaError err;
	int cudaDevice;
	err = cudaD3D11GetDevice(&cudaDevice, pAdapter);
	if (err == cudaSuccess) {
		cudaDeviceProp prop;
		if(cudaSuccess != cudaGetDeviceProperties(&prop, cudaDevice)) {
			exit(-1);
		}
		if(prop.major >= 2)
			deviceAcceptable = true;
	}

	pAdapter->Release();
	pFactory->Release();

	// clear any errors we got while querying invalid compute devices
	cudaGetLastError();

	return deviceAcceptable;
}


//--------------------------------------------------------------------------------------
// Called right before creating a device, allowing the app to modify the device settings as needed
//--------------------------------------------------------------------------------------
bool CALLBACK ModifyDeviceSettings( DXUTDeviceSettings* pDeviceSettings, void* pUserContext )
{
	return true;
}


//--------------------------------------------------------------------------------------
// Create any D3D11 resources that aren't dependent on the back buffer
//--------------------------------------------------------------------------------------
HRESULT CALLBACK OnD3D11CreateDevice( ID3D11Device* pd3dDevice, const DXGI_SURFACE_DESC* pBackBufferSurfaceDesc, void* pUserContext )
{
	//wprintf(L"Device: %s\n", DXUTGetDeviceStats());

#if 0
	//According to http://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__D3D11__DEPRECATED.html,
	//this is no longer required
	cudaError error = cudaD3D11SetDirect3DDevice(pd3dDevice);
	if(cudaSuccess != error) {
		printf("cudaD3D11SetDirect3DDevice returned error %u: %s\n", uint(error), cudaGetErrorString(error));
		return E_FAIL;
	}
#endif

	if(!InitCudaDevices()) {
		printf("InitCudaDevices returned false");
		return E_FAIL;
	}

	// don't page-lock brick data - CUDA doesn't seem to like when we page-lock lots of smallish mem areas..
	//g_volume.SetPageLockBrickData(true);


	HRESULT hr;

	if(FAILED(hr = g_screenEffect.Create(pd3dDevice))) {
		return hr;
	}

	if(FAILED(hr = g_progressBarEffect.Create(pd3dDevice))) {
		return hr;
	}


	if(g_volume.IsOpen()) {
		// this creates the cudaCompress instance etc
		if(FAILED(hr = CreateVolumeDependentResources(pd3dDevice))) {
			return hr;
		}
	}


	InitTwBars(pd3dDevice, pBackBufferSurfaceDesc->Height);

	if(FAILED(hr = g_tfEdt.onCreateDevice( pd3dDevice ))) {
		return hr;
	}
	g_particleRenderParams.m_pTransferFunction = g_tfEdt.getSRV(TF_LINE_MEASURES);
	g_heatMapParams.m_pTransferFunction = g_tfEdt.getSRV(TF_HEAT_MAP);
	cudaSafeCall(cudaGraphicsD3D11RegisterResource(&g_pTfEdtSRVCuda, g_tfEdt.getTexture(TF_RAYTRACE), cudaGraphicsRegisterFlagsNone));

	//TracingBenchmark bench;
	//bench.RunBenchmark(g_particleTraceParams, 64, 2048, 5, 0);
	//bench.RunBenchmark(g_particleTraceParams, 64, 2048, 5, 1);
	//exit(42);

	return S_OK;
}


//--------------------------------------------------------------------------------------
// Create any D3D11 resources that depend on the back buffer
//--------------------------------------------------------------------------------------
HRESULT CALLBACK OnD3D11ResizedSwapChain( ID3D11Device* pd3dDevice, IDXGISwapChain* pSwapChain,
                                          const DXGI_SURFACE_DESC* pBackBufferSurfaceDesc, void* pUserContext )
{
	g_redraw = true;

	g_windowSize.x() = pBackBufferSurfaceDesc->Width;
	g_windowSize.y() = pBackBufferSurfaceDesc->Height;


	ResizeRenderBuffer(pd3dDevice);


	HRESULT hr;

	// create texture to hold last finished raycasted image
	D3D11_TEXTURE2D_DESC desc;
	desc.ArraySize = 1;
	desc.BindFlags = D3D11_BIND_RENDER_TARGET;
	desc.CPUAccessFlags = 0;
	desc.Format = pBackBufferSurfaceDesc->Format;
	desc.MipLevels = 1;
	desc.MiscFlags = 0;
	desc.SampleDesc.Count = 1;
	desc.SampleDesc.Quality = 0;
	desc.Usage = D3D11_USAGE_DEFAULT;
	desc.Width = g_windowSize.x();
	desc.Height = g_windowSize.y();
	// create texture for the last finished image from the raycaster
	hr = pd3dDevice->CreateTexture2D(&desc, nullptr, &g_pRaycastFinishedTex);
	if(FAILED(hr)) return hr;
	hr = pd3dDevice->CreateRenderTargetView(g_pRaycastFinishedTex, nullptr, &g_pRaycastFinishedRTV);
	if(FAILED(hr)) return hr;


	// create staging texture for taking screenshots
	desc.ArraySize = 1;
	desc.BindFlags = 0;
	desc.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
	desc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
	desc.MipLevels = 1;
	desc.MiscFlags = 0;
	desc.SampleDesc.Count = 1;
	desc.SampleDesc.Quality = 0;
	desc.Usage = D3D11_USAGE_STAGING;
	desc.Width = g_windowSize.x();
	desc.Height = g_windowSize.y();
	hr = pd3dDevice->CreateTexture2D(&desc, nullptr, &g_pStagingTex);
	if(FAILED(hr)) return hr;


	// GUI
	TwWindowSize(g_windowSize.x(), g_windowSize.y());
	// can't set fontsize before window has been initialized, so do it here instead of init()...
	TwDefine("GLOBAL fontsize=3 fontresizable=false");

	// main gui on left side
	int posMain[2] = { 10, 10 };
	int sizeMain[2];
	TwGetParam(g_pTwBarMain, nullptr, "size", TW_PARAM_INT32, 2, sizeMain);
	sizeMain[1] = max(static_cast<int>(pBackBufferSurfaceDesc->Height) - 40, 200);

	TwSetParam(g_pTwBarMain, nullptr, "position", TW_PARAM_INT32, 2, posMain);
	TwSetParam(g_pTwBarMain, nullptr, "size", TW_PARAM_INT32, 2, sizeMain);

	// image sequence recorder in upper right corner
	int sizeImgSeq[2];
	TwGetParam(g_pTwBarImageSequence, nullptr, "size", TW_PARAM_INT32, 2, sizeImgSeq);
	int posImgSeq[2] = { max(10, (int)g_windowSize.x() - sizeImgSeq[0] - 10), 10 };
	TwSetParam(g_pTwBarImageSequence, nullptr, "position", TW_PARAM_INT32, 2, posImgSeq);

	g_tfEdt.onResizeSwapChain( pBackBufferSurfaceDesc->Width, pBackBufferSurfaceDesc->Height );

	// batch tracing in middle right
	int sizeBatch[2];
	TwGetParam(g_pTwBarBatchTrace, nullptr, "size", TW_PARAM_INT32, 2, sizeBatch);
	int posBatch[2] = { max(10, (int)g_windowSize.x() - sizeBatch[0] - 10), max(10, ((int)g_windowSize.y() - sizeBatch[1]) / 2 - 10) };
	TwSetParam(g_pTwBarBatchTrace, nullptr, "position", TW_PARAM_INT32, 2, posBatch);

	g_tfEdt.onResizeSwapChain( pBackBufferSurfaceDesc->Width, pBackBufferSurfaceDesc->Height );


	return S_OK;
}


// save a screenshot from the framebuffer
void SaveScreenshot(ID3D11DeviceContext* pd3dImmediateContext, const std::string& filename)
{
	ID3D11Resource* pSwapChainTex;
	DXUTGetD3D11RenderTargetView()->GetResource(&pSwapChainTex);
	pd3dImmediateContext->CopyResource(g_pStagingTex, pSwapChainTex);
	SAFE_RELEASE(pSwapChainTex);

	D3D11_MAPPED_SUBRESOURCE mapped = { 0 };
	pd3dImmediateContext->Map(g_pStagingTex, 0, D3D11_MAP_READ, 0, &mapped);

	stbi_write_png(filename.c_str(), g_windowSize.x(), g_windowSize.y(), 4, mapped.pData, mapped.RowPitch);

	pd3dImmediateContext->Unmap(g_pStagingTex, 0);
}

// save a screenshot from the (possibly higher-resolution) render buffer
void SaveRenderBufferScreenshot(ID3D11DeviceContext* pd3dImmediateContext, const std::string& filename)
{
	if(!g_renderingManager.IsCreated()) return;

	bool singleGPU = (!g_useAllGPUs || (g_cudaDevices.size() == 1));
	//ID3D11Texture2D* pSrcTex = (singleGPU ? g_renderingManager.GetRaycastTex() : g_pRenderBufferTempTex);
	//HACK: for now, save just the opaque tex - actually need to blend here...
	ID3D11Texture2D* pSrcTex = (singleGPU ? g_renderingManager.GetOpaqueTex() : g_pRenderBufferTempTex);
	pd3dImmediateContext->CopyResource(g_pRenderBufferStagingTex, pSrcTex);

	D3D11_MAPPED_SUBRESOURCE mapped = { 0 };
	pd3dImmediateContext->Map(g_pRenderBufferStagingTex, 0, D3D11_MAP_READ, 0, &mapped);

	stbi_write_png(filename.c_str(), g_projParams.m_imageWidth, g_projParams.m_imageHeight, 4, mapped.pData, mapped.RowPitch);

	pd3dImmediateContext->Unmap(g_pRenderBufferStagingTex, 0);
}



float Luminance(const Vec3f& color)
{
	return 0.2126f * color.x() + 0.7152f * color.y() + 0.0722f * color.z();
}


//--------------------------------------------------------------------------------------
// Render the scene using the D3D11 device
//--------------------------------------------------------------------------------------
void CALLBACK OnD3D11FrameRender( ID3D11Device* pd3dDevice, ID3D11DeviceContext* pd3dImmediateContext,
                                  double fTime, float fElapsedTime, void* pUserContext )
{
	//fflush(stdout);

	//FIXME have to reset these on destroy/create device...
	static bool s_isFiltering = false;
	static bool s_isTracing = false;
	static bool s_isRendering = false;



	clock_t curTime = clock();

	// Clear render target and the depth stencil
	float ClearColor[4] = { 0.0f, 0.0f, 0.0f, 0.0f };

	ID3D11RenderTargetView* pRTV = DXUTGetD3D11RenderTargetView();
	ID3D11DepthStencilView* pDSV = DXUTGetD3D11DepthStencilView();
	pd3dImmediateContext->ClearRenderTargetView( pRTV, ClearColor );
	pd3dImmediateContext->ClearDepthStencilView( pDSV, D3D11_CLEAR_DEPTH, 1.0f, 0 );


	static std::string volumeFilePrev = "";
	bool volumeChanged = (g_volume.GetFilename() != volumeFilePrev);
	volumeFilePrev = g_volume.GetFilename();
	g_redraw = g_redraw || volumeChanged;
	g_retrace = g_retrace || volumeChanged;


	static int timestepPrev = -1;
	bool timestepChanged = (g_volume.GetCurNearestTimestepIndex() != timestepPrev);
	timestepPrev = g_volume.GetCurNearestTimestepIndex();
	g_redraw = g_redraw || timestepChanged;
	g_retrace = g_retrace || timestepChanged;


	static float renderBufferSizeFactorPrev = 0.0f;
	bool renderBufferSizeChanged = (g_renderBufferSizeFactor != renderBufferSizeFactorPrev);
	renderBufferSizeFactorPrev = g_renderBufferSizeFactor;
	if(renderBufferSizeChanged)
	{
		ResizeRenderBuffer(pd3dDevice);
		g_lastRenderParamsUpdate = curTime;
	}


	static bool renderDomainBoxPrev = g_bRenderDomainBox;
	static bool renderBrickBoxesPrev = g_bRenderBrickBoxes;
	static bool renderClipBoxPrev = g_bRenderClipBox;
	static bool renderSeedBoxPrev = g_bRenderSeedBox;
	if(renderDomainBoxPrev != g_bRenderDomainBox || renderBrickBoxesPrev != g_bRenderBrickBoxes || renderClipBoxPrev != g_bRenderClipBox || renderSeedBoxPrev != g_bRenderSeedBox)
	{
		renderDomainBoxPrev = g_bRenderDomainBox;
		renderBrickBoxesPrev = g_bRenderBrickBoxes;
		renderClipBoxPrev = g_bRenderClipBox;
		renderSeedBoxPrev = g_bRenderSeedBox;
		g_redraw = true;
	}


	static ProjectionParams projParamsPrev;
	bool projParamsChanged = (g_projParams != projParamsPrev);
	projParamsPrev = g_projParams;
	g_redraw = g_redraw || projParamsChanged;

	static Range1D rangePrev;
	bool rangeChanged = (g_cudaDevices[g_primaryCudaDeviceIndex].range != rangePrev);
	rangePrev = g_cudaDevices[g_primaryCudaDeviceIndex].range;
	g_redraw = g_redraw || rangeChanged;

	if(projParamsChanged || rangeChanged)
	{
		// cancel rendering
		g_renderingManager.CancelRendering();
		if(g_useAllGPUs)
		{
			for(size_t i = 0; i < g_cudaDevices.size(); i++)
			{
				if(g_cudaDevices[i].pThread == nullptr) continue;
				g_cudaDevices[i].pThread->CancelRendering();
			}
		}

		// forward the new params
		g_renderingManager.SetProjectionParams(g_projParams, g_cudaDevices[g_primaryCudaDeviceIndex].range);
		if(g_useAllGPUs)
		{
			for(size_t i = 0; i < g_cudaDevices.size(); i++)
			{
				if(g_cudaDevices[i].pThread == nullptr) continue;
				g_cudaDevices[i].pThread->SetProjectionParams(g_projParams, g_cudaDevices[i].range);
			}
		}
		g_lastRenderParamsUpdate = curTime;
	}


	static StereoParams stereoParamsPrev;
	bool stereoParamsChanged = (g_stereoParams != stereoParamsPrev);
	stereoParamsPrev = g_stereoParams;
	g_redraw = g_redraw || stereoParamsChanged;

	if(stereoParamsChanged)
	{
		g_lastRenderParamsUpdate = curTime;
	}

	static ViewParams viewParamsPrev;
	bool viewParamsChanged = (g_viewParams != viewParamsPrev);
	viewParamsPrev = g_viewParams;
	g_redraw = g_redraw || viewParamsChanged;

	if(viewParamsChanged)
	{
		g_lastRenderParamsUpdate = curTime;
	}


	if(g_tfEdt.getTimestamp() != g_tfTimestamp) 
	{
		g_redraw = true;
		g_lastRenderParamsUpdate = curTime;
		g_tfTimestamp = g_tfEdt.getTimestamp();
	}


	// Get raycast params from the TF Editor's UI
	g_raycastParams.m_alphaScale				= g_tfEdt.getAlphaScale(TF_RAYTRACE);
	g_raycastParams.m_transferFunctionRangeMin	= g_tfEdt.getTfRangeMin(TF_RAYTRACE);
	g_raycastParams.m_transferFunctionRangeMax	= g_tfEdt.getTfRangeMax(TF_RAYTRACE);
	g_particleRenderParams.m_transferFunctionRangeMin = g_tfEdt.getTfRangeMin(TF_LINE_MEASURES);
	g_particleRenderParams.m_transferFunctionRangeMax = g_tfEdt.getTfRangeMax(TF_LINE_MEASURES);
	g_heatMapParams.m_tfAlphaScale = g_tfEdt.getAlphaScale(TF_HEAT_MAP);
	g_heatMapParams.m_tfRangeMin = g_tfEdt.getTfRangeMin(TF_HEAT_MAP);
	g_heatMapParams.m_tfRangeMax = g_tfEdt.getTfRangeMax(TF_HEAT_MAP);

	static FilterParams filterParamsPrev;
	bool filterParamsChanged = (g_filterParams != filterParamsPrev);
	filterParamsPrev = g_filterParams;

	// clear filtered bricks if something relevant changed
	if(volumeChanged || timestepChanged || filterParamsChanged)
	{
		g_filteringManager.ClearResult();
		s_isFiltering = false;
		g_redraw = true;
	}

	static ParticleTraceParams particleTraceParamsPrev;
	bool particleTraceParamsChanged = g_particleTraceParams.hasChangesForRetracing(particleTraceParamsPrev);
	//bool particleTraceParamsChanged = (g_particleTraceParams != particleTraceParamsPrev);
    bool seedBoxChanged = (g_particleTraceParams.m_seedBoxMin != particleTraceParamsPrev.m_seedBoxMin || g_particleTraceParams.m_seedBoxSize != particleTraceParamsPrev.m_seedBoxSize);
	particleTraceParamsPrev = g_particleTraceParams;
	g_retrace = g_retrace || particleTraceParamsChanged;
    g_redraw = g_redraw || seedBoxChanged;

	// heat map parameters
	static HeatMapParams heatMapParamsPrev;
	static bool heatMapDoubleRedraw = false;
	bool debugHeatMapPrint = false;
	//g_retrace = g_retrace || g_heatMapParams.HasChangesForRetracing(heatMapParamsPrev, g_particleTraceParams);
	//g_redraw = g_redraw || g_heatMapParams.HasChangesForRedrawing(heatMapParamsPrev);
	if (g_heatMapParams.HasChangesForRetracing(heatMapParamsPrev, g_particleTraceParams)) {
		g_retrace = true;
		std::cout << "heat map has changes for retracing" << std::endl;
		debugHeatMapPrint = true;
		heatMapDoubleRedraw = true;
	}
	if (g_heatMapParams.HasChangesForRedrawing(heatMapParamsPrev)) {
		g_redraw = true;
		std::cout << "heat map has changes for redrawing" << std::endl;
		debugHeatMapPrint = true;
		heatMapDoubleRedraw = true;
	}
	heatMapParamsPrev = g_heatMapParams;
	g_heatMapManager.SetParams(g_heatMapParams);
	if (debugHeatMapPrint) {
#if 0
		g_heatMapManager.DebugPrintParams();
#endif
	}
	if (heatMapDoubleRedraw && !g_redraw) {
		//Hack: For some reasons, the heat map manager only applies changes after the second rendering
		//Or does the RenderManager not update the render targets?
		g_redraw = true;
		heatMapDoubleRedraw = false;
	}

	if(particleTraceParamsChanged)
	{
		g_lastTraceParamsUpdate = curTime;
		g_lastRenderParamsUpdate = curTime;
	}


	// clear particle tracer if something relevant changed
	if(g_retrace && s_isTracing)
	{
		g_tracingManager.CancelTracing();
		s_isTracing = false;
		g_tracingManager.ClearResult();
		g_redraw = true;
		g_lastRenderParamsUpdate = curTime;
	}



	static RaycastParams raycastParamsPrev = g_raycastParams;
	bool raycastParamsChanged = (g_raycastParams != raycastParamsPrev);
	raycastParamsPrev = g_raycastParams;
	g_redraw = g_redraw || raycastParamsChanged;
	
	if(raycastParamsChanged)
	{
		g_lastTraceParamsUpdate = curTime;
	}

	static ParticleRenderParams particleRenderParamsPrev = g_particleRenderParams;
	bool particleRenderParamsChanged = (g_particleRenderParams != particleRenderParamsPrev);
	particleRenderParamsPrev = g_particleRenderParams;
	g_redraw = g_redraw || particleRenderParamsChanged;
	
	//if(particleRenderParamsChanged)
	//{
	//	g_lastTraceParamsUpdate = curTime;
	//}


	Vec3f camPos = g_viewParams.GetCameraPosition();


	static std::vector<uint> s_timestampLastUpdate; // last time the result image from each GPU was taken


	if(!g_volume.IsOpen()) goto NoVolumeLoaded;

	//-----------------------------------------------------------
	// FILTERING
	//-----------------------------------------------------------

	// start filtering if required
	bool needFilteredBricks = g_filterParams.HasNonZeroRadius();
	if(needFilteredBricks && !s_isFiltering && g_filteringManager.GetResultCount() == 0)
	{
		g_renderingManager.CancelRendering();
		if(g_useAllGPUs)
		{
			for(size_t i = 0; i < g_cudaDevices.size(); i++)
			{
				if(g_cudaDevices[i].pThread == nullptr) continue;
				g_cudaDevices[i].pThread->CancelRendering();
			}
		}
		// release other resources - we'll need a lot of memory for filtering
		g_tracingManager.ReleaseResources();
		g_renderingManager.ReleaseResources();

		assert(g_filteringManager.IsCreated());

		s_isFiltering = g_filteringManager.StartFiltering(g_volume, g_filterParams);
	}

	//-----------------------------------------------------------
	// TRACING
	//-----------------------------------------------------------

	// set parameters even if tracing is currently enabled.
	// This allows changes to the parameters in the particle mode, even if they are currently running
	if (!g_retrace && s_isTracing) {
		g_tracingManager.SetParams(g_particleTraceParams);
	}
	// start particle tracing if required
	float timeSinceTraceUpdate = float(curTime - g_lastTraceParamsUpdate) / float(CLOCKS_PER_SEC);
	bool traceDelayPassed = (timeSinceTraceUpdate >= g_startWorkingDelay);
	//bool traceStartNow = !s_isFiltering && g_particleRenderParams.m_linesEnabled && traceDelayPassed; //TODO: enable tracing also when rendering is disabled?
	bool traceStartNow = !s_isFiltering && traceDelayPassed;
	if(g_retrace && traceStartNow)
	{
		g_renderingManager.CancelRendering();
		if(g_useAllGPUs)
		{
			for(size_t i = 0; i < g_cudaDevices.size(); i++)
			{
				if(g_cudaDevices[i].pThread == nullptr) continue;
				g_cudaDevices[i].pThread->CancelRendering();
			}
		}
		// release other resources - we're gonna need a lot of memory
		g_renderingManager.ReleaseResources();
		g_filteringManager.ReleaseResources();

		assert(g_tracingManager.IsCreated());

		g_tracingManager.ClearResult();

		g_tracingManager.ReleaseResources();

		s_isTracing = g_tracingManager.StartTracing(g_volume, g_particleTraceParams, g_flowGraph);
		g_timerTracing.Start();

		//notify the heat map manager
		g_heatMapManager.SetVolumeAndReset(g_volume);

		g_retrace = false;
	}


	const std::vector<const TimeVolumeIO::Brick*>& bricksToLoad =
		s_isFiltering ? g_filteringManager.GetBricksToLoad() :
		(s_isTracing ? g_tracingManager.GetBricksToLoad() : g_renderingManager.GetBricksToLoad());
	//TODO when nothing's going on, load bricks in any order? (while memory is available etc..)
	g_volume.UpdateLoadingQueue(bricksToLoad);
	g_volume.UnloadLRUBricks();

	//Check if tracing is done and if so, start rendering
	if(s_isTracing && !g_particleTracingPaused)
	{
		//std::cout << "Trace" << std::endl;
		bool finished = g_tracingManager.Trace();
		if (finished || LineModeIsIterative(g_particleTraceParams.m_lineMode)) {
			g_heatMapManager.ProcessLines(g_tracingManager.GetResult());
		}

		if(finished)
		{
			s_isTracing = false;
			g_timerTracing.Stop();
			g_redraw = true;
		}
		else if(g_showPreview)
		{
			g_tracingManager.BuildIndexBuffer();
			g_redraw = true;
		}
	}
	else if(s_isFiltering)
	{
		bool finished = g_filteringManager.Filter();

		if(finished)
		{
			s_isFiltering = false;
			g_redraw = true;
		}
	}


	//-----------------------------------------------------------
	// RENDERING
	//-----------------------------------------------------------

	bool renderingUpdated = false;
	if (g_redraw)
	{
		// cancel rendering of current image
		g_renderingManager.CancelRendering();
		if(g_useAllGPUs)
		{
			for(size_t i = 0; i < g_cudaDevices.size(); i++)
			{
				if(g_cudaDevices[i].pThread == nullptr) continue;
				g_cudaDevices[i].pThread->CancelRendering();
			}
		}

		// release other resources if possible
		if(!s_isFiltering)
		{
			g_filteringManager.ReleaseResources();
		}
		if(!s_isTracing)
		{
			g_tracingManager.ReleaseResources();
		}

		// while tracing/filtering is in progress, don't start raycasting
		bool linesOnly = s_isTracing || s_isFiltering || !g_raycastParams.m_raycastingEnabled;

		cudaArray* pTfArray = nullptr;
		bool needTF = (g_raycastParams.m_raycastingEnabled && RaycastModeNeedsTransferFunction(g_raycastParams.m_raycastMode));
		if(needTF)
		{
			cudaSafeCall(cudaGraphicsMapResources(1, &g_pTfEdtSRVCuda));
			cudaSafeCall(cudaGraphicsSubResourceGetMappedArray(&pTfArray, g_pTfEdtSRVCuda, 0, 0));
		}
		std::vector<LineBuffers*> lineBuffers = g_lineBuffers;
		LineBuffers* pTracedLines = g_tracingManager.GetResult().get();
		if(pTracedLines != nullptr)
		{
			lineBuffers.push_back(pTracedLines);
		}
		RenderingManager::eRenderState state = g_renderingManager.StartRendering(
			g_volume, g_filteringManager.GetResults(),
			g_viewParams, g_stereoParams,
			g_bRenderDomainBox, g_bRenderClipBox, g_bRenderSeedBox, g_bRenderBrickBoxes,
			g_particleTraceParams, g_particleRenderParams,
			lineBuffers, linesOnly,
			g_ballBuffers, g_ballRadius,
			&g_heatMapManager,
			g_raycastParams, pTfArray);


		if(state == RenderingManager::STATE_ERROR)
		{
			printf("RenderingManager::StartRendering returned STATE_ERROR.\n");
			s_isRendering = false;
		}
		else
		{
			s_isRendering = true; // set to true even if STATE_DONE - other GPUs might have something to do
			renderingUpdated = true;

			if(g_useAllGPUs)
			{
				for(size_t i = 0; i < g_cudaDevices.size(); i++)
				{
					if(g_cudaDevices[i].pThread != nullptr)
					{
						int primaryDevice = g_cudaDevices[g_primaryCudaDeviceIndex].device;
						g_cudaDevices[i].pThread->StartRendering(g_filteringManager.GetResults(), g_viewParams, g_stereoParams, g_raycastParams, pTfArray, primaryDevice);
					}
				}
			}

			s_timestampLastUpdate.clear();
			s_timestampLastUpdate.resize(g_cudaDevices.size(), 0);

			g_timerRendering.Start();
		}
		if(needTF)
		{
			cudaSafeCall(cudaGraphicsUnmapResources(1, &g_pTfEdtSRVCuda));
		}

		g_redraw = false;
	}

	bool renderingFinished = true; // default to "true", set to false if any GPU is not finished
	//TODO only call g_renderingManager.Render() when renderDelayPassed?
	//float timeSinceRenderUpdate = float(curTime - g_lastRenderParamsUpdate) / float(CLOCKS_PER_SEC);
	//bool renderDelayPassed = (timeSinceRenderUpdate >= g_startWorkingDelay);
	if (s_isRendering)
	{
		if(g_renderingManager.IsRendering())
		{
			// render next brick on primary GPU
			renderingFinished = (g_renderingManager.Render() == RenderingManager::STATE_DONE);
			renderingUpdated = true;
		}

		// if primary GPU is done, check if other threads are finished as well
		if(renderingFinished && g_useAllGPUs)
		{
			for(size_t i = 0; i < g_cudaDevices.size(); i++)
			{
				if(g_cudaDevices[i].pThread == nullptr) continue;

				renderingFinished = renderingFinished && !g_cudaDevices[i].pThread->IsWorking();
				renderingUpdated = true;
			}
		}

		if(renderingFinished)
		{
			s_isRendering = false;
			g_timerRendering.Stop();
		}
	}

	//-----------------------------------------------------------
	// COMBINE RESULTS AND DRAW ON SCREEN
	//-----------------------------------------------------------

	// if this was the last brick, or we want to see unfinished images, copy from raycast target into finished image tex
	if (renderingUpdated && (renderingFinished || g_showPreview))
	{
		//if (s_isTracing) std::cout << "Render while still tracing" << std::endl;

		// common shader vars for fullscreen pass
		Vec2f screenMin(-1.0f, -1.0f);
		Vec2f screenMax( 1.0f,  1.0f);
		g_screenEffect.m_pvScreenMinVariable->SetFloatVector(screenMin);
		g_screenEffect.m_pvScreenMaxVariable->SetFloatVector(screenMax);
		Vec2f texCoordMin(0.0f, 0.0f);
		Vec2f texCoordMax(1.0f, 1.0f);
		g_screenEffect.m_pvTexCoordMinVariable->SetFloatVector(texCoordMin);
		g_screenEffect.m_pvTexCoordMaxVariable->SetFloatVector(texCoordMax);

		pd3dImmediateContext->IASetInputLayout(NULL);
		pd3dImmediateContext->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLESTRIP);


		if(!g_useAllGPUs || (g_cudaDevices.size() == 1))
		{
			// single-GPU case - blend together rendering manager's "opaque" and "raycast" textures
			pd3dImmediateContext->OMSetRenderTargets(1, &g_pRenderBufferTempRTV, nullptr);

			// save old viewport
			UINT viewportCount = 1;
			D3D11_VIEWPORT viewportOld;
			pd3dImmediateContext->RSGetViewports(&viewportCount, &viewportOld);

			// set viewport for offscreen render buffer
			D3D11_VIEWPORT viewport;
			viewport.TopLeftX = 0;
			viewport.TopLeftY = 0;
			viewport.Width = (FLOAT)g_projParams.m_imageWidth;
			viewport.Height = (FLOAT)g_projParams.m_imageHeight;
			viewport.MinDepth = 0.0f;
			viewport.MaxDepth = 1.0f;
			pd3dImmediateContext->RSSetViewports(1, &viewport);

			// blit opaque stuff
			g_screenEffect.m_pTexVariable->SetResource(g_renderingManager.GetOpaqueSRV());
			g_screenEffect.m_pTechnique->GetPassByIndex(1)->Apply(0, pd3dImmediateContext);
			pd3dImmediateContext->Draw(4, 0);

			// blend raycaster result over
			g_screenEffect.m_pTexVariable->SetResource(g_renderingManager.GetRaycastSRV());
			g_screenEffect.m_pTechnique->GetPassByIndex(2)->Apply(0, pd3dImmediateContext);
			pd3dImmediateContext->Draw(4, 0);

			// restore viewport
			pd3dImmediateContext->RSSetViewports(1, &viewportOld);
		}
		else
		{
			// multi-GPU case - copy all raycast textures together
			//TODO what about opaque stuff..?
			for(size_t i = 0; i < g_cudaDevices.size(); i++)
			{
				int left   = g_projParams.GetImageLeft  (g_cudaDevices[i].range);
				int width  = g_projParams.GetImageWidth (g_cudaDevices[i].range);
				int height = g_projParams.GetImageHeight(g_cudaDevices[i].range);

				if(g_cudaDevices[i].pThread == nullptr)
				{
					// this is the main GPU, copy over directly
					D3D11_BOX box;
					box.left = 0;
					box.right = width;
					box.top = 0;
					box.bottom = height;
					box.front = 0;
					box.back = 1;

					pd3dImmediateContext->CopySubresourceRegion(g_pRenderBufferTempTex, 0, left, 0, 0, g_renderingManager.GetRaycastTex(), 0, &box);
				}
				else
				{
					// get image from thread and upload to this GPU
					byte* pData = nullptr;
					uint timestamp = g_cudaDevices[i].pThread->LockResultImage(pData);
					if(timestamp > s_timestampLastUpdate[i])
					{
						s_timestampLastUpdate[i] = timestamp;

						D3D11_BOX box;
						box.left = left;
						box.right = box.left + width;
						box.top = 0;
						box.bottom = box.top + height;
						box.front = 0;
						box.back = 1;

						pd3dImmediateContext->UpdateSubresource(g_pRenderBufferTempTex, 0, &box, pData, width * sizeof(uchar4), 0);
					}
					g_cudaDevices[i].pThread->UnlockResultImage();
				}
			}
		}

		// blit over into "raycast finished" tex
		pd3dImmediateContext->OMSetRenderTargets(1, &g_pRaycastFinishedRTV, nullptr);

		//TODO if g_renderBufferSizeFactor > 2, generate mipmaps first?
		g_screenEffect.m_pTexVariable->SetResource(g_pRenderBufferTempSRV);
		g_screenEffect.m_pTechnique->GetPassByIndex(1)->Apply(0, pd3dImmediateContext);
		pd3dImmediateContext->Draw(4, 0);

		ID3D11ShaderResourceView* pNullSRV[1] = { nullptr };
		pd3dImmediateContext->PSSetShaderResources(0, 1, pNullSRV);

		// reset render target
		ID3D11RenderTargetView* pRTV = DXUTGetD3D11RenderTargetView();
		pd3dImmediateContext->OMSetRenderTargets(1, &pRTV, DXUTGetD3D11DepthStencilView());
	}


	//-----------------------------------------------------------
	// MISCELANOUS TASKS (Screenshots, ...)
	//-----------------------------------------------------------

NoVolumeLoaded:

	// copy last finished image into back buffer
	ID3D11Resource* pSwapChainTex;
	DXUTGetD3D11RenderTargetView()->GetResource(&pSwapChainTex);
	pd3dImmediateContext->CopyResource(pSwapChainTex, g_pRaycastFinishedTex);
	SAFE_RELEASE(pSwapChainTex);


	// draw background
	pd3dImmediateContext->IASetInputLayout(NULL);
	pd3dImmediateContext->IASetPrimitiveTopology(D3D10_PRIMITIVE_TOPOLOGY_TRIANGLESTRIP);
	g_screenEffect.m_pvColorVariable->SetFloatVector(g_backgroundColor);
	Vec2f screenMin(-1.0f, -1.0f);
	Vec2f screenMax( 1.0f,  1.0f);
	g_screenEffect.m_pvScreenMinVariable->SetFloatVector(screenMin);
	g_screenEffect.m_pvScreenMaxVariable->SetFloatVector(screenMax);
	Vec2f texCoordMin(0.0f, 0.0f);
	Vec2f texCoordMax(1.0f, 1.0f);
	g_screenEffect.m_pvTexCoordMinVariable->SetFloatVector(texCoordMin);
	g_screenEffect.m_pvTexCoordMaxVariable->SetFloatVector(texCoordMax);
	g_screenEffect.m_pTechnique->GetPassByIndex(0)->Apply(0, pd3dImmediateContext);
	pd3dImmediateContext->Draw(4, 0);


	// Draw Transfer function
	if (g_bRenderUI)
	{
		g_tfEdt.setVisible(
			(g_raycastParams.m_raycastingEnabled && RaycastModeNeedsTransferFunction(g_raycastParams.m_raycastMode))
			|| (g_particleRenderParams.m_lineColorMode == eLineColorMode::MEASURE)
			|| (g_heatMapParams.m_enableRendering));
		//Draw Transfer function editor
		g_tfEdt.onFrameRender((float)fTime, fElapsedTime);
	}

	// save screenshot before drawing progress bar and gui
	if(g_saveScreenshot) {
		std::wstring filenameW = tum3d::FindNextSequenceNameEX(L"screenshot", L"png", CSysTools::GetExePath());
		std::string filename(filenameW.begin(), filenameW.end());
		SaveScreenshot(pd3dImmediateContext, filename);

		g_saveScreenshot = false;
	}

	if(g_saveRenderBufferScreenshot) {
		std::wstring filenameW = tum3d::FindNextSequenceNameEX(L"screenshot", L"png", CSysTools::GetExePath());
		std::string filename(filenameW.begin(), filenameW.end());
		SaveRenderBufferScreenshot(pd3dImmediateContext, filename);

		g_saveRenderBufferScreenshot = false;
	}


	if(g_imageSequence.Running) {
		if(!g_redraw && !s_isFiltering && !s_isRendering) {
			// current frame is finished, save (if specified) and advance
			if(g_imageSequence.Record) {
				std::wstring filenameW = tum3d::FindNextSequenceNameEX(L"video", L"png", CSysTools::GetExePath());
				std::string filename(filenameW.begin(), filenameW.end());
				if(g_imageSequence.FromRenderBuffer) {
					SaveRenderBufferScreenshot(pd3dImmediateContext, filename);
				} else {
					SaveScreenshot(pd3dImmediateContext, filename);
				}
			}

			g_imageSequence.FrameCur++;

			if(g_imageSequence.FrameCur >= g_imageSequence.FrameCount && g_imageSequence.Record) {
				g_imageSequence.Running = false;
				g_imageSequence.FrameCur = 0;
			} else {
				float angle = float(g_imageSequence.FrameCur) * g_imageSequence.AngleInc * PI / 180.0f;
				Vec4f rotationQuatCurFrame; rotationQuaternion(angle, Vec3f(0.0f, 1.0f, 0.0f), rotationQuatCurFrame);
				multQuaternion(g_imageSequence.BaseRotationQuat, rotationQuatCurFrame, g_viewParams.m_rotationQuat);

				g_viewParams.m_viewDistance += g_imageSequence.ViewDistInc;

				int32 timestep = g_imageSequence.BaseTimestep + (g_imageSequence.FrameCur / g_imageSequence.FramesPerTimestep);
				timestep %= g_volume.GetTimestepCount();
				float time = float(timestep) * g_volume.GetTimeSpacing();
				g_volume.SetCurTime(time);
			}
		}
	}


	if(g_batchTrace.Running) {
		if(!g_retrace && !s_isTracing) {
			// current trace is finished
			printf("\n------ Batch trace file %u step %u done.\n\n", g_batchTrace.FileCur, g_batchTrace.StepCur);

			// build output filename
			std::string volumeFileName = tum3d::RemoveExt(tum3d::GetFilename(g_batchTrace.VolumeFiles[g_batchTrace.FileCur]));
			std::ostringstream stream;
			stream << volumeFileName
				<< "_" << GetAdvectModeName(g_particleTraceParams.m_advectMode)
				<< "_" << GetTextureFilterModeName(g_particleTraceParams.m_filterMode);
			std::string strOutFileBaseNoSuffix = stream.str();

			if(g_batchTraceParams.m_qualityStepCount > 1) {
				stream << "_Q" << g_batchTraceParams.GetQualityStep(g_batchTrace.StepCur);
			}
			if(g_batchTraceParams.m_heuristicStepCount > 1) {
				stream << "_B" << g_batchTraceParams.GetHeuristicBonusStep(g_batchTrace.StepCur);
				stream << "P" << g_batchTraceParams.GetHeuristicPenaltyStep(g_batchTrace.StepCur);
			}
			std::string strOutFileBase = stream.str();

			g_batchTrace.Timings.push_back(g_tracingManager.GetTimings().TraceWall);

			if(g_batchTrace.WriteLinebufs) {
				// save result
				g_tracingManager.GetResult()->Write(g_batchTrace.OutPath + strOutFileBase + ".linebuf");
				// save config
				ConfigFile config;
				g_particleTraceParams.WriteConfig(config);
				config.Write(g_batchTrace.OutPath + strOutFileBase + ".cfg");
				// save stats
				std::ofstream statsFile(g_batchTrace.OutPath + strOutFileBase + ".txt");
				g_tracingManager.GetStats().WriteToFile(statsFile);
				statsFile << "\n";
				g_tracingManager.GetTimings().WriteToFile(statsFile);
				statsFile.close();
			}

			std::vector<std::string> extraColumns;
			extraColumns.push_back(strOutFileBase);
			extraColumns.push_back(toString(g_particleTraceParams.m_heuristicBonusFactor));
			extraColumns.push_back(toString(g_particleTraceParams.m_heuristicPenaltyFactor));
			g_tracingManager.GetStats().WriteToCSVFile(g_batchTrace.FileStats, extraColumns);
			g_tracingManager.GetTimings().WriteToCSVFile(g_batchTrace.FileTimings, extraColumns);

			// advance step
			g_batchTrace.StepCur++;

			if(g_batchTrace.StepCur >= g_batchTraceParams.GetTotalStepCount()) {
				if(g_batchTraceParams.m_heuristicBPSeparate)
				{
					// write this file's B/P timings into csv in matrix form
					std::ofstream fileTimings(g_batchTrace.OutPath + strOutFileBaseNoSuffix + "_Timings.csv");
					fileTimings << ";;Penalty\n";
					fileTimings << ";";
					for(uint p = 0; p < g_batchTraceParams.m_heuristicStepCount; p++)
					{
						fileTimings << ";" << g_batchTraceParams.GetHeuristicFactor(p);
					}
					fileTimings << "\n";
					size_t i = 0;
					for(uint b = 0; b < g_batchTraceParams.m_heuristicStepCount; b++)
					{
						fileTimings << (b == 0 ? "Bonus" : "");
						fileTimings << ";" << g_batchTraceParams.GetHeuristicFactor(b);
						for(uint p = 0; p < g_batchTraceParams.m_heuristicStepCount; p++)
						{
							fileTimings << ";" << g_batchTrace.Timings[i++] / 1000.0f;
						}
						fileTimings << "\n";
					}
					fileTimings.close();
				}
				g_batchTrace.Timings.clear();

				g_batchTrace.StepCur = 0;
				g_batchTrace.FileCur++;
				if(g_batchTrace.FileCur >= g_batchTrace.VolumeFiles.size()) {
					// finished
					g_batchTrace.Running = false;
					printf("\n------ Batch trace finished.\n\n");
					g_batchTrace.FileStats.close();
					g_batchTrace.FileTimings.close();
					g_batchTrace.Timings.clear();

					if(g_batchTrace.ExitAfterFinishing) {
						g_volume.Close(); // kill the loading thread!
						exit(0); // hrm..
					}
				} else {
					// open next volume file
					CloseVolumeFile();
					OpenVolumeFile(g_batchTrace.VolumeFiles[g_batchTrace.FileCur], pd3dDevice);
					if(!LineModeIsTimeDependent(g_particleTraceParams.m_lineMode) && g_volume.IsCompressed()) {
						// semi-HACK: fully load the whole timestep, so we get results consistently without IO
						printf("Loading file %u...", g_batchTrace.FileCur);
						g_volume.LoadNearestTimestep();
						printf("\n");
					}
					if(g_particleTraceParams.HeuristicUseFlowGraph()) {
						LoadOrBuildFlowGraph();
					}

					// start next step
					g_batchTraceParams.ApplyToTraceParams(g_particleTraceParams, g_batchTrace.StepCur);
				}
			} else {
				// start next step
				g_batchTraceParams.ApplyToTraceParams(g_particleTraceParams, g_batchTrace.StepCur);
			}
		}
	}


	// if the current task isn't finished, show progress bar
	if (s_isFiltering || s_isTracing || s_isRendering)
	{
		float pos[2]  = { -0.5f, 0.8f };
		float size[2] = { 1.0f, 0.1f };
		float color[4] = { 1.0f, 1.0f, 1.0f, 0.7f };
		if(Luminance(g_backgroundColor.xyz()) > 0.5f) {
			color[0] = color[1] = color[2] = 0.0f; color[3] = 0.5f;
		}
		float progress = 1.0f;
		if(s_isFiltering) {
			progress = g_filteringManager.GetFilteringProgress();
		} else if(s_isTracing) {
			progress = g_tracingManager.GetTracingProgress();
		} else if(s_isRendering) {
			progress = g_renderingManager.GetRenderingProgress();
		}
		if(g_useAllGPUs)
		{
			for(size_t i = 0; i < g_cudaDevices.size(); i++)
			{
				if(i == g_primaryCudaDeviceIndex) continue;

				progress = min(progress, g_cudaDevices[i].pThread->GetProgress());
			}
		}

		if(g_stereoParams.m_stereoEnabled) {
			pos[1] = 0.925f;
			size[1] *= 0.5f;
			DrawProgressBar(pd3dImmediateContext, pos, size, color, progress);
			pos[1] -= 1.0f;
			DrawProgressBar(pd3dImmediateContext, pos, size, color, progress);
		} else {
			DrawProgressBar(pd3dImmediateContext, pos, size, color, progress);
		}
	}

	// If any bricks are still being loaded, show progress bar
	float loadingProgress = g_volume.GetLoadingProgress();
	if (loadingProgress < 1.0f)
	{
		float pos[2]  = { -0.5f, 0.75f };
		float size[2] = { 1.0f, 0.03f };
		float color[4] = { 1.0f, 1.0f, 1.0f, 0.7f };
		if(Luminance(g_backgroundColor.xyz()) > 0.5f) {
			color[0] = color[1] = color[2] = 0.0f; color[3] = 0.5f;
		}

		if(g_stereoParams.m_stereoEnabled) {
			pos[1] = 0.9f;
			size[1] *= 0.5f;
			DrawProgressBar(pd3dImmediateContext, pos, size, color, loadingProgress);
			pos[1] -= 1.0f;
			DrawProgressBar(pd3dImmediateContext, pos, size, color, loadingProgress);
		} else {
			DrawProgressBar(pd3dImmediateContext, pos, size, color, loadingProgress);
		}
	}


	// GUI
	if(g_bRenderUI)
	{
		// switch to dark text for bright backgrounds
		if(Luminance(g_backgroundColor.xyz()) > 0.7f) {
			TwDefine("Main text=dark");
			TwDefine("ImageSequence text=dark");
		} else {
			TwDefine("Main text=light");
			TwDefine("ImageSequence text=light");
		}

		TwRefreshBar(g_pTwBarMain);
		TwRefreshBar(g_pTwBarImageSequence);
		TwDraw();
	}
}


//--------------------------------------------------------------------------------------
// Release D3D11 resources created in OnD3D11ResizedSwapChain 
//--------------------------------------------------------------------------------------
void CALLBACK OnD3D11ReleasingSwapChain( void* pUserContext )
{
	g_windowSize.x() = 0;
	g_windowSize.y() = 0;
	TwWindowSize(0, 0);

	ResizeRenderBuffer(nullptr);

	SAFE_RELEASE(g_pStagingTex);

	SAFE_RELEASE(g_pRaycastFinishedRTV);
	SAFE_RELEASE(g_pRaycastFinishedTex);

	g_tfEdt.onReleasingSwapChain();
}


//--------------------------------------------------------------------------------------
// Release D3D11 resources created in OnD3D11CreateDevice 
//--------------------------------------------------------------------------------------
void CALLBACK OnD3D11DestroyDevice( void* pUserContext )
{
	cudaSafeCall(cudaGraphicsUnregisterResource(g_pTfEdtSRVCuda));
	g_pTfEdtSRVCuda = nullptr;
	g_tfEdt.onDestroyDevice();

	TwTerminate();
	g_pTwBarImageSequence = nullptr;
	g_pTwBarMain = nullptr;


	ReleaseVolumeDependentResources();

	ShutdownCudaDevices();


	g_progressBarEffect.SafeRelease();

	g_screenEffect.SafeRelease();


	cudaSafeCall(cudaDeviceSynchronize());
	g_volume.SetPageLockBrickData(false);
	cudaSafeCallNoSync(cudaDeviceReset());

	//release slice texture
	SAFE_RELEASE(g_particleRenderParams.m_pSliceTexture);
	SAFE_RELEASE(g_particleRenderParams.m_pColorTexture);
}


//--------------------------------------------------------------------------------------
// Call if device was removed.  Return true to find a new device, false to quit
//--------------------------------------------------------------------------------------
bool CALLBACK OnDeviceRemoved( void* pUserContext )
{
	return true;
}


// Picks the seed from the seed texture and places is into 'seed' if there is a intersection.
// The function returns true iff there was an intersection
// 'intersection' will contain the 3D-coordinate of intersection
bool PickSeed(unsigned int* pSeed, Vec3f* pIntersection) {
	std::cout << "Pick seed, mouse position = (" << g_mouseScreenPosition.x() << ", " << g_mouseScreenPosition.y() << ")" << std::endl;
	//create ray through the mouse
	Mat4f viewProjMat = g_projParams.BuildProjectionMatrix(EYE_CYCLOP, 0.0f, g_cudaDevices[g_primaryCudaDeviceIndex].range)
		* g_viewParams.BuildViewMatrix(EYE_CYCLOP, 0.0f);
	Mat4f invViewProjMat;
	tum3D::invert4x4(viewProjMat, invViewProjMat);
	Vec4f start4 = invViewProjMat * Vec4f(g_mouseScreenPosition.x(), g_mouseScreenPosition.y(), 0.01f, 1.0f);
	Vec3f start = start4.xyz() / start4.w();
	Vec4f end4 = invViewProjMat * Vec4f(g_mouseScreenPosition.x(), g_mouseScreenPosition.y(), 0.99f, 1.0f);
	Vec3f end = end4.xyz() / end4.w();
	std::cout << "Ray, start=" << start << ", end=" << end << std::endl;
	//cut ray with the xy-plane
	Vec3f dir = end - start;
	normalize(dir);
	Vec3f n = Vec3f(0, 0, 1); //normal of the plane
	float d = (-start).dot(n) / (dir.dot(n));
	if (d < 0) return false; //we are behind the plane
	Vec3f intersection = start + d * dir;
	std::cout << "Intersection: " << intersection << std::endl;
	if (pIntersection) *pIntersection = intersection;
	//Test if seed texture is loaded
	if (g_particleTraceParams.m_seedTexture.m_colors == NULL) {
		//no texture
		*pSeed = 0;
		return true;
	}
	else {
		//a seed texture was found
		//check if intersection is in the volume
		if (intersection > -g_volume.GetVolumeHalfSizeWorld()
			&& intersection < g_volume.GetVolumeHalfSizeWorld()) {
			//inside, convert to texture coordinates
			Vec3f localIntersection = (intersection + g_volume.GetVolumeHalfSizeWorld()) / (2 * g_volume.GetVolumeHalfSizeWorld());
			int texX = (int)(localIntersection.x() * g_particleTraceParams.m_seedTexture.m_width);
			int texY = (int)(localIntersection.y() * g_particleTraceParams.m_seedTexture.m_height);
			texY = g_particleTraceParams.m_seedTexture.m_height - texY - 1;
			unsigned int color = g_particleTraceParams.m_seedTexture.m_colors[texX + texY * g_particleTraceParams.m_seedTexture.m_height];
			printf("Pick color at position (%d, %d): 0x%08x\n", texX, texY, color);
			*pSeed = color;
			return true;
		}
		else {
			std::cout << "Outside the bounds" << std::endl;
			return false;
		}
	}
}

// NOTE: OnKeyboard and OnMouse are *not* registered as DXUT callbacks because this doesn't work well with AntTweakBar
//--------------------------------------------------------------------------------------
// Handle key presses
//--------------------------------------------------------------------------------------
void OnKeyboard( UINT nChar, bool bKeyDown, bool bAltDown, bool bHandledByGUI )
{
	if (nChar == VK_RSHIFT || nChar == VK_SHIFT) {
		g_keyboardShiftPressed = bKeyDown;
	}

	if(bKeyDown)
	{
		switch(nChar)
		{
			case VK_RETURN :
			{
				if(bAltDown) DXUTToggleFullScreen();
				break;
			}
			case 'U' : 
			{
				g_bRenderUI = !g_bRenderUI; 
				break;
			}
			case VK_F5 :
			{
				SaveRenderingParams("quicksave.cfg");
				break;
			}
			case VK_F6 :
			//case VK_F9 :
			{
				LoadRenderingParams("quicksave.cfg");
				break;
			}
			case 'P' : 
			{
				unsigned int color;
				Vec3f pos;
				bool ret = PickSeed(&color, &pos);

				//Test if seed texture is loaded
				if (g_particleTraceParams.m_seedTexture.m_colors == NULL) {
					//no texture, just adjust seed box
					g_particleTraceParams.m_seedBoxMin = pos - Vec3f(0.05f);
					g_particleTraceParams.m_seedBoxSize = Vec3f(0.1f);
				}
				else {
					//a seed texture was found
					//check if intersection is in the volume
					if (ret) {
						if (g_keyboardShiftPressed) {
							if (g_particleTraceParams.m_seedTexture.m_picked.count(color)==0)
								g_particleTraceParams.m_seedTexture.m_picked.insert(color);
							else //toggle selection status
								g_particleTraceParams.m_seedTexture.m_picked.erase(color);
						} else {
							g_particleTraceParams.m_seedTexture.m_picked.clear();
							g_particleTraceParams.m_seedTexture.m_picked.insert(color);
						}
					} else {
						if (!g_keyboardShiftPressed) {
							g_particleTraceParams.m_seedTexture.m_picked.clear(); //disable seed from texture
						}
					}
					//update heat map params
					g_heatMapParams.m_recordTexture = g_particleTraceParams.m_seedTexture;
					//check if the rendered channels are still alive
					if (g_heatMapParams.m_recordTexture.m_picked.count(g_heatMapParams.m_renderedChannels[0]) == 0) {
						if (g_heatMapParams.m_recordTexture.m_picked.empty()) {
							g_heatMapParams.m_renderedChannels[0] = 0;
						}
						else {
							g_heatMapParams.m_renderedChannels[0] = *g_heatMapParams.m_recordTexture.m_picked.begin();
						}
					}
					if (g_heatMapParams.m_recordTexture.m_picked.count(g_heatMapParams.m_renderedChannels[1]) == 0) {
						g_heatMapParams.m_renderedChannels[1] = 0;
					}
				}

				break;
			}
			case '1':
			case '2':
			{
				//Update rendered channel
				int channel = nChar - '1';
				
				unsigned int color;
				Vec3f pos;
				bool ret = PickSeed(&color, &pos);
				if (ret) {
					g_heatMapParams.m_renderedChannels[channel] = color;
				}
				else {
					g_heatMapParams.m_renderedChannels[channel] = 0;
				}

				break;
			}
			default : return;
		}
	}
}


// NOTE: OnKeyboard and OnMouse are *not* registered as DXUT callbacks because this doesn't work well with AntTweakBar
//--------------------------------------------------------------------------------------
// Handle mouse button presses
//--------------------------------------------------------------------------------------
void OnMouse( bool bLeftButtonDown, bool bRightButtonDown, bool bMiddleButtonDown,
              bool bSideButton1Down, bool bSideButton2Down, int nMouseWheelDelta,
              int xPos, int yPos, bool bCtrlDown, bool bShiftDown, bool bHandledByGUI )
{
	static bool bLeftButtonDownPrev = bLeftButtonDown;
	static bool bRightButtonDownPrev = bRightButtonDown;
	static bool bMiddleButtonDownPrev = bMiddleButtonDown;
	static int xPosPrev = xPos;
	static int yPosPrev = yPos;

	g_mouseScreenPosition = Vec2f((xPos / (float)g_windowSize.x()) * 2 - 1, -((yPos / (float)g_windowSize.y()) * 2 - 1));

	if(!bHandledByGUI && !g_imageSequence.Running) {
		int xPosDelta = xPos - xPosPrev;
		int yPosDelta = yPos - yPosPrev;

		float xDelta = float(xPosDelta) / float(g_windowSize.y()); // g_windowSize.y() not a typo: consistent speed in x and y
		float yDelta = float(yPosDelta) / float(g_windowSize.y());

		Mat4f inverseView;
		tum3D::invert4x4(g_viewParams.BuildViewMatrix(EYE_CYCLOP, 0.0f), inverseView);
		Vec3f majorX, majorY;
		GetMajorWorldPlane(Vec3f(1.0f, 0.0f, 0.0f), Vec3f(0.0f, 1.0f, 0.0f), inverseView, majorX, majorY);

		if(bLeftButtonDown && bLeftButtonDownPrev) {
			// rotate
			Vec4f rotationX;
			tum3D::rotationQuaternion(xDelta * PI, Vec3f(0.0f, 1.0f, 0.0f), rotationX);
			Vec4f rotationY;
			tum3D::rotationQuaternion(yDelta * PI, Vec3f(1.0f, 0.0f, 0.0f), rotationY);

			//add to global rotation
			Vec4f temp;
			tum3D::multQuaternion(rotationX, g_rotationX, temp); g_rotationX = temp;
			tum3D::multQuaternion(rotationY, g_rotationY, temp); g_rotationY = temp;

			//combine and set to view params
			tum3D::multQuaternion(g_rotationY, g_rotationX, temp);
			tum3D::multQuaternion(temp, g_rotation, g_viewParams.m_rotationQuat);
			//tum3D::multQuaternion(g_rotationY, g_rotationX, g_viewParams.m_rotationQuat);
		}
		if (!bLeftButtonDown && bLeftButtonDownPrev) {
			//mouse released, so store the current rotation and reset the partial rotation
			//This is needed so that the trackball starts with a new projection
			g_rotation = g_viewParams.m_rotationQuat;
			g_rotationX = Vec4f(1, 0, 0, 0);
			g_rotationY = Vec4f(1, 0, 0, 0);
			printf("push rotation\n");
		}
		if(bMiddleButtonDown && bMiddleButtonDownPrev) {
			if(bShiftDown) {
				// scale seed region
				Vec3f expX = majorX *  xDelta;
				Vec3f expY = majorY * -yDelta;
				Vec3f scalingX(pow(4.0f, expX.x()), pow(4.0f, expX.y()), pow(4.0f, expX.z()));
				Vec3f scalingY(pow(4.0f, expY.x()), pow(4.0f, expY.y()), pow(4.0f, expY.z()));
				g_particleTraceParams.ScaleSeedBox(scalingX * scalingY);
			} else {
				// zoom
				g_viewParams.m_viewDistance *= pow(5.0f, -yDelta);
			}
		}
		if(bRightButtonDown && bRightButtonDownPrev) {
			if(bShiftDown) {
				// move seed region
				Vec3f translation = xDelta * majorX - yDelta * majorY;
				g_particleTraceParams.MoveSeedBox(translation);
			} else {
				// move domain/camera
				Mat3f rotationMat;
				tum3D::convertQuaternionToRotMat(g_viewParams.m_rotationQuat, rotationMat);
				Vec3f xVec = rotationMat.getRow(0);
				Vec3f yVec = rotationMat.getRow(1);
				g_viewParams.m_lookAt -= tan(0.5f * g_projParams.m_fovy) * g_viewParams.m_viewDistance * (xDelta * xVec - yDelta * yVec);
			}
		}
	}

	// update "previous" state for next call
	bLeftButtonDownPrev = bLeftButtonDown;
	bRightButtonDownPrev = bRightButtonDown;
	bMiddleButtonDownPrev = bMiddleButtonDown;
	xPosPrev = xPos;
	yPosPrev = yPos;
}


//--------------------------------------------------------------------------------------
// Handle messages to the application
//--------------------------------------------------------------------------------------
LRESULT CALLBACK MsgProc( HWND hWnd, UINT uMsg, WPARAM wParam, LPARAM lParam,
                          bool* pbNoFurtherProcessing, void* pUserContext )
{
	bool bHandledByGUI = false;

	*pbNoFurtherProcessing = g_tfEdt.msgProc( hWnd, uMsg, wParam, lParam );
	if(*pbNoFurtherProcessing) {
		bHandledByGUI = true;
	}

	if(g_bRenderUI) // only handle UI events if it is visible
	{
		if(TwEventWin(hWnd, uMsg, wParam, lParam))
		{
			bHandledByGUI = true;
			*pbNoFurtherProcessing = true;
			// don't return, still pass event to keyboard/mouse callbacks (so they can keep their state valid)
		}
	}

	// Copied from DXUT: Consolidate the keyboard messages and pass them to the app's keyboard callback
	if( uMsg == WM_KEYDOWN ||
	    uMsg == WM_SYSKEYDOWN ||
	    uMsg == WM_KEYUP ||
	    uMsg == WM_SYSKEYUP )
	{
		bool bKeyDown = ( uMsg == WM_KEYDOWN || uMsg == WM_SYSKEYDOWN );
		DWORD dwMask = ( 1 << 29 );
		bool bAltDown = ( ( lParam & dwMask ) != 0 );
		//bool bShiftDown = ((nMouseButtonState & MK_SHIFT) != 0);

		OnKeyboard( ( UINT )wParam, bKeyDown, bAltDown, bHandledByGUI );
	}

	// Copied from DXUT: Consolidate the mouse button messages and pass them to the app's mouse callback
	if( uMsg == WM_LBUTTONDOWN ||
	    uMsg == WM_LBUTTONUP ||
	    uMsg == WM_LBUTTONDBLCLK ||
	    uMsg == WM_MBUTTONDOWN ||
	    uMsg == WM_MBUTTONUP ||
	    uMsg == WM_MBUTTONDBLCLK ||
	    uMsg == WM_RBUTTONDOWN ||
	    uMsg == WM_RBUTTONUP ||
	    uMsg == WM_RBUTTONDBLCLK ||
	    uMsg == WM_XBUTTONDOWN ||
	    uMsg == WM_XBUTTONUP ||
	    uMsg == WM_XBUTTONDBLCLK ||
	    uMsg == WM_MOUSEWHEEL ||
	    uMsg == WM_MOUSEMOVE )
	{
		int xPos = ( short )LOWORD( lParam );
		int yPos = ( short )HIWORD( lParam );

		if( uMsg == WM_MOUSEWHEEL )
		{
			// WM_MOUSEWHEEL passes screen mouse coords
			// so convert them to client coords
			POINT pt;
			pt.x = xPos; pt.y = yPos;
			ScreenToClient( hWnd, &pt );
			xPos = pt.x; yPos = pt.y;
		}

		int nMouseWheelDelta = 0;
		if( uMsg == WM_MOUSEWHEEL )
			nMouseWheelDelta = ( short )HIWORD( wParam );

		int nMouseButtonState = LOWORD( wParam );
		bool bLeftButton = ( ( nMouseButtonState & MK_LBUTTON ) != 0 );
		bool bRightButton = ( ( nMouseButtonState & MK_RBUTTON ) != 0 );
		bool bMiddleButton = ( ( nMouseButtonState & MK_MBUTTON ) != 0 );
		bool bSideButton1 = ( ( nMouseButtonState & MK_XBUTTON1 ) != 0 );
		bool bSideButton2 = ( ( nMouseButtonState & MK_XBUTTON2 ) != 0 );
		bool bCtrlDown = ( ( nMouseButtonState & MK_CONTROL ) != 0 );
		bool bShiftDown = ( ( nMouseButtonState & MK_SHIFT ) != 0 );

		OnMouse( bLeftButton, bRightButton, bMiddleButton, bSideButton1, bSideButton2, nMouseWheelDelta, xPos, yPos, bCtrlDown, bShiftDown, bHandledByGUI );
	}

	return 0;
}


//--------------------------------------------------------------------------------------
// Handle updates to the scene
//--------------------------------------------------------------------------------------
void CALLBACK OnFrameMove( double dTime, float fElapsedTime, void* pUserContext )
{
	// check if we should update the window title
	bool update = false;

	// update if image or window size changed
	static uint s_imageWidth = 0;
	static uint s_imageHeight = 0;
	if (s_imageWidth != g_projParams.m_imageWidth || s_imageHeight != g_projParams.m_imageHeight) {
		s_imageWidth = g_projParams.m_imageWidth;
		s_imageHeight = g_projParams.m_imageHeight;
		update = true;
	}

	static Vec2i s_windowSize(0, 0);
	if (s_windowSize != g_windowSize) {
		s_windowSize = g_windowSize;
		update = true;
	}

	// update fps once per second
	static float s_fps = 0.0f;
	static float s_mspf = 0.0f;
	static double s_dStatLastFPSUpdate = 0.0;
	double dt = dTime - s_dStatLastFPSUpdate;
	if (dt >= 1.0) {
		s_fps = DXUTGetFPS();
		s_mspf = 1000.0f / s_fps;
		s_dStatLastFPSUpdate = dTime;
		update = true;
	}

	//// update if RenderingManager's timings were updated
	//const RenderingManager::Timings& timings = g_renderingManager.GetTimings();
	//static RenderingManager::Timings s_timings;
	//if (s_timings != timings)
	//{
	//	s_timings = timings;
	//	update = true;
	//}

	static float s_timeTrace = 0.0f;
	float timeTraceNew = g_timerTracing.GetElapsedTimeMS();
	if(s_timeTrace != timeTraceNew)
	{
		s_timeTrace = timeTraceNew;
		update = true;
	}

	static float s_timeRender = 0.0f;
	float timeRenderNew = g_timerRendering.GetElapsedTimeMS();
	if(s_timeRender != timeRenderNew)
	{
		s_timeRender = timeRenderNew;
		update = true;
	}

	// update window title if something relevant changed
	if (update) {
		const size_t len = 512;
		wchar_t str[len];
		int pos = swprintf_s(str, L"TurbulenceRenderer %ux%u (%ux%u) @ %.2f fps / %.2f ms", s_windowSize.x(), s_windowSize.y(), s_imageWidth, s_imageHeight, s_fps, s_mspf);
		//pos += swprintf_s(str + pos, len - pos, L"   Upload/Decomp (GPU): %.2f ms (%.2f-%.2f-%.2f : %u)", s_timings.UploadDecompressGPU.Total, s_timings.UploadDecompressGPU.Min, s_timings.UploadDecompressGPU.Avg, s_timings.UploadDecompressGPU.Max, s_timings.UploadDecompressGPU.Count);
		//pos += swprintf_s(str + pos, len - pos, L"   Raycast (GPU): %.2f ms (%.2f-%.2f-%.2f : %u)", s_timings.RaycastGPU.Total, s_timings.RaycastGPU.Min, s_timings.RaycastGPU.Avg, s_timings.RaycastGPU.Max, s_timings.RaycastGPU.Count);
		//pos += swprintf_s(str + pos, len - pos, L"   Render (Wall): %.2f ms", s_timings.RenderWall);
		pos += swprintf_s(str + pos, len - pos, L"   Trace Time: %.2f ms", s_timeTrace);
		pos += swprintf_s(str + pos, len - pos, L"   Render Time: %.2f ms", s_timeRender);
		SetWindowText(DXUTGetHWND(), str);
	}
}






// ============================================================
// Init / Exit / Main
// ============================================================

bool InitApp()
{
	return GetCudaDevices();
}


// note: this is called *before* OnD3D11DestroyDevice !!
void ExitApp()
{
	CloseVolumeFile();
	ClearCudaDevices();
}


//--------------------------------------------------------------------------------------
// Initialize everything and go into a render loop
//--------------------------------------------------------------------------------------
//#include "TracingTestData.h"
int main(int argc, char* argv[])
{
	// Enable run-time memory check for debug builds.
#if defined(DEBUG) | defined(_DEBUG)
	_CrtSetDbgFlag( _CRTDBG_ALLOC_MEM_DF | _CRTDBG_LEAK_CHECK_DF );
#endif


#ifdef _DEBUG
	printf("---- WARNING: DEBUG BUILD - SHIT WILL BE SLOW! ----\n\n");
#endif


	//if(!freopen("log.txt", "w", stdout)) {
	//	printf("FUBAR\n");
	//}


	// init default number of omp threads
	g_threadCount = omp_get_num_procs();
	omp_set_num_threads(g_threadCount);

	// Set general DXUT callbacks
	DXUTSetCallbackMsgProc( MsgProc );
	// do *not* register OnKeyboard and OnMouse!

	DXUTSetCallbackFrameMove( OnFrameMove );

	DXUTSetCallbackDeviceChanging( ModifyDeviceSettings );
	DXUTSetCallbackDeviceRemoved( OnDeviceRemoved );

	// Set the D3D11 DXUT callbacks
	DXUTSetCallbackD3D11DeviceAcceptable( IsD3D11DeviceAcceptable );
	DXUTSetCallbackD3D11DeviceCreated( OnD3D11CreateDevice );
	DXUTSetCallbackD3D11SwapChainResized( OnD3D11ResizedSwapChain );
	DXUTSetCallbackD3D11FrameRender( OnD3D11FrameRender );
	DXUTSetCallbackD3D11SwapChainReleasing( OnD3D11ReleasingSwapChain );
	DXUTSetCallbackD3D11DeviceDestroyed( OnD3D11DestroyDevice );

	// Perform application-level initialization
	if(!InitApp()) {
		return EXIT_FAILURE;
	}

	DXUTInit( true, true, NULL ); // Parse the command line, show msgboxes on error, no extra command line params
	DXUTSetIsInGammaCorrectMode( false );
	DXUTSetCursorSettings( true, true ); // Show the cursor and clip it when in full screen
	DXUTCreateWindow( L"TurbulenceRenderer" );
	DXUTCreateDevice( D3D_FEATURE_LEVEL_11_0, true, 1280, 1024 );


	//OpenVolumeFile("F:\\Turbulence\\IsoHigh\\IsotropicHigh.timevol", DXUTGetD3D11Device());


	if(argc > 1 && std::string(argv[1]) == "dumpla3d") {
		if(argc != 4) {
			printf("Usage: %s dumpla3d file.timevol outfile.la3d", argv[0]);
			return 1;
		}
		std::string filenameVol(argv[2]);
		std::string filenameLa3d(argv[3]);

		if(!OpenVolumeFile(filenameVol, DXUTGetD3D11Device())) {
			printf("Failed opening %s\n", filenameVol.c_str());
			return 2;
		}

		// create outpath if it doesn't exist yet
		system((string("mkdir \"") + tum3d::GetPath(filenameLa3d) + "\"").c_str());

		// remove extension
		if(filenameLa3d.substr(filenameLa3d.size() - 5) == ".la3d")
		{
			filenameLa3d = filenameLa3d.substr(0, filenameLa3d.size() - 5);
		}
		// build per-channel filenames
		std::vector<std::string> filenames;
		for(int c = 0; c < g_volume.GetChannelCount(); c++)
		{
			std::ostringstream str;
			str << filenameLa3d << char('X' + c) << ".la3d";
			filenames.push_back(str.str());
		}
		// and write la3d
		if(!g_renderingManager.WriteCurTimestepToLA3Ds(g_volume, filenames)) {
			printf("Failed writing la3ds\n");
			return 3;
		}
		return 0;
	}


	// parse cmdline
	// usage: argv[0] [config.cfg [volume.timevol [batchoutprefix [heuristicflags [options...]]]]]
	if(argc > 1)
	{
		std::string configFile(argv[1]);
		if(!LoadRenderingParams(configFile))
		{
			printf("Failed loading config from %s\n", configFile.c_str());
			return 1;
		}
	}
	if(argc > 2)
	{
		std::string volumeFile(argv[2]);
		g_batchTrace.VolumeFiles.push_back(volumeFile);
		g_batchTrace.OutPath = tum3d::GetPath(volumeFile);
		if(argc > 3)
		{
			g_batchTrace.OutPath += std::string(argv[3]);
		}
		if(argc > 4)
		{
			g_particleTraceParams.m_heuristicFlags = atoi(argv[4]);
		}
		for(int i = 5; i < argc; i++)
		{
			if(argv[i] == std::string("WriteLinebufs"))
			{
				g_batchTrace.WriteLinebufs = true;
			}
			//else if ... more options?
			else
			{
				printf("WARNING: unknown option \"%s\"\n", argv[i]);
			}
		}
		g_batchTrace.ExitAfterFinishing = true;
		StartBatchTracing();
	}

	DXUTMainLoop(); // Enter into the DXUT render loop

	// Perform application-level cleanup
	ExitApp();

	return DXUTGetExitCode();
}
