#pragma once

#include <vector>
#include <fstream>

#include <d3d11.h>

#include <cuda_runtime.h>
#include <cuda_d3d11_interop.h>
#include <cuda_profiler_api.h>

#include <TimeVolume.h>
#include <FilterParams.h>
#include <RaycastParams.h>
#include <ParticleTraceParams.h>
#include <ParticleTraceParams.h>
#include <HeatMap.h>
#include <FilteringManager.h>
#include <TracingManager.h>
#include <RenderingManager.h>
#include <HeatMapManager.h>
#include <ProjectionParams.h>
#include <StereoParams.h>
#include <ViewParams.h>
#include <BatchTraceParams.h>
#include <WorkerThread.h>
#include <TimerCPU.h>
#include <RenderTexture.h>
#include <ScreenEffect.h>

#include <Vec.h>


#pragma region AuxStructures
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
};


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

	tum3D::Vec4f BaseRotationQuat;
	int32 BaseTimestep;
	// ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

	bool  Running;
	int32 FrameCur;
};
#pragma endregion


class FlowVisTool
{
public:
	ID3D11Device*			m_d3dDevice = nullptr;
	ID3D11DeviceContext*	m_d3dDeviceContex = nullptr;
	//ID3D11RenderTargetView*	m_mainRenderTargetView = nullptr;

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

	RenderTexture g_renderTexture;
	std::vector<MyCudaDevice>	g_cudaDevices;
	int							g_primaryCudaDeviceIndex = -1;
	bool						g_useAllGPUs = false;

	cudaGraphicsResource*		g_pTfEdtSRVCuda = nullptr;

	tum3D::Vec2i g_windowSize;

	bool g_showPreview = true;
	bool m_redraw = false;
	bool m_retrace = false;
	bool g_particleTracingPaused = false;
	bool g_bRenderDomainBox = true;
	bool g_bRenderBrickBoxes = false;
	bool g_bRenderClipBox = true;
	bool g_bRenderSeedBox = true;

	float g_renderBufferSizeFactor = 2.0f;

	BatchTrace				g_batchTrace;
	ImageSequence			g_imageSequence;

	FilterParams			g_filterParams;
	RaycastParams			g_raycastParams;
	ParticleTraceParams		g_particleTraceParams;
	ParticleRenderParams	g_particleRenderParams;
	HeatMapParams			g_heatMapParams;
	ProjectionParams		g_projParams;
	StereoParams			g_stereoParams;
	ViewParams				g_viewParams;
	BatchTraceParams		g_batchTraceParams;

	GPUResources            g_compressShared;
	CompressVolumeResources g_compressVolume;
	TimeVolume				g_volume;
	FlowGraph				g_flowGraph;

	std::vector<LineBuffers*> g_lineBuffers;
	std::vector<BallBuffers*> g_ballBuffers;
	float                     g_ballRadius = 0.011718750051f;

	// resources on primary GPU
	FilteringManager g_filteringManager;
	TracingManager   g_tracingManager;
	RenderingManager g_renderingManager;
	HeatMapManager   g_heatMapManager;

	clock_t	g_lastRenderParamsUpdate = 0;
	clock_t	g_lastTraceParamsUpdate = 0;
	float	g_startWorkingDelay = 0.1f;

	TimerCPU	g_timerTracing;
	TimerCPU	g_timerRendering;

	ScreenEffect		g_screenEffect;

	tum3D::Vec4f		g_backgroundColor = tum3D::Vec4f(0.1f, 0.1f, 0.1f, 1.0f);

public:
	FlowVisTool();

	bool Initialize(ID3D11Device* d3dDevice, ID3D11DeviceContext* d3dDeviceContex, const std::vector<MyCudaDevice>& cudaDevices);
	void Release();
	void OnFrame(float deltaTime);

	void ReleaseLineBuffers();
	void ReleaseBallBuffers();

	void CloseVolumeFile();
	bool OpenVolumeFile(const std::string& filename);

	bool ResizeViewport(int width, int height);

	void BuildFlowGraph(const std::string& filenameTxt = "");
	bool SaveFlowGraph();
	bool LoadFlowGraph();
	void LoadOrBuildFlowGraph();

	void SetBoundingBoxToDomainSize();
	
private:
	void ReleaseVolumeDependentResources();
	bool CreateVolumeDependentResources();

	bool InitCudaDevices();
	void ShutdownCudaDevices();

	bool ResizeRenderBuffer();

private:
	// disable copy and assignment
	FlowVisTool(const FlowVisTool&);
	FlowVisTool& operator=(const FlowVisTool&);
};