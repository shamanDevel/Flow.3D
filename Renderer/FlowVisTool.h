#ifndef __FLOWVISTOOL_H_
#define __FLOWVISTOOL_H_

#pragma once

#include <vector>
#include <fstream>

#include <d3d11.h>
#include "WICTextureLoader.h"

#include <cuda_runtime.h>
#include <cuda_d3d11_interop.h>
#include <cuda_profiler_api.h>

#include <FilterParams.h>
#include <RaycastParams.h>
#include <ParticleTraceParams.h>
#include <ParticleTraceParams.h>
#include <RenderingParams.h>
#include <ProjectionParams.h>
#include <StereoParams.h>
#include <ViewParams.h>
#include <BatchTraceParams.h>

#include <FilteringManager.h>
#include <TracingManager.h>
#include <HeatMapManager.h>
#include <RenderingManager.h>
#include <RaycasterManager.h>

#include <FlowVisToolVolumeData.h>
#include <HeatMap.h>
#include <WorkerThread.h>
#include <TimerCPU.h>
#include <RenderTexture.h>
#include <ScreenEffect.h>

// windows codec header
#include <wincodec.h>

#include <Vec.h>
#include <DirectXMath.h>
#include <ScreenGrab.h>

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
#pragma endregion


class FlowVisTool
{
public:
	ID3D11Device*			m_d3dDevice = nullptr;
	ID3D11DeviceContext*	m_d3dDeviceContex = nullptr;

	// texture to hold results from other GPUs
	ID3D11Texture2D*			g_pRenderBufferTempTex = nullptr;
	ID3D11ShaderResourceView*	g_pRenderBufferTempSRV = nullptr;
	ID3D11RenderTargetView*		g_pRenderBufferTempRTV = nullptr;

	// swapchain especially needed for screenshot
	IDXGISwapChain*				g_pSwapchain = nullptr;

	// render target to hold last finished image
	ID3D11Texture2D*        g_pRaycastFinishedTex = nullptr;
	ID3D11RenderTargetView* g_pRaycastFinishedRTV = nullptr;

	// staging textures for screenshots
	ID3D11Texture2D*        g_pStagingTex = nullptr;
	ID3D11Texture2D*		g_pRenderBufferStagingTex = nullptr;

	RenderTexture				g_renderTexture;

	std::vector<MyCudaDevice>	g_cudaDevices;
	int							g_primaryCudaDeviceIndex = -1;
	bool						g_useAllGPUs = false;

	std::vector<LineBuffers*> g_lineBuffers;
	std::vector<BallBuffers*> g_ballBuffers;
	float                     g_ballRadius = 0.011718750051f;

	TimerCPU		g_timerRendering;

	ScreenEffect	g_screenEffect;

	RenderingParameters		g_renderingParams;
	ProjectionParams		g_projParams;
	StereoParams			g_stereoParams;
	ViewParams				g_viewParams;

#pragma region VolumeDependent
	clock_t		g_lastRenderParamsUpdate = 0;

	TimerCPU	g_timerTracing;

	bool		m_isFiltering = false;
	bool		m_isRaycasting = false;

	bool		m_restartFiltering = false;

	// adding screenshot functionality
	bool		g_saveScreenshot = false;

	cudaGraphicsResource*	g_pTfEdtSRVCuda = nullptr;

	FilterParams			g_filterParams;
	RaycastParams			g_raycastParams;
	HeatMapParams			g_heatMapParams;

	std::vector<FlowVisToolVolumeData*>	g_volumes;

	// resources on primary GPU
	FlowGraph			g_flowGraph;
	FilteringManager	g_filteringManager;
	HeatMapManager		g_heatMapManager;
	RenderingManager	g_renderingManager;
	RaycasterManager	g_raycasterManager;
#pragma endregion

private:
	// Currently only particle tracing can be simultaneous. We can filter and raycast one volume at a time. This variable indicated which one should be used with everything else besides particle tracing.
	int m_selectedVolume = -1;

	

public:
	FlowVisTool();

	bool Initialize(ID3D11Device* d3dDevice, ID3D11DeviceContext* d3dDeviceContex, const std::vector<MyCudaDevice>& cudaDevices, IDXGISwapChain* _g_pSwapchain);
	void Release();
	void OnFrame(float deltaTime);

	void ReleaseLineBuffers();
	void ReleaseBallBuffers();

	void CloseVolumeFile(int idx);
	bool OpenVolumeFile(const std::string& filename);

	bool ResizeViewport(int width, int height);

	void BuildFlowGraph(const std::string& filenameTxt = "");
	bool SaveFlowGraph();
	
	// saves screenshots to file
	void SaveScreenshot_toFile(std::string  filename);
	bool SaveScreenShot();

	// save animation
	bool SaveAnimation(FlowVisTool & flowVisTool, TimeVolume* volume);


	bool LoadFlowGraph(FlowVisToolVolumeData* volumeData);
#ifdef Single
	void LoadOrBuildFlowGraph();
#endif

	void SetSelectedVolume(int selected);
	int GetSelectedvolume() { return m_selectedVolume; }

private:
	void ReleaseVolumeDependentResources();
	bool CreateVolumeDependentResources(FlowVisToolVolumeData* volumeData);

	bool InitCudaDevices();
	void ShutdownCudaDevices();

	bool ResizeRenderBuffer();

	void CheckForChanges();
	void Filtering();
	void Tracing();
	bool Rendering();
	void BlitRaycastingResults();

	void CheckForChanges(FlowVisToolVolumeData* volumeData);
	bool Tracing(FlowVisToolVolumeData* volumeData, FlowGraph& flowGraph);
	bool RenderTracingResults();

	bool ShouldStartFiltering();
	bool ShouldStartTracing(FlowVisToolVolumeData* volumeData);
	bool ShouldStartRaycasting();

	bool CanFilter();
	bool CanTrace();
	bool CanRaycast();

	void StartFiltering();
	void StartTracing(FlowVisToolVolumeData* volumeData, FlowGraph& flowGraph);
	void StartRaycasting();

	void StopFiltering();
	void StopTracing();
	void StopRaycasting();
	
	void UpdateBricks(TimeVolume* volume, const std::vector<const TimeVolumeIO::Brick*>& bricks);
	void UpdateFiltering();
	void UpdateTracing(FlowVisToolVolumeData* volumeData);
	void UpdateRaycasting();

private:
	// disable copy and assignment
	FlowVisTool(const FlowVisTool&);
	FlowVisTool& operator=(const FlowVisTool&);
};



#endif //__FLOWVISTOOL_H_