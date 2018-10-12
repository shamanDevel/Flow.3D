#include <global.h>

#include <cstdio>
#include <ctime>
#include <algorithm>
#include <fstream>
#include <sstream>
#include <vector>
#include <iostream>

#include <tchar.h>

#include <omp.h>

#include <cuda_runtime.h>
#include <cuda_d3d11_interop.h>
#include <cuda_profiler_api.h>

//#include <AntTweakBar/AntTweakBar.h>
#include <D3DX11Effect/d3dx11effect.h>

#include <cudaCompress/Instance.h>

#include <SysTools.h>
#include <CSysTools.h>
#include <cudaUtil.h>
#include <Vec.h>

#include "TransferFunctionEditor/TransferFunctionEditor.h"

//#include "DXUT.h"
#include <WICTextureLoader.h>

#include "stb_image_write.h"

#include "ProgressBarEffect.h"
#include "ScreenEffect.h"

#include "Range.h"
#include "LineMode.h"
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

#include "RenderTexture.h"

#include <imgui.h>
#include <imgui_impl_win32.h>
#include <imgui_impl_dx11.h>
#include <algorithm>


//#include "TracingBenchmark.h"
//#if 0
//#include <vld.h>
//#endif

#include <FlowVisTool.h>

#pragma region Definitions
// ImGui stuff
#ifndef WM_DPICHANGED
#define WM_DPICHANGED 0x02E0 // From Windows SDK 8.1+ headers
#endif
#pragma endregion

#pragma region GlobalVariables

FlowVisTool g_flowVisTool;

static WNDCLASSEX				g_wc;
static HWND						g_hwnd;
static ID3D11Device*            g_pd3dDevice = NULL;
static ID3D11DeviceContext*     g_pd3dDeviceContext = NULL;
static IDXGISwapChain*          g_pSwapChain = NULL;
static ID3D11RenderTargetView*  g_mainRenderTargetView = NULL;
static ID3D11DepthStencilView*	g_mainDepthStencilView = NULL;
static ID3D11Texture2D*			g_mainDepthStencilTexture = NULL;

//RenderTexture g_renderTexture;

ID3D11Texture2D* m_renderTargetTexture;
ID3D11RenderTargetView* m_renderTargetView;
ID3D11ShaderResourceView* m_shaderResourceView;

//tum3D::Vec2i            g_windowSize(0, 0);
//float            g_renderBufferSizeFactor = 2.0f;
//bool             g_keyboardShiftPressed = false;

//ProjectionParams g_projParams;
//StereoParams     g_stereoParams;
//ViewParams       g_viewParams;
//Vec2f            g_mouseScreenPosition;

//FilterParams         g_filterParams;
//RaycastParams        g_raycastParams;
//ParticleTraceParams  g_particleTraceParams;
//bool                 g_particleTracingPaused = false;
//ParticleRenderParams g_particleRenderParams;
//HeatMapParams        g_heatMapParams;

//BatchTraceParams g_batchTraceParams;


//TimeVolume       g_volume(0.8f);
//FlowGraph        g_flowGraph;

// the thread at g_primaryCudaDeviceIndex will not be started!
std::vector<MyCudaDevice> g_cudaDevices;
//int                       g_primaryCudaDeviceIndex = -1;
//bool                      g_useAllGPUs = false;


// resources on primary GPU
//FilteringManager g_filteringManager;
//TracingManager   g_tracingManager;
//RenderingManager g_renderingManager;
//HeatMapManager   g_heatMapManager;

//GPUResources            g_compressShared;
//CompressVolumeResources g_compressVolume;

//std::vector<LineBuffers*> g_lineBuffers;
//std::vector<BallBuffers*> g_ballBuffers;
//float                     g_ballRadius = 0.011718750051f;

int                       g_lineIDOverride = -1;

// texture to hold results from other GPUs
//ID3D11Texture2D*			g_pRenderBufferTempTex = nullptr;
//ID3D11ShaderResourceView*	g_pRenderBufferTempSRV = nullptr;
//ID3D11RenderTargetView*		g_pRenderBufferTempRTV = nullptr;

// render target to hold last finished image
//ID3D11Texture2D*        g_pRaycastFinishedTex = nullptr;
//ID3D11RenderTargetView* g_pRaycastFinishedRTV = nullptr;

// staging textures for screenshots
//ID3D11Texture2D*        g_pStagingTex = nullptr;
//ID3D11Texture2D*		g_pRenderBufferStagingTex = nullptr;

//ScreenEffect			g_screenEffect;
//ProgressBarEffect		g_progressBarEffect;


//Vec4f		g_backgroundColor(0.1f, 0.1f, 0.1f, 1.0f);

//bool		g_showPreview = true;

//bool		g_bRenderDomainBox = true;
//bool		g_bRenderBrickBoxes = false;
//bool		g_bRenderClipBox = true;
//bool		g_bRenderSeedBox = true;
//
//
//bool		g_redraw = true;
//bool		g_retrace = true;
//
//
//TimerCPU	g_timerTracing;
//TimerCPU	g_timerRendering;


// flag to indicate to the render callback to save a screenshot next time it is called
//bool g_saveScreenshot = false;
//bool g_saveRenderBufferScreenshot = false;


// GUI
//bool   g_bRenderUI = true;
//TwBar* g_pTwBarMain = nullptr;
//TwBar* g_pTwBarImageSequence = nullptr;
//TwBar* g_pTwBarBatchTrace = nullptr;

//clock_t	g_lastRenderParamsUpdate = 0;
//clock_t	g_lastTraceParamsUpdate = 0;
//float	g_startWorkingDelay = 0.1f;

// cached gui state - apparently values can't be queried from a TwBar?!
//uint g_guiFilterRadius[3] = { 0, 0, 0 };


//Transfer function editor
//const int				TF_LINE_MEASURES = 0;
//const int				TF_HEAT_MAP = 1;
//const int				TF_RAYTRACE = 2;
//TransferFunctionEditor  g_tfEdt(400, 250, std::vector<std::string>( {"Measures", "HeatMap", "Raytrace"} ));
//int                     g_tfTimestamp = -1;
//cudaGraphicsResource*   g_pTfEdtSRVCuda = nullptr;


// OpenMP thread count (because omp_get_num_threads is stupid)
int g_threadCount = 0;

#pragma endregion

#pragma region Utility
//bool LoadRenderingParams(const std::string& filename)
//{
//	ConfigFile config;
//	if(!config.Read(filename))
//		return false;
//
//	Vec2i windowSizeNew = g_windowSize;
//	for(size_t s = 0; s < config.GetSections().size(); s++) {
//		const ConfigSection& section = config.GetSections()[s];
//
//		std::string sectionName = section.GetName();
//
//		if(section.GetName() == "Main") {
//
//			// this is our section - parse entries
//			for(size_t e = 0; e < section.GetEntries().size(); e++) {
//				const ConfigEntry& entry = section.GetEntries()[e];
//
//				std::string entryName = entry.GetName();
//
//				if(entryName == "WindowSize") {
//					entry.GetAsVec2i(windowSizeNew);
//				} else if(entryName == "RenderBufferSizeFactor") {
//					entry.GetAsFloat(g_renderBufferSizeFactor);
//				} else if(entryName == "RenderDomainBox" || entryName == "RenderBBox") {
//					entry.GetAsBool(g_bRenderDomainBox);
//				} else if(entryName == "RenderBrickBoxes") {
//					entry.GetAsBool(g_bRenderBrickBoxes);
//				} else if(entryName == "RenderClipBox" || entryName == "RenderClipBBox") {
//					entry.GetAsBool(g_bRenderClipBox);
//				} else if(entryName == "RenderSeedBox") {
//					entry.GetAsBool(g_bRenderSeedBox);
//				} else if(entryName == "RenderUI") {
//					entry.GetAsBool(g_bRenderUI);
//				} else if(entryName == "BackgroundColor") {
//					entry.GetAsVec4f(g_backgroundColor);
//				} else {
//					printf("WARNING: LoadRenderingParams: unknown config entry \"%s\" ignored\n", entryName.c_str());
//				}
//			}
//		}
//	}
//
//	g_viewParams.ApplyConfig(config);
//	g_stereoParams.ApplyConfig(config);
//	g_filterParams.ApplyConfig(config);
//	g_raycastParams.ApplyConfig(config);
//	g_particleTraceParams.ApplyConfig(config);
//	g_particleRenderParams.ApplyConfig(config);
//	g_batchTraceParams.ApplyConfig(config);
//
//	// update filter gui state
//	for(uint i = 0; i < 3; i++)
//	{
//		g_guiFilterRadius[i] = g_filterParams.m_radius.size() > i ? g_filterParams.m_radius[i] : 0;
//	}
//
//	// forward params to tf editor
//	//g_tfEdt.setAlphaScale(TF_RAYTRACE, g_raycastParams.m_alphaScale);
//	//g_tfEdt.setTfRangeMin(TF_RAYTRACE, g_raycastParams.m_transferFunctionRangeMin);
//	//g_tfEdt.setTfRangeMax(TF_RAYTRACE, g_raycastParams.m_transferFunctionRangeMax);
//
//	// load transfer function from separate binary file
//	//std::ifstream fileTF(filename + ".tf", std::ios_base::binary);
//	//if(fileTF.good()) {
//	//	g_tfEdt.loadTransferFunction(TF_RAYTRACE, &fileTF);
//	//	fileTF.close();
//	//}
//
//	// resize window if necessary
//	if(windowSizeNew != g_windowSize)
//	{
//		g_windowSize = windowSizeNew;
//
//		// SetWindowPos wants the total size including borders, so determine how large they are...
//		RECT oldWindowRect; GetWindowRect(g_hwnd, &oldWindowRect);
//		RECT oldClientRect; GetClientRect(g_hwnd, &oldClientRect);
//		uint borderX = (oldWindowRect.right - oldWindowRect.left) - (oldClientRect.right - oldClientRect.left);
//		uint borderY = (oldWindowRect.bottom - oldWindowRect.top) - (oldClientRect.bottom - oldClientRect.top);
//		SetWindowPos(g_hwnd, HWND_TOP, 0, 0, g_windowSize.x() + borderX, g_windowSize.y() + borderY, SWP_NOMOVE);
//	}
//
//	return true;
//}
//
//bool SaveRenderingParams(const std::string& filename)
//{
//	ConfigFile config;
//
//	ConfigSection section("Main");
//	section.AddEntry(ConfigEntry("WindowSize", g_windowSize));
//	section.AddEntry(ConfigEntry("RenderBufferSizeFactor", g_renderBufferSizeFactor));
//	section.AddEntry(ConfigEntry("RenderDomainBox", g_bRenderDomainBox));
//	section.AddEntry(ConfigEntry("RenderBrickBoxes", g_bRenderBrickBoxes));
//	section.AddEntry(ConfigEntry("RenderClipBox", g_bRenderClipBox));
//	section.AddEntry(ConfigEntry("RenderSeedBox", g_bRenderSeedBox));
//	section.AddEntry(ConfigEntry("RenderUI", g_bRenderUI));
//	section.AddEntry(ConfigEntry("BackgroundColor", g_backgroundColor));
//
//	config.AddSection(section);
//
//	g_viewParams.WriteConfig(config);
//	g_stereoParams.WriteConfig(config);
//	g_filterParams.WriteConfig(config);
//	g_raycastParams.WriteConfig(config);
//	g_particleTraceParams.WriteConfig(config);
//	g_particleRenderParams.WriteConfig(config);
//
//	config.Write(filename);
//
//	std::ofstream fileTF(filename + ".tf", std::ios_base::binary);
//	//g_tfEdt.saveTransferFunction(TF_RAYTRACE, &fileTF);
//	fileTF.close();
//
//	return true;
//}

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


void GetMajorWorldPlane(const tum3D::Vec3f& vecViewX, const tum3D::Vec3f& vecViewY, const tum3D::Mat4f& matInv, tum3D::Vec3f& vecWorldX, tum3D::Vec3f& vecWorldY)
{
	tum3D::Vec4f transformedX = matInv * tum3D::Vec4f(vecViewX, 0.0f);
	tum3D::Vec4f transformedY = matInv * tum3D::Vec4f(vecViewY, 0.0f);
	vecWorldX = transformedX.xyz();
	vecWorldY = transformedY.xyz();

	tum3D::Vec3f normal; tum3D::crossProd(vecWorldX, vecWorldY, normal);

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

	tum3D::normalize(vecWorldX);
	tum3D::normalize(vecWorldY);
}

void LoadSeedTexture()
{
	std::string filename;
	if (tum3d::GetFilenameDialog("Load Texture", "Images (jpg, png, bmp)\0*.png;*.jpg;*.jpeg;*.bmp\0", filename, false)) {
		//create new texture
		ID3D11Device* pd3dDevice = g_pd3dDevice;
		std::wstring wfilename(filename.begin(), filename.end());
		ID3D11Resource* res = NULL;
		//ID3D11ShaderResourceView* srv = NULL;
		if (!FAILED(DirectX::CreateWICTextureFromFileEx(pd3dDevice, wfilename.c_str(), 0Ui64, D3D11_USAGE_STAGING, 0, D3D11_CPU_ACCESS_READ, 0, false, &res, NULL))) {
			std::cout << "Seed texture " << filename << " loaded" << std::endl;
			//delete old data
			delete[] g_flowVisTool.g_particleTraceParams.m_seedTexture.m_colors;
			g_flowVisTool.g_particleTraceParams.m_seedTexture.m_colors = NULL;
			//Copy to cpu memory
			ID3D11Texture2D* tex = NULL;
			res->QueryInterface(&tex);
			D3D11_TEXTURE2D_DESC desc;
			tex->GetDesc(&desc);
			g_flowVisTool.g_particleTraceParams.m_seedTexture.m_width = desc.Width;
			g_flowVisTool.g_particleTraceParams.m_seedTexture.m_height = desc.Height;
			g_flowVisTool.g_particleTraceParams.m_seedTexture.m_colors = new unsigned int[desc.Width * desc.Height];
			D3D11_MAPPED_SUBRESOURCE mappedResource;
			ID3D11DeviceContext* context = NULL;
			pd3dDevice->GetImmediateContext(&context);
			if (!FAILED(context->Map(tex, 0, D3D11_MAP_READ, 0, &mappedResource))) {
				for (int y = 0; y < desc.Width; ++y) {
					memcpy(&g_flowVisTool.g_particleTraceParams.m_seedTexture.m_colors[y*desc.Width], ((char*)mappedResource.pData) + (y*mappedResource.RowPitch), sizeof(unsigned int) * desc.Width);
				}
				context->Unmap(tex, 0);
			}
			SAFE_RELEASE(context);
			SAFE_RELEASE(tex);
			//reset color
			g_flowVisTool.g_particleTraceParams.m_seedTexture.m_picked.clear();
			//set seed box to domain
			g_flowVisTool.SetBoundingBoxToDomainSize();
			std::cout << "Seed texture copied to cpu memory" << std::endl;
		}
		else {
			std::cerr << "Failed to load seed texture" << std::endl;
		}
		//SAFE_RELEASE(srv);
		SAFE_RELEASE(res);
	}
}
#pragma endregion

#pragma region GUI

void SaveLinesDialog()
{
	//bool bFullscreen = (!DXUTIsWindowed());

	//if( bFullscreen ) DXUTToggleFullScreen();

	std::string filename;
	if (tum3d::GetFilenameDialog("Save Lines", "*.linebuf\0*.linebuf", filename, true)) 
	{
		filename = tum3d::RemoveExt(filename) + ".linebuf";
		float posOffset = 0.0f;
		if (g_flowVisTool.g_particleTraceParams.m_upsampledVolumeHack)
		{
			// upsampled volume is offset by half a grid spacing...
			float gridSpacingWorld = 2.0f / float(g_flowVisTool.g_volume.GetVolumeSize().maximum());
			posOffset = 0.5f * gridSpacingWorld;
		}
		if (!g_flowVisTool.g_tracingManager.GetResult()->Write(filename, posOffset))
		{
			printf("Saving lines to file %s failed!\n", filename.c_str());
		}
	}

	//if( bFullscreen ) DXUTToggleFullScreen();
}

void LoadLinesDialog()
{
	//bool bFullscreen = (!DXUTIsWindowed());

	//if( bFullscreen ) DXUTToggleFullScreen();

	std::string filename;
	if (tum3d::GetFilenameDialog("Load Lines", "*.linebuf\0*.linebuf", filename, false))
	{
		LineBuffers* pBuffers = new LineBuffers(g_pd3dDevice);
		if(!pBuffers->Read(filename, g_lineIDOverride))
		{
			printf("Loading lines from file %s failed!\n", filename.c_str());
			delete pBuffers;
		}
		else
		{
			g_flowVisTool.g_lineBuffers.push_back(pBuffers);
		}
	}

	//if( bFullscreen ) DXUTToggleFullScreen();

	g_flowVisTool.m_redraw = true;
}

void ClearLinesCallback()
{
	g_flowVisTool.ReleaseLineBuffers();
	g_flowVisTool.m_redraw = true;
}

void LoadBallsDialog()
{
	//bool bFullscreen = (!DXUTIsWindowed());

	//if( bFullscreen ) DXUTToggleFullScreen();

	std::string filename;
	if (tum3d::GetFilenameDialog("Load Balls", "*.*\0*.*", filename, false))
	{
		BallBuffers* pBuffers = new BallBuffers(g_pd3dDevice);
		if(!pBuffers->Read(filename))
		{
			printf("Loading balls from file %s failed!\n", filename.c_str());
			delete pBuffers;
		}
		else
		{
			g_flowVisTool.g_ballBuffers.push_back(pBuffers);
		}
	}

	//if( bFullscreen ) DXUTToggleFullScreen();

	g_flowVisTool.m_redraw = true;
}

void ClearBallsCallback()
{
	g_flowVisTool.ReleaseBallBuffers();
	g_flowVisTool.m_redraw = true;
}

void SaveRenderingParamsDialog()
{
	//bool bFullscreen = (!DXUTIsWindowed());

	//if( bFullscreen ) DXUTToggleFullScreen();

	//std::string filename;
	//if ( tum3d::GetFilenameDialog("Save Settings", "*.cfg\0*.cfg", filename, true) ) 
	//{
	//	filename = tum3d::RemoveExt(filename) + ".cfg";
	//	SaveRenderingParams( filename );
	//}
	
	//if( bFullscreen ) DXUTToggleFullScreen();
}

void LoadRenderingParamsDialog()
{
	//bool bFullscreen = (!DXUTIsWindowed());

	//if( bFullscreen ) DXUTToggleFullScreen();

	//std::string filename;
	//if ( tum3d::GetFilenameDialog("Load Settings", "*.cfg\0*.cfg", filename, false) ) 
	//	LoadRenderingParams( filename );

	//if( bFullscreen ) DXUTToggleFullScreen();	
}

void BuildFlowGraphCallback()
{
	g_flowVisTool.BuildFlowGraph("flowgraph.txt");
}

void SaveFlowGraphCallback()
{
	g_flowVisTool.SaveFlowGraph();
}

void LoadFlowGraphCallback()
{
	g_flowVisTool.LoadFlowGraph();
}

void LoadSliceTexture()
{
	std::string filename;
	if (tum3d::GetFilenameDialog("Load Texture", "Images (jpg, png, bmp)\0*.png;*.jpg;*.jpeg;*.bmp\0", filename, false)) {
		//release old texture
		SAFE_RELEASE(g_flowVisTool.g_particleRenderParams.m_pSliceTexture);
		//create new texture
		ID3D11Device* pd3dDevice = g_pd3dDevice;
		std::wstring wfilename(filename.begin(), filename.end());
		ID3D11Resource* tmp = NULL;
		if (!FAILED(DirectX::CreateWICTextureFromFile(pd3dDevice, wfilename.c_str(), &tmp, &g_flowVisTool.g_particleRenderParams.m_pSliceTexture))) {
			std::cout << "Slice texture " << filename << " loaded" << std::endl;
			g_flowVisTool.g_particleRenderParams.m_showSlice = true;
			g_flowVisTool.m_redraw = true;
		}
		else {
			std::cerr << "Failed to load slice texture" << std::endl;
		}
		SAFE_RELEASE(tmp);
	}
}

void LoadColorTexture()
{
	std::string filename;
	if (tum3d::GetFilenameDialog("Load Texture", "Images (jpg, png, bmp)\0*.png;*.jpg;*.jpeg;*.bmp\0", filename, false)) {
		//release old texture
		SAFE_RELEASE(g_flowVisTool.g_particleRenderParams.m_pColorTexture);
		//create new texture
		ID3D11Device* pd3dDevice = g_pd3dDevice;
		std::wstring wfilename(filename.begin(), filename.end());
		ID3D11Resource* tmp = NULL;
		if (!FAILED(DirectX::CreateWICTextureFromFile(pd3dDevice, wfilename.c_str(), &tmp, &g_flowVisTool.g_particleRenderParams.m_pColorTexture))) {
			std::cout << "Color texture " << filename << " loaded" << std::endl;
			g_flowVisTool.g_particleRenderParams.m_lineColorMode = eLineColorMode::TEXTURE;
			g_flowVisTool.m_redraw = true;
		}
		else {
			std::cerr << "Failed to load color texture" << std::endl;
		}
		SAFE_RELEASE(tmp);
	}
}
#pragma endregion

#pragma region DXUTCallbacks

//--------------------------------------------------------------------------------------
// Reject any D3D11 devices that aren't acceptable by returning false
//--------------------------------------------------------------------------------------
//bool CALLBACK IsD3D11DeviceAcceptable( const CD3D11EnumAdapterInfo *AdapterInfo, UINT Output, const CD3D11EnumDeviceInfo *DeviceInfo,
//                                       DXGI_FORMAT BackBufferFormat, bool bWindowed, void* pUserContext )
//{
//	bool deviceAcceptable = false;
//
//	IDXGIFactory *pFactory;
//	if(FAILED(CreateDXGIFactory(__uuidof(IDXGIFactory), (void**)(&pFactory) )))
//		return false;
//
//	// get a candidate DXGI adapter
//	IDXGIAdapter* pAdapter = 0;
//	if(FAILED(pFactory->EnumAdapters(AdapterInfo->AdapterOrdinal, &pAdapter))) {
//		pFactory->Release();
//		return false;
//	}
//
//	// Check if adapter is CUDA capable
//	// query to see if there exists a corresponding compute device
//	cudaError err;
//	int cudaDevice;
//	err = cudaD3D11GetDevice(&cudaDevice, pAdapter);
//	if (err == cudaSuccess) {
//		cudaDeviceProp prop;
//		if(cudaSuccess != cudaGetDeviceProperties(&prop, cudaDevice)) {
//			exit(-1);
//		}
//		if(prop.major >= 2)
//			deviceAcceptable = true;
//	}
//
//	pAdapter->Release();
//	pFactory->Release();
//
//	// clear any errors we got while querying invalid compute devices
//	cudaGetLastError();
//
//	return deviceAcceptable;
//}
//
//
////--------------------------------------------------------------------------------------
//// Called right before creating a device, allowing the app to modify the device settings as needed
////--------------------------------------------------------------------------------------
//bool CALLBACK ModifyDeviceSettings( DXUTDeviceSettings* pDeviceSettings, void* pUserContext )
//{
//	return true;
//}
//
//
//--------------------------------------------------------------------------------------
// Create any D3D11 resources that aren't dependent on the back buffer
//--------------------------------------------------------------------------------------
HRESULT OnD3D11CreateDevice()
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

	if(!g_flowVisTool.InitCudaDevices()) {
		printf("InitCudaDevices returned false");
		return E_FAIL;
	}

	// don't page-lock brick data - CUDA doesn't seem to like when we page-lock lots of smallish mem areas..
	//g_volume.SetPageLockBrickData(true);

	HRESULT hr;

	if (FAILED(hr = g_flowVisTool.g_screenEffect.Create(g_pd3dDevice))) {
		return hr;
	}

	if (FAILED(hr = g_flowVisTool.g_progressBarEffect.Create(g_pd3dDevice))) {
		return hr;
	}


	if (g_flowVisTool.g_volume.IsOpen()) {
		// this creates the cudaCompress instance etc
		if (FAILED(hr = g_flowVisTool.CreateVolumeDependentResources())) {
			return hr;
		}
	}

	//InitTwBars(pd3dDevice, pBackBufferSurfaceDesc->Height);

	//if(FAILED(hr = g_tfEdt.onCreateDevice( pd3dDevice ))) {
	//	return hr;
	//}
	//g_particleRenderParams.m_pTransferFunction = g_tfEdt.getSRV(TF_LINE_MEASURES);
	//g_heatMapParams.m_pTransferFunction = g_tfEdt.getSRV(TF_HEAT_MAP);
	//cudaSafeCall(cudaGraphicsD3D11RegisterResource(&g_pTfEdtSRVCuda, g_tfEdt.getTexture(TF_RAYTRACE), cudaGraphicsRegisterFlagsNone));

	//TracingBenchmark bench;
	//bench.RunBenchmark(g_particleTraceParams, 64, 2048, 5, 0);
	//bench.RunBenchmark(g_particleTraceParams, 64, 2048, 5, 1);
	//exit(42);

	return S_OK;
}
//
//
//--------------------------------------------------------------------------------------
// Create any D3D11 resources that depend on the back buffer
//--------------------------------------------------------------------------------------

// save a screenshot from the framebuffer
//void SaveScreenshot(ID3D11DeviceContext* pd3dImmediateContext, const std::string& filename)
//{
//	ID3D11Resource* pSwapChainTex;
//	g_mainRenderTargetView->GetResource(&pSwapChainTex);
//	pd3dImmediateContext->CopyResource(g_pStagingTex, pSwapChainTex);
//	SAFE_RELEASE(pSwapChainTex);
//
//	D3D11_MAPPED_SUBRESOURCE mapped = { 0 };
//	pd3dImmediateContext->Map(g_pStagingTex, 0, D3D11_MAP_READ, 0, &mapped);
//
//	stbi_write_png(filename.c_str(), g_windowSize.x(), g_windowSize.y(), 4, mapped.pData, mapped.RowPitch);
//
//	//stbi_write_bmp(filename.c_str(), g_windowSize.x(), g_windowSize.y(), 4, mapped.pData);
//
//	pd3dImmediateContext->Unmap(g_pStagingTex, 0);
//}
//
//// save a screenshot from the (possibly higher-resolution) render buffer
//void SaveRenderBufferScreenshot(ID3D11DeviceContext* pd3dImmediateContext, const std::string& filename)
//{
//	if(!g_renderingManager.IsCreated()) return;
//
//	bool singleGPU = (!g_useAllGPUs || (g_cudaDevices.size() == 1));
//	//ID3D11Texture2D* pSrcTex = (singleGPU ? g_renderingManager.GetRaycastTex() : g_pRenderBufferTempTex);
//	//HACK: for now, save just the opaque tex - actually need to blend here...
//	ID3D11Texture2D* pSrcTex = (singleGPU ? g_renderingManager.GetOpaqueTex() : g_pRenderBufferTempTex);
//	pd3dImmediateContext->CopyResource(g_pRenderBufferStagingTex, pSrcTex);
//
//	D3D11_MAPPED_SUBRESOURCE mapped = { 0 };
//	pd3dImmediateContext->Map(g_pRenderBufferStagingTex, 0, D3D11_MAP_READ, 0, &mapped);
//
//	stbi_write_png(filename.c_str(), g_projParams.m_imageWidth, g_projParams.m_imageHeight, 4, mapped.pData, mapped.RowPitch);
//
//	//stbi_write_bmp(filename.c_str(), g_projParams.m_imageWidth, g_projParams.m_imageHeight, 4, mapped.pData);
//
//	pd3dImmediateContext->Unmap(g_pRenderBufferStagingTex, 0);
//}


//--------------------------------------------------------------------------------------
// Release D3D11 resources created in OnD3D11ResizedSwapChain 
//--------------------------------------------------------------------------------------
//void OnD3D11ReleasingSwapChain( void* pUserContext )
//{
//	g_windowSize.x() = 0;
//	g_windowSize.y() = 0;
//	//TwWindowSize(0, 0);
//
//	g_flowVisTool.ResizeRenderBuffer();
//
//	SAFE_RELEASE(g_pStagingTex);
//
//	SAFE_RELEASE(g_pRaycastFinishedRTV);
//	SAFE_RELEASE(g_pRaycastFinishedTex);
//
//	//g_tfEdt.onReleasingSwapChain();
//}


//--------------------------------------------------------------------------------------
// Release D3D11 resources created in OnD3D11CreateDevice 
//--------------------------------------------------------------------------------------



//--------------------------------------------------------------------------------------
// Call if device was removed.  Return true to find a new device, false to quit
//--------------------------------------------------------------------------------------
//bool OnDeviceRemoved( void* pUserContext )
//{
//	return true;
//}


// Picks the seed from the seed texture and places is into 'seed' if there is a intersection.
// The function returns true iff there was an intersection
// 'intersection' will contain the 3D-coordinate of intersection
//bool PickSeed(unsigned int* pSeed, Vec3f* pIntersection) {
//	std::cout << "Pick seed, mouse position = (" << g_mouseScreenPosition.x() << ", " << g_mouseScreenPosition.y() << ")" << std::endl;
//	//create ray through the mouse
//	Mat4f viewProjMat = g_projParams.BuildProjectionMatrix(EYE_CYCLOP, 0.0f, g_cudaDevices[g_primaryCudaDeviceIndex].range)
//		* g_viewParams.BuildViewMatrix(EYE_CYCLOP, 0.0f);
//	Mat4f invViewProjMat;
//	tum3D::invert4x4(viewProjMat, invViewProjMat);
//	Vec4f start4 = invViewProjMat * Vec4f(g_mouseScreenPosition.x(), g_mouseScreenPosition.y(), 0.01f, 1.0f);
//	Vec3f start = start4.xyz() / start4.w();
//	Vec4f end4 = invViewProjMat * Vec4f(g_mouseScreenPosition.x(), g_mouseScreenPosition.y(), 0.99f, 1.0f);
//	Vec3f end = end4.xyz() / end4.w();
//	std::cout << "Ray, start=" << start << ", end=" << end << std::endl;
//	//cut ray with the xy-plane
//	Vec3f dir = end - start;
//	normalize(dir);
//	Vec3f n = Vec3f(0, 0, 1); //normal of the plane
//	float d = (-start).dot(n) / (dir.dot(n));
//	if (d < 0) return false; //we are behind the plane
//	Vec3f intersection = start + d * dir;
//	std::cout << "Intersection: " << intersection << std::endl;
//	if (pIntersection) *pIntersection = intersection;
//	//Test if seed texture is loaded
//	if (g_particleTraceParams.m_seedTexture.m_colors == NULL) {
//		//no texture
//		*pSeed = 0;
//		return true;
//	}
//	else {
//		//a seed texture was found
//		//check if intersection is in the volume
//		if (intersection > -g_volume.GetVolumeHalfSizeWorld()
//			&& intersection < g_volume.GetVolumeHalfSizeWorld()) {
//			//inside, convert to texture coordinates
//			Vec3f localIntersection = (intersection + g_volume.GetVolumeHalfSizeWorld()) / (2 * g_volume.GetVolumeHalfSizeWorld());
//			int texX = (int)(localIntersection.x() * g_particleTraceParams.m_seedTexture.m_width);
//			int texY = (int)(localIntersection.y() * g_particleTraceParams.m_seedTexture.m_height);
//			texY = g_particleTraceParams.m_seedTexture.m_height - texY - 1;
//			unsigned int color = g_particleTraceParams.m_seedTexture.m_colors[texX + texY * g_particleTraceParams.m_seedTexture.m_height];
//			printf("Pick color at position (%d, %d): 0x%08x\n", texX, texY, color);
//			*pSeed = color;
//			return true;
//		}
//		else {
//			std::cout << "Outside the bounds" << std::endl;
//			return false;
//		}
//	}
//}

// NOTE: OnKeyboard and OnMouse are *not* registered as DXUT callbacks because this doesn't work well with AntTweakBar
//--------------------------------------------------------------------------------------
// Handle key presses
//--------------------------------------------------------------------------------------
//void OnKeyboard( UINT nChar, bool bKeyDown, bool bAltDown, bool bHandledByGUI )
//{
//	if (nChar == VK_RSHIFT || nChar == VK_SHIFT) {
//		g_keyboardShiftPressed = bKeyDown;
//	}
//
//	if(bKeyDown)
//	{
//		switch(nChar)
//		{
//			case VK_RETURN :
//			{
//				if(bAltDown) DXUTToggleFullScreen();
//				break;
//			}
//			case 'U' : 
//			{
//				g_bRenderUI = !g_bRenderUI; 
//				break;
//			}
//			case VK_F5 :
//			{
//				SaveRenderingParams("quicksave.cfg");
//				break;
//			}
//			case VK_F6 :
//			//case VK_F9 :
//			{
//				LoadRenderingParams("quicksave.cfg");
//				break;
//			}
//			case 'P' : 
//			{
//				unsigned int color;
//				Vec3f pos;
//				bool ret = PickSeed(&color, &pos);
//
//				//Test if seed texture is loaded
//				if (g_particleTraceParams.m_seedTexture.m_colors == NULL) {
//					//no texture, just adjust seed box
//					g_particleTraceParams.m_seedBoxMin = pos - Vec3f(0.05f);
//					g_particleTraceParams.m_seedBoxSize = Vec3f(0.1f);
//				}
//				else {
//					//a seed texture was found
//					//check if intersection is in the volume
//					if (ret) {
//						if (g_keyboardShiftPressed) {
//							if (g_particleTraceParams.m_seedTexture.m_picked.count(color)==0)
//								g_particleTraceParams.m_seedTexture.m_picked.insert(color);
//							else //toggle selection status
//								g_particleTraceParams.m_seedTexture.m_picked.erase(color);
//						} else {
//							g_particleTraceParams.m_seedTexture.m_picked.clear();
//							g_particleTraceParams.m_seedTexture.m_picked.insert(color);
//						}
//					} else {
//						if (!g_keyboardShiftPressed) {
//							g_particleTraceParams.m_seedTexture.m_picked.clear(); //disable seed from texture
//						}
//					}
//					//update heat map params
//					g_heatMapParams.m_recordTexture = g_particleTraceParams.m_seedTexture;
//					//check if the rendered channels are still alive
//					if (g_heatMapParams.m_recordTexture.m_picked.count(g_heatMapParams.m_renderedChannels[0]) == 0) {
//						if (g_heatMapParams.m_recordTexture.m_picked.empty()) {
//							g_heatMapParams.m_renderedChannels[0] = 0;
//						}
//						else {
//							g_heatMapParams.m_renderedChannels[0] = *g_heatMapParams.m_recordTexture.m_picked.begin();
//						}
//					}
//					if (g_heatMapParams.m_recordTexture.m_picked.count(g_heatMapParams.m_renderedChannels[1]) == 0) {
//						g_heatMapParams.m_renderedChannels[1] = 0;
//					}
//				}
//
//				break;
//			}
//			case '1':
//			case '2':
//			{
//				//Update rendered channel
//				int channel = nChar - '1';
//				
//				unsigned int color;
//				Vec3f pos;
//				bool ret = PickSeed(&color, &pos);
//				if (ret) {
//					g_heatMapParams.m_renderedChannels[channel] = color;
//				}
//				else {
//					g_heatMapParams.m_renderedChannels[channel] = 0;
//				}
//
//				break;
//			}
//			default : return;
//		}
//	}
//}
//
//
//// NOTE: OnKeyboard and OnMouse are *not* registered as DXUT callbacks because this doesn't work well with AntTweakBar
////--------------------------------------------------------------------------------------
//// Handle mouse button presses
////--------------------------------------------------------------------------------------
//void OnMouse( bool bLeftButtonDown, bool bRightButtonDown, bool bMiddleButtonDown,
//              bool bSideButton1Down, bool bSideButton2Down, int nMouseWheelDelta,
//              int xPos, int yPos, bool bCtrlDown, bool bShiftDown, bool bHandledByGUI )
//{
//	static bool bLeftButtonDownPrev = bLeftButtonDown;
//	static bool bRightButtonDownPrev = bRightButtonDown;
//	static bool bMiddleButtonDownPrev = bMiddleButtonDown;
//	static int xPosPrev = xPos;
//	static int yPosPrev = yPos;
//
//	g_mouseScreenPosition = Vec2f((xPos / (float)g_windowSize.x()) * 2 - 1, -((yPos / (float)g_windowSize.y()) * 2 - 1));
//
//	if(!bHandledByGUI && !g_imageSequence.Running) {
//		int xPosDelta = xPos - xPosPrev;
//		int yPosDelta = yPos - yPosPrev;
//
//		float xDelta = float(xPosDelta) / float(g_windowSize.y()); // g_windowSize.y() not a typo: consistent speed in x and y
//		float yDelta = float(yPosDelta) / float(g_windowSize.y());
//
//		Mat4f inverseView;
//		tum3D::invert4x4(g_viewParams.BuildViewMatrix(EYE_CYCLOP, 0.0f), inverseView);
//		Vec3f majorX, majorY;
//		GetMajorWorldPlane(Vec3f(1.0f, 0.0f, 0.0f), Vec3f(0.0f, 1.0f, 0.0f), inverseView, majorX, majorY);
//
//		if(bLeftButtonDown && bLeftButtonDownPrev) {
//			// rotate
//			Vec4f rotationX;
//			tum3D::rotationQuaternion(xDelta * PI, Vec3f(0.0f, 1.0f, 0.0f), rotationX);
//			Vec4f rotationY;
//			tum3D::rotationQuaternion(yDelta * PI, Vec3f(1.0f, 0.0f, 0.0f), rotationY);
//
//			//add to global rotation
//			Vec4f temp;
//			tum3D::multQuaternion(rotationX, g_rotationX, temp); g_rotationX = temp;
//			tum3D::multQuaternion(rotationY, g_rotationY, temp); g_rotationY = temp;
//
//			//combine and set to view params
//			tum3D::multQuaternion(g_rotationY, g_rotationX, temp);
//			tum3D::multQuaternion(temp, g_rotation, g_viewParams.m_rotationQuat);
//			//tum3D::multQuaternion(g_rotationY, g_rotationX, g_viewParams.m_rotationQuat);
//		}
//		if (!bLeftButtonDown && bLeftButtonDownPrev) {
//			//mouse released, so store the current rotation and reset the partial rotation
//			//This is needed so that the trackball starts with a new projection
//			g_rotation = g_viewParams.m_rotationQuat;
//			g_rotationX = Vec4f(1, 0, 0, 0);
//			g_rotationY = Vec4f(1, 0, 0, 0);
//			printf("push rotation\n");
//		}
//		if(bMiddleButtonDown && bMiddleButtonDownPrev) {
//			if(bShiftDown) {
//				// scale seed region
//				Vec3f expX = majorX *  xDelta;
//				Vec3f expY = majorY * -yDelta;
//				Vec3f scalingX(pow(4.0f, expX.x()), pow(4.0f, expX.y()), pow(4.0f, expX.z()));
//				Vec3f scalingY(pow(4.0f, expY.x()), pow(4.0f, expY.y()), pow(4.0f, expY.z()));
//				g_particleTraceParams.ScaleSeedBox(scalingX * scalingY);
//			} else {
//				// zoom
//				g_viewParams.m_viewDistance *= pow(5.0f, -yDelta);
//			}
//		}
//		if(bRightButtonDown && bRightButtonDownPrev) {
//			if(bShiftDown) {
//				// move seed region
//				Vec3f translation = xDelta * majorX - yDelta * majorY;
//				g_particleTraceParams.MoveSeedBox(translation);
//			} else {
//				// move domain/camera
//				Mat3f rotationMat;
//				tum3D::convertQuaternionToRotMat(g_viewParams.m_rotationQuat, rotationMat);
//				Vec3f xVec = rotationMat.getRow(0);
//				Vec3f yVec = rotationMat.getRow(1);
//				g_viewParams.m_lookAt -= tan(0.5f * g_projParams.m_fovy) * g_viewParams.m_viewDistance * (xDelta * xVec - yDelta * yVec);
//			}
//		}
//	}
//
//	// update "previous" state for next call
//	bLeftButtonDownPrev = bLeftButtonDown;
//	bRightButtonDownPrev = bRightButtonDown;
//	bMiddleButtonDownPrev = bMiddleButtonDown;
//	xPosPrev = xPos;
//	yPosPrev = yPos;
//}


//--------------------------------------------------------------------------------------
// Handle messages to the application
//--------------------------------------------------------------------------------------
//LRESULT CALLBACK MsgProc( HWND hWnd, UINT uMsg, WPARAM wParam, LPARAM lParam, bool* pbNoFurtherProcessing, void* pUserContext )
//{
//	bool bHandledByGUI = false;
//
//	//*pbNoFurtherProcessing = g_tfEdt.msgProc( hWnd, uMsg, wParam, lParam );
//	//if(*pbNoFurtherProcessing) {
//	//	bHandledByGUI = true;
//	//}
//
//	if(g_bRenderUI) // only handle UI events if it is visible
//	{
//		//if(TwEventWin(hWnd, uMsg, wParam, lParam))
//		//{
//		//	bHandledByGUI = true;
//		//	*pbNoFurtherProcessing = true;
//		//	// don't return, still pass event to keyboard/mouse callbacks (so they can keep their state valid)
//		//}
//	}
//
//	// Copied from DXUT: Consolidate the keyboard messages and pass them to the app's keyboard callback
//	if( uMsg == WM_KEYDOWN ||
//	    uMsg == WM_SYSKEYDOWN ||
//	    uMsg == WM_KEYUP ||
//	    uMsg == WM_SYSKEYUP )
//	{
//		bool bKeyDown = ( uMsg == WM_KEYDOWN || uMsg == WM_SYSKEYDOWN );
//		DWORD dwMask = ( 1 << 29 );
//		bool bAltDown = ( ( lParam & dwMask ) != 0 );
//		//bool bShiftDown = ((nMouseButtonState & MK_SHIFT) != 0);
//
//		OnKeyboard( ( UINT )wParam, bKeyDown, bAltDown, bHandledByGUI );
//	}
//
//	// Copied from DXUT: Consolidate the mouse button messages and pass them to the app's mouse callback
//	if( uMsg == WM_LBUTTONDOWN ||
//	    uMsg == WM_LBUTTONUP ||
//	    uMsg == WM_LBUTTONDBLCLK ||
//	    uMsg == WM_MBUTTONDOWN ||
//	    uMsg == WM_MBUTTONUP ||
//	    uMsg == WM_MBUTTONDBLCLK ||
//	    uMsg == WM_RBUTTONDOWN ||
//	    uMsg == WM_RBUTTONUP ||
//	    uMsg == WM_RBUTTONDBLCLK ||
//	    uMsg == WM_XBUTTONDOWN ||
//	    uMsg == WM_XBUTTONUP ||
//	    uMsg == WM_XBUTTONDBLCLK ||
//	    uMsg == WM_MOUSEWHEEL ||
//	    uMsg == WM_MOUSEMOVE )
//	{
//		int xPos = ( short )LOWORD( lParam );
//		int yPos = ( short )HIWORD( lParam );
//
//		if( uMsg == WM_MOUSEWHEEL )
//		{
//			// WM_MOUSEWHEEL passes screen mouse coords
//			// so convert them to client coords
//			POINT pt;
//			pt.x = xPos; pt.y = yPos;
//			ScreenToClient( hWnd, &pt );
//			xPos = pt.x; yPos = pt.y;
//		}
//
//		int nMouseWheelDelta = 0;
//		if( uMsg == WM_MOUSEWHEEL )
//			nMouseWheelDelta = ( short )HIWORD( wParam );
//
//		int nMouseButtonState = LOWORD( wParam );
//		bool bLeftButton = ( ( nMouseButtonState & MK_LBUTTON ) != 0 );
//		bool bRightButton = ( ( nMouseButtonState & MK_RBUTTON ) != 0 );
//		bool bMiddleButton = ( ( nMouseButtonState & MK_MBUTTON ) != 0 );
//		bool bSideButton1 = ( ( nMouseButtonState & MK_XBUTTON1 ) != 0 );
//		bool bSideButton2 = ( ( nMouseButtonState & MK_XBUTTON2 ) != 0 );
//		bool bCtrlDown = ( ( nMouseButtonState & MK_CONTROL ) != 0 );
//		bool bShiftDown = ( ( nMouseButtonState & MK_SHIFT ) != 0 );
//
//		OnMouse( bLeftButton, bRightButton, bMiddleButton, bSideButton1, bSideButton2, nMouseWheelDelta, xPos, yPos, bCtrlDown, bShiftDown, bHandledByGUI );
//	}
//
//	return 0;
//}


//--------------------------------------------------------------------------------------
// Handle updates to the scene
//--------------------------------------------------------------------------------------
void OnFrameMove( double dTime, float fElapsedTime, void* pUserContext )
{
	// check if we should update the window title
	bool update = false;

	// update if image or window size changed
	static uint s_imageWidth = 0;
	static uint s_imageHeight = 0;
	if (s_imageWidth != g_flowVisTool.g_projParams.m_imageWidth || s_imageHeight != g_flowVisTool.g_projParams.m_imageHeight) {
		s_imageWidth = g_flowVisTool.g_projParams.m_imageWidth;
		s_imageHeight = g_flowVisTool.g_projParams.m_imageHeight;
		update = true;
	}

	static tum3D::Vec2i s_windowSize(0, 0);
	if (s_windowSize != g_flowVisTool.g_windowSize) {
		s_windowSize = g_flowVisTool.g_windowSize;
		update = true;
	}

	// update fps once per second
	static float s_fps = 0.0f;
	static float s_mspf = 0.0f;
	static double s_dStatLastFPSUpdate = 0.0;
	double dt = dTime - s_dStatLastFPSUpdate;
	if (dt >= 1.0) {
		s_fps = ImGui::GetIO().Framerate;
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
	float timeTraceNew = g_flowVisTool.g_timerTracing.GetElapsedTimeMS();
	if(s_timeTrace != timeTraceNew)
	{
		s_timeTrace = timeTraceNew;
		update = true;
	}

	static float s_timeRender = 0.0f;
	float timeRenderNew = g_flowVisTool.g_timerRendering.GetElapsedTimeMS();
	if(s_timeRender != timeRenderNew)
	{
		s_timeRender = timeRenderNew;
		update = true;
	}

	// update window title if something relevant changed
	if (update) {
		const size_t len = 512;
		wchar_t str[len];
		int pos = swprintf_s(str, L"FlowVisTool %ux%u (%ux%u) @ %.2f fps / %.2f ms", s_windowSize.x(), s_windowSize.y(), s_imageWidth, s_imageHeight, s_fps, s_mspf);
		//pos += swprintf_s(str + pos, len - pos, L"   Upload/Decomp (GPU): %.2f ms (%.2f-%.2f-%.2f : %u)", s_timings.UploadDecompressGPU.Total, s_timings.UploadDecompressGPU.Min, s_timings.UploadDecompressGPU.Avg, s_timings.UploadDecompressGPU.Max, s_timings.UploadDecompressGPU.Count);
		//pos += swprintf_s(str + pos, len - pos, L"   Raycast (GPU): %.2f ms (%.2f-%.2f-%.2f : %u)", s_timings.RaycastGPU.Total, s_timings.RaycastGPU.Min, s_timings.RaycastGPU.Avg, s_timings.RaycastGPU.Max, s_timings.RaycastGPU.Count);
		//pos += swprintf_s(str + pos, len - pos, L"   Render (Wall): %.2f ms", s_timings.RenderWall);
		pos += swprintf_s(str + pos, len - pos, L"   Trace Time: %.2f ms", s_timeTrace);
		pos += swprintf_s(str + pos, len - pos, L"   Render Time: %.2f ms", s_timeRender);
		SetWindowText(g_hwnd, str);
	}
}

#pragma endregion

#pragma region InitExitMain

D3D11_TEXTURE2D_DESC GetBackBufferSurfaceDesc()
{
	//IDXGISurface* pBackBuffer;
	//g_pSwapChain->GetBuffer(0, __uuidof(IDXGISurface), (LPVOID*)&pBackBuffer);

	//DXGI_SURFACE_DESC backBufferSurfaceDesc;
	//pBackBuffer->GetDesc(&backBufferSurfaceDesc);

	//return backBufferSurfaceDesc;

	ID3D11Texture2D* pBackBuffer;
	HRESULT hr = g_pSwapChain->GetBuffer(0, IID_PPV_ARGS(&pBackBuffer));
	//if (FAILED(hr))
	//	return hr;
	D3D11_TEXTURE2D_DESC backBufferSurfaceDesc;
	pBackBuffer->GetDesc(&backBufferSurfaceDesc);

	SAFE_RELEASE(pBackBuffer);

	return backBufferSurfaceDesc;
}

bool InitApp()
{
	return GetCudaDevices();
}

void SetupImGui()
{
	//ImGui_ImplWin32_EnableDpiAwareness();

	// Setup Dear ImGui binding
	IMGUI_CHECKVERSION();
	ImGui::CreateContext();
	ImGuiIO& io = ImGui::GetIO(); (void)io;

	io.ConfigFlags = 0;

	io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;       // Enable Keyboard Controls
	io.ConfigFlags |= ImGuiConfigFlags_DockingEnable;           // Enable Docking
	//io.ConfigFlags |= ImGuiConfigFlags_ViewportsEnable;         // Enable Multi-Viewport / Platform Windows
	//io.ConfigFlags |= ImGuiConfigFlags_ViewportsNoTaskBarIcons;
	//io.ConfigFlags |= ImGuiConfigFlags_ViewportsNoMerge;
	//io.ConfigFlags |= ImGuiConfigFlags_DpiEnableScaleFonts;     // FIXME-DPI: THIS CURRENTLY DOESN'T WORK AS EXPECTED. DON'T USE IN USER APP!
	//io.ConfigFlags |= ImGuiConfigFlags_DpiEnableScaleViewports; // FIXME-DPI
	
	io.ConfigResizeWindowsFromEdges = true;
	io.ConfigDockingWithShift = false;

	if (!ImGui_ImplWin32_Init(g_hwnd))
		std::cerr << "Failed to initialize 'ImGui_ImplWin32'" << std::endl;

	//ID3D11Device* device = DXUTGetD3D11Device();
	//ID3D11DeviceContext* deviceContext;

	//device->GetImmediateContext(&deviceContext);

	if (!ImGui_ImplDX11_Init(g_pd3dDevice, g_pd3dDeviceContext))
		std::cerr << "Failed to initialize 'ImGui_ImplDX11'" << std::endl;

	//SAFE_RELEASE(deviceContext);

	// Setup style
	ImGui::GetStyle().TabRounding = 0.0f;
	ImGui::GetStyle().ScrollbarRounding = 0.0f;
	ImGui::GetStyle().WindowRounding = 0.0f;
	ImGui::StyleColorsDark();
	//ImGui::StyleColorsClassic();

	ImGui::GetStyle().Colors[ImGuiCol_Tab] = ImVec4(0.125f, 0.243f, 0.404f, 0.863f);
	ImGui::GetStyle().Colors[ImGuiCol_TitleBgActive] = ImVec4(27/255.0f, 27/255.0f, 27/255.0f, 1.0f);

	// Load Fonts
	// - If no fonts are loaded, dear imgui will use the default font. You can also load multiple fonts and use ImGui::PushFont()/PopFont() to select them. 
	// - AddFontFromFileTTF() will return the ImFont* so you can store it if you need to select the font among multiple. 
	// - If the file cannot be loaded, the function will return NULL. Please handle those errors in your application (e.g. use an assertion, or display an error and quit).
	// - The fonts will be rasterized at a given size (w/ oversampling) and stored into a texture when calling ImFontAtlas::Build()/GetTexDataAsXXXX(), which ImGui_ImplXXXX_NewFrame below will call.
	// - Read 'misc/fonts/README.txt' for more instructions and details.
	// - Remember that in C/C++ if you want to include a backslash \ in a string literal you need to write a double backslash \\ !
	//io.Fonts->AddFontDefault();
	io.Fonts->AddFontFromFileTTF("../resources/Roboto-Medium.ttf", 16.0f);
	//io.Fonts->AddFontFromFileTTF("./resources/DroidSans.ttf", 16.0f);
	//io.Fonts->AddFontFromFileTTF("./resources/Cousine-Regular.ttf", 16.0f);

	//ImFont* font = io.Fonts->AddFontFromFileTTF("c:\\Windows\\Fonts\\ArialUni.ttf", 18.0f, NULL, io.Fonts->GetGlyphRangesJapanese());
	//IM_ASSERT(font != NULL);
}

void ImGuiCleanup()
{
	ImGui_ImplDX11_Shutdown();
	ImGui_ImplWin32_Shutdown();
	ImGui::DestroyContext();
}

//--------------------------------------------------------------------------------------
// Sets the viewport, render target view, and depth stencil view.
//--------------------------------------------------------------------------------------
HRESULT SetupD3D11Views()
{
	HRESULT hr = S_OK;

	// Setup the viewport to match the backbuffer
	D3D11_VIEWPORT vp;
	vp.Width = (FLOAT)GetBackBufferSurfaceDesc().Width;
	vp.Height = (FLOAT)GetBackBufferSurfaceDesc().Height;
	vp.MinDepth = 0;
	vp.MaxDepth = 1;
	vp.TopLeftX = 0;
	vp.TopLeftY = 0;
	g_pd3dDeviceContext->RSSetViewports(1, &vp);

	// Set the render targets
	g_pd3dDeviceContext->OMSetRenderTargets(1, &g_mainRenderTargetView, g_mainDepthStencilView);

	return hr;
}

//--------------------------------------------------------------------------------------
// Creates a render target view, and depth stencil texture and view.
//--------------------------------------------------------------------------------------
HRESULT CreateD3D11Views()
{
	HRESULT hr = S_OK;

	// Get the back buffer and desc
	ID3D11Texture2D* pBackBuffer;
	//hr = g_pSwapChain->GetBuffer(0, IID_PPV_ARGS(&pBackBuffer));
	hr = g_pSwapChain->GetBuffer(0, __uuidof(ID3D11Texture2D), (LPVOID*)&pBackBuffer);
	if (FAILED(hr))
		return hr;

	hr = g_pd3dDevice->CreateRenderTargetView(pBackBuffer, NULL, &g_mainRenderTargetView);

	//D3D11_TEXTURE2D_DESC backBufferSurfaceDesc;
	//pBackBuffer->GetDesc(&backBufferSurfaceDesc);

	// Create the render target view
	//hr = g_pd3dDevice->CreateRenderTargetView(pBackBuffer, nullptr, &g_mainRenderTargetView);
	SAFE_RELEASE(pBackBuffer);

	if (FAILED(hr))
		return hr;

	//if (pDeviceSettings->d3d11.AutoCreateDepthStencil)
	{
		// Create depth stencil texture
		D3D11_TEXTURE2D_DESC descDepth;
		descDepth.Width = GetBackBufferSurfaceDesc().Width;
		descDepth.Height = GetBackBufferSurfaceDesc().Height;
		descDepth.MipLevels = 1;
		descDepth.ArraySize = 1;
		descDepth.Format = DXGI_FORMAT_D24_UNORM_S8_UINT;
		descDepth.SampleDesc.Count = 1;
		descDepth.SampleDesc.Quality = 0;
		descDepth.Usage = D3D11_USAGE_DEFAULT;
		descDepth.BindFlags = D3D11_BIND_DEPTH_STENCIL;
		descDepth.CPUAccessFlags = 0;
		descDepth.MiscFlags = 0;

		SAFE_RELEASE(g_mainDepthStencilTexture);

		hr = g_pd3dDevice->CreateTexture2D(&descDepth, nullptr, &g_mainDepthStencilTexture);
		if (FAILED(hr))
			return hr;
		//DXUT_SetDebugName(pDepthStencil, "DXUT");

		// Create the depth stencil view
		D3D11_DEPTH_STENCIL_VIEW_DESC descDSV;
		descDSV.Format = descDepth.Format;
		descDSV.Flags = 0;
		if (descDepth.SampleDesc.Count > 1)
			descDSV.ViewDimension = D3D11_DSV_DIMENSION_TEXTURE2DMS;
		else
			descDSV.ViewDimension = D3D11_DSV_DIMENSION_TEXTURE2D;
		descDSV.Texture2D.MipSlice = 0;
		hr = g_pd3dDevice->CreateDepthStencilView(g_mainDepthStencilTexture, &descDSV, &g_mainDepthStencilView);
		if (FAILED(hr))
			return hr;
	}

	hr = SetupD3D11Views();
	if (FAILED(hr))
		return hr;

	g_flowVisTool.SetDXStuff(g_pd3dDevice, g_pd3dDeviceContext, g_mainRenderTargetView, g_mainDepthStencilView);

	return hr;
}

//void CreateRenderTarget()
//{
//	ID3D11Texture2D* pBackBuffer;
//	g_pSwapChain->GetBuffer(0, __uuidof(ID3D11Texture2D), (LPVOID*)&pBackBuffer);
//	g_pd3dDevice->CreateRenderTargetView(pBackBuffer, NULL, &g_mainRenderTargetView);
//	pBackBuffer->Release();
//}

void CleanupRenderTarget()
{
	if (g_mainRenderTargetView) { g_mainRenderTargetView->Release(); g_mainRenderTargetView = NULL; }
	if (g_mainDepthStencilView) { g_mainDepthStencilView->Release(); g_mainDepthStencilView = NULL; }
}

HRESULT CreateDeviceD3D(HWND hWnd)
{
	// Setup swap chain
	DXGI_SWAP_CHAIN_DESC sd;
	ZeroMemory(&sd, sizeof(sd));
	sd.BufferCount = 2;
	sd.BufferDesc.Width = 0;
	sd.BufferDesc.Height = 0;
	sd.BufferDesc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
	sd.BufferDesc.RefreshRate.Numerator = 60;
	sd.BufferDesc.RefreshRate.Denominator = 1;
	sd.Flags = DXGI_SWAP_CHAIN_FLAG_ALLOW_MODE_SWITCH;
	sd.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
	sd.OutputWindow = hWnd;
	sd.SampleDesc.Count = 1;
	sd.SampleDesc.Quality = 0;
	sd.Windowed = TRUE;
	sd.SwapEffect = DXGI_SWAP_EFFECT_DISCARD;

	UINT createDeviceFlags = 0;
	//createDeviceFlags |= D3D11_CREATE_DEVICE_DEBUG;
	D3D_FEATURE_LEVEL featureLevel;
	const D3D_FEATURE_LEVEL featureLevelArray[2] = { D3D_FEATURE_LEVEL_11_0, D3D_FEATURE_LEVEL_10_0, };
	if (D3D11CreateDeviceAndSwapChain(NULL, D3D_DRIVER_TYPE_HARDWARE, NULL, createDeviceFlags, featureLevelArray, 2, D3D11_SDK_VERSION, &sd, &g_pSwapChain, &g_pd3dDevice, &featureLevel, &g_pd3dDeviceContext) != S_OK)
		return E_FAIL;

	//CreateRenderTarget();
	if (CreateD3D11Views() != S_OK)
		std::cout << "Something went wrong!" << std::endl;

	return S_OK;
}

void CleanupDeviceD3D()
{
	g_flowVisTool.OnD3D11DestroyDevice();

	CleanupRenderTarget();
	if (g_pSwapChain) { g_pSwapChain->Release(); g_pSwapChain = NULL; }
	if (g_pd3dDeviceContext) { g_pd3dDeviceContext->Release(); g_pd3dDeviceContext = NULL; }
	if (g_pd3dDevice) { g_pd3dDevice->Release(); g_pd3dDevice = NULL; }
}

extern LRESULT ImGui_ImplWin32_WndProcHandler(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam);
LRESULT WINAPI WndProc(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam)
{
	if (ImGui_ImplWin32_WndProcHandler(hWnd, msg, wParam, lParam))
		return true;

	switch (msg)
	{
	case WM_SIZE:
		if (g_pd3dDevice != NULL && wParam != SIZE_MINIMIZED)
		{
			ImGui_ImplDX11_InvalidateDeviceObjects();
			CleanupRenderTarget();
			g_pSwapChain->ResizeBuffers(0, (UINT)LOWORD(lParam), (UINT)HIWORD(lParam), DXGI_FORMAT_UNKNOWN, 0);
			//CreateRenderTarget();
			if (CreateD3D11Views() != S_OK)
				std::cout << "Something went wrong!" << std::endl;
			ImGui_ImplDX11_CreateDeviceObjects();
		}
		return 0;
	case WM_SYSCOMMAND:
		if ((wParam & 0xfff0) == SC_KEYMENU) // Disable ALT application menu
			return 0;
		break;
	case WM_DESTROY:
		PostQuitMessage(0);
		return 0;
	case WM_DPICHANGED:
		if (ImGui::GetIO().ConfigFlags & ImGuiConfigFlags_DpiEnableScaleViewports)
		{
			//const int dpi = HIWORD(wParam);
			//printf("WM_DPICHANGED to %d (%.0f%%)\n", dpi, (float)dpi / 96.0f * 100.0f);
			const RECT* suggested_rect = (RECT*)lParam;
			::SetWindowPos(hWnd, NULL, suggested_rect->left, suggested_rect->top, suggested_rect->right - suggested_rect->left, suggested_rect->bottom - suggested_rect->top, SWP_NOZORDER | SWP_NOACTIVATE);
		}
		break;
	}
	return DefWindowProc(hWnd, msg, wParam, lParam);
}

void ExitApp()
{
	g_flowVisTool.CloseVolumeFile();
	ClearCudaDevices();
	ImGuiCleanup();

	CleanupDeviceD3D();
	DestroyWindow(g_hwnd);
	UnregisterClass(_T("FlowVisTool"), g_wc.hInstance);
}

void DockSpace()
{
	static bool opt_fullscreen_persistant = true;
	static ImGuiDockNodeFlags opt_flags = ImGuiDockNodeFlags_None;
	bool opt_fullscreen = opt_fullscreen_persistant;

	// We are using the ImGuiWindowFlags_NoDocking flag to make the parent window not dockable into,
	// because it would be confusing to have two docking targets within each others.
	ImGuiWindowFlags window_flags = ImGuiWindowFlags_MenuBar | ImGuiWindowFlags_NoDocking;
	if (opt_fullscreen)
	{
		ImGuiViewport* viewport = ImGui::GetMainViewport();
		ImGui::SetNextWindowPos(viewport->Pos);
		ImGui::SetNextWindowSize(viewport->Size);
		ImGui::SetNextWindowViewport(viewport->ID);
		ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 0.0f);
		ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 0.0f);
		window_flags |= ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove;
		window_flags |= ImGuiWindowFlags_NoBringToFrontOnFocus | ImGuiWindowFlags_NoNavFocus;
	}

	// When using ImGuiDockNodeFlags_RenderWindowBg or ImGuiDockNodeFlags_InvisibleDockspace, DockSpace() will render our background and handle the pass-thru hole, so we ask Begin() to not render a background.
	if (opt_flags & ImGuiDockNodeFlags_RenderWindowBg)
		ImGui::SetNextWindowBgAlpha(0.0f);

	ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0.0f, 0.0f));
	ImGui::Begin("DockSpace Demo", nullptr, window_flags);
	ImGui::PopStyleVar();

	if (opt_fullscreen)
		ImGui::PopStyleVar(2);

	// Dockspace
	ImGuiIO& io = ImGui::GetIO();
	if (io.ConfigFlags & ImGuiConfigFlags_DockingEnable)
	{
		ImGuiID dockspace_id = ImGui::GetID("MyDockspace");
		ImGui::DockSpace(dockspace_id, ImVec2(0.0f, 0.0f), opt_flags);
	}
	//else
	//{
	//	ShowDockingDisabledMessage();
	//}
	//
	//if (ImGui::BeginMenuBar())
	//{
	//	if (ImGui::BeginMenu("Docking"))
	//	{
	//		// Disabling fullscreen would allow the window to be moved to the front of other windows, 
	//		// which we can't undo at the moment without finer window depth/z control.
	//		//ImGui::MenuItem("Fullscreen", NULL, &opt_fullscreen_persistant);
	//		if (ImGui::MenuItem("Flag: NoSplit", "", (opt_flags & ImGuiDockNodeFlags_NoSplit) != 0))                opt_flags ^= ImGuiDockNodeFlags_NoSplit;
	//		if (ImGui::MenuItem("Flag: NoDockingInCentralNode", "", (opt_flags & ImGuiDockNodeFlags_NoDockingInCentralNode) != 0)) opt_flags ^= ImGuiDockNodeFlags_NoDockingInCentralNode;
	//		if (ImGui::MenuItem("Flag: PassthruInEmptyNodes", "", (opt_flags & ImGuiDockNodeFlags_PassthruInEmptyNodes) != 0))   opt_flags ^= ImGuiDockNodeFlags_PassthruInEmptyNodes;
	//		if (ImGui::MenuItem("Flag: RenderWindowBg", "", (opt_flags & ImGuiDockNodeFlags_RenderWindowBg) != 0))         opt_flags ^= ImGuiDockNodeFlags_RenderWindowBg;
	//		if (ImGui::MenuItem("Flag: PassthruDockspace (all 3 above)", "", (opt_flags & ImGuiDockNodeFlags_PassthruDockspace) == ImGuiDockNodeFlags_PassthruDockspace))
	//			opt_flags = (opt_flags & ~ImGuiDockNodeFlags_PassthruDockspace) | ((opt_flags & ImGuiDockNodeFlags_PassthruDockspace) == ImGuiDockNodeFlags_PassthruDockspace) ? 0 : ImGuiDockNodeFlags_PassthruDockspace;
	//		ImGui::Separator();
	//		ImGui::EndMenu();
	//	}
	//	ImGui::EndMenuBar();
	//}

	ImGui::End();
}



bool g_showRenderingOptionsWindow = true;
bool g_showTracingOptionsWindow = true;
bool g_showFTLEWindow = false;
bool g_showHeatmapWindow = false;
bool g_showExtraWindow = false;
bool g_showDatasetWindow = true;



void MainMenu()
{
	if (ImGui::BeginMainMenuBar())
	{
		if (ImGui::BeginMenu("File"))
		{
			if (ImGui::MenuItem("Load dataset", "CTRL+L"))
			{
				std::string filename;
				if (tum3d::GetFilenameDialog("Select TimeVolume file", "TimeVolume (*.timevol)\0*.timevol\0", filename, false))
				{
					g_flowVisTool.CloseVolumeFile();
					g_flowVisTool.OpenVolumeFile(filename);
				}
			}
			ImGui::Separator();
			if (ImGui::MenuItem("Save", "CTRL+S")) {}
			ImGui::EndMenu();
		}
		if (ImGui::BeginMenu("View"))
		{
			if (ImGui::MenuItem("Rendering Options", nullptr, g_showRenderingOptionsWindow))
				g_showRenderingOptionsWindow = !g_showRenderingOptionsWindow;
			if (ImGui::MenuItem("Tracing Options", nullptr, g_showTracingOptionsWindow))
				g_showTracingOptionsWindow = !g_showTracingOptionsWindow;
			if (ImGui::MenuItem("Dataset", nullptr, g_showDatasetWindow))
				g_showDatasetWindow = !g_showDatasetWindow;
			if (ImGui::MenuItem("FTLE", nullptr, g_showFTLEWindow))
				g_showFTLEWindow = !g_showFTLEWindow;
			if (ImGui::MenuItem("Extra", nullptr, g_showExtraWindow))
				g_showExtraWindow = !g_showExtraWindow;
			if (ImGui::MenuItem("Heatmap", nullptr, g_showHeatmapWindow))
				g_showHeatmapWindow = !g_showHeatmapWindow;

			ImGui::EndMenu();
		}
		ImGui::EndMainMenuBar();
	}
}


//--------------------------------------------------------------------------------------
// Initialize everything and go into a render loop
//--------------------------------------------------------------------------------------
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

	std::cout.precision(3);
	std::cout << std::fixed;

	// init default number of omp threads
	g_threadCount = omp_get_num_procs();
	omp_set_num_threads(g_threadCount);

	// Set general DXUT callbacks
	//DXUTSetCallbackMsgProc( MsgProc );
	// do *not* register OnKeyboard and OnMouse!
	//DXUTSetCallbackFrameMove( OnFrameMove );
	//DXUTSetCallbackDeviceChanging( ModifyDeviceSettings );
	//DXUTSetCallbackDeviceRemoved( OnDeviceRemoved );
	//// Set the D3D11 DXUT callbacks
	//DXUTSetCallbackD3D11DeviceAcceptable( IsD3D11DeviceAcceptable );
	//DXUTSetCallbackD3D11DeviceCreated( OnD3D11CreateDevice );
	//DXUTSetCallbackD3D11SwapChainResized( OnD3D11ResizedSwapChain );
	//DXUTSetCallbackD3D11FrameRender( OnD3D11FrameRender );
	//DXUTSetCallbackD3D11SwapChainReleasing( OnD3D11ReleasingSwapChain );
	//DXUTSetCallbackD3D11DeviceDestroyed( OnD3D11DestroyDevice );

	// Create application window
	g_wc = { sizeof(WNDCLASSEX), CS_CLASSDC, WndProc, 0L, 0L, GetModuleHandle(NULL), NULL, NULL, NULL, NULL, _T("FlowVisTool"), NULL };
	RegisterClassEx(&g_wc);
	g_hwnd = CreateWindow(_T("FlowVisTool"), _T("FlowVisTool"), WS_OVERLAPPEDWINDOW, 100, 100, 1280, 800, NULL, NULL, g_wc.hInstance, NULL);

	// Initialize Direct3D
	if (CreateDeviceD3D(g_hwnd) < 0)
	{
		CleanupDeviceD3D();
		UnregisterClass(_T("FlowVisTool"), g_wc.hInstance);
		return 1;
	}

	// Show the window
	ShowWindow(g_hwnd, SW_SHOWDEFAULT);
	UpdateWindow(g_hwnd);


	// Perform application-level initialization
	if(!InitApp())
		return EXIT_FAILURE;
	
	g_flowVisTool.g_cudaDevices = g_cudaDevices;

	//DXUTInit( true, true, NULL ); // Parse the command line, show msgboxes on error, no extra command line params
	//DXUTSetIsInGammaCorrectMode( false );
	//DXUTSetCursorSettings( true, true ); // Show the cursor and clip it when in full screen
	//DXUTCreateWindow( L"TurbulenceRenderer" );
	//DXUTCreateDevice( D3D_FEATURE_LEVEL_11_0, true, 1360, 1360 );


	OnD3D11CreateDevice();

	g_flowVisTool.SetDXStuff(g_pd3dDevice, g_pd3dDeviceContext, g_mainRenderTargetView, g_mainDepthStencilView);

	SetupImGui();

	//g_flowVisTool.OpenVolumeFile("C:\\Users\\ge25ben\\Data\\TimeVol\\avg-wsize-170-wbegin-001.timevol");
	//OpenVolumeFile("C:\\Users\\alexf\\Desktop\\pacificvis-stuff\\TimeVol\\turb-data.timevol", g_pd3dDevice);

	//if(argc > 1 && std::string(argv[1]) == "dumpla3d") {
	//	if(argc != 4) {
	//		printf("Usage: %s dumpla3d file.timevol outfile.la3d", argv[0]);
	//		return 1;
	//	}
	//	std::string filenameVol(argv[2]);
	//	std::string filenameLa3d(argv[3]);
	//	if(!OpenVolumeFile(filenameVol, DXUTGetD3D11Device())) {
	//		printf("Failed opening %s\n", filenameVol.c_str());
	//		return 2;
	//	}
	//	// create outpath if it doesn't exist yet
	//	system((string("mkdir \"") + tum3d::GetPath(filenameLa3d) + "\"").c_str());
	//	// remove extension
	//	if(filenameLa3d.substr(filenameLa3d.size() - 5) == ".la3d")
	//	{
	//		filenameLa3d = filenameLa3d.substr(0, filenameLa3d.size() - 5);
	//	}
	//	// build per-channel filenames
	//	std::vector<std::string> filenames;
	//	for(int c = 0; c < g_volume.GetChannelCount(); c++)
	//	{
	//		std::ostringstream str;
	//		str << filenameLa3d << char('X' + c) << ".la3d";
	//		filenames.push_back(str.str());
	//	}
	//	// and write la3d
	//	if(!g_renderingManager.WriteCurTimestepToLA3Ds(g_volume, filenames)) {
	//		printf("Failed writing la3ds\n");
	//		return 3;
	//	}
	//	return 0;
	//}
	//// parse cmdline
	//// usage: argv[0] [config.cfg [volume.timevol [batchoutprefix [heuristicflags [options...]]]]]
	//if(argc > 1)
	//{
	//	std::string configFile(argv[1]);
	//	if(!LoadRenderingParams(configFile))
	//	{
	//		printf("Failed loading config from %s\n", configFile.c_str());
	//		return 1;
	//	}
	//}
	//if(argc > 2)
	//{
	//	std::string volumeFile(argv[2]);
	//	g_batchTrace.VolumeFiles.push_back(volumeFile);
	//	g_batchTrace.OutPath = tum3d::GetPath(volumeFile);
	//	if(argc > 3)
	//	{
	//		g_batchTrace.OutPath += std::string(argv[3]);
	//	}
	//	if(argc > 4)
	//	{
	//		g_particleTraceParams.m_heuristicFlags = atoi(argv[4]);
	//	}
	//	for(int i = 5; i < argc; i++)
	//	{
	//		if(argv[i] == std::string("WriteLinebufs"))
	//		{
	//			g_batchTrace.WriteLinebufs = true;
	//		}
	//		//else if ... more options?
	//		else
	//		{
	//			printf("WARNING: unknown option \"%s\"\n", argv[i]);
	//		}
	//	}
	//	g_batchTrace.ExitAfterFinishing = true;
	//	//StartBatchTracing();
	//}

	//DXUTMainLoop(); // Enter into the DXUT render loop


	// Main loop
	bool resizeNextFrame = false;
	ImVec2 sceneWindowSize;
	bool show_demo_window = false;
	//bool show_another_window = false;
	//ImVec4 clear_color = ImVec4(0.45f, 0.55f, 0.60f, 1.00f);

	MSG msg;
	ZeroMemory(&msg, sizeof(msg));
	while (msg.message != WM_QUIT)
	{
		// Poll and handle messages (inputs, window resize, etc.)
		// You can read the io.WantCaptureMouse, io.WantCaptureKeyboard flags to tell if dear imgui wants to use your inputs.
		// - When io.WantCaptureMouse is true, do not dispatch mouse input data to your main application.
		// - When io.WantCaptureKeyboard is true, do not dispatch keyboard input data to your main application.
		// Generally you may always pass all inputs to dear imgui, and hide them from your application based on those two flags.
		if (PeekMessage(&msg, NULL, 0U, 0U, PM_REMOVE))
		{
			TranslateMessage(&msg);
			DispatchMessage(&msg);
			continue;
		}

		// Start the Dear ImGui frame
		ImGui_ImplDX11_NewFrame();
		ImGui_ImplWin32_NewFrame();
		ImGui::NewFrame();

		if (resizeNextFrame)
		{
			resizeNextFrame = false;
			g_flowVisTool.OnD3D11ResizedSwapChain(sceneWindowSize.x, sceneWindowSize.y);
		}

		DockSpace();

		static float time = 0.0f;
		time += ImGui::GetIO().DeltaTime;

		{
			OnFrameMove(time, ImGui::GetIO().DeltaTime, nullptr);

			g_flowVisTool.g_renderTexture.SetRenderTarget(g_pd3dDeviceContext, g_mainDepthStencilView);

			// save old viewport
			UINT viewportCount = 1;
			D3D11_VIEWPORT viewportOld;
			g_pd3dDeviceContext->RSGetViewports(&viewportCount, &viewportOld);

			// set viewport for offscreen render buffer
			D3D11_VIEWPORT viewport;
			viewport.TopLeftX = 0;
			viewport.TopLeftY = 0;
			viewport.Width = (FLOAT)g_flowVisTool.g_windowSize.x();
			viewport.Height = (FLOAT)g_flowVisTool.g_windowSize.y();
			viewport.MinDepth = 0.0f;
			viewport.MaxDepth = 1.0f;
			g_pd3dDeviceContext->RSSetViewports(1, &viewport);

			g_flowVisTool.OnFrame(ImGui::GetIO().DeltaTime);

			// restore viewport
			g_pd3dDeviceContext->RSSetViewports(1, &viewportOld);
		}

		// Show the big demo window (Most of the sample code is in ImGui::ShowDemoWindow()! You can browse its code to learn more about Dear ImGui!).
		if (show_demo_window)
			ImGui::ShowDemoWindow(&show_demo_window);

		MainMenu();

		const float buttonWidth = 200;

		// Dataset window
		if (g_showDatasetWindow)
		{
			ImGui::SetNextWindowSize(ImVec2(300, 300), ImGuiCond_FirstUseEver);
			if (ImGui::Begin("Dataset", &g_showDatasetWindow))
			{
				ImGui::PushItemWidth(-150);
				{
					if (g_flowVisTool.g_volume.IsOpen())
					{
						ImGui::Spacing();
						ImGui::Separator();

						if (ImGui::Button("Preload nearest timestep", ImVec2(buttonWidth, 0)))
						{
							std::cout << "Loading timestep...";
							TimerCPU timer;
							timer.Start();
							g_flowVisTool.g_volume.LoadNearestTimestep();
							timer.Stop();
							std::cout << " done in " << timer.GetElapsedTimeMS() / 1000.0f << "s" << std::endl;
						}

						int32 timestepMax = g_flowVisTool.g_volume.GetTimestepCount() - 1;
						float timeSpacing = g_flowVisTool.g_volume.GetTimeSpacing();
						float timeMax = timeSpacing * float(timestepMax);

						float t = g_flowVisTool.g_volume.GetCurTime();
						if (ImGui::InputFloat("Time", &t, timeSpacing, timeSpacing * 2.0f, 0))
						{
							t = std::max(0.0f, std::min(t, timeMax));

							g_flowVisTool.g_volume.SetCurTime(t);
						}

						t = g_flowVisTool.g_volume.GetCurNearestTimestepIndex();
						if (ImGui::SliderFloat("Timestep", &t, 0.0f, timestepMax, "%.0f"))
						{
							t = t * g_flowVisTool.g_volume.GetTimeSpacing();

							g_flowVisTool.g_volume.SetCurTime(t);
						}

						t = g_flowVisTool.g_volume.GetTimeSpacing();
						if (ImGui::DragFloat("Time spacing", &t, 0.05f, 0.05f, timeMax, "%.2f"))
						{
							t = std::max(0.05f, std::min(t, timeMax));

							g_flowVisTool.g_volume.SetTimeSpacing(t);
						}

						ImGui::Spacing();
						ImGui::Separator();

						if (ImGui::Button("Save as raw", ImVec2(buttonWidth, 0)))
						{
							std::string filename;
							if (tum3d::GetFilenameDialog("Select output file", "Raw (*.raw)\0*.raw\0", filename, true))
							{
								// remove extension
								if (filename.substr(filename.size() - 4) == ".raw")
									filename = filename.substr(0, filename.size() - 4);

								std::vector<std::string> filenames;
								for (int c = 0; c < g_flowVisTool.g_volume.GetChannelCount(); c++)
								{
									std::ostringstream str;
									str << filename << char('X' + c) << ".raw";
									filenames.push_back(str.str());
								}

								g_flowVisTool.g_renderingManager.WriteCurTimestepToRaws(g_flowVisTool.g_volume, filenames);
							}
						}

						if (ImGui::Button("Save as la3d", ImVec2(buttonWidth, 0)))
						{
							std::string filename;
							if (tum3d::GetFilenameDialog("Select output file", "LargeArray3D (*.la3d)\0*.la3d\0", filename, true))
							{
								// remove extension
								if (filename.substr(filename.size() - 5) == ".la3d")
									filename = filename.substr(0, filename.size() - 5);

								std::vector<std::string> filenames;
								for (int c = 0; c < g_flowVisTool.g_volume.GetChannelCount(); c++)
								{
									std::ostringstream str;
									str << filename << char('X' + c) << ".la3d";
									filenames.push_back(str.str());
								}
								g_flowVisTool.g_renderingManager.WriteCurTimestepToLA3Ds(g_flowVisTool.g_volume, filenames);
							}
						}
					}
				}
				ImGui::PopItemWidth();
			}
			ImGui::End();
		}


		// Particle tracing config window
		if (g_showTracingOptionsWindow)
		{
			ImGui::SetNextWindowSize(ImVec2(300, 300), ImGuiCond_FirstUseEver);
			if (ImGui::Begin("Tracing Options", &g_showTracingOptionsWindow))
			{
				ImGui::PushItemWidth(-150);
				{
					ImGui::Checkbox("Verbose", &g_flowVisTool.g_tracingManager.GetVerbose());

					// Seeding options
					ImGui::Spacing();
					ImGui::Separator();
					{
						if (ImGui::Button("Load seed texture", ImVec2(buttonWidth, 0)))
							LoadSeedTexture();

						if (ImGui::Button("Set seed box to domain", ImVec2(buttonWidth, 0)))
							g_flowVisTool.SetBoundingBoxToDomainSize();

						ImGui::DragFloat3("Seed box min", (float*)&g_flowVisTool.g_particleTraceParams.m_seedBoxMin, 0.005f, 0.0f, 0.0f, "%.3f");
						ImGui::DragFloat3("Seed box size", (float*)&g_flowVisTool.g_particleTraceParams.m_seedBoxSize, 0.005f, 0.0f, 0.0f, "%.3f");

						static auto getterSeedingPattern = [](void* data, int idx, const char** out_str)
						{
							if (idx >= ParticleTraceParams::eSeedPattern::COUNT) return false;
							*out_str = ParticleTraceParams::GetSeedPatternName(ParticleTraceParams::eSeedPattern(idx));
							return true;
						};
						ImGui::Combo("Seeding pattern", (int*)&g_flowVisTool.g_particleTraceParams.m_seedPattern, getterSeedingPattern, nullptr, ParticleTraceParams::eSeedPattern::COUNT);
					}

					// Tracing
					ImGui::Spacing();
					ImGui::Separator();
					{
						static auto getterAdvectMode = [](void* data, int idx, const char** out_str)
						{
							if (idx >= ADVECT_MODE_COUNT) return false;
							*out_str = GetAdvectModeName(eAdvectMode(idx));
							return true;
						};

						ImGui::Combo("Advection", (int*)&g_flowVisTool.g_particleTraceParams.m_advectMode, getterAdvectMode, nullptr, ADVECT_MODE_COUNT);
						ImGui::Checkbox("Dense output", &g_flowVisTool.g_particleTraceParams.m_enableDenseOutput);

						static auto getterFilterMode = [](void* data, int idx, const char** out_str)
						{
							if (idx >= TEXTURE_FILTER_MODE_COUNT) return false;
							*out_str = GetTextureFilterModeName(eTextureFilterMode(idx));
							return true;
						};
						ImGui::Combo("Interpolation", (int*)&g_flowVisTool.g_particleTraceParams.m_filterMode, getterFilterMode, nullptr, TEXTURE_FILTER_MODE_COUNT);

						static auto getterLineMode = [](void* data, int idx, const char** out_str)
						{
							if (idx >= LINE_MODE_COUNT) return false;
							*out_str = GetLineModeName(eLineMode(idx));
							return true;
						};
						ImGui::Combo("Line mode", (int*)&g_flowVisTool.g_particleTraceParams.m_lineMode, getterLineMode, nullptr, LINE_MODE_COUNT);

						if (ImGui::DragInt("Line count", &g_flowVisTool.g_particleTraceParams.m_lineCount, 1.0f, 1.0f, INT_MAX))
							g_flowVisTool.g_particleTraceParams.m_lineCount = std::max(1, g_flowVisTool.g_particleTraceParams.m_lineCount);
						if (ImGui::DragInt("Line max lenght", &g_flowVisTool.g_particleTraceParams.m_lineLengthMax, 1.0f, 2.0f, INT_MAX))
							g_flowVisTool.g_particleTraceParams.m_lineLengthMax = std::max(2, g_flowVisTool.g_particleTraceParams.m_lineLengthMax);
						if (ImGui::DragFloat("Line max age", &g_flowVisTool.g_particleTraceParams.m_lineAgeMax, 0.05f, 0.0f, FLT_MAX, "%.3f"))
							g_flowVisTool.g_particleTraceParams.m_lineAgeMax = std::max(0.0f, g_flowVisTool.g_particleTraceParams.m_lineAgeMax);

						ImGui::DragFloat("Min velocity", &g_flowVisTool.g_particleTraceParams.m_minVelocity, 0.01f, 0.0f, 0.0f, "%.2f");
						ImGui::DragFloat("Particles per second", &g_flowVisTool.g_particleTraceParams.m_particlesPerSecond, 0.01f, 0.0f, 0.0f, "%.2f");
						ImGui::DragFloat("Advection Delta T", &g_flowVisTool.g_particleTraceParams.m_advectDeltaT, 0.001f, 0.0f, 0.0f, "%.5f");
						if (ImGui::Button("Seed many particles", ImVec2(buttonWidth, 0)))
							g_flowVisTool.g_tracingManager.SeedManyParticles();
						ImGui::DragFloat("Cell Change Time Threshold", &g_flowVisTool.g_particleTraceParams.m_cellChangeThreshold, 0.001f, 0.0f, 0.0f, "%.5f");

						ImGui::Spacing();
						ImGui::Separator();

						if (ImGui::Button("Retrace", ImVec2(buttonWidth, 0)))
							g_flowVisTool.m_retrace = true;

						if (ImGui::Button(g_flowVisTool.g_particleTracingPaused ? "Continue" : "Pause", ImVec2(buttonWidth, 0)))
							g_flowVisTool.g_particleTracingPaused = !g_flowVisTool.g_particleTracingPaused;
					}

					ImGui::Spacing();
					ImGui::Separator();

					ImGui::Checkbox("CPU Tracing", &g_flowVisTool.g_particleTraceParams.m_cpuTracing);

					if (ImGui::SliderInt("# CPU Threads", &g_threadCount, 1, 16))
						omp_set_num_threads(g_threadCount);

					ImGui::Spacing();
					ImGui::Separator();

					int v = g_flowVisTool.g_tracingManager.GetBrickSlotCountMax();
					if (ImGui::DragInt("Max Brick Slot Count", &v, 1, 0, INT_MAX))
						g_flowVisTool.g_tracingManager.GetBrickSlotCountMax() = (unsigned int)std::max(0, v);

					v = g_flowVisTool.g_tracingManager.GetTimeSlotCountMax();
					if (ImGui::DragInt("Max Time Slot Count", &v, 1, 0, INT_MAX))
						g_flowVisTool.g_tracingManager.GetTimeSlotCountMax() = (unsigned int)std::max(0, v);

					ImGui::Spacing();
					ImGui::Separator();

					if (ImGui::DragFloat("Advection Error Tolerance (Voxels)", &g_flowVisTool.g_particleTraceParams.m_advectErrorTolerance, 0.001f, 0.0f, FLT_MAX, "%.5f"))
						g_flowVisTool.g_particleTraceParams.m_advectErrorTolerance = std::max(0.0f, g_flowVisTool.g_particleTraceParams.m_advectErrorTolerance);

					if (ImGui::DragFloat("Advection Delta T Min", &g_flowVisTool.g_particleTraceParams.m_advectDeltaTMin, 0.001f, 0.0f, FLT_MAX, "%.5f"))
						g_flowVisTool.g_particleTraceParams.m_advectDeltaTMin = std::max(0.0f, g_flowVisTool.g_particleTraceParams.m_advectDeltaTMin);

					if (ImGui::DragFloat("Advection Delta T Max", &g_flowVisTool.g_particleTraceParams.m_advectDeltaTMax, 0.001f, 0.0f, FLT_MAX, "%.5f"))
						g_flowVisTool.g_particleTraceParams.m_advectDeltaTMax = std::max(0.0f, g_flowVisTool.g_particleTraceParams.m_advectDeltaTMax);

					v = g_flowVisTool.g_particleTraceParams.m_advectStepsPerRound;
					if (ImGui::DragInt("Advect Steps per Round", &v, 1, 0, INT_MAX, "%d steps"))
						g_flowVisTool.g_particleTraceParams.m_advectStepsPerRound = (unsigned int)std::max(0, v);

					v = g_flowVisTool.g_particleTraceParams.m_purgeTimeoutInRounds;
					if (ImGui::DragInt("Brick Purge Timeout", &v, 1, 0, INT_MAX, "%d rounds"))
						g_flowVisTool.g_particleTraceParams.m_purgeTimeoutInRounds = (unsigned int)std::max(0, v);

					ImGui::Spacing();
					ImGui::Separator();

					ImGui::Text("Heuristic");

					if (ImGui::DragFloat("Bonus Factor", &g_flowVisTool.g_particleTraceParams.m_heuristicBonusFactor, 0.01f, 0.0f, FLT_MAX, "%.5f"))
						g_flowVisTool.g_particleTraceParams.m_heuristicBonusFactor = std::max(0.0f, g_flowVisTool.g_particleTraceParams.m_heuristicBonusFactor);

					if (ImGui::DragFloat("Penalty Factor", &g_flowVisTool.g_particleTraceParams.m_heuristicPenaltyFactor, 0.01f, 0.0f, FLT_MAX, "%.5f"))
						g_flowVisTool.g_particleTraceParams.m_heuristicPenaltyFactor = std::max(0.0f, g_flowVisTool.g_particleTraceParams.m_heuristicPenaltyFactor);

					// TODO: this should be a combo, no?
					v = g_flowVisTool.g_particleTraceParams.m_heuristicFlags;
					if (ImGui::DragInt("Flags", &v, 1, 0, INT_MAX))
						g_flowVisTool.g_particleTraceParams.m_heuristicFlags = (unsigned int)std::max(0, v);

					ImGui::Spacing();
					ImGui::Separator();

					if (ImGui::DragFloat("Output Pos Diff (Voxels)", &g_flowVisTool.g_particleTraceParams.m_outputPosDiff, 0.01f, 0.0f, FLT_MAX, "%.5f"))
						g_flowVisTool.g_particleTraceParams.m_outputPosDiff = std::max(0.0f, g_flowVisTool.g_particleTraceParams.m_outputPosDiff);

					if (ImGui::DragFloat("Output Time Diff", &g_flowVisTool.g_particleTraceParams.m_outputTimeDiff, 0.01f, 0.0f, FLT_MAX, "%.5f"))
						g_flowVisTool.g_particleTraceParams.m_outputTimeDiff = std::max(0.0f, g_flowVisTool.g_particleTraceParams.m_outputTimeDiff);

					ImGui::Checkbox("Wait for Disk", &g_flowVisTool.g_particleTraceParams.m_waitForDisk);
					ImGui::Checkbox("Prefetching", &g_flowVisTool.g_particleTraceParams.m_enablePrefetching);
					ImGui::Checkbox("Upsampled Volume Hack", &g_flowVisTool.g_particleTraceParams.m_upsampledVolumeHack);
				}
				ImGui::PopItemWidth();
			}
			ImGui::End();
		}


		// Extra window
		if (g_showExtraWindow)
		{
			ImGui::SetNextWindowSize(ImVec2(300, 300), ImGuiCond_FirstUseEver);
			if (ImGui::Begin("Extra", &g_showExtraWindow))
			{
				ImGui::PushItemWidth(-150);
				{
					float v = g_flowVisTool.g_volume.GetSystemMemoryUsage().GetSystemMemoryLimitMBytes();
					if (ImGui::DragFloat("Mem Usage Limit", &v, 10.0f, 0.0f, FLT_MAX, "%.1f MB"))
						g_flowVisTool.g_volume.GetSystemMemoryUsage().SetSystemMemoryLimitMBytes(std::max(0.0f, v));

					ImGui::Spacing();
					ImGui::Separator();


					if (ImGui::Button("Save Traced Lines", ImVec2(buttonWidth, 0)))
						SaveLinesDialog();
					if (ImGui::Button("Load Lines", ImVec2(buttonWidth, 0)))
						LoadLinesDialog();

					if (ImGui::DragInt("Line ID Override", &g_lineIDOverride, 1, -1, INT_MAX))
						g_lineIDOverride = std::max(-1, g_lineIDOverride);

					if (ImGui::Button("Clear Loaded Lines", ImVec2(buttonWidth, 0)))
						ClearLinesCallback();

					ImGui::Spacing();
					ImGui::Separator();

					if (ImGui::Button("Load Balls", ImVec2(buttonWidth, 0)))
						LoadBallsDialog();

					if (ImGui::DragFloat("Ball Radius", &g_flowVisTool.g_ballRadius, 0.001f, 0.0f, FLT_MAX, "%.3f"))
						g_flowVisTool.g_ballRadius = std::max(0.0f, g_flowVisTool.g_ballRadius);

					if (ImGui::Button("Clear Loaded Balls", ImVec2(buttonWidth, 0)))
						ClearBallsCallback();

					ImGui::Spacing();
					ImGui::Separator();

					if (ImGui::Button("Build Flow Graph", ImVec2(buttonWidth, 0)))
						BuildFlowGraphCallback();

					if (ImGui::Button("Save Flow Graph", ImVec2(buttonWidth, 0)))
						SaveFlowGraphCallback();

					if (ImGui::Button("Load Flow Graph", ImVec2(buttonWidth, 0)))
						LoadFlowGraphCallback();

					ImGui::Spacing();
					ImGui::Separator();

					if (ImGui::Button("Save Settings", ImVec2(buttonWidth, 0)))
						SaveRenderingParamsDialog();

					if (ImGui::Button("Load Settings", ImVec2(buttonWidth, 0)))
						LoadRenderingParamsDialog();

					ImGui::Spacing();
					ImGui::Separator();

					//if (ImGui::Button("Save Screenshot", ImVec2(buttonWidth, 0)))
					//	g_flowVisTool.g_saveScreenshot = true;

					//if (ImGui::Button("Save Renderbuffer", ImVec2(buttonWidth, 0)))
					//	g_flowVisTool.g_saveRenderBufferScreenshot = true;
				}
				ImGui::PopItemWidth();
			}
			ImGui::End();
		}

		// FTLE window
		if (g_showFTLEWindow)
		{
			ImGui::SetNextWindowSize(ImVec2(300, 300), ImGuiCond_FirstUseEver);
			if (ImGui::Begin("FTLE", &g_showFTLEWindow))
			{
				ImGui::PushItemWidth(-150);
				{
					static auto ftleComputeParticleCount = []()
					{
						return g_flowVisTool.g_particleTraceParams.m_lineCount = g_flowVisTool.g_particleTraceParams.m_ftleResolution * g_flowVisTool.g_particleTraceParams.m_ftleResolution * 6;
					};

					if (ImGui::Checkbox("Enabled", &g_flowVisTool.g_particleTraceParams.m_ftleEnabled))
					{
						if (g_flowVisTool.g_particleTraceParams.m_ftleEnabled)
						{
							//TwDefine("Main/LineCount readonly=true");
							//TwDefine("Main/LineMode readonly=true");
							//TwDefine("Main/LineLengthMax readonly=true");
							//TwDefine("Main/LineLengthMax readonly=true");
							//TwDefine("Main/SeedingPattern readonly=true"); 
							g_flowVisTool.g_particleTraceParams.m_lineMode = eLineMode::LINE_PATH_FTLE;
							g_flowVisTool.g_particleTraceParams.m_seedPattern = ParticleTraceParams::eSeedPattern::FTLE;
							g_flowVisTool.g_particleTraceParams.m_lineLengthMax = 2;
							g_flowVisTool.g_particleTraceParams.m_lineAgeMax = 0.1f;
							ftleComputeParticleCount();
						}
						else
						{
							g_flowVisTool.g_particleTraceParams.m_lineMode = eLineMode::LINE_STREAM;
							g_flowVisTool.g_particleTraceParams.m_seedPattern = ParticleTraceParams::eSeedPattern::RANDOM;
						}
					}

					ImGui::DragFloat("Scale", &g_flowVisTool.g_renderingManager.m_ftleScale, 0.001f);

					ImGui::Checkbox("Invert velocity", &g_flowVisTool.g_particleTraceParams.m_ftleInvertVelocity);

					int u = g_flowVisTool.g_particleTraceParams.m_ftleResolution;
					if (ImGui::DragInt("Resolution", &u, 1, 64, INT_MAX))
					{
						g_flowVisTool.g_particleTraceParams.m_ftleResolution = std::max(64, u);
						if (g_flowVisTool.g_particleTraceParams.m_ftleEnabled)
							ftleComputeParticleCount();
					}

					ImGui::DragFloat("Slice (Y)", &g_flowVisTool.g_particleTraceParams.m_ftleSliceY, 0.001f);

					ImGui::SliderFloat("Slice Alpha", &g_flowVisTool.g_particleRenderParams.m_ftleTextureAlpha, 0.0f, 1.0f);

					ImGui::DragFloat3("Separation Dist", (float*)&g_flowVisTool.g_particleTraceParams.m_ftleSeparationDistance, 0.0000001f, 0.0f, FLT_MAX, "%.7f");
				}
				ImGui::PopItemWidth();
			}
			ImGui::End();
		}

		// Heatmap window
		if (g_showHeatmapWindow)
		{
			ImGui::SetNextWindowSize(ImVec2(300, 300), ImGuiCond_FirstUseEver);
			if (ImGui::Begin("Heatmap", &g_showHeatmapWindow))
			{
				ImGui::PushItemWidth(-150);
				{
					ImGui::Checkbox("Enable Recording", &g_flowVisTool.g_heatMapParams.m_enableRecording);
					ImGui::Checkbox("Enable Rendering", &g_flowVisTool.g_heatMapParams.m_enableRendering);
					ImGui::Checkbox("Auto Reset", &g_flowVisTool.g_heatMapParams.m_autoReset);

					if (ImGui::Button("Reset", ImVec2(buttonWidth, 0)))
					{
						g_flowVisTool.g_heatMapManager.ClearChannels();
						g_flowVisTool.m_redraw = true;
					}

					static auto getterNormalizationMode = [](void* data, int idx, const char** out_str)
					{
						if (idx >= HEAT_MAP_NORMALIZATION_MODE_COUNT) return false;
						*out_str = GetHeatMapNormalizationModeName(eHeatMapNormalizationMode(idx));
						return true;
					};
					ImGui::Combo("Normalization", (int*)&g_flowVisTool.g_heatMapParams.m_normalizationMode, getterNormalizationMode, nullptr, HEAT_MAP_NORMALIZATION_MODE_COUNT);

					if (ImGui::DragFloat("Step Size", &g_flowVisTool.g_heatMapParams.m_stepSize, 0.001f, 0.001f, FLT_MAX))
						g_flowVisTool.g_heatMapParams.m_stepSize = std::max(0.001f, g_flowVisTool.g_heatMapParams.m_stepSize);

					if (ImGui::DragFloat("Density Scale", &g_flowVisTool.g_heatMapParams.m_densityScale, 0.01f, 0.0f, FLT_MAX))
						g_flowVisTool.g_heatMapParams.m_densityScale = std::max(0.0f, g_flowVisTool.g_heatMapParams.m_densityScale);

					ImGui::ColorEdit3("First displayed channel (1)", (float*)&g_flowVisTool.g_heatMapParams.m_renderedChannels[0]);
					ImGui::ColorEdit3("Second displayed channel (2)", (float*)&g_flowVisTool.g_heatMapParams.m_renderedChannels[1]);

					ImGui::Checkbox("Isosurface Rendering", &g_flowVisTool.g_heatMapParams.m_isosurface);

					ImGui::SliderFloat("Isovalue", &g_flowVisTool.g_heatMapParams.m_isovalue, 0.0f, 1.0f);
				}
				ImGui::PopItemWidth();
			}
			ImGui::End();
		}

		// Rendering config window
		if (g_showRenderingOptionsWindow)
		{
			ImGui::SetNextWindowSize(ImVec2(300, 300), ImGuiCond_FirstUseEver);
			if (ImGui::Begin("Rendering Options", &g_showRenderingOptionsWindow))
			{
				ImGui::PushItemWidth(-150);
				{
					if (ImGui::Button("Redraw", ImVec2(buttonWidth, 0)))
						g_flowVisTool.m_redraw = true;

					ImGui::Checkbox("Rendering Preview", &g_flowVisTool.g_showPreview);

					if (ImGui::ColorEdit3("Background color", (float*)&g_flowVisTool.g_backgroundColor))
						g_flowVisTool.m_redraw = true;

					ImGui::Checkbox("Show Seed Box", &g_flowVisTool.g_bRenderSeedBox);
					ImGui::Checkbox("Show Domain Box", &g_flowVisTool.g_bRenderDomainBox);

					if (ImGui::DragFloat("Domain Box Thickness", &g_flowVisTool.g_renderingManager.m_DomainBoxThickness, 0.0001f, 0.0f, FLT_MAX, "%.4f"))
					{
						g_flowVisTool.g_renderingManager.m_DomainBoxThickness = std::max(0.0f, g_flowVisTool.g_renderingManager.m_DomainBoxThickness);
						g_flowVisTool.m_redraw = true;
					}

					ImGui::Checkbox("Show Brick Boxes", &g_flowVisTool.g_bRenderBrickBoxes);

					ImGui::Spacing();
					ImGui::Separator();

					ImGui::Checkbox("Fixed Light Dir", &g_flowVisTool.g_particleRenderParams.m_FixedLightDir);

					ImGui::SliderFloat3("Light Dir", (float*)&g_flowVisTool.g_particleRenderParams.m_lightDir, -1.0f, 1.0f);

					int f = g_flowVisTool.g_renderBufferSizeFactor;
					if (ImGui::SliderInt("SuperSample Factor", &f, 1.0f, 8.0f))
						g_flowVisTool.g_renderBufferSizeFactor = f;

					if (ImGui::Checkbox("Perspective", &g_flowVisTool.g_projParams.m_perspective))
					{
						if (g_flowVisTool.g_projParams.m_perspective)
							g_flowVisTool.g_projParams.m_fovy = 30.0f * PI / 180.0f; // this should be 24 deg, but a bit larger fov looks better...
						else
							g_flowVisTool.g_projParams.m_fovy = 3.1f;
					}

					float degfovy = g_flowVisTool.g_projParams.m_fovy * 180.0f / PI;
					if (ImGui::SliderFloat("FoVY", &degfovy, 1.0f, 180.0f, "%.2f deg"))
						g_flowVisTool.g_projParams.m_fovy = degfovy * PI / 180.0f;

					ImGui::Checkbox("Stereo", &g_flowVisTool.g_stereoParams.m_stereoEnabled);

					if (ImGui::DragFloat("Eye Distance", &g_flowVisTool.g_stereoParams.m_eyeDistance, 0.001f, 0.0f, FLT_MAX, "%.3f"))
						g_flowVisTool.g_stereoParams.m_eyeDistance = std::max(0.0f, g_flowVisTool.g_stereoParams.m_eyeDistance);

					ImGui::Spacing();
					ImGui::Separator();

					ImGui::Checkbox("Particle rendering", &g_flowVisTool.g_particleRenderParams.m_linesEnabled);

					static auto getterLineRenderMode = [](void* data, int idx, const char** out_str)
					{
						if (idx >= LINE_RENDER_MODE_COUNT) return false;
						*out_str = GetLineRenderModeName(eLineRenderMode(idx));
						return true;
					};
					ImGui::Combo("Line render mode", (int*)&g_flowVisTool.g_particleRenderParams.m_lineRenderMode, getterLineRenderMode, nullptr, LINE_RENDER_MODE_COUNT);

					if (ImGui::DragFloat("Ribbon Width", &g_flowVisTool.g_particleRenderParams.m_ribbonWidth, 0.001f, 0.0f, FLT_MAX, "%.3f"))
						g_flowVisTool.g_particleRenderParams.m_ribbonWidth = std::max(0.0f, g_flowVisTool.g_particleRenderParams.m_ribbonWidth);

					if (ImGui::DragFloat("Tube Radius", &g_flowVisTool.g_particleRenderParams.m_tubeRadius, 0.001f, 0.0f, FLT_MAX, "%.3f"))
						g_flowVisTool.g_particleRenderParams.m_tubeRadius = std::max(0.0f, g_flowVisTool.g_particleRenderParams.m_tubeRadius);

					if (ImGui::DragFloat("Particle Size", &g_flowVisTool.g_particleRenderParams.m_particleSize, 0.001f, 0.0f, FLT_MAX, "%.3f"))
						g_flowVisTool.g_particleRenderParams.m_particleSize = std::max(0.0f, g_flowVisTool.g_particleRenderParams.m_particleSize);

					ImGui::Checkbox("Display Velocity", &g_flowVisTool.g_particleRenderParams.m_tubeRadiusFromVelocity);

					ImGui::DragFloat("Reference Velocity", &g_flowVisTool.g_particleRenderParams.m_referenceVelocity, 0.001f);

					ImGui::Spacing();
					ImGui::Separator();

					static auto getterParticleRenderMode = [](void* data, int idx, const char** out_str)
					{
						if (idx >= PARTICLE_RENDER_MODE_COUNT) return false;
						*out_str = GetParticleRenderModeName(eParticleRenderMode(idx));
						return true;
					};
					ImGui::Combo("Particle render mode", (int*)&g_flowVisTool.g_particleRenderParams.m_particleRenderMode, getterParticleRenderMode, nullptr, PARTICLE_RENDER_MODE_COUNT);

					if (ImGui::DragFloat("Particle Transparency", &g_flowVisTool.g_particleRenderParams.m_particleTransparency, 0.001f, 0.0f, 1.0f, "%.3f"))
						g_flowVisTool.g_particleRenderParams.m_particleTransparency = std::min(1.0f, std::max(0.0f, g_flowVisTool.g_particleRenderParams.m_particleTransparency));

					ImGui::Checkbox("Sort Particles", &g_flowVisTool.g_particleRenderParams.m_sortParticles);

					ImGui::Spacing();
					ImGui::Separator();


					static auto getterLineColorMode = [](void* data, int idx, const char** out_str)
					{
						if (idx >= LINE_COLOR_MODE_COUNT) return false;
						*out_str = GetLineColorModeName(eLineColorMode(idx));
						return true;
					};
					ImGui::Combo("Color Mode", (int*)&g_flowVisTool.g_particleRenderParams.m_lineColorMode, getterLineColorMode, nullptr, LINE_COLOR_MODE_COUNT);

					ImGui::ColorEdit3("Color 0", (float*)&g_flowVisTool.g_particleRenderParams.m_color0);
					ImGui::ColorEdit3("Color 1", (float*)&g_flowVisTool.g_particleRenderParams.m_color1);

					if (ImGui::Button("Load Color Texture", ImVec2(buttonWidth, 0)))
						LoadColorTexture();

					static auto getterMeasureMode = [](void* data, int idx, const char** out_str)
					{
						if (idx >= MEASURE_COUNT) return false;
						*out_str = GetMeasureName(eMeasure(idx));
						return true;
					};
					ImGui::Combo("Measure", (int*)&g_flowVisTool.g_particleRenderParams.m_measure, getterMeasureMode, nullptr, MEASURE_COUNT);

					if (ImGui::DragFloat("Measure scale", &g_flowVisTool.g_particleRenderParams.m_measureScale, 0.001f, 0.0f, 1.0f, "%.3f"))
						g_flowVisTool.g_particleRenderParams.m_measureScale = std::min(1.0f, std::max(0.0f, g_flowVisTool.g_particleRenderParams.m_measureScale));

					ImGui::Spacing();
					ImGui::Separator();

					ImGui::Checkbox("Time Stripes", &g_flowVisTool.g_particleRenderParams.m_timeStripes);

					if (ImGui::DragFloat("Time Stripe Length", &g_flowVisTool.g_particleRenderParams.m_timeStripeLength, 0.001f, 0.001f, FLT_MAX, "%.3f"))
						g_flowVisTool.g_particleRenderParams.m_timeStripeLength = std::max(0.001f, g_flowVisTool.g_particleRenderParams.m_timeStripeLength);

					ImGui::Spacing();
					ImGui::Separator();

					if (ImGui::Button("Load Slice Texture", ImVec2(buttonWidth, 0)))
						LoadSliceTexture();

					ImGui::Checkbox("Show Slice", &g_flowVisTool.g_particleRenderParams.m_showSlice);

					ImGui::DragFloat("Slice Position", &g_flowVisTool.g_particleRenderParams.m_slicePosition, 0.001f);

					if (ImGui::DragFloat("Slice Transparency", &g_flowVisTool.g_particleRenderParams.m_sliceAlpha, 0.001f, 0.0f, 1.0f, "%.3f"))
						g_flowVisTool.g_particleRenderParams.m_sliceAlpha = std::min(1.0f, std::max(0.0f, g_flowVisTool.g_particleRenderParams.m_sliceAlpha));
				}
				ImGui::PopItemWidth();
			}
			ImGui::End();
		}


		// Scene view window
		ImGui::SetNextWindowSize(ImVec2(400, 400), ImGuiCond_FirstUseEver);
		if (ImGui::Begin("Scene view"))
		{
			float height;
			float width;

			ImVec2 availableRegion = ImGui::GetContentRegionAvail();

			if (!ImGui::IsMouseDragging(0))
			{
				static ImVec2 lastFrameWindowSize;

				if (availableRegion.x != lastFrameWindowSize.x || availableRegion.y != lastFrameWindowSize.y)
				{
					sceneWindowSize = availableRegion;
					lastFrameWindowSize = availableRegion;
					resizeNextFrame = true;
				}
			}
			float windowAspectRatio = availableRegion.x / (float)availableRegion.y;

			if (windowAspectRatio < g_flowVisTool.g_renderTexture.GetAspectRatio())
			{
				width = availableRegion.x - 2;
				height = width * 1.0f / g_flowVisTool.g_renderTexture.GetAspectRatio();
				ImGui::SetCursorPosY(ImGui::GetCursorPos().y + (availableRegion.y - height) / 2.0f);
			}
			else
			{
				height = availableRegion.y - 2;
				width = height * g_flowVisTool.g_renderTexture.GetAspectRatio();
				ImGui::SetCursorPosX(ImGui::GetCursorPos().x + (availableRegion.x - width) / 2.0f);
			}


			static float orbitSens = 100.0f;
			static float panSens = 40.0f;
			static float zoomSens = 3.25f;

			ImGui::Begin("Debug");
			{
				ImGui::SliderFloat("Orbit Sens", &orbitSens, 0.0f, 1000.0f, "%.2f");
				ImGui::SliderFloat("Pan Sens", &panSens, 0.0f, 100.0f, "%.2f");
				ImGui::SliderFloat("Zoom Sens", &zoomSens, 0.0f, 100.0f, "%.2f");
				ImGui::DragFloat("View distance", &g_flowVisTool.g_viewParams.m_viewDistance);
			}
			ImGui::End();

			bool userInteraction = false;

			// ImageButton prevents mouse dragging from moving the window as well.
			ImGui::ImageButton((void *)(intptr_t)g_flowVisTool.g_renderTexture.GetShaderResourceView(), ImVec2(width, height), ImVec2(0, 0), ImVec2(1, 1), 0, ImColor(0, 0, 0, 255), ImColor(255, 255, 255, 255));
			//ImGui::Image((void *)(intptr_t)g_renderTexture.GetShaderResourceView(), ImVec2(width, height), ImVec2(0, 0), ImVec2(1, 1), ImColor(255, 255, 255, 255), ImColor(255, 255, 255, 25));
			
			if (ImGui::IsItemHovered(ImGuiHoveredFlags_::ImGuiHoveredFlags_None))
			//if (ImGui::IsWindowHovered(ImGuiHoveredFlags_::ImGuiHoveredFlags_None))
			{
				// Zoom
				g_flowVisTool.g_viewParams.m_viewDistance -= ImGui::GetIO().MouseWheel * ImGui::GetIO().DeltaTime * zoomSens * g_flowVisTool.g_viewParams.m_viewDistance;
				g_flowVisTool.g_viewParams.m_viewDistance = std::max(0.0f, g_flowVisTool.g_viewParams.m_viewDistance);

				// Orbit
				if (ImGui::IsMouseDragging(0))
				{
					if (ImGui::GetIO().MouseDelta.x != 0 || ImGui::GetIO().MouseDelta.y != 0)
					{
						userInteraction = true;

						tum3D::Vec2d normDelta = tum3D::Vec2d((double)ImGui::GetIO().MouseDelta.x / (double)g_flowVisTool.g_windowSize.x(), (double)ImGui::GetIO().MouseDelta.y / (double)g_flowVisTool.g_windowSize.y());

						tum3D::Vec2d delta = normDelta * (double)ImGui::GetIO().DeltaTime * (double)orbitSens;

						// Don't trust this code. Seriously, I have no idea how this is working.
						tum3D::Vec4f rotationX;

						tum3D::Vec3f up;
						tum3D::rotateVecByQuaternion(tum3D::Vec3f(0.0f, 0.0f, 1.0f), g_flowVisTool.g_viewParams.m_rotationQuat, up);
						up = tum3D::normalize(up);

						tum3D::rotationQuaternion((float)(up.y() < 0.0f ? -delta.x() : delta.x()) * PI, up, rotationX);
						//tum3D::rotationQuaternion(delta.x() * PI, Vec3f(0.0f, 0.0f, 1.0f), rotationX);

						tum3D::Vec4f rotation = tum3D::Vec4f(1, 0, 0, 0);

						tum3D::multQuaternion(rotationX, g_flowVisTool.g_viewParams.m_rotationQuat, rotation); g_flowVisTool.g_viewParams.m_rotationQuat = rotation;

						tum3D::Vec4f rotationY;
						tum3D::rotationQuaternion((float)delta.y() * PI, tum3D::Vec3f(1.0f, 0.0f, 0.0f), rotationY);

						tum3D::multQuaternion(rotationY, g_flowVisTool.g_viewParams.m_rotationQuat, rotation); g_flowVisTool.g_viewParams.m_rotationQuat = rotation;
					}	
				}

				// Pan on xy plane
				if (ImGui::IsMouseDragging(2))
				{
					if (ImGui::GetIO().MouseDelta.x != 0 || ImGui::GetIO().MouseDelta.y != 0)
					{
						userInteraction = true;
						
						tum3D::Vec2d normDelta = tum3D::Vec2d((double)ImGui::GetIO().MouseDelta.x / (double)g_flowVisTool.g_windowSize.x(), (double)ImGui::GetIO().MouseDelta.y / (double)g_flowVisTool.g_windowSize.y());

						tum3D::Vec2d delta = normDelta * (double)ImGui::GetIO().DeltaTime * (double)g_flowVisTool.g_viewParams.m_viewDistance * (double)panSens;

						tum3D::Vec2f target = g_flowVisTool.g_viewParams.m_lookAt.xy();

						tum3D::Vec2f right = g_flowVisTool.g_viewParams.GetRightVector().xy(); right = tum3D::normalize(right);
						target = target - right * delta.x();

						tum3D::Vec2f forward = g_flowVisTool.g_viewParams.GetViewDir().xy(); forward = tum3D::normalize(forward);

						if (forward.x() == 0.0f && forward.y() == 0.0f)
						{
							tum3D::Vec3f for3d;
							tum3D::crossProd(tum3D::Vec3f(0.0f, 0.0f, -g_flowVisTool.g_viewParams.GetViewDir().z()), tum3D::Vec3f(right.x(), right.y(), 0.0f), for3d);
							forward = for3d.xy();
						}

						target = target - forward * delta.y();

						g_flowVisTool.g_viewParams.m_lookAt.x() = target.x();
						g_flowVisTool.g_viewParams.m_lookAt.y() = target.y();
					}
				}
			}

			ImVec2 sceneViewPos = ImVec2(ImGui::GetWindowPos().x + ImGui::GetWindowContentRegionMin().x, ImGui::GetWindowPos().y + ImGui::GetWindowContentRegionMin().y);
			ImVec2 sceneViewSize = ImVec2(ImGui::GetWindowContentRegionMax().x - ImGui::GetWindowContentRegionMin().x, ImGui::GetWindowContentRegionMax().y - ImGui::GetWindowContentRegionMin().y);

			// Orientation overlay
			// FIXME-VIEWPORT-ABS: Select a default viewport
			const float DISTANCE = 10.0f;
			static int corner = 1;

			ImVec2 window_pos = ImVec2((corner & 1) ? (sceneViewPos.x + sceneViewSize.x - DISTANCE) : (sceneViewPos.x + DISTANCE), (corner & 2) ? (sceneViewPos.y + sceneViewSize.y - DISTANCE) : (sceneViewPos.y + DISTANCE));
			ImVec2 window_pos_pivot = ImVec2((corner & 1) ? 1.0f : 0.0f, (corner & 2) ? 1.0f : 0.0f);

			ImGui::SetNextWindowPos(window_pos, ImGuiCond_Always, window_pos_pivot);
			ImGui::SetNextWindowViewport(ImGui::GetWindowViewport()->ID);

			static bool p_open = true;

			ImGui::SetNextWindowBgAlpha(0.1f); // Transparent background
			if (ImGui::Begin("Example: Simple Overlay", &p_open, ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoDocking | ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_AlwaysAutoResize | ImGuiWindowFlags_NoSavedSettings | ImGuiWindowFlags_NoFocusOnAppearing | ImGuiWindowFlags_NoNav))
			{
				static auto makeRotationFromDir = [](tum3D::Vec3f direction)
				{
					tum3D::Mat3f mat;

					tum3D::Vec3f up(0.0f, 0.0f, 1.0f);

					tum3D::Vec3f xaxis;
					tum3D::crossProd(up, direction, xaxis);
					xaxis = tum3D::normalize(xaxis);

					tum3D::Vec3f yaxis;
					tum3D::crossProd(direction, xaxis, yaxis);
					yaxis = tum3D::normalize(yaxis);

					mat.get(0, 0) = xaxis.x();
					mat.get(1, 0) = yaxis.x();
					mat.get(2, 0) = direction.x();

					mat.get(0, 1) = xaxis.y();
					mat.get(1, 1) = yaxis.y();
					mat.get(2, 1) = direction.y();

					mat.get(0, 2) = xaxis.z();
					mat.get(1, 2) = yaxis.z();
					mat.get(2, 2) = direction.z();

					return mat;
				};

				tum3D::Vec3f dir = tum3D::Vec3f(0.0f, 0.0f, 0.0f);

				const float _epsilon = 0.001f;

				ImVec4 bColor = ImGui::GetStyle().Colors[ImGuiCol_::ImGuiCol_Button];
				ImVec4 tColor = ImGui::GetStyle().Colors[ImGuiCol_::ImGuiCol_Text];
				ImVec4 fColor = ImGui::GetStyle().Colors[ImGuiCol_::ImGuiCol_FrameBg];

				if (!ImGui::IsWindowHovered()) { bColor.w = 0.15f; tColor.w = 0.15f; fColor.w = 0.15f; }

				ImGui::PushStyleColor(ImGuiCol_::ImGuiCol_Button, bColor);
				ImGui::PushStyleColor(ImGuiCol_::ImGuiCol_Text, tColor);
				ImGui::PushStyleColor(ImGuiCol_::ImGuiCol_FrameBg, fColor);
				{
					ImVec2 bsize(60, 0);

					if (ImGui::Button("Top", bsize))
						dir = tum3D::Vec3f(_epsilon, 0.0f, 1.0f);
					ImGui::SameLine();
					if (ImGui::Button("Bottom", bsize))
						dir = tum3D::Vec3f(_epsilon, 0.0f, -1.0f);
					ImGui::SameLine();
					if (ImGui::Button("Right", bsize))
						dir = tum3D::Vec3f(-1.0f, 0.0f, _epsilon);

					if (ImGui::Button("Top0", bsize))
						dir = tum3D::Vec3f(0.5f, 0.5f, 0.5f);
					ImGui::SameLine();
					if (ImGui::Button("Bottom0", bsize))
						dir = tum3D::Vec3f(0.5f, 0.5f, -0.5f);
					ImGui::SameLine();
					if (ImGui::Button("Left", bsize))
						dir = tum3D::Vec3f(1.0f, 0.0f, _epsilon);

					if (ImGui::Button("Top1", bsize))
						dir = tum3D::Vec3f(0.5f, -0.5f, 0.5f);
					ImGui::SameLine();
					if (ImGui::Button("Bottom1", bsize))
						dir = tum3D::Vec3f(0.5f, -0.5f, -0.5f);
					ImGui::SameLine();
					if (ImGui::Button("Front", bsize))
						dir = tum3D::Vec3f(_epsilon, -1.0f, 0.0f);

					if (ImGui::Button("Top2", bsize))
						dir = tum3D::Vec3f(-0.5f, -0.5f, 0.5f);
					ImGui::SameLine();
					if (ImGui::Button("Bottom2", bsize))
						dir = tum3D::Vec3f(-0.5f, -0.5f, -0.5f);
					ImGui::SameLine();
					if (ImGui::Button("Back", bsize))
						dir = tum3D::Vec3f(_epsilon, 1.0f, 0.0f);

					if (ImGui::Button("Top3", bsize))
						dir = tum3D::Vec3f(-0.5f, 0.5f, 0.5f);
					ImGui::SameLine();
					if (ImGui::Button("Bottom3", bsize))
						dir = tum3D::Vec3f(-0.5f, 0.5f, -0.5f);


					ImGui::Separator();

					ImGui::PushItemWidth(100);
					ImGui::DragFloat3("Pivot", (float*)&g_flowVisTool.g_viewParams.m_lookAt, 0.01f, 0.0f, 0.0f, "%.2f");
					ImGui::PopItemWidth();
					ImGui::SameLine();
					if (ImGui::Button("Reset", ImVec2(-1, 0)))
						g_flowVisTool.g_viewParams.m_lookAt = tum3D::Vec3f(0.0f, 0.0f, 0.0f);
				}
				ImGui::PopStyleColor(3);

				static tum3D::Vec4f targetQuat;
				static bool interp = false;
				static float rotInterpSpeed = 5.0f;

				if (dir.normSqr() != 0.0f)
				{
					dir = tum3D::normalize(dir);
					tum3D::convertRotMatToQuaternion(makeRotationFromDir(dir), targetQuat);

					interp = true;
				}

				if (userInteraction) interp = false;
			
				if (interp)
				{
					tum3D::Vec4f res;
					g_flowVisTool.g_viewParams.m_rotationQuat = tum3D::slerpQuaternion(rotInterpSpeed * ImGui::GetIO().DeltaTime, g_flowVisTool.g_viewParams.m_rotationQuat, targetQuat, res);

					if (std::abs(tum3D::dotProd(targetQuat, g_flowVisTool.g_viewParams.m_rotationQuat)) > 1.0f - 0.000001f) // Epsilon
						interp = false;
				}
				
				if (ImGui::BeginPopupContextWindow())
				{
					if (ImGui::MenuItem("Top-left", NULL, corner == 0)) corner = 0;
					if (ImGui::MenuItem("Top-right", NULL, corner == 1)) corner = 1;
					if (ImGui::MenuItem("Bottom-left", NULL, corner == 2)) corner = 2;
					if (ImGui::MenuItem("Bottom-right", NULL, corner == 3)) corner = 3;
					ImGui::EndPopup();
				}
			}
			ImGui::End();
		}
		ImGui::End();




		ImGui::Begin("Debug");
		{
			static float sleepAmount = 0.0f;
			ImGui::SliderFloat("Thread sleep", &sleepAmount, 0.0f, 1000.0f);

			std::this_thread::sleep_for(std::chrono::milliseconds((long long)sleepAmount));
		}
		ImGui::End();


		// Rendering
		ImGui::Render();

		g_pd3dDeviceContext->OMSetRenderTargets(1, &g_mainRenderTargetView, g_mainDepthStencilView);

		//g_pd3dDeviceContext->OMSetRenderTargets(1, &g_mainRenderTargetView, NULL);
		//g_pd3dDeviceContext->ClearRenderTargetView(g_mainRenderTargetView, (float*)&clear_color);
		ImGui_ImplDX11_RenderDrawData(ImGui::GetDrawData());

		// Update and Render additional Platform Windows
		if (ImGui::GetIO().ConfigFlags & ImGuiConfigFlags_ViewportsEnable)
		{
			ImGui::UpdatePlatformWindows();
			ImGui::RenderPlatformWindowsDefault();
		}

		g_pSwapChain->Present(1, 0); // Present with vsync
		//g_pSwapChain->Present(0, 0); // Present without vsync
	}




	// Perform application-level cleanup
	ExitApp();

	//return DXUTGetExitCode();

	return 0;
}

#pragma endregion