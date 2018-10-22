#include <tchar.h>
#include <omp.h>

#include <imgui.h>
#include <imgui_impl_win32.h>
#include <imgui_impl_dx11.h>

#include <FlowVisTool.h>
#include <FlowVisToolGUI.h>


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

// the thread at g_primaryCudaDeviceIndex will not be started!
std::vector<MyCudaDevice> g_cudaDevices;
#pragma endregion


#pragma region ImGuiStuff
#ifndef WM_DPICHANGED
#define WM_DPICHANGED 0x02E0 // From Windows SDK 8.1+ headers
#endif

void ImGuiCustomTheme()
{
	ImVec4* colors = ImGui::GetStyle().Colors;
	colors[ImGuiCol_Text] = ImVec4(0.74f, 0.74f, 0.74f, 1.00f);
	colors[ImGuiCol_TextDisabled] = ImVec4(0.42f, 0.42f, 0.42f, 1.00f);
	colors[ImGuiCol_WindowBg] = ImVec4(0.18f, 0.18f, 0.18f, 1.00f);
	colors[ImGuiCol_ChildBg] = ImVec4(1.00f, 1.00f, 1.00f, 0.00f);
	colors[ImGuiCol_PopupBg] = ImVec4(0.08f, 0.08f, 0.08f, 0.94f);
	colors[ImGuiCol_Border] = ImVec4(0.14f, 0.14f, 0.14f, 1.00f);
	colors[ImGuiCol_BorderShadow] = ImVec4(0.00f, 0.00f, 0.00f, 0.00f);
	colors[ImGuiCol_FrameBg] = ImVec4(0.22f, 0.22f, 0.22f, 1.00f);
	colors[ImGuiCol_FrameBgHovered] = ImVec4(0.35f, 0.16f, 0.16f, 1.00f);
	colors[ImGuiCol_FrameBgActive] = ImVec4(0.55f, 0.25f, 0.25f, 1.00f);
	colors[ImGuiCol_TitleBg] = ImVec4(0.00f, 0.00f, 0.00f, 1.00f);
	colors[ImGuiCol_TitleBgActive] = ImVec4(0.06f, 0.07f, 0.08f, 1.00f);
	colors[ImGuiCol_TitleBgCollapsed] = ImVec4(0.00f, 0.00f, 0.00f, 0.51f);
	colors[ImGuiCol_MenuBarBg] = ImVec4(0.14f, 0.14f, 0.14f, 1.00f);
	colors[ImGuiCol_ScrollbarBg] = ImVec4(0.14f, 0.14f, 0.14f, 1.00f);
	colors[ImGuiCol_ScrollbarGrab] = ImVec4(0.31f, 0.31f, 0.31f, 1.00f);
	colors[ImGuiCol_ScrollbarGrabHovered] = ImVec4(0.41f, 0.41f, 0.41f, 1.00f);
	colors[ImGuiCol_ScrollbarGrabActive] = ImVec4(0.51f, 0.51f, 0.51f, 1.00f);
	colors[ImGuiCol_CheckMark] = ImVec4(0.72f, 0.33f, 0.33f, 0.69f);
	colors[ImGuiCol_SliderGrab] = ImVec4(0.72f, 0.33f, 0.33f, 0.69f);
	colors[ImGuiCol_SliderGrabActive] = ImVec4(0.82f, 0.38f, 0.38f, 1.00f);
	colors[ImGuiCol_Button] = ImVec4(0.24f, 0.27f, 0.29f, 1.00f);
	colors[ImGuiCol_ButtonHovered] = ImVec4(0.35f, 0.16f, 0.16f, 1.00f);
	colors[ImGuiCol_ButtonActive] = ImVec4(0.55f, 0.25f, 0.25f, 1.00f);
	colors[ImGuiCol_Header] = ImVec4(0.27f, 0.27f, 0.27f, 1.00f);
	colors[ImGuiCol_HeaderHovered] = ImVec4(0.36f, 0.36f, 0.36f, 1.00f);
	colors[ImGuiCol_HeaderActive] = ImVec4(0.21f, 0.21f, 0.21f, 1.00f);
	colors[ImGuiCol_Separator] = ImVec4(0.12f, 0.12f, 0.12f, 1.00f);
	colors[ImGuiCol_SeparatorHovered] = ImVec4(0.10f, 0.40f, 0.75f, 0.78f);
	colors[ImGuiCol_SeparatorActive] = ImVec4(0.10f, 0.40f, 0.75f, 1.00f);
	colors[ImGuiCol_ResizeGrip] = ImVec4(0.26f, 0.59f, 0.98f, 0.25f);
	colors[ImGuiCol_ResizeGripHovered] = ImVec4(0.26f, 0.59f, 0.98f, 0.67f);
	colors[ImGuiCol_ResizeGripActive] = ImVec4(0.26f, 0.59f, 0.98f, 0.95f);
	colors[ImGuiCol_Tab] = ImVec4(0.11f, 0.11f, 0.11f, 1.00f);
	colors[ImGuiCol_TabHovered] = ImVec4(0.33f, 0.33f, 0.33f, 1.00f);
	colors[ImGuiCol_TabActive] = ImVec4(0.18f, 0.18f, 0.18f, 1.00f);
	colors[ImGuiCol_TabUnfocused] = ImVec4(0.11f, 0.11f, 0.11f, 1.00f);
	colors[ImGuiCol_TabUnfocusedActive] = ImVec4(0.18f, 0.18f, 0.18f, 1.00f);
	colors[ImGuiCol_DockingPreview] = ImVec4(0.26f, 0.59f, 0.98f, 0.70f);
	colors[ImGuiCol_DockingEmptyBg] = ImVec4(0.00f, 0.00f, 0.00f, 1.00f);
	colors[ImGuiCol_PlotLines] = ImVec4(0.61f, 0.61f, 0.61f, 1.00f);
	colors[ImGuiCol_PlotLinesHovered] = ImVec4(1.00f, 0.43f, 0.35f, 1.00f);
	colors[ImGuiCol_PlotHistogram] = ImVec4(0.39f, 0.30f, 0.00f, 1.00f);
	colors[ImGuiCol_PlotHistogramHovered] = ImVec4(0.67f, 0.52f, 0.00f, 1.00f);
	colors[ImGuiCol_TextSelectedBg] = ImVec4(0.26f, 0.59f, 0.98f, 0.35f);
	colors[ImGuiCol_DragDropTarget] = ImVec4(1.00f, 1.00f, 0.00f, 0.90f);
	colors[ImGuiCol_NavHighlight] = ImVec4(0.26f, 0.59f, 0.98f, 1.00f);
	colors[ImGuiCol_NavWindowingHighlight] = ImVec4(1.00f, 1.00f, 1.00f, 0.70f);
	colors[ImGuiCol_NavWindowingDimBg] = ImVec4(0.80f, 0.80f, 0.80f, 0.20f);
	colors[ImGuiCol_ModalWindowDimBg] = ImVec4(0.80f, 0.80f, 0.80f, 0.35f);
}

bool InitializeImGui()
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
	{
		std::cerr << "Failed to initialize 'ImGui_ImplWin32'" << std::endl;
		return false;
	}

	//ID3D11Device* device = DXUTGetD3D11Device();
	//ID3D11DeviceContext* deviceContext;

	//device->GetImmediateContext(&deviceContext);

	if (!ImGui_ImplDX11_Init(g_pd3dDevice, g_pd3dDeviceContext))
	{
		std::cerr << "Failed to initialize 'ImGui_ImplDX11'" << std::endl;
		return false;
	}

	//SAFE_RELEASE(deviceContext);

	// Setup style
	ImGui::GetStyle().TabRounding = 0.0f;
	ImGui::GetStyle().ScrollbarRounding = 0.0f;
	ImGui::GetStyle().WindowRounding = 0.0f;
	ImGui::StyleColorsDark();
	//ImGui::StyleColorsClassic();

	// Load Fonts
	// - If no fonts are loaded, dear imgui will use the default font. You can also load multiple fonts and use ImGui::PushFont()/PopFont() to select them. 
	// - AddFontFromFileTTF() will return the ImFont* so you can store it if you need to select the font among multiple. 
	// - If the file cannot be loaded, the function will return NULL. Please handle those errors in your application (e.g. use an assertion, or display an error and quit).
	// - The fonts will be rasterized at a given size (w/ oversampling) and stored into a texture when calling ImFontAtlas::Build()/GetTexDataAsXXXX(), which ImGui_ImplXXXX_NewFrame below will call.
	// - Read 'misc/fonts/README.txt' for more instructions and details.
	// - Remember that in C/C++ if you want to include a backslash \ in a string literal you need to write a double backslash \\ !
	//io.Fonts->AddFontDefault();

	if (tum3d::FileExists("./resources/Roboto-Medium.ttf"))
		io.Fonts->AddFontFromFileTTF("./resources/Roboto-Medium.ttf", 16.0f);
	else if (tum3d::FileExists("../resources/Roboto-Medium.ttf"))
		io.Fonts->AddFontFromFileTTF("../resources/Roboto-Medium.ttf", 16.0f);
	//io.Fonts->AddFontFromFileTTF("./resources/DroidSans.ttf", 16.0f);
	//io.Fonts->AddFontFromFileTTF("./resources/Cousine-Regular.ttf", 16.0f);

	//ImFont* font = io.Fonts->AddFontFromFileTTF("c:\\Windows\\Fonts\\ArialUni.ttf", 18.0f, NULL, io.Fonts->GetGlyphRangesJapanese());
	//IM_ASSERT(font != NULL);

	ImGuiCustomTheme();

	return true;
}

void ImGuiCleanup()
{
	ImGui_ImplDX11_Shutdown();
	ImGui_ImplWin32_Shutdown();
	ImGui::DestroyContext();
}
#pragma endregion


#pragma region CUDA
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
#pragma endregion


#pragma region DX
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

	//g_flowVisTool.m_mainRenderTargetView = g_mainRenderTargetView;

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
	CleanupRenderTarget();
	if (g_pSwapChain) { g_pSwapChain->Release(); g_pSwapChain = NULL; }
	if (g_pd3dDeviceContext) { g_pd3dDeviceContext->Release(); g_pd3dDeviceContext = NULL; }
	if (g_pd3dDevice) { g_pd3dDevice->Release(); g_pd3dDevice = NULL; }
}

bool InitializeDirect3D()
{
	// Initialize Direct3D
	if (CreateDeviceD3D(g_hwnd) < 0)
	{
		//CleanupDeviceD3D();
		//UnregisterClass(_T("FlowVisTool"), g_wc.hInstance);
		return false;
	}
	return true;
}
#pragma endregion


#pragma region WindowStuff
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

bool CreateApplicationWindow()
{
	// Create application window
	g_wc = { sizeof(WNDCLASSEX), CS_CLASSDC, WndProc, 0L, 0L, GetModuleHandle(NULL), NULL, NULL, NULL, NULL, _T("FlowVisTool"), NULL };
	RegisterClassEx(&g_wc);
	g_hwnd = CreateWindow(_T("FlowVisTool"), _T("FlowVisTool"), WS_OVERLAPPEDWINDOW, 100, 100, 1360, 768, NULL, NULL, g_wc.hInstance, NULL);

	// Show the window
	//ShowWindow(g_hwnd, SW_SHOWDEFAULT);
	ShowWindow(g_hwnd, SW_SHOWMAXIMIZED);
	UpdateWindow(g_hwnd);

	return true;
}
#pragma endregion


#pragma region Main
bool InitializeFlowVisTool()
{
	if (!g_flowVisTool.Initialize(g_pd3dDevice, g_pd3dDeviceContext, g_cudaDevices))
		return false;

	//g_flowVisTool.OpenVolumeFile("C:\\Users\\ge25ben\\Data\\TimeVol\\avg-wsize-170-wbegin-001.timevol");
	//g_flowVisTool.OpenVolumeFile("C:\\Users\\alexf\\Desktop\\pacificvis-stuff\\TimeVol\\turb-data.timevol", g_pd3dDevice);

	// init default number of omp threads
	omp_set_num_threads(omp_get_num_procs());

	std::cout.precision(3);
	std::cout << std::fixed;

	return true;
}

void MainLoop()
{
	bool resizeNextFrame = false;
	ImVec2 sceneWindowSize;

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

		// Start the Dear ImGui frame.
		ImGui_ImplDX11_NewFrame();
		ImGui_ImplWin32_NewFrame();
		ImGui::NewFrame();

		FlowVisToolGUI::DockSpace();

		if (resizeNextFrame)
		{
			resizeNextFrame = false;
			g_flowVisTool.ResizeViewport(sceneWindowSize.x, sceneWindowSize.y);
		}

		{
			//g_flowVisTool.g_renderTexture.SetRenderTarget(g_pd3dDeviceContext, g_mainDepthStencilView);

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

		FlowVisToolGUI::RenderGUI(g_flowVisTool, resizeNextFrame, sceneWindowSize);

		// Set and clear main render target.
		{
			g_pd3dDeviceContext->OMSetRenderTargets(1, &g_mainRenderTargetView, g_mainDepthStencilView);

			//g_pd3dDeviceContext->OMSetRenderTargets(1, &g_mainRenderTargetView, NULL);
			//g_pd3dDeviceContext->ClearRenderTargetView(g_mainRenderTargetView, (float*)&clear_color);

			static float clearColor[4] = { 154.0f / 255.0f, 72.0f / 255.0f, 162.0f / 255.0f, 1.0f };

			ImGui::Begin("Debug");
			ImGui::ColorEdit4("Clear color", clearColor);
			ImGui::End();

			g_pd3dDeviceContext->ClearRenderTargetView(g_mainRenderTargetView, (float*)&clearColor);
			g_pd3dDeviceContext->ClearDepthStencilView(g_mainDepthStencilView, D3D11_CLEAR_DEPTH, 1.0f, 0);
		}

		// Render GUI.
		{
			ImGui::Render();
			ImGui_ImplDX11_RenderDrawData(ImGui::GetDrawData());

			// Update and Render additional Platform Windows
			if (ImGui::GetIO().ConfigFlags & ImGuiConfigFlags_ViewportsEnable)
			{
				ImGui::UpdatePlatformWindows();
				ImGui::RenderPlatformWindowsDefault();
			}
		}

		g_pSwapChain->Present(1, 0); // Present with vsync
		//g_pSwapChain->Present(0, 0); // Present without vsync
	}
}

void ExitApplication()
{
	g_flowVisTool.Release();
	ClearCudaDevices();
	ImGuiCleanup();

	CleanupDeviceD3D();
	DestroyWindow(g_hwnd);
	UnregisterClass(_T("FlowVisTool"), g_wc.hInstance);
}

int main(int argc, char* argv[])
{
	// Enable run-time memory check for debug builds.
#if defined(DEBUG) | defined(_DEBUG)
	_CrtSetDbgFlag( _CRTDBG_ALLOC_MEM_DF | _CRTDBG_LEAK_CHECK_DF );
	printf("---- WARNING: DEBUG BUILD - SHIT WILL BE SLOW! ----\n\n");
#endif

	if (!CreateApplicationWindow())
	{
		ExitApplication();
		return EXIT_FAILURE;
	}

	if (!InitializeDirect3D())
	{
		ExitApplication();
		return EXIT_FAILURE;
	}
	
	if (!InitializeImGui())
	{
		ExitApplication();
		return EXIT_FAILURE;
	}

	if (!GetCudaDevices())
	{
		ExitApplication();
		return EXIT_FAILURE;
	}

	if (!InitializeFlowVisTool())
	{
		ExitApplication();
		return EXIT_FAILURE;
	}

	MainLoop();

	ExitApplication();

	return 0;
}
#pragma endregion