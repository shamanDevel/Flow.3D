#include <FlowVisToolGUI.h>


#include <Vec.h>
#include <WICTextureLoader.h>
#include <omp.h>
#include <fstream>
#include <vector>


bool FlowVisToolGUI::g_showRenderingOptionsWindow = true;
bool FlowVisToolGUI::g_showTracingOptionsWindow = true;
bool FlowVisToolGUI::g_showFTLEWindow = false;
bool FlowVisToolGUI::g_showHeatmapWindow = false;
bool FlowVisToolGUI::g_showExtraWindow = false;
bool FlowVisToolGUI::g_showDatasetWindow = true;
bool FlowVisToolGUI::g_show_demo_window = false; 
bool FlowVisToolGUI::g_showProfilerWindow = true;


const float FlowVisToolGUI::buttonWidth = 200;
int FlowVisToolGUI::g_threadCount = omp_get_num_procs();
int FlowVisToolGUI::g_lineIDOverride = -1;


const int max_frames = 150;
std::vector<float> frameTimes(max_frames);
int currentFrameTimeIndex = 0;


#pragma region Dialogs
void FlowVisToolGUI::SaveLinesDialog(FlowVisTool& g_flowVisTool)
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

void FlowVisToolGUI::LoadLinesDialog(FlowVisTool& g_flowVisTool)
{
	//bool bFullscreen = (!DXUTIsWindowed());

	//if( bFullscreen ) DXUTToggleFullScreen();

	std::string filename;
	if (tum3d::GetFilenameDialog("Load Lines", "*.linebuf\0*.linebuf", filename, false))
	{
		LineBuffers* pBuffers = new LineBuffers(g_flowVisTool.m_d3dDevice);
		if (!pBuffers->Read(filename, g_lineIDOverride))
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

void FlowVisToolGUI::LoadBallsDialog(FlowVisTool& g_flowVisTool)
{
	//bool bFullscreen = (!DXUTIsWindowed());

	//if( bFullscreen ) DXUTToggleFullScreen();

	std::string filename;
	if (tum3d::GetFilenameDialog("Load Balls", "*.*\0*.*", filename, false))
	{
		BallBuffers* pBuffers = new BallBuffers(g_flowVisTool.m_d3dDevice);
		if (!pBuffers->Read(filename))
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

void FlowVisToolGUI::SaveRenderingParamsDialog(FlowVisTool& g_flowVisTool)
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

void FlowVisToolGUI::LoadRenderingParamsDialog(FlowVisTool& g_flowVisTool)
{
	//bool bFullscreen = (!DXUTIsWindowed());

	//if( bFullscreen ) DXUTToggleFullScreen();

	//std::string filename;
	//if ( tum3d::GetFilenameDialog("Load Settings", "*.cfg\0*.cfg", filename, false) ) 
	//	LoadRenderingParams( filename );

	//if( bFullscreen ) DXUTToggleFullScreen();	
}

void FlowVisToolGUI::LoadSliceTexture(FlowVisTool& g_flowVisTool)
{
	std::string filename;
	if (tum3d::GetFilenameDialog("Load Texture", "Images (jpg, png, bmp)\0*.png;*.jpg;*.jpeg;*.bmp\0", filename, false)) {
		//release old texture
		SAFE_RELEASE(g_flowVisTool.g_particleRenderParams.m_pSliceTexture);
		//create new texture
		std::wstring wfilename(filename.begin(), filename.end());
		ID3D11Resource* tmp = NULL;
		if (!FAILED(DirectX::CreateWICTextureFromFile(g_flowVisTool.m_d3dDevice, wfilename.c_str(), &tmp, &g_flowVisTool.g_particleRenderParams.m_pSliceTexture))) {
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

void FlowVisToolGUI::LoadColorTexture(FlowVisTool& g_flowVisTool)
{
	std::string filename;
	if (tum3d::GetFilenameDialog("Load Texture", "Images (jpg, png, bmp)\0*.png;*.jpg;*.jpeg;*.bmp\0", filename, false)) {
		//release old texture
		SAFE_RELEASE(g_flowVisTool.g_particleRenderParams.m_pColorTexture);
		//create new texture
		std::wstring wfilename(filename.begin(), filename.end());
		ID3D11Resource* tmp = NULL;
		if (!FAILED(DirectX::CreateWICTextureFromFile(g_flowVisTool.m_d3dDevice, wfilename.c_str(), &tmp, &g_flowVisTool.g_particleRenderParams.m_pColorTexture))) {
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

void FlowVisToolGUI::LoadSeedTexture(FlowVisTool& g_flowVisTool)
{
	std::string filename;
	if (tum3d::GetFilenameDialog("Load Texture", "Images (jpg, png, bmp)\0*.png;*.jpg;*.jpeg;*.bmp\0", filename, false)) {
		//create new texture
		std::wstring wfilename(filename.begin(), filename.end());
		ID3D11Resource* res = NULL;
		//ID3D11ShaderResourceView* srv = NULL;
		if (!FAILED(DirectX::CreateWICTextureFromFileEx(g_flowVisTool.m_d3dDevice, wfilename.c_str(), 0Ui64, D3D11_USAGE_STAGING, 0, D3D11_CPU_ACCESS_READ, 0, false, &res, NULL))) {
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
			g_flowVisTool.m_d3dDevice->GetImmediateContext(&context);
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

void FlowVisToolGUI::RenderGUI(FlowVisTool& g_flowVisTool, bool& resizeNextFrame, ImVec2& sceneWindowSize)
{
	if (g_show_demo_window)
		ImGui::ShowDemoWindow(&g_show_demo_window);

	MainMenu(g_flowVisTool);

	DatasetWindow(g_flowVisTool);
	TracingWindow(g_flowVisTool);
	ExtraWindow(g_flowVisTool);
	FTLEWindow(g_flowVisTool);
	HeatmapWindow(g_flowVisTool);
	RenderingWindow(g_flowVisTool);
	SceneWindow(g_flowVisTool, resizeNextFrame, sceneWindowSize);
	ProfilerWindow(g_flowVisTool);

	ImGui::Begin("Debug");
	{
		static float sleepAmount = 0.0f;
		ImGui::SliderFloat("Thread sleep", &sleepAmount, 0.0f, 1000.0f);

		std::this_thread::sleep_for(std::chrono::milliseconds((long long)sleepAmount));
	}
	ImGui::End();
}

void FlowVisToolGUI::DockSpace()
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

void FlowVisToolGUI::MainMenu(FlowVisTool& g_flowVisTool)
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
			ImGui::Separator();
			if (ImGui::MenuItem("ImGui Demo", nullptr, g_show_demo_window))
				g_show_demo_window = !g_show_demo_window;
			if (ImGui::MenuItem("Profiler", nullptr, g_showProfilerWindow))
				g_showProfilerWindow = !g_showProfilerWindow;

			ImGui::EndMenu();
		}
		ImGui::EndMainMenuBar();
	}
}

void FlowVisToolGUI::DatasetWindow(FlowVisTool& g_flowVisTool)
{
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
}

void FlowVisToolGUI::TracingWindow(FlowVisTool& g_flowVisTool)
{
	// Particle tracing config window
	if (g_showTracingOptionsWindow)
	{
		ImGui::SetNextWindowSize(ImVec2(300, 300), ImGuiCond_FirstUseEver);
		if (ImGui::Begin("Tracing Options", &g_showTracingOptionsWindow))
		{
			ImGui::PushItemWidth(-150);
			{
				if (ImGui::Button("Retrace", ImVec2(buttonWidth, 0)))
					g_flowVisTool.m_retrace = true;

				if (ImGui::Button(g_flowVisTool.g_particleTracingPaused ? "Continue" : "Pause", ImVec2(buttonWidth, 0)))
					g_flowVisTool.g_particleTracingPaused = !g_flowVisTool.g_particleTracingPaused;

				ImGui::Spacing();
				ImGui::Separator();

				ImGui::Checkbox("Verbose", &g_flowVisTool.g_tracingManager.GetVerbose());

				// Seeding options
				ImGui::Spacing();
				ImGui::Separator();
				{
					if (ImGui::Button("Load seed texture", ImVec2(buttonWidth, 0)))
						LoadSeedTexture(g_flowVisTool);

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
}

void FlowVisToolGUI::ExtraWindow(FlowVisTool& g_flowVisTool)
{
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
					SaveLinesDialog(g_flowVisTool);
				if (ImGui::Button("Load Lines", ImVec2(buttonWidth, 0)))
					LoadLinesDialog(g_flowVisTool);

				if (ImGui::DragInt("Line ID Override", &g_lineIDOverride, 1, -1, INT_MAX))
					g_lineIDOverride = std::max(-1, g_lineIDOverride);

				if (ImGui::Button("Clear Loaded Lines", ImVec2(buttonWidth, 0)))
				{
					g_flowVisTool.ReleaseLineBuffers();
					g_flowVisTool.m_redraw = true;
				}

				ImGui::Spacing();
				ImGui::Separator();

				if (ImGui::Button("Load Balls", ImVec2(buttonWidth, 0)))
					LoadBallsDialog(g_flowVisTool);

				if (ImGui::DragFloat("Ball Radius", &g_flowVisTool.g_ballRadius, 0.001f, 0.0f, FLT_MAX, "%.3f"))
					g_flowVisTool.g_ballRadius = std::max(0.0f, g_flowVisTool.g_ballRadius);

				if (ImGui::Button("Clear Loaded Balls", ImVec2(buttonWidth, 0)))
				{
					g_flowVisTool.ReleaseBallBuffers();
					g_flowVisTool.m_redraw = true;
				}

				ImGui::Spacing();
				ImGui::Separator();

				if (ImGui::Button("Build Flow Graph", ImVec2(buttonWidth, 0)))
					g_flowVisTool.BuildFlowGraph("flowgraph.txt");

				if (ImGui::Button("Save Flow Graph", ImVec2(buttonWidth, 0)))
					g_flowVisTool.SaveFlowGraph();

				if (ImGui::Button("Load Flow Graph", ImVec2(buttonWidth, 0)))
					g_flowVisTool.LoadFlowGraph();

				ImGui::Spacing();
				ImGui::Separator();

				if (ImGui::Button("Save Settings", ImVec2(buttonWidth, 0)))
					SaveRenderingParamsDialog(g_flowVisTool);

				if (ImGui::Button("Load Settings", ImVec2(buttonWidth, 0)))
					LoadRenderingParamsDialog(g_flowVisTool);

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
}

void FlowVisToolGUI::FTLEWindow(FlowVisTool& g_flowVisTool)
{
	// FTLE window
	if (g_showFTLEWindow)
	{
		ImGui::SetNextWindowSize(ImVec2(300, 300), ImGuiCond_FirstUseEver);
		if (ImGui::Begin("FTLE", &g_showFTLEWindow))
		{
			ImGui::PushItemWidth(-150);
			{
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
						g_flowVisTool.g_particleTraceParams.m_lineCount = g_flowVisTool.g_particleTraceParams.m_ftleResolution * g_flowVisTool.g_particleTraceParams.m_ftleResolution * 6;
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
						g_flowVisTool.g_particleTraceParams.m_lineCount = g_flowVisTool.g_particleTraceParams.m_ftleResolution * g_flowVisTool.g_particleTraceParams.m_ftleResolution * 6;
				}

				ImGui::DragFloat("Slice (Y)", &g_flowVisTool.g_particleTraceParams.m_ftleSliceY, 0.001f);

				ImGui::SliderFloat("Slice Alpha", &g_flowVisTool.g_particleRenderParams.m_ftleTextureAlpha, 0.0f, 1.0f);

				ImGui::DragFloat3("Separation Dist", (float*)&g_flowVisTool.g_particleTraceParams.m_ftleSeparationDistance, 0.0000001f, 0.0f, FLT_MAX, "%.7f");
			}
			ImGui::PopItemWidth();
		}
		ImGui::End();
	}
}

void FlowVisToolGUI::HeatmapWindow(FlowVisTool& g_flowVisTool)
{
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
}

void FlowVisToolGUI::RenderingWindow(FlowVisTool& g_flowVisTool)
{
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
					LoadColorTexture(g_flowVisTool);

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
					LoadSliceTexture(g_flowVisTool);

				ImGui::Checkbox("Show Slice", &g_flowVisTool.g_particleRenderParams.m_showSlice);

				ImGui::DragFloat("Slice Position", &g_flowVisTool.g_particleRenderParams.m_slicePosition, 0.001f);

				if (ImGui::DragFloat("Slice Transparency", &g_flowVisTool.g_particleRenderParams.m_sliceAlpha, 0.001f, 0.0f, 1.0f, "%.3f"))
					g_flowVisTool.g_particleRenderParams.m_sliceAlpha = std::min(1.0f, std::max(0.0f, g_flowVisTool.g_particleRenderParams.m_sliceAlpha));
			}
			ImGui::PopItemWidth();
		}
		ImGui::End();
	}
}

void FlowVisToolGUI::SceneWindow(FlowVisTool& g_flowVisTool, bool& resizeNextFrame, ImVec2& sceneWindowSize)
{
	// Scene view window
	ImGui::PushStyleVar(ImGuiStyleVar_::ImGuiStyleVar_WindowPadding, ImVec2(2, 2));
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
	ImGui::PopStyleVar(ImGuiStyleVar_::ImGuiStyleVar_WindowPadding);
}

void FlowVisToolGUI::ProfilerWindow(FlowVisTool& g_flowVisTool)
{
	currentFrameTimeIndex = (currentFrameTimeIndex + 1) % frameTimes.size();
	frameTimes[currentFrameTimeIndex] = ImGui::GetIO().DeltaTime * 1000.0f;

	if (g_showProfilerWindow)
	{
		if (ImGui::Begin("Profiler"))
		{
			//ImGui::Text("Average %.2f ms (%.1f FPS)", 1000.0f / ImGui::GetIO().Framerate, ImGui::GetIO().Framerate);
			ImGui::Text("Current %.2f ms (%.1f FPS)", ImGui::GetIO().DeltaTime * 1000.0f, 1.0f / ImGui::GetIO().DeltaTime);

			float max = FLT_MIN;
			for (size_t i = 0; i < frameTimes.size(); i++)
				max = std::max(frameTimes[i], max);
			//max = std::ceil(max * 1.1f);
			max = std::ceil(max / 5.0f) * 5.0f;

			ImGui::BeginGroup();
			ImGui::Text("%.0f", max);
			ImGui::Text(" ");
			ImGui::Text(" ");
			ImGui::Text("0");
			ImGui::EndGroup();
			
			// Capture the group size and create widgets using the same size
			ImVec2 size = ImGui::GetItemRectSize();

			ImGui::SameLine();

			ImGui::PushItemWidth(-1);
			ImGui::PlotLines("", &frameTimes[0], frameTimes.size(), currentFrameTimeIndex, "", 0.0f, max, ImVec2(0, size.y));
			ImGui::PopItemWidth();

			ImGui::Separator();
			ImGui::Spacing();

			ImGui::Text("Window resolution: %dx%d", g_flowVisTool.g_windowSize.x(), g_flowVisTool.g_windowSize.y());
			ImGui::Text("Viewport resolution: %dx%d", g_flowVisTool.g_projParams.m_imageWidth, g_flowVisTool.g_projParams.m_imageHeight);
			ImGui::Text("Trace Time: %.2f ms", g_flowVisTool.g_timerTracing.GetElapsedTimeMS());
			ImGui::Text("Render Time: %.2f ms", g_flowVisTool.g_timerRendering.GetElapsedTimeMS());
		}
		ImGui::End();
	}
}