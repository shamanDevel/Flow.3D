#include <FlowVisToolGUI.h>


#include <Vec.h>
#include <WICTextureLoader.h>
#include <omp.h>
#include <fstream>
#include <vector>


bool FlowVisToolGUI::g_showTrajectoriesRenderingSettingsWindow = true;
bool FlowVisToolGUI::g_showGeneralRenderingSettingsWindow = true;
bool FlowVisToolGUI::g_showTracingOptionsWindow = true;
bool FlowVisToolGUI::g_showFTLEWindow = false;
bool FlowVisToolGUI::g_showRaycastingWindow = true;
bool FlowVisToolGUI::g_showHeatmapWindow = false;
bool FlowVisToolGUI::g_showExtraWindow = false;
bool FlowVisToolGUI::g_showDatasetWindow = true;
bool FlowVisToolGUI::g_show_demo_window = false; 
bool FlowVisToolGUI::g_showProfilerWindow = true;
bool FlowVisToolGUI::g_showStatusWindow = true;


const float buttonWidth = 200;
int g_threadCount = omp_get_num_procs();
int FlowVisToolGUI::g_lineIDOverride = -1;


const int max_frames = 150;
std::vector<float> frameTimes(max_frames);
int currentFrameTimeIndex = 0;


ImVec4 sectionTextColor = ImVec4(0.67f, 0.52f, 0.00f, 1.00f);
ImVec4 datasetNameTextColor = ImVec4(0.26f, 0.59f, 0.98f, 0.95f);


#pragma region Utils
void SectionText(const char* str)
{
	ImGui::PushStyleColor(ImGuiCol_Text, sectionTextColor);
	ImGui::Text(str);
	ImGui::PopStyleColor();
}
#pragma endregion


#pragma region Dialogs

#ifdef Single
void FlowVisToolGUI::SaveLinesDialog(FlowVisTool& flowVisTool)
{
	//bool bFullscreen = (!DXUTIsWindowed());

	//if( bFullscreen ) DXUTToggleFullScreen();

	std::string filename;
	if (tum3d::GetFilenameDialog("Save Lines", "*.linebuf\0*.linebuf", filename, true))
	{
		filename = tum3d::RemoveExt(filename) + ".linebuf";
		float posOffset = 0.0f;
		if (flowVisTool.g_particleTraceParams.m_upsampledVolumeHack)
		{
			// upsampled volume is offset by half a grid spacing...
			float gridSpacingWorld = 2.0f / float(flowVisTool.g_volumes[0]->GetVolumeSize().maximum());
			posOffset = 0.5f * gridSpacingWorld;
		}
		if (!flowVisTool.g_tracingManager.GetResult()->Write(filename, posOffset))
		{
			printf("Saving lines to file %s failed!\n", filename.c_str());
		}
	}

	//if( bFullscreen ) DXUTToggleFullScreen();
}
#endif

void FlowVisToolGUI::LoadLinesDialog(FlowVisTool& flowVisTool)
{
	//bool bFullscreen = (!DXUTIsWindowed());

	//if( bFullscreen ) DXUTToggleFullScreen();

	std::string filename;
	if (tum3d::GetFilenameDialog("Load Lines", "*.linebuf\0*.linebuf", filename, false))
	{
		LineBuffers* pBuffers = new LineBuffers(flowVisTool.m_d3dDevice);
		if (!pBuffers->Read(filename, g_lineIDOverride))
		{
			printf("Loading lines from file %s failed!\n", filename.c_str());
			delete pBuffers;
		}
		else
		{
			flowVisTool.g_lineBuffers.push_back(pBuffers);
		}
	}

	//if( bFullscreen ) DXUTToggleFullScreen();

	flowVisTool.g_raycastParams.m_redraw = true;
}

void FlowVisToolGUI::LoadBallsDialog(FlowVisTool& flowVisTool)
{
	//bool bFullscreen = (!DXUTIsWindowed());

	//if( bFullscreen ) DXUTToggleFullScreen();

	std::string filename;
	if (tum3d::GetFilenameDialog("Load Balls", "*.*\0*.*", filename, false))
	{
		BallBuffers* pBuffers = new BallBuffers(flowVisTool.m_d3dDevice);
		if (!pBuffers->Read(filename))
		{
			printf("Loading balls from file %s failed!\n", filename.c_str());
			delete pBuffers;
		}
		else
		{
			flowVisTool.g_ballBuffers.push_back(pBuffers);
		}
	}

	//if( bFullscreen ) DXUTToggleFullScreen();

	flowVisTool.g_raycastParams.m_redraw = true;
}

void FlowVisToolGUI::SaveRenderingParamsDialog(FlowVisTool& flowVisTool)
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

void FlowVisToolGUI::LoadRenderingParamsDialog(FlowVisTool& flowVisTool)
{
	//bool bFullscreen = (!DXUTIsWindowed());

	//if( bFullscreen ) DXUTToggleFullScreen();

	//std::string filename;
	//if ( tum3d::GetFilenameDialog("Load Settings", "*.cfg\0*.cfg", filename, false) ) 
	//	LoadRenderingParams( filename );

	//if( bFullscreen ) DXUTToggleFullScreen();	
}

#ifdef Single
void FlowVisToolGUI::LoadSliceTexture(FlowVisTool& flowVisTool)
{
	std::string filename;
	if (tum3d::GetFilenameDialog("Load Texture", "Images (jpg, png, bmp)\0*.png;*.jpg;*.jpeg;*.bmp\0", filename, false)) {
		//release old texture
		SAFE_RELEASE(flowVisTool.g_particleRenderParams.m_pSliceTexture);
		//create new texture
		std::wstring wfilename(filename.begin(), filename.end());
		ID3D11Resource* tmp = NULL;
		if (!FAILED(DirectX::CreateWICTextureFromFile(flowVisTool.m_d3dDevice, wfilename.c_str(), &tmp, &flowVisTool.g_particleRenderParams.m_pSliceTexture))) {
			std::cout << "Slice texture " << filename << " loaded" << std::endl;
			flowVisTool.g_particleRenderParams.m_showSlice = true;
			flowVisTool.g_renderingParams.m_redraw = true;
		}
		else {
			std::cerr << "Failed to load slice texture" << std::endl;
		}
		SAFE_RELEASE(tmp);
	}
}
#endif

bool DialogColorTexture(std::string& path)
{
	return tum3d::GetFilenameDialog("Load Texture", "Images (jpg, png, bmp)\0*.png;*.jpg;*.jpeg;*.bmp\0", path, false);
}


#ifdef Single
void FlowVisToolGUI::LoadSeedTexture(FlowVisTool& flowVisTool)
{
	std::string filename;
	if (tum3d::GetFilenameDialog("Load Texture", "Images (jpg, png, bmp)\0*.png;*.jpg;*.jpeg;*.bmp\0", filename, false)) {
		//create new texture
		std::wstring wfilename(filename.begin(), filename.end());
		ID3D11Resource* res = NULL;
		//ID3D11ShaderResourceView* srv = NULL;
		if (!FAILED(DirectX::CreateWICTextureFromFileEx(flowVisTool.m_d3dDevice, wfilename.c_str(), 0Ui64, D3D11_USAGE_STAGING, 0, D3D11_CPU_ACCESS_READ, 0, false, &res, NULL))) {
			std::cout << "Seed texture " << filename << " loaded" << std::endl;
			//delete old data
			delete[] flowVisTool.g_particleTraceParams.m_seedTexture.m_colors;
			flowVisTool.g_particleTraceParams.m_seedTexture.m_colors = NULL;
			//Copy to cpu memory
			ID3D11Texture2D* tex = NULL;
			res->QueryInterface(&tex);
			D3D11_TEXTURE2D_DESC desc;
			tex->GetDesc(&desc);
			flowVisTool.g_particleTraceParams.m_seedTexture.m_width = desc.Width;
			flowVisTool.g_particleTraceParams.m_seedTexture.m_height = desc.Height;
			flowVisTool.g_particleTraceParams.m_seedTexture.m_colors = new unsigned int[desc.Width * desc.Height];
			D3D11_MAPPED_SUBRESOURCE mappedResource;
			ID3D11DeviceContext* context = NULL;
			flowVisTool.m_d3dDevice->GetImmediateContext(&context);
			if (!FAILED(context->Map(tex, 0, D3D11_MAP_READ, 0, &mappedResource))) {
				for (int y = 0; y < desc.Width; ++y) {
					memcpy(&flowVisTool.g_particleTraceParams.m_seedTexture.m_colors[y*desc.Width], ((char*)mappedResource.pData) + (y*mappedResource.RowPitch), sizeof(unsigned int) * desc.Width);
				}
				context->Unmap(tex, 0);
			}
			SAFE_RELEASE(context);
			SAFE_RELEASE(tex);
			//reset color
			flowVisTool.g_particleTraceParams.m_seedTexture.m_picked.clear();
			//set seed box to domain
			flowVisTool.SetBoundingBoxToDomainSize();
			std::cout << "Seed texture copied to cpu memory" << std::endl;
		}
		else {
			std::cerr << "Failed to load seed texture" << std::endl;
		}
		//SAFE_RELEASE(srv);
		SAFE_RELEASE(res);
	}
}
#endif
#pragma endregion


void TimeVolGUI(TimeVolume& volume)
{
	//int32 timestepMax = g_volumes[0]->GetTimestepCount() - 1;
	//float timeSpacing = g_volumes[0]->GetTimeSpacing();
	//float timeMax = timeSpacing * float(timestepMax);
	//TwSetParam(g_pTwBarMain, "Time", "max", TW_PARAM_FLOAT, 1, &timeMax);
	//TwSetParam(g_pTwBarMain, "Time", "step", TW_PARAM_FLOAT, 1, &timeSpacing);
	//TwSetParam(g_pTwBarMain, "Timestep", "max", TW_PARAM_INT32, 1, &timestepMax);

	const TimeVolumeInfo& volInfo = volume.GetInfo();

	ImGui::Text("Brick count: %d, %d, %d", volInfo.GetBrickCount().x(), volInfo.GetBrickCount().y(), volInfo.GetBrickCount().z());
	ImGui::Text("World size: %.3f, %.3f, %.3f", volInfo.GetBrickSizeWorld().x(), volInfo.GetBrickSizeWorld().y(), volInfo.GetBrickSizeWorld().z());
	ImGui::Text("Volume size: %d, %d, %d", volInfo.GetVolumeSize().x(), volInfo.GetVolumeSize().y(), volInfo.GetVolumeSize().z());
	ImGui::Text("Timestep count: %d", volInfo.GetTimestepCount());
	ImGui::Text("Time spacing: %.5f", volInfo.GetTimeSpacing());

	ImGui::Spacing();
	ImGui::Separator();

	if (ImGui::Button("Preload nearest timestep", ImVec2(buttonWidth, 0)))
	{
		std::cout << "Loading timestep...";
		TimerCPU timer;
		timer.Start();
		volume.LoadNearestTimestep();
		timer.Stop();
		std::cout << " done in " << timer.GetElapsedTimeMS() / 1000.0f << "s" << std::endl;
	}

	int32 timestepMax = volume.GetTimestepCount() - 1;
	float timeSpacing = volume.GetTimeSpacing();
	float timeMax = timeSpacing * float(timestepMax);

	float t = volume.GetCurTime();
	if (ImGui::InputFloat("Time", &t, timeSpacing, timeSpacing * 2.0f, 0))
	{
		t = std::max(0.0f, std::min(t, timeMax));

		volume.SetCurTime(t);
	}

	t = volume.GetCurNearestTimestepIndex();
	if (ImGui::SliderFloat("Timestep", &t, 0.0f, timestepMax, "%.0f"))
	{
		t = t * volume.GetTimeSpacing();

		volume.SetCurTime(t);
	}

	t = volume.GetTimeSpacing();
	if (ImGui::DragFloat("Time spacing", &t, 0.05f, 0.05f, timeMax, "%.2f"))
	{
		t = std::max(0.05f, std::min(t, timeMax));

		volume.SetTimeSpacing(t);
	}

	if (ImGui::Button("Save as raw", ImVec2(buttonWidth, 0)))
	{
		std::string filename;
		if (tum3d::GetFilenameDialog("Select output file", "Raw (*.raw)\0*.raw\0", filename, true))
		{
			// remove extension
			if (filename.substr(filename.size() - 4) == ".raw")
				filename = filename.substr(0, filename.size() - 4);

			std::vector<std::string> filenames;
			for (int c = 0; c < volume.GetChannelCount(); c++)
			{
				std::ostringstream str;
				str << filename << char('X' + c) << ".raw";
				filenames.push_back(str.str());
			}

			std::cout << "Not implemented." << std::endl;

			//flowVisTool.g_renderingManager.WriteCurTimestepToRaws(flowVisTool.g_volume, filenames);
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
			for (int c = 0; c < volume.GetChannelCount(); c++)
			{
				std::ostringstream str;
				str << filename << char('X' + c) << ".la3d";
				filenames.push_back(str.str());
			}
			std::cout << "Not implemented." << std::endl;

			//flowVisTool.g_renderingManager.WriteCurTimestepToLA3Ds(flowVisTool.g_volume, filenames);
		}
	}
}

void TraceParamsGUI(FlowVisToolVolumeData* selected)
{
	ParticleTraceParams& traceParams = selected->m_traceParams;

	if (ImGui::Button("Retrace", ImVec2(buttonWidth, 0)))
		selected->m_retrace = true;

	{
		bool wasPaused = selected->m_tracingPaused;
		if (wasPaused)
			ImGui::PushStyleColor(ImGuiCol_::ImGuiCol_Button, sectionTextColor);

		if (ImGui::Button(selected->m_tracingPaused ? "Continue" : "Pause", ImVec2(buttonWidth, 0)))
			selected->m_tracingPaused = !selected->m_tracingPaused;

		if (wasPaused)
			ImGui::PopStyleColor();
	}


	ImGui::Spacing();
	ImGui::Separator();

	ImGui::Checkbox("Verbose", &selected->m_tracingManager.GetVerbose());

	// Seeding options
	ImGui::Spacing();
	ImGui::Separator();
	{
#ifdef Single
		if (ImGui::Button("Load seed texture", ImVec2(buttonWidth, 0)))
			LoadSeedTexture(flowVisTool);
#endif
		if (ImGui::Button("Set seed box to domain", ImVec2(buttonWidth, 0)))
			selected->SetSeedingBoxToDomainSize();

		ImGui::DragFloat3("Seed box min", (float*)&traceParams.m_seedBoxMin, 0.005f, 0.0f, 0.0f, "%.3f");
		ImGui::DragFloat3("Seed box size", (float*)&traceParams.m_seedBoxSize, 0.005f, 0.0f, 0.0f, "%.3f");

		static auto getterSeedingPattern = [](void* data, int idx, const char** out_str)
		{
			if (idx >= ParticleTraceParams::eSeedPattern::COUNT) return false;
			*out_str = ParticleTraceParams::GetSeedPatternName(ParticleTraceParams::eSeedPattern(idx));
			return true;
		};
		ImGui::Combo("Seeding pattern", (int*)&traceParams.m_seedPattern, getterSeedingPattern, nullptr, ParticleTraceParams::eSeedPattern::COUNT);
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

		ImGui::Combo("Advection", (int*)&traceParams.m_advectMode, getterAdvectMode, nullptr, ADVECT_MODE_COUNT);
		ImGui::Checkbox("Dense output", &traceParams.m_enableDenseOutput);

		static auto getterFilterMode = [](void* data, int idx, const char** out_str)
		{
			if (idx >= TEXTURE_FILTER_MODE_COUNT) return false;
			*out_str = GetTextureFilterModeName(eTextureFilterMode(idx));
			return true;
		};
		ImGui::Combo("Interpolation", (int*)&traceParams.m_filterMode, getterFilterMode, nullptr, TEXTURE_FILTER_MODE_COUNT);

		static auto getterLineMode = [](void* data, int idx, const char** out_str)
		{
			if (idx >= LINE_MODE_COUNT) return false;
			*out_str = GetLineModeName(eLineMode(idx));
			return true;
		};
		if (ImGui::Combo("Line mode", (int*)&traceParams.m_lineMode, getterLineMode, nullptr, LINE_MODE_COUNT))
		{
			switch (traceParams.m_lineMode)
			{
			case eLineMode::LINE_PARTICLE_STREAM:
			case eLineMode::LINE_PARTICLES:
			{
				selected->m_renderParams.m_lineRenderMode = eLineRenderMode::LINE_RENDER_PARTICLES;
				break;
			}
			case eLineMode::LINE_PATH:
			case eLineMode::LINE_STREAM:
			{
				selected->m_renderParams.m_lineRenderMode = eLineRenderMode::LINE_RENDER_TUBE;
				break;
			}
			}
		}

		if (ImGui::DragInt("Line count", &traceParams.m_lineCount, 1.0f, 1.0f, INT_MAX))
			traceParams.m_lineCount = std::max(1, traceParams.m_lineCount);
		if (ImGui::DragInt("Line max lenght", &traceParams.m_lineLengthMax, 1.0f, 2.0f, INT_MAX))
			traceParams.m_lineLengthMax = std::max(2, traceParams.m_lineLengthMax);
		if (ImGui::DragFloat("Line max age", &traceParams.m_lineAgeMax, 0.05f, 0.0f, FLT_MAX, "%.3f"))
			traceParams.m_lineAgeMax = std::max(0.0f, traceParams.m_lineAgeMax);

		ImGui::DragFloat("Min velocity", &traceParams.m_minVelocity, 0.01f, 0.0f, 0.0f, "%.2f");
		ImGui::DragFloat("Particles per second", &traceParams.m_particlesPerSecond, 0.01f, 0.0f, 0.0f, "%.2f");
		ImGui::DragFloat("Advection Delta T", &traceParams.m_advectDeltaT, 0.001f, 0.0f, 0.0f, "%.5f");
		if (ImGui::Button("Seed many particles", ImVec2(buttonWidth, 0)))
			selected->m_tracingManager.SeedManyParticles();
		ImGui::DragFloat("Cell Change Time Threshold", &traceParams.m_cellChangeThreshold, 0.001f, 0.0f, 0.0f, "%.5f");
	}

	ImGui::Spacing();
	ImGui::Separator();

	ImGui::Checkbox("CPU Tracing", &traceParams.m_cpuTracing);

	if (ImGui::SliderInt("# CPU Threads", &g_threadCount, 1, 16))
		omp_set_num_threads(g_threadCount);

	ImGui::Spacing();
	ImGui::Separator();

	int v;
	//v = flowVisTool.g_tracingManager.GetBrickSlotCountMax();
	//if (ImGui::DragInt("Max Brick Slot Count", &v, 1, 0, INT_MAX))
	//	flowVisTool.g_tracingManager.GetBrickSlotCountMax() = (unsigned int)std::max(0, v);

	//v = flowVisTool.g_tracingManager.GetTimeSlotCountMax();
	//if (ImGui::DragInt("Max Time Slot Count", &v, 1, 0, INT_MAX))
	//	flowVisTool.g_tracingManager.GetTimeSlotCountMax() = (unsigned int)std::max(0, v);

	ImGui::Spacing();
	ImGui::Separator();

	if (ImGui::DragFloat("Advection Error Tolerance (Voxels)", &traceParams.m_advectErrorTolerance, 0.001f, 0.0f, FLT_MAX, "%.5f"))
		traceParams.m_advectErrorTolerance = std::max(0.0f, traceParams.m_advectErrorTolerance);

	if (ImGui::DragFloat("Advection Delta T Min", &traceParams.m_advectDeltaTMin, 0.001f, 0.0f, FLT_MAX, "%.5f"))
		traceParams.m_advectDeltaTMin = std::max(0.0f, traceParams.m_advectDeltaTMin);

	if (ImGui::DragFloat("Advection Delta T Max", &traceParams.m_advectDeltaTMax, 0.001f, 0.0f, FLT_MAX, "%.5f"))
		traceParams.m_advectDeltaTMax = std::max(0.0f, traceParams.m_advectDeltaTMax);

	v = traceParams.m_advectStepsPerRound;
	if (ImGui::DragInt("Advect Steps per Round", &v, 1, 0, INT_MAX, "%d steps"))
		traceParams.m_advectStepsPerRound = (unsigned int)std::max(0, v);

	v = traceParams.m_purgeTimeoutInRounds;
	if (ImGui::DragInt("Brick Purge Timeout", &v, 1, 0, INT_MAX, "%d rounds"))
		traceParams.m_purgeTimeoutInRounds = (unsigned int)std::max(0, v);

	ImGui::Spacing();
	ImGui::Separator();

	ImGui::Text("Heuristic");

	if (ImGui::DragFloat("Bonus Factor", &traceParams.m_heuristicBonusFactor, 0.01f, 0.0f, FLT_MAX, "%.5f"))
		traceParams.m_heuristicBonusFactor = std::max(0.0f, traceParams.m_heuristicBonusFactor);

	if (ImGui::DragFloat("Penalty Factor", &traceParams.m_heuristicPenaltyFactor, 0.01f, 0.0f, FLT_MAX, "%.5f"))
		traceParams.m_heuristicPenaltyFactor = std::max(0.0f, traceParams.m_heuristicPenaltyFactor);

	// TODO: this should be a combo, no?
	v = traceParams.m_heuristicFlags;
	if (ImGui::DragInt("Flags", &v, 1, 0, INT_MAX))
		traceParams.m_heuristicFlags = (unsigned int)std::max(0, v);

	ImGui::Spacing();
	ImGui::Separator();

	if (ImGui::DragFloat("Output Pos Diff (Voxels)", &traceParams.m_outputPosDiff, 0.01f, 0.0f, FLT_MAX, "%.5f"))
		traceParams.m_outputPosDiff = std::max(0.0f, traceParams.m_outputPosDiff);

	if (ImGui::DragFloat("Output Time Diff", &traceParams.m_outputTimeDiff, 0.01f, 0.0f, FLT_MAX, "%.5f"))
		traceParams.m_outputTimeDiff = std::max(0.0f, traceParams.m_outputTimeDiff);

	ImGui::Checkbox("Wait for Disk", &traceParams.m_waitForDisk);
	ImGui::Checkbox("Prefetching", &traceParams.m_enablePrefetching);
	ImGui::Checkbox("Upsampled Volume Hack", &traceParams.m_upsampledVolumeHack);
}

void TransferFunctionGUI(TransferFunction* tf)
{
	if (ImGui::ColorEdit4("color0", tf->m_color0, ImGuiColorEditFlags_NoInputs | ImGuiColorEditFlags_NoLabel | ImGuiColorEditFlags_AlphaPreviewHalf))
		tf->UpdateTexture();
	ImGui::SameLine();
	ImGui::DragFloat("Min", &tf->m_rangeMin, 0.01f);
	
	if (ImGui::ColorEdit4("color1", tf->m_color1, ImGuiColorEditFlags_NoInputs | ImGuiColorEditFlags_NoLabel | ImGuiColorEditFlags_AlphaPreviewHalf))
		tf->UpdateTexture();
	ImGui::SameLine();
	ImGui::DragFloat("Max", &tf->m_rangeMax, 0.01f);
}

void TrajectoriesRenderingParamsGUI(FlowVisToolVolumeData* selected, ID3D11Device* device)
{
	ParticleRenderParams& renderParams = selected->m_renderParams;

	ImGui::Checkbox("Enabled", &renderParams.m_linesEnabled);

	ImGui::Checkbox("Show Brick Boxes", &selected->m_renderBrickBoxes);
	ImGui::Checkbox("Show Seed Box", &selected->m_renderSeedBox);
	ImGui::Checkbox("Show Domain Box", &selected->m_renderDomainBox);

	ImGui::Spacing();
	ImGui::Separator();

	SectionText("Rendering Mode");

	static auto getterLineRenderMode = [](void* data, int idx, const char** out_str)
	{
		if (idx >= LINE_RENDER_MODE_COUNT) return false;
		*out_str = GetLineRenderModeName(eLineRenderMode(idx));
		return true;
	};
	ImGui::Combo("Line render mode", (int*)&renderParams.m_lineRenderMode, getterLineRenderMode, nullptr, LINE_RENDER_MODE_COUNT);


	switch (renderParams.m_lineRenderMode)
	{
	case eLineRenderMode::LINE_RENDER_TUBE:
	{
		ImGui::Spacing();
		ImGui::Separator();
		SectionText("Tube Rendering Settings");
		if (ImGui::DragFloat("Radius", &renderParams.m_tubeRadius, 0.001f, 0.0f, FLT_MAX, "%.3f"))
			renderParams.m_tubeRadius = std::max(0.0f, renderParams.m_tubeRadius);
		break;
	}
	case eLineRenderMode::LINE_RENDER_RIBBON:
	{
		ImGui::Spacing();
		ImGui::Separator();
		SectionText("Ribbon Rendering Settings");
		if (ImGui::DragFloat("Width", &renderParams.m_ribbonWidth, 0.001f, 0.0f, FLT_MAX, "%.3f"))
			renderParams.m_ribbonWidth = std::max(0.0f, renderParams.m_ribbonWidth);
		break;
	}
	case eLineRenderMode::LINE_RENDER_PARTICLES:
	{
		ImGui::Spacing();
		ImGui::Separator();
		SectionText("Particle Rendering Settings");
		if (ImGui::DragFloat("Size", &renderParams.m_particleSize, 0.001f, 0.0f, FLT_MAX, "%.3f"))
			renderParams.m_particleSize = std::max(0.0f, renderParams.m_particleSize);

		static auto getterParticleRenderMode = [](void* data, int idx, const char** out_str)
		{
			if (idx >= PARTICLE_RENDER_MODE_COUNT) return false;
			*out_str = GetParticleRenderModeName(eParticleRenderMode(idx));
			return true;
		};
		ImGui::Combo("Render mode", (int*)&renderParams.m_particleRenderMode, getterParticleRenderMode, nullptr, PARTICLE_RENDER_MODE_COUNT);

		if (ImGui::DragFloat("Transparency", &renderParams.m_particleTransparency, 0.001f, 0.0f, 1.0f, "%.3f"))
			renderParams.m_particleTransparency = std::min(1.0f, std::max(0.0f, renderParams.m_particleTransparency));

		ImGui::Checkbox("Sort Particles", &renderParams.m_sortParticles);

		break;
	}
	}

	if (renderParams.m_lineRenderMode == eLineRenderMode::LINE_RENDER_TUBE || renderParams.m_lineRenderMode == eLineRenderMode::LINE_RENDER_PARTICLES)
	{
		ImGui::Checkbox("Display Velocity", &renderParams.m_tubeRadiusFromVelocity);
		ImGui::SameLine();
		ImGui::DragFloat("Reference", &renderParams.m_referenceVelocity, 0.001f);
	}


	ImGui::Spacing();
	ImGui::Separator();


	SectionText("Color Mode");
	static auto getterLineColorMode = [](void* data, int idx, const char** out_str)
	{
		if (idx >= LINE_COLOR_MODE_COUNT) return false;
		*out_str = GetLineColorModeName(eLineColorMode(idx));
		return true;
	};
	ImGui::Combo("Color Mode", (int*)&renderParams.m_lineColorMode, getterLineColorMode, nullptr, LINE_COLOR_MODE_COUNT);

	switch (renderParams.m_lineColorMode)
	{
	case eLineColorMode::AGE:
	{
		ImGui::ColorEdit3("Color 0", (float*)&renderParams.m_color0);
		ImGui::ColorEdit3("Color 1", (float*)&renderParams.m_color1);
		break;
	}
	case eLineColorMode::TEXTURE:
	{
		if (ImGui::Button("Load Color Texture", ImVec2(buttonWidth, 0)))
		{
			std::string path("");
			if (DialogColorTexture(path))
				renderParams.LoadColorTexture(path, device);
		}
		break;
	}
	case eLineColorMode::MEASURE:
	{
		static auto getterMeasureMode = [](void* data, int idx, const char** out_str)
		{
			if (idx >= MEASURE_COUNT) return false;
			*out_str = GetMeasureName(eMeasure(idx));
			return true;
		};
		ImGui::Combo("Measure", (int*)&renderParams.m_measure, getterMeasureMode, nullptr, MEASURE_COUNT);

		ImGui::DragFloat("Scale", &renderParams.m_measureScale, 0.001f);

		TransferFunctionGUI(&renderParams.m_transferFunction);
		break;
	}
	}

	ImGui::Spacing();
	ImGui::Separator();

	SectionText("Time Stripes");
	ImGui::Checkbox("Time Stripes", &renderParams.m_timeStripes);

	if (ImGui::DragFloat("Time Stripe Length", &renderParams.m_timeStripeLength, 0.001f, 0.001f, FLT_MAX, "%.3f"))
		renderParams.m_timeStripeLength = std::max(0.001f, renderParams.m_timeStripeLength);

	ImGui::Spacing();
	ImGui::Separator();
}

FlowVisToolVolumeData* VolumeDataSelectionCombo(FlowVisTool& flowVisTool, FlowVisToolVolumeData* selected)
{
	bool currentExists = false;
	for (size_t i = 0; i < flowVisTool.g_volumes.size(); i++)
		if (selected == flowVisTool.g_volumes[i])
			currentExists = true;

	if (selected == nullptr || !currentExists)
		selected = flowVisTool.g_volumes.front();

	ImVec4 originalTextCol = ImGui::GetStyleColorVec4(ImGuiCol_Text);

	ImGui::PushStyleColor(ImGuiCol_Text, datasetNameTextColor);
	ImGui::PushItemWidth(-1);
	if (ImGui::BeginCombo("##combo", selected->m_volume->GetName().c_str())) // The second parameter is the label previewed before opening the combo.
	{
		for (int n = 0; n < flowVisTool.g_volumes.size(); n++)
		{
			bool is_selected = (selected == flowVisTool.g_volumes[n]); // You can store your selection however you want, outside or inside your objects

			ImGui::PushStyleColor(ImGuiCol_Text, originalTextCol);
			ImGui::PushID((const void*)flowVisTool.g_volumes[n]);
			if (ImGui::Selectable(flowVisTool.g_volumes[n]->m_volume->GetName().c_str(), is_selected))
				selected = flowVisTool.g_volumes[n];
			ImGui::PopID();
			ImGui::PopStyleColor();

			if (is_selected)
				ImGui::SetItemDefaultFocus();   // You may set the initial focus when opening the combo (scrolling + for keyboard navigation support)
		}
		ImGui::EndCombo();
	}
	ImGui::PopItemWidth();
	ImGui::PopStyleColor();

	return selected;
}


void FlowVisToolGUI::RenderGUI(FlowVisTool& flowVisTool, bool& resizeNextFrame, ImVec2& sceneWindowSize)
{
	if (g_show_demo_window)
		ImGui::ShowDemoWindow(&g_show_demo_window);

	MainMenu(flowVisTool);

	DatasetWindow(flowVisTool);
	TracingWindow(flowVisTool);
	ExtraWindow(flowVisTool);
	//FTLEWindow(flowVisTool);
	HeatmapWindow(flowVisTool);
	TrajectoriesRenderingWindow(flowVisTool);
	GeneralRenderingWindow(flowVisTool);
	RaycastingWindow(flowVisTool);
	SceneWindow(flowVisTool, resizeNextFrame, sceneWindowSize);
	ProfilerWindow(flowVisTool);
	StatusWindow(flowVisTool);

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
	static ImGuiDockNodeFlags opt_flags = ImGuiDockNodeFlags_None;// | ImGuiDockNodeFlags_PassthruInEmptyNodes | ImGuiDockNodeFlags_RenderWindowBg;
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

	//if (ImGui::BeginMainMenuBar())
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
	//}
	//ImGui::EndMainMenuBar();

	ImGui::End();
}

void FlowVisToolGUI::MainMenu(FlowVisTool& flowVisTool)
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
					//flowVisTool.CloseVolumeFile();
					flowVisTool.OpenVolumeFile(filename);
				}
			}
			ImGui::Separator();
			if (ImGui::MenuItem("Save", "CTRL+S")) {}
			ImGui::EndMenu();
		}
		if (ImGui::BeginMenu("View"))
		{
			if (ImGui::MenuItem("Trajectories Rendering Settings", nullptr, g_showTrajectoriesRenderingSettingsWindow))
				g_showTrajectoriesRenderingSettingsWindow = !g_showTrajectoriesRenderingSettingsWindow;
			if (ImGui::MenuItem("General Rendering Settings", nullptr, g_showGeneralRenderingSettingsWindow))
				g_showGeneralRenderingSettingsWindow = !g_showGeneralRenderingSettingsWindow;
			if (ImGui::MenuItem("Tracing Options", nullptr, g_showTracingOptionsWindow))
				g_showTracingOptionsWindow = !g_showTracingOptionsWindow;
			if (ImGui::MenuItem("Dataset", nullptr, g_showDatasetWindow))
				g_showDatasetWindow = !g_showDatasetWindow;
			if (ImGui::MenuItem("Raycasting", nullptr, g_showRaycastingWindow))
				g_showRaycastingWindow = !g_showRaycastingWindow;
			if (ImGui::MenuItem("FTLE", nullptr, g_showFTLEWindow))
				g_showFTLEWindow = !g_showFTLEWindow;
			if (ImGui::MenuItem("Extra", nullptr, g_showExtraWindow))
				g_showExtraWindow = !g_showExtraWindow;
			if (ImGui::MenuItem("Heatmap", nullptr, g_showHeatmapWindow))
				g_showHeatmapWindow = !g_showHeatmapWindow;
			if (ImGui::MenuItem("Status", nullptr, g_showStatusWindow))
				g_showStatusWindow = !g_showStatusWindow;
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

#pragma region Windows
void FlowVisToolGUI::DatasetWindow(FlowVisTool& flowVisTool)
{
	// Dataset window
	if (g_showDatasetWindow)
	{
		ImGui::SetNextWindowSize(ImVec2(300, 300), ImGuiCond_FirstUseEver);
		if (ImGui::Begin("Datasets", &g_showDatasetWindow))
		{
			ImGui::PushItemWidth(-150);
			{
				if (flowVisTool.g_volumes.empty())
				{
					ImGui::Text("No dataset available.");
				}
				else
				{
					for (size_t i = 0; i < flowVisTool.g_volumes.size(); i++)
					{
						assert(flowVisTool.g_volumes[i]->m_volume);

						//SectionText(flowVisTool.g_volumes[i]->m_volume->GetName().c_str());

						ImGui::PushStyleColor(ImGuiCol_Text, datasetNameTextColor);
						ImGui::Text(flowVisTool.g_volumes[i]->m_volume->GetName().c_str());
						ImGui::PopStyleColor();

						ImGui::Spacing();
						ImGui::Separator();

						ImGui::PushItemWidth(0);
						//if (flowVisTool.g_volumes[i]->m_tracingManager.IsTracing())
						{
							float progress = 0.0f;
							if (flowVisTool.g_volumes[i]->m_tracingManager.IsTracing())
								flowVisTool.g_volumes[i]->m_tracingManager.GetTracingProgress();
							ImGui::ProgressBar(progress, ImVec2(0.0f, 0.0f));
							ImGui::SameLine(0.0f, ImGui::GetStyle().ItemInnerSpacing.x);
							ImGui::Text("Tracing");
						}

						//if (flowVisTool.g_volumes[0]->m_volume->GetLoadingProgress() > 0.0f && flowVisTool.g_volumes[0]->m_volume->GetLoadingProgress() < 1.0f)
						{
							ImGui::ProgressBar(flowVisTool.g_volumes[0]->m_volume->GetLoadingProgress(), ImVec2(0.0f, 0.0f));
							ImGui::SameLine(0.0f, ImGui::GetStyle().ItemInnerSpacing.x);
							ImGui::Text("Loading");
						}
						ImGui::PopItemWidth();

						ImGui::Spacing();
						ImGui::Separator();

						ImGui::PushID((const void*)flowVisTool.g_volumes[i]);

						TimeVolGUI(*flowVisTool.g_volumes[i]->m_volume);

						if (ImGui::Button("Close", ImVec2(buttonWidth, 0)))
							flowVisTool.CloseVolumeFile(i);

						ImGui::PopID();

						ImGui::Spacing();
						ImGui::Separator();
					}
				}
			}
			ImGui::PopItemWidth();
		}
		ImGui::End();
	}
}

void FlowVisToolGUI::ExtraWindow(FlowVisTool& flowVisTool)
{
	// Extra window
	if (g_showExtraWindow)
	{
		ImGui::SetNextWindowSize(ImVec2(300, 300), ImGuiCond_FirstUseEver);
		if (ImGui::Begin("Extra", &g_showExtraWindow))
		{
			ImGui::PushItemWidth(-150);
			{
#ifdef Single
				float v = flowVisTool.g_volumes[0]->GetSystemMemoryUsage().GetSystemMemoryLimitMBytes();
				if (ImGui::DragFloat("Mem Usage Limit", &v, 10.0f, 0.0f, FLT_MAX, "%.1f MB"))
					flowVisTool.g_volumes[0]->GetSystemMemoryUsage().SetSystemMemoryLimitMBytes(std::max(0.0f, v));
#endif
				ImGui::Spacing();
				ImGui::Separator();

#ifdef Single
				if (ImGui::Button("Save Traced Lines", ImVec2(buttonWidth, 0)))
					SaveLinesDialog(flowVisTool);
#endif				
				if (ImGui::Button("Load Lines", ImVec2(buttonWidth, 0)))
					LoadLinesDialog(flowVisTool);

				if (ImGui::DragInt("Line ID Override", &g_lineIDOverride, 1, -1, INT_MAX))
					g_lineIDOverride = std::max(-1, g_lineIDOverride);

				if (ImGui::Button("Clear Loaded Lines", ImVec2(buttonWidth, 0)))
				{
					flowVisTool.ReleaseLineBuffers();
					flowVisTool.g_raycastParams.m_redraw = true;
				}

				ImGui::Spacing();
				ImGui::Separator();

				if (ImGui::Button("Load Balls", ImVec2(buttonWidth, 0)))
					LoadBallsDialog(flowVisTool);

				if (ImGui::DragFloat("Ball Radius", &flowVisTool.g_ballRadius, 0.001f, 0.0f, FLT_MAX, "%.3f"))
					flowVisTool.g_ballRadius = std::max(0.0f, flowVisTool.g_ballRadius);

				if (ImGui::Button("Clear Loaded Balls", ImVec2(buttonWidth, 0)))
				{
					flowVisTool.ReleaseBallBuffers();
					flowVisTool.g_raycastParams.m_redraw = true;
				}

				ImGui::Spacing();
				ImGui::Separator();

				if (ImGui::Button("Build Flow Graph", ImVec2(buttonWidth, 0)))
					flowVisTool.BuildFlowGraph("flowgraph.txt");

				if (ImGui::Button("Save Flow Graph", ImVec2(buttonWidth, 0)))
					flowVisTool.SaveFlowGraph();

#ifdef Single
				if (ImGui::Button("Load Flow Graph", ImVec2(buttonWidth, 0)))
					flowVisTool.LoadFlowGraph();
#endif

				ImGui::Spacing();
				ImGui::Separator();

				if (ImGui::Button("Save Settings", ImVec2(buttonWidth, 0)))
					SaveRenderingParamsDialog(flowVisTool);

				if (ImGui::Button("Load Settings", ImVec2(buttonWidth, 0)))
					LoadRenderingParamsDialog(flowVisTool);

				ImGui::Spacing();
				ImGui::Separator();

				//if (ImGui::Button("Save Screenshot", ImVec2(buttonWidth, 0)))
				//	flowVisTool.g_saveScreenshot = true;

				//if (ImGui::Button("Save Renderbuffer", ImVec2(buttonWidth, 0)))
				//	flowVisTool.g_saveRenderBufferScreenshot = true;
			}
			ImGui::PopItemWidth();
		}
		ImGui::End();
	}
}

#ifdef Single
void FlowVisToolGUI::FTLEWindow(FlowVisTool& flowVisTool)
{
	// FTLE window
	if (g_showFTLEWindow)
	{
		ImGui::SetNextWindowSize(ImVec2(300, 300), ImGuiCond_FirstUseEver);
		if (ImGui::Begin("FTLE", &g_showFTLEWindow))
		{
			ImGui::PushItemWidth(-150);
			{
				if (ImGui::Checkbox("Enabled", &flowVisTool.g_particleTraceParams.m_ftleEnabled))
				{
					if (flowVisTool.g_particleTraceParams.m_ftleEnabled)
					{
						//TwDefine("Main/LineCount readonly=true");
						//TwDefine("Main/LineMode readonly=true");
						//TwDefine("Main/LineLengthMax readonly=true");
						//TwDefine("Main/LineLengthMax readonly=true");
						//TwDefine("Main/SeedingPattern readonly=true"); 
						flowVisTool.g_particleTraceParams.m_lineMode = eLineMode::LINE_PATH_FTLE;
						flowVisTool.g_particleTraceParams.m_seedPattern = ParticleTraceParams::eSeedPattern::FTLE;
						flowVisTool.g_particleTraceParams.m_lineLengthMax = 2;
						flowVisTool.g_particleTraceParams.m_lineAgeMax = 0.1f;
						flowVisTool.g_particleTraceParams.m_lineCount = flowVisTool.g_particleTraceParams.m_ftleResolution * flowVisTool.g_particleTraceParams.m_ftleResolution * 6;
					}
					else
					{
						flowVisTool.g_particleTraceParams.m_lineMode = eLineMode::LINE_STREAM;
						flowVisTool.g_particleTraceParams.m_seedPattern = ParticleTraceParams::eSeedPattern::RANDOM;
					}
				}

				ImGui::DragFloat("Scale", &flowVisTool.g_renderingManager.m_ftleScale, 0.001f);

				ImGui::Checkbox("Invert velocity", &flowVisTool.g_particleTraceParams.m_ftleInvertVelocity);

				int u = flowVisTool.g_particleTraceParams.m_ftleResolution;
				if (ImGui::DragInt("Resolution", &u, 1, 64, INT_MAX))
				{
					flowVisTool.g_particleTraceParams.m_ftleResolution = std::max(64, u);
					if (flowVisTool.g_particleTraceParams.m_ftleEnabled)
						flowVisTool.g_particleTraceParams.m_lineCount = flowVisTool.g_particleTraceParams.m_ftleResolution * flowVisTool.g_particleTraceParams.m_ftleResolution * 6;
				}

				ImGui::DragFloat("Slice (Y)", &flowVisTool.g_particleTraceParams.m_ftleSliceY, 0.001f);

				ImGui::SliderFloat("Slice Alpha", &flowVisTool.g_particleRenderParams.m_ftleTextureAlpha, 0.0f, 1.0f);

				ImGui::DragFloat3("Separation Dist", (float*)&flowVisTool.g_particleTraceParams.m_ftleSeparationDistance, 0.0000001f, 0.0f, FLT_MAX, "%.7f");
			}
			ImGui::PopItemWidth();
		}
		ImGui::End();
	}
}
#endif

void FlowVisToolGUI::HeatmapWindow(FlowVisTool& flowVisTool)
{
	// Heatmap window
	if (g_showHeatmapWindow)
	{
		ImGui::SetNextWindowSize(ImVec2(300, 300), ImGuiCond_FirstUseEver);
		if (ImGui::Begin("Heatmap", &g_showHeatmapWindow))
		{
			ImGui::PushItemWidth(-150);
			{
				ImGui::Checkbox("Enable Recording", &flowVisTool.g_heatMapParams.m_enableRecording);
				ImGui::Checkbox("Enable Rendering", &flowVisTool.g_heatMapParams.m_enableRendering);
				ImGui::Checkbox("Auto Reset", &flowVisTool.g_heatMapParams.m_autoReset);

				if (ImGui::Button("Reset", ImVec2(buttonWidth, 0)))
				{
					flowVisTool.g_heatMapManager.ClearChannels();
					flowVisTool.g_raycastParams.m_redraw = true;
				}

				static auto getterNormalizationMode = [](void* data, int idx, const char** out_str)
				{
					if (idx >= HEAT_MAP_NORMALIZATION_MODE_COUNT) return false;
					*out_str = GetHeatMapNormalizationModeName(eHeatMapNormalizationMode(idx));
					return true;
				};
				ImGui::Combo("Normalization", (int*)&flowVisTool.g_heatMapParams.m_normalizationMode, getterNormalizationMode, nullptr, HEAT_MAP_NORMALIZATION_MODE_COUNT);

				if (ImGui::DragFloat("Step Size", &flowVisTool.g_heatMapParams.m_stepSize, 0.001f, 0.001f, FLT_MAX))
					flowVisTool.g_heatMapParams.m_stepSize = std::max(0.001f, flowVisTool.g_heatMapParams.m_stepSize);

				if (ImGui::DragFloat("Density Scale", &flowVisTool.g_heatMapParams.m_densityScale, 0.01f, 0.0f, FLT_MAX))
					flowVisTool.g_heatMapParams.m_densityScale = std::max(0.0f, flowVisTool.g_heatMapParams.m_densityScale);

				ImGui::ColorEdit3("First displayed channel (1)", (float*)&flowVisTool.g_heatMapParams.m_renderedChannels[0]);
				ImGui::ColorEdit3("Second displayed channel (2)", (float*)&flowVisTool.g_heatMapParams.m_renderedChannels[1]);

				ImGui::Checkbox("Isosurface Rendering", &flowVisTool.g_heatMapParams.m_isosurface);

				ImGui::SliderFloat("Isovalue", &flowVisTool.g_heatMapParams.m_isovalue, 0.0f, 1.0f);
			}
			ImGui::PopItemWidth();
		}
		ImGui::End();
	}
}

void FlowVisToolGUI::TracingWindow(FlowVisTool& flowVisTool)
{
	// Particle tracing config window
	if (g_showTracingOptionsWindow)
	{
		ImGui::SetNextWindowSize(ImVec2(300, 300), ImGuiCond_FirstUseEver);
		if (ImGui::Begin("Tracing Options", &g_showTracingOptionsWindow))
		{
			ImGui::PushItemWidth(-150);
			if (flowVisTool.g_volumes.empty())
			{
				ImGui::Text("No dataset available.");
			}
			else
			{
				static FlowVisToolVolumeData* selected = nullptr;

				selected = VolumeDataSelectionCombo(flowVisTool, selected);

				ImGui::Spacing();
				ImGui::Separator();

				TraceParamsGUI(selected);
			}
			ImGui::PopItemWidth();
		}
		ImGui::End();
	}
}

void FlowVisToolGUI::TrajectoriesRenderingWindow(FlowVisTool& flowVisTool)
{
	// Rendering config window
	if (g_showTrajectoriesRenderingSettingsWindow)
	{
		ImGui::SetNextWindowSize(ImVec2(300, 300), ImGuiCond_FirstUseEver);
		if (ImGui::Begin("Trajectories Rendering Settings", &g_showTrajectoriesRenderingSettingsWindow))
		{
			ImGui::PushItemWidth(-150);
			if (flowVisTool.g_volumes.empty())
			{
				ImGui::Text("No dataset available.");
			}
			else
			{
				static FlowVisToolVolumeData* selected = nullptr;

				selected = VolumeDataSelectionCombo(flowVisTool, selected);

				ImGui::Spacing();
				ImGui::Separator();

				TrajectoriesRenderingParamsGUI(selected, flowVisTool.m_d3dDevice);
			}
			ImGui::PopItemWidth();
		}
		ImGui::End();
	}
}

void FlowVisToolGUI::GeneralRenderingWindow(FlowVisTool& flowVisTool)
{
	// Rendering config window
	if (g_showGeneralRenderingSettingsWindow)
	{
		ImGui::SetNextWindowSize(ImVec2(300, 300), ImGuiCond_FirstUseEver);
		if (ImGui::Begin("General Rendering Settings", &g_showGeneralRenderingSettingsWindow))
		{
			ImGui::PushItemWidth(-150);
			{
				if (ImGui::DragFloat("Domain Box Thickness", &flowVisTool.g_renderingManager.m_DomainBoxThickness, 0.0001f, 0.0f, FLT_MAX, "%.4f"))
				{
					flowVisTool.g_renderingManager.m_DomainBoxThickness = std::max(0.0f, flowVisTool.g_renderingManager.m_DomainBoxThickness);
				}

				ImGui::ColorEdit4("Background color", (float*)&flowVisTool.g_renderingParams.m_backgroundColor);

				ImGui::Checkbox("Fixed Light Dir", &flowVisTool.g_renderingParams.m_FixedLightDir);

				ImGui::SliderFloat3("Light Dir", (float*)&flowVisTool.g_renderingParams.m_lightDir, -1.0f, 1.0f);

				int f = flowVisTool.g_renderingParams.m_renderBufferSizeFactor;
				if (ImGui::SliderInt("SuperSample Factor", &f, 1.0f, 8.0f))
					flowVisTool.g_renderingParams.m_renderBufferSizeFactor = f;

				if (ImGui::Checkbox("Perspective", &flowVisTool.g_projParams.m_perspective))
				{
					if (flowVisTool.g_projParams.m_perspective)
						flowVisTool.g_projParams.m_fovy = 30.0f * PI / 180.0f; // this should be 24 deg, but a bit larger fov looks better...
					else
						flowVisTool.g_projParams.m_fovy = 3.1f;
				}

				float degfovy = flowVisTool.g_projParams.m_fovy * 180.0f / PI;
				if (ImGui::SliderFloat("FoVY", &degfovy, 1.0f, 180.0f, "%.2f deg"))
					flowVisTool.g_projParams.m_fovy = degfovy * PI / 180.0f;

				ImGui::Checkbox("Stereo", &flowVisTool.g_stereoParams.m_stereoEnabled);

				if (ImGui::DragFloat("Eye Distance", &flowVisTool.g_stereoParams.m_eyeDistance, 0.001f, 0.0f, FLT_MAX, "%.3f"))
					flowVisTool.g_stereoParams.m_eyeDistance = std::max(0.0f, flowVisTool.g_stereoParams.m_eyeDistance);

#ifdef Si
				SectionText("Slice");
				if (ImGui::Button("Load Slice Texture", ImVec2(buttonWidth, 0)))
					LoadSliceTexture(flowVisTool);

				ImGui::Checkbox("Show Slice", &renderParams.m_showSlice);

				ImGui::DragFloat("Slice Position", &renderParams.m_slicePosition, 0.001f);

				if (ImGui::DragFloat("Slice Transparency", &renderParams.m_sliceAlpha, 0.001f, 0.0f, 1.0f, "%.3f"))
					renderParams.m_sliceAlpha = std::min(1.0f, std::max(0.0f, renderParams.m_sliceAlpha));
#endif
			}
			ImGui::PopItemWidth();
		}
		ImGui::End();
	}
}

void FlowVisToolGUI::RaycastingWindow(FlowVisTool& flowVisTool)
{
	if (g_showRaycastingWindow)
	{
		ImGui::SetNextWindowSize(ImVec2(300, 300), ImGuiCond_FirstUseEver);
		if (ImGui::Begin("Raycasting", &g_showRaycastingWindow))
		{
			ImGui::PushItemWidth(-150);
			if (flowVisTool.g_volumes.empty())
			{
				ImGui::Text("No dataset available.");
			}
			else
			{
				FlowVisToolVolumeData* current = nullptr;

				if (flowVisTool.GetSelectedvolume() >= 0 && flowVisTool.GetSelectedvolume() < flowVisTool.g_volumes.size())
					current = flowVisTool.g_volumes[flowVisTool.GetSelectedvolume()];

				FlowVisToolVolumeData* selected = VolumeDataSelectionCombo(flowVisTool, current);

				if (current != selected)
				{
					for (size_t i = 0; i < flowVisTool.g_volumes.size(); i++)
					{
						if (flowVisTool.g_volumes[i] == selected)
							flowVisTool.SetSelectedVolume(i);
					}
				}

				ImGui::Spacing();
				ImGui::Separator();

				ImGui::PushItemWidth(0);
				if (flowVisTool.g_raycasterManager.IsRendering())
				{
					ImGui::ProgressBar(flowVisTool.g_raycasterManager.GetRenderingProgress(), ImVec2(0.0f, 0.0f));
					ImGui::SameLine(0.0f, ImGui::GetStyle().ItemInnerSpacing.x);
					ImGui::Text("Raycasting");
				}
				ImGui::PopItemWidth();

				ImGui::Checkbox("Enabled", &flowVisTool.g_raycastParams.m_raycastingEnabled);

				if (ImGui::Button("Redraw", ImVec2(buttonWidth, 0)))
					flowVisTool.g_raycastParams.m_redraw = true;

				int v = flowVisTool.g_raycasterManager.m_bricksPerFrame;
				if (ImGui::SliderInt("Bricks per frame", &v, 1, 20))
					flowVisTool.g_raycasterManager.m_bricksPerFrame = (unsigned int)v;

				static auto getterMeasureComputeMode = [](void* data, int idx, const char** out_str)
				{
					if (idx >= MEASURE_COMPUTE_COUNT) return false;
					*out_str = GetMeasureComputeModeName(eMeasureComputeMode(idx));
					return true;
				};
				ImGui::Combo("Measure Computation", (int*)&flowVisTool.g_raycastParams.m_measureComputeMode, getterMeasureComputeMode, nullptr, MEASURE_COMPUTE_COUNT);

				static auto getterFilterModeRayCaster = [](void* data, int idx, const char** out_str)
				{
					//Raycaster only supports the first two modes, linear and cubic
					if (idx >= 2) return false;
					*out_str = GetTextureFilterModeName(eTextureFilterMode(idx));
					return true;
				};
				ImGui::Combo("Interpolation", (int*)&flowVisTool.g_raycastParams.m_textureFilterMode, getterFilterModeRayCaster, nullptr, 2);

				static auto getterRaycastMode = [](void* data, int idx, const char** out_str)
				{
					if (idx >= RAYCAST_MODE_COUNT) return false;
					*out_str = GetRaycastModeName(eRaycastMode(idx));
					return true;
				};
				ImGui::Combo("Raycast Mode", (int*)&flowVisTool.g_raycastParams.m_raycastMode, getterRaycastMode, nullptr, RAYCAST_MODE_COUNT);

				static auto getterMeasureMode = [](void* data, int idx, const char** out_str)
				{
					if (idx >= MEASURE_COUNT) return false;
					*out_str = GetMeasureName(eMeasure(idx));
					return true;
				};

				if (ImGui::DragFloat("Sample Rate", &flowVisTool.g_raycastParams.m_sampleRate, 0.01f, 0.01f, 20.0f, "%.2f"))
					flowVisTool.g_raycastParams.m_sampleRate = std::min(20.0f, std::max(0.01f, flowVisTool.g_raycastParams.m_sampleRate));

				if (ImGui::DragFloat("Density", &flowVisTool.g_raycastParams.m_density, 0.1f, 0.01f, 1000.0f, "%.1f"))
					flowVisTool.g_raycastParams.m_density = std::min(1000.0f, std::max(0.1f, flowVisTool.g_raycastParams.m_density));

				
				ImGui::Spacing();
				ImGui::Separator();


				ImGui::PushItemWidth(45);
				ImGui::PushID(1);
				ImGui::DragFloat("", &flowVisTool.g_raycastParams.m_measureScale1, 0.001f, 0.0f, 0.0f, "%.3f");
				ImGui::PopID();
				ImGui::PopItemWidth();

				ImGui::SameLine();
				if (ImGui::Combo("Measure 1", (int*)&flowVisTool.g_raycastParams.m_measure1, getterMeasureMode, nullptr, MEASURE_COUNT))
					flowVisTool.g_raycastParams.m_measureScale1 = GetDefaultMeasureScale(flowVisTool.g_raycastParams.m_measure1);
				
	
				ImGui::PushItemWidth(45);
				ImGui::PushID(3);
				ImGui::DragFloat("", &flowVisTool.g_raycastParams.m_measureScale2, 0.001f, 0.0f, 0.0f, "%.3f");
				ImGui::PopID();
				ImGui::PopItemWidth();

				ImGui::SameLine();
				if (ImGui::Combo("Measure 2", (int*)&flowVisTool.g_raycastParams.m_measure2, getterMeasureMode, nullptr, MEASURE_COUNT))
					flowVisTool.g_raycastParams.m_measureScale2 = GetDefaultMeasureScale(flowVisTool.g_raycastParams.m_measure2);
				

				ImGui::Spacing();
				ImGui::Separator();
				ImGui::Text("Iso Settings");

				ImGui::ColorEdit4("Color1", (float*)&flowVisTool.g_raycastParams.m_isoColor1, ImGuiColorEditFlags_::ImGuiColorEditFlags_NoInputs | ImGuiColorEditFlags_::ImGuiColorEditFlags_NoLabel);
				ImGui::SameLine();
				ImGui::DragFloat("IsoValue 1", &flowVisTool.g_raycastParams.m_isoValue1, 0.001f, 0.0f, 0.0f, "%.4f");
				
				ImGui::ColorEdit4("Color2", (float*)&flowVisTool.g_raycastParams.m_isoColor2, ImGuiColorEditFlags_::ImGuiColorEditFlags_NoInputs | ImGuiColorEditFlags_::ImGuiColorEditFlags_NoLabel);
				ImGui::SameLine();
				ImGui::DragFloat("IsoValue 2", &flowVisTool.g_raycastParams.m_isoValue2, 0.001f, 0.0f, 0.0f, "%.4f");
				
				ImGui::ColorEdit4("Color3", (float*)&flowVisTool.g_raycastParams.m_isoColor3, ImGuiColorEditFlags_::ImGuiColorEditFlags_NoInputs | ImGuiColorEditFlags_::ImGuiColorEditFlags_NoLabel);
				ImGui::SameLine();
				ImGui::DragFloat("IsoValue 3", &flowVisTool.g_raycastParams.m_isoValue3, 0.001f, 0.0f, 0.0f, "%.4f");
				
				ImGui::Spacing();

				static auto getterColorMode = [](void* data, int idx, const char** out_str)
				{
					if (idx >= COLOR_MODE_COUNT) return false;
					*out_str = GetColorModeName(eColorMode(idx));
					return true;
				};
				ImGui::Combo("Color Mode", (int*)&flowVisTool.g_raycastParams.m_colorMode, getterColorMode, nullptr, COLOR_MODE_COUNT);

				ImGui::Spacing();
				ImGui::Separator();
				ImGui::Text("Filter");

				v = flowVisTool.g_filterParams.m_radius[0];
				if (ImGui::SliderInt("Radius 1", &v, 0, 64))
					flowVisTool.g_filterParams.m_radius[0] = (unsigned int)v;

				v = flowVisTool.g_filterParams.m_radius[1];
				if (ImGui::SliderInt("Radius 2", &v, 0, 64))
					flowVisTool.g_filterParams.m_radius[1] = (unsigned int)v;

				v = flowVisTool.g_filterParams.m_radius[2];
				if (ImGui::SliderInt("Radius 3", &v, 0, 64))
					flowVisTool.g_filterParams.m_radius[2] = (unsigned int)v;

				ImGui::Spacing();

				v = flowVisTool.g_raycastParams.m_filterOffset;
				if (ImGui::SliderInt("Offset (Scale)", &v, 0, 2))
					flowVisTool.g_raycastParams.m_filterOffset = (unsigned int)v;

				ImGui::Spacing();
				ImGui::Separator();

				//ImGui::Checkbox("Show Clip Box", &flowVisTool.g_renderingManager.m_renderClipBox);

				ImGui::DragFloat3("ClipBoxMin", (float*)&flowVisTool.g_raycastParams.m_clipBoxMin, 0.005f, 0.0f, 0.0f, "%.3f");
				ImGui::DragFloat3("ClipBoxMax", (float*)&flowVisTool.g_raycastParams.m_clipBoxMax, 0.005f, 0.0f, 0.0f, "%.3f");
			}
			ImGui::PopItemWidth();
		}
		ImGui::End();
	}
}

void FlowVisToolGUI::SceneWindow(FlowVisTool& flowVisTool, bool& resizeNextFrame, ImVec2& sceneWindowSize)
{
	// Scene view window
	ImGui::PushStyleVar(ImGuiStyleVar_::ImGuiStyleVar_WindowPadding, ImVec2(0, 0));
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

		if (windowAspectRatio < flowVisTool.g_renderTexture.GetAspectRatio())
		{
			width = availableRegion.x - 2;
			height = width * 1.0f / flowVisTool.g_renderTexture.GetAspectRatio();
			ImGui::SetCursorPosY(ImGui::GetCursorPos().y + (availableRegion.y - height) / 2.0f);
		}
		else
		{
			height = availableRegion.y - 2;
			width = height * flowVisTool.g_renderTexture.GetAspectRatio();
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
		}
		ImGui::End();

		bool userInteraction = false;

		// ImageButton prevents mouse dragging from moving the window as well.
		ImGui::ImageButton((void *)(intptr_t)flowVisTool.g_renderTexture.GetShaderResourceView(), ImVec2(width, height), ImVec2(0, 0), ImVec2(1, 1), 0, ImColor(0, 0, 0, 255), ImColor(255, 255, 255, 255));
		//ImGui::Image((void *)(intptr_t)g_renderTexture.GetShaderResourceView(), ImVec2(width, height), ImVec2(0, 0), ImVec2(1, 1), ImColor(255, 255, 255, 255), ImColor(255, 255, 255, 25));

		if (ImGui::IsItemHovered(ImGuiHoveredFlags_::ImGuiHoveredFlags_None))
			//if (ImGui::IsWindowHovered(ImGuiHoveredFlags_::ImGuiHoveredFlags_None))
		{
			// Zoom
			flowVisTool.g_viewParams.m_viewDistance -= ImGui::GetIO().MouseWheel * ImGui::GetIO().DeltaTime * zoomSens * flowVisTool.g_viewParams.m_viewDistance;
			flowVisTool.g_viewParams.m_viewDistance = std::max(0.0001f, flowVisTool.g_viewParams.m_viewDistance);

			// Orbit
			if (ImGui::IsMouseDragging(0))
			{
				if (ImGui::GetIO().MouseDelta.x != 0 || ImGui::GetIO().MouseDelta.y != 0)
				{
					userInteraction = true;

					tum3D::Vec2d normDelta = tum3D::Vec2d((double)ImGui::GetIO().MouseDelta.x / (double)flowVisTool.g_renderingParams.m_windowSize.x(), (double)ImGui::GetIO().MouseDelta.y / (double)flowVisTool.g_renderingParams.m_windowSize.y());

					tum3D::Vec2d delta = normDelta * (double)ImGui::GetIO().DeltaTime * (double)orbitSens;

					// Don't trust this code. Seriously, I have no idea how this is working.
					tum3D::Vec4f rotationX;

					tum3D::Vec3f up;
					tum3D::rotateVecByQuaternion(tum3D::Vec3f(0.0f, 0.0f, 1.0f), flowVisTool.g_viewParams.m_rotationQuat, up);
					up = tum3D::normalize(up);

					tum3D::rotationQuaternion((float)(up.y() < 0.0f ? -delta.x() : delta.x()) * PI, up, rotationX);
					//tum3D::rotationQuaternion(delta.x() * PI, Vec3f(0.0f, 0.0f, 1.0f), rotationX);

					tum3D::Vec4f rotation = tum3D::Vec4f(1, 0, 0, 0);

					tum3D::multQuaternion(rotationX, flowVisTool.g_viewParams.m_rotationQuat, rotation); flowVisTool.g_viewParams.m_rotationQuat = rotation;

					tum3D::Vec4f rotationY;
					tum3D::rotationQuaternion((float)delta.y() * PI, tum3D::Vec3f(1.0f, 0.0f, 0.0f), rotationY);

					tum3D::multQuaternion(rotationY, flowVisTool.g_viewParams.m_rotationQuat, rotation); flowVisTool.g_viewParams.m_rotationQuat = rotation;
				}
			}

			// Pan on xy plane
			if (ImGui::IsMouseDragging(2))
			{
				if (ImGui::GetIO().MouseDelta.x != 0 || ImGui::GetIO().MouseDelta.y != 0)
				{
					userInteraction = true;

					tum3D::Vec2d normDelta = tum3D::Vec2d((double)ImGui::GetIO().MouseDelta.x / (double)flowVisTool.g_renderingParams.m_windowSize.x(), (double)ImGui::GetIO().MouseDelta.y / (double)flowVisTool.g_renderingParams.m_windowSize.y());

					tum3D::Vec2d delta = normDelta * (double)ImGui::GetIO().DeltaTime * (double)flowVisTool.g_viewParams.m_viewDistance * (double)panSens;

					tum3D::Vec2f target = flowVisTool.g_viewParams.m_lookAt.xy();

					tum3D::Vec2f right = flowVisTool.g_viewParams.GetRightVector().xy(); right = tum3D::normalize(right);
					target = target - right * delta.x();

					tum3D::Vec2f forward = flowVisTool.g_viewParams.GetViewDir().xy(); forward = tum3D::normalize(forward);

					if (forward.x() == 0.0f && forward.y() == 0.0f)
					{
						tum3D::Vec3f for3d;
						tum3D::crossProd(tum3D::Vec3f(0.0f, 0.0f, -flowVisTool.g_viewParams.GetViewDir().z()), tum3D::Vec3f(right.x(), right.y(), 0.0f), for3d);
						forward = for3d.xy();
					}

					target = target - forward * delta.y();

					flowVisTool.g_viewParams.m_lookAt.x() = target.x();
					flowVisTool.g_viewParams.m_lookAt.y() = target.y();
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

		static bool isHelperWindowHovered = false;

		ImGui::SetNextWindowBgAlpha(isHelperWindowHovered ? 0.75f : 0.15f);
		
		ImGui::PushStyleColor(ImGuiCol_::ImGuiCol_Border, ImVec4(0.0f, 0.0f, 0.0f, 0.0f));

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
			ImVec4 sColor = ImGui::GetStyle().Colors[ImGuiCol_::ImGuiCol_Separator];
			

			if (!isHelperWindowHovered) { bColor.w = 0.15f; tColor.w = 0.15f; fColor.w = 0.15f; sColor.w = 0.15f; }

			ImGui::PushStyleColor(ImGuiCol_::ImGuiCol_Button, bColor);
			ImGui::PushStyleColor(ImGuiCol_::ImGuiCol_Text, tColor);
			ImGui::PushStyleColor(ImGuiCol_::ImGuiCol_FrameBg, fColor);
			ImGui::PushStyleColor(ImGuiCol_::ImGuiCol_Separator, sColor);
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
				ImGui::SameLine();
				ImGui::PushID(0);
				if (ImGui::Button("Reset", bsize))
					tum3D::convertRotMatToQuaternion(makeRotationFromDir(tum3D::normalize(tum3D::Vec3f(0.5f, 0.5f, 0.5f))), flowVisTool.g_viewParams.m_rotationQuat);
				ImGui::PopID();

				ImGui::Separator();

				ImGui::PushItemWidth(100);
				ImGui::DragFloat3("Pivot", (float*)&flowVisTool.g_viewParams.m_lookAt, 0.01f, 0.0f, 0.0f, "%.2f");
				ImGui::PopItemWidth();
				ImGui::SameLine();
				ImGui::PushID(1);
				if (ImGui::Button("Reset", ImVec2(-1, 0)))
					flowVisTool.g_viewParams.m_lookAt = tum3D::Vec3f(0.0f, 0.0f, 0.0f);
				ImGui::PopID();

				ImGui::Separator();
				ImGui::PushItemWidth(100);
				if (ImGui::DragFloat("Dist  ", &flowVisTool.g_viewParams.m_viewDistance, 0.1f))
					flowVisTool.g_viewParams.m_viewDistance = std::max(0.0001f, flowVisTool.g_viewParams.m_viewDistance);
				ImGui::PopItemWidth();
				ImGui::SameLine();
				ImGui::PushID(2);
				if (ImGui::Button("Reset", ImVec2(-1, 0)))
					flowVisTool.g_viewParams.m_viewDistance = 5.0f;
				ImGui::PopID();
			}
			ImGui::PopStyleColor(4);

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
				flowVisTool.g_viewParams.m_rotationQuat = tum3D::slerpQuaternion(rotInterpSpeed * ImGui::GetIO().DeltaTime, flowVisTool.g_viewParams.m_rotationQuat, targetQuat, res);

				if (std::abs(tum3D::dotProd(targetQuat, flowVisTool.g_viewParams.m_rotationQuat)) > 1.0f - 0.000001f) // Epsilon
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

			isHelperWindowHovered = ImGui::IsWindowHovered();
		}
		ImGui::End();
		ImGui::PopStyleColor();
	}
	ImGui::End();
	ImGui::PopStyleVar(ImGuiStyleVar_::ImGuiStyleVar_WindowPadding);
}

void FlowVisToolGUI::ProfilerWindow(FlowVisTool& flowVisTool)
{
	currentFrameTimeIndex = (currentFrameTimeIndex + 1) % frameTimes.size();
	frameTimes[currentFrameTimeIndex] = ImGui::GetIO().DeltaTime * 1000.0f;

	if (g_showProfilerWindow)
	{
		ImGui::SetNextWindowSize(ImVec2(300, 300), ImGuiCond_FirstUseEver);
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
			ImGui::PlotLines("ms", &frameTimes[0], frameTimes.size(), currentFrameTimeIndex, "", 0.0f, max, ImVec2(0, size.y));
			ImGui::PopItemWidth();

			ImGui::Separator();
			ImGui::Spacing();

			//ImGui::Text("Window resolution: %dx%d", flowVisTool.g_renderingParams.m_windowSize.x(), flowVisTool.g_renderingParams.m_windowSize.y());
			ImGui::Text("Viewport resolution: %dx%d", flowVisTool.g_projParams.m_imageWidth, flowVisTool.g_projParams.m_imageHeight);
			//ImGui::Text("Trace Time: %.2f ms", flowVisTool.g_timerTracing.GetElapsedTimeMS());
			//ImGui::Text("Render Time: %.2f ms", flowVisTool.g_timerRendering.GetElapsedTimeMS());
		}
		ImGui::End();
	}
}

void FlowVisToolGUI::StatusWindow(FlowVisTool& flowVisTool)
{
	if (g_showStatusWindow)
	{
		ImGui::SetNextWindowSize(ImVec2(300, 300), ImGuiCond_FirstUseEver);
		if (ImGui::Begin("Status"))
		{
			if (flowVisTool.g_filteringManager.IsFiltering())
			{
				ImGui::ProgressBar(flowVisTool.g_filteringManager.GetFilteringProgress(), ImVec2(0.0f, 0.0f));
				ImGui::SameLine(0.0f, ImGui::GetStyle().ItemInnerSpacing.x);
				ImGui::Text("Filtering");
			}

			if (flowVisTool.g_raycasterManager.IsRendering())
			{
				ImGui::ProgressBar(flowVisTool.g_raycasterManager.GetRenderingProgress(), ImVec2(0.0f, 0.0f));
				ImGui::SameLine(0.0f, ImGui::GetStyle().ItemInnerSpacing.x);
				ImGui::Text("Raycasting");
			}

			//if (g_useAllGPUs)
			//{
			//	for (size_t i = 0; i < g_cudaDevices.size(); i++)
			//	{
			//		if (i == g_primaryCudaDeviceIndex) continue;

			//		progress = min(progress, g_cudaDevices[i].pThread->GetProgress());
			//	}
			//}
		}
		ImGui::End();
	}
}
#pragma endregion