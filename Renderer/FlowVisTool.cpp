#include <FlowVisTool.h>

#include <cudaUtil.h>
#include <CSysTools.h>
#include <winerror.h>

#include <ctime>
#include <string>

#include <imgui\imgui.h>

FlowVisTool::FlowVisTool()
{
}

bool FlowVisTool::Initialize(ID3D11Device* d3dDevice, ID3D11DeviceContext* d3dDeviceContex, const std::vector<MyCudaDevice>& cudaDevices)
{
	m_d3dDevice = d3dDevice;
	m_d3dDeviceContex = d3dDeviceContex;
	//m_mainRenderTargetView = mainRenderTargetView;
	g_cudaDevices = cudaDevices;

	if (!InitCudaDevices())
	{
		printf("InitCudaDevices returned false");
		return false;
	}

	// don't page-lock brick data - CUDA doesn't seem to like when we page-lock lots of smallish mem areas..
	//g_volume.SetPageLockBrickData(true);

	HRESULT hr;

	if (FAILED(hr = g_screenEffect.Create(m_d3dDevice)))
		return false;

	if (g_volume.IsOpen())
	{
		// this creates the cudaCompress instance etc
		if (!CreateVolumeDependentResources())
			return false;
	}

	//if(FAILED(hr = g_tfEdt.onCreateDevice( pd3dDevice ))) {
	//	return hr;
	//}
	//g_particleRenderParams.m_pTransferFunction = g_tfEdt.getSRV(TF_LINE_MEASURES);
	//g_heatMapParams.m_pTransferFunction = g_tfEdt.getSRV(TF_HEAT_MAP);
	//cudaSafeCall(cudaGraphicsD3D11RegisterResource(&g_pTfEdtSRVCuda, g_tfEdt.getTexture(TF_RAYTRACE), cudaGraphicsRegisterFlagsNone));

	return true;
}

void FlowVisTool::Release()
{
	CloseVolumeFile();

	if (g_pTfEdtSRVCuda != nullptr)
	{
		cudaSafeCall(cudaGraphicsUnregisterResource(g_pTfEdtSRVCuda));
		g_pTfEdtSRVCuda = nullptr;
	}
	//g_tfEdt.onDestroyDevice();

	//TwTerminate();
	//g_pTwBarImageSequence = nullptr;
	//g_pTwBarMain = nullptr;


	ReleaseVolumeDependentResources();

	ShutdownCudaDevices();

	g_screenEffect.SafeRelease();


	cudaSafeCall(cudaDeviceSynchronize());
	g_volume.SetPageLockBrickData(false);
	g_renderingManager.m_ftleTexture.UnregisterCudaResources();
	cudaSafeCallNoSync(cudaDeviceReset());

	//release slice texture
	SAFE_RELEASE(g_particleRenderParams.m_pSliceTexture);
	SAFE_RELEASE(g_particleRenderParams.m_pColorTexture);
	g_renderingManager.m_ftleTexture.ReleaseResources();
}

void FlowVisTool::SetBoundingBoxToDomainSize()
{
	g_particleTraceParams.m_seedBoxMin = -g_volume.GetVolumeHalfSizeWorld();
	g_particleTraceParams.m_seedBoxSize = 2 * g_volume.GetVolumeHalfSizeWorld();
}


void FlowVisTool::BuildFlowGraph(const std::string& filenameTxt)
{
	//TODO might need to cancel tracing first?
	uint advectStepsPerRoundBak = g_particleTraceParams.m_advectStepsPerRound;
	g_particleTraceParams.m_advectStepsPerRound = 256;
	g_flowGraph.Build(&g_compressShared, &g_compressVolume, 1024, g_particleTraceParams, filenameTxt);
	g_particleTraceParams.m_advectStepsPerRound = advectStepsPerRoundBak;
}

bool FlowVisTool::SaveFlowGraph()
{
	std::string filename = tum3d::RemoveExt(g_volume.GetFilename()) + ".fg";
	return g_flowGraph.SaveToFile(filename);
}

bool FlowVisTool::LoadFlowGraph()
{
	std::string filename = tum3d::RemoveExt(g_volume.GetFilename()) + ".fg";
	return g_flowGraph.LoadFromFile(filename);
}

void FlowVisTool::LoadOrBuildFlowGraph()
{
	if (!LoadFlowGraph()) {
		BuildFlowGraph();
		SaveFlowGraph();
	}
}


void FlowVisTool::ReleaseLineBuffers()
{
	for (size_t i = 0; i < g_lineBuffers.size(); i++)
	{
		delete g_lineBuffers[i];
	}
	g_lineBuffers.clear();
}

void FlowVisTool::ReleaseBallBuffers()
{
	for (size_t i = 0; i < g_ballBuffers.size(); i++)
	{
		delete g_ballBuffers[i];
	}
	g_ballBuffers.clear();
}


void FlowVisTool::ReleaseVolumeDependentResources()
{
	g_flowGraph.Shutdown();

	if (g_useAllGPUs)
	{
		for (size_t i = 0; i < g_cudaDevices.size(); i++)
		{
			if (g_cudaDevices[i].pThread)
			{
				g_cudaDevices[i].pThread->CancelCurrentTask();
			}
		}
		for (size_t i = 0; i < g_cudaDevices.size(); i++)
		{
			if (g_cudaDevices[i].pThread)
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

bool FlowVisTool::CreateVolumeDependentResources()
{
	std::cout << "Creating volume dependent reources..." << std::endl;

	ReleaseVolumeDependentResources();

	if (g_volume.IsCompressed())
	{
		uint brickSize = g_volume.GetBrickSizeWithOverlap();
		// do multi-channel decoding only for small bricks; for large bricks, mem usage gets too high
		uint channelCount = (brickSize <= 128) ? g_volume.GetChannelCount() : 1;
		uint huffmanBits = g_volume.GetHuffmanBitsMax();
		g_compressShared.create(CompressVolumeResources::getRequiredResources(brickSize, brickSize, brickSize, channelCount, huffmanBits));
		g_compressVolume.create(g_compressShared.getConfig());
	}

	HRESULT hr;

	if (!g_filteringManager.Create(&g_compressShared, &g_compressVolume))
		return false;

	if (!g_tracingManager.Create(&g_compressShared, &g_compressVolume, m_d3dDevice))
		return false;

	if (!g_renderingManager.Create(&g_compressShared, &g_compressVolume, m_d3dDevice))
		return false;
	
	if (!g_heatMapManager.Create(&g_compressShared, &g_compressVolume, m_d3dDevice))
		return false;


	if (g_useAllGPUs)
	{
		for (size_t i = 0; i < g_cudaDevices.size(); i++)
		{
			if (g_cudaDevices[i].pThread)
			{
				g_cudaDevices[i].pThread->CreateVolumeDependentResources();
			}
		}
	}

	g_flowGraph.Init(g_volume);

	std::cout << "Volume dependent reources created." << std::endl;

	return true;
}




void FlowVisTool::CloseVolumeFile()
{
	ReleaseVolumeDependentResources();
	g_volume.Close();
}

bool FlowVisTool::OpenVolumeFile(const std::string& filename)
{
	CloseVolumeFile();


	if (!g_volume.Open(filename))
		return false;

	// recreate brick slots and re-init cudaCompress - brick size may have changed
	CreateVolumeDependentResources();


	int32 timestepMax = g_volume.GetTimestepCount() - 1;
	float timeSpacing = g_volume.GetTimeSpacing();
	float timeMax = timeSpacing * float(timestepMax);
	//TwSetParam(g_pTwBarMain, "Time", "max", TW_PARAM_FLOAT, 1, &timeMax);
	//TwSetParam(g_pTwBarMain, "Time", "step", TW_PARAM_FLOAT, 1, &timeSpacing);
	//TwSetParam(g_pTwBarMain, "Timestep", "max", TW_PARAM_INT32, 1, &timestepMax);


	g_volume.SetCurTime(0.0f);
	m_redraw = true;
	m_retrace = true;

	g_raycastParams.m_clipBoxMin = -g_volume.GetVolumeHalfSizeWorld();
	g_raycastParams.m_clipBoxMax = g_volume.GetVolumeHalfSizeWorld();


	g_imageSequence.FrameCount = g_volume.GetTimestepCount();

	LoadFlowGraph();

	SetBoundingBoxToDomainSize();

	return true;
}

bool FlowVisTool::ResizeRenderBuffer()
{
	g_projParams.m_imageWidth = uint(g_windowSize.x() * g_renderBufferSizeFactor);
	g_projParams.m_imageHeight = uint(g_windowSize.y() * g_renderBufferSizeFactor);
	g_projParams.m_aspectRatio = float(g_projParams.m_imageWidth) / float(g_projParams.m_imageHeight);


	SAFE_RELEASE(g_pRenderBufferTempRTV);
	SAFE_RELEASE(g_pRenderBufferTempSRV);
	SAFE_RELEASE(g_pRenderBufferTempTex);
	SAFE_RELEASE(g_pRenderBufferStagingTex);


	if (g_projParams.m_imageWidth * g_projParams.m_imageHeight > 0 && m_d3dDevice)
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
		hr = m_d3dDevice->CreateTexture2D(&desc, nullptr, &g_pRenderBufferStagingTex);
		if (FAILED(hr)) return false;

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
		hr = m_d3dDevice->CreateTexture2D(&desc, nullptr, &g_pRenderBufferTempTex);
		if (FAILED(hr)) return false;
		hr = m_d3dDevice->CreateShaderResourceView(g_pRenderBufferTempTex, nullptr, &g_pRenderBufferTempSRV);
		if (FAILED(hr)) return false;
		hr = m_d3dDevice->CreateRenderTargetView(g_pRenderBufferTempTex, nullptr, &g_pRenderBufferTempRTV);
		if (FAILED(hr)) return false;
	}

	return true;
}

void FlowVisTool::OnFrame(float deltatime)
{
	if (ImGui::Begin("Debug"))
	{
		bool b;
		b = s_isFiltering; ImGui::Checkbox("IsFiltering", &b);
		b = s_isTracing; ImGui::Checkbox("IsTracing", &b);
		b = s_isRendering; ImGui::Checkbox("IsRendering", &b);
		b = m_redraw; ImGui::Checkbox("Redraw", &b);
		b = m_retrace; ImGui::Checkbox("Retrace", &b);
		b = g_showPreview; ImGui::Checkbox("ShowPreview", &b);
	}
	ImGui::End();

	CheckForChanges();

	if (g_volume.IsOpen())
	{
		Filtering();

		Tracing();

		Rendering();
	}

	// copy last finished image into back buffer
	//ID3D11Resource* pSwapChainTex;
	//m_mainRenderTargetView->GetResource(&pSwapChainTex);
	//m_d3dDeviceContex->CopyResource(pSwapChainTex, g_pRaycastFinishedTex);
	//SAFE_RELEASE(pSwapChainTex);
	if (g_renderTexture.IsInitialized())
	{
		g_renderTexture.ClearRenderTarget(m_d3dDeviceContex, nullptr, g_backgroundColor.x(), g_backgroundColor.y(), g_backgroundColor.z(), g_backgroundColor.w());
		g_renderTexture.SetRenderTarget(m_d3dDeviceContex, nullptr);

		g_screenEffect.m_pTexVariable->SetResource(g_pRenderBufferTempSRV);
		g_screenEffect.m_pTechnique->GetPassByIndex(ScreenEffect::BlitBlendOver)->Apply(0, m_d3dDeviceContex);
		m_d3dDeviceContex->Draw(4, 0);
		//g_renderTexture.CopyFromTexture(m_d3dDeviceContex, g_pRaycastFinishedTex);
	}

	// draw background
	//m_d3dDeviceContex->IASetInputLayout(NULL);
	//m_d3dDeviceContex->IASetPrimitiveTopology(D3D10_PRIMITIVE_TOPOLOGY_TRIANGLESTRIP);
	//g_screenEffect.m_pvColorVariable->SetFloatVector(g_backgroundColor);
	//tum3D::Vec2f screenMin(-1.0f, -1.0f);
	//tum3D::Vec2f screenMax(1.0f, 1.0f);
	//g_screenEffect.m_pvScreenMinVariable->SetFloatVector(screenMin);
	//g_screenEffect.m_pvScreenMaxVariable->SetFloatVector(screenMax);
	//tum3D::Vec2f texCoordMin(0.0f, 0.0f);
	//tum3D::Vec2f texCoordMax(1.0f, 1.0f);
	//g_screenEffect.m_pvTexCoordMinVariable->SetFloatVector(texCoordMin);
	//g_screenEffect.m_pvTexCoordMaxVariable->SetFloatVector(texCoordMax);
	//g_screenEffect.m_pTechnique->GetPassByIndex(ScreenEffect::BlitBlendOver)->Apply(0, m_d3dDeviceContex);
	//m_d3dDeviceContex->Draw(4, 0);

#if 0
	if (g_imageSequence.Running) {
		if (!m_redraw && !s_isFiltering && !s_isRendering) {
			// current frame is finished, save (if specified) and advance
			if (g_imageSequence.Record) {
				std::wstring filenameW = tum3d::FindNextSequenceNameEX(L"video", L"png", CSysTools::GetExePath());
				std::string filename(filenameW.begin(), filenameW.end());
				if (g_imageSequence.FromRenderBuffer) {
					//SaveRenderBufferScreenshot(m_d3dDeviceContex, filename);
				}
				else {
					//SaveScreenshot(m_d3dDeviceContex, filename);
				}
			}

			g_imageSequence.FrameCur++;

			if (g_imageSequence.FrameCur >= g_imageSequence.FrameCount && g_imageSequence.Record) {
				g_imageSequence.Running = false;
				g_imageSequence.FrameCur = 0;
			}
			else {
				float angle = float(g_imageSequence.FrameCur) * g_imageSequence.AngleInc * PI / 180.0f;
				tum3D::Vec4f rotationQuatCurFrame; tum3D::rotationQuaternion(angle, tum3D::Vec3f(0.0f, 1.0f, 0.0f), rotationQuatCurFrame);
				tum3D::multQuaternion(g_imageSequence.BaseRotationQuat, rotationQuatCurFrame, g_viewParams.m_rotationQuat);

				g_viewParams.m_viewDistance += g_imageSequence.ViewDistInc;

				int32 timestep = g_imageSequence.BaseTimestep + (g_imageSequence.FrameCur / g_imageSequence.FramesPerTimestep);
				timestep %= g_volume.GetTimestepCount();
				float time = float(timestep) * g_volume.GetTimeSpacing();
				g_volume.SetCurTime(time);
			}
		}
	}


	if (g_batchTrace.Running) {
		if (!m_retrace && !s_isTracing) {
			// current trace is finished
			printf("\n------ Batch trace file %u step %u done.\n\n", g_batchTrace.FileCur, g_batchTrace.StepCur);

			// build output filename
			std::string volumeFileName = tum3d::RemoveExt(tum3d::GetFilename(g_batchTrace.VolumeFiles[g_batchTrace.FileCur]));
			std::ostringstream stream;
			stream << volumeFileName
				<< "_" << GetAdvectModeName(g_particleTraceParams.m_advectMode)
				<< "_" << GetTextureFilterModeName(g_particleTraceParams.m_filterMode);
			std::string strOutFileBaseNoSuffix = stream.str();

			if (g_batchTraceParams.m_qualityStepCount > 1) {
				stream << "_Q" << g_batchTraceParams.GetQualityStep(g_batchTrace.StepCur);
			}
			if (g_batchTraceParams.m_heuristicStepCount > 1) {
				stream << "_B" << g_batchTraceParams.GetHeuristicBonusStep(g_batchTrace.StepCur);
				stream << "P" << g_batchTraceParams.GetHeuristicPenaltyStep(g_batchTrace.StepCur);
			}
			std::string strOutFileBase = stream.str();

			g_batchTrace.Timings.push_back(g_tracingManager.GetTimings().TraceWall);

			if (g_batchTrace.WriteLinebufs) {
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
			extraColumns.push_back(std::to_string(g_particleTraceParams.m_heuristicBonusFactor));
			extraColumns.push_back(std::to_string(g_particleTraceParams.m_heuristicPenaltyFactor));
			g_tracingManager.GetStats().WriteToCSVFile(g_batchTrace.FileStats, extraColumns);
			g_tracingManager.GetTimings().WriteToCSVFile(g_batchTrace.FileTimings, extraColumns);

			// advance step
			g_batchTrace.StepCur++;

			if (g_batchTrace.StepCur >= g_batchTraceParams.GetTotalStepCount()) {
				if (g_batchTraceParams.m_heuristicBPSeparate)
				{
					// write this file's B/P timings into csv in matrix form
					std::ofstream fileTimings(g_batchTrace.OutPath + strOutFileBaseNoSuffix + "_Timings.csv");
					fileTimings << ";;Penalty\n";
					fileTimings << ";";
					for (uint p = 0; p < g_batchTraceParams.m_heuristicStepCount; p++)
					{
						fileTimings << ";" << g_batchTraceParams.GetHeuristicFactor(p);
					}
					fileTimings << "\n";
					size_t i = 0;
					for (uint b = 0; b < g_batchTraceParams.m_heuristicStepCount; b++)
					{
						fileTimings << (b == 0 ? "Bonus" : "");
						fileTimings << ";" << g_batchTraceParams.GetHeuristicFactor(b);
						for (uint p = 0; p < g_batchTraceParams.m_heuristicStepCount; p++)
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
				if (g_batchTrace.FileCur >= g_batchTrace.VolumeFiles.size()) {
					// finished
					g_batchTrace.Running = false;
					printf("\n------ Batch trace finished.\n\n");
					g_batchTrace.FileStats.close();
					g_batchTrace.FileTimings.close();
					g_batchTrace.Timings.clear();

					if (g_batchTrace.ExitAfterFinishing) {
						g_volume.Close(); // kill the loading thread!
						exit(0); // hrm..
					}
				}
				else {
					// open next volume file
					CloseVolumeFile();
					OpenVolumeFile(g_batchTrace.VolumeFiles[g_batchTrace.FileCur]);
					if (!LineModeIsTimeDependent(g_particleTraceParams.m_lineMode) && g_volume.IsCompressed()) {
						// semi-HACK: fully load the whole timestep, so we get results consistently without IO
						printf("Loading file %u...", g_batchTrace.FileCur);
						g_volume.LoadNearestTimestep();
						printf("\n");
					}
					if (g_particleTraceParams.HeuristicUseFlowGraph()) {
						LoadOrBuildFlowGraph();
					}

					// start next step
					g_batchTraceParams.ApplyToTraceParams(g_particleTraceParams, g_batchTrace.StepCur);
				}
			}
			else {
				// start next step
				g_batchTraceParams.ApplyToTraceParams(g_particleTraceParams, g_batchTrace.StepCur);
			}
		}
	}
#endif
}

void FlowVisTool::CheckForChanges()
{
	clock_t curTime = clock();

	static std::string volumeFilePrev = "";
	bool volumeChanged = (g_volume.GetFilename() != volumeFilePrev);
	volumeFilePrev = g_volume.GetFilename();
	m_redraw = m_redraw || volumeChanged;
	m_retrace = m_retrace || volumeChanged;


	static int timestepPrev = -1;
	bool timestepChanged = (g_volume.GetCurNearestTimestepIndex() != timestepPrev);
	timestepPrev = g_volume.GetCurNearestTimestepIndex();
	m_redraw = m_redraw || timestepChanged;
	m_retrace = m_retrace || timestepChanged;


	static float renderBufferSizeFactorPrev = 0.0f;
	bool renderBufferSizeChanged = (g_renderBufferSizeFactor != renderBufferSizeFactorPrev);
	renderBufferSizeFactorPrev = g_renderBufferSizeFactor;
	if (renderBufferSizeChanged)
	{
		ResizeRenderBuffer();
		g_lastRenderParamsUpdate = curTime;
	}


	static bool renderDomainBoxPrev = g_bRenderDomainBox;
	static bool renderBrickBoxesPrev = g_bRenderBrickBoxes;
	static bool renderClipBoxPrev = g_bRenderClipBox;
	static bool renderSeedBoxPrev = g_bRenderSeedBox;
	if (renderDomainBoxPrev != g_bRenderDomainBox || renderBrickBoxesPrev != g_bRenderBrickBoxes || renderClipBoxPrev != g_bRenderClipBox || renderSeedBoxPrev != g_bRenderSeedBox)
	{
		renderDomainBoxPrev = g_bRenderDomainBox;
		renderBrickBoxesPrev = g_bRenderBrickBoxes;
		renderClipBoxPrev = g_bRenderClipBox;
		renderSeedBoxPrev = g_bRenderSeedBox;
		m_redraw = true;
	}


	static ProjectionParams projParamsPrev;
	bool projParamsChanged = (g_projParams != projParamsPrev);
	projParamsPrev = g_projParams;
	m_redraw = m_redraw || projParamsChanged;

	static Range1D rangePrev;
	bool rangeChanged = (g_cudaDevices[g_primaryCudaDeviceIndex].range != rangePrev);
	rangePrev = g_cudaDevices[g_primaryCudaDeviceIndex].range;
	m_redraw = m_redraw || rangeChanged;

	if (projParamsChanged || rangeChanged)
	{
		// cancel rendering
		g_renderingManager.CancelRendering();
		if (g_useAllGPUs)
		{
			for (size_t i = 0; i < g_cudaDevices.size(); i++)
			{
				if (g_cudaDevices[i].pThread == nullptr) continue;
				g_cudaDevices[i].pThread->CancelRendering();
			}
		}

		// forward the new params
		g_renderingManager.SetProjectionParams(g_projParams, g_cudaDevices[g_primaryCudaDeviceIndex].range);
		if (g_useAllGPUs)
		{
			for (size_t i = 0; i < g_cudaDevices.size(); i++)
			{
				if (g_cudaDevices[i].pThread == nullptr) continue;
				g_cudaDevices[i].pThread->SetProjectionParams(g_projParams, g_cudaDevices[i].range);
			}
		}
		g_lastRenderParamsUpdate = curTime;
	}


	static StereoParams stereoParamsPrev;
	bool stereoParamsChanged = (g_stereoParams != stereoParamsPrev);
	stereoParamsPrev = g_stereoParams;
	m_redraw = m_redraw || stereoParamsChanged;

	if (stereoParamsChanged)
		g_lastRenderParamsUpdate = curTime;

	static ViewParams viewParamsPrev;
	bool viewParamsChanged = (g_viewParams != viewParamsPrev);
	viewParamsPrev = g_viewParams;
	m_redraw = m_redraw || viewParamsChanged;

	if (viewParamsChanged)
		g_lastRenderParamsUpdate = curTime;

	//if(g_tfEdt.getTimestamp() != g_tfTimestamp) 
	//{
	//	m_redraw = true;
	//	g_lastRenderParamsUpdate = curTime;
	//	g_tfTimestamp = g_tfEdt.getTimestamp();
	//}


	// Get raycast params from the TF Editor's UI
	//g_raycastParams.m_alphaScale				= g_tfEdt.getAlphaScale(TF_RAYTRACE);
	//g_raycastParams.m_transferFunctionRangeMin	= g_tfEdt.getTfRangeMin(TF_RAYTRACE);
	//g_raycastParams.m_transferFunctionRangeMax	= g_tfEdt.getTfRangeMax(TF_RAYTRACE);
	//g_particleRenderParams.m_transferFunctionRangeMin = g_tfEdt.getTfRangeMin(TF_LINE_MEASURES);
	//g_particleRenderParams.m_transferFunctionRangeMax = g_tfEdt.getTfRangeMax(TF_LINE_MEASURES);
	//g_heatMapParams.m_tfAlphaScale = g_tfEdt.getAlphaScale(TF_HEAT_MAP);
	//g_heatMapParams.m_tfRangeMin = g_tfEdt.getTfRangeMin(TF_HEAT_MAP);
	//g_heatMapParams.m_tfRangeMax = g_tfEdt.getTfRangeMax(TF_HEAT_MAP);

	static FilterParams filterParamsPrev;
	bool filterParamsChanged = (g_filterParams != filterParamsPrev);
	filterParamsPrev = g_filterParams;

	// clear filtered bricks if something relevant changed
	if (volumeChanged || timestepChanged || filterParamsChanged)
	{
		g_filteringManager.ClearResult();
		s_isFiltering = false;
		m_redraw = true;
	}

	static ParticleTraceParams particleTraceParamsPrev;
	bool particleTraceParamsChanged = g_particleTraceParams.hasChangesForRetracing(particleTraceParamsPrev);
	//bool particleTraceParamsChanged = (g_particleTraceParams != particleTraceParamsPrev);
	bool seedBoxChanged = (g_particleTraceParams.m_seedBoxMin != particleTraceParamsPrev.m_seedBoxMin || g_particleTraceParams.m_seedBoxSize != particleTraceParamsPrev.m_seedBoxSize);
	particleTraceParamsPrev = g_particleTraceParams;
	m_retrace = m_retrace || particleTraceParamsChanged;
	m_redraw = m_redraw || seedBoxChanged;

	// heat map parameters
	static HeatMapParams heatMapParamsPrev;
	static bool heatMapDoubleRedraw = false;
	bool debugHeatMapPrint = false;
	//m_retrace = m_retrace || g_heatMapParams.HasChangesForRetracing(heatMapParamsPrev, g_particleTraceParams);
	//m_redraw = m_redraw || g_heatMapParams.HasChangesForRedrawing(heatMapParamsPrev);
	if (g_heatMapParams.HasChangesForRetracing(heatMapParamsPrev, g_particleTraceParams))
	{
		m_retrace = true;
		std::cout << "heat map has changes for retracing" << std::endl;
		debugHeatMapPrint = true;
		heatMapDoubleRedraw = true;
	}
	if (g_heatMapParams.HasChangesForRedrawing(heatMapParamsPrev))
	{
		m_redraw = true;
		std::cout << "heat map has changes for redrawing" << std::endl;
		debugHeatMapPrint = true;
		heatMapDoubleRedraw = true;
	}
	heatMapParamsPrev = g_heatMapParams;
	g_heatMapManager.SetParams(g_heatMapParams);

#if 0
	if (debugHeatMapPrint)
		g_heatMapManager.DebugPrintParams();
#endif

	if (heatMapDoubleRedraw && !m_redraw)
	{
		//Hack: For some reasons, the heat map manager only applies changes after the second rendering
		//Or does the RenderManager not update the render targets?
		m_redraw = true;
		heatMapDoubleRedraw = false;
	}

	if (particleTraceParamsChanged)
	{
		g_lastTraceParamsUpdate = curTime;
		g_lastRenderParamsUpdate = curTime;
	}


	// clear particle tracer if something relevant changed
	if (m_retrace && s_isTracing)
	{
		g_tracingManager.CancelTracing();
		s_isTracing = false;
		g_tracingManager.ClearResult();
		m_redraw = true;
		g_lastRenderParamsUpdate = curTime;
	}



	static RaycastParams raycastParamsPrev = g_raycastParams;
	bool raycastParamsChanged = (g_raycastParams != raycastParamsPrev);
	raycastParamsPrev = g_raycastParams;
	m_redraw = m_redraw || raycastParamsChanged;

	if (raycastParamsChanged)
		g_lastTraceParamsUpdate = curTime;

	static ParticleRenderParams particleRenderParamsPrev = g_particleRenderParams;
	bool particleRenderParamsChanged = (g_particleRenderParams != particleRenderParamsPrev);
	particleRenderParamsPrev = g_particleRenderParams;
	m_redraw = m_redraw || particleRenderParamsChanged;

	//if(particleRenderParamsChanged)
	//{
	//	g_lastTraceParamsUpdate = curTime;
	//}
}

void FlowVisTool::Filtering()
{
	// start filtering if required
	bool needFilteredBricks = g_filterParams.HasNonZeroRadius();
	if (needFilteredBricks && !s_isFiltering && g_filteringManager.GetResultCount() == 0)
	{
		g_renderingManager.CancelRendering();
		if (g_useAllGPUs)
		{
			for (size_t i = 0; i < g_cudaDevices.size(); i++)
			{
				if (g_cudaDevices[i].pThread == nullptr) continue;
				g_cudaDevices[i].pThread->CancelRendering();
			}
		}
		// release other resources - we'll need a lot of memory for filtering
		g_tracingManager.ReleaseResources();
		g_renderingManager.ReleaseResources();

		assert(g_filteringManager.IsCreated());

		s_isFiltering = g_filteringManager.StartFiltering(g_volume, g_filterParams);
	}
}

void FlowVisTool::Tracing()
{
	// set parameters even if tracing is currently enabled.
	// This allows changes to the parameters in the particle mode, even if they are currently running
	if (!m_retrace && s_isTracing)
		g_tracingManager.SetParams(g_particleTraceParams);

	// start particle tracing if required
	float timeSinceTraceUpdate = float(clock() - g_lastTraceParamsUpdate) / float(CLOCKS_PER_SEC);
	bool traceDelayPassed = (timeSinceTraceUpdate >= g_startWorkingDelay);
	//bool traceStartNow = !s_isFiltering && g_particleRenderParams.m_linesEnabled && traceDelayPassed; //TODO: enable tracing also when rendering is disabled?
	bool traceStartNow = !s_isFiltering && traceDelayPassed;
	if (m_retrace && traceStartNow)
	{
		g_renderingManager.CancelRendering();
		if (g_useAllGPUs)
		{
			for (size_t i = 0; i < g_cudaDevices.size(); i++)
			{
				if (g_cudaDevices[i].pThread == nullptr) continue;
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

		m_retrace = false;
	}


	const std::vector<const TimeVolumeIO::Brick*>& bricksToLoad =
		s_isFiltering ? g_filteringManager.GetBricksToLoad() :
		(s_isTracing ? g_tracingManager.GetBricksToLoad() : g_renderingManager.GetBricksToLoad());
	//TODO when nothing's going on, load bricks in any order? (while memory is available etc..)
	g_volume.UpdateLoadingQueue(bricksToLoad);
	g_volume.UnloadLRUBricks();

	//Check if tracing is done and if so, start rendering
	if (s_isTracing && !g_particleTracingPaused)
	{
		//std::cout << "Trace" << std::endl;
		bool finished = g_tracingManager.Trace();
		if (finished || LineModeIsIterative(g_particleTraceParams.m_lineMode))
			g_heatMapManager.ProcessLines(g_tracingManager.GetResult());

		if (finished)
		{
			s_isTracing = false;
			g_timerTracing.Stop();
			m_redraw = true;
		}
		else if (g_showPreview)
		{
			g_tracingManager.BuildIndexBuffer();
			m_redraw = true;
		}
	}
	else if (s_isFiltering)
	{
		bool finished = g_filteringManager.Filter();

		if (finished)
		{
			s_isFiltering = false;
			m_redraw = true;
		}
	}
}

void FlowVisTool::Rendering()
{
	static std::vector<uint> s_timestampLastUpdate; // last time the result image from each GPU was taken

	bool renderingUpdated = false;
	if (m_redraw)
	{
		// cancel rendering of current image
		g_renderingManager.CancelRendering();
		if (g_useAllGPUs)
		{
			for (size_t i = 0; i < g_cudaDevices.size(); i++)
			{
				if (g_cudaDevices[i].pThread == nullptr) continue;
				g_cudaDevices[i].pThread->CancelRendering();
			}
		}

		// release other resources if possible
		if (!s_isFiltering)
			g_filteringManager.ReleaseResources();

		if (!s_isTracing)
			g_tracingManager.ReleaseResources();

		// while tracing/filtering is in progress, don't start raycasting
		bool linesOnly = s_isTracing || s_isFiltering || !g_raycastParams.m_raycastingEnabled;

		cudaArray* pTfArray = nullptr;
		bool needTF = (g_raycastParams.m_raycastingEnabled && RaycastModeNeedsTransferFunction(g_raycastParams.m_raycastMode));
		if (needTF)
		{
			cudaSafeCall(cudaGraphicsMapResources(1, &g_pTfEdtSRVCuda));
			cudaSafeCall(cudaGraphicsSubResourceGetMappedArray(&pTfArray, g_pTfEdtSRVCuda, 0, 0));
		}
		std::vector<LineBuffers*> lineBuffers = g_lineBuffers;
		LineBuffers* pTracedLines = g_tracingManager.GetResult().get();
		if (pTracedLines != nullptr)
		{
			lineBuffers.push_back(pTracedLines);
		}
		RenderingManager::eRenderState state = g_renderingManager.StartRendering(g_tracingManager.IsTracing(),
			g_volume, g_filteringManager.GetResults(),
			g_viewParams, g_stereoParams,
			g_bRenderDomainBox, g_bRenderClipBox, g_bRenderSeedBox, g_bRenderBrickBoxes,
			g_particleTraceParams, g_particleRenderParams,
			lineBuffers, linesOnly,
			g_ballBuffers, g_ballRadius,
			&g_heatMapManager,
			g_raycastParams, pTfArray, g_tracingManager.m_dpParticles);


		if (state == RenderingManager::STATE_ERROR)
		{
			printf("RenderingManager::StartRendering returned STATE_ERROR.\n");
			s_isRendering = false;
		}
		else
		{
			s_isRendering = true; // set to true even if STATE_DONE - other GPUs might have something to do
			renderingUpdated = true;

			if (g_useAllGPUs)
			{
				for (size_t i = 0; i < g_cudaDevices.size(); i++)
				{
					if (g_cudaDevices[i].pThread != nullptr)
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
		if (needTF)
			cudaSafeCall(cudaGraphicsUnmapResources(1, &g_pTfEdtSRVCuda));

		m_redraw = false;
	}

	bool renderingFinished = true; // default to "true", set to false if any GPU is not finished
	//TODO only call g_renderingManager.Render() when renderDelayPassed?
	//float timeSinceRenderUpdate = float(curTime - g_lastRenderParamsUpdate) / float(CLOCKS_PER_SEC);
	//bool renderDelayPassed = (timeSinceRenderUpdate >= g_startWorkingDelay);
	if (s_isRendering)
	{
		if (g_renderingManager.IsRendering())
		{
			// render next brick on primary GPU
			renderingFinished = (g_renderingManager.Render() == RenderingManager::STATE_DONE);
			renderingUpdated = true;
		}

		// if primary GPU is done, check if other threads are finished as well
		if (renderingFinished && g_useAllGPUs)
		{
			for (size_t i = 0; i < g_cudaDevices.size(); i++)
			{
				if (g_cudaDevices[i].pThread == nullptr) continue;

				renderingFinished = renderingFinished && !g_cudaDevices[i].pThread->IsWorking();
				renderingUpdated = true;
			}
		}

		if (renderingFinished)
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
		tum3D::Vec2f screenMin(-1.0f, -1.0f);
		tum3D::Vec2f screenMax(1.0f, 1.0f);
		g_screenEffect.m_pvScreenMinVariable->SetFloatVector(screenMin);
		g_screenEffect.m_pvScreenMaxVariable->SetFloatVector(screenMax);
		tum3D::Vec2f texCoordMin(0.0f, 0.0f);
		tum3D::Vec2f texCoordMax(1.0f, 1.0f);
		g_screenEffect.m_pvTexCoordMinVariable->SetFloatVector(texCoordMin);
		g_screenEffect.m_pvTexCoordMaxVariable->SetFloatVector(texCoordMax);

		m_d3dDeviceContex->IASetInputLayout(NULL);
		m_d3dDeviceContex->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLESTRIP);


		if (!g_useAllGPUs || (g_cudaDevices.size() == 1))
		{
			//m_d3dDeviceContex->ClearRenderTargetView(g_pRenderBufferTempRTV, (float*)&g_backgroundColor);
			// single-GPU case - blend together rendering manager's "opaque" and "raycast" textures
			m_d3dDeviceContex->OMSetRenderTargets(1, &g_pRenderBufferTempRTV, nullptr);

			// save old viewport
			UINT viewportCount = 1;
			D3D11_VIEWPORT viewportOld;
			m_d3dDeviceContex->RSGetViewports(&viewportCount, &viewportOld);

			// set viewport for offscreen render buffer
			D3D11_VIEWPORT viewport;
			viewport.TopLeftX = 0;
			viewport.TopLeftY = 0;
			viewport.Width = (FLOAT)g_projParams.m_imageWidth;
			viewport.Height = (FLOAT)g_projParams.m_imageHeight;
			viewport.MinDepth = 0.0f;
			viewport.MaxDepth = 1.0f;
			m_d3dDeviceContex->RSSetViewports(1, &viewport);

			// blit opaque stuff
			g_screenEffect.m_pTexVariable->SetResource(g_renderingManager.GetOpaqueSRV());
			g_screenEffect.m_pTechnique->GetPassByIndex(ScreenEffect::Blit)->Apply(0, m_d3dDeviceContex);
			m_d3dDeviceContex->Draw(4, 0);

			// blend raycaster result over
			g_screenEffect.m_pTexVariable->SetResource(g_renderingManager.GetRaycastSRV());
			g_screenEffect.m_pTechnique->GetPassByIndex(ScreenEffect::BlitBlendOver)->Apply(0, m_d3dDeviceContex);
			m_d3dDeviceContex->Draw(4, 0);

			// restore viewport
			m_d3dDeviceContex->RSSetViewports(1, &viewportOld);
		}
		else
		{
			// multi-GPU case - copy all raycast textures together
			//TODO what about opaque stuff..?
			for (size_t i = 0; i < g_cudaDevices.size(); i++)
			{
				int left = g_projParams.GetImageLeft(g_cudaDevices[i].range);
				int width = g_projParams.GetImageWidth(g_cudaDevices[i].range);
				int height = g_projParams.GetImageHeight(g_cudaDevices[i].range);

				if (g_cudaDevices[i].pThread == nullptr)
				{
					// this is the main GPU, copy over directly
					D3D11_BOX box;
					box.left = 0;
					box.right = width;
					box.top = 0;
					box.bottom = height;
					box.front = 0;
					box.back = 1;

					m_d3dDeviceContex->CopySubresourceRegion(g_pRenderBufferTempTex, 0, left, 0, 0, g_renderingManager.GetRaycastTex(), 0, &box);
				}
				else
				{
					// get image from thread and upload to this GPU
					byte* pData = nullptr;
					uint timestamp = g_cudaDevices[i].pThread->LockResultImage(pData);
					if (timestamp > s_timestampLastUpdate[i])
					{
						s_timestampLastUpdate[i] = timestamp;

						D3D11_BOX box;
						box.left = left;
						box.right = box.left + width;
						box.top = 0;
						box.bottom = box.top + height;
						box.front = 0;
						box.back = 1;

						m_d3dDeviceContex->UpdateSubresource(g_pRenderBufferTempTex, 0, &box, pData, width * sizeof(uchar4), 0);
					}
					g_cudaDevices[i].pThread->UnlockResultImage();
				}
			}
		}

		// blit over into "raycast finished" tex
		//g_renderTexture.ClearRenderTarget(m_d3dDeviceContex, nullptr, g_backgroundColor.x(), g_backgroundColor.y(), g_backgroundColor.z(), g_backgroundColor.w());
		//g_renderTexture.SetRenderTarget(m_d3dDeviceContex, nullptr);

		m_d3dDeviceContex->OMSetRenderTargets(1, &g_pRaycastFinishedRTV, nullptr);
		//m_d3dDeviceContex->ClearRenderTargetView(g_pRaycastFinishedRTV, (float*)&g_backgroundColor);

		//TODO if g_renderBufferSizeFactor > 2, generate mipmaps first?
		g_screenEffect.m_pTexVariable->SetResource(g_pRenderBufferTempSRV);
		g_screenEffect.m_pTechnique->GetPassByIndex(ScreenEffect::Blit)->Apply(0, m_d3dDeviceContex);
		m_d3dDeviceContex->Draw(4, 0);

		ID3D11ShaderResourceView* pNullSRV[1] = { nullptr };
		m_d3dDeviceContex->PSSetShaderResources(0, 1, pNullSRV);

		// reset render target
		//ID3D11RenderTargetView* pRTV = DXUTGetD3D11RenderTargetView();
		//g_pd3dDeviceContext->OMSetRenderTargets(1, &g_mainRenderTargetView, g_mainDepthStencilView);
	}
}

bool FlowVisTool::ResizeViewport(int width, int height)
{
	std::cout << "Scene window resize: " << width << ", " << height << std::endl;

	g_renderTexture.Release();
	g_renderTexture.Initialize(m_d3dDevice, width, height);

	m_redraw = true;

	g_windowSize.x() = width;
	g_windowSize.y() = height;


	ResizeRenderBuffer();


	HRESULT hr;

	// create texture to hold last finished raycasted image
	D3D11_TEXTURE2D_DESC desc;
	desc.ArraySize = 1;
	desc.BindFlags = D3D11_BIND_RENDER_TARGET;
	desc.CPUAccessFlags = 0;
	desc.Format = DXGI_FORMAT_R32G32B32A32_FLOAT;
	desc.MipLevels = 1;
	desc.MiscFlags = 0;
	desc.SampleDesc.Count = 1;
	desc.SampleDesc.Quality = 0;
	desc.Usage = D3D11_USAGE_DEFAULT;
	desc.Width = g_windowSize.x();
	desc.Height = g_windowSize.y();
	// create texture for the last finished image from the raycaster
	hr = m_d3dDevice->CreateTexture2D(&desc, nullptr, &g_pRaycastFinishedTex);
	if (FAILED(hr)) return false;
	hr = m_d3dDevice->CreateRenderTargetView(g_pRaycastFinishedTex, nullptr, &g_pRaycastFinishedRTV);
	if (FAILED(hr)) return false;


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
	hr = m_d3dDevice->CreateTexture2D(&desc, nullptr, &g_pStagingTex);
	if (FAILED(hr)) return false;


	// GUI
	//TwWindowSize(g_windowSize.x(), g_windowSize.y());
	//// can't set fontsize before window has been initialized, so do it here instead of init()...
	//TwDefine("GLOBAL fontsize=3 fontresizable=false");

	//// main gui on left side
	//int posMain[2] = { 10, 10 };
	//int sizeMain[2];
	//TwGetParam(g_pTwBarMain, nullptr, "size", TW_PARAM_INT32, 2, sizeMain);
	//sizeMain[1] = max(static_cast<int>(pBackBufferSurfaceDesc->Height) - 40, 200);

	//TwSetParam(g_pTwBarMain, nullptr, "position", TW_PARAM_INT32, 2, posMain);
	//TwSetParam(g_pTwBarMain, nullptr, "size", TW_PARAM_INT32, 2, sizeMain);

	//// image sequence recorder in upper right corner
	//int sizeImgSeq[2];
	//TwGetParam(g_pTwBarImageSequence, nullptr, "size", TW_PARAM_INT32, 2, sizeImgSeq);
	//int posImgSeq[2] = { max(10, (int)g_windowSize.x() - sizeImgSeq[0] - 10), 10 };
	//TwSetParam(g_pTwBarImageSequence, nullptr, "position", TW_PARAM_INT32, 2, posImgSeq);

	//g_tfEdt.onResizeSwapChain( pBackBufferSurfaceDesc->Width, pBackBufferSurfaceDesc->Height );

	//// batch tracing in middle right
	//int sizeBatch[2];
	//TwGetParam(g_pTwBarBatchTrace, nullptr, "size", TW_PARAM_INT32, 2, sizeBatch);
	//int posBatch[2] = { max(10, (int)g_windowSize.x() - sizeBatch[0] - 10), max(10, ((int)g_windowSize.y() - sizeBatch[1]) / 2 - 10) };
	//TwSetParam(g_pTwBarBatchTrace, nullptr, "position", TW_PARAM_INT32, 2, posBatch);

	//g_tfEdt.onResizeSwapChain( pBackBufferSurfaceDesc->Width, pBackBufferSurfaceDesc->Height );


	return true;
}

void FlowVisTool::ShutdownCudaDevices()
{
	for (size_t index = 0; index < g_cudaDevices.size(); index++)
	{
		if (g_cudaDevices[index].pThread == nullptr) continue;

		g_cudaDevices[index].pThread->CancelCurrentTask();
	}
	for (size_t index = 0; index < g_cudaDevices.size(); index++)
	{
		if (g_cudaDevices[index].pThread == nullptr) continue;

		g_cudaDevices[index].pThread->Stop();
		delete g_cudaDevices[index].pThread;
	}

	g_primaryCudaDeviceIndex = -1;
}

bool FlowVisTool::InitCudaDevices()
{
	// current device has been set to the primary one by cudaD3D11SetDirect3DDevice
	int deviceCur = -1;
	cudaSafeCallNoSync(cudaGetDevice(&deviceCur));

	// find primary device index, and sum up total compute/memory power
	g_primaryCudaDeviceIndex = -1;
	float totalComputePower = 0.0f;
	float totalMemoryPower = 0.0f;
	for (size_t index = 0; index < g_cudaDevices.size(); index++)
	{
		totalComputePower += g_cudaDevices[index].computePower;
		totalMemoryPower += g_cudaDevices[index].memoryPower;
		if (g_cudaDevices[index].device == deviceCur)
		{
			g_primaryCudaDeviceIndex = (int)index;
		}
	}
	if (g_primaryCudaDeviceIndex == -1)
	{
		printf("ERROR: Did not find primary CUDA device %i among all eligible CUDA devices\n", deviceCur);
		return false;
	}

	if (g_useAllGPUs)
	{
		printf("Using all CUDA devices; %i is primary\n", deviceCur);

		// create and start rendering threads
		float rangeMin = 0.0f, rangeMax = 0.0f;
		for (size_t index = 0; index < g_cudaDevices.size(); index++)
		{
			float myPower = g_cudaDevices[index].computePower / totalComputePower;
			//TODO also/only use memory power?

			rangeMin = rangeMax;
			rangeMax = rangeMin + myPower;
			if (index == g_cudaDevices.size() - 1) rangeMax = 1.0f;
			//rangeMax = float(index + 1) / float(g_cudaDevices.size());

			g_cudaDevices[index].range.Set(rangeMin, rangeMax);

			// don't create extra thread for main GPU
			if (index == g_primaryCudaDeviceIndex) continue;

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