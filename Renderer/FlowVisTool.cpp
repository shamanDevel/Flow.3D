#include <FlowVisTool.h>

#include <ctime>
#include <string>
#include <map>
#include <algorithm>

#include <cudaUtil.h>
#include <CSysTools.h>
#include <winerror.h>

#include <imgui\imgui.h>


#define g_startWorkingDelay 0.1f


FlowVisTool::FlowVisTool()
{
}


bool FlowVisTool::Initialize(ID3D11Device* d3dDevice, ID3D11DeviceContext* d3dDeviceContex, const std::vector<MyCudaDevice>& cudaDevices)
{
	m_d3dDevice = d3dDevice;
	m_d3dDeviceContex = d3dDeviceContex;
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

	//if(FAILED(hr = g_tfEdt.onCreateDevice( pd3dDevice )))
	//	return hr;
	//g_particleRenderParams.m_pTransferFunction = g_tfEdt.getSRV(TF_LINE_MEASURES);
	//g_heatMapParams.m_pTransferFunction = g_tfEdt.getSRV(TF_HEAT_MAP);
	//cudaSafeCall(cudaGraphicsD3D11RegisterResource(&g_pTfEdtSRVCuda, g_tfEdt.getTexture(TF_RAYTRACE), cudaGraphicsRegisterFlagsNone));

	if (!g_filteringManager.Create())
		return false;

	if (!g_renderingManager.Create(m_d3dDevice))
		return false;

	if (!g_raycasterManager.Create(m_d3dDevice))
		return false;

	if (!g_heatMapManager.Create(m_d3dDevice))
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

	return true;
}

void FlowVisTool::Release()
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
	g_raycasterManager.Release();

	g_heatMapManager.Release();

	g_filteringManager.ClearResult();
	g_filteringManager.Release();

	for (size_t i = 0; i < g_volumes.size(); i++)
		CloseVolumeFile(i);
	
	if (g_pTfEdtSRVCuda != nullptr)
	{
		cudaSafeCall(cudaGraphicsUnregisterResource(g_pTfEdtSRVCuda));
		g_pTfEdtSRVCuda = nullptr;
	}
	//g_tfEdt.onDestroyDevice();


	ReleaseVolumeDependentResources();

	ShutdownCudaDevices();

	g_screenEffect.SafeRelease();


	cudaSafeCall(cudaDeviceSynchronize());
	g_renderingManager.m_ftleTexture.UnregisterCudaResources();
	cudaSafeCallNoSync(cudaDeviceReset());

	g_renderingManager.m_ftleTexture.ReleaseResources();
}


void FlowVisTool::BuildFlowGraph(const std::string& filenameTxt)
{
	ParticleTraceParams& traceParamns = g_volumes[m_selectedVolume]->m_traceParams;

	//TODO might need to cancel tracing first?
	uint advectStepsPerRoundBak = traceParamns.m_advectStepsPerRound;
	traceParamns.m_advectStepsPerRound = 256;
	g_flowGraph.Build(1024, traceParamns, filenameTxt);
	traceParamns.m_advectStepsPerRound = advectStepsPerRoundBak;
}

bool FlowVisTool::SaveFlowGraph()
{
	std::string filename = tum3d::RemoveExt(g_volumes[m_selectedVolume]->m_volume->GetFilename()) + ".fg";
	return g_flowGraph.SaveToFile(filename);
}

bool FlowVisTool::LoadFlowGraph(FlowVisToolVolumeData* volumeData)
{
	std::string filename = tum3d::RemoveExt(volumeData->m_volume->GetFilename()) + ".fg";
	return g_flowGraph.LoadFromFile(filename);
}

#ifdef Single
void FlowVisTool::LoadOrBuildFlowGraph()
{
	if (!LoadFlowGraph()) {
		BuildFlowGraph();
		SaveFlowGraph();
	}
}
#endif


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
}

bool FlowVisTool::CreateVolumeDependentResources(FlowVisToolVolumeData* volumeData)
{
	std::cout << "Creating volume dependent reources..." << std::endl;

	ReleaseVolumeDependentResources();
	
	g_flowGraph.Init(*volumeData->m_volume);

	std::cout << "Volume dependent reources created." << std::endl;

	return true;
}


void FlowVisTool::SetSelectedVolume(int selected)
{
	m_selectedVolume = selected;

	// recreate brick slots and re-init cudaCompress - brick size may have changed
	CreateVolumeDependentResources(g_volumes[m_selectedVolume]);

	g_raycastParams.m_clipBoxMin = -g_volumes[m_selectedVolume]->m_volume->GetVolumeHalfSizeWorld();
	g_raycastParams.m_clipBoxMax = g_volumes[m_selectedVolume]->m_volume->GetVolumeHalfSizeWorld();
	g_raycastParams.m_redraw = true;

	m_restartFiltering = true;

#if defined(BATCH_IMAGE_SEQUENCE)
	g_imageSequence.FrameCount = g_volume.GetTimestepCount();
#endif

	LoadFlowGraph(g_volumes[m_selectedVolume]);
}


void FlowVisTool::CloseVolumeFile(int idx)
{
	assert(!g_volumes.empty());
	assert(idx >= 0 && idx < g_volumes.size());
	assert(g_volumes[idx]);
	assert(g_volumes[idx]->m_volume);

	if (idx == m_selectedVolume)
	{
		ReleaseVolumeDependentResources();
		m_selectedVolume = -1;
	}

	g_volumes[idx]->m_volume->SetPageLockBrickData(false);
	g_volumes[idx]->m_volume->Close();
	delete g_volumes[idx]->m_volume;
	g_volumes[idx]->m_volume = nullptr;
	g_volumes[idx]->ReleaseResources();
	delete g_volumes[idx];
	g_volumes[idx] = nullptr;
	g_volumes.erase(g_volumes.begin() + idx);
}

bool FlowVisTool::OpenVolumeFile(const std::string& filename)
{
	//CloseVolumeFile();

	TimeVolume* vol = new TimeVolume();

	if (!vol->Open(filename))
	{
		delete vol;
		return false;
	}

	FlowVisToolVolumeData* volumeData = new FlowVisToolVolumeData();

	volumeData->m_volume = vol;
	volumeData->m_volume->SetCurTime(0.0f);
	volumeData->m_retrace = true;
	volumeData->SetSeedingBoxToDomainSize();
	volumeData->CreateResources(m_d3dDevice);

//	if (g_volumes.empty())
//	{
//		// recreate brick slots and re-init cudaCompress - brick size may have changed
//		CreateVolumeDependentResources(volumeData);
//
//		g_raycastParams.m_clipBoxMin = -volumeData->m_volume->GetVolumeHalfSizeWorld();
//		g_raycastParams.m_clipBoxMax = volumeData->m_volume->GetVolumeHalfSizeWorld();
//		g_raycastParams.m_redraw = true;
//
//		m_restartFiltering = true;
//
//#if defined(BATCH_IMAGE_SEQUENCE)
//		g_imageSequence.FrameCount = g_volume.GetTimestepCount();
//#endif
//
//		LoadFlowGraph(volumeData);
//	}

	g_volumes.push_back(volumeData);
	

	return true;
}


void FlowVisTool::OnFrame(float deltatime)
{
	if (g_volumes.empty())
	{
		if (g_renderTexture.IsInitialized())
			g_renderTexture.ClearRenderTarget(m_d3dDeviceContex, nullptr, g_renderingParams.m_backgroundColor);
	}

	// Check if stuff changed
	CheckForChanges();

	for (size_t i = 0; i < g_volumes.size(); i++)
		CheckForChanges(g_volumes[i]);

	// Start stuff
	if (ShouldStartFiltering())
		StartFiltering();

	for (size_t i = 0; i < g_volumes.size(); i++)
		if (ShouldStartTracing(g_volumes[i]))
			StartTracing(g_volumes[i], g_flowGraph);

	if (ShouldStartRaycasting())
		StartRaycasting();

	// Update stuff
	if (m_isFiltering)
		UpdateFiltering();

	for (size_t i = 0; i < g_volumes.size(); i++)
		if (g_volumes[i]->m_isTracing && !g_volumes[i]->m_tracingPaused)
			UpdateTracing(g_volumes[i]);

	if (m_isRaycasting)
		UpdateRaycasting();

	// Render stuff
	RenderTracingResults();

	BlitRaycastingResults();

	// Present stuff
	if (g_renderTexture.IsInitialized())
	{
		g_renderTexture.ClearRenderTarget(m_d3dDeviceContex, nullptr, g_renderingParams.m_backgroundColor);
		g_renderTexture.SetRenderTarget(m_d3dDeviceContex, nullptr);

		g_screenEffect.m_pTexVariable->SetResource(g_pRenderBufferTempSRV);
		g_screenEffect.m_pTechnique->GetPassByIndex(ScreenEffect::BlitBlendOver)->Apply(0, m_d3dDeviceContex);
		m_d3dDeviceContex->Draw(4, 0);
		//g_renderTexture.CopyFromTexture(m_d3dDeviceContex, g_pRaycastFinishedTex);

		m_d3dDeviceContex->GenerateMips(g_renderTexture.GetShaderResourceView());
	}

#ifdef BATCH_IMAGE_SEQUENCE
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


bool FlowVisTool::CheckForChanges(FlowVisToolVolumeData* volumeData)
{
	assert(volumeData);
	assert(volumeData->m_volume);
	assert(volumeData->m_volume->IsOpen());

	bool redraw = false;

	clock_t curTime = clock();

	static std::map<FlowVisToolVolumeData*, int> timestepPrev;

	bool timestepChanged = (volumeData->m_volume->GetCurNearestTimestepIndex() != timestepPrev[volumeData]);
	timestepPrev[volumeData] = volumeData->m_volume->GetCurNearestTimestepIndex();
	redraw = redraw || timestepChanged;
	volumeData->m_retrace = volumeData->m_retrace || timestepChanged;

	static std::map<FlowVisToolVolumeData*, ParticleTraceParams> traceParamsPrev;

	bool particleTraceParamsChanged = volumeData->m_traceParams.hasChangesForRetracing(traceParamsPrev[volumeData]);
	//bool particleTraceParamsChanged = (g_particleTraceParams != particleTraceParamsPrev);
	bool seedBoxChanged = (volumeData->m_traceParams.m_seedBoxMin != traceParamsPrev[volumeData].m_seedBoxMin || volumeData->m_traceParams.m_seedBoxSize != traceParamsPrev[volumeData].m_seedBoxSize);

	traceParamsPrev[volumeData] = volumeData->m_traceParams;

	volumeData->m_retrace = volumeData->m_retrace || particleTraceParamsChanged;
	redraw = redraw || seedBoxChanged;

	if (particleTraceParamsChanged)
	{
		volumeData->m_lastTraceParamsUpdate = curTime;
		//g_lastRenderParamsUpdate = curTime;
	}

	// clear particle tracer if something relevant changed
	if (volumeData->m_retrace && volumeData->m_isTracing)
	{
		std::cout << volumeData->m_volume->GetName() << " should retrace. Cancelling current tracing." << std::endl;
		volumeData->m_tracingManager.CancelTracing();
		volumeData->m_isTracing = false;
		volumeData->m_tracingManager.ClearResult();
		redraw = true;
	}

	static std::map<FlowVisToolVolumeData*, ParticleRenderParams> renderParamsPrev;

	bool particleRenderParamsChanged = volumeData->m_renderParams != renderParamsPrev[volumeData];
	renderParamsPrev[volumeData] = volumeData->m_renderParams;
	redraw = redraw || particleRenderParamsChanged;

	return redraw;
}

bool FlowVisTool::Tracing(FlowVisToolVolumeData* volumeData, FlowGraph& flowGraph)
{
	assert(volumeData);
	assert(volumeData->m_volume);
	assert(volumeData->m_volume->IsOpen());

	// set parameters even if tracing is currently enabled.
	// This allows changes to the parameters in the particle mode, even if they are currently running
	if (!volumeData->m_retrace && volumeData->m_isTracing)
		volumeData->m_tracingManager.SetParams(volumeData->m_traceParams);

	// start particle tracing if required
	float timeSinceTraceUpdate = float(clock() - volumeData->m_lastTraceParamsUpdate) / float(CLOCKS_PER_SEC);
	bool traceDelayPassed = timeSinceTraceUpdate >= g_startWorkingDelay;
	//bool traceStartNow = !s_isFiltering && g_particleRenderParams.m_linesEnabled && traceDelayPassed; //TODO: enable tracing also when rendering is disabled?


	//bool traceStartNow = !s_isFiltering && traceDelayPassed;
	bool traceStartNow = traceDelayPassed;
	if (volumeData->m_retrace && traceStartNow)
		StartTracing(volumeData, flowGraph);


	// Update bricks
	//UpdateBricks(volumeData);

	bool shouldRedraw = false;

	//Check if tracing is done and if so, start rendering
	if (volumeData->m_isTracing && !volumeData->m_tracingPaused)
	{
		//std::cout << "Trace" << std::endl;
		bool finished = volumeData->m_tracingManager.Trace();
		//if (finished || LineModeIsIterative(volumeData.m_traceParams.m_lineMode)) 
		//	g_heatMapManager.ProcessLines(volumeData.m_tracingManager.GetResult());

		if (finished)
		{
			volumeData->m_isTracing = false;
			volumeData->m_timerTracing.Stop();
			shouldRedraw = true;
		}
		//else if (g_showPreview)
		else
		{
			volumeData->m_tracingManager.BuildIndexBuffer();
			shouldRedraw = true;
		}
	}
	//else if (s_isFiltering)
	//{
	//	bool finished = g_filteringManager.Filter();

	//	if (finished)
	//	{
	//		s_isFiltering = false;
	//		shouldRedraw = true;
	//	}
	//}

	return shouldRedraw;
}

bool FlowVisTool::RenderTracingResults()
{
	assert(g_renderingManager.IsCreated());

	cudaArray* pTfArray = nullptr;

	RenderingManager::eRenderState stateRenderer = g_renderingManager.Render(
		g_volumes,
		g_renderingParams,
		g_viewParams,
		g_stereoParams,
		g_raycastParams,
		g_lineBuffers,
		g_ballBuffers,
		g_ballRadius,
		&g_heatMapManager,
		pTfArray);

	if (stateRenderer == RenderingManager::STATE_ERROR)
		printf("RenderingManager::Render returned STATE_ERROR.\n");

	return true;
}

bool FlowVisTool::CheckForChanges()
{
	clock_t curTime = clock();

	//static std::string volumeFilePrev = "";
	//bool volumeChanged = (g_volumes[0]->m_volume->GetFilename() != volumeFilePrev);
	//volumeFilePrev = g_volumes[0]->m_volume->GetFilename();
	//m_redraw = m_redraw || volumeChanged;
	//m_retrace = m_retrace || volumeChanged;


	//static int timestepPrev = -1;
	//bool timestepChanged = (g_volumes[0]->m_volume->GetCurNearestTimestepIndex() != timestepPrev);
	//timestepPrev = g_volumes[0]->m_volume->GetCurNearestTimestepIndex();
	//m_redraw = m_redraw || timestepChanged;
	//m_retrace = m_retrace || timestepChanged;


	static float renderBufferSizeFactorPrev = 0.0f;
	bool renderBufferSizeChanged = (g_renderingParams.m_renderBufferSizeFactor != renderBufferSizeFactorPrev);
	renderBufferSizeFactorPrev = g_renderingParams.m_renderBufferSizeFactor;
	if (renderBufferSizeChanged)
	{
		ResizeRenderBuffer();
		g_lastRenderParamsUpdate = curTime;
	}


	static bool renderDomainBoxPrev = g_renderingManager.m_renderDomainBox;
	static bool renderBrickBoxesPrev = g_renderingManager.m_renderBrickBoxes;
	static bool renderClipBoxPrev = g_renderingManager.m_renderClipBox;
	static bool renderSeedBoxPrev = g_renderingManager.m_renderSeedBox;
	if (renderDomainBoxPrev != g_renderingManager.m_renderDomainBox || renderBrickBoxesPrev != g_renderingManager.m_renderBrickBoxes || renderClipBoxPrev != g_renderingManager.m_renderClipBox || renderSeedBoxPrev != g_renderingManager.m_renderSeedBox)
	{
		renderDomainBoxPrev = g_renderingManager.m_renderDomainBox;
		renderBrickBoxesPrev = g_renderingManager.m_renderBrickBoxes;
		renderClipBoxPrev = g_renderingManager.m_renderClipBox;
		renderSeedBoxPrev = g_renderingManager.m_renderSeedBox;
		g_raycastParams.m_redraw = true;
	}


	static ProjectionParams projParamsPrev;
	bool projParamsChanged = (g_projParams != projParamsPrev);
	projParamsPrev = g_projParams;
	g_raycastParams.m_redraw = g_raycastParams.m_redraw || projParamsChanged;

	static Range1D rangePrev;
	bool rangeChanged = (g_cudaDevices[g_primaryCudaDeviceIndex].range != rangePrev);
	rangePrev = g_cudaDevices[g_primaryCudaDeviceIndex].range;
	g_raycastParams.m_redraw = g_raycastParams.m_redraw || rangeChanged;

	if (projParamsChanged || rangeChanged)
	{
		// cancel rendering
		g_raycasterManager.CancelRendering();
		if (g_useAllGPUs)
		{
			for (size_t i = 0; i < g_cudaDevices.size(); i++)
			{
				if (g_cudaDevices[i].pThread == nullptr) continue;
				g_cudaDevices[i].pThread->CancelRendering();
			}
		}

		// forward the new params
		g_raycasterManager.SetProjectionParams(g_projParams, g_cudaDevices[g_primaryCudaDeviceIndex].range);
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
	g_raycastParams.m_redraw = g_raycastParams.m_redraw || stereoParamsChanged;

	if (stereoParamsChanged)
		g_lastRenderParamsUpdate = curTime;

	static ViewParams viewParamsPrev;
	bool viewParamsChanged = (g_viewParams != viewParamsPrev);
	viewParamsPrev = g_viewParams;
	g_raycastParams.m_redraw = g_raycastParams.m_redraw || viewParamsChanged;

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
	//if (volumeChanged || timestepChanged || filterParamsChanged)
	if (filterParamsChanged)
	{
		m_restartFiltering = true;
		g_raycastParams.m_redraw = true;
	}

	//static ParticleTraceParams particleTraceParamsPrev;
	//bool particleTraceParamsChanged = g_volumes[0]->m_traceParams.hasChangesForRetracing(particleTraceParamsPrev);
	////bool particleTraceParamsChanged = (g_particleTraceParams != particleTraceParamsPrev);
	//bool seedBoxChanged = (g_volumes[0]->m_traceParams.m_seedBoxMin != particleTraceParamsPrev.m_seedBoxMin || g_volumes[0]->m_traceParams.m_seedBoxSize != particleTraceParamsPrev.m_seedBoxSize);
	//particleTraceParamsPrev = g_volumes[0]->m_traceParams;
	//m_retrace = m_retrace || particleTraceParamsChanged;
	//m_redraw = m_redraw || seedBoxChanged;

	//if (particleTraceParamsChanged)
	//{
	//	g_lastTraceParamsUpdate = curTime;
	//	g_lastRenderParamsUpdate = curTime;
	//}

	// heat map parameters
//	static HeatMapParams heatMapParamsPrev;
//	static bool heatMapDoubleRedraw = false;
//	bool debugHeatMapPrint = false;
//	//m_retrace = m_retrace || g_heatMapParams.HasChangesForRetracing(heatMapParamsPrev, g_particleTraceParams);
//	//m_redraw = m_redraw || g_heatMapParams.HasChangesForRedrawing(heatMapParamsPrev);
//	if (g_heatMapParams.HasChangesForRetracing(heatMapParamsPrev, g_volumes[0]->m_traceParams))
//	{
//		m_retrace = true;
//		std::cout << "heat map has changes for retracing" << std::endl;
//		debugHeatMapPrint = true;
//		heatMapDoubleRedraw = true;
//	}
//	if (g_heatMapParams.HasChangesForRedrawing(heatMapParamsPrev))
//	{
//		m_redraw = true;
//		std::cout << "heat map has changes for redrawing" << std::endl;
//		debugHeatMapPrint = true;
//		heatMapDoubleRedraw = true;
//	}
//	heatMapParamsPrev = g_heatMapParams;
//	g_heatMapManager.SetParams(g_heatMapParams);
//
//#if 0
//	if (debugHeatMapPrint)
//		g_heatMapManager.DebugPrintParams();
//#endif
//
//	if (heatMapDoubleRedraw && !m_redraw)
//	{
//		//Hack: For some reasons, the heat map manager only applies changes after the second rendering
//		//Or does the RenderManager not update the render targets?
//		m_redraw = true;
//		heatMapDoubleRedraw = false;
//	}


	// clear particle tracer if something relevant changed
	//if (m_retrace && s_isTracing)
	//{
	//	g_volumes[0]->m_tracingManager.CancelTracing();
	//	s_isTracing = false;
	//	g_volumes[0]->m_tracingManager.ClearResult();
	//	m_redraw = true;
	//	g_lastRenderParamsUpdate = curTime;
	//}

	static RaycastParams raycastParamsPrev = g_raycastParams;
	bool raycastParamsChanged = (g_raycastParams != raycastParamsPrev);
	raycastParamsPrev = g_raycastParams;
	g_raycastParams.m_redraw = g_raycastParams.m_redraw || raycastParamsChanged;

	if (!g_raycastParams.m_raycastingEnabled)
	{
		g_raycastParams.m_redraw = false;
		StopRaycasting();
	}

	return g_raycastParams.m_redraw;
}

#ifdef Single
void FlowVisTool::Filtering()
{
	// start filtering if required
	bool needFilteredBricks = g_filterParams.HasNonZeroRadius();
	if (needFilteredBricks && !s_isFiltering && g_filteringManager.GetResultCount() == 0)
	{
		g_raycasterManager.CancelRendering();
		if (g_useAllGPUs)
		{
			for (size_t i = 0; i < g_cudaDevices.size(); i++)
			{
				if (g_cudaDevices[i].pThread == nullptr) continue;
				g_cudaDevices[i].pThread->CancelRendering();
			}
		}
		// release other resources - we'll need a lot of memory for filtering
		g_volumes[0]->m_tracingManager.ReleaseResources();
		g_raycasterManager.ReleaseResources();

		assert(g_filteringManager.IsCreated());

		s_isFiltering = g_filteringManager.StartFiltering(*g_volumes[0]->m_volume, g_filterParams);
	}
}

void FlowVisTool::Tracing()
{
	// set parameters even if tracing is currently enabled.
	// This allows changes to the parameters in the particle mode, even if they are currently running
	if (!m_retrace && s_isTracing)
		g_volumes[0]->m_tracingManager.SetParams(g_volumes[0]->m_traceParams);

	// start particle tracing if required
	float timeSinceTraceUpdate = float(clock() - g_lastTraceParamsUpdate) / float(CLOCKS_PER_SEC);
	bool traceDelayPassed = (timeSinceTraceUpdate >= g_startWorkingDelay);
	//bool traceStartNow = !s_isFiltering && g_particleRenderParams.m_linesEnabled && traceDelayPassed; //TODO: enable tracing also when rendering is disabled?
	bool traceStartNow = !s_isFiltering && traceDelayPassed;
	if (m_retrace && traceStartNow)
	{
		g_raycasterManager.CancelRendering();
		if (g_useAllGPUs)
		{
			for (size_t i = 0; i < g_cudaDevices.size(); i++)
			{
				if (g_cudaDevices[i].pThread == nullptr) continue;
				g_cudaDevices[i].pThread->CancelRendering();
			}
		}
		// release other resources - we're gonna need a lot of memory
		g_raycasterManager.ReleaseResources();
		g_filteringManager.ReleaseResources();

		assert(g_volumes[0]->m_tracingManager.IsCreated());

		g_volumes[0]->m_tracingManager.ClearResult();
		g_volumes[0]->m_tracingManager.ReleaseResources();

		s_isTracing = g_volumes[0]->m_tracingManager.StartTracing(*g_volumes[0]->m_volume, g_volumes[0]->m_traceParams, g_flowGraph);
		g_timerTracing.Start();

		//notify the heat map manager
		g_heatMapManager.SetVolumeAndReset(*g_volumes[0]->m_volume);

		m_retrace = false;
	}


	const std::vector<const TimeVolumeIO::Brick*>& bricksToLoad =	s_isFiltering ? g_filteringManager.GetBricksToLoad() :
		(s_isTracing ? g_volumes[0]->m_tracingManager.GetBricksToLoad() : g_raycasterManager.GetBricksToLoad());
	//TODO when nothing's going on, load bricks in any order? (while memory is available etc..)
	g_volumes[0]->m_volume->UpdateLoadingQueue(bricksToLoad);
	g_volumes[0]->m_volume->UnloadLRUBricks();

	//Check if tracing is done and if so, start rendering
	if (s_isTracing && !g_particleTracingPaused)
	{
		//std::cout << "Trace" << std::endl;
		bool finished = g_volumes[0]->m_tracingManager.Trace();
		if (finished || LineModeIsIterative(g_volumes[0]->m_traceParams.m_lineMode))
			g_heatMapManager.ProcessLines(g_volumes[0]->m_tracingManager.GetResult());

		if (finished)
		{
			s_isTracing = false;
			g_timerTracing.Stop();
			m_redraw = true;
		}
		else if (g_showPreview)
		{
			g_volumes[0]->m_tracingManager.BuildIndexBuffer();
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

std::vector<uint> s_timestampLastUpdate;

bool FlowVisTool::Rendering()
{
	// last time the result image from each GPU was taken

	bool renderingUpdated = false;
	if (m_redraw)
	{
		// cancel rendering of current image
		g_raycasterManager.CancelRendering();
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
			g_volumes[0]->m_tracingManager.ReleaseResources();

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
		LineBuffers* pTracedLines = g_volumes[0]->m_tracingManager.GetResult().get();
		if (pTracedLines != nullptr)
		{
			lineBuffers.push_back(pTracedLines);
		}

		RenderingManager::eRenderState stateRenderer = g_renderingManager.Render(
			g_volumes[0]->m_tracingManager.IsTracing(),
			*g_volumes[0]->m_volume,
			g_viewParams, 
			g_stereoParams,
			g_volumes[0]->m_traceParams,
			g_particleRenderParams,
			g_raycastParams,
			lineBuffers,
			g_ballBuffers, 
			g_ballRadius,
			&g_heatMapManager,
			pTfArray,
			g_volumes[0]->m_tracingManager.m_dpParticles);

		if (stateRenderer == RenderingManager::STATE_ERROR)
			printf("RenderingManager::Render returned STATE_ERROR.\n");

		RaycasterManager::eRenderState stateRaycaster = RaycasterManager::STATE_DONE;
		if (g_raycastParams.m_raycastingEnabled && !linesOnly)
		{
			g_renderingManager.CopyDepthTexture(m_d3dDeviceContex, g_raycasterManager.m_pDepthTexCopy);
			stateRaycaster = g_raycasterManager.StartRendering(*g_volumes[0]->m_volume, g_viewParams, g_stereoParams, g_raycastParams, g_filteringManager.GetResults(), pTfArray);
		}

		if (stateRaycaster == RaycasterManager::STATE_ERROR)
		{
			printf("RaycasterManager::StartRendering returned STATE_ERROR.\n");
			s_isRaycasting = false;
		}
		else
		{
			s_isRaycasting = true; // set to true even if STATE_DONE - other GPUs might have something to do
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

	// default to "true", set to false if any GPU is not finished
	bool raycastingFinished = true;

	//TODO only call g_renderingManager.Render() when renderDelayPassed?
	//float timeSinceRenderUpdate = float(curTime - g_lastRenderParamsUpdate) / float(CLOCKS_PER_SEC);
	//bool renderDelayPassed = (timeSinceRenderUpdate >= g_startWorkingDelay);

	if (s_isRaycasting)
	{
		if (g_raycasterManager.IsRendering())
		{
			// render next brick on primary GPU
			raycastingFinished = (g_raycasterManager.Render() == RaycasterManager::STATE_DONE);
			renderingUpdated = true;
		}

		// if primary GPU is done, check if other threads are finished as well
		if (raycastingFinished && g_useAllGPUs)
		{
			for (size_t i = 0; i < g_cudaDevices.size(); i++)
			{
				if (g_cudaDevices[i].pThread == nullptr) continue;

				raycastingFinished = raycastingFinished && !g_cudaDevices[i].pThread->IsWorking();
				renderingUpdated = true;
			}
		}

		if (raycastingFinished)
		{
			s_isRaycasting = false;
			g_timerRendering.Stop();
		}
	}

	// if this was the last brick, or we want to see unfinished images, copy from raycast target into finished image tex
	if (renderingUpdated && (raycastingFinished || g_showPreview))
		BlitRenderingResults();

	return renderingUpdated;
}
#endif


bool FlowVisTool::ResizeViewport(int width, int height)
{
	std::cout << "Scene window resize: " << width << ", " << height << std::endl;

	g_renderTexture.Release();
	g_renderTexture.Initialize(m_d3dDevice, width, height);

	g_raycastParams.m_redraw = true;

	g_renderingParams.m_windowSize.x() = width;
	g_renderingParams.m_windowSize.y() = height;


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
	desc.Width = g_renderingParams.m_windowSize.x();
	desc.Height = g_renderingParams.m_windowSize.y();
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
	desc.Width = g_renderingParams.m_windowSize.x();
	desc.Height = g_renderingParams.m_windowSize.y();
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

bool FlowVisTool::ResizeRenderBuffer()
{
	g_projParams.m_imageWidth = uint(g_renderingParams.m_windowSize.x() * g_renderingParams.m_renderBufferSizeFactor);
	g_projParams.m_imageHeight = uint(g_renderingParams.m_windowSize.y() * g_renderingParams.m_renderBufferSizeFactor);
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

			g_cudaDevices[index].pThread = new WorkerThread(g_cudaDevices[index].device, *g_volumes[m_selectedVolume]->m_volume);
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


void FlowVisTool::BlitRaycastingResults()
{
	//-----------------------------------------------------------
	// COMBINE RESULTS AND DRAW ON SCREEN
	//-----------------------------------------------------------
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
		g_screenEffect.m_pTexVariable->SetResource(g_raycasterManager.GetRaycastSRV());
		g_screenEffect.m_pTechnique->GetPassByIndex(ScreenEffect::BlitBlendOver)->Apply(0, m_d3dDeviceContex);
		m_d3dDeviceContex->Draw(4, 0);

		// restore viewport
		m_d3dDeviceContex->RSSetViewports(1, &viewportOld);
	}
#ifdef Single
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

				m_d3dDeviceContex->CopySubresourceRegion(g_pRenderBufferTempTex, 0, left, 0, 0, g_raycasterManager.GetRaycastTex(), 0, &box);
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
#endif

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


bool FlowVisTool::ShouldStartFiltering()
{
	bool needFilteredBricks = g_filterParams.HasNonZeroRadius();

	return	CanFilter() && needFilteredBricks && m_restartFiltering;
}

bool FlowVisTool::ShouldStartTracing(FlowVisToolVolumeData* volumeData)
{
	float timeSinceTraceUpdate = float(clock() - volumeData->m_lastTraceParamsUpdate) / float(CLOCKS_PER_SEC);
	bool traceDelayPassed = (timeSinceTraceUpdate >= g_startWorkingDelay);

	return CanTrace() && volumeData->m_retrace && traceDelayPassed;
}

bool FlowVisTool::ShouldStartRaycasting()
{
	return CanRaycast() && g_raycastParams.m_raycastingEnabled && g_raycastParams.m_redraw;
}


bool FlowVisTool::CanFilter()
{
	return true;
}

bool FlowVisTool::CanTrace()
{
	return !m_isFiltering;
}

bool FlowVisTool::CanRaycast()
{
	bool anyIsTracing = std::any_of(g_volumes.begin(), g_volumes.end(), [](FlowVisToolVolumeData* e) { return e->m_isTracing; });

	return !m_isFiltering && !anyIsTracing;
}


void FlowVisTool::StartFiltering()
{
	assert(CanFilter());
	assert(g_filteringManager.IsCreated());
	assert(m_selectedVolume >= 0 && m_selectedVolume < g_volumes.size());

	StopRaycasting();

	StopTracing();
	
	StopFiltering();

	// Release resources - we'll need a lot of memory for filtering.
	g_raycasterManager.ReleaseResources();

	for (size_t i = 0; i < g_volumes.size(); i++)
		g_volumes[i]->m_tracingManager.ReleaseResources();

	// Go!
	m_isFiltering = g_filteringManager.StartFiltering(*g_volumes[m_selectedVolume]->m_volume, g_filterParams);

	m_restartFiltering = false;
}

void FlowVisTool::StartTracing(FlowVisToolVolumeData* volumeData, FlowGraph& flowGraph)
{
	assert(CanTrace());
	assert(volumeData);
	assert(volumeData->m_volume);
	assert(volumeData->m_tracingManager.IsCreated());

	StopRaycasting();

	// release other resources - we're gonna need a lot of memory
	g_raycasterManager.ReleaseResources();
	g_filteringManager.ReleaseResources();

	volumeData->m_tracingManager.ClearResult();
	volumeData->m_tracingManager.ReleaseResources();

	volumeData->m_isTracing = volumeData->m_tracingManager.StartTracing(*volumeData->m_volume, volumeData->m_traceParams, flowGraph);
	volumeData->m_timerTracing.Start();

	//g_heatMapManager.SetVolumeAndReset(*g_volumes[0]); //notify the heat map manager

	volumeData->m_retrace = false;
}

void FlowVisTool::StartRaycasting()
{
	assert(CanRaycast());
	assert(!g_volumes.empty());
	assert(m_selectedVolume >= 0 && m_selectedVolume < g_volumes.size());

	StopRaycasting();

	//// Release other resources if possible
	//if (!m_isFiltering)
	//	g_filteringManager.ReleaseResources();
	//for (size_t i = 0; i < g_volumes.size(); i++)
	//	if (!g_volumes[i]->m_isTracing)
	//		g_volumes[i]->m_tracingManager.ReleaseResources();
	
	g_renderingManager.CopyDepthTexture(m_d3dDeviceContex, g_raycasterManager.m_pDepthTexCopy);

	cudaArray* pTfArray = nullptr;
	RaycasterManager::eRenderState stateRaycaster = RaycasterManager::STATE_DONE;

	stateRaycaster = g_raycasterManager.StartRendering(*g_volumes[m_selectedVolume]->m_volume, g_viewParams, g_stereoParams, g_raycastParams, g_filteringManager.GetResults(), pTfArray);

	if (stateRaycaster == RaycasterManager::STATE_ERROR)
	{
		printf("RaycasterManager::StartRendering returned STATE_ERROR.\n");
		m_isRaycasting = false;
	}
	else
	{
		m_isRaycasting = true; // set to true even if STATE_DONE - other GPUs might have something to do

		//if (g_useAllGPUs)
		//{
		//	for (size_t i = 0; i < g_cudaDevices.size(); i++)
		//	{
		//		if (g_cudaDevices[i].pThread != nullptr)
		//		{
		//			int primaryDevice = g_cudaDevices[g_primaryCudaDeviceIndex].device;
		//			g_cudaDevices[i].pThread->StartRendering(g_filteringManager.GetResults(), g_viewParams, g_stereoParams, g_raycastParams, pTfArray, primaryDevice);
		//		}
		//	}
		//}

		//s_timestampLastUpdate.clear();
		//s_timestampLastUpdate.resize(g_cudaDevices.size(), 0);

		g_timerRendering.Start();
	}

	g_raycastParams.m_redraw = false;
}


void FlowVisTool::StopFiltering()
{
	g_filteringManager.ClearResult();
	m_isFiltering = false;
}

void FlowVisTool::StopTracing()
{
	for (size_t i = 0; i < g_volumes.size(); i++)
	{
		if (g_volumes[i]->m_isTracing)
			g_volumes[i]->m_tracingManager.CancelTracing();
		g_volumes[i]->m_isTracing = false;
	}
}

void FlowVisTool::StopRaycasting()
{
	g_raycasterManager.CancelRendering();
	// FIXME: cancel raycasting here for g_useAllGPUs.
	m_isRaycasting = false;
}


void FlowVisTool::UpdateBricks(TimeVolume* volume, const std::vector<const TimeVolumeIO::Brick*>& bricks)
{
	//TODO when nothing's going on, load bricks in any order? (while memory is available etc..)
	volume->UpdateLoadingQueue(bricks);
	volume->UnloadLRUBricks();
}

void FlowVisTool::UpdateFiltering()
{
	assert(m_isFiltering);

	UpdateBricks(g_volumes[m_selectedVolume]->m_volume, g_filteringManager.GetBricksToLoad());

	bool finished = g_filteringManager.Filter();

	if (finished)
		m_isFiltering = false;
}

void FlowVisTool::UpdateTracing(FlowVisToolVolumeData* volumeData)
{
	assert(volumeData->m_isTracing && !volumeData->m_tracingPaused);

	UpdateBricks(volumeData->m_volume, volumeData->m_tracingManager.GetBricksToLoad());

	bool finished = volumeData->m_tracingManager.Trace();
	//if (finished || LineModeIsIterative(volumeData.m_traceParams.m_lineMode)) 
	//	g_heatMapManager.ProcessLines(volumeData.m_tracingManager.GetResult());

	if (finished)
	{
		volumeData->m_isTracing = false;
		volumeData->m_timerTracing.Stop();
		g_raycastParams.m_redraw = true;
	}
	else
		volumeData->m_tracingManager.BuildIndexBuffer();
}

void FlowVisTool::UpdateRaycasting()
{
	assert(m_isRaycasting);

	UpdateBricks(g_volumes[m_selectedVolume]->m_volume, g_raycasterManager.GetBricksToLoad());

	bool raycastingFinished = true;

	if (g_raycasterManager.IsRendering())
	{
		// render next brick on primary GPU
		raycastingFinished = (g_raycasterManager.Render() == RaycasterManager::STATE_DONE);
	}

	// if primary GPU is done, check if other threads are finished as well
	if (raycastingFinished && g_useAllGPUs)
	{
		for (size_t i = 0; i < g_cudaDevices.size(); i++)
		{
			if (g_cudaDevices[i].pThread == nullptr) 
				continue;

			raycastingFinished = raycastingFinished && !g_cudaDevices[i].pThread->IsWorking();
		}
	}

	if (raycastingFinished)
	{
		m_isRaycasting = false;
		g_timerRendering.Stop();
	}
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