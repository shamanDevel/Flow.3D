#include "TracingManager.h"

#include <cassert>
#include <fstream>
#include <random>
#include <chrono>

#include "cudaUtil.h"
#include "cudaTum3D.h"

#include "BrickUpload.h"
#include "TextureCPU.h"

using namespace tum3D;


TextureCPU<float4> g_volume;


namespace
{
	inline bool InsideOfDomain(float3 pos, Vec3f volumeHalfSize)
	{
		return pos.x >= -volumeHalfSize.x() && pos.y >= -volumeHalfSize.y() && pos.z >= -volumeHalfSize.z()
			&& pos.x <=  volumeHalfSize.x() && pos.y <=  volumeHalfSize.y() && pos.z <=  volumeHalfSize.z();
	}

	inline bool OutsideOfDomain(float3 pos, Vec3f volumeHalfSize)
	{
		return !InsideOfDomain(pos, volumeHalfSize);
	}
}


TracingManager::TracingManager()
	: m_brickSlotCountMax(1024), m_timeSlotCountMax(8)
	, m_roundsPerFrame(0), m_uploadsPerFrame(0)
	, m_isCreated(false), m_pCompressShared(nullptr), m_pCompressVolume(nullptr), m_pDevice(nullptr)
	, m_pBrickToSlot(nullptr), m_pSlotTimestepMin(nullptr), m_pSlotTimestepMax(nullptr), m_brickIndexUploadEvent(0)
	, m_pBrickRequestCounts(nullptr), m_pBrickTimestepMins(nullptr), m_brickRequestsDownloadEvent(0)
	, m_currentTimestamp(0)
	, m_brickSlotCount(0, 0, 0)
	, m_bricksToDoDirty(true), m_bricksToDoPrioritiesDirty(true)
	, m_dpLineCheckpoints(nullptr), m_dpLineVertexCounts(nullptr)
	, m_pResult(nullptr), m_indexBufferDirty(true)
	, m_pVolume(nullptr), m_pFlowGraph(nullptr), m_timestepMax(0)
	, m_progress(0.0f)
	, m_verbose(false)
	, m_engine(std::chrono::system_clock::now().time_since_epoch().count())
	, m_numRejections(20)
	, m_seedManyParticles(false)
{
	m_brickIndexGPU.Init();
	m_brickRequestsGPU.Init();
}

TracingManager::~TracingManager()
{
	assert(!IsCreated());
}


bool TracingManager::Create(GPUResources* pCompressShared, CompressVolumeResources* pCompressVolume, ID3D11Device* pDevice)
{
	std::cout << "Creating TracingManager..." << std::endl;

	m_pCompressShared = pCompressShared;
	m_pCompressVolume = pCompressVolume;
	m_pDevice = pDevice;

	cudaSafeCall(cudaHostRegister(&m_volumeInfoGPU, sizeof(m_volumeInfoGPU), cudaHostRegisterDefault));

	cudaSafeCall(cudaHostRegister(&m_brickIndexGPU, sizeof(m_brickIndexGPU), cudaHostRegisterDefault));
	cudaSafeCall(cudaEventCreate(&m_brickIndexUploadEvent, cudaEventDisableTiming));
	cudaSafeCall(cudaEventRecord(m_brickIndexUploadEvent));

	cudaSafeCall(cudaHostRegister(&m_brickRequestsGPU, sizeof(m_brickRequestsGPU), cudaHostRegisterDefault));
	cudaSafeCall(cudaEventCreate(&m_brickRequestsDownloadEvent, cudaEventDisableTiming));
	cudaSafeCall(cudaEventRecord(m_brickRequestsDownloadEvent));

	if(!m_integrator.Create())
	{
		Release();
		return false;
	}

	m_isCreated = true;

	std::cout << "TracingManager created." << std::endl;

	return true;
}

void TracingManager::Release()
{
	if(!m_isCreated) return;

	CancelTracing();

	m_timerBuildIndexBuffer.ReleaseTimers();
	m_timerIntegrate.ReleaseTimers();
	m_timerUploadDecompress.ReleaseTimers();

	m_integrator.Release();

	if(m_brickRequestsDownloadEvent)
	{
		cudaSafeCall(cudaEventDestroy(m_brickRequestsDownloadEvent));
		m_brickRequestsDownloadEvent = 0;
	}
	cudaSafeCall(cudaHostUnregister(&m_brickRequestsGPU));

	if(m_brickIndexUploadEvent)
	{
		cudaSafeCall(cudaEventDestroy(m_brickIndexUploadEvent));
		m_brickIndexUploadEvent = 0;
	}
	cudaSafeCall(cudaHostUnregister(&m_brickIndexGPU));

	cudaSafeCall(cudaHostUnregister(&m_volumeInfoGPU));

	ReleaseResultResources();
	ReleaseParamDependentResources();
	ReleaseVolumeDependentResources();

	m_pDevice = nullptr;
	m_pCompressVolume = nullptr;
	m_pCompressShared = nullptr;

	m_isCreated = false;
}

void TracingManager::SetParams(const ParticleTraceParams& traceParams)
{
	m_traceParams = traceParams;
	SetCheckpointTimeStep(m_traceParams.m_advectDeltaT);
}

bool TracingManager::StartTracing(const TimeVolume& volume, const ParticleTraceParams& traceParams, const FlowGraph& flowGraph)
{
	if(!volume.IsOpen()) return false;

	if(IsTracing()) CancelTracing();

	//HACK for now, release resources first
	ReleaseResources();
	ReleaseResultResources();

	printf("\n----------------------------------------------------------------------\nTracingManager::StartTracing\n");

	m_pVolume = &volume;
	m_traceParams = traceParams;
	m_pFlowGraph = &flowGraph;
	m_progress = 0.0f;

	m_currentTimestamp = 0;

	//TODO recreate only if something relevant changed
	if(FAILED(CreateParamDependentResources()))
	{
		MessageBoxA(nullptr, "TracingManager::StartRendering: Failed creating param-dependent resources! (probably not enough GPU memory)", "Fail", MB_OK | MB_ICONINFORMATION);
		CancelTracing();
		return false;
	}
	if(FAILED(CreateResultResources()))
	{
		MessageBoxA(nullptr, "TracingManager::StartRendering: Failed creating result resources! (probably not enough GPU memory)", "Fail", MB_OK | MB_ICONINFORMATION);
		CancelTracing();
		return false;
	}
	// create volume-dependent resources last - they will take up all available GPU memory
	if(FAILED(CreateVolumeDependentResources()))
	{
		MessageBoxA(nullptr, "TracingManager::StartRendering: Failed creating volume-dependent resources! (probably not enough GPU memory)", "Fail", MB_OK | MB_ICONINFORMATION);
		CancelTracing();
		return false;
	}


	// probably no need to sync on the previous upload here..?
	m_volumeInfoGPU.Fill(m_pVolume->GetInfo());
	m_volumeInfoGPU.Upload(m_traceParams.m_cpuTracing);

	m_brickIndexGPU.Upload(m_traceParams.m_cpuTracing);
	m_brickRequestsGPU.Upload(m_traceParams.m_cpuTracing);

	m_integrator.SetVolumeInfo(m_pVolume->GetInfo());

	float spawnTime = GetLineSpawnTime();

	LineInfo lineInfo(spawnTime, m_traceParams.m_lineCount, m_dpLineCheckpoints, nullptr, m_dpLineVertexCounts, m_traceParams.m_lineLengthMax);

	m_integrator.ForceParamUpdate(m_traceParams, lineInfo);

	UpdateFrameBudgets();

	//create initial checkpoints
	CreateInitialCheckpoints(spawnTime);


	if (m_traceParams.m_ftleEnabled)
	{
		std::vector<SimpleParticleVertexDeltaT> particles(m_traceParams.m_lineCount);

		for (size_t i = 0; i < m_checkpoints.size(); i++)
		{
			particles[i].DeltaT = m_checkpoints[i].DeltaT;
			particles[i].Position = m_checkpoints[i].Position;
			particles[i].Time = m_checkpoints[i].Time;
		}

		cudaSafeCall(cudaMemcpy(m_dpParticles, particles.data(), particles.size() * sizeof(SimpleParticleVertexDeltaT), cudaMemcpyHostToDevice));
	}


	//and upload the seed texture (if available)
	if (traceParams.m_seedTexture.m_colors != NULL) 
	{
		IntegratorTimeInCell::Upload(m_cellTextureGPU, traceParams.m_seedTexture.m_colors, traceParams.m_seedTexture.m_width, traceParams.m_seedTexture.m_height);
	}

	// compute last timestep that will be needed for integration
	m_timestepMax = GetLineFloorTimestepIndex(spawnTime);
	if(LineModeIsTimeDependent(m_traceParams.m_lineMode))
	{
		float timeMax = spawnTime + m_traceParams.m_lineAgeMax;
		m_timestepMax = GetLineFloorTimestepIndex(timeMax) + 1;
	}
	m_timestepMax = min(m_timestepMax, m_pVolume->GetTimestepCount() - 1);


	m_stats.Clear();
	m_stats.DiskBusyTimeMS = m_pVolume->GetTotalDiskBusyTimeMS();
	m_stats.DiskBytesLoaded = m_pVolume->GetTotalLoadedBytes();
	m_bricksLoaded.clear();


	if (m_traceParams.HeuristicUseFlowGraph() && !m_pFlowGraph->IsBuilt())
		std::cout << "TracingManager: Warning: heuristic wants to use flow graph, but it's not built\n" << std::endl;

	// start timing
	m_timerUploadDecompress.ResetTimers();
	m_timerIntegrate.ResetTimers();
	m_timerBuildIndexBuffer.ReleaseTimers();

	m_timings.IntegrateCPU = 0.0f;

	m_timings.WaitDiskWall = 0.0f;

	m_timerTrace.Start();

	if (LineModeIsIterative(m_traceParams.m_lineMode)) 
		StartTracingParticlesIteratively();
	else 
	{
		// run integration kernel immediately to populate brick request list
		TraceRound();
		UpdateBricksToLoad();
	}

	return true;
}

void TracingManager::CreateInitialCheckpoints(float spawnTime)
{
	m_checkpoints.resize(m_traceParams.m_lineCount);


	float3 seedBoxMin = make_float3(m_traceParams.m_seedBoxMin);
	float3 seedBoxSize = make_float3(m_traceParams.m_seedBoxSize);

	if (m_traceParams.m_upsampledVolumeHack)
	{
		// upsampled volume is offset by half a grid spacing...
		float gridSpacingWorld = 2.0f / float(m_pVolume->GetVolumeSize().maximum());
		seedBoxMin -= 0.5f * gridSpacingWorld;
	}


	switch (m_traceParams.m_seedPattern)
	{
	case ParticleTraceParams::eSeedPattern::REGULAR_GRID:
		CreateInitialCheckpointsRegularGrid(seedBoxMin, seedBoxSize);
		break;
	case ParticleTraceParams::eSeedPattern::FTLE:
		CreateInitialCheckpointsFTLE(seedBoxMin, seedBoxSize);
		break;
	case ParticleTraceParams::eSeedPattern::RANDOM:
		CreateInitialCheckpointsRandom(seedBoxMin, seedBoxSize);
		break;
	default:
		CreateInitialCheckpointsRandom(seedBoxMin, seedBoxSize);
		break;
	}

	// create initial checkpoints
	//float3 seedBoxMin = make_float3(m_traceParams.m_seedBoxMin);
	//float3 seedBoxSize = make_float3(m_traceParams.m_seedBoxSize);
	//// create random seed positions
	//if (m_traceParams.m_upsampledVolumeHack)
	//{
	//	// upsampled volume is offset by half a grid spacing...
	//	float gridSpacingWorld = 2.0f / float(m_pVolume->GetVolumeSize().maximum());
	//	seedBoxMin -= 0.5f * gridSpacingWorld;
	//}
//#ifdef GRID
//
//	int n = std::floor(std::sqrt(m_traceParams.m_lineCount));
//
//#endif
//
//
//	for (uint i = 0; i < m_traceParams.m_lineCount; i++)
//	{
//
//#ifdef GRID
//
//		uint ii = i % n;
//		uint jj = std::floor(i / (float)n);
//
//		m_checkpoints[i].Position = make_float3(seedBoxMin.x + seedBoxSize.x * ii / (float)n,
//			seedBoxMin.y + seedBoxSize.y * jj / (float)n,
//			seedBoxMin.z);
//#else
//
//		if (m_traceParams.m_seedTexture.m_colors == NULL || m_traceParams.m_seedTexture.m_picked.empty())
//		{
//			//seed directly from the seed box
//			m_checkpoints[i].Position = GetRandomVectorInBox(seedBoxMin, seedBoxSize, m_rng, m_engine);
//		}
//		else {
//			//sample texture until a point with the same color is found
//			m_checkpoints[i].Position = make_float3(Vec3f(-10000, -10000, -10000)); //somewhere outside, so that the point is deleted if no sample could be found
//			for (int j = 0; j < m_numRejections; ++j) {
//				float3 pos = GetRandomVectorInBox(seedBoxMin, seedBoxSize, m_rng, m_engine);
//				Vec3f posAsVec = Vec3f(&pos.x);
//				Vec3f posInVolume = (posAsVec + m_pVolume->GetVolumeHalfSizeWorld()) / (2 * m_pVolume->GetVolumeHalfSizeWorld());
//				int texX = (int)(posInVolume.x() * m_traceParams.m_seedTexture.m_width);
//				int texY = (int)(posInVolume.y() * m_traceParams.m_seedTexture.m_height);
//				texY = m_traceParams.m_seedTexture.m_height - texY - 1;
//				if (texX < 0 || texY < 0 || texX >= m_traceParams.m_seedTexture.m_width || texY >= m_traceParams.m_seedTexture.m_height) {
//					continue;
//				}
//				unsigned int color = m_traceParams.m_seedTexture.m_colors[texX + texY * m_traceParams.m_seedTexture.m_height];
//				if (m_traceParams.m_seedTexture.m_picked.find(color) != m_traceParams.m_seedTexture.m_picked.end()) {
//					//we found it
//					m_checkpoints[i].Position = pos;
//					break;
//				}
//			}
//		}
//
//#endif
//
//		/*if (!InsideOfDomain(m_checkpoints[i].Position, Vec3f(1.0f, 1.0f, 1.0f)))
//		printf("WARNING: seed %i is outside of domain!\n", i);*/
//		m_checkpoints[i].SeedPosition = m_checkpoints[i].Position;
//		m_checkpoints[i].Time = spawnTime;
//		m_checkpoints[i].Normal = make_float3(0.0f, 0.0f, 0.0f);
//		m_checkpoints[i].DeltaT = m_traceParams.m_advectDeltaT;
//
//		m_checkpoints[i].StepsTotal = 0;
//		m_checkpoints[i].StepsAccepted = 0;
//	}

	for (uint i = 0; i < m_traceParams.m_lineCount; i++)
	{
		m_checkpoints[i].SeedPosition = m_checkpoints[i].Position;
		m_checkpoints[i].Time = spawnTime;
		m_checkpoints[i].Normal = make_float3(0.0f, 0.0f, 0.0f);
		m_checkpoints[i].DeltaT = m_traceParams.m_advectDeltaT;

		m_checkpoints[i].StepsTotal = 0;
		m_checkpoints[i].StepsAccepted = 0;
	}

	if (!m_traceParams.m_ftleEnabled)
	{
		// upload checkpoints
		cudaMemcpyKind copyDir = m_traceParams.m_cpuTracing ? cudaMemcpyHostToHost : cudaMemcpyHostToDevice;
		cudaSafeCall(cudaMemcpy(m_dpLineCheckpoints, m_checkpoints.data(), m_checkpoints.size() * sizeof(LineCheckpoint), copyDir));
	}
}


void TracingManager::CreateInitialCheckpointsRegularGrid(float3 seedBoxMin, float3 seedBoxSize)
{
	int n = std::floor(std::sqrt(m_traceParams.m_lineCount));

	for (uint i = 0; i < m_traceParams.m_lineCount; i++)
	{
		uint ii = i % n;
		uint jj = std::floor(i / (float)n);

		m_checkpoints[i].Position = make_float3(seedBoxMin.x + seedBoxSize.x * ii / (float)n, 
												seedBoxMin.y + seedBoxSize.y * jj / (float)n, 
												seedBoxMin.z);
	}
}

void TracingManager::CreateInitialCheckpointsRandom(float3 seedBoxMin, float3 seedBoxSize)
{
	for (uint i = 0; i < m_traceParams.m_lineCount; i++)
	{

		if (m_traceParams.m_seedTexture.m_colors == NULL || m_traceParams.m_seedTexture.m_picked.empty())
		{
			//seed directly from the seed box
			m_checkpoints[i].Position = GetRandomVectorInBox(seedBoxMin, seedBoxSize, m_rng, m_engine);
		}
		else 
		{
			//sample texture until a point with the same color is found
			m_checkpoints[i].Position = make_float3(Vec3f(-10000, -10000, -10000)); //somewhere outside, so that the point is deleted if no sample could be found
			for (int j = 0; j < m_numRejections; ++j) {
				float3 pos = GetRandomVectorInBox(seedBoxMin, seedBoxSize, m_rng, m_engine);
				Vec3f posAsVec = Vec3f(&pos.x);
				Vec3f posInVolume = (posAsVec + m_pVolume->GetVolumeHalfSizeWorld()) / (2 * m_pVolume->GetVolumeHalfSizeWorld());
				int texX = (int)(posInVolume.x() * m_traceParams.m_seedTexture.m_width);
				int texY = (int)(posInVolume.y() * m_traceParams.m_seedTexture.m_height);
				texY = m_traceParams.m_seedTexture.m_height - texY - 1;
				if (texX < 0 || texY < 0 || texX >= m_traceParams.m_seedTexture.m_width || texY >= m_traceParams.m_seedTexture.m_height) {
					continue;
				}
				unsigned int color = m_traceParams.m_seedTexture.m_colors[texX + texY * m_traceParams.m_seedTexture.m_height];
				if (m_traceParams.m_seedTexture.m_picked.find(color) != m_traceParams.m_seedTexture.m_picked.end()) {
					//we found it
					m_checkpoints[i].Position = pos;
					break;
				}
			}
		}
	}
}

void TracingManager::CreateInitialCheckpointsFTLE(float3 seedBoxMin, float3 seedBoxSize)
{
	assert(m_traceParams.m_lineCount == m_traceParams.m_ftleResolution * m_traceParams.m_ftleResolution * 6);

	int lineIdx = 0;

	for (size_t i = 0; i < m_traceParams.m_ftleResolution; i++)
	{
		for (size_t j = 0; j < m_traceParams.m_ftleResolution; j++)
		{
			float3 pos = make_float3(	seedBoxMin.x + seedBoxSize.x * (float)i / (float)m_traceParams.m_ftleResolution, 
										seedBoxMin.y + seedBoxSize.y * (float)j / (float)m_traceParams.m_ftleResolution, 
										m_traceParams.m_ftleSliceY);

			//for (size_t k = 0; k < 6; k++)
			//	m_checkpoints[lineIdx++].Position = pos;

			m_checkpoints[lineIdx++].Position = pos - make_float3(m_traceParams.m_ftleSeparationDistance.x(), 0, 0);
			m_checkpoints[lineIdx++].Position = pos + make_float3(m_traceParams.m_ftleSeparationDistance.x(), 0, 0);

			m_checkpoints[lineIdx++].Position = pos - make_float3(0, m_traceParams.m_ftleSeparationDistance.y(), 0);
			m_checkpoints[lineIdx++].Position = pos + make_float3(0, m_traceParams.m_ftleSeparationDistance.y(), 0);

			m_checkpoints[lineIdx++].Position = pos - make_float3(0, 0, m_traceParams.m_ftleSeparationDistance.z());
			m_checkpoints[lineIdx++].Position = pos + make_float3(0, 0, m_traceParams.m_ftleSeparationDistance.z());
		}
	}
}


void TracingManager::SetCheckpointTimeStep(float deltaT)
{
	if (!LineModeIsIterative(m_traceParams.m_lineMode)) {
		//only update checkpoints if we trace particles
		//otherwise, the current status of the lines would be overwritten
		return;
	}
	for (size_t i = 0; i < m_checkpoints.size(); ++i) {
		m_checkpoints[i].DeltaT = deltaT;
	}
	// upload checkpoints
	cudaMemcpyKind copyDir = m_traceParams.m_cpuTracing ? cudaMemcpyHostToHost : cudaMemcpyHostToDevice;
	cudaSafeCall(cudaMemcpy(m_dpLineCheckpoints, m_checkpoints.data(), m_checkpoints.size() * sizeof(LineCheckpoint), copyDir));
}

//static void writeOutFinishedLine() {
//	// write out finished line
//	cudaSafeCall(cudaGraphicsMapResources(1, &m_pResult->m_pVBCuda));
//	LineVertex* dpVB = nullptr;
//	cudaSafeCall(cudaGraphicsResourceGetMappedPointer((void**)&dpVB, nullptr, m_pResult->m_pVBCuda));
//	uint lineIndex = 5;
//	std::vector<LineVertex> vertices(m_traceParams.m_lineLengthMax);
//	cudaSafeCall(cudaMemcpy(vertices.data(), dpVB + lineIndex * m_traceParams.m_lineLengthMax, m_traceParams.m_lineLengthMax * sizeof(LineVertex), cudaMemcpyDeviceToHost));
//	cudaSafeCall(cudaGraphicsUnmapResources(1, &m_pResult->m_pVBCuda));
//	FILE* file = fopen("E:\\_line.txt", "w");
//	for(size_t i = 0; i < vertices.size(); i++) {
//		float timePrev = vertices[max(i,1)-1].Time;
//		fprintf(file, "%1.10f %1.10f %1.10f %1.10f %1.10f\n", vertices[i].Position.x, vertices[i].Position.y, vertices[i].Position.z, vertices[i].Time, vertices[i].Time - timePrev);
//	}
//	fclose(file);
//	// write out brick texture
//	Vec3ui size = m_brickAtlas.GetSize() * m_brickAtlas.GetSlotCount();
//	std::vector<float> tex(size.volume() * 4);
//	cudaMemcpy3DParms memcpyParams = { 0 };
//	memcpyParams.srcArray = const_cast<cudaArray*>(m_brickAtlas.GetCudaArray());
//	memcpyParams.dstPtr   = make_cudaPitchedPtr(tex.data(), size.x() * 4 * sizeof(float), size.x(), size.y());
//	memcpyParams.extent   = make_cudaExtent(size.x(), size.y(), size.z());
//	memcpyParams.kind     = cudaMemcpyDeviceToHost;
//	cudaSafeCall(cudaMemcpy3D(&memcpyParams));
//	std::vector<float> tex3(size.volume() * 3);
//	for(uint z = 0; z < size.z(); z++) {
//		for(uint y = 0; y < size.y(); y++) {
//			for(uint x = 0; x < size.x(); x++) {
//				uint i = x + size.x() * (y + size.y() * z);
//				tex3[3*i+0] = tex[4*i+0];
//				tex3[3*i+1] = tex[4*i+1];
//				tex3[3*i+2] = tex[4*i+2];
//			}
//		}
//	}
//	file = fopen("E:\\_tex.raw", "wb");
//	fwrite(tex3.data(), sizeof(float), tex3.size(), file);
//	fclose(file);
//}

bool TracingManager::Trace()
{
	assert(IsTracing());
	if(!IsTracing()) return false;

	if(m_timerDiskWait.IsRunning())
	{
		m_timerDiskWait.Stop();
		m_timings.WaitDiskWall += m_timerDiskWait.GetElapsedTimeMS();
	}

	if (LineModeIsIterative(m_traceParams.m_lineMode)) {
		//particles
		return TraceParticlesIteratively();
	}

	uint uploadBudget = m_uploadsPerFrame;
	for(uint round = 0; round < m_roundsPerFrame; round++)
	{
		UpdateBricksToDo();

		// if there are no more lines to do, we're done
		if(m_bricksToDo.empty())
		{
			BuildIndexBuffer();

			ComputeTimings();
			PrintTimings();

			ComputeStats();
			PrintStats();

			//writeOutFinishedLine();

			m_bricksToLoad.clear();

			m_pVolume = nullptr;
			m_pFlowGraph = nullptr;

			return true;
		}

		//if(m_verbose)
		//{
		//	printf("TracingManager::Trace: bricks");
		//	for(auto it = bricksToDo.begin(); it != bricksToDo.end(); ++it)
		//	{
		//		printf(" %u-%u(%u)", it->m_linearBrickIndex, it->m_timestepFrom, it->m_lineCount);
		//	}
		//	printf("\n");
		//}

		//TODO kick out everything older than oldestTimestep?

		// upload new bricks into unused slots
		if(!UploadBricks(uploadBudget, false))
		{
			// couldn't upload all bricks we want, so bail out
			break;
		}

		// check if there's still anything to do with the bricks we have on the GPU
		bool anythingToDo = false;
		// in time-dependent case, provide at least 3 consecutive timesteps [n,n+2] to properly handle edge cases..
		uint timestepInterpolation = LineModeIsTimeDependent(m_traceParams.m_lineMode) ? 2 : 0;
		if(!m_traceParams.m_cpuTracing)
			cudaSafeCall(cudaEventSynchronize(m_brickRequestsDownloadEvent));
		for(size_t i = 0; i < m_brickSlotInfo.size(); i++)
		{
			uint brickIndex = m_brickSlotInfo[i].brickIndex;
			uint timestepAvailableMin = m_brickSlotInfo[i].timestepFirst;
			uint timestepAvailableMax = timestepAvailableMin + m_brickSlotInfo[i].timestepCount - 1;
			if(brickIndex != ~0u)
			{
				uint requestCount = m_pBrickRequestCounts[brickIndex];
				uint timestepRequiredMin = m_pBrickTimestepMins[brickIndex];
				uint timestepRequiredMax = min(m_pBrickTimestepMins[brickIndex] + timestepInterpolation, m_timestepMax);
				if(requestCount > 0 && timestepAvailableMin <= timestepRequiredMin && timestepRequiredMax <= timestepAvailableMax)
				{
					anythingToDo = true;
				}
			}
		}

		// if all bricks on the GPU are done, kick out all the finished bricks
		if(!anythingToDo)
		{
			if(m_verbose)
			{
				printf("TracingManager::Trace: Clearing all brick slots\n");
			}
			bool reallyKick = false; // if false, only clear slot timestamps to avoid waiting for purge timeout
			if(reallyKick)
			{
				m_bricksOnGPU.clear();
				cudaSafeCall(cudaEventSynchronize(m_brickIndexUploadEvent));
			}
			for(size_t slotIndex = 0; slotIndex < m_brickSlotInfo.size(); slotIndex++)
			{
				SlotInfo& slot = m_brickSlotInfo[slotIndex];
				if(reallyKick)
				{
					// update brickindex
					m_pBrickToSlot[slot.brickIndex].x = BrickIndexGPU::INVALID;
					m_pBrickToSlot[slot.brickIndex].y = BrickIndexGPU::INVALID;
					m_pSlotTimestepMin[slotIndex] = BrickIndexGPU::INVALID;
					m_pSlotTimestepMax[slotIndex] = BrickIndexGPU::INVALID;

					slot.Clear();
				}
				else
				{
					slot.InvalidateTimestamp();
				}
			}

			m_bricksToDoPrioritiesDirty = true;
			break;
		}

		// still here -> all the bricks we want were uploaded, so go tracing
		TraceRound();
	}

	//TODO this syncs on the last trace round - move to top?
	UpdateBricksToLoad();

	return false;
}

bool TracingManager::TraceParticlesIteratively()
{
	//1. Upload all bricks to the GPU
	if (m_particlesNeedsUploadTimestep) {
		int timestep = m_pVolume->GetCurNearestTimestepIndex();
		printf("Upload all bricks at timestep %d\n", timestep);
		if (!UploadWholeTimestep(timestep, false)) {
			return false; //still uploading
		}
		m_particlesNeedsUploadTimestep = false;
		printf("uploading done\n");
	}

	// map d3d result buffer
	cudaSafeCall(cudaGraphicsMapResources(1, &m_pResult->m_pVBCuda));
	LineVertex* dpVB = nullptr;
	cudaSafeCall(cudaGraphicsResourceGetMappedPointer((void**)&dpVB, nullptr, m_pResult->m_pVBCuda));

	//build structures
	LineVertex* pVertices = (m_traceParams.m_cpuTracing ? m_lineVerticesCPU.data() : dpVB);
	LineInfo lineInfo(GetLineSpawnTime(), m_traceParams.m_lineCount, m_dpLineCheckpoints, pVertices, m_dpLineVertexCounts, m_traceParams.m_lineLengthMax);

	//check if particles should be seeded
	int seed = -1;
	std::chrono::steady_clock::time_point tp = std::chrono::steady_clock::now();
	std::chrono::duration<double> time_passed 
		= std::chrono::duration_cast<std::chrono::duration<double >> (tp - m_particlesLastTime);
	double timeBetweenSeeds = 1.0 / m_traceParams.m_particlesPerSecond;
	if (timeBetweenSeeds < time_passed.count() || m_seedManyParticles) {
		if (LineModeGenerateAlwaysNewSeeds(m_traceParams.m_lineMode)) {
			//generate new seeds
			float spawnTime = GetLineSpawnTime();
			CreateInitialCheckpoints(spawnTime);
		}
		//seed it
		seed = m_particlesSeedPosition;
		printf("TracingManager::TraceParticlesIteratively: seed particles at index %d\n", seed);
		m_particlesSeedPosition = (m_particlesSeedPosition + 1) % m_traceParams.m_lineLengthMax;
		m_particlesLastTime = tp;
	}

	//compute time per frame and update delta time
	double tpf = std::chrono::duration_cast<std::chrono::duration<double>> (tp - m_particlesLastFrame).count();
	m_particlesLastFrame = tp;
	if (m_verbose) {
		printf("TPF: %f\n", tpf);
	}
	//clamp tpf to prevent extreme large steps when the simulation is slow
	tpf = std::min(tpf, 0.1); //10fps

	// integrate
	m_timerIntegrateCPU.Start();
	m_timerIntegrate.StartNextTimer();
	//cudaSafeCall(cudaDeviceSynchronize());
	m_integrator.IntegrateParticles(m_brickAtlas, lineInfo, m_traceParams, seed, tpf * 100);

	// seed many particles
	if (m_seedManyParticles && LineModeGenerateAlwaysNewSeeds(m_traceParams.m_lineMode)) {
		//only for line mode 'PARTICLES (new seeds)' it makes sense to seed more particles
		static const int particlesToSeed = 20;
		float spawnTime = GetLineSpawnTime();
		for (int i = 0; i < particlesToSeed; ++i) {
			CreateInitialCheckpoints(spawnTime);
			seed = m_particlesSeedPosition;
			m_particlesSeedPosition = (m_particlesSeedPosition + 1) % m_traceParams.m_lineLengthMax;
			m_integrator.IntegrateParticlesExtraSeed(lineInfo, m_traceParams, seed);
		}
	}
	//cudaSafeCall(cudaDeviceSynchronize());
	m_timerIntegrate.StopCurrentTimer();
	m_timerIntegrateCPU.Stop();
	m_timings.IntegrateCPU += m_timerIntegrateCPU.GetElapsedTimeMS();

	if (m_traceParams.m_cpuTracing)
	{
		// copy vertices into d3d buffer
		cudaSafeCall(cudaMemcpy(dpVB, m_lineVerticesCPU.data(), m_lineVerticesCPU.size() * sizeof(LineVertex), cudaMemcpyHostToDevice));
	}

	// unmap d3d buffer again
	cudaSafeCall(cudaGraphicsUnmapResources(1, &m_pResult->m_pVBCuda));

	m_seedManyParticles = false;

	//never return true, it never ends
	return false;
}

void TracingManager::StartTracingParticlesIteratively()
{
	//start load all blocks
	m_particlesNeedsUploadTimestep = true;

	// map d3d result buffer
	cudaSafeCall(cudaGraphicsMapResources(1, &m_pResult->m_pVBCuda));
	LineVertex* dpVB = nullptr;
	cudaSafeCall(cudaGraphicsResourceGetMappedPointer((void**)&dpVB, nullptr, m_pResult->m_pVBCuda));

	//build structures
	LineVertex* pVertices = (m_traceParams.m_cpuTracing ? m_lineVerticesCPU.data() : dpVB);
	LineInfo lineInfo(GetLineSpawnTime(), m_traceParams.m_lineCount, m_dpLineCheckpoints, pVertices, m_dpLineVertexCounts, m_traceParams.m_lineLengthMax);

	//init particles
	m_integrator.InitIntegrateParticles(lineInfo, m_traceParams);

	// unmap d3d buffer again
	cudaSafeCall(cudaGraphicsUnmapResources(1, &m_pResult->m_pVBCuda));

	//init timings
	m_particlesSeedPosition = 0;
	m_particlesLastTime = std::chrono::steady_clock::now();
	m_particlesLastFrame = std::chrono::steady_clock::now();

	//write index buffer
	BuildParticlesIndexBuffer();
}

void TracingManager::CancelTracing()
{
	if(IsTracing())
	{
		printf("TracingManager::CancelTracing\n");
		ReleaseResultResources(); //TODO hrm..?

		m_bricksToDo.clear();
		m_bricksToLoad.clear();
		m_pVolume = nullptr;
		m_pFlowGraph = nullptr;
	}
}

bool TracingManager::IsTracing() const
{
	return m_pVolume != nullptr;
}

float TracingManager::GetTracingProgress() const
{
	if(!IsTracing()) return 0.0f;

	return m_progress;
}


void TracingManager::ClearResult()
{
	ReleaseResultResources();
}


bool TracingManager::Timings::WriteToFile(const std::string& filename) const
{
	std::ofstream file(filename);
	return WriteToFile(file);
}

bool TracingManager::Timings::WriteToFile(std::ostream& file) const
{
	if(!file.good()) return false;

	file << "WallTime: " << TraceWall << "\n";
	file << "DiskWaitTime: " << WaitDiskWall << "\n";
	file << "UploadDecompressTimeAvg: " << UploadDecompressGPU.Avg << "\n";
	file << "UploadDecompressTimeSum: " << UploadDecompressGPU.Total << "\n";
	file << "IntegrateTimeAvg: " << IntegrateGPU.Avg << "\n";
	file << "IntegrateTimeSum: " << IntegrateGPU.Total << "\n";
	file << "BuildIndexBufferTimeAvg: " << BuildIndexBufferGPU.Avg << "\n";
	file << "BuildIndexBufferTimeSum: " << BuildIndexBufferGPU.Total << "\n";

	return true;
}

bool TracingManager::Timings::WriteCSVHeader(std::ostream& file, const std::vector<std::string>& extraColumns)
{
	if(!file.good()) return false;

	for(auto& str : extraColumns)
	{
		file << str << ";";
	}
	file << "WallTime;";
	file << "DiskWaitTime;";
	file << "UploadDecompressTimeAvg;";
	file << "UploadDecompressTimeSum;";
	file << "IntegrateTimeAvg;";
	file << "IntegrateTimeSum;";
	file << "BuildIndexBufferTimeAvg;";
	file << "BuildIndexBufferTimeSum" << std::endl;

	return true;
}

bool TracingManager::Timings::WriteToCSVFile(std::ostream& file, const std::vector<std::string>& extraColumns) const
{
	if(!file.good()) return false;

	for(auto& str : extraColumns)
	{
		file << str << ";";
	}
	file << TraceWall << ";";
	file << WaitDiskWall << ";";
	file << UploadDecompressGPU.Avg << ";";
	file << UploadDecompressGPU.Total << ";";
	file << IntegrateGPU.Avg << ";";
	file << IntegrateGPU.Total << ";";
	file << BuildIndexBufferGPU.Avg << ";";
	file << BuildIndexBufferGPU.Total << std::endl;

	return true;
}


void TracingManager::Stats::Clear()
{
	BricksUploadedTotal = 0;
	BricksUploadedUnique = 0;
	StepsTotal = 0;
	StepsAccepted = 0;
	Evaluations = 0;
	LinesReachedMaxAge = 0;
	LinesLeftDomain = 0;
	DiskBusyTimeMS = 0.0f;
	DiskBytesLoaded = 0;
	DiskBytesUsed = 0;
}


bool TracingManager::Stats::WriteToFile(const std::string& filename) const
{
	std::ofstream file(filename);
	return WriteToFile(file);
}

bool TracingManager::Stats::WriteToFile(std::ostream& file) const
{
	if(!file.good()) return false;

	file << "TotalBricksUploaded: " << BricksUploadedTotal << "\n";
	file << "UniqueBricksUploaded: " << BricksUploadedUnique << "\n";

	file << "StepsTotal: " << StepsTotal << "\n";
	file << "StepsAccepted: " << StepsAccepted << "\n";
	file << "Evaluations: " << Evaluations << "\n";

	file << "LinesReachedMaxAge: " << LinesReachedMaxAge << "\n";
	file << "LinesLeftDomain: " << LinesLeftDomain << "\n";

	file << "DiskBusyTimeMS: " << DiskBusyTimeMS << "\n";
	file << "DiskBytesLoaded: " << DiskBytesLoaded << "\n";
	file << "DiskBytesUsed: " << DiskBytesUsed << "\n";

	return true;
}

bool TracingManager::Stats::WriteCSVHeader(std::ostream& file, const std::vector<std::string>& extraColumns)
{
	if(!file.good()) return false;

	for(auto& str : extraColumns)
	{
		file << str << ";";
	}
	file << "TotalBricksUploaded;";
	file << "UniqueBricksUploaded;";
	file << "StepsTotal;";
	file << "StepsAccepted;";
	file << "Evaluations;";
	file << "LinesReachedMaxAge;";
	file << "LinesLeftDomain;";
	file << "DiskBusyTimeMS;";
	file << "DiskBytesLoaded;";
	file << "DiskBytesUsed" << std::endl;

	return true;
}

bool TracingManager::Stats::WriteToCSVFile(std::ostream& file, const std::vector<std::string>& extraColumns) const
{
	if(!file.good()) return false;

	for(auto& str : extraColumns)
	{
		file << str << ";";
	}
	file << BricksUploadedTotal << ";";
	file << BricksUploadedUnique << ";";
	file << StepsTotal << ";";
	file << StepsAccepted << ";";
	file << Evaluations << ";";
	file << LinesReachedMaxAge << ";";
	file << LinesLeftDomain << ";";
	file << DiskBusyTimeMS << ";";
	file << DiskBytesLoaded << ";";
	file << DiskBytesUsed << std::endl;

	return true;
}


static Vec2ui Fit2DSlotCount(uint slotCountTotal, uint slotCount1DMax)
{
	Vec2ui slotCountBest(min(slotCountTotal, slotCount1DMax), 1);

	uint slotCountXMin = max(1, uint(sqrt((double)slotCount1DMax)));
	for(uint slotCountX = slotCount1DMax; slotCountX >= slotCountXMin; --slotCountX)
	{
		uint slotCountY = slotCountTotal / slotCountX;
		if(slotCountX * slotCountY > slotCountBest.area())
		{
			slotCountBest.set(slotCountX, slotCountY);
		}
	}

	return slotCountBest;
}

HRESULT TracingManager::CreateVolumeDependentResources()
{
	assert(m_pVolume != nullptr);

	ReleaseVolumeDependentResources();


	int device = -1;
	cudaSafeCall(cudaGetDevice(&device));
	cudaDeviceProp devProp;
	cudaSafeCall(cudaGetDeviceProperties(&devProp, device));


	uint brickSize = m_pVolume->GetBrickSizeWithOverlap();
	uint channelCount = m_pVolume->GetChannelCount();
	uint channelCountTex = (channelCount == 3) ? 4 : channelCount;


	// allocate channel buffers for decompression
	size_t brickSizeBytePerChannel = brickSize * brickSize * brickSize * sizeof(float);
	m_dpChannelBuffer.resize(channelCount);
	m_pChannelBufferCPU.resize(channelCount);
	for(size_t channel = 0; channel < m_dpChannelBuffer.size(); channel++)
	{
		cudaSafeCall(cudaMalloc2(&m_dpChannelBuffer[channel], brickSizeBytePerChannel));
		cudaSafeCall(cudaMallocHost(&m_pChannelBufferCPU[channel], brickSizeBytePerChannel));
	}

	uint brickCount = m_pVolume->GetBrickCount().volume();


	// allocate brick requests structure
	cudaSafeCall(cudaMallocHost(&m_pBrickRequestCounts, brickCount * sizeof(uint)));
	memset(m_pBrickRequestCounts, 0, brickCount * sizeof(uint));
	cudaSafeCall(cudaMallocHost(&m_pBrickTimestepMins, brickCount * sizeof(uint)));
	uint timestepMinInit = LineModeIsTimeDependent(m_traceParams.m_lineMode) ? ~0u : GetLineFloorTimestepIndex(GetLineSpawnTime());
	for(uint i = 0; i < brickCount; i++)
	{
		m_pBrickTimestepMins[i] = timestepMinInit;
	}
	m_brickRequestsGPU.Allocate(m_traceParams.m_cpuTracing, brickCount);
	m_brickRequestsGPU.Clear(m_traceParams.m_cpuTracing, brickCount);


	// allocate brick slots
	size_t memFree = 0;
	size_t memTotal = 0;
	cudaSafeCall(cudaMemGetInfo(&memFree, &memTotal));

	std::cout << "\tAvailable: " << float(memFree) / (1024.0f * 1024.0f) << "MB" << std::endl;

	size_t memPerTimeSlot = brickSize * brickSize * brickSize * channelCountTex * sizeof(float);

	// leave some wiggle room - arbitrarily chosen to be the size of a brick, or at least min(128 MB, 0.1 * totalMemory)
	// with large bricks, CUDA tends to start paging otherwise...
	size_t memBuffer = max(memPerTimeSlot, min(size_t(32) * 1024 * 1024, memTotal / 100));
	//size_t memAvailable = memFree - min(memBuffer, memFree);
	size_t memAvailable = 1024ll * 1024ll * 1024ll;

	float memAvailableMB = float(memAvailable) / (1024.0f * 1024.0f);

	// calculate max number of time slots
	uint timeSlotCountMax = min(devProp.maxTexture3D[0] / brickSize, m_timeSlotCountMax);
	timeSlotCountMax = min(max(1, uint(memAvailable / memPerTimeSlot)), timeSlotCountMax);
	uint timeSlotCount = LineModeIsTimeDependent(m_traceParams.m_lineMode) ? timeSlotCountMax : 1;

	// calculate max number of brick slots based on the available memory
	size_t memPerBrickSlot = brickSize * brickSize * brickSize * channelCountTex * timeSlotCount * sizeof(float);
	uint brickSlotCount1DMax = devProp.maxTexture3D[1] / brickSize;
	uint brickSlotCountMax = min(max(1, uint(memAvailable / memPerBrickSlot)), m_brickSlotCountMax);
	Vec2ui brickSlotCount = Fit2DSlotCount(brickSlotCountMax, brickSlotCount1DMax);
	m_brickSlotCount = Vec3ui(timeSlotCount, brickSlotCount);

	if(m_traceParams.m_cpuTracing)
	{
		Vec3ui size = brickSize * m_brickSlotCount;
		g_volume.size = make_uint3(size);
		g_volume.data.resize(size.volume());
		printf("TracingManager::CreateVolumeDependentResources:\n\tAllocated %ux%u brick slot(s) (target %u) with %u time slot(s) each\n",
			m_brickSlotCount.y(), m_brickSlotCount.z(), brickSlotCountMax, m_brickSlotCount.x());
	}
	else
	{
		if(!m_brickAtlas.Create(brickSize, channelCount, m_brickSlotCount))
		{
			// out of memory
			//TODO retry with fewer brick/time slots
			printf("TracingManager::CreateVolumeDependentResources: Failed to create brick slots\n");
			ReleaseVolumeDependentResources();
			return E_FAIL;
		}
		printf("TracingManager::CreateVolumeDependentResources: %.2f MB available\n\tCreated %ux%u brick slot(s) (target %u) with %u time slot(s) each\n",
			memAvailableMB, m_brickAtlas.GetSlotCount().y(), m_brickAtlas.GetSlotCount().z(), brickSlotCountMax, m_brickAtlas.GetSlotCount().x());
	}


	m_brickSlotInfo.resize(m_brickSlotCount.yz().area());


	memFree = 0;
	memTotal = 0;
	cudaMemGetInfo(&memFree, &memTotal);
	std::cout << "cudaMallocHost: " << float(brickCount * sizeof(uint2)) / 1024.0f << "KB" << "\tAvailable: " << float(memFree) / (1024.0f * 1024.0f) << "MB" << std::endl;

	// allocate brick index *after* brick slots - need to know the slot count!
	cudaSafeCall(cudaMallocHost(&m_pBrickToSlot, brickCount * sizeof(uint2)));
	for(uint i = 0; i < brickCount; i++)
	{
		m_pBrickToSlot[i].x = BrickIndexGPU::INVALID;
		m_pBrickToSlot[i].y = BrickIndexGPU::INVALID;
	}

	cudaMemGetInfo(&memFree, &memTotal);
	std::cout << "cudaMallocHost: " << float(brickSlotCount.area() * sizeof(uint) * 2) / 1024.0f << "KB" << "\tAvailable: " << float(memFree) / (1024.0f * 1024.0f) << "MB" << std::endl;

	cudaSafeCall(cudaMallocHost(&m_pSlotTimestepMin, brickSlotCount.area() * sizeof(uint)));
	cudaSafeCall(cudaMallocHost(&m_pSlotTimestepMax, brickSlotCount.area() * sizeof(uint)));
	for(uint i = 0; i < brickCount; i++)
	{
		m_pSlotTimestepMin[i] = BrickIndexGPU::INVALID;
		m_pSlotTimestepMax[i] = BrickIndexGPU::INVALID;
	}
	m_brickIndexGPU.Allocate(m_traceParams.m_cpuTracing, brickCount, make_uint2(brickSlotCount));
	m_brickIndexGPU.Update(m_traceParams.m_cpuTracing, m_pBrickToSlot, m_pSlotTimestepMin, m_pSlotTimestepMax);
	cudaSafeCall(cudaEventRecord(m_brickIndexUploadEvent));

	std::cout << "TracingManager::CreateVolumeDependentResources done." << std::endl;

	return S_OK;
}

void TracingManager::ReleaseVolumeDependentResources()
{
	m_brickSlotInfo.clear();
	m_brickAtlas.Release();
	m_brickSlotCount.set(0, 0, 0);

	g_volume.data.clear();
	g_volume.data.shrink_to_fit();
	g_volume.size = make_uint3(0, 0, 0);

	m_bricksOnGPU.clear();

	m_brickRequestsGPU.Deallocate();
	cudaSafeCall(cudaFreeHost(m_pBrickTimestepMins));
	m_pBrickTimestepMins = nullptr;
	cudaSafeCall(cudaFreeHost(m_pBrickRequestCounts));
	m_pBrickRequestCounts = nullptr;
	m_brickIndexGPU.Deallocate();
	cudaSafeCall(cudaFreeHost(m_pSlotTimestepMax));
	m_pSlotTimestepMax = nullptr;
	cudaSafeCall(cudaFreeHost(m_pSlotTimestepMin));
	m_pSlotTimestepMin = nullptr;
	cudaSafeCall(cudaFreeHost(m_pBrickToSlot));
	m_pBrickToSlot = nullptr;

	for(size_t channel = 0; channel < m_dpChannelBuffer.size(); channel++)
	{
		cudaSafeCall(cudaFree(m_dpChannelBuffer[channel]));
		cudaSafeCall(cudaFreeHost(m_pChannelBufferCPU[channel]));
	}
	m_dpChannelBuffer.clear();
	m_pChannelBufferCPU.clear();
}


HRESULT TracingManager::CreateParamDependentResources()
{
	std::cout << "CreateParamDependentResources()" << std::endl;

	ReleaseParamDependentResources();

	if(m_traceParams.m_cpuTracing)
	{
		m_dpLineCheckpoints = new LineCheckpoint[m_traceParams.m_lineCount];
		m_dpLineVertexCounts = new uint[m_traceParams.m_lineCount];
		memset(m_dpLineVertexCounts, 0, m_traceParams.m_lineCount * sizeof(uint));

		m_lineVerticesCPU.resize(m_traceParams.m_lineCount * m_traceParams.m_lineLengthMax);
	}
	else if (!m_traceParams.m_ftleEnabled)
	{
		cudaSafeCall(cudaMalloc2(&m_dpLineCheckpoints, m_traceParams.m_lineCount * sizeof(LineCheckpoint)));
		cudaSafeCall(cudaMalloc2(&m_dpLineVertexCounts, m_traceParams.m_lineCount * sizeof(uint)));
		cudaSafeCall(cudaMemset(m_dpLineVertexCounts, 0, m_traceParams.m_lineCount * sizeof(uint)));
	}

	std::cout << "CreateParamDependentResources(): done" << std::endl;


	return S_OK;
}

void TracingManager::ReleaseParamDependentResources()
{
	if(m_traceParams.m_cpuTracing)
	{
		m_lineVerticesCPU.clear();
		m_lineVerticesCPU.shrink_to_fit();
		delete[] m_dpLineVertexCounts;
		delete[] m_dpLineCheckpoints;
	}
	else
	{
		cudaSafeCall(cudaFree(m_dpLineVertexCounts));
		cudaSafeCall(cudaFree(m_dpLineCheckpoints));
	}
	m_dpLineVertexCounts = nullptr;
	m_dpLineCheckpoints = nullptr;
}


HRESULT TracingManager::CreateResultResources()
{
	ReleaseResultResources();

	assert(m_pDevice);

	if (m_traceParams.m_ftleEnabled)
		cudaSafeCall(cudaMalloc2(&m_dpParticles, m_traceParams.m_lineCount * sizeof(SimpleParticleVertexDeltaT)));
	else
		m_pResult = std::make_shared<LineBuffers>(m_pDevice, m_traceParams.m_lineCount, m_traceParams.m_lineLengthMax);

	return S_OK;
}

void TracingManager::ReleaseResultResources()
{
	m_pResult = nullptr;

	if (m_dpParticles != nullptr)
	{
		cudaSafeCall(cudaFree(m_dpParticles));
		m_dpParticles = nullptr;
	}
}


void TracingManager::UpdateFrameBudgets()
{
	// HACK: for uploads, scale with linear brick size (instead of volume)
	//       to account for larger overhead with smaller bricks...
	const uint brickSizeRef = 256;
	const uint brickSizeRef2 = brickSizeRef * brickSizeRef;
	uint brickSize = m_pVolume->GetBrickSizeWithOverlap();
	uint timeSlotCount = m_brickSlotCount.x();
	m_uploadsPerFrame = max(1, brickSizeRef / (brickSize * timeSlotCount)); //TODO remove timeSlotCount here, instead count one upload per time slot

	m_roundsPerFrame = 8; //TODO scale with interpolation mode
}


static Vec3i SpatialNeighbor(const Vec3i& vec, int dim, int dist)
{
	Vec3i result(vec);
	result[dim] += dist;
	return result;
}

std::vector<uint> TracingManager::GetBrickNeighbors(uint linearIndex) const
{
	std::vector<uint> result;

	uint neighbors[6];
	GetBrickNeighbors(linearIndex, neighbors);
	for(uint i = 0; i < 6; i++)
	{
		if(neighbors[i] != uint(-1))
		{
			result.push_back(neighbors[i]);
		}
	}

	return result;
}

void TracingManager::GetBrickNeighbors(uint linearIndex, uint neighborIndices[6]) const
{
	Vec3i brickCount = m_pVolume->GetBrickCount();
	Vec3i spatialIndex = m_pVolume->GetBrickSpatialIndex(linearIndex);

	auto left  = spatialIndex.compEqual(Vec3i(0, 0, 0));
	auto right = spatialIndex.compEqual(brickCount - 1);

	neighborIndices[0] = left[0]  ? uint(-1) : m_pVolume->GetBrickLinearIndex(SpatialNeighbor(spatialIndex, 0, -1));
	neighborIndices[1] = left[1]  ? uint(-1) : m_pVolume->GetBrickLinearIndex(SpatialNeighbor(spatialIndex, 1, -1));
	neighborIndices[2] = left[2]  ? uint(-1) : m_pVolume->GetBrickLinearIndex(SpatialNeighbor(spatialIndex, 2, -1));
	neighborIndices[3] = right[0] ? uint(-1) : m_pVolume->GetBrickLinearIndex(SpatialNeighbor(spatialIndex, 0,  1));
	neighborIndices[4] = right[1] ? uint(-1) : m_pVolume->GetBrickLinearIndex(SpatialNeighbor(spatialIndex, 1,  1));
	neighborIndices[5] = right[2] ? uint(-1) : m_pVolume->GetBrickLinearIndex(SpatialNeighbor(spatialIndex, 2,  1));
}


bool TracingManager::BrickIsLoaded(uint linearIndex, int timestepFrom, int timestepTo) const
{
	if(!m_pVolume) return false;

	bool loaded = true;
	for(int timestep = timestepFrom; timestep <= timestepTo; timestep++)
	{
		if(!m_pVolume->GetBrick(timestep, linearIndex).IsLoaded())
		{
			loaded = false;
			break;
		}
	}

	return loaded;
}

bool TracingManager::BrickIsOnGPU(uint linearIndex, int timestepFrom, int timestepTo) const
{
	assert(m_pVolume != nullptr);

	auto it = m_bricksOnGPU.find(linearIndex);
	if(it == m_bricksOnGPU.end())
	{
		// brick isn't on the GPU at all
		return false;
	}

	uint slotIndex = it->second;
	const SlotInfo& slotInfo = m_brickSlotInfo[slotIndex];
	assert(slotInfo.brickIndex == linearIndex);

	// check if all required timesteps are there
	int timestepAvailableFirst = slotInfo.timestepFirst;
	int timestepAvailableLast  = slotInfo.timestepFirst + slotInfo.timestepCount - 1;
	return (timestepAvailableFirst <= timestepFrom) && (timestepAvailableLast >= timestepTo);
}


int TracingManager::FindAvailableBrickSlot(bool forcePurgeFinished) const
{
	// in time-dependent case, provide at least 3 consecutive timesteps [n,n+2] to properly handle edge cases..
	uint timestepInterpolation = LineModeIsTimeDependent(m_traceParams.m_lineMode) ? 2 : 0;

	int oldestFinishedSlotIndex = -1;
	int oldestFinishedTimestamp = std::numeric_limits<int>::max();
	for(size_t i = 0; i < m_brickSlotInfo.size(); i++)
	{
		const SlotInfo& info = m_brickSlotInfo[i];

		// if slot isn't filled at all, use this one immediately
		if(!info.IsFilled()) return int(i);

		uint timestepAvailableMin = info.timestepFirst;
		uint timestepAvailableMax = info.timestepFirst + info.timestepCount - 1;

		uint requestCount = m_pBrickRequestCounts[info.brickIndex];
		uint timestepRequiredMin = m_pBrickTimestepMins[info.brickIndex];
		uint timestepRequiredMax = timestepRequiredMin + timestepInterpolation;

		bool brickIsFinished = (requestCount == 0) || (timestepAvailableMin > timestepRequiredMin) || (timestepRequiredMax > timestepAvailableMax);
		bool purgeTimeoutPassed = (info.lastUseTimestamp + int(m_traceParams.m_purgeTimeoutInRounds) <= m_currentTimestamp);
		bool canPurge = brickIsFinished && (purgeTimeoutPassed || forcePurgeFinished);
		if(canPurge && info.lastUseTimestamp < oldestFinishedTimestamp)
		{
			oldestFinishedSlotIndex = int(i);
			oldestFinishedTimestamp = info.lastUseTimestamp;
		}
	}

	return oldestFinishedSlotIndex;
}


bool TracingManager::UploadBricks(uint& uploadBudget, bool forcePurgeFinished)
{
	uint uploadedCount;
	return UploadBricks(uploadBudget, forcePurgeFinished, uploadedCount);
}

bool TracingManager::UploadBricks(uint& uploadBudget, bool forcePurgeFinished, uint& uploadedCount)
{
	if(m_verbose)
	{
		printf("TracingManager::UploadBricks (budget %u forcePurge %i)\n", uploadBudget, int(forcePurgeFinished));
	}

	uploadedCount = 0;

	uint timeSlotCount = m_brickSlotCount.x();

	while(uploadBudget > 0)
	{
		// first update brick priorities
		UpdateBricksToDo();

		// find a brick that needs to be uploaded
		bool finished = true;
		for(auto itNextBrick = m_bricksToDo.begin(); itNextBrick != m_bricksToDo.end(); ++itNextBrick)
		{
			uint linearBrickIndex = itNextBrick->m_linearBrickIndex;
			uint timestepFrom = itNextBrick->m_timestepFrom;
			uint timestepTo = min(itNextBrick->m_timestepFrom + timeSlotCount - 1, m_timestepMax);

			if(BrickIsOnGPU(linearBrickIndex, timestepFrom, timestepTo))
			{
				assert(m_bricksOnGPU.count(linearBrickIndex) > 0);
				// update slot's lastUseTimestamp
				uint slotIndex = m_bricksOnGPU[linearBrickIndex];
				m_brickSlotInfo[slotIndex].lastUseTimestamp = m_currentTimestamp;
				continue;
			}

			bool brickIsLoaded = BrickIsLoaded(linearBrickIndex, timestepFrom, timestepTo);

			if(!brickIsLoaded && m_traceParams.m_waitForDisk)
			{
				// bail out - have to wait for disk IO
				// track disk io waiting time
				if(!m_timerDiskWait.IsRunning())
				{
					m_timerDiskWait.Start();
					// this somewhat un-fucks CUDA's timings...
					cudaSafeCall(cudaDeviceSynchronize());
				}
				return false;
			}

			if(brickIsLoaded)
			{
				// okay, we found a brick to upload which is available in CPU memory
				int slotIndex = -1;
				// check if the brick is already on the GPU (at different time steps); if so, reuse the slot
				auto itSlot = m_bricksOnGPU.find(linearBrickIndex);
				if(itSlot != m_bricksOnGPU.end())
				{
					slotIndex = itSlot->second;
				}
				else
				{
					slotIndex = FindAvailableBrickSlot(forcePurgeFinished);
				}
				// if there is no available slot, we're good for now
				if(slotIndex == -1)
				{
					return true;
				}

				if(m_verbose)
				{
					printf("TracingManager::UploadBricks: uploading brick %u prio %.2f\n", linearBrickIndex, itNextBrick->m_priority);
				}
				if(!UpdateBrickSlot(uploadBudget, slotIndex, linearBrickIndex, timestepFrom, timestepTo))
				{
					// budget ran out
					return false;
				}
				++uploadedCount;

				// stop iterating over bricksToDo (will be updated now, and we'll start over!)
				finished = false;
				break;
			}
		}

		if(finished)
		{
			// no more bricks to process - we're good
			return true;
		}
	}

	// upload budget ran out
	return false;
}

bool TracingManager::UpdateBrickSlot(uint& uploadBudget, uint brickSlotIndex, uint brickLinearIndex, int timestepFrom, int timestepTo)
{
	Vec3i brickSpatialIndex = m_pVolume->GetBrickSpatialIndex(brickLinearIndex);

	uint timeSlotCount = m_brickSlotCount.x();

	assert(BrickIsLoaded(brickLinearIndex, timestepFrom, timestepTo));


	SlotInfo& slot = m_brickSlotInfo[brickSlotIndex];

	// check if the brick is already in the correct slot (though maybe at a different timestep)
	bool brickOnGPU = (slot.brickIndex == brickLinearIndex);
	// if the brick is already on the GPU, it should be in the correct slot
	assert((m_bricksOnGPU.count(brickLinearIndex) > 0) == brickOnGPU);


	// de-linearize slot index - they're stacked in y and z in the texture
	Vec2ui brickSlotIndexGPU(brickSlotIndex % m_brickSlotCount.y(), brickSlotIndex / m_brickSlotCount.y());

	// find out which timesteps we need to upload
	//TODO handle case when timestepFirst != timestepFrom, but we don't need more timesteps
	int timeSlotFirstToUpload = 0;
	int timeSlotLastToUpload = timestepTo - timestepFrom;
	if(brickOnGPU)
	{
		// if some (but not all) required time steps are there already, shift them to the correct time slots
		int timeSlotShift = slot.timestepFirst - timestepFrom;
		if(uint(abs(timeSlotShift)) < timeSlotCount)
		{
			if(timeSlotShift < 0)
			{
				assert(!m_traceParams.m_cpuTracing);
				// move existing contents to the left
				timeSlotFirstToUpload = min(timeSlotCount + timeSlotShift, timeSlotLastToUpload + 1);
				for(int timeSlot = 0; timeSlot < timeSlotFirstToUpload; timeSlot++)
				{
					m_brickAtlas.CopySlot(Vec3ui(timeSlot - timeSlotShift, brickSlotIndexGPU), Vec3ui(timeSlot, brickSlotIndexGPU));
				}
			}
			else if(timeSlotShift > 0)
			{
				assert(!m_traceParams.m_cpuTracing);
				// move existing contents to the right
				int timeSlotMax = timeSlotLastToUpload;
				timeSlotLastToUpload = min(timeSlotMax, max(timeSlotShift - 1, timeSlotFirstToUpload - 1));
				for(int timeSlot = timeSlotLastToUpload + 1; timeSlot < timeSlotMax; timeSlot++)
				{
					m_brickAtlas.CopySlot(Vec3ui(timeSlot - timeSlotShift, brickSlotIndexGPU), Vec3ui(timeSlot, brickSlotIndexGPU));
				}
			}
			else
			{
				// nothing to do
				timeSlotFirstToUpload = 0;
				timeSlotLastToUpload = -1;
			}
		}
		//else: none of the required time steps are there already, have to load all
	}
	else
	{
		if(slot.brickIndex != ~0u)
		{
			// kick out old brick
			m_bricksOnGPU.erase(slot.brickIndex);
			// update brickindex
			cudaSafeCall(cudaEventSynchronize(m_brickIndexUploadEvent));
			m_pBrickToSlot[slot.brickIndex].x = BrickIndexGPU::INVALID;
			m_pBrickToSlot[slot.brickIndex].y = BrickIndexGPU::INVALID;
			m_pSlotTimestepMin[brickSlotIndex] = BrickIndexGPU::INVALID;
			m_pSlotTimestepMax[brickSlotIndex] = BrickIndexGPU::INVALID;

			slot.brickIndex = ~0u;
		}
	}

	//TODO limit number of time steps according to budget (-> properly record later!)
	bool result = true;
	assert(uploadBudget > 0);
	if(timeSlotLastToUpload - timeSlotFirstToUpload + 1 > (int)uploadBudget)
	{
		uploadBudget = 0;
	}
	else
	{
		uploadBudget -= timeSlotLastToUpload - timeSlotFirstToUpload + 1;
	}
	//if(timeSlotLastToUpload - timeSlotFirstToUpload + 1 > (int)uploadBudget)
	//{
	//	timeSlotLastToUpload = timeSlotFirstToUpload + uploadBudget - 1;
	//	result = false;
	//}
	//// ..and update the budget
	//uploadBudget -= timeSlotLastToUpload - timeSlotFirstToUpload + 1;


	if(m_verbose)
	{
		printf("TracingManager::UpdateBrickSlot: slot %ux%u brick %u (%i,%i,%i) ts %i-%i (upload %i)\n",
			brickSlotIndexGPU.x(), brickSlotIndexGPU.y(),
			brickLinearIndex, brickSpatialIndex.x(), brickSpatialIndex.y(), brickSpatialIndex.z(),
			timestepFrom, timestepTo,
			timeSlotLastToUpload + 1 - timeSlotFirstToUpload);
	}

	// upload new brick data
	uint brickCount = m_pVolume->GetBrickCount().volume();
	for(int timeSlot = timeSlotFirstToUpload; timeSlot <= timeSlotLastToUpload; timeSlot++)
	{
		int timestep = timestepFrom + timeSlot;
		const TimeVolumeIO::Brick& brick = m_pVolume->GetBrick(timestep, brickLinearIndex);
		assert(brick.IsLoaded());

		//cudaSafeCall(cudaDeviceSynchronize());
		Vec3ui slotIndex(timeSlot, brickSlotIndexGPU);
		if(m_traceParams.m_cpuTracing)
		{
			UploadBrick(m_pCompressShared, m_pCompressVolume, m_pVolume->GetInfo(), brick, m_dpChannelBuffer.data(), nullptr, slotIndex, &m_timerUploadDecompress);

			// download to CPU and copy into interleaved g_volume
			Vec3ui brickSize = brick.GetSize();
			size_t brickSizeBytePerChannel = brickSize.volume() * sizeof(float);
			int channelCount = m_pVolume->GetInfo().GetChannelCount();
			for(int c = 0; c < channelCount; c++)
			{
				cudaSafeCall(cudaMemcpy(m_pChannelBufferCPU[c], m_dpChannelBuffer[c], brickSizeBytePerChannel, cudaMemcpyDeviceToHost));
			}

			uint brickSizeFull = m_pVolume->GetInfo().GetBrickSizeWithOverlap();
			Vec3ui atlasSize = m_brickSlotCount * brickSizeFull;
			Vec3ui offset = slotIndex * brickSizeFull;
			for(uint z = 0; z < brickSize.z(); z++)
			{
				for(uint y = 0; y < brickSize.y(); y++)
				{
					for(uint x = 0; x < brickSize.x(); x++)
					{
						uint indexSrc = x + brickSize.x() * (y + brickSize.y() * z);
						uint indexDst = (offset.x()+x) + atlasSize.x() * ((offset.y()+y) + atlasSize.y() * (offset.z()+z));
						// HACK: hardcoded for 4 channels...
						assert(channelCount == 4);
						g_volume.data[indexDst] = make_float4(m_pChannelBufferCPU[0][indexSrc], m_pChannelBufferCPU[1][indexSrc], m_pChannelBufferCPU[2][indexSrc], m_pChannelBufferCPU[3][indexSrc]);
					}
				}
			}
		}
		else
		{
			UploadBrick(m_pCompressShared, m_pCompressVolume, m_pVolume->GetInfo(), brick, m_dpChannelBuffer.data(), &m_brickAtlas, slotIndex, &m_timerUploadDecompress);
		}
		//cudaSafeCall(cudaDeviceSynchronize());

		m_stats.BricksUploadedTotal++;
		m_bricksLoaded.insert(timestep * brickCount + brickLinearIndex);
	}

	// record new brick as being on the GPU
	slot.brickIndex = brickLinearIndex;
	slot.timestepFirst = timestepFrom;
	slot.timestepCount = timestepTo - timestepFrom + 1;
	slot.lastUseTimestamp = m_currentTimestamp;
	m_bricksOnGPU[brickLinearIndex] = brickSlotIndex;
	// update brick index
	cudaSafeCall(cudaEventSynchronize(m_brickIndexUploadEvent));
	// de-linearize slot index - they're stacked in y and z
	m_pBrickToSlot[brickLinearIndex] = make_uint2(brickSlotIndexGPU);
	m_pSlotTimestepMin[brickSlotIndex] = timestepFrom;
	m_pSlotTimestepMax[brickSlotIndex] = timestepTo;

	if (m_verbose)
	{
		printf("TracingManager::UpdateBrickSlot: gpu slot %d assigned to a brick\n",
			brickLinearIndex);
	}

	m_bricksToDoPrioritiesDirty = true;

	return result;
}

bool TracingManager::UploadWholeTimestep(int timestep, bool forcePurgeFinished)
{
	m_bricksToLoad.clear();

	//This is more or less a modified version of UploadBricks

	uint uploadBudget = 8;

	if (m_verbose)
	{
		printf("TracingManager::UploadWholeTimestep (budget %u)\n", uploadBudget);
	}

	uint timeSlotCount = m_brickSlotCount.x();

	uint brickCount = m_pVolume->GetBrickCount().volume();

	// find a brick that needs to be uploaded
	bool finished = true;
	for (uint linearBrickIndex = 0; linearBrickIndex < brickCount; ++linearBrickIndex)
	{
		uint timestepFrom = timestep;
		uint timestepTo = timestep;

		if (BrickIsOnGPU(linearBrickIndex, timestepFrom, timestepTo))
		{
			assert(m_bricksOnGPU.count(linearBrickIndex) > 0);
			// update slot's lastUseTimestamp
			uint slotIndex = m_bricksOnGPU[linearBrickIndex];
			m_brickSlotInfo[slotIndex].lastUseTimestamp = m_currentTimestamp;
			continue;
		}

		bool brickIsLoaded = BrickIsLoaded(linearBrickIndex, timestepFrom, timestepTo);

		if (!brickIsLoaded) {
			//add to loading queue
			m_bricksToLoad.push_back(&m_pVolume->GetBrick(timestep, linearBrickIndex));
			if (m_verbose)
			{
				printf("TracingManager::UploadWholeTimestep: brick %u is not loaded\n", linearBrickIndex);
			}
		}

		if (!brickIsLoaded && m_traceParams.m_waitForDisk)
		{
			// bail out - have to wait for disk IO
			// track disk io waiting time
			if (!m_timerDiskWait.IsRunning())
			{
				m_timerDiskWait.Start();
				// this somewhat un-fucks CUDA's timings...
				cudaSafeCall(cudaDeviceSynchronize());
			}
			return false;
		}

		if (brickIsLoaded)
		{
			// okay, we found a brick to upload which is available in CPU memory
			int slotIndex = -1;
			// check if the brick is already on the GPU (at different time steps); if so, reuse the slot
			auto itSlot = m_bricksOnGPU.find(linearBrickIndex);
			if (itSlot != m_bricksOnGPU.end())
			{
				slotIndex = itSlot->second;
			}
			else
			{
				slotIndex = FindAvailableBrickSlot(forcePurgeFinished);
			}
			// if there is no available slot, we're good for now
			if (slotIndex == -1)
			{
				printf("TracingManager::UploadWholeTimestep: no available slot found!!\n");
				break;
			}

			if (m_verbose)
			{
				printf("TracingManager::UploadWholeTimestep: uploading brick %u\n", linearBrickIndex);
			}
			if (!UpdateBrickSlot(uploadBudget, slotIndex, linearBrickIndex, timestepFrom, timestepTo))
			{
				// budget ran out
				return false;
			}
			if (uploadBudget == 0) {
				return false;
			}
		}
	}

	if (finished) {
		// update brick index
		m_brickIndexGPU.Update(m_traceParams.m_cpuTracing, m_pBrickToSlot, m_pSlotTimestepMin, m_pSlotTimestepMax);
		cudaSafeCall(cudaEventRecord(m_brickIndexUploadEvent));
		printf("TracingManager::UploadWholeTimestep: update brick index\n");
	}

	return finished;
}

void TracingManager::TraceRound()
{
	++m_currentTimestamp;

	if(m_verbose)
	{
		printf("TracingManager::TraceRound %i: bricks", m_currentTimestamp);
		for(size_t i = 0; i < m_brickSlotInfo.size(); i++)
		{
			// cast to signed int, so "invalid" will show up as "-1"
			int brickIndex = int(m_brickSlotInfo[i].brickIndex);
			printf(" %i", brickIndex);
			if(LineModeIsTimeDependent(m_traceParams.m_lineMode))
			{
				int timestepMin = int(m_brickSlotInfo[i].timestepFirst);
				int timestepMax = timestepMin + int(m_brickSlotInfo[i].timestepCount) - 1;
				printf("(%i-%i)", timestepMin, timestepMax);
			}
		}
		printf("\n");
	}

	uint brickCount = m_pVolume->GetBrickCount().volume();
	uint slotCount = (uint)m_brickSlotInfo.size();

	// clear all brick requests
	m_brickRequestsGPU.Clear(m_traceParams.m_cpuTracing, brickCount);
	// update brick index. TODO only if something changed
	m_brickIndexGPU.Update(m_traceParams.m_cpuTracing, m_pBrickToSlot, m_pSlotTimestepMin, m_pSlotTimestepMax);
	cudaSafeCall(cudaEventRecord(m_brickIndexUploadEvent));

	// map d3d result buffer
	LineVertex* dpVB = nullptr;

	if (!m_traceParams.m_ftleEnabled)
	{
		cudaSafeCall(cudaGraphicsMapResources(1, &m_pResult->m_pVBCuda));
		cudaSafeCall(cudaGraphicsResourceGetMappedPointer((void**)&dpVB, nullptr, m_pResult->m_pVBCuda));
	}

	// integrate
	LineVertex* pVertices = (m_traceParams.m_cpuTracing ? m_lineVerticesCPU.data() : dpVB);
	LineInfo lineInfo(GetLineSpawnTime(), m_traceParams.m_lineCount, m_dpLineCheckpoints, pVertices, m_dpLineVertexCounts, m_traceParams.m_lineLengthMax);
	
	m_timerIntegrateCPU.Start();
	m_timerIntegrate.StartNextTimer();
	//cudaSafeCall(cudaDeviceSynchronize());

	if (m_traceParams.m_ftleEnabled)
		m_integrator.IntegrateLines(m_brickAtlas, lineInfo, m_traceParams, m_dpParticles);
	else
		m_integrator.IntegrateLines(m_brickAtlas, lineInfo, m_traceParams);

	//cudaSafeCall(cudaDeviceSynchronize());
	m_timerIntegrate.StopCurrentTimer();
	m_timerIntegrateCPU.Stop();
	m_timings.IntegrateCPU += m_timerIntegrateCPU.GetElapsedTimeMS();

	// copy vertices into d3d buffer
	if(m_traceParams.m_cpuTracing)
		cudaSafeCall(cudaMemcpy(dpVB, m_lineVerticesCPU.data(), m_lineVerticesCPU.size() * sizeof(LineVertex), cudaMemcpyHostToDevice));

#if 0
	//TEST!!
	m_lineVerticesCPU.resize(m_traceParams.m_lineCount * m_traceParams.m_lineLengthMax);
	cudaSafeCall(cudaMemcpy(m_lineVerticesCPU.data(), dpVB, m_lineVerticesCPU.size() * sizeof(LineVertex), cudaMemcpyDeviceToHost));
	for (const auto& v : m_lineVerticesCPU) 
	{
		printf("vertex (%f, %f, %f) : t=%f\n", v.Position.x, v.Position.y, v.Position.z, v.Heat);
	}
#endif

	// unmap d3d buffer again
	if (!m_traceParams.m_ftleEnabled)
		cudaSafeCall(cudaGraphicsUnmapResources(1, &m_pResult->m_pVBCuda));

	m_indexBufferDirty = true;

	// start download of brick requests. for stream lines, do *not* download timestep indices (not filled by the kernel!)
	uint* pBrickTimestepMins = LineModeIsTimeDependent(m_traceParams.m_lineMode) ? m_pBrickTimestepMins : nullptr;
	m_brickRequestsGPU.Download(m_traceParams.m_cpuTracing, m_pBrickRequestCounts, pBrickTimestepMins, brickCount);
	
	if(!m_traceParams.m_cpuTracing)
		cudaSafeCall(cudaEventRecord(m_brickRequestsDownloadEvent));

	m_bricksToDoDirty = true;
}


void TracingManager::BuildIndexBuffer()
{
	assert(IsTracing());
	if(!IsTracing()) return;

	if(!m_indexBufferDirty) return;

	if (m_pResult == nullptr)
		return;

	m_timerBuildIndexBuffer.StartNextTimer();

	// map d3d buffer
	cudaSafeCall(cudaGraphicsMapResources(1, &m_pResult->m_pIBCuda));
	uint* dpIB = nullptr;
	cudaSafeCall(cudaGraphicsResourceGetMappedPointer((void**)&dpIB, nullptr, m_pResult->m_pIBCuda));

	if(m_traceParams.m_cpuTracing)
	{
		std::vector<uint> indices(m_traceParams.m_lineCount * (m_traceParams.m_lineLengthMax - 1) * 2);
		m_pResult->m_indexCountTotal = m_integrator.BuildLineIndexBufferCPU(m_dpLineVertexCounts, m_traceParams.m_lineLengthMax, indices.data(), m_traceParams.m_lineCount);
		cudaSafeCall(cudaMemcpy(dpIB, indices.data(), m_pResult->m_indexCountTotal * sizeof(uint), cudaMemcpyHostToDevice));
	}
	else
	{
		m_pResult->m_indexCountTotal = m_integrator.BuildLineIndexBuffer(m_dpLineVertexCounts, m_traceParams.m_lineLengthMax, dpIB, m_traceParams.m_lineCount);
	}

	//if(m_verbose)
	//{
	//	printf("TracingManager::BuildIndexBuffer: total index count %u\n", m_pResult->m_indexCountTotal);
	//}

	//if(false) {
	//	std::vector<LineVertex> vertices(m_traceParams.m_lineCount * m_traceParams.m_lineLengthMax);
	//	cudaSafeCall(cudaMemcpy(vertices.data(), dpVB, vertices.size() * sizeof(LineVertex), cudaMemcpyDeviceToHost));
	//	std::vector<uint> indices(m_pResult->m_indexCountTotal);
	//	cudaSafeCall(cudaMemcpy(indices.data(), dpIB, indices.size() * sizeof(uint), cudaMemcpyDeviceToHost));
	//	int a=0;
	//}

	// unmap d3d buffer again
	cudaSafeCall(cudaGraphicsUnmapResources(1, &m_pResult->m_pIBCuda));

	m_timerBuildIndexBuffer.StopCurrentTimer();

	m_indexBufferDirty = false;
}


void TracingManager::BuildParticlesIndexBuffer()
{
	if (m_traceParams.m_cpuTracing)
	{
		printf("TracingManager::BuildParticlesIndexBuffer - ERROR: cpu tracing not supported\n");
		return;
	}

	m_timerBuildIndexBuffer.StartNextTimer();

	// map d3d buffer
	cudaSafeCall(cudaGraphicsMapResources(1, &m_pResult->m_pIBCuda));
	uint* dpIB = nullptr;
	cudaSafeCall(cudaGraphicsResourceGetMappedPointer((void**)&dpIB, nullptr, m_pResult->m_pIBCuda));

	m_pResult->m_indexCountTotal = m_integrator.BuildParticleIndexBuffer(dpIB, m_traceParams.m_lineLengthMax, m_traceParams.m_lineCount);

	// unmap d3d buffer again
	cudaSafeCall(cudaGraphicsUnmapResources(1, &m_pResult->m_pIBCuda));

	m_timerBuildIndexBuffer.StopCurrentTimer();
}



void TracingManager::ReleaseResources()
{
	CancelTracing();

	ReleaseParamDependentResources();
	ReleaseVolumeDependentResources();

	IntegratorTimeInCell::Free(m_cellTextureGPU);
}

void TracingManager::UpdateBricksToDo()
{
	bool resort = false;

	if(m_bricksToDoDirty)
	{
		if(m_verbose)
		{
			printf("TracingManager::UpdateBricksToDo: collecting bricks\n");
		}

		// collect all bricks with lines to do
		uint brickCount = m_pVolume->GetBrickCount().volume();
		m_bricksToDo.clear();
		if(!m_traceParams.m_cpuTracing)
			cudaSafeCall(cudaEventSynchronize(m_brickRequestsDownloadEvent));
		//if(m_verbose)
		//{
		//	printf("TracingManager::UpdateBricksToDo: request counts");
		//}
		for(uint brickIndex = 0; brickIndex < brickCount; brickIndex++)
		{
			uint requestCount = m_pBrickRequestCounts[brickIndex];
			//if(m_verbose)
			//{
			//	printf(" %u", requestCount);
			//}
			if(requestCount != 0)
			{
				uint timestep = m_pBrickTimestepMins[brickIndex];
				m_bricksToDo.push_back(BrickSortItem(brickIndex, timestep, float(requestCount)));
			}
		}
		//if(m_verbose)
		//{
		//	printf("\n");
		//}

		m_bricksToDoDirty = false;
		m_bricksToDoPrioritiesDirty = true;
		resort = true;
	}

	if(m_bricksToDoPrioritiesDirty)
	{
		if(m_verbose)
		{
			printf("TracingManager::UpdateBricksToDo: updating priorities\n");
		}

		if(!m_traceParams.m_cpuTracing)
			cudaSafeCall(cudaEventSynchronize(m_brickRequestsDownloadEvent));
		for(auto it = m_bricksToDo.begin(); it != m_bricksToDo.end(); ++it)
		{
			BrickSortItem& brick = *it;
			uint brickParticleCount = m_pBrickRequestCounts[brick.m_linearBrickIndex];
			float brickParticleFactor = float(brickParticleCount);
			if(m_traceParams.HeuristicDoSqrtBPCount())
			{
				brickParticleFactor = sqrtf(brickParticleFactor);
			}

			uint neighbors[6];
			GetBrickNeighbors(brick.m_linearBrickIndex, neighbors);
			float bonus = 0.0f;
			float penalty = 0.0f;
			for(int dir = 0; dir < 6; ++dir)
			{
				uint neighbor = neighbors[dir];
				// skip neighbors that don't exist (at the border of the domain)
				if(neighbor == uint(-1)) continue;

				uint neighborParticleCount = m_pBrickRequestCounts[neighbor];
				float neighborParticleFactor = float(neighborParticleCount);
				if(m_traceParams.HeuristicDoSqrtBPCount())
				{
					neighborParticleFactor = sqrtf(neighborParticleFactor);
				}

				if(BrickIsOnGPU(neighbor, brick.m_timestepFrom, brick.m_timestepFrom) && m_brickSlotInfo[m_bricksOnGPU[neighbor]].HasValidTimestamp())
				{
					// going out
					float probabilityOut = 1.0f / 6.0f;
					if(m_traceParams.HeuristicUseFlowGraph() && m_pFlowGraph->IsBuilt())
					{
						const FlowGraph::BrickInfo& info = m_pFlowGraph->GetBrickInfo(brick.m_linearBrickIndex);
						probabilityOut = info.OutFreq[dir];
					}
					if(m_traceParams.HeuristicDoSqrtBPProb())
					{
						probabilityOut = sqrtf(probabilityOut);
					}
					bonus += probabilityOut * brickParticleFactor;

					// coming in
					float probabilityIn = 1.0f / 6.0f;
					if(m_traceParams.HeuristicUseFlowGraph() && m_pFlowGraph->IsBuilt())
					{
						const FlowGraph::BrickInfo& info = m_pFlowGraph->GetBrickInfo(neighbor);
						int n = (dir + 3) % 6; // flip sign of direction (DIR_POS_X <-> DIR_NEG_X etc)
						probabilityIn = info.OutFreq[n];
					}
					if(m_traceParams.HeuristicDoSqrtBPProb())
					{
						probabilityIn = sqrtf(probabilityIn);
					}
					bonus += probabilityIn * neighborParticleFactor;
				}
				else
				{
					// coming in
					float probabilityIn = 1.0f / 6.0f;
					if(m_traceParams.HeuristicUseFlowGraph() && m_pFlowGraph->IsBuilt())
					{
						const FlowGraph::BrickInfo& info = m_pFlowGraph->GetBrickInfo(neighbor);
						int n = (dir + 3) % 6; // flip sign of direction (DIR_POS_X <-> DIR_NEG_X etc)
						probabilityIn = info.OutFreq[n];
					}
					if(m_traceParams.HeuristicDoSqrtBPProb())
					{
						probabilityIn = sqrtf(probabilityIn);
					}
					penalty += probabilityIn * neighborParticleFactor;
				}
			}

			// base priority: number of particles in this brick
			brick.m_priority = brickParticleFactor;

			// modify priority according to heuristic
			brick.m_priority += m_traceParams.m_heuristicBonusFactor   * bonus;
			brick.m_priority -= m_traceParams.m_heuristicPenaltyFactor * penalty;
		}

		m_bricksToDoPrioritiesDirty = false;
		resort = true;
	}

	if(resort)
	{
		// sort bricks in todo list by priority
		std::sort(m_bricksToDo.begin(), m_bricksToDo.end());
	}
}

void TracingManager::UpdateBricksToLoad()
{
	m_bricksToLoad.clear();

	if(m_pVolume == nullptr) return;

	// collect all bricks with lines to do
	UpdateBricksToDo();

	// ..and add to bricks to load
	uint timeSlotCount = m_brickSlotCount.x();
	for(auto it = m_bricksToDo.cbegin(); it != m_bricksToDo.cend(); it++)
	{
		int timestepFrom = it->m_timestepFrom;
		int timestepTo = min(timestepFrom + timeSlotCount - 1, m_timestepMax);
		for(int timestep = timestepFrom; timestep <= timestepTo; timestep++)
		{
			m_bricksToLoad.push_back(&m_pVolume->GetBrick(timestep, it->m_linearBrickIndex));
		}
	}

	// prefetching
	if(m_traceParams.m_enablePrefetching)
	{
		if(LineModeIsTimeDependent(m_traceParams.m_lineMode))
		{
			// for path lines, add later timesteps of bricks in todo list
			for(auto it = m_bricksToDo.cbegin(); it != m_bricksToDo.cend(); it++)
			{
				int timestepFrom = it->m_timestepFrom + timeSlotCount;
				int timestepTo = min(timestepFrom + timeSlotCount - 1, m_timestepMax);
				for(int timestep = timestepFrom; timestep <= timestepTo; timestep++)
				{
					m_bricksToLoad.push_back(&m_pVolume->GetBrick(timestep, it->m_linearBrickIndex));
				}
			}
		}
		else
		{
			// for stream lines, add neighbors of bricks in todo list
			Vec3i brickCount = m_pVolume->GetBrickCount();

			std::unordered_set<uint> indicesDone;
			for(auto it = m_bricksToDo.cbegin(); it != m_bricksToDo.cend(); it++)
			{
				indicesDone.insert(it->m_linearBrickIndex);
			}

			int timestep = GetLineFloorTimestepIndex(0.0f);
			for(auto it = m_bricksToDo.cbegin(); it != m_bricksToDo.cend(); it++)
			{
				std::vector<uint> neighbors = GetBrickNeighbors(it->m_linearBrickIndex);
				for(size_t i = 0; i < neighbors.size(); ++i)
				{
					uint neighbor = neighbors[i];
					if(indicesDone.count(neighbor) == 0)
					{
						indicesDone.insert(neighbor);
						m_bricksToLoad.push_back(&m_pVolume->GetBrick(timestep, neighbor));
					}
				}
			}
		}

		//// add other bricks (with no lines to do) in arbitrary order (by index)
		////TODO right now, this only makes sense for stream lines
		////TODO sort by distance to nearest brick with lines to do or something?
		//auto& bricks = m_pVolume->GetNearestTimestep().bricks;
		//for(uint index = 0; index < bricks.size(); index++)
		//{
		//	// we add some bricks again here, but that shouldn't matter
		//	m_bricksToLoad.push_back(&bricks[index]);
		//}

		//TODO prefetch next timesteps?
	}
}


void TracingManager::UpdateProgress()
{
	float spawnTime = GetLineSpawnTime();

	float linesDone = 0.0f;
	//for(uint i = 0; i < m_traceParams.m_lineCount; i++)
	//{
	//	if(InsideOfDomain(m_pLineCheckpoints[i].Position, m_pVolume->GetVolumeHalfSizeWorld()))
	//	{
	//		float progressBySteps = min(float(m_pLineVertexCounts[i]) / float(m_traceParams.m_lineLengthMax), 1.0f);
	//		float progressByAge = min((m_pLineCheckpoints[i].Time - spawnTime) / m_traceParams.m_lineAgeMax, 1.0f);
	//		linesDone += max(progressBySteps, progressByAge);
	//	}
	//	else
	//	{
	//		linesDone += 1.0f;
	//	}
	//}
	m_progress = linesDone / float(m_traceParams.m_lineCount);
}


void TracingManager::ComputeTimings()
{
	m_timings.UploadDecompressGPU = m_timerUploadDecompress.GetStats();
	m_timings.IntegrateGPU        = m_timerIntegrate.GetStats();
	m_timings.BuildIndexBufferGPU = m_timerBuildIndexBuffer.GetStats();

	if(m_timerDiskWait.IsRunning())
	{
		m_timerDiskWait.Stop();
		m_timings.WaitDiskWall += m_timerDiskWait.GetElapsedTimeMS();
	}

	m_timerTrace.Stop();
	m_timings.TraceWall = m_timerTrace.GetElapsedTimeMS();
}

void TracingManager::PrintTimings() const
{
	printf("Done tracing in %.2f s (disk wait: %.2f s).\n", m_timings.TraceWall / 1000.0f, m_timings.WaitDiskWall / 1000.0f);
	printf("Upload/Decomp (GPU): %.2f ms (%.2f-%.2f-%.2f : %u)\n",
		m_timings.UploadDecompressGPU.Total,
		m_timings.UploadDecompressGPU.Min, m_timings.UploadDecompressGPU.Avg, m_timings.UploadDecompressGPU.Max,
		m_timings.UploadDecompressGPU.Count);
	printf("Integrate (GPU): %.2f ms (%.2f-%.2f-%.2f : %u)\n",
		m_timings.IntegrateGPU.Total,
		m_timings.IntegrateGPU.Min, m_timings.IntegrateGPU.Avg, m_timings.IntegrateGPU.Max,
		m_timings.IntegrateGPU.Count);
	printf("Integrate (CPU): %.2f ms\n", m_timings.IntegrateCPU);
	printf("Build Index Buffer (GPU): %.2f ms (%.2f-%.2f-%.2f : %u)\n",
		m_timings.BuildIndexBufferGPU.Total,
		m_timings.BuildIndexBufferGPU.Min, m_timings.BuildIndexBufferGPU.Avg, m_timings.BuildIndexBufferGPU.Max,
		m_timings.BuildIndexBufferGPU.Count);
}


void TracingManager::ComputeStats()
{
	m_stats.BricksUploadedUnique = uint(m_bricksLoaded.size());

	// download final checkpoints and gather stats
	std::vector<LineCheckpoint> checkpoints(m_traceParams.m_lineCount);
	cudaMemcpyKind copyDir = m_traceParams.m_cpuTracing ? cudaMemcpyHostToHost : cudaMemcpyDeviceToHost;
	cudaSafeCall(cudaMemcpy(checkpoints.data(), m_dpLineCheckpoints, checkpoints.size() * sizeof(LineCheckpoint), copyDir));

	float lineTimeMax = GetLineTimeMax();
	for(size_t i = 0; i < checkpoints.size(); i++)
	{
		m_stats.StepsTotal    += checkpoints[i].StepsTotal;
		m_stats.StepsAccepted += checkpoints[i].StepsAccepted;

		if(checkpoints[i].Time >= lineTimeMax) m_stats.LinesReachedMaxAge++;
		else if(OutsideOfDomain(checkpoints[i].Position, m_pVolume->GetVolumeHalfSizeWorld())) m_stats.LinesLeftDomain++;
	}
	assert(m_stats.StepsTotal >= m_stats.StepsAccepted);
	uint stepsRejected = m_stats.StepsTotal - m_stats.StepsAccepted;
	m_stats.Evaluations = m_stats.StepsAccepted * GetAdvectModeEvaluationsPerAcceptedStep(m_traceParams.m_advectMode);
	m_stats.Evaluations += stepsRejected * GetAdvectModeEvaluationsPerRejectedStep(m_traceParams.m_advectMode);

	// DiskBusyTimeMS/DiskBytesLoaded atm contain the numbers when we started tracing
	m_stats.DiskBusyTimeMS = m_pVolume->GetTotalDiskBusyTimeMS() - m_stats.DiskBusyTimeMS;
	m_stats.DiskBytesLoaded = m_pVolume->GetTotalLoadedBytes() - m_stats.DiskBytesLoaded;

	m_stats.DiskBytesUsed = 0;
	uint brickCount = m_pVolume->GetBrickCount().volume();
	for(auto it = m_bricksLoaded.cbegin(); it != m_bricksLoaded.cend(); it++)
	{
		uint index = *it;

		uint timestep = index / brickCount;
		uint brickLinearIndex = index % brickCount;
		m_stats.DiskBytesUsed += m_pVolume->GetBrick(timestep, brickLinearIndex).GetPaddedDataSize();
	}
}

void TracingManager::PrintStats() const
{
	printf("Total (unique) bricks uploaded: %u (%u) in %u slots\n", m_stats.BricksUploadedTotal, m_stats.BricksUploadedUnique, m_brickSlotCount.yz().area());
	printf("Total steps: %u  accepted: %u    Evaluations: %u\n", m_stats.StepsTotal, m_stats.StepsAccepted, m_stats.Evaluations);
	uint linesRest = m_traceParams.m_lineCount - m_stats.LinesReachedMaxAge - m_stats.LinesLeftDomain;
	printf("#Lines reached max age: %u  left domain: %u  other: %u\n", m_stats.LinesReachedMaxAge, m_stats.LinesLeftDomain, linesRest);
	printf("Disk IO: %.2f MB  Used: %.2f MB  Busy: %.2f s\n", float(m_stats.DiskBytesLoaded) / (1024.0f * 1024.0f), float(m_stats.DiskBytesUsed) / (1024.0f * 1024.0f), m_stats.DiskBusyTimeMS / 1000.0f);
}


float TracingManager::GetLineSpawnTime() const
{
	return (LineModeIsTimeDependent(m_traceParams.m_lineMode)) ? m_pVolume->GetCurTime() : 0.0f;
}

int TracingManager::GetLineFloorTimestepIndex(float lineTime) const
{
	return (LineModeIsTimeDependent(m_traceParams.m_lineMode)) ? m_pVolume->GetFloorTimestepIndex(lineTime) : m_pVolume->GetCurNearestTimestepIndex();
}

float TracingManager::GetLineTimeMax() const
{
	float spawnTime = GetLineSpawnTime();
	float lineTimeMax = spawnTime + m_traceParams.m_lineAgeMax;
	if(LineModeIsTimeDependent(m_traceParams.m_lineMode))
	{
		float volumeTimeMax = (m_pVolume->GetTimestepCount() - 1) * m_pVolume->GetTimeSpacing();
		lineTimeMax = min(lineTimeMax, volumeTimeMax);
	}
	// HACK: subtract small epsilon to avoid numerical snafu
	return lineTimeMax - 1e-4f;
}
