#include "TracingManager.h"

#include <cassert>
#include <fstream>
#include <random>
#include <chrono>

#include "cudaUtil.h"
#include "cudaTum3D.h"

#include "BrickUpload.h"


using namespace tum3D;


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

#pragma region Timings
bool TracingManager::Timings::WriteToFile(const std::string& filename) const
{
	std::ofstream file(filename);
	return WriteToFile(file);
}

bool TracingManager::Timings::WriteToFile(std::ostream& file) const
{
	if (!file.good()) return false;

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
	if (!file.good()) return false;

	for (auto& str : extraColumns)
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
	if (!file.good()) return false;

	for (auto& str : extraColumns)
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
#pragma endregion

#pragma region Stats
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
	if (!file.good()) return false;

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
	if (!file.good()) return false;

	for (auto& str : extraColumns)
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
	if (!file.good()) return false;

	for (auto& str : extraColumns)
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
#pragma endregion


TracingManager::TracingManager()
	: m_roundsPerFrame(0), m_uploadsPerFrame(0)
	, m_isCreated(false), m_pDevice(nullptr)
	, m_currentTimestamp(0)
	, m_dpLineCheckpoints(nullptr), m_dpLineVertexCounts(nullptr)
	, m_pResult(nullptr), m_indexBufferDirty(true)
	, m_pFlowGraph(nullptr), m_timestepMax(0)
	, m_progress(0.0f)
	, m_verbose(false)
	, m_engine(std::chrono::system_clock::now().time_since_epoch().count())
	, m_numRejections(20)
	, m_seedManyParticles(false)
	, m_traceableVol(nullptr)
{
}

TracingManager::~TracingManager()
{
	assert(!IsCreated());
}


bool TracingManager::Create(ID3D11Device* pDevice)
{
	std::cout << "Creating TracingManager..." << std::endl;

	m_pDevice = pDevice;

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

	m_integrator.Release();

	if (m_traceableVol)
	{
		m_traceableVol->ReleaseResources();
		delete m_traceableVol;
		m_traceableVol = nullptr;
	}

	ReleaseResultResources();
	ReleaseParamDependentResources();
	
	m_pDevice = nullptr;
	m_isCreated = false;
}

void TracingManager::ReleaseResources()
{
	CancelTracing();

	ReleaseParamDependentResources();
	
	if (m_traceableVol)
	{
		m_traceableVol->ReleaseResources();
		delete m_traceableVol;
		m_traceableVol = nullptr;
	}

	IntegratorTimeInCell::Free(m_cellTextureGPU);
}

HRESULT TracingManager::CreateParamDependentResources()
{
	std::cout << "CreateParamDependentResources()" << std::endl;

	ReleaseParamDependentResources();

	if (m_traceParams.m_cpuTracing)
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
	if (m_traceParams.m_cpuTracing)
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


void TracingManager::SetParams(const ParticleTraceParams& traceParams)
{
	m_traceParams = traceParams;
	SetCheckpointTimeStep(m_traceParams.m_advectDeltaT);
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


#pragma region InitialCheckpoints
void TracingManager::CreateInitialCheckpoints(float spawnTime)
{
	m_checkpoints.resize(m_traceParams.m_lineCount);

	float3 seedBoxMin = make_float3(m_traceParams.m_seedBoxMin);
	float3 seedBoxSize = make_float3(m_traceParams.m_seedBoxSize);

	if (m_traceParams.m_upsampledVolumeHack)
	{
		// upsampled volume is offset by half a grid spacing...
		float gridSpacingWorld = 2.0f / float(m_traceableVol->m_pVolume->GetVolumeSize().maximum());
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
				Vec3f posInVolume = (posAsVec + m_traceableVol->m_pVolume->GetVolumeHalfSizeWorld()) / (2 * m_traceableVol->m_pVolume->GetVolumeHalfSizeWorld());
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
#pragma endregion


bool TracingManager::StartTracing(const TimeVolume& volume, const ParticleTraceParams& traceParams, const FlowGraph& flowGraph)
{
	if (!volume.IsOpen())
		return false;

	if (IsTracing())
	{
		std::cout << volume.GetName() << " was tracing. Cancelling current tracing." << std::endl;
		CancelTracing();
	}

	//HACK for now, release resources first
	ReleaseResources();
	ReleaseResultResources();

	printf("\n----------------------------------------------------------------------\nTracingManager::StartTracing: %s\n", volume.GetName().c_str());

	m_traceableVol = new TraceableVolume(&volume);

	m_traceParams = traceParams;
	m_pFlowGraph = &flowGraph;
	m_progress = 0.0f;

	m_currentTimestamp = 0;

	//TODO recreate only if something relevant changed
	if (FAILED(CreateParamDependentResources()))
	{
		MessageBoxA(nullptr, "TracingManager::StartTracing: Failed creating param-dependent resources! (probably not enough GPU memory)", "Fail", MB_OK | MB_ICONINFORMATION);
		CancelTracing();
		return false;
	}
	if (FAILED(CreateResultResources()))
	{
		MessageBoxA(nullptr, "TracingManager::StartTracing: Failed creating result resources! (probably not enough GPU memory)", "Fail", MB_OK | MB_ICONINFORMATION);
		CancelTracing();
		return false;
	}

	uint timestepMinInit = LineModeIsTimeDependent(m_traceParams.m_lineMode) ? ~0u : GetLineFloorTimestepIndex(GetLineSpawnTime());

	// create volume-dependent resources last - they will take up all available GPU memory
	//if (FAILED(CreateVolumeDependentResources()))
	if (!m_traceableVol->CreateResources(timestepMinInit, m_traceParams.m_cpuTracing, LineModeIsTimeDependent(m_traceParams.m_lineMode)))
	{
		MessageBoxA(nullptr, "TracingManager::StartTracing: Failed creating volume-dependent resources! (probably not enough GPU memory)", "Fail", MB_OK | MB_ICONINFORMATION);
		CancelTracing();
		return false;
	}

	m_integrator.SetVolumeInfo(volume.GetInfo());

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
	if (LineModeIsTimeDependent(m_traceParams.m_lineMode))
	{
		float timeMax = spawnTime + m_traceParams.m_lineAgeMax;
		m_timestepMax = GetLineFloorTimestepIndex(timeMax) + 1;
	}
	m_timestepMax = min(m_timestepMax, volume.GetTimestepCount() - 1);


	m_stats.Clear();
	m_stats.DiskBusyTimeMS = volume.GetTotalDiskBusyTimeMS();
	m_stats.DiskBytesLoaded = volume.GetTotalLoadedBytes();
	m_traceableVol->m_bricksLoaded.clear();


	if (m_traceParams.HeuristicUseFlowGraph() && !m_pFlowGraph->IsBuilt())
		std::cout << "TracingManager: Warning: heuristic wants to use flow graph, but it's not built\n" << std::endl;

	// start timing
	m_traceableVol->m_timerUploadDecompress.ResetTimers();
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
		m_traceableVol->UpdateBricksToLoad(m_traceParams.m_cpuTracing, m_traceParams.m_enablePrefetching, LineModeIsTimeDependent(m_traceParams.m_lineMode), GetLineFloorTimestepIndex(0.0f), m_timestepMax, GetHeuristics());
	}

	return true;
}

bool TracingManager::Trace()
{
	assert(IsTracing());
	if(!IsTracing()) return false;

	if (m_traceableVol->m_timerDiskWait.IsRunning())
	{
		m_traceableVol->m_timerDiskWait.Stop();
		m_timings.WaitDiskWall += m_traceableVol->m_timerDiskWait.GetElapsedTimeMS();
	}

	//particles
	if (LineModeIsIterative(m_traceParams.m_lineMode))
		return TraceParticlesIteratively();

	uint uploadBudget = m_uploadsPerFrame;
	for(uint round = 0; round < m_roundsPerFrame; round++)
	{
		m_traceableVol->UpdateBricksToDo(m_traceParams.m_cpuTracing, GetHeuristics());

		// if there are no more lines to do, we're done
		//if (m_bricksToDo.empty())
		if (!m_traceableVol->HasThingsToDo())
		{
			BuildIndexBuffer();

			ComputeTimings();
			PrintTimings();

			ComputeStats();
			PrintStats();

			//writeOutFinishedLine();

			//m_traceableVol->m_bricksToLoad.clear();

			m_traceableVol->ReleaseResources();
			delete m_traceableVol;
			m_traceableVol = nullptr;

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
		if (!m_traceableVol->UploadBricks(m_traceParams.m_cpuTracing, m_traceParams.m_waitForDisk, uploadBudget, false, m_currentTimestamp, m_timestepMax, LineModeIsTimeDependent(m_traceParams.m_lineMode), m_traceParams.m_purgeTimeoutInRounds, GetHeuristics()))
			break; // couldn't upload all bricks we want, so bail out

		if (m_traceableVol->TryToKickBricksFromGPU(LineModeIsTimeDependent(m_traceParams.m_lineMode), m_traceParams.m_cpuTracing, m_timestepMax))
			break;

		// still here -> all the bricks we want were uploaded, so go tracing
		TraceRound();
	}

	//TODO this syncs on the last trace round - move to top?
	m_traceableVol->UpdateBricksToLoad(m_traceParams.m_cpuTracing, m_traceParams.m_enablePrefetching, LineModeIsTimeDependent(m_traceParams.m_lineMode), GetLineFloorTimestepIndex(0.0f), m_timestepMax, GetHeuristics());

	return false;
}

void TracingManager::TraceRound()
{
	++m_currentTimestamp;

	if (m_verbose)
	{
		printf("TracingManager::TraceRound %i: bricks", m_currentTimestamp);
		for (size_t i = 0; i < m_traceableVol->m_brickSlotInfo.size(); i++)
		{
			// cast to signed int, so "invalid" will show up as "-1"
			int brickIndex = int(m_traceableVol->m_brickSlotInfo[i].brickIndex);
			printf(" %i", brickIndex);
			if (LineModeIsTimeDependent(m_traceParams.m_lineMode))
			{
				int timestepMin = int(m_traceableVol->m_brickSlotInfo[i].timestepFirst);
				int timestepMax = timestepMin + int(m_traceableVol->m_brickSlotInfo[i].timestepCount) - 1;
				printf("(%i-%i)", timestepMin, timestepMax);
			}
		}
		printf("\n");
	}

	m_traceableVol->ClearRequestsAndIndices(m_traceParams.m_cpuTracing);

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
		m_integrator.IntegrateLines(*m_traceableVol, lineInfo, m_traceParams, m_dpParticles);
	else
		m_integrator.IntegrateLines(*m_traceableVol, lineInfo, m_traceParams);

	//cudaSafeCall(cudaDeviceSynchronize());
	m_timerIntegrate.StopCurrentTimer();
	m_timerIntegrateCPU.Stop();
	m_timings.IntegrateCPU += m_timerIntegrateCPU.GetElapsedTimeMS();

	// copy vertices into d3d buffer
	if (m_traceParams.m_cpuTracing)
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

	m_traceableVol->DownloadRequests(m_traceParams.m_cpuTracing, LineModeIsTimeDependent(m_traceParams.m_lineMode));
}

bool TracingManager::TraceParticlesIteratively()
{
	//1. Upload all bricks to the GPU
	if (m_particlesNeedsUploadTimestep) 
	{
		int timestep = m_traceableVol->m_pVolume->GetCurNearestTimestepIndex();
		printf("Upload all bricks at timestep %d\n", timestep);
		if (!m_traceableVol->UploadWholeTimestep(m_traceParams.m_cpuTracing, m_traceParams.m_waitForDisk, timestep, false, m_currentTimestamp, LineModeIsTimeDependent(m_traceParams.m_lineMode), m_traceParams.m_purgeTimeoutInRounds)) {
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
	std::chrono::duration<double> time_passed = std::chrono::duration_cast<std::chrono::duration<double >> (tp - m_particlesLastTime);
	double timeBetweenSeeds = 1.0 / m_traceParams.m_particlesPerSecond;

	if (timeBetweenSeeds < time_passed.count() || m_seedManyParticles) 
	{
		if (LineModeGenerateAlwaysNewSeeds(m_traceParams.m_lineMode)) 
		{
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
	m_integrator.IntegrateParticles(*m_traceableVol, lineInfo, m_traceParams, seed, tpf * 100);

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
		assert(m_traceableVol);

		//printf("TracingManager::CancelTracing: %s\n", m_traceableVol->m_pVolume->GetName().c_str());
		ReleaseResultResources(); //TODO hrm..?

		//m_traceableVol->m_bricksToDo.clear();
		//m_traceableVol->m_bricksToLoad.clear();
		m_traceableVol->ReleaseResources();
		delete m_traceableVol;
		m_traceableVol = nullptr;
		m_pFlowGraph = nullptr;
	}
}

bool TracingManager::IsTracing() const
{
	return m_traceableVol != nullptr;
}

float TracingManager::GetTracingProgress() const
{
	if(!IsTracing()) return 0.0f;

	return m_progress;
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


void TracingManager::ClearResult()
{
	ReleaseResultResources();
}


void TracingManager::UpdateFrameBudgets()
{
	// HACK: for uploads, scale with linear brick size (instead of volume)
	//       to account for larger overhead with smaller bricks...
	const uint brickSizeRef = 256;
	const uint brickSizeRef2 = brickSizeRef * brickSizeRef;
	uint brickSize = m_traceableVol->m_pVolume->GetBrickSizeWithOverlap();
	uint timeSlotCount = m_traceableVol->m_brickSlotCount.x();
	m_uploadsPerFrame = max(1, brickSizeRef / (brickSize * timeSlotCount)); //TODO remove timeSlotCount here, instead count one upload per time slot

	m_roundsPerFrame = 8; //TODO scale with interpolation mode
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


void TracingManager::ComputeTimings()
{
	m_timings.UploadDecompressGPU = m_traceableVol->m_timerUploadDecompress.GetStats();
	m_timings.IntegrateGPU        = m_timerIntegrate.GetStats();
	m_timings.BuildIndexBufferGPU = m_timerBuildIndexBuffer.GetStats();

	if (m_traceableVol->m_timerDiskWait.IsRunning())
	{
		m_traceableVol->m_timerDiskWait.Stop();
		m_timings.WaitDiskWall += m_traceableVol->m_timerDiskWait.GetElapsedTimeMS();
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
	m_stats.BricksUploadedUnique = uint(m_traceableVol->m_bricksLoaded.size());

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
		else if (OutsideOfDomain(checkpoints[i].Position, m_traceableVol->m_pVolume->GetVolumeHalfSizeWorld())) m_stats.LinesLeftDomain++;
	}
	assert(m_stats.StepsTotal >= m_stats.StepsAccepted);
	uint stepsRejected = m_stats.StepsTotal - m_stats.StepsAccepted;
	m_stats.Evaluations = m_stats.StepsAccepted * GetAdvectModeEvaluationsPerAcceptedStep(m_traceParams.m_advectMode);
	m_stats.Evaluations += stepsRejected * GetAdvectModeEvaluationsPerRejectedStep(m_traceParams.m_advectMode);

	// DiskBusyTimeMS/DiskBytesLoaded atm contain the numbers when we started tracing
	m_stats.DiskBusyTimeMS = m_traceableVol->m_pVolume->GetTotalDiskBusyTimeMS() - m_stats.DiskBusyTimeMS;
	m_stats.DiskBytesLoaded = m_traceableVol->m_pVolume->GetTotalLoadedBytes() - m_stats.DiskBytesLoaded;

	m_stats.DiskBytesUsed = 0;
	uint brickCount = m_traceableVol->m_pVolume->GetBrickCount().volume();
	for (auto it = m_traceableVol->m_bricksLoaded.cbegin(); it != m_traceableVol->m_bricksLoaded.cend(); it++)
	{
		uint index = *it;

		uint timestep = index / brickCount;
		uint brickLinearIndex = index % brickCount;
		m_stats.DiskBytesUsed += m_traceableVol->m_pVolume->GetBrick(timestep, brickLinearIndex).GetPaddedDataSize();
	}
}

void TracingManager::PrintStats() const
{
	printf("Total (unique) bricks uploaded: %u (%u) in %u slots\n", m_stats.BricksUploadedTotal, m_stats.BricksUploadedUnique, m_traceableVol->m_brickSlotCount.yz().area());
	printf("Total steps: %u  accepted: %u    Evaluations: %u\n", m_stats.StepsTotal, m_stats.StepsAccepted, m_stats.Evaluations);
	uint linesRest = m_traceParams.m_lineCount - m_stats.LinesReachedMaxAge - m_stats.LinesLeftDomain;
	printf("#Lines reached max age: %u  left domain: %u  other: %u\n", m_stats.LinesReachedMaxAge, m_stats.LinesLeftDomain, linesRest);
	printf("Disk IO: %.2f MB  Used: %.2f MB  Busy: %.2f s\n", float(m_stats.DiskBytesLoaded) / (1024.0f * 1024.0f), float(m_stats.DiskBytesUsed) / (1024.0f * 1024.0f), m_stats.DiskBusyTimeMS / 1000.0f);
}


float TracingManager::GetLineSpawnTime() const
{
	return (LineModeIsTimeDependent(m_traceParams.m_lineMode)) ? m_traceableVol->m_pVolume->GetCurTime() : 0.0f;
}

int TracingManager::GetLineFloorTimestepIndex(float lineTime) const
{
	return (LineModeIsTimeDependent(m_traceParams.m_lineMode)) ? m_traceableVol->m_pVolume->GetFloorTimestepIndex(lineTime) : m_traceableVol->m_pVolume->GetCurNearestTimestepIndex();
}

float TracingManager::GetLineTimeMax() const
{
	float spawnTime = GetLineSpawnTime();
	float lineTimeMax = spawnTime + m_traceParams.m_lineAgeMax;
	if(LineModeIsTimeDependent(m_traceParams.m_lineMode))
	{
		float volumeTimeMax = (m_traceableVol->m_pVolume->GetTimestepCount() - 1) * m_traceableVol->m_pVolume->GetTimeSpacing();
		lineTimeMax = min(lineTimeMax, volumeTimeMax);
	}
	// HACK: subtract small epsilon to avoid numerical snafu
	return lineTimeMax - 1e-4f;
}
