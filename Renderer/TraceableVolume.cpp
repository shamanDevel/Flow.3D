#include <TraceableVolume.h>


#include <cudaUtil.h>
#include <cudaTum3D.h>
#include <BrickUpload.h>
#include <TextureCPU.h>

using namespace tum3D;

TextureCPU<float4>			g_volume;

static Vec3i SpatialNeighbor(const Vec3i& vec, int dim, int dist)
{
	Vec3i result(vec);
	result[dim] += dist;
	return result;
}

Vec2ui Fit2DSlotCount(uint slotCountTotal, uint slotCount1DMax)
{
	Vec2ui slotCountBest(min(slotCountTotal, slotCount1DMax), 1);

	uint slotCountXMin = max(1, uint(sqrt((double)slotCount1DMax)));
	for (uint slotCountX = slotCount1DMax; slotCountX >= slotCountXMin; --slotCountX)
	{
		uint slotCountY = slotCountTotal / slotCountX;
		if (slotCountX * slotCountY > slotCountBest.area())
		{
			slotCountBest.set(slotCountX, slotCountY);
		}
	}

	return slotCountBest;
}




TraceableVolume::TraceableVolume(const TimeVolume* volume)
	: m_brickSlotCountMax(1024), m_timeSlotCountMax(8)
	, m_verbose(false)
	, m_isCreated(false)
	, m_pBrickToSlot(nullptr), m_pSlotTimestepMin(nullptr), m_pSlotTimestepMax(nullptr), m_brickIndexUploadEvent(0)
	, m_pBrickRequestCounts(nullptr), m_pBrickTimestepMins(nullptr), m_brickRequestsDownloadEvent(0)
	, m_brickSlotCount(0, 0, 0)
	, m_bricksToDoDirty(true), m_bricksToDoPrioritiesDirty(true)
	, m_pCompressShared(nullptr)
	, m_pCompressVolume(nullptr)
	, m_pVolume(nullptr)
	, m_volumeInfoGPU(nullptr)
	, m_brickIndexGPU(nullptr)
	, m_brickRequestsGPU(nullptr)
	, m_brickAtlas(nullptr)
{
	m_pVolume = volume;
}

TraceableVolume::~TraceableVolume()
{
	assert(!m_isCreated);

	ReleaseResources();

	m_pVolume = nullptr;
}


size_t TraceableVolume::BrickTimeSizeInBytes(const TimeVolume* volume, size_t timeSlotCount)
{
	assert(volume);

	size_t brickSize = volume->GetBrickSizeWithOverlap();
	size_t channelCount = volume->GetChannelCount();
	size_t channelCountTex = (channelCount == 3) ? 4 : channelCount;

	return brickSize * brickSize * brickSize * channelCountTex * timeSlotCount * sizeof(float);
}

size_t TraceableVolume::TimeStepSizeInBytes(const TimeVolume* volume)
{
	return volume->GetBrickCount().volume() * BrickTimeSizeInBytes(volume, 1);
}


bool TraceableVolume::CreateResources(uint minTimestepIndex, bool cpuTracing, bool timeDependent, int gpuMemUsageLimitMB)
{
	//if (m_pVolume == nullptr || !m_pVolume->IsOpen())
	//	return false;
	assert(m_pVolume != nullptr);
	assert(m_pVolume->IsOpen());

	ReleaseResources();

	m_pCompressShared = new GPUResources();
	m_pCompressVolume = new CompressVolumeResources();

	if (m_pVolume->IsCompressed())
	{
		uint brickSize = m_pVolume->GetBrickSizeWithOverlap();
		// do multi-channel decoding only for small bricks; for large bricks, mem usage gets too high
		uint channelCount = (brickSize <= 128) ? m_pVolume->GetChannelCount() : 1;
		uint huffmanBits = m_pVolume->GetHuffmanBitsMax();
		m_pCompressShared->create(CompressVolumeResources::getRequiredResources(brickSize, brickSize, brickSize, channelCount, huffmanBits));
		m_pCompressVolume->create(m_pCompressShared->getConfig());
	}

	m_brickIndexGPU = new BrickIndexGPU();
	m_brickIndexGPU->Init();

	m_brickRequestsGPU = new BrickRequestsGPU();
	m_brickRequestsGPU->Init();

	m_volumeInfoGPU = new VolumeInfoGPU();

	// probably no need to sync on the previous upload here..?
	m_volumeInfoGPU->Fill(m_pVolume->GetInfo());
	m_volumeInfoGPU->Upload(cpuTracing);

	m_brickIndexGPU->Upload(cpuTracing);
	m_brickRequestsGPU->Upload(cpuTracing);

	cudaSafeCall(cudaHostRegister(&m_volumeInfoGPU, sizeof(m_volumeInfoGPU), cudaHostRegisterDefault));

	cudaSafeCall(cudaHostRegister(&m_brickIndexGPU, sizeof(m_brickIndexGPU), cudaHostRegisterDefault));
	cudaSafeCall(cudaEventCreate(&m_brickIndexUploadEvent, cudaEventDisableTiming));
	cudaSafeCall(cudaEventRecord(m_brickIndexUploadEvent));

	cudaSafeCall(cudaHostRegister(&m_brickRequestsGPU, sizeof(m_brickRequestsGPU), cudaHostRegisterDefault));
	cudaSafeCall(cudaEventCreate(&m_brickRequestsDownloadEvent, cudaEventDisableTiming));
	cudaSafeCall(cudaEventRecord(m_brickRequestsDownloadEvent));

	int device = -1;
	cudaSafeCall(cudaGetDevice(&device));
	cudaDeviceProp devProp;
	cudaSafeCall(cudaGetDeviceProperties(&devProp, device));


	uint brickSize = m_pVolume->GetBrickSizeWithOverlap();
	uint channelCount = m_pVolume->GetChannelCount();
	uint channelCountTex = (channelCount == 3) ? 4 : channelCount;

	//@Behdad: We do not use compression any more!
	// allocate channel buffers for decompression 
	size_t brickSizeBytePerChannel = brickSize * brickSize * brickSize * sizeof(float);
	
	m_dpChannelBuffer.resize(channelCount);
	m_pChannelBufferCPU.resize(channelCount);
	for (size_t channel = 0; channel < m_dpChannelBuffer.size(); channel++)
	{
		cudaSafeCall(cudaMalloc2(&m_dpChannelBuffer[channel], brickSizeBytePerChannel));
		cudaSafeCall(cudaMallocHost(&m_pChannelBufferCPU[channel], brickSizeBytePerChannel));
	}

	uint brickCount = m_pVolume->GetBrickCount().volume();

	// allocate brick requests structure
	cudaSafeCall(cudaMallocHost(&m_pBrickRequestCounts, brickCount * sizeof(uint)));
	memset(m_pBrickRequestCounts, 0, brickCount * sizeof(uint));
	cudaSafeCall(cudaMallocHost(&m_pBrickTimestepMins, brickCount * sizeof(uint)));
	//uint timestepMinInit = LineModeIsTimeDependent(m_traceParams.m_lineMode) ? ~0u : GetLineFloorTimestepIndex(GetLineSpawnTime());
	uint timestepMinInit = minTimestepIndex;
	for (uint i = 0; i < brickCount; i++)
	{
		m_pBrickTimestepMins[i] = timestepMinInit;
	}

	m_brickRequestsGPU->Allocate(cpuTracing, brickCount);
	m_brickRequestsGPU->Clear(cpuTracing, brickCount);


	// allocate brick slots
	size_t memFree = 0;
	size_t memTotal = 0;
	cudaSafeCall(cudaMemGetInfo(&memFree, &memTotal));

	std::cout << "\tAvailable: " << float(memFree) / (1024.0f * 1024.0f) << "MB" << std::endl;

	size_t memPerTimeSlot = brickSize * brickSize * brickSize * channelCountTex * sizeof(float);
	std::cout << "TraceableVolume::memPerTimeSlot1: " << memPerTimeSlot << std::endl;
	memPerTimeSlot = BrickTimeSizeInBytes(m_pVolume, 1);
	std::cout << "TraceableVolume::memPerTimeSlot2: " << memPerTimeSlot << std::endl;


	std::cout << "TraceableVolume::brickCount: " << m_pVolume->GetBrickCount().volume() << std::endl;

	// leave some wiggle room - arbitrarily chosen to be the size of a brick, or at least min(128 MB, 0.1 * totalMemory)
	// with large bricks, CUDA tends to start paging otherwise...
	//size_t memBuffer = max(memPerTimeSlot, min(size_t(32) * 1024 * 1024, memTotal / 100));
	//size_t memAvailable = memFree - min(memBuffer, memFree);
	//float memAvailableMB = float(memAvailable) / (1024.0f * 1024.0f);

	size_t memAvailable = 1024ll * 1024ll * (size_t)gpuMemUsageLimitMB;
	float memAvailableMB = gpuMemUsageLimitMB;

	// calculate max number of time slots
	uint timeSlotCountMax = min(devProp.maxTexture3D[0] / brickSize, m_timeSlotCountMax);
	timeSlotCountMax = min(max(1, uint(memAvailable / memPerTimeSlot)), timeSlotCountMax);

	uint timeSlotCount = timeDependent ? timeSlotCountMax : 1;
	//uint timeSlotCount = timeSlotCountMax;

	// calculate max number of brick slots based on the available memory
	size_t memPerBrickSlot = brickSize * brickSize * brickSize * channelCountTex * timeSlotCount * sizeof(float);
	std::cout << "TraceableVolume::memPerBrickSlot1: " << memPerBrickSlot << std::endl;
	memPerBrickSlot = BrickTimeSizeInBytes(m_pVolume, timeSlotCount);
	std::cout << "TraceableVolume::memPerBrickSlot2: " << memPerBrickSlot << std::endl;

	uint brickSlotCount1DMax = devProp.maxTexture3D[1] / brickSize;
	uint brickSlotCountMax = min(max(1, uint(memAvailable / memPerBrickSlot)), m_brickSlotCountMax);
	Vec2ui brickSlotCount = Fit2DSlotCount(brickSlotCountMax, brickSlotCount1DMax);
	m_brickSlotCount = Vec3ui(timeSlotCount, brickSlotCount);

	if (cpuTracing)
	{
		
		Vec3ui size = brickSize * m_brickSlotCount;
		g_volume.size = make_uint3(size);
		g_volume.data.resize(size.volume());
		printf("TraceableVolume::CreateVolumeDependentResources:\n\tAllocated %ux%u brick slot(s) (target %u) with %u time slot(s) each\n",
			m_brickSlotCount.y(), m_brickSlotCount.z(), brickSlotCountMax, m_brickSlotCount.x());
	}
	
	// GPU Tracing (This part should be fine, but before that we have messed up with GPU memory)
	else
	{
		
		m_brickAtlas = new BrickSlot();
		if (!m_brickAtlas->Create(brickSize, channelCount, m_brickSlotCount))
		{
			// out of memory
			//TODO retry with fewer brick/time slots
			printf("TraceableVolume::CreateVolumeDependentResources: Failed to create brick slots\n");
			ReleaseResources();
			return false;
		}
		printf("TraceableVolume::CreateVolumeDependentResources: %.2f MB available\n\tCreated %ux%u brick slot(s) (target %u) with %u time slot(s) each\n",
			memAvailableMB, m_brickAtlas->GetSlotCount().y(), m_brickAtlas->GetSlotCount().z(), brickSlotCountMax, m_brickAtlas->GetSlotCount().x());
	}


	m_brickSlotInfo.resize(m_brickSlotCount.yz().area());


	memFree = 0;
	memTotal = 0;
	cudaMemGetInfo(&memFree, &memTotal);
	std::cout << "cudaMallocHost: " << float(brickCount * sizeof(uint2)) / 1024.0f << "KB" << "\tAvailable: " << float(memFree) / (1024.0f * 1024.0f) << "MB" << std::endl;

	// allocate brick index *after* brick slots - need to know the slot count!
	cudaSafeCall(cudaMallocHost(&m_pBrickToSlot, brickCount * sizeof(uint2)));
	for (uint i = 0; i < brickCount; i++)
	{
		m_pBrickToSlot[i].x = BrickIndexGPU::INVALID;
		m_pBrickToSlot[i].y = BrickIndexGPU::INVALID;
	}

	cudaMemGetInfo(&memFree, &memTotal);
	std::cout << "cudaMallocHost: " << float(brickSlotCount.area() * sizeof(uint) * 2) / 1024.0f << "KB" << "\tAvailable: " << float(memFree) / (1024.0f * 1024.0f) << "MB" << std::endl;

	uint brickSlotCountArea = brickSlotCount.area();
	cudaSafeCall(cudaMallocHost(&m_pSlotTimestepMin, brickSlotCountArea * sizeof(uint)));
	cudaSafeCall(cudaMallocHost(&m_pSlotTimestepMax, brickSlotCountArea * sizeof(uint)));
	for (uint i = 0; i < brickSlotCountArea; i++) // TODO: brickCount
	{
		m_pSlotTimestepMin[i] = BrickIndexGPU::INVALID;
		m_pSlotTimestepMax[i] = BrickIndexGPU::INVALID;
	}
	m_brickIndexGPU->Allocate(cpuTracing, brickCount, make_uint2(brickSlotCount));
	m_brickIndexGPU->Update(cpuTracing, m_pBrickToSlot, m_pSlotTimestepMin, m_pSlotTimestepMax);
	cudaSafeCall(cudaEventRecord(m_brickIndexUploadEvent));
	
	m_isCreated = true;

	return true;
}

void TraceableVolume::ReleaseResources()
{
	if (m_brickRequestsDownloadEvent)
	{
		cudaSafeCall(cudaEventDestroy(m_brickRequestsDownloadEvent));
		m_brickRequestsDownloadEvent = 0;
	}

	if (m_brickIndexUploadEvent)
	{
		cudaSafeCall(cudaEventDestroy(m_brickIndexUploadEvent));
		m_brickIndexUploadEvent = 0;
	}

	if (m_brickRequestsGPU)
	{
		cudaSafeCall(cudaHostUnregister(&m_brickRequestsGPU));
		m_brickRequestsGPU->Deallocate();
		delete m_brickRequestsGPU;
		m_brickRequestsGPU = nullptr;
	}

	if (m_brickIndexGPU)
	{
		cudaSafeCall(cudaHostUnregister(&m_brickIndexGPU));
		m_brickIndexGPU->Deallocate();
		delete m_brickIndexGPU;
		m_brickIndexGPU = nullptr;
	}
	
	if (m_volumeInfoGPU)
	{
		cudaSafeCall(cudaHostUnregister(&m_volumeInfoGPU));
		delete m_volumeInfoGPU;
		m_volumeInfoGPU = nullptr;
	}

	if (m_brickAtlas)
	{
		m_brickAtlas->Release();
		delete m_brickAtlas;
		m_brickAtlas = nullptr;
	}

	if (m_pCompressVolume)
	{
		m_pCompressVolume->destroy();
		delete m_pCompressVolume;
		m_pCompressVolume = nullptr;
	}

	if (m_pCompressShared)
	{
		m_pCompressShared->destroy();
		delete m_pCompressShared;
		m_pCompressShared = nullptr;
	}

	if (m_pBrickTimestepMins)
	{
		cudaSafeCall(cudaFreeHost(m_pBrickTimestepMins));
		m_pBrickTimestepMins = nullptr;
	}
	if (m_pBrickRequestCounts)
	{
		cudaSafeCall(cudaFreeHost(m_pBrickRequestCounts));
		m_pBrickRequestCounts = nullptr;
	}

	if (m_pSlotTimestepMax)
	{
		cudaSafeCall(cudaFreeHost(m_pSlotTimestepMax));
		m_pSlotTimestepMax = nullptr;
	}

	if (m_pSlotTimestepMin)
	{
		cudaSafeCall(cudaFreeHost(m_pSlotTimestepMin));
		m_pSlotTimestepMin = nullptr;
	}

	if (m_pBrickToSlot)
	{
		cudaSafeCall(cudaFreeHost(m_pBrickToSlot));
		m_pBrickToSlot = nullptr;
	}

	for (size_t channel = 0; channel < m_dpChannelBuffer.size(); channel++)
	{
		cudaSafeCall(cudaFree(m_dpChannelBuffer[channel]));
		cudaSafeCall(cudaFreeHost(m_pChannelBufferCPU[channel]));
	}
	
	m_dpChannelBuffer.clear();
	m_pChannelBufferCPU.clear();

	m_timerUploadDecompress.ReleaseTimers();

	m_brickSlotInfo.clear();
	m_brickSlotCount.set(0, 0, 0);

	g_volume.data.clear();
	g_volume.data.shrink_to_fit();
	g_volume.size = make_uint3(0, 0, 0);

	m_bricksOnGPU.clear();
	m_bricksToLoad.clear();
	m_bricksToDo.clear();
	m_bricksLoaded.clear();

	m_bricksUploadedTotal = 0;

	m_bricksToDoPrioritiesDirty = true;
	m_bricksToDoDirty = true;

	m_isCreated = false;
}

bool TraceableVolume::IsCreated()
{
	return m_isCreated;
}

bool TraceableVolume::TryToKickBricksFromGPU(bool timeDependent, bool cpuTracing, int timeStepMax)
{
	// check if there's still anything to do with the bricks we have on the GPU
	bool anythingToDo = false;
	// in time-dependent case, provide at least 3 consecutive timesteps [n,n+2] to properly handle edge cases..
	//uint timestepInterpolation = LineModeIsTimeDependent(m_traceParams.m_lineMode) ? 2 : 0;
	uint timestepInterpolation = timeDependent ? 2 : 0;
	if (!cpuTracing)
		cudaSafeCall(cudaEventSynchronize(m_brickRequestsDownloadEvent));
	for (size_t i = 0; i < m_brickSlotInfo.size(); i++)
	{
		uint brickIndex = m_brickSlotInfo[i].brickIndex;
		uint timestepAvailableMin = m_brickSlotInfo[i].timestepFirst;
		uint timestepAvailableMax = timestepAvailableMin + m_brickSlotInfo[i].timestepCount - 1;
		if (brickIndex != ~0u)
		{
			uint requestCount = m_pBrickRequestCounts[brickIndex];
			uint timestepRequiredMin = m_pBrickTimestepMins[brickIndex];
			uint timestepRequiredMax = min(m_pBrickTimestepMins[brickIndex] + timestepInterpolation, timeStepMax);
			if (requestCount > 0 && timestepAvailableMin <= timestepRequiredMin && timestepRequiredMax <= timestepAvailableMax)
			{
				anythingToDo = true;
			}
		}
	}

	// if all bricks on the GPU are done, kick out all the finished bricks
	//if (!m_pVolume->HasThingsToDo(LineModeIsTimeDependent(m_traceParams.m_lineMode), m_traceParams.m_cpuTracing, m_timestepMax))
	if (!anythingToDo)
	{
		if (m_verbose)
		{
			printf("TraceableVolume::Trace: Clearing all brick slots\n");
		}
		bool reallyKick = false; // if false, only clear slot timestamps to avoid waiting for purge timeout
		if (reallyKick)
		{
			m_bricksOnGPU.clear();
			cudaSafeCall(cudaEventSynchronize(m_brickIndexUploadEvent));
		}
		for (size_t slotIndex = 0; slotIndex < m_brickSlotInfo.size(); slotIndex++)
		{
			SlotInfo& slot = m_brickSlotInfo[slotIndex];
			if (reallyKick)
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
		return true;
	}

	return false;
}

void TraceableVolume::ClearRequestsAndIndices(bool cpuTracing)
{
	uint brickCount = m_pVolume->GetBrickCount().volume();
	uint slotCount = (uint)m_brickSlotInfo.size();

	// clear all brick requests
	m_brickRequestsGPU->Clear(cpuTracing, brickCount);
	// update brick index. TODO only if something changed
	m_brickIndexGPU->Update(cpuTracing, m_pBrickToSlot, m_pSlotTimestepMin, m_pSlotTimestepMax);
	cudaSafeCall(cudaEventRecord(m_brickIndexUploadEvent));
}

void TraceableVolume::DownloadRequests(bool cpuTracing, bool timeDependent)
{
	uint brickCount = m_pVolume->GetBrickCount().volume();

	// start download of brick requests. for stream lines, do *not* download timestep indices (not filled by the kernel!)
	uint* pBrickTimestepMins = timeDependent ? m_pBrickTimestepMins : nullptr;
	m_brickRequestsGPU->Download(cpuTracing, m_pBrickRequestCounts, pBrickTimestepMins, brickCount);

	if (!cpuTracing)
		cudaSafeCall(cudaEventRecord(m_brickRequestsDownloadEvent));

	m_bricksToDoDirty = true;
}

std::vector<uint> TraceableVolume::GetBrickNeighbors(uint linearIndex) const
{
	std::vector<uint> result;

	uint neighbors[6];
	GetBrickNeighbors(linearIndex, neighbors);
	for (uint i = 0; i < 6; i++)
	{
		if (neighbors[i] != uint(-1))
		{
			result.push_back(neighbors[i]);
		}
	}

	return result;
}

void TraceableVolume::GetBrickNeighbors(uint linearIndex, uint neighborIndices[6]) const
{
	Vec3i brickCount = m_pVolume->GetBrickCount();
	Vec3i spatialIndex = m_pVolume->GetBrickSpatialIndex(linearIndex);

	auto left = spatialIndex.compEqual(Vec3i(0, 0, 0));
	auto right = spatialIndex.compEqual(brickCount - 1);

	neighborIndices[0] = left[0] ? uint(-1) : m_pVolume->GetBrickLinearIndex(SpatialNeighbor(spatialIndex, 0, -1));
	neighborIndices[1] = left[1] ? uint(-1) : m_pVolume->GetBrickLinearIndex(SpatialNeighbor(spatialIndex, 1, -1));
	neighborIndices[2] = left[2] ? uint(-1) : m_pVolume->GetBrickLinearIndex(SpatialNeighbor(spatialIndex, 2, -1));
	neighborIndices[3] = right[0] ? uint(-1) : m_pVolume->GetBrickLinearIndex(SpatialNeighbor(spatialIndex, 0, 1));
	neighborIndices[4] = right[1] ? uint(-1) : m_pVolume->GetBrickLinearIndex(SpatialNeighbor(spatialIndex, 1, 1));
	neighborIndices[5] = right[2] ? uint(-1) : m_pVolume->GetBrickLinearIndex(SpatialNeighbor(spatialIndex, 2, 1));
}

bool TraceableVolume::BrickIsLoaded(uint linearIndex, int timestepFrom, int timestepTo) const
{
	if (!m_pVolume) return false;

	bool loaded = true;
	for (int timestep = timestepFrom; timestep <= timestepTo; timestep++)
	{
		if (!m_pVolume->GetBrick(timestep, linearIndex).IsLoaded())
		{
			loaded = false;
			break;
		}
	}

	return loaded;
}

bool TraceableVolume::BrickIsOnGPU(uint linearIndex, int timestepFrom, int timestepTo) const
{
	assert(m_pVolume != nullptr);

	auto it = m_bricksOnGPU.find(linearIndex);
	if (it == m_bricksOnGPU.end())
	{
		// brick isn't on the GPU at all
		return false;
	}

	uint slotIndex = it->second;
	const SlotInfo& slotInfo = m_brickSlotInfo[slotIndex];
	assert(slotInfo.brickIndex == linearIndex);

	// check if all required timesteps are there
	int timestepAvailableFirst = slotInfo.timestepFirst;
	int timestepAvailableLast = slotInfo.timestepFirst + slotInfo.timestepCount - 1;
	return (timestepAvailableFirst <= timestepFrom) && (timestepAvailableLast >= timestepTo);
}

int TraceableVolume::FindAvailableBrickSlot(uint purgeTimeoutInRounds, bool forcePurgeFinished, bool timeDependent, int currentTimestamp) const
{
	// in time-dependent case, provide at least 3 consecutive timesteps [n,n+2] to properly handle edge cases..
	//uint timestepInterpolation = LineModeIsTimeDependent(m_traceParams.m_lineMode) ? 2 : 0;
	uint timestepInterpolation = timeDependent ? 2 : 0;

	int oldestFinishedSlotIndex = -1;
	int oldestFinishedTimestamp = std::numeric_limits<int>::max();
	for (size_t i = 0; i < m_brickSlotInfo.size(); i++)
	{
		const SlotInfo& info = m_brickSlotInfo[i];

		// if slot isn't filled at all, use this one immediately
		if (!info.IsFilled()) return int(i);

		uint timestepAvailableMin = info.timestepFirst;
		uint timestepAvailableMax = info.timestepFirst + info.timestepCount - 1;

		uint requestCount = m_pBrickRequestCounts[info.brickIndex];
		uint timestepRequiredMin = m_pBrickTimestepMins[info.brickIndex];
		uint timestepRequiredMax = timestepRequiredMin + timestepInterpolation;

		bool brickIsFinished = (requestCount == 0) || (timestepAvailableMin > timestepRequiredMin) || (timestepRequiredMax > timestepAvailableMax);
		//bool purgeTimeoutPassed = (info.lastUseTimestamp + int(m_traceParams.m_purgeTimeoutInRounds) <= m_currentTimestamp);
		//bool purgeTimeoutPassed = (info.lastUseTimestamp + int(purgeTimeoutInRounds) <= m_currentTimestamp);
		bool purgeTimeoutPassed = (info.lastUseTimestamp + int(purgeTimeoutInRounds) <= currentTimestamp);
		bool canPurge = brickIsFinished && (purgeTimeoutPassed || forcePurgeFinished);
		if (canPurge && info.lastUseTimestamp < oldestFinishedTimestamp)
		{
			oldestFinishedSlotIndex = int(i);
			oldestFinishedTimestamp = info.lastUseTimestamp;
		}
	}

	return oldestFinishedSlotIndex;
}

bool TraceableVolume::UploadBricks(bool cpuTracing, bool waitForDisk, uint& uploadBudget, bool forcePurgeFinished, int currentTimestamp, int timestepMax, bool timeDependent, uint purgeTimeoutInRounds, const TraceableVolumeHeuristics& heuristics)
{
	uint uploadedCount;
	return UploadBricks(cpuTracing, waitForDisk, uploadBudget, forcePurgeFinished, uploadedCount, currentTimestamp, timestepMax, timeDependent, purgeTimeoutInRounds, heuristics);
}

bool TraceableVolume::UploadBricks(bool cpuTracing, bool waitForDisk, uint& uploadBudget, bool forcePurgeFinished, uint& uploadedCount, int currentTimestamp, int timestepMax, bool timeDependent, uint purgeTimeoutInRounds, const TraceableVolumeHeuristics& heuristics)
{
	if (m_verbose)
	{
		printf("TraceableVolume::UploadBricks (budget %u forcePurge %i)\n", uploadBudget, int(forcePurgeFinished));
	}

	uploadedCount = 0;

	uint timeSlotCount = m_brickSlotCount.x();

	while (uploadBudget > 0)
	{
		// first update brick priorities
		UpdateBricksToDo(cpuTracing, heuristics);

		// find a brick that needs to be uploaded
		bool finished = true;
		for (auto itNextBrick = m_bricksToDo.begin(); itNextBrick != m_bricksToDo.end(); ++itNextBrick)
		{
			uint linearBrickIndex = itNextBrick->m_linearBrickIndex;
			uint timestepFrom = itNextBrick->m_timestepFrom;
			//uint timestepTo = min(itNextBrick->m_timestepFrom + timeSlotCount - 1, m_timestepMax);
			uint timestepTo = min(itNextBrick->m_timestepFrom + timeSlotCount - 1, timestepMax);

			if (BrickIsOnGPU(linearBrickIndex, timestepFrom, timestepTo))
			{
				assert(m_bricksOnGPU.count(linearBrickIndex) > 0);
				// update slot's lastUseTimestamp
				uint slotIndex = m_bricksOnGPU[linearBrickIndex];
				//m_brickSlotInfo[slotIndex].lastUseTimestamp = m_currentTimestamp;
				m_brickSlotInfo[slotIndex].lastUseTimestamp = currentTimestamp;
				continue;
			}

			bool brickIsLoaded = BrickIsLoaded(linearBrickIndex, timestepFrom, timestepTo);

			//if (!brickIsLoaded && m_traceParams.m_waitForDisk)
			if (!brickIsLoaded && waitForDisk)
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
					slotIndex = FindAvailableBrickSlot(purgeTimeoutInRounds, forcePurgeFinished, timeDependent, currentTimestamp);
				}
				// if there is no available slot, we're good for now
				if (slotIndex == -1)
				{
					return true;
				}

				if (m_verbose)
				{
					printf("TraceableVolume::UploadBricks: uploading brick %u prio %.2f\n", linearBrickIndex, itNextBrick->m_priority);
				}
				if (!UpdateBrickSlot(cpuTracing, uploadBudget, slotIndex, linearBrickIndex, timestepFrom, timestepTo, currentTimestamp))
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

		if (finished)
		{
			// no more bricks to process - we're good
			return true;
		}
	}

	// upload budget ran out
	return false;
}

bool TraceableVolume::UpdateBrickSlot(bool cpuTracing, uint& uploadBudget, uint brickSlotIndex, uint brickLinearIndex, int timestepFrom, int timestepTo, int currentTimestamp)
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
	if (brickOnGPU)
	{
		// if some (but not all) required time steps are there already, shift them to the correct time slots
		int timeSlotShift = slot.timestepFirst - timestepFrom;
		if (uint(abs(timeSlotShift)) < timeSlotCount)
		{
			if (timeSlotShift < 0)
			{
				//assert(!m_traceParams.m_cpuTracing);
				assert(!cpuTracing);
				// move existing contents to the left
				timeSlotFirstToUpload = min(timeSlotCount + timeSlotShift, timeSlotLastToUpload + 1);
				for (int timeSlot = 0; timeSlot < timeSlotFirstToUpload; timeSlot++)
				{
					m_brickAtlas->CopySlot(Vec3ui(timeSlot - timeSlotShift, brickSlotIndexGPU), Vec3ui(timeSlot, brickSlotIndexGPU));
				}
			}
			else if (timeSlotShift > 0)
			{
				//assert(!m_traceParams.m_cpuTracing);
				assert(!cpuTracing);
				// move existing contents to the right
				int timeSlotMax = timeSlotLastToUpload;
				timeSlotLastToUpload = min(timeSlotMax, max(timeSlotShift - 1, timeSlotFirstToUpload - 1));
				for (int timeSlot = timeSlotLastToUpload + 1; timeSlot < timeSlotMax; timeSlot++)
				{
					m_brickAtlas->CopySlot(Vec3ui(timeSlot - timeSlotShift, brickSlotIndexGPU), Vec3ui(timeSlot, brickSlotIndexGPU));
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
		if (slot.brickIndex != ~0u)
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
	if (timeSlotLastToUpload - timeSlotFirstToUpload + 1 > (int)uploadBudget)
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


	if (m_verbose)
	{
		printf("TraceableVolume::UpdateBrickSlot: slot %ux%u brick %u (%i,%i,%i) ts %i-%i (upload %i)\n",
			brickSlotIndexGPU.x(), brickSlotIndexGPU.y(),
			brickLinearIndex, brickSpatialIndex.x(), brickSpatialIndex.y(), brickSpatialIndex.z(),
			timestepFrom, timestepTo,
			timeSlotLastToUpload + 1 - timeSlotFirstToUpload);
	}

	// upload new brick data
	uint brickCount = m_pVolume->GetBrickCount().volume();
	for (int timeSlot = timeSlotFirstToUpload; timeSlot <= timeSlotLastToUpload; timeSlot++)
	{
		int timestep = timestepFrom + timeSlot;
		const TimeVolumeIO::Brick& brick = m_pVolume->GetBrick(timestep, brickLinearIndex);
		assert(brick.IsLoaded());

		//cudaSafeCall(cudaDeviceSynchronize());
		Vec3ui slotIndex(timeSlot, brickSlotIndexGPU);
		//if (m_traceParams.m_cpuTracing)
		if (cpuTracing)
		{
			UploadBrick(m_pCompressShared, m_pCompressVolume, m_pVolume->GetInfo(), brick, m_dpChannelBuffer.data(), nullptr, slotIndex, &m_timerUploadDecompress);

			// download to CPU and copy into interleaved g_volume
			Vec3ui brickSize = brick.GetSize();
			size_t brickSizeBytePerChannel = brickSize.volume() * sizeof(float);
			int channelCount = m_pVolume->GetInfo().GetChannelCount();
			for (int c = 0; c < channelCount; c++)
			{
				cudaSafeCall(cudaMemcpy(m_pChannelBufferCPU[c], m_dpChannelBuffer[c], brickSizeBytePerChannel, cudaMemcpyDeviceToHost));
			}

			uint brickSizeFull = m_pVolume->GetInfo().GetBrickSizeWithOverlap();
			Vec3ui atlasSize = m_brickSlotCount * brickSizeFull;
			Vec3ui offset = slotIndex * brickSizeFull;
			for (uint z = 0; z < brickSize.z(); z++)
			{
				for (uint y = 0; y < brickSize.y(); y++)
				{
					for (uint x = 0; x < brickSize.x(); x++)
					{
						uint indexSrc = x + brickSize.x() * (y + brickSize.y() * z);
						uint indexDst = (offset.x() + x) + atlasSize.x() * ((offset.y() + y) + atlasSize.y() * (offset.z() + z));
						// HACK: hardcoded for 4 channels...
						assert(channelCount == 4);
						g_volume.data[indexDst] = make_float4(m_pChannelBufferCPU[0][indexSrc], m_pChannelBufferCPU[1][indexSrc], m_pChannelBufferCPU[2][indexSrc], m_pChannelBufferCPU[3][indexSrc]);
					}
				}
			}
		}
		else
		{
			UploadBrick(m_pCompressShared, m_pCompressVolume, m_pVolume->GetInfo(), brick, m_dpChannelBuffer.data(), m_brickAtlas, slotIndex, &m_timerUploadDecompress);
		}
		//cudaSafeCall(cudaDeviceSynchronize());

		m_bricksUploadedTotal++;
		m_bricksLoaded.insert(timestep * brickCount + brickLinearIndex);
	}

	// record new brick as being on the GPU
	slot.brickIndex = brickLinearIndex;
	slot.timestepFirst = timestepFrom;
	slot.timestepCount = timestepTo - timestepFrom + 1;
	//slot.lastUseTimestamp = m_currentTimestamp;
	slot.lastUseTimestamp = currentTimestamp;
	m_bricksOnGPU[brickLinearIndex] = brickSlotIndex;
	// update brick index
	cudaSafeCall(cudaEventSynchronize(m_brickIndexUploadEvent));
	// de-linearize slot index - they're stacked in y and z
	m_pBrickToSlot[brickLinearIndex] = make_uint2(brickSlotIndexGPU);
	m_pSlotTimestepMin[brickSlotIndex] = timestepFrom;
	m_pSlotTimestepMax[brickSlotIndex] = timestepTo;

	if (m_verbose)
	{
		printf("TraceableVolume::UpdateBrickSlot: gpu slot %d assigned to a brick\n",
			brickLinearIndex);
	}

	m_bricksToDoPrioritiesDirty = true;

	return result;
}

bool TraceableVolume::UploadWholeTimestep(bool cpuTracing, bool waitForDisk, int timestep, bool forcePurgeFinished, int currentTimestamp, bool timeDependent, uint purgeTimeoutInRounds)
{
	m_bricksToLoad.clear();

	//This is more or less a modified version of UploadBricks

	uint uploadBudget = 8;

	if (m_verbose)
		printf("TraceableVolume::UploadWholeTimestep (budget %u)\n", uploadBudget);

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
			//m_brickSlotInfo[slotIndex].lastUseTimestamp = m_currentTimestamp;
			m_brickSlotInfo[slotIndex].lastUseTimestamp = currentTimestamp;
			continue;
		}

		bool brickIsLoaded = BrickIsLoaded(linearBrickIndex, timestepFrom, timestepTo);

		if (!brickIsLoaded) {
			//add to loading queue
			m_bricksToLoad.push_back(&m_pVolume->GetBrick(timestep, linearBrickIndex));
			if (m_verbose)
			{
				printf("TraceableVolume::UploadWholeTimestep: brick %u is not loaded\n", linearBrickIndex);
			}
		}

		//if (!brickIsLoaded && m_traceParams.m_waitForDisk)
		if (!brickIsLoaded && waitForDisk)
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
				slotIndex = FindAvailableBrickSlot(purgeTimeoutInRounds, forcePurgeFinished, timeDependent, currentTimestamp);
			}
			// if there is no available slot, we're good for now
			if (slotIndex == -1)
			{
				printf("TraceableVolume::UploadWholeTimestep: no available slot found!!\n");
				break;
			}

			if (m_verbose)
			{
				printf("TraceableVolume::UploadWholeTimestep: uploading brick %u\n", linearBrickIndex);
			}
			if (!UpdateBrickSlot(cpuTracing, uploadBudget, slotIndex, linearBrickIndex, timestepFrom, timestepTo, currentTimestamp))
			{
				// budget ran out
				return false;
			}
			if (uploadBudget == 0) {
				return false;
			}
		}
	}

	if (finished) 
	{
		// update brick index
		//m_brickIndexGPU.Update(m_traceParams.m_cpuTracing, m_pBrickToSlot, m_pSlotTimestepMin, m_pSlotTimestepMax);
		m_brickIndexGPU->Update(cpuTracing, m_pBrickToSlot, m_pSlotTimestepMin, m_pSlotTimestepMax);
		cudaSafeCall(cudaEventRecord(m_brickIndexUploadEvent));
		printf("TraceableVolume::UploadWholeTimestep: update brick index\n");
	}

	return finished;
}

void TraceableVolume::UpdateBricksToLoad(bool cpuTracing, bool enablePrefetching, bool timeDependent, int floorTimestepIndex, int maxTimestep, const TraceableVolumeHeuristics& heuristics)
{
	m_bricksToLoad.clear();

	if (m_pVolume == nullptr) return;

	// collect all bricks with lines to do
	UpdateBricksToDo(cpuTracing, heuristics);

	// ..and add to bricks to load
	uint timeSlotCount = m_brickSlotCount.x();
	for (auto it = m_bricksToDo.cbegin(); it != m_bricksToDo.cend(); it++)
	{
		int timestepFrom = it->m_timestepFrom;
		//int timestepTo = min(timestepFrom + timeSlotCount - 1, m_timestepMax);
		int timestepTo = min(timestepFrom + timeSlotCount - 1, maxTimestep);
		for (int timestep = timestepFrom; timestep <= timestepTo; timestep++)
		{
			m_bricksToLoad.push_back(&m_pVolume->GetBrick(timestep, it->m_linearBrickIndex));
		}
	}

	// prefetching
	//if (m_traceParams.m_enablePrefetching)
	if (enablePrefetching)
	{
		//if (LineModeIsTimeDependent(m_traceParams.m_lineMode))
		if (timeDependent)
		{
			// for path lines, add later timesteps of bricks in todo list
			for (auto it = m_bricksToDo.cbegin(); it != m_bricksToDo.cend(); it++)
			{
				int timestepFrom = it->m_timestepFrom + timeSlotCount;
				//int timestepTo = min(timestepFrom + timeSlotCount - 1, m_timestepMax);
				int timestepTo = min(timestepFrom + timeSlotCount - 1, maxTimestep);
				for (int timestep = timestepFrom; timestep <= timestepTo; timestep++)
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
			for (auto it = m_bricksToDo.cbegin(); it != m_bricksToDo.cend(); it++)
			{
				indicesDone.insert(it->m_linearBrickIndex);
			}

			//int timestep = GetLineFloorTimestepIndex(0.0f);
			int timestep = floorTimestepIndex;
			for (auto it = m_bricksToDo.cbegin(); it != m_bricksToDo.cend(); it++)
			{
				std::vector<uint> neighbors = GetBrickNeighbors(it->m_linearBrickIndex);
				for (size_t i = 0; i < neighbors.size(); ++i)
				{
					uint neighbor = neighbors[i];
					if (indicesDone.count(neighbor) == 0)
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

void TraceableVolume::UpdateBricksToDo(bool cpuTracing, const TraceableVolumeHeuristics& heuristics)
{
	bool resort = false;

	if (m_bricksToDoDirty)
	{
		if (m_verbose)
		{
			printf("TraceableVolume::UpdateBricksToDo: collecting bricks\n");
		}

		// collect all bricks with lines to do
		uint brickCount = m_pVolume->GetBrickCount().volume();
		m_bricksToDo.clear();
		//if (!m_traceParams.m_cpuTracing)
		if (!cpuTracing)
			cudaSafeCall(cudaEventSynchronize(m_brickRequestsDownloadEvent));
		//if(m_verbose)
		//{
		//	printf("TraceableVolume::UpdateBricksToDo: request counts");
		//}
		for (uint brickIndex = 0; brickIndex < brickCount; brickIndex++)
		{
			uint requestCount = m_pBrickRequestCounts[brickIndex];
			//if(m_verbose)
			//{
			//	printf(" %u", requestCount);
			//}
			if (requestCount != 0)
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

	if (m_bricksToDoPrioritiesDirty)
	{
		if (m_verbose)
		{
			printf("TraceableVolume::UpdateBricksToDo: updating priorities\n");
		}

		//if (!m_traceParams.m_cpuTracing)
		if (!cpuTracing)
			cudaSafeCall(cudaEventSynchronize(m_brickRequestsDownloadEvent));
		for (auto it = m_bricksToDo.begin(); it != m_bricksToDo.end(); ++it)
		{
			BrickSortItem& brick = *it;
			uint brickParticleCount = m_pBrickRequestCounts[brick.m_linearBrickIndex];
			float brickParticleFactor = float(brickParticleCount);
			//if (m_traceParams.HeuristicDoSqrtBPCount())
			if (heuristics.doSqrtBPCount)
				brickParticleFactor = sqrtf(brickParticleFactor);

			uint neighbors[6];
			GetBrickNeighbors(brick.m_linearBrickIndex, neighbors);
			float bonus = 0.0f;
			float penalty = 0.0f;
			for (int dir = 0; dir < 6; ++dir)
			{
				uint neighbor = neighbors[dir];
				// skip neighbors that don't exist (at the border of the domain)
				if (neighbor == uint(-1)) continue;

				uint neighborParticleCount = m_pBrickRequestCounts[neighbor];
				float neighborParticleFactor = float(neighborParticleCount);
				//if (m_traceParams.HeuristicDoSqrtBPCount())
				if (heuristics.doSqrtBPCount)
					neighborParticleFactor = sqrtf(neighborParticleFactor);

				if (BrickIsOnGPU(neighbor, brick.m_timestepFrom, brick.m_timestepFrom) && m_brickSlotInfo[m_bricksOnGPU[neighbor]].HasValidTimestamp())
				{
					// going out
					float probabilityOut = 1.0f / 6.0f;
					//if (m_traceParams.HeuristicUseFlowGraph() && m_pFlowGraph->IsBuilt())
					if (heuristics.useFlowGraph && heuristics.flowGraph->IsBuilt())
					{
						const FlowGraph::BrickInfo& info = heuristics.flowGraph->GetBrickInfo(brick.m_linearBrickIndex);
						probabilityOut = info.OutFreq[dir];
					}
					//if (m_traceParams.HeuristicDoSqrtBPProb())
					if (heuristics.doSqrtBPProb)
					{
						probabilityOut = sqrtf(probabilityOut);
					}
					bonus += probabilityOut * brickParticleFactor;

					// coming in
					float probabilityIn = 1.0f / 6.0f;
					//if (m_traceParams.HeuristicUseFlowGraph() && m_pFlowGraph->IsBuilt())
					if (heuristics.useFlowGraph && heuristics.flowGraph->IsBuilt())
					{
						const FlowGraph::BrickInfo& info = heuristics.flowGraph->GetBrickInfo(neighbor);
						int n = (dir + 3) % 6; // flip sign of direction (DIR_POS_X <-> DIR_NEG_X etc)
						probabilityIn = info.OutFreq[n];
					}
					//if (m_traceParams.HeuristicDoSqrtBPProb())
					if (heuristics.doSqrtBPProb)
					{
						probabilityIn = sqrtf(probabilityIn);
					}
					bonus += probabilityIn * neighborParticleFactor;
				}
				else
				{
					// coming in
					float probabilityIn = 1.0f / 6.0f;
					//if (m_traceParams.HeuristicUseFlowGraph() && m_pFlowGraph->IsBuilt())
					if (heuristics.useFlowGraph && heuristics.flowGraph->IsBuilt())
					{
						const FlowGraph::BrickInfo& info = heuristics.flowGraph->GetBrickInfo(neighbor);
						int n = (dir + 3) % 6; // flip sign of direction (DIR_POS_X <-> DIR_NEG_X etc)
						probabilityIn = info.OutFreq[n];
					}
					//if (m_traceParams.HeuristicDoSqrtBPProb())
					if (heuristics.doSqrtBPProb)
					{
						probabilityIn = sqrtf(probabilityIn);
					}
					penalty += probabilityIn * neighborParticleFactor;
				}
			}

			// base priority: number of particles in this brick
			brick.m_priority = brickParticleFactor;
			
			// modify priority according to heuristic
			//brick.m_priority += m_traceParams.m_heuristicBonusFactor   * bonus;
			//brick.m_priority -= m_traceParams.m_heuristicPenaltyFactor * penalty;
			brick.m_priority += heuristics.bonusFactor   * bonus;
			brick.m_priority -= heuristics.penaltyFactor * penalty;
		}

		m_bricksToDoPrioritiesDirty = false;
		resort = true;
	}

	if (resort)
	{
		// sort bricks in todo list by priority
		std::sort(m_bricksToDo.begin(), m_bricksToDo.end());
	}
}
