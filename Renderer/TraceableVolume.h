#pragma once

#include <unordered_set>

#include <GPUResources.h>
#include <CompressVolume.h>
#include <TimeVolume.h>
#include <VolumeInfoGPU.h>
#include <BrickIndexGPU.h>
#include <BrickRequestsGPU.h>
#include <BrickSlot.h>
#include <FlowGraph.h>
#include <MultiTimerGPU.h>

struct TraceableVolumeHeuristics
{
	TraceableVolumeHeuristics(bool doSqrtBPCount, bool useFlowGraph, bool doSqrtBPProb, float bonusFactor, float penaltyFactor, const FlowGraph* flowGraph)
		: doSqrtBPCount(doSqrtBPCount), useFlowGraph(useFlowGraph), doSqrtBPProb(doSqrtBPProb), bonusFactor(bonusFactor), penaltyFactor(penaltyFactor), flowGraph(flowGraph)
	{
	}

	bool doSqrtBPCount;
	bool useFlowGraph;
	bool doSqrtBPProb;
	float bonusFactor;
	float penaltyFactor;
	const FlowGraph* flowGraph;
};

class TraceableVolume
{
public:



	//===========================================================================================================================
	struct SlotInfo
	{
		SlotInfo() { Clear(); }

		uint       brickIndex;
		uint       timestepFirst;
		uint       timestepCount;
		int        lastUseTimestamp;

		void Clear() { brickIndex = ~0u; timestepFirst = ~0u; timestepCount = 0; ClearTimestamp(); }
		void ClearTimestamp() { lastUseTimestamp = std::numeric_limits<int>::min(); }
		void InvalidateTimestamp() { if (lastUseTimestamp >= 0) lastUseTimestamp -= std::numeric_limits<int>::max(); else ClearTimestamp(); }
		bool HasValidTimestamp() const { return lastUseTimestamp >= 0; }
		bool IsFilled() const { return timestepCount > 0; }
	};

	struct BrickSortItem
	{
		BrickSortItem(uint linearBrickIndex, int timestepFrom, float priority)
			: m_linearBrickIndex(linearBrickIndex), m_timestepFrom(timestepFrom), m_priority(priority) {}

		// sort by priority: oldest required timestep first, break ties by precomputed priority
		bool operator<(const BrickSortItem& other) const
		{
			// oldest required timestep always has priority
			if (m_timestepFrom != other.m_timestepFrom) return (m_timestepFrom < other.m_timestepFrom);
			// otherwise, sort by stored priority
			if (m_priority != other.m_priority) return (m_priority > other.m_priority);
			// if still no difference, arbitrarily break ties by brick index
			return (m_linearBrickIndex < other.m_linearBrickIndex);
		}

		uint  m_linearBrickIndex;
		int   m_timestepFrom;
		float m_priority;
	};






	//===========================================================================================================================
	TraceableVolume(const TimeVolume* volume);
	~TraceableVolume();

	bool CreateResources(uint minTimestepIndex, bool cpuTracing, bool timeDependent, int gpuMemUsageLimitMB);
	void ReleaseResources();
	bool IsCreated();

	// How many bytes of a texture a single brick with 'timeSlotCount' time steps takes.
	static size_t BrickTimeSizeInBytes(const TimeVolume* volume, size_t timeSlotCount);
	// How many bytes of a texture a full time steps takes.
	static size_t TimeStepSizeInBytes(const TimeVolume* volume);

	//===========================================================================================================================
	std::vector<uint> GetBrickNeighbors(uint linearIndex) const;
	// get neighbors in -x,-y,-z,+x,+y,+z order; return uint(-1) if non-existing
	void GetBrickNeighbors(uint linearIndex, uint neighborIndices[6]) const;

	// Check if all specified timesteps of the given brick have been loaded from disk.
	bool BrickIsLoaded(uint linearIndex, int timestepFrom, int timestepTo) const;
	// Check if all specified timesteps of the given brick are available on the GPU.
	bool BrickIsOnGPU(uint linearIndex, int timestepFrom, int timestepTo) const;

	// Find a brick slot which is available for uploading a new brick.
	// If forcePurgeFinished is set, this includes any filled brick slot with no lines to do,
	// otherwise only empty brick slots or filled slots where the last use was more than purgeTimeoutInRounds ago.
	// Returns -1 if no appropriate slot is found.
	// Note: The download of brick requests must be finished when this is called!
	int FindAvailableBrickSlot(uint m_purgeTimeoutInRounds, bool forcePurgeFinished, bool timedependent, int currentTimestamp) const;

	// Upload bricks (from m_bricksToDo) into available slots.
	// Returns true if all available slots could be filled, false if we had to bail out
	// (if a brick isn't loaded from disk yet, or the upload budget ran out).
	bool UploadBricks(bool cpuTracing, bool waitForDisk, uint& uploadBudget, bool forcePurgeFinished, int currentTimestamp, int timestepMax, bool timedependent, uint purgeTimeoutInRounds, const TraceableVolumeHeuristics& heuristics);
	bool UploadBricks(bool cpuTracing, bool waitForDisk, uint& uploadBudget, bool forcePurgeFinished, uint& uploadedCount, int currentTimestamp, int timestepMax, bool timedependent, uint purgeTimeoutInRounds, const TraceableVolumeHeuristics& heuristics);
	// Uploads a whole timestep
	// Returns true if all bricks of the timestep fit into memory, false if we had to bail out
	// (if a brick isn't loaded from disk yet, or the upload budget ran out).
	bool UploadWholeTimestep(bool cpuTracing, bool waitForDisk, int timestep, bool forcePurgeFinished, int currentTimestamp, bool timedependent, uint purgeTimeoutInRounds);

	// Update data in the given brick slot: Upload the specified time steps as required, limited by the budget.
	// Returns true if all required time steps were uploaded, false if the budget ran out.
	bool UpdateBrickSlot(bool cpuTracing, uint& uploadBudget, uint brickSlotIndex, uint brickLinearIndex, int timestepFrom, int timestepTo, int currentTimestamp);
	void UpdateBricksToDo(bool cpuTracing, const TraceableVolumeHeuristics& heuristics);
	void UpdateBricksToLoad(bool cpuTracing, bool enablePrefetching, bool timedependent, int floorTimestepIndex, int maxTimestep, const TraceableVolumeHeuristics& heuristics);

	bool TryToKickBricksFromGPU(bool isTimeDependent, bool cpuTracing, int timeStepMax);
	bool HasThingsToDo() { return !m_bricksToDo.empty(); };

	void ClearRequestsAndIndices(bool cpuTracing);
	void DownloadRequests(bool cpuTracing, bool timeDependent);


	//===========================================================================================================================
	// "global" resource settings
	uint m_brickSlotCountMax;
	uint m_timeSlotCountMax;

	GPUResources*				m_pCompressShared;
	CompressVolumeResources*	m_pCompressVolume;

	const TimeVolume*	m_pVolume;
	VolumeInfoGPU*		m_volumeInfoGPU;
	BrickIndexGPU*		m_brickIndexGPU;
	BrickRequestsGPU*	m_brickRequestsGPU;
	BrickSlot*			m_brickAtlas;	// Size depends on maxTexture3D size, bricksize and available GPU memory.

	// channel buffers for decompression
	std::vector<float*>	m_dpChannelBuffer;
	std::vector<float*>	m_pChannelBufferCPU;

	// cudaMallocHost for uploading to BrickIndexGPU
	uint2*      m_pBrickToSlot;			// Size depends on brickcount
	uint*       m_pSlotTimestepMin;		// Size depends on maxTexture3D size, bricksize and available GPU memory.
	uint*       m_pSlotTimestepMax;		// Size depends on maxTexture3D size, bricksize and available GPU memory.

	// cudaMallocHost for downloading from BrickRequestsGPU
	uint*       m_pBrickRequestCounts;	// Size depends on brickcount
	uint*		m_pBrickTimestepMins;	// Size depends on brickcount. Only used if linemode is time dependent

	// Used to synchronize host and device on download of requests and upload of indexes.
	cudaEvent_t	m_brickRequestsDownloadEvent;
	cudaEvent_t m_brickIndexUploadEvent;

	
	tum3D::Vec3ui         m_brickSlotCount;	// X: timeSlotCount; Y,Z: brickSlotCount; Depends on maxTexture3D size, bricksize and available GPU memory.
	std::vector<SlotInfo> m_brickSlotInfo;	// Size: brickSlotCount.x * brickSlotCount.y
	std::map<uint, uint>  m_bricksOnGPU;	// map from brick index to slot index

	std::vector<BrickSortItem>	m_bricksToDo;
	bool						m_bricksToDoDirty;
	bool						m_bricksToDoPrioritiesDirty;

	std::vector<const TimeVolumeIO::Brick*> m_bricksToLoad;

	// Stats stuff
	MultiTimerGPU				m_timerUploadDecompress;
	TimerCPU					m_timerDiskWait;
	int							m_bricksUploadedTotal;
	std::unordered_set<uint>	m_bricksLoaded; // 4D linear brick index: timestep * brickCount + brickIndex

	bool m_verbose;

private:
	bool m_isCreated;

private:
	TraceableVolume() = delete;
	// disable copy and assignment
	TraceableVolume(const TraceableVolume&);
	TraceableVolume& operator=(const TraceableVolume&);
};