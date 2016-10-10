#ifndef __TUM3D__TRACINGMANAGER_H__
#define __TUM3D__TRACINGMANAGER_H__


#include <global.h>

#include <limits>
#include <memory>
#include <map>
#include <unordered_set>
#include <vector>

#include <cuda_runtime.h>
#include <cuda_d3d11_interop.h>
#include <D3D11.h>

#include <TimerCPU.h>
#include <MultiTimerGPU.h>
#include <Vec.h>

#include "cutil_math.h"

#include "BrickIndexGPU.h"
#include "BrickRequestsGPU.h"
#include "BrickSlot.h"
#include "FlowGraph.h"
#include "Integrator.h"
#include "LineBuffers.h"
#include "ParticleTraceParams.h"
#include "TimeVolume.h"
#include "TracingCommon.h"
#include "VolumeInfoGPU.h"

#include "GPUResources.h"
#include "CompressVolume.h"


class TracingManager
{
public:
	TracingManager();
	~TracingManager();

	bool Create(GPUResources* pCompressShared, CompressVolumeResources* pCompressVolume, ID3D11Device* pDevice);
	void Release();
	bool IsCreated() const { return m_isCreated; }

	// settings for resource limits. applied on the next StartTracing.
	uint GetBrickSlotCountMax() const { return m_brickSlotCountMax; }
	uint& GetBrickSlotCountMax() { return m_brickSlotCountMax; }
	void SetBrickSlotCountMax(uint brickSlotCountMax) { m_brickSlotCountMax = brickSlotCountMax; }
	uint GetTimeSlotCountMax() const { return m_timeSlotCountMax; }
	uint& GetTimeSlotCountMax() { return m_timeSlotCountMax; }
	void SetTimeSlotCountMax(uint timeSlotCountMax) { m_timeSlotCountMax = timeSlotCountMax; }

	bool& GetVerbose() { return m_verbose; }

	bool StartTracing(const TimeVolume& volume, const ParticleTraceParams& traceParams, const FlowGraph& flowGraph);
	bool Trace(); // returns true if finished TODO error code
	void CancelTracing();
	bool IsTracing() const;
	float GetTracingProgress() const;

	const std::vector<const TimeVolumeIO::Brick*>& GetBricksToLoad() const { return m_bricksToLoad; }

	void BuildIndexBuffer();
	bool NeedIndexBufferRebuild() { return m_indexBufferDirty; }

	// release memory-heavy resources (will be recreated on StartTracing)
	void ReleaseResources();

	std::shared_ptr<LineBuffers> GetResult() { return m_pResult; }
	void ClearResult();


	struct Timings
	{
		Timings() : IntegrateCPU(0.0f), WaitDiskWall(0.0f), TraceWall(0.0f) {}
		bool operator==(const Timings& other) { return memcmp(this, &other, sizeof(*this)) == 0; }
		bool operator!=(const Timings& other) { return !(*this == other); }

		bool WriteToFile(const std::string& filename) const;
		bool WriteToFile(std::ostream& file) const;

		static bool WriteCSVHeader(std::ostream& file, const std::vector<std::string>& extraColumns);
		bool WriteToCSVFile(std::ostream& file, const std::vector<std::string>& extraColumns) const;

		MultiTimerGPU::Stats UploadDecompressGPU;
		MultiTimerGPU::Stats IntegrateGPU;
		MultiTimerGPU::Stats BuildIndexBufferGPU;

		float                IntegrateCPU;

		float                WaitDiskWall;

		float                TraceWall;
	};
	const Timings& GetTimings() const { return m_timings; }

	struct Stats
	{
		Stats() { Clear(); }
		void Clear();

		bool WriteToFile(const std::string& filename) const;
		bool WriteToFile(std::ostream& file) const;

		static bool WriteCSVHeader(std::ostream& file, const std::vector<std::string>& extraColumns);
		bool WriteToCSVFile(std::ostream& file, const std::vector<std::string>& extraColumns) const;

		uint BricksUploadedTotal;
		uint BricksUploadedUnique;

		uint StepsTotal;
		uint StepsAccepted;
		uint Evaluations;

		uint LinesReachedMaxAge;
		uint LinesLeftDomain;

		float DiskBusyTimeMS;

		size_t DiskBytesLoaded;
		size_t DiskBytesUsed;
	};
	const Stats& GetStats() const { return m_stats; }

private:
	struct BrickSortItem
	{
		BrickSortItem(uint linearBrickIndex, int timestepFrom, float priority)
			: m_linearBrickIndex(linearBrickIndex), m_timestepFrom(timestepFrom), m_priority(priority) {}

		// sort by priority: oldest required timestep first, break ties by precomputed priority
		bool operator<(const BrickSortItem& other) const
		{
			// oldest required timestep always has priority
			if(m_timestepFrom != other.m_timestepFrom) return (m_timestepFrom < other.m_timestepFrom);
			// otherwise, sort by stored priority
			if(m_priority != other.m_priority) return (m_priority > other.m_priority);
			// if still no difference, arbitrarily break ties by brick index
			return (m_linearBrickIndex < other.m_linearBrickIndex);
		}

		uint  m_linearBrickIndex;
		int   m_timestepFrom;
		float m_priority;
	};


	HRESULT CreateVolumeDependentResources();
	void ReleaseVolumeDependentResources();

	HRESULT CreateParamDependentResources();
	void ReleaseParamDependentResources();

	HRESULT CreateResultResources();
	void ReleaseResultResources();


	void UpdateFrameBudgets();


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
	int FindAvailableBrickSlot(bool forcePurgeFinished) const;


	// Upload bricks (from m_bricksToDo) into available slots.
	// Returns true if all available slots could be filled, false if we had to bail out
	// (if a brick isn't loaded from disk yet, or the upload budget ran out).
	bool UploadBricks(uint& uploadBudget, bool forcePurgeFinished);
	bool UploadBricks(uint& uploadBudget, bool forcePurgeFinished, uint& uploadedCount);
	// Update data in the given brick slot: Upload the specified time steps as required, limited by the budget.
	// Returns true if all required time steps were uploaded, false if the budget ran out.
	bool UpdateBrickSlot(uint& uploadBudget, uint brickSlotIndex, uint brickLinearIndex, int timestepFrom, int timestepTo);
	// Perform one round of particle tracing, using the bricks which are currently on the GPU.
	void TraceRound();


	void UpdateBricksToDo();
	void UpdateBricksToLoad();

	void UpdateProgress();

	void ComputeTimings();
	void PrintTimings() const;
	void ComputeStats();
	void PrintStats() const;


	float GetLineSpawnTime() const;
	int GetLineFloorTimestepIndex(float lineTime) const;
	float GetLineTimeMax() const;


	// "global" resource settings
	uint m_brickSlotCountMax;
	uint m_timeSlotCountMax;


	// per-frame budgets
	uint m_uploadsPerFrame;
	uint m_roundsPerFrame;


	bool          m_isCreated;
	// valid between create/release
	GPUResources*            m_pCompressShared;
	CompressVolumeResources* m_pCompressVolume;
	ID3D11Device*            m_pDevice;

	VolumeInfoGPU    m_volumeInfoGPU;
	BrickIndexGPU    m_brickIndexGPU;
	uint2*           m_pBrickToSlot;
	uint*            m_pSlotTimestepMin;
	uint*            m_pSlotTimestepMax;
	cudaEvent_t      m_brickIndexUploadEvent;
	BrickRequestsGPU m_brickRequestsGPU;
	uint*            m_pBrickRequestCounts;
	uint*            m_pBrickTimestepMins;
	cudaEvent_t      m_brickRequestsDownloadEvent;

	Integrator       m_integrator;

	struct SlotInfo
	{
		SlotInfo() { Clear(); }

		uint       brickIndex;
		uint       timestepFirst;
		uint       timestepCount;
		int        lastUseTimestamp;

		void Clear() { brickIndex = ~0u; timestepFirst = ~0u; timestepCount = 0; ClearTimestamp(); }
		void ClearTimestamp() { lastUseTimestamp = std::numeric_limits<int>::min(); }
		void InvalidateTimestamp() { if(lastUseTimestamp >= 0) lastUseTimestamp -= std::numeric_limits<int>::max(); else ClearTimestamp(); }
		bool HasValidTimestamp() const { return lastUseTimestamp >= 0; }
		bool IsFilled() const { return timestepCount > 0; }
	};
	int m_currentTimestamp; // incremented per trace round

	// volume-dependent resources
	std::vector<float*>   m_dpChannelBuffer;
	std::vector<float*>   m_pChannelBufferCPU;
	BrickSlot             m_brickAtlas;
	tum3D::Vec3ui         m_brickSlotCount;
	std::vector<SlotInfo> m_brickSlotInfo;
	std::map<uint, uint>  m_bricksOnGPU; // map from brick index to slot index
	std::vector<BrickSortItem> m_bricksToDo;
	bool                  m_bricksToDoDirty;
	bool                  m_bricksToDoPrioritiesDirty;

	// param-dependent resources
	// note: if m_traceParams.m_cpuTracing, then these are actually CPU arrays!
	LineCheckpoint*       m_dpLineCheckpoints;
	uint*                 m_dpLineVertexCounts;

	// result resources
	std::shared_ptr<LineBuffers> m_pResult;
	bool                         m_indexBufferDirty;
	std::vector<LineVertex>      m_lineVerticesCPU;


	// only valid while tracing
	const TimeVolume*         m_pVolume;
	ParticleTraceParams       m_traceParams;
	const FlowGraph*          m_pFlowGraph;
	int                       m_timestepMax; // last timestep that might be needed (limited by lineMaxAge!)

	float                     m_progress;


	std::vector<const TimeVolumeIO::Brick*> m_bricksToLoad;


	// timing
	MultiTimerGPU m_timerUploadDecompress;
	MultiTimerGPU m_timerIntegrate;
	MultiTimerGPU m_timerBuildIndexBuffer;
	TimerCPU      m_timerIntegrateCPU;
	TimerCPU      m_timerDiskWait;
	TimerCPU      m_timerTrace;

	Timings       m_timings;

	// stats
	std::unordered_set<uint> m_bricksLoaded; // 4D linear brick index: timestep * brickCount + brickIndex
	Stats                    m_stats;


	bool m_verbose;
};


#endif
