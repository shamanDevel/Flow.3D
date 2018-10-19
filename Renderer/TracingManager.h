#ifndef __TUM3D__TRACINGMANAGER_H__
#define __TUM3D__TRACINGMANAGER_H__


#include <global.h>

#include <limits>
#include <memory>
#include <map>
#include <unordered_set>
#include <vector>
#include <chrono>
#include <random>

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

#include "IntegratorTimeInCell.cuh"

#include <TraceableVolume.h>


class TracingManager
{
public:
	TracingManager();
	~TracingManager();

	bool Create(ID3D11Device* pDevice);
	void Release();
	bool IsCreated() const { return m_isCreated; }

	// settings for resource limits. applied on the next StartTracing.
	uint GetBrickSlotCountMax() const { return m_traceableVol->m_brickSlotCountMax; }
	uint& GetBrickSlotCountMax() { return m_traceableVol->m_brickSlotCountMax; }
	void SetBrickSlotCountMax(uint brickSlotCountMax) { m_traceableVol->m_brickSlotCountMax = brickSlotCountMax; }
	uint GetTimeSlotCountMax() const { return m_traceableVol->m_timeSlotCountMax; }
	uint& GetTimeSlotCountMax() { return m_traceableVol->m_timeSlotCountMax; }
	void SetTimeSlotCountMax(uint timeSlotCountMax) { m_traceableVol->m_timeSlotCountMax = timeSlotCountMax; }

	bool& GetVerbose() { return m_verbose; }

	//Sets the parameters even when the tracing is not restarted (as an alternate to StartTracing)
	//This is needed for particles that allow changes of some parameters (e.g. the seed box) without retracing
	void SetParams(const ParticleTraceParams& traceParams);
	//Starts the tracing, performs the first tracing step
	bool StartTracing(const TimeVolume& volume, GPUResources* pCompressShared, CompressVolumeResources* pCompressVolume, const ParticleTraceParams& traceParams, const FlowGraph& flowGraph);

	//Performs one tracing step. Is called multiple times (once per frame), until it is done and returns true
	bool Trace(); // returns true if finished TODO error code
	
	//Cancels the tracing: resets all parameters
	void CancelTracing();
	bool IsTracing() const;
	float GetTracingProgress() const;

	//Called by a button in the ui: many particles should be seeded now
	void SeedManyParticles() { m_seedManyParticles = true; }

	const std::vector<const TimeVolumeIO::Brick*>& GetBricksToLoad() const { return m_traceableVol->m_bricksToLoad; }

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

	SimpleParticleVertexDeltaT* m_dpParticles = nullptr;

private:

	HRESULT CreateParamDependentResources();
	void ReleaseParamDependentResources();

	HRESULT CreateResultResources();
	void ReleaseResultResources();


	void UpdateFrameBudgets();


	

	
	
	// Perform one round of particle tracing, using the bricks which are currently on the GPU.
	void TraceRound();


	//Traces particles. If LineModeIsIterative returns true for the selected line mode,
	//then the line should be treated as particles rather than lines:
	//The current checkpoint is the seeding point and all particles are advected
	bool TraceParticlesIteratively();
	//Start particle tracing. This fills all bricks of the current time steps in the loading queue
	void StartTracingParticlesIteratively();
	//Builds the index buffer for particles. This is a dummy index buffer / the identity.
	//Therefore, all particles are rendered
	void BuildParticlesIndexBuffer();

	//Creates the initial checkpoints / seeds
	void CreateInitialCheckpoints(float spawnTime);
	void CreateInitialCheckpointsRegularGrid(float3 seedBoxMin, float3 seedBoxSize);
	void CreateInitialCheckpointsRandom(float3 seedBoxMin, float3 seedBoxSize);
	void CreateInitialCheckpointsFTLE(float3 seedBoxMin, float3 seedBoxSize);
	//Sets the delta-t for the checkpoints
	void SetCheckpointTimeStep(float deltaT);

	
	TraceableVolumeHeuristics GetHeuristics() { return TraceableVolumeHeuristics(m_traceParams.HeuristicDoSqrtBPCount(), m_traceParams.HeuristicUseFlowGraph(), m_traceParams.HeuristicDoSqrtBPProb(), m_traceParams.m_heuristicBonusFactor, m_traceParams.m_heuristicPenaltyFactor, m_pFlowGraph); }

	void UpdateProgress();

	void ComputeTimings();
	void PrintTimings() const;
	void ComputeStats();
	void PrintStats() const;


	float GetLineSpawnTime() const;
	int GetLineFloorTimestepIndex(float lineTime) const;
	float GetLineTimeMax() const;


	


	// per-frame budgets
	uint m_uploadsPerFrame;
	uint m_roundsPerFrame;


	bool          m_isCreated;
	// valid between create/release
	ID3D11Device*            m_pDevice;

	

	Integrator       m_integrator;

	


	int m_currentTimestamp; // incremented per trace round

	// param-dependent resources
	// note: if m_traceParams.m_cpuTracing, then these are actually CPU arrays!
	LineCheckpoint*       m_dpLineCheckpoints;
	uint*                 m_dpLineVertexCounts;
	

	// result resources
	std::shared_ptr<LineBuffers> m_pResult;
	bool                         m_indexBufferDirty;
	std::vector<LineVertex>      m_lineVerticesCPU;


	// only valid while tracing
	
	ParticleTraceParams       m_traceParams;
	const FlowGraph*          m_pFlowGraph;
	int                       m_timestepMax; // last timestep that might be needed (limited by lineMaxAge!)

	float                     m_progress;

	// for particle tracing
	bool									m_particlesNeedsUploadTimestep;
	int										m_particlesSeedPosition;
	bool									m_seedManyParticles;
	std::chrono::steady_clock::time_point	m_particlesLastTime;
	std::chrono::steady_clock::time_point	m_particlesLastFrame;

	

	// for checkpoint generation
	std::mt19937 m_engine;
	std::uniform_real_distribution<float> m_rng;
	std::vector<LineCheckpoint> m_checkpoints;
	//number of attempts to sample in the seed texture
	int m_numRejections;

	// timing

	MultiTimerGPU m_timerIntegrate;
	MultiTimerGPU m_timerBuildIndexBuffer;
	TimerCPU      m_timerIntegrateCPU;
	
	TimerCPU      m_timerTrace;
	Timings       m_timings;

	// stats
	Stats                    m_stats;

	// for cell time tracking
	CellTextureGPU m_cellTextureGPU;

	bool m_verbose;

	TraceableVolume* m_traceableVol;

private:
	// disable copy and assignment
	TracingManager(const TracingManager&);
	TracingManager& operator=(const TracingManager&);
};


#endif
