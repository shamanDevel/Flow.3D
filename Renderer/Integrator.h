#ifndef __TUM3D__INTEGRATOR_H__
#define __TUM3D__INTEGRATOR_H__


#include <global.h>

#include <vector>

#include <cuda_runtime.h>

#include <Vec.h>
#include <TimeVolumeInfo.h>

#include "BrickSlot.h"
#include "IntegrationParamsGPU.h"
#include "LineInfoGPU.h"
#include "ParticleTraceParams.h"
#include "TracingCommon.h"


class Integrator
{
public:
	Integrator();
	~Integrator();

	bool Create();
	void Release();
	bool IsCreated() const { return m_isCreated; }

	void SetVolumeInfo(const TimeVolumeInfo& volumeInfo) { m_volumeInfo = volumeInfo; }


	void ForceParamUpdate(const ParticleTraceParams& params);
	void ForceParamUpdate(const ParticleTraceParams& params, const LineInfo& lineInfo);

	void IntegrateSimpleParticles(const BrickSlot& brickAtlas, SimpleParticleVertex* dpParticles, uint particleCount, const ParticleTraceParams& params);

	void IntegrateLines(const BrickSlot& brickAtlas, const LineInfo& lineInfo, const ParticleTraceParams& params, SimpleParticleVertexDeltaT* dpParticles = nullptr);

	//Initializes the particles. This must be called once before IntegrateParticles(...)
	void InitIntegrateParticles(const LineInfo& lineInfo, const ParticleTraceParams& params);

	//Integrates particles
	//line vertices are the particles in a circular-buffer fashion per line;
	//the checkpoints contains the seed.
	//It can be used interchangeable to IntegrateLines
	//seed: the seeding position in the line or -1 if nothing should be seeded
	void IntegrateParticles(const BrickSlot& brickAtlas, const LineInfo& lineInfo, const ParticleTraceParams& params, int seed, double tpf);

	//Triggers an additional seeding kernel without advection
	void IntegrateParticlesExtraSeed(const LineInfo& lineInfo, const ParticleTraceParams& params, int seed);

	// returns total number of indices
	uint BuildLineIndexBuffer(const uint* dpLineVertexCounts, uint lineVertexStride, uint* dpIndices, uint lineCount);
	uint BuildLineIndexBufferCPU(const uint* pLineVertexCounts, uint lineVertexStride, uint* pIndices, uint lineCount);

	uint BuildParticleIndexBuffer(uint* dpIndices, uint lineVertexStride, uint lineCount);

private:
	void UpdateIntegrationParams(const ParticleTraceParams& params, float timeMax, bool force = false);
	void UpdateLineInfo(const ParticleTraceParams& params, const LineInfo& lineInfo, bool force = false);

	TimeVolumeInfo m_volumeInfo;

	bool m_isCreated;

	IntegrationParamsGPU m_integrationParamsGPU;
	bool m_integrationParamsCpuTracingPrev;
	cudaEvent_t m_integrationParamsUploadEvent;
	LineInfoGPU m_lineInfoGPU;
	bool m_lineInfoCpuTracingPrev;
	cudaEvent_t m_lineInfoUploadEvent;

	uint* m_dpIndexOffset;
	uint m_indexOffsetSize;
	uint m_indexCountTotal;
	cudaEvent_t m_indexCountTotalDownloadEvent;
};


#endif
