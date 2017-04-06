#include "Integrator.h"

#include <cassert>
#include <cstring>

#include <thrust/device_ptr.h>
#include <thrust/transform_scan.h>

#include "cudaUtil.h"
#include "cudaTum3D.h"

#include "AdvectMode.h"
#include "BrickIndexGPU.h"
#include "VolumeInfoGPU.h"

#include "IntegratorKernels.h"
#include "IntegratorKernelsCPU.h"
#include "IntegratorParticleKernels.h"

using namespace tum3D;



// HACK: we use the texture reference defined in Raycaster.cu here...
extern texture<float4, cudaTextureType3D, cudaReadModeElementType> g_texVolume1;

extern std::vector<float4> g_volume;



Integrator::Integrator()
	: m_isCreated(false)
	, m_integrationParamsCpuTracingPrev(false), m_integrationParamsUploadEvent(0)
	, m_lineInfoCpuTracingPrev(false), m_lineInfoUploadEvent(0)
	, m_dpIndexOffset(nullptr), m_indexOffsetSize(0)
	, m_indexCountTotal(0), m_indexCountTotalDownloadEvent(0)
{
}

Integrator::~Integrator()
{
	assert(!IsCreated());
}


bool Integrator::Create()
{
	if(IsCreated()) return true;

	cudaSafeCall(cudaHostRegister(&m_integrationParamsGPU, sizeof(m_integrationParamsGPU), cudaHostRegisterDefault));
	cudaSafeCall(cudaEventCreate(&m_integrationParamsUploadEvent, cudaEventDisableTiming));
	cudaSafeCall(cudaEventRecord(m_integrationParamsUploadEvent));

	cudaSafeCall(cudaHostRegister(&m_lineInfoGPU, sizeof(m_lineInfoGPU), cudaHostRegisterDefault));
	cudaSafeCall(cudaEventCreate(&m_lineInfoUploadEvent, cudaEventDisableTiming));
	cudaSafeCall(cudaEventRecord(m_lineInfoUploadEvent));

	cudaSafeCall(cudaHostRegister(&m_indexCountTotal, sizeof(m_indexCountTotal), cudaHostRegisterDefault));
	cudaSafeCall(cudaEventCreate(&m_indexCountTotalDownloadEvent, cudaEventDisableTiming));

	g_texVolume1.addressMode[0] = cudaAddressModeClamp;
	g_texVolume1.addressMode[1] = cudaAddressModeClamp;
	g_texVolume1.addressMode[2] = cudaAddressModeClamp;
	g_texVolume1.normalized = false;

	m_isCreated = true;

	return true;
}

void Integrator::Release()
{
	if(!IsCreated()) return;

	cudaSafeCall(cudaEventDestroy(m_indexCountTotalDownloadEvent));
	m_indexCountTotalDownloadEvent = 0;
	cudaSafeCall(cudaHostUnregister(&m_indexCountTotal));

	cudaSafeCall(cudaFree(m_dpIndexOffset));
	m_dpIndexOffset = nullptr;
	m_indexOffsetSize = 0;

	cudaSafeCall(cudaEventSynchronize(m_lineInfoUploadEvent));
	cudaSafeCall(cudaEventDestroy(m_lineInfoUploadEvent));
	m_lineInfoUploadEvent = 0;
	cudaSafeCall(cudaHostUnregister(&m_lineInfoGPU));

	cudaSafeCall(cudaEventSynchronize(m_integrationParamsUploadEvent));
	cudaSafeCall(cudaEventDestroy(m_integrationParamsUploadEvent));
	m_integrationParamsUploadEvent = 0;
	cudaSafeCall(cudaHostUnregister(&m_integrationParamsGPU));

	m_isCreated = false;
}


void Integrator::ForceParamUpdate(const ParticleTraceParams& params)
{
	float timeMax = 1e10f;
	UpdateIntegrationParams(params, timeMax, true);
}

void Integrator::ForceParamUpdate(const ParticleTraceParams& params, const LineInfo& lineInfo)
{
	float timeMax = lineInfo.lineSeedTime + params.m_lineAgeMax;
	if(LineModeIsTimeDependent(params.m_lineMode))
	{
		timeMax = min(timeMax, (m_volumeInfo.GetTimestepCount() - 1) * m_volumeInfo.GetTimeSpacing());
	}
	UpdateIntegrationParams(params, timeMax, true);
	UpdateLineInfo(params, lineInfo, true);
}


void Integrator::IntegrateSimpleParticles(const BrickSlot& brickAtlas, SimpleParticleVertex* dpParticles, uint particleCount, const ParticleTraceParams& params)
{
	float timeMax = 1e10f;
	UpdateIntegrationParams(params, timeMax);


	g_texVolume1.filterMode = GetCudaTextureFilterMode(params.m_filterMode);
	cudaSafeCall(cudaBindTextureToArray(g_texVolume1, brickAtlas.GetCudaArray()));

	integratorKernelSimpleParticles(dpParticles, particleCount, params.m_advectDeltaT, params.m_advectStepsPerRound, params.m_advectMode, params.m_filterMode);

	cudaSafeCall(cudaUnbindTexture(g_texVolume1));
}


void Integrator::IntegrateLines(const BrickSlot& brickAtlas, const LineInfo& lineInfo, const ParticleTraceParams& params)
{
	float timeMax = lineInfo.lineSeedTime + params.m_lineAgeMax;
	if(LineModeIsTimeDependent(params.m_lineMode))
	{
		timeMax = min(timeMax, (m_volumeInfo.GetTimestepCount() - 1) * m_volumeInfo.GetTimeSpacing());
	}
	UpdateIntegrationParams(params, timeMax);
	UpdateLineInfo(params, lineInfo);


	if(!params.m_cpuTracing)
	{
		g_texVolume1.filterMode = GetCudaTextureFilterMode(params.m_filterMode);
		cudaSafeCall(cudaBindTextureToArray(g_texVolume1, brickAtlas.GetCudaArray()));
	}

	// launch appropriate kernel
	if(params.m_cpuTracing)
	{
		if(params.m_lineMode != LINE_STREAM || !params.m_enableDenseOutput) {
			printf("Integrator::IntegrateLines warning: CPU tracing only supports stream lines with dense output!\n");
		}
		integratorKernelStreamLinesDenseCPU(lineInfo, params.m_advectMode, params.m_filterMode);
	}
	else
	{
		switch(params.m_lineMode)
		{
			case LINE_STREAM:
				if(params.m_enableDenseOutput && IsAdvectModeDenseOutput(params.m_advectMode)) {
					integratorKernelStreamLinesDense(lineInfo, params.m_advectMode, params.m_filterMode);
				} else {
					integratorKernelStreamLines(lineInfo, params.m_advectMode, params.m_filterMode);
				}
				break;
			case LINE_PATH:
				integratorKernelPathLines(lineInfo, params.m_advectMode, params.m_filterMode);
				break;
		}
	}

	if(!params.m_cpuTracing)
	{
		cudaSafeCall(cudaUnbindTexture(g_texVolume1));
	}
}

void Integrator::InitIntegrateParticles(const LineInfo& lineInfo, const ParticleTraceParams& params)
{
	if (params.m_cpuTracing) {
		printf("Integrator::InitIntegrateParticles - ERROR: cpu tracing not supported\n");
		return;
	}

	UpdateLineInfo(params, lineInfo);
	integratorKernelInitParticles(lineInfo);
	printf("Integrator::InitIntegrateParticles: particles initialized\n");
}

void Integrator::IntegrateParticles(const BrickSlot& brickAtlas, const LineInfo& lineInfo, 
	const ParticleTraceParams& params, int seed)
{
	if (params.m_cpuTracing) {
		return;
	}

	float timeMax = lineInfo.lineSeedTime + params.m_lineAgeMax;
	if (LineModeIsTimeDependent(params.m_lineMode))
	{
		timeMax = min(timeMax, (m_volumeInfo.GetTimestepCount() - 1) * m_volumeInfo.GetTimeSpacing());
	}
	UpdateIntegrationParams(params, timeMax);
	UpdateLineInfo(params, lineInfo);

	g_texVolume1.filterMode = GetCudaTextureFilterMode(params.m_filterMode);
	cudaSafeCall(cudaBindTextureToArray(g_texVolume1, brickAtlas.GetCudaArray()));

	if (seed >= 0) {
		//launch seeding kernel
		integratorKernelSeedParticles(lineInfo, seed);
	}

	// launch advection kernel
	integratorKernelParticles(lineInfo, params.m_advectMode, params.m_filterMode);

	cudaSafeCall(cudaUnbindTexture(g_texVolume1));
}


void Integrator::UpdateIntegrationParams(const ParticleTraceParams& params, float timeMax, bool force)
{
	// output distance and tolerance are specified in voxels, so convert to world space here
	float voxelSizeWorld = m_volumeInfo.GetVoxelToWorldFactor();
	float outputPosDiffWorld = params.m_outputPosDiff * voxelSizeWorld;
	float toleranceWorld = params.m_advectErrorTolerance * voxelSizeWorld;

	float brickSafeMarginVoxels = float(m_volumeInfo.GetBrickOverlap()) - 0.5f - float(GetTextureFilterModeCellRadius(params.m_filterMode));
	const float brickSafeMarginVoxelsMin = 0.1f;
	if(brickSafeMarginVoxels < brickSafeMarginVoxelsMin)
	{
		printf("Integrator::IntegrateLines: WARNING: Increasing safe margin from %.2f to %.2f\n", brickSafeMarginVoxels, brickSafeMarginVoxelsMin);
		brickSafeMarginVoxels = brickSafeMarginVoxelsMin;
	}
	float brickSafeMarginWorld = brickSafeMarginVoxels * voxelSizeWorld;

	float velocityMax = 3.4f; //TODO actually get this from the volume
	float velocityMaxWorld = velocityMax * m_volumeInfo.GetPhysicalToWorldFactor().maximum();

	// in upsampled volume, double everything that's relative to the grid spacing
	if(params.m_upsampledVolumeHack)
	{
		outputPosDiffWorld *= 2.0f;
		toleranceWorld *= 2.0f;
	}

	// upload new params only if something changed
	IntegrationParamsGPU integrationParamsGPU;
	integrationParamsGPU.timeMax = timeMax;
	integrationParamsGPU.toleranceSquared = toleranceWorld * toleranceWorld;
	integrationParamsGPU.deltaTimeMin = params.m_advectDeltaTMin;
	integrationParamsGPU.deltaTimeMax = params.m_advectDeltaTMax;
	integrationParamsGPU.brickSafeMarginWorld = brickSafeMarginWorld;
	integrationParamsGPU.velocityMaxWorld = velocityMaxWorld;
	integrationParamsGPU.stepCountMax = params.m_advectStepsPerRound;
	integrationParamsGPU.outputPosDiffSquared = outputPosDiffWorld * outputPosDiffWorld;
	integrationParamsGPU.outputTimeDiff = params.m_outputTimeDiff;
	if(force || params.m_cpuTracing != m_integrationParamsCpuTracingPrev || memcmp(&m_integrationParamsGPU, &integrationParamsGPU, sizeof(m_integrationParamsGPU)) != 0)
	{
		cudaSafeCall(cudaEventSynchronize(m_integrationParamsUploadEvent));
		m_integrationParamsGPU = integrationParamsGPU;
		m_integrationParamsGPU.Upload(params.m_cpuTracing);
		cudaSafeCall(cudaEventRecord(m_integrationParamsUploadEvent));
		m_integrationParamsCpuTracingPrev = params.m_cpuTracing;
	}
}

void Integrator::UpdateLineInfo(const ParticleTraceParams& params, const LineInfo& lineInfo, bool force)
{
	// upload new params only if something changed
	LineInfoGPU lineInfoGPU;
	lineInfoGPU.lineCount = lineInfo.lineCount;
	lineInfoGPU.pCheckpoints = lineInfo.dpCheckpoints;
	lineInfoGPU.pVertices = lineInfo.dpVertices;
	lineInfoGPU.pVertexCounts = lineInfo.dpVertexCounts;
	lineInfoGPU.vertexStride = lineInfo.lineVertexStride;
	lineInfoGPU.lineLengthMax = params.m_lineLengthMax;
	if (lineInfo.lineVertexStride != params.m_lineLengthMax) {
		printf("Integrator::UpdateLineInfo - ERROR: vertex stride (%d) != line length max (%d)\n",
			lineInfo.lineVertexStride, params.m_lineLengthMax);
	}
	if(force || params.m_cpuTracing != m_lineInfoCpuTracingPrev || memcmp(&m_lineInfoGPU, &lineInfoGPU, sizeof(m_lineInfoGPU)) != 0)
	{
		cudaSafeCall(cudaEventSynchronize(m_lineInfoUploadEvent));
		m_lineInfoGPU = lineInfoGPU;
		m_lineInfoGPU.Upload(params.m_cpuTracing);
		cudaSafeCall(cudaEventRecord(m_lineInfoUploadEvent));
		m_lineInfoCpuTracingPrev = params.m_cpuTracing;
	}
}


struct LineIndexCount
{
	__device__ inline uint operator() (uint lineLength) { return (max(lineLength, 1) - 1) * 2; }
};

__global__ void fillLineIndexBufferKernel(uint* pIndices, const uint* pLengths, const uint* pIndexOffsets, uint lineCount, uint lineVertexStride)
{
	uint index = blockIdx.x * blockDim.x + threadIdx.x;
	uint line = blockIdx.y;

	uint lineLength = pLengths[line];

	if(lineLength == 0 || index >= (lineLength - 1) * 2) return;

	uint vertex = line * lineVertexStride + (index + 1) / 2;

	pIndices[pIndexOffsets[line] + index] = vertex;
}

uint Integrator::BuildLineIndexBuffer(const uint* dpLineVertexCounts, uint lineVertexStride, uint* dpIndices, uint lineCount)
{
	uint indexOffsetSize = lineCount + 1;
	if(m_indexOffsetSize != indexOffsetSize)
	{
		cudaSafeCall(cudaFree(m_dpIndexOffset));
		m_indexOffsetSize = indexOffsetSize;
		cudaSafeCall(cudaMalloc(&m_dpIndexOffset, m_indexOffsetSize * sizeof(uint)));
	}

	thrust::device_ptr<const uint> dpLengthThrust(dpLineVertexCounts);
	thrust::device_ptr<uint> dpIndexOffsetThrust(m_dpIndexOffset);
	thrust::transform_exclusive_scan(dpLengthThrust, dpLengthThrust + lineCount + 1, dpIndexOffsetThrust, LineIndexCount(), 0u, thrust::plus<uint>());

	cudaSafeCall(cudaMemcpyAsync(&m_indexCountTotal, m_dpIndexOffset + lineCount, sizeof(uint), cudaMemcpyDeviceToHost));
	cudaSafeCall(cudaEventRecord(m_indexCountTotalDownloadEvent));

	//if(false) {
	//	std::vector<uint> length(lineCount);
	//	std::vector<uint> offset(lineCount);
	//	cudaSafeCall(cudaMemcpy(length.data(), dpLineVertexCounts, lineCount * sizeof(uint), cudaMemcpyDeviceToHost));
	//	cudaSafeCall(cudaMemcpy(offset.data(), m_dpIndexOffset, lineCount * sizeof(uint), cudaMemcpyDeviceToHost));
	//}

	uint indicesPerLine = (lineVertexStride - 1) * 2;
	uint blockSize = 128;
	dim3 blockCount((indicesPerLine + blockSize - 1) / blockSize, lineCount);

	fillLineIndexBufferKernel<<<blockCount, blockSize>>>(dpIndices, dpLineVertexCounts, m_dpIndexOffset, lineCount, lineVertexStride);
	cudaCheckMsg("fillIndexBufferKernel execution failed");

	//std::vector<uint> indices(lineCount * (lineVertexStride - 1) * 2);
	//cudaSafeCall(cudaMemcpy(indices.data(), dpIndices, indices.size() * sizeof(uint), cudaMemcpyDeviceToHost));

	cudaSafeCall(cudaEventSynchronize(m_indexCountTotalDownloadEvent));
	return m_indexCountTotal;
}

uint Integrator::BuildLineIndexBufferCPU(const uint* pLineVertexCounts, uint lineVertexStride, uint* pIndices, uint lineCount)
{
	uint indexCount = 0;
	for(uint line = 0; line < lineCount; line++)
	{
		uint lineVertexCount = pLineVertexCounts[line];
		for(uint lineVertex = 1; lineVertex < lineVertexCount; lineVertex++)
		{
			uint vertex = line * lineVertexStride + lineVertex;
			pIndices[indexCount++] = vertex - 1;
			pIndices[indexCount++] = vertex;
		}
	}
	return indexCount;
}

__global__ void fillParticleIndexBufferKernel(uint* pIndices, uint count)
{
	uint index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index >= count)
		return;
	pIndices[index] = index;
}

uint Integrator::BuildParticleIndexBuffer(uint* dpIndices, uint lineVertexStride, uint lineCount)
{
	uint blockSize = 128;
	uint blockCount = (lineCount * lineVertexStride + blockSize - 1) / blockSize;
	fillParticleIndexBufferKernel <<< blockCount, blockSize >>> (dpIndices, (uint) (lineCount * lineVertexStride));
	return lineCount * lineVertexStride;
}
