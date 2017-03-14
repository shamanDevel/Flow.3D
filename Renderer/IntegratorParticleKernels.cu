#include <global.h>

#include "AdvectMode.h"
#include "BrickIndexGPU.h"
#include "BrickRequestsGPU.h"
#include "IntegrationParamsGPU.h"
#include "LineInfoGPU.h"
#include "TextureFilterMode.h"
#include "TracingCommon.h"
#include "VolumeInfoGPU.h"

#include "Advect.cuh"
#include "Coords.cuh"
#include "IntegratorCommon.cuh"
#include "TextureFilter.cuh"

extern __constant__ VolumeInfoGPU c_volumeInfo;
extern __constant__ BrickIndexGPU c_brickIndex;
extern __constant__ BrickRequestsGPU c_brickRequests;
extern __constant__ IntegrationParamsGPU c_integrationParams;
extern __constant__ LineInfoGPU c_lineInfo;

extern texture<float4, cudaTextureType3D, cudaReadModeElementType> g_texVolume1;



template<eAdvectMode advectMode, eTextureFilterMode filterMode>
__global__ void integrateParticlesKernel()
{
	uint index = blockIdx.x * blockDim.x + threadIdx.x;

	if (index >= c_lineInfo.lineCount * c_lineInfo.lineLengthMax)
		return;

	//get vertex
	LineVertex vertex = c_lineInfo.pVertices[index];

	if (vertex.Time < 0) {
		return; //this particle is invalidated
	}

	if (c_volumeInfo.isOutsideOfDomain(vertex.Position)) {
		//we are outside, invalidate particle.  (This should not happen)
		c_lineInfo.pVertices[index].Time = -1;
		return;
	}

	// find brick we're in
	float3 brickBoxMin;
	float3 brickBoxMax;
	float3 world2texOffset;
	float  world2texScale;
	if (!findBrick(vertex.Position, brickBoxMin, brickBoxMax, world2texOffset, world2texScale)) {
		//no brick found, this should not happen
		c_lineInfo.pVertices[index].Time = -1;
		return;
	}

	// get velocity at initial position
	vertex.Velocity = c_volumeInfo.velocityScale * sampleVolume<filterMode, float4, float3>(g_texVolume1, w2t(vertex.Position));

	int lineIndex = index / c_lineInfo.lineLengthMax;
	float deltaTime = c_lineInfo.pCheckpoints[lineIndex].DeltaT;

	//hard-coded euler
	bool stepAccepted = advect<ADVECT_EULER, filterMode>(
		g_texVolume1,
		vertex.Position, vertex.Time, vertex.Velocity,
		deltaTime,
		world2texOffset, world2texScale,
		c_volumeInfo.velocityScale);

	// check if we left the current brick
	if (!isInBrick(vertex.Position, brickBoxMin, brickBoxMax)) {
		bool isOutOfDomain = c_volumeInfo.isOutsideOfDomain(vertex.Position);
		if (isOutOfDomain || !findBrick(vertex.Position, brickBoxMin, brickBoxMax, world2texOffset, world2texScale)) {
			// new brick isn't available or we went out of the domain - delete particle
			vertex.Time = -1;
		}
		else {
			// semi-HACK: update velocity from new brick (can be different to previous one because of lossy compression)
			//            this avoids excessively small time steps at some brick boundaries
			vertex.Velocity = c_volumeInfo.velocityScale * sampleVolume<filterMode, float4, float3>(g_texVolume1, w2t(vertex.Position));
		}
	}

	//write vertex back
	c_lineInfo.pVertices[index] = vertex;
}


__global__ void seedParticlesKernel(int particleEntry)
{
	uint lineIndex = blockIdx.x * blockDim.x + threadIdx.x;

	if (lineIndex >= c_lineInfo.lineCount)
		return;

	//set initial position
	LineVertex* pVertices = c_lineInfo.pVertices + lineIndex * c_lineInfo.vertexStride + particleEntry;
	pVertices->Position = c_lineInfo.pCheckpoints[lineIndex].Position;
	pVertices->Time = 0.001;
}

__global__ void initParticlesKernel()
{
	uint index = blockIdx.x * blockDim.x + threadIdx.x;

	if (index >= c_lineInfo.lineCount * c_lineInfo.lineLengthMax)
		return;

	//set time to <0 -> they are marked invalid
	c_lineInfo.pVertices[index].Time = -1;
	c_lineInfo.pVertices[index].LineID = index / c_lineInfo.lineLengthMax;
}


#include "cudaUtil.h"

#include "IntegratorKernelDefines.h"


void integratorKernelParticles(const LineInfo& lineInfo, eAdvectMode advectMode, eTextureFilterMode filterMode)
{
	uint blockSize = 128;
	uint blockCount = (lineInfo.lineCount * lineInfo.lineVertexStride + blockSize - 1) / blockSize;

#define INTEGRATE(advect, filter) integrateParticlesKernel <advect, filter> <<<blockCount, blockSize>>> ()

	ADVECT_SWITCH;
	cudaCheckMsg("integrateStreamLinesKernel execution failed");

#undef INTEGRATE
}

void integratorKernelSeedParticles(const LineInfo& lineInfo, int particleEntry)
{
	uint blockSize = 128;
	uint blockCount = (lineInfo.lineCount + blockSize - 1) / blockSize;

	seedParticlesKernel <<<blockCount, blockSize>>> (particleEntry);
}

void integratorKernelInitParticles(const LineInfo& lineInfo)
{
	uint blockSize = 128;
	uint blockCount = (lineInfo.lineCount * lineInfo.lineVertexStride + blockSize - 1) / blockSize;
	initParticlesKernel <<< blockCount, blockSize >>> ();
}