#include <global.h>

#include "AdvectMode.h"
#include "BrickIndexGPU.h"
#include "BrickRequestsGPU.h"
#include "IntegrationParamsGPU.h"
#include "LineInfoGPU.h"
#include "TextureFilterMode.h"
#include "TracingCommon.h"
#include "VolumeInfoGPU.h"
#include "IntegratorTimeInCell.cuh"

#include "Advect.cuh"
#include "Coords.cuh"
#include "IntegratorCommon.cuh"
#include "TextureFilter.cuh"
#include "Jacobian.cuh"

//extern __constant__ VolumeInfoGPU c_volumeInfo;
extern __constant__ BrickIndexGPU c_brickIndex;
extern __constant__ BrickRequestsGPU c_brickRequests;
extern __constant__ IntegrationParamsGPU c_integrationParams;
//extern __constant__ LineInfoGPU c_lineInfo;

extern texture<float4, cudaTextureType3D, cudaReadModeElementType> g_texVolume1;



template<eAdvectMode advectMode, eTextureFilterMode filterMode>
__global__ void integrateParticlesKernel(LineInfoGPU c_lineInfo, VolumeInfoGPU c_volumeInfo, double tpf)
{
	uint index = blockIdx.x * blockDim.x + threadIdx.x;

	if (index >= c_lineInfo.lineCount * c_lineInfo.lineLengthMax)
		return;

	int lineIndex = index / c_lineInfo.lineLengthMax;
	float deltaTime = c_lineInfo.pCheckpoints[lineIndex].DeltaT * tpf;
	if (deltaTime == 0) return;

	//get vertex
	LineVertex vertex = c_lineInfo.pVertices[index];

	if (vertex.Time < 0) {
		return; //this particle is invalidated
	}

	if (c_volumeInfo.isOutsideOfDomain(vertex.Position)) {
		//we are outside, invalidate particle.
		c_lineInfo.pVertices[index].Time = -1;
		return;
	}

	// find brick we're in
	float3 brickBoxMin;
	float3 brickBoxMax;
	float3 world2texOffset;
	float3 world2texScale;
	if (!findBrick(c_volumeInfo, vertex.Position, brickBoxMin, brickBoxMax, world2texOffset, world2texScale)) {
		//no brick found, this should not happen
		printf("i=%d: pos=(%5.3f,%5.3f,%5.3f) no brick found\n", index, vertex.Position.x, vertex.Position.y, vertex.Position.z);
		c_lineInfo.pVertices[index].Time = -1;
		return;
	}

	// get velocity at initial position
	float3 tmpTexPos = w2t(vertex.Position);
	float4 vel4 = sampleVolume<filterMode, float4, float4>(g_texVolume1, tmpTexPos);
	vertex.Velocity = c_volumeInfo.velocityScale * make_float3(vel4.x, vel4.y, vel4.z);

	if (dot(vertex.Velocity, vertex.Velocity) < c_integrationParams.minVelocitySquared) {
		c_lineInfo.pVertices[index].Time = -1;
		return;
	}

#if 0
	if (length(vertex.Velocity) < 0.00001) {
		printf("i=%d: zero velocity! pos=(%+5.3f,%+5.3f,%+5.3f) brick w2tOffset=(%+5.3f,%+5.3f,%+5.3f), s2tScale=(%5.3f,%5.3f,%5.3f), texPos=(%5.3f,%5.3f,%5.3f)\n",
			index, vertex.Position.x, vertex.Position.y, vertex.Position.z,
			world2texOffset.x, world2texOffset.y, world2texOffset.z,
			world2texScale.x, world2texScale.y, world2texScale.z,
			tmpTexPos.x, tmpTexPos.y, tmpTexPos.z);
	}
	else {
		printf("i=%d: non-null-vel!  pos=(%+5.3f,%+5.3f,%+5.3f) brick w2tOffset=(%+5.3f,%+5.3f,%+5.3f), s2tScale=(%5.3f,%5.3f,%5.3f), texPos=(%5.3f,%5.3f,%5.3f)\n",
			index, vertex.Position.x, vertex.Position.y, vertex.Position.z,
			world2texOffset.x, world2texOffset.y, world2texOffset.z,
			world2texScale.x, world2texScale.y, world2texScale.z,
			tmpTexPos.x, tmpTexPos.y, tmpTexPos.z);
	}
#endif

	uint step = 0;

	float timeMax = vertex.Time + deltaTime; //this is the goal to reach
	float difference = deltaTime / 100.0f; //resulting deltaT can be +- 1% different from the target deltaT
	                                       //otherwise, high-order integrations take a lot of time, because
	                                       //they execute towards the end many, many small time steps
	bool stayedInAvailableBrick = true;
	while (step < c_integrationParams.stepCountMax &&
		vertex.Time + difference < timeMax
		&& stayedInAvailableBrick)
	{
		float deltaTimeBak = deltaTime;
		// limit deltaTime ..
		// .. so we don't integrate past timeMax
		deltaTime = min(deltaTime, timeMax - vertex.Time);
		// .. so we don't leave the current brick's safe region
		float distMax = c_integrationParams.brickSafeMarginWorld + distanceToBrickBorder(vertex.Position, brickBoxMin, brickBoxMax);
		deltaTime = min(deltaTime, distMax / c_integrationParams.velocityMaxWorld);
		// integrate
		bool stepAccepted = advect<advectMode, filterMode>(
			g_texVolume1,
			vertex.Position, vertex.Time, vertex.Velocity,
			deltaTime,
			world2texOffset, world2texScale,
			c_volumeInfo.velocityScale);
		++step;
		if (stepAccepted) {
			// if we artificially limited deltaTime earlier, reset it now
			// (if we didn't, the new deltaTime is larger than the backup anyway)
			deltaTime = fmax(deltaTime, deltaTimeBak);
			// check if we left the current brick
			if (!isInBrick(vertex.Position, brickBoxMin, brickBoxMax)) {
				bool isOutOfDomain = c_volumeInfo.isOutsideOfDomain(vertex.Position);
				if (isOutOfDomain || !findBrick(c_volumeInfo, vertex.Position, brickBoxMin, brickBoxMax, world2texOffset, world2texScale)) {
					// new brick isn't available (or we went out of the domain) - get outta here
					// (if we're still inside the domain, the new brick has already been requested in findBrick!)
					stayedInAvailableBrick = false;
					break;
				}
				else {
					// semi-HACK: update velocity from new brick (can be different to previous one because of lossy compression)
					//            this avoids excessively small time steps at some brick boundaries
					vertex.Velocity = c_volumeInfo.velocityScale * sampleVolume<filterMode, float4, float3>(g_texVolume1, w2t(vertex.Position));
				}
			}
		}
	}

	//get jacobian and heat for measures
	vertex.Jacobian = getJacobian<filterMode>(g_texVolume1, w2t(vertex.Position), c_integrationParams.gridSpacing);
	float3 gradT = sampleScalarGradient<filterMode>(g_texVolume1, w2t(vertex.Position), c_integrationParams.gridSpacing);
	vertex.Heat = vel4.w;
	vertex.HeatCurrent = gradT;

	//compute time-in-cell measures
	IntegratorTimeInCell::processParticle(&vertex, c_volumeInfo, deltaTime);

	//write vertex back
	c_lineInfo.pVertices[index] = vertex;
}


__global__ void seedParticlesKernel(LineInfoGPU c_lineInfo, int particleEntry)
{
	uint lineIndex = blockIdx.x * blockDim.x + threadIdx.x;

	if (lineIndex >= c_lineInfo.lineCount)
		return;

	//set initial position
	LineVertex vertex = c_lineInfo.pVertices[lineIndex * c_lineInfo.vertexStride + particleEntry];
	vertex.Position = c_lineInfo.pCheckpoints[lineIndex].Position;
	vertex.SeedPosition = vertex.Position;
	vertex.Time = 0;
	c_lineInfo.pVertices[lineIndex * c_lineInfo.vertexStride + particleEntry] = vertex;
}

__global__ void initParticlesKernel(LineInfoGPU c_lineInfo)
{
	uint index = blockIdx.x * blockDim.x + threadIdx.x;

	if (index >= c_lineInfo.lineCount * c_lineInfo.lineLengthMax)
		return;

	//set time to <0 -> they are marked invalid
	LineVertex v = {};
	v.Time = -1;
	v.LineID = index / c_lineInfo.lineLengthMax;
	c_lineInfo.pVertices[index] = v;
}


#include "cudaUtil.h"

#include "IntegratorKernelDefines.h"


void integratorKernelParticles(LineInfoGPU lineInfo, VolumeInfoGPU volumeInfo, eAdvectMode advectMode, eTextureFilterMode filterMode, double tpf)
{
	uint blockSize = 128;
	uint blockCount = (lineInfo.lineCount * lineInfo.vertexStride + blockSize - 1) / blockSize;

#define INTEGRATE(advect, filter) integrateParticlesKernel <advect, filter> <<<blockCount, blockSize>>> (lineInfo, volumeInfo, tpf)

	ADVECT_SWITCH;
	cudaCheckMsg("integrateParticlesKernel execution failed");

#undef INTEGRATE
}

void integratorKernelSeedParticles(LineInfoGPU lineInfo, int particleEntry)
{
	uint blockSize = 128;
	uint blockCount = (lineInfo.lineCount + blockSize - 1) / blockSize;

	seedParticlesKernel << <blockCount, blockSize >> > (lineInfo, particleEntry);
}

void integratorKernelInitParticles(LineInfoGPU lineInfo)
{
	uint blockSize = 128;
	uint blockCount = (lineInfo.lineCount * lineInfo.vertexStride + blockSize - 1) / blockSize;
	initParticlesKernel << < blockCount, blockSize >> > (lineInfo);
}