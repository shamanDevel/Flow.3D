#include <global.h>

#include "AdvectMode.h"
#include "BrickIndexGPU.h"
#include "BrickRequestsGPU.h"
#include "IntegrationParamsGPU.h"
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

extern texture<float4, cudaTextureType3D, cudaReadModeElementType> g_texVolume1;


template<eAdvectMode advectMode, eTextureFilterMode filterMode>
__global__ void integrateSimpleParticlesKernel(
	SimpleParticleVertex* pParticles, uint particleCount,
	float deltaTime, uint stepCountMax)
{
	uint index = blockIdx.x * blockDim.x + threadIdx.x;

	if(index >= particleCount)
		return;

	SimpleParticleVertex vertex = pParticles[index];

	// find brick we're in
	float3 brickBoxMin;
	float3 brickBoxMax;
	float3 world2texOffset;
	float  world2texScale;
	if(!findBrick(vertex.Position, brickBoxMin, brickBoxMax, world2texOffset, world2texScale)) {
		return;
	}

	// get velocity at initial position
	float3 velocity = c_volumeInfo.velocityScale * sampleVolume<filterMode, float4, float3>(g_texVolume1, w2t(vertex.Position));

	uint step = 0;
	uint stepsAccepted = 0;

	while(step < stepCountMax) {
		float deltaTimeBak = deltaTime;
		// limit deltaTime ..
		// .. so we don't integrate past timeMax
		deltaTime = min(deltaTime, c_integrationParams.timeMax - vertex.Time);
		// .. so we don't leave the current brick's safe region
		float distMax = c_integrationParams.brickSafeMarginWorld + distanceToBrickBorder(vertex.Position, brickBoxMin, brickBoxMax);
		deltaTime = min(deltaTime, distMax / c_integrationParams.velocityMaxWorld);
		// integrate
		bool stepAccepted = advect<advectMode, filterMode>(
			g_texVolume1,
			vertex.Position, vertex.Time, velocity,
			deltaTime,
			world2texOffset, world2texScale,
			c_volumeInfo.velocityScale);
		++step;

		if(stepAccepted) {
			++stepsAccepted;

			// if we artificially limited deltaTime earlier, reset it now
			// (if we didn't, the new deltaTime is larger than the backup anyway)
			deltaTime = fmax(deltaTime, deltaTimeBak);

			// check if we left the current brick
			if(!isInBrick(vertex.Position, brickBoxMin, brickBoxMax)) {
				bool isOutOfDomain = c_volumeInfo.isOutsideOfDomain(vertex.Position);
				if(isOutOfDomain || !findBrick(vertex.Position, brickBoxMin, brickBoxMax, world2texOffset, world2texScale)) {
					// new brick isn't available (or we went out of the domain) - get outta here
					// (if we're still inside the domain, the new brick has already been requested in findBrick!)
					break;
				} else {
					// semi-HACK: update velocity from new brick (can be different to previous one because of lossy compression)
					//            this avoids excessively small time steps at some brick boundaries
					velocity = c_volumeInfo.velocityScale * sampleVolume<filterMode, float4, float3>(g_texVolume1, w2t(vertex.Position));
				}
			}
		}
	}

	// write out result
	pParticles[index] = vertex;
}


#include "cudaUtil.h"

#include "IntegratorKernelDefines.h"


void integratorKernelSimpleParticles(SimpleParticleVertex* dpParticles, uint particleCount, float deltaT, uint stepCountMax, eAdvectMode advectMode, eTextureFilterMode filterMode)
{
	uint blockSize = 128;
	uint blockCount = (particleCount + blockSize - 1) / blockSize;

#define INTEGRATE(advect, filter) \
		integrateSimpleParticlesKernel \
		<advect, filter> \
		<<<blockCount, blockSize>>> \
		(dpParticles, particleCount, deltaT, stepCountMax)

	ADVECT_SWITCH;
	cudaCheckMsg("integrateSimpleParticlesKernel execution failed");

#undef INTEGRATE
}
