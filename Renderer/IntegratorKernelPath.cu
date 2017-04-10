#include <global.h>

#include "AdvectMode.h"
#include "BrickIndexGPU.h"
#include "BrickRequestsGPU.h"
#include "IntegrationParamsGPU.h"
#include "LineInfoGPU.h"
#include "TextureFilterMode.h"
#include "TracingCommon.h"
#include "VolumeInfoGPU.h"

#include "AdvectTime.cuh"
#include "Coords.cuh"
#include "IntegratorCommon.cuh"
#include "TextureFilterTime.cuh"
#include "Jacobian.cuh"

extern __constant__ VolumeInfoGPU c_volumeInfo;
extern __constant__ BrickIndexGPU c_brickIndex;
extern __constant__ BrickRequestsGPU c_brickRequests;
extern __constant__ IntegrationParamsGPU c_integrationParams;
extern __constant__ LineInfoGPU c_lineInfo;

extern texture<float4, cudaTextureType3D, cudaReadModeElementType> g_texVolume1;


template<eAdvectMode advectMode, eTextureFilterMode filterMode>
__global__ void integratePathLinesKernel()
{
	uint lineIndex = blockIdx.x * blockDim.x + threadIdx.x;

	if(lineIndex >= c_lineInfo.lineCount)
		return;

	uint lineLength = c_lineInfo.pVertexCounts[lineIndex];

	if(lineLength >= c_lineInfo.lineLengthMax)
		return;

	LineVertex vertex;
	// get initial position from checkpoints array
	vertex.Position = c_lineInfo.pCheckpoints[lineIndex].Position;
	vertex.Time     = c_lineInfo.pCheckpoints[lineIndex].Time;

	if(vertex.Time >= c_integrationParams.timeMax || c_volumeInfo.isOutsideOfDomain(vertex.Position))
		return;


	// find brick we're in
	float3 brickBoxMin;
	float3 brickBoxMax;
	float3 world2texOffset;
	float3 world2texScale;
	float brickTimeMin;
	float brickTimeMax;
	float time2texOffset;
	float time2texScale;
	if(!findBrickTime(
		vertex.Position, vertex.Time,
		brickBoxMin, brickBoxMax, world2texOffset, world2texScale,
		brickTimeMin, brickTimeMax, time2texOffset, time2texScale))
	{
		return;
	}
	//assert(vertex.Time < brickTimeMax);


	const float timestepInc = (float)c_volumeInfo.brickSizeVoxelsWithOverlap;

	// get velocity at initial position
	float4 vel4 = sampleVolume<filterMode, float4, float4>(g_texVolume1, w2t(vertex.Position));
	vertex.Velocity = c_volumeInfo.velocityScale * make_float3(vel4.x, vel4.y, vel4.z);

	vertex.LineID = lineIndex;


	LineVertex* pVertices = c_lineInfo.pVertices + lineIndex * c_lineInfo.vertexStride + lineLength;

	if(lineLength == 0) {
		// new line - build normal: arbitrary vector perpendicular to velocity
		float3 tangent = normalize(vertex.Velocity);
		vertex.Normal = cross(tangent, make_float3(1.0f, 0.0f, 0.0f));
		if(length(vertex.Normal) < 0.01f) vertex.Normal = cross(tangent, make_float3(0.0f, 1.0f, 0.0f));
		vertex.Normal = normalize(vertex.Normal);
		vertex.Jacobian = getJacobian<filterMode>(g_texVolume1, w2t(vertex.Position), c_integrationParams.gridSpacing);
		float3 gradT = sampleScalarGradient<filterMode>(g_texVolume1, w2t(vertex.Position), c_integrationParams.gridSpacing);
		vertex.heat = make_float4(gradT, vel4.w);

		// write out initial vertex
		*pVertices++ = vertex;
		++lineLength;
	} else {
		// existing line - get old normal
		vertex.Normal = c_lineInfo.pCheckpoints[lineIndex].Normal;
	}

	// get the last vertex that was written out
	float3 lastOutPos  = (pVertices - 1)->Position;
	float  lastOutTime = (pVertices - 1)->Time;

	float deltaTime = c_lineInfo.pCheckpoints[lineIndex].DeltaT;

	uint step = 0;
	uint stepsAccepted = 0;
	
	bool stayedInAvailableBrick = true;
	while(step < c_integrationParams.stepCountMax &&
		  vertex.Time < brickTimeMax &&
		  lineLength < c_lineInfo.lineLengthMax)
	{
		float deltaTimeBak = deltaTime;
		// limit deltaTime ..
		// .. so we don't integrate past timeMax
		deltaTime = min(deltaTime, brickTimeMax - vertex.Time);
		// .. so we don't leave the current brick's safe region
		float distMax = c_integrationParams.brickSafeMarginWorld + distanceToBrickBorder(vertex.Position, brickBoxMin, brickBoxMax);
		deltaTime = min(deltaTime, distMax / c_integrationParams.velocityMaxWorld);
		// integrate
		
		bool stepAccepted = advectTime<advectMode, filterMode>(
			g_texVolume1,
			vertex.Position, vertex.Time, vertex.Velocity,
			deltaTime,
			world2texOffset, world2texScale,
			time2texOffset, time2texScale, timestepInc,
			c_volumeInfo.velocityScale);

		//bool stepAccepted = false;
	
		++step;
		if(stepAccepted) {
			++stepsAccepted;

			// if we artificially limited deltaTime earlier, reset it now
			// (if we didn't, the new deltaTime is larger than the backup anyway)
			deltaTime = fmax(deltaTime, deltaTimeBak);

			// re-orthogonalize normal wrt. tangent == velocity direction
			float3 binormal = cross(vertex.Velocity, vertex.Normal);
			vertex.Normal = normalize(cross(binormal, vertex.Velocity));

			float3 posDiff = vertex.Position - lastOutPos;
			float timeDiff = vertex.Time     - lastOutTime;
			if((dot(posDiff, posDiff) >= c_integrationParams.outputPosDiffSquared) || (timeDiff >= c_integrationParams.outputTimeDiff)) {
				//get jacobian and heat for measures
				vertex.Jacobian = getJacobian<filterMode>(g_texVolume1, w2t(vertex.Position), c_integrationParams.gridSpacing);
				vel4 = sampleVolume<filterMode, float4, float4>(g_texVolume1, w2t(vertex.Position));
				float3 gradT = sampleScalarGradient<filterMode>(g_texVolume1, w2t(vertex.Position), c_integrationParams.gridSpacing);
				vertex.heat = make_float4(gradT, vel4.w);
				
				// write out (intermediate) result
				*pVertices++ = vertex;
				++lineLength;

				lastOutPos  = vertex.Position;
				lastOutTime = vertex.Time;
			}

			// check if we left the current brick
			if(!isInBrickTime(vertex.Position, vertex.Time, brickBoxMin, brickBoxMax, brickTimeMin, brickTimeMax)) {
				bool isOutOfDomain = c_volumeInfo.isOutsideOfDomain(vertex.Position);
				if(isOutOfDomain ||
					!findBrickTime(
						vertex.Position, vertex.Time,
						brickBoxMin, brickBoxMax, world2texOffset, world2texScale,
						brickTimeMin, brickTimeMax, time2texOffset, time2texScale))
				{
					// new brick isn't available (or we went out of the domain) - get outta here
					// (if we're still inside the domain, the new brick has already been requested in findBrickTime!)
					stayedInAvailableBrick = false;
					break;
				} else {
					// semi-HACK: update velocity from new brick (can be different to previous one because of lossy compression)
					//            this avoids excessively small time steps at some brick boundaries
					vertex.Velocity = c_volumeInfo.velocityScale * sampleVolumeTime<filterMode, float4, float3>(g_texVolume1, w2t(vertex.Position), time2tex(vertex.Time), timestepInc);
				}
			}
		}
	}
	
	c_lineInfo.pVertexCounts[lineIndex] = lineLength;
	//assert(c_lineInfo.pVertexCounts[lineIndex] < lineLengthMax);

	// update checkpoint for next integration round
	c_lineInfo.pCheckpoints[lineIndex].Position = vertex.Position;
	c_lineInfo.pCheckpoints[lineIndex].Time     = vertex.Time;
	c_lineInfo.pCheckpoints[lineIndex].Normal   = vertex.Normal;
	c_lineInfo.pCheckpoints[lineIndex].DeltaT   = deltaTime;

	c_lineInfo.pCheckpoints[lineIndex].StepsAccepted += stepsAccepted;
	c_lineInfo.pCheckpoints[lineIndex].StepsTotal    += step;

	// if the line is still alive and in an available brick, request it again for next round
	if(vertex.Time < c_integrationParams.timeMax &&
	   lineLength < c_lineInfo.lineLengthMax &&
	   stayedInAvailableBrick)
	{
		// find out which brick we're in now
		uint3 brickIndex = c_volumeInfo.getBrickIndex(vertex.Position);
		uint brickLinearIndex = c_volumeInfo.getBrickLinearIndex(brickIndex);
		uint timestepIndex = c_volumeInfo.getFloorTimestepIndex(vertex.Time);
		// request it to be loaded
		c_brickRequests.requestBrickTime(brickLinearIndex, timestepIndex);
	}
}


#include "cudaUtil.h"

#include "IntegratorKernelDefines.h"


void integratorKernelPathLines(const LineInfo& lineInfo, eAdvectMode advectMode, eTextureFilterMode filterMode)
{
	uint blockSize = 128;
	uint blockCount = (lineInfo.lineCount + blockSize - 1) / blockSize;

#define INTEGRATE(advect, filter) integratePathLinesKernel <advect, filter> <<<blockCount, blockSize>>> ()

	ADVECT_SWITCH;
	cudaCheckMsg("integratePathLinesKernel execution failed");

#undef INTEGRATE
}
