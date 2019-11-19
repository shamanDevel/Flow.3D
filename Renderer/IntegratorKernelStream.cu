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
#include "Jacobian.cuh"

//extern __constant__ VolumeInfoGPU c_volumeInfo;
//extern __constant__ BrickIndexGPU c_brickIndex;
//extern __constant__ BrickRequestsGPU c_brickRequests;
extern __constant__ IntegrationParamsGPU c_integrationParams;
//extern __constant__ LineInfoGPU c_lineInfo;

extern cudaTextureObject_t g_texVolume1;


template<eAdvectMode advectMode, eTextureFilterMode filterMode>
__global__ void integrateStreamLinesKernel(LineInfoGPU c_lineInfo, VolumeInfoGPU c_volumeInfo, BrickIndexGPU c_brickIndex, BrickRequestsGPU c_brickRequests)
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
	vertex.SeedPosition = c_lineInfo.pCheckpoints[lineIndex].SeedPosition;

	if(vertex.Time >= c_integrationParams.timeMax || c_volumeInfo.isOutsideOfDomain(vertex.Position))
		return;


	// find brick we're in
	float3 brickBoxMin;
	float3 brickBoxMax;
	float3 world2texOffset;
	float3 world2texScale;
	if (!findBrick(c_volumeInfo, c_brickIndex, c_brickRequests, vertex.Position, brickBoxMin, brickBoxMax, world2texOffset, world2texScale)) {
		return;
	}

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
		vertex.Heat = vel4.w;
		vertex.HeatCurrent = gradT;

		// write out initial vertex
		*pVertices++ = vertex;
		++lineLength;
		//printf("line=%d, index=0: pos=(%f, %f, %f), temp=%f\n", lineIndex, vertex.Position.x, vertex.Position.y, vertex.Position.z, vertex.Heat);
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
		  vertex.Time < c_integrationParams.timeMax &&
		  lineLength < c_lineInfo.lineLengthMax)
	{
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
			vertex.Position, vertex.Time, vertex.Velocity,
			deltaTime,
			world2texOffset, world2texScale,
			c_volumeInfo.velocityScale);
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
				vertex.Heat = vel4.w;
				vertex.HeatCurrent = gradT;

				// write out (intermediate) result
				*pVertices++ = vertex;
				++lineLength;
				//printf("line=%d, index=%d: pos=(%f, %f, %f), temp=%f\n", lineIndex, lineLength, vertex.Position.x, vertex.Position.y, vertex.Position.z, vertex.Heat);

				lastOutPos  = vertex.Position;
				lastOutTime = vertex.Time;
			}

			// check if we left the current brick
			if(!isInBrick(vertex.Position, brickBoxMin, brickBoxMax)) {
				bool isOutOfDomain = c_volumeInfo.isOutsideOfDomain(vertex.Position);
				if(isOutOfDomain) {
					// write out final position
					*pVertices++ = vertex;
					++lineLength;

					lastOutPos  = vertex.Position;
					lastOutTime = vertex.Time;
				}
				if (isOutOfDomain || !findBrick(c_volumeInfo, c_brickIndex, c_brickRequests, vertex.Position, brickBoxMin, brickBoxMax, world2texOffset, world2texScale)) {
					// new brick isn't available (or we went out of the domain) - get outta here
					// (if we're still inside the domain, the new brick has already been requested in findBrick!)
					stayedInAvailableBrick = false;
					break;
				} else {
					// semi-HACK: update velocity from new brick (can be different to previous one because of lossy compression)
					//            this avoids excessively small time steps at some brick boundaries
					vertex.Velocity = c_volumeInfo.velocityScale * sampleVolume<filterMode, float4, float3>(g_texVolume1, w2t(vertex.Position));
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
		// request it to be loaded
		c_brickRequests.requestBrick(brickLinearIndex);
	}
}


#include "cudaUtil.h"

#include "IntegratorKernelDefines.h"


void integratorKernelStreamLines(LineInfoGPU lineInfo, VolumeInfoGPU volumeInfo, BrickIndexGPU brickIndex, BrickRequestsGPU brickRequests, eAdvectMode advectMode, eTextureFilterMode filterMode)
{
	uint blockSize = 128;
	uint blockCount = (lineInfo.lineCount + blockSize - 1) / blockSize;

#define INTEGRATE(advect, filter) integrateStreamLinesKernel <advect, filter> <<<blockCount, blockSize>>> (lineInfo, volumeInfo, brickIndex, brickRequests)

	ADVECT_SWITCH;
	cudaCheckMsg("integrateStreamLinesKernel execution failed");

#undef INTEGRATE
}
