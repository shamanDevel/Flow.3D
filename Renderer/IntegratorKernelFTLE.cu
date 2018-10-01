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

#include "Advect.cuh"

extern __constant__ VolumeInfoGPU c_volumeInfo;
extern __constant__ BrickIndexGPU c_brickIndex;
extern __constant__ BrickRequestsGPU c_brickRequests;
extern __constant__ IntegrationParamsGPU c_integrationParams;
//extern __constant__ LineInfoGPU c_lineInfo;

extern texture<float4, cudaTextureType3D, cudaReadModeElementType> g_texVolume1;


template<eAdvectMode advectMode, eTextureFilterMode filterMode>
__global__ void computePathFTLEKernel(SimpleParticleVertexDeltaT* dpParticles, uint particleCount)
{
	uint index = blockIdx.x * blockDim.x + threadIdx.x;

	if (index >= particleCount)
		return;

	//uint lineLength = c_lineInfo.pVertexCounts[index];

	//if(lineLength >= c_lineInfo.lineLengthMax)
	//	return;

	SimpleParticleVertexDeltaT vertex = dpParticles[index];

	//LineVertex vertex;
	// get initial position from checkpoints array
	//vertex.Position = c_lineInfo.pCheckpoints[index].Position;
	//vertex.Time     = c_lineInfo.pCheckpoints[index].Time;
	
	if (vertex.Time >= c_integrationParams.timeMax || c_volumeInfo.isOutsideOfDomain(vertex.Position))
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
	if (!findBrickTime(	vertex.Position, vertex.Time,
						brickBoxMin, brickBoxMax, world2texOffset, world2texScale,
						brickTimeMin, brickTimeMax, time2texOffset, time2texScale))
	{
		return;
	}
	//assert(vertex.Time < brickTimeMax);


	const float timestepInc = (float)c_volumeInfo.brickSizeVoxelsWithOverlap;

	// get velocity at initial position
	//float4 vel4 = sampleVolume<filterMode, float4, float4>(g_texVolume1, w2t(vertex.Position));
	//float3 velocity = c_volumeInfo.velocityScale * make_float3(vel4.x, vel4.y, vel4.z);

	float3 velocity = c_volumeInfo.velocityScale * sampleVolume<filterMode, float4, float3>(g_texVolume1, w2t(vertex.Position));

	//vertex.LineID = index;


	//LineVertex* pVertices = c_lineInfo.pVertices + index * c_lineInfo.vertexStride;
	//LineVertex* pVertices = c_lineInfo.pVertices + index * c_lineInfo.vertexStride + lineLength;

	//if(lineLength == 0) {
	//	// new line - build normal: arbitrary vector perpendicular to velocity
	//	float3 tangent = normalize(velocity);
	//	vertex.Normal = cross(tangent, make_float3(1.0f, 0.0f, 0.0f));
	//	if(length(vertex.Normal) < 0.01f) vertex.Normal = cross(tangent, make_float3(0.0f, 1.0f, 0.0f));
	//	vertex.Normal = normalize(vertex.Normal);
	//	vertex.Jacobian = getJacobian<filterMode>(g_texVolume1, w2t(vertex.Position), c_integrationParams.gridSpacing);
	//	float3 gradT = sampleScalarGradient<filterMode>(g_texVolume1, w2t(vertex.Position), c_integrationParams.gridSpacing);
	//	vertex.Heat = vel4.w;
	//	vertex.HeatCurrent = gradT;

	//	// write out initial vertex
	//	*(pVertices + 1) = vertex;
	//	//++lineLength;
	//	lineLength = 1;
	//} else {
	//	// existing line - get old normal
	//	vertex.Normal = c_lineInfo.pCheckpoints[index].Normal;
	//}

	// get the last vertex that was written out
	//float3 lastOutPos  = (pVertices + 1)->Position;
	//float  lastOutTime = (pVertices + 1)->Time;

	bool stayedInAvailableBrick = true;
	float deltaTime = vertex.DeltaT;
	uint step = 0;
	uint stepsAccepted = 0;

	while (step < c_integrationParams.stepCountMax && vertex.Time < brickTimeMax)
	{
		float deltaTimeBak = deltaTime;
		// limit deltaTime ..
		// .. so we don't integrate past timeMax
		deltaTime = min(deltaTime, brickTimeMax - vertex.Time);
		// .. so we don't leave the current brick's safe region
		float distMax = c_integrationParams.brickSafeMarginWorld + distanceToBrickBorder(vertex.Position, brickBoxMin, brickBoxMax);
		deltaTime = min(deltaTime, distMax / c_integrationParams.velocityMaxWorld);
		
		// integrate
		bool stepAccepted = advectTime<advectMode, filterMode>(	g_texVolume1,
																vertex.Position, vertex.Time, velocity,
																deltaTime,
																world2texOffset, world2texScale,
																time2texOffset, time2texScale, timestepInc,
																c_volumeInfo.velocityScale);

		//bool stepAccepted = false;
	
		++step;
		if(stepAccepted) 
		{
			++stepsAccepted;

			// if we artificially limited deltaTime earlier, reset it now
			// (if we didn't, the new deltaTime is larger than the backup anyway)
			deltaTime = fmax(deltaTime, deltaTimeBak);

			// re-orthogonalize normal wrt. tangent == velocity direction
			//float3 binormal = cross(velocity, vertex.Normal);
			//vertex.Normal = normalize(cross(binormal, velocity));

			//float3 posDiff = vertex.Position - lastOutPos;
			//float timeDiff = vertex.Time     - lastOutTime;
			//if((dot(posDiff, posDiff) >= c_integrationParams.outputPosDiffSquared) || (timeDiff >= c_integrationParams.outputTimeDiff)) 
			//{
			//	//get jacobian and heat for measures
			//	vertex.Jacobian = getJacobian<filterMode>(g_texVolume1, w2t(vertex.Position), c_integrationParams.gridSpacing);
			//	vel4 = sampleVolume<filterMode, float4, float4>(g_texVolume1, w2t(vertex.Position));
			//	float3 gradT = sampleScalarGradient<filterMode>(g_texVolume1, w2t(vertex.Position), c_integrationParams.gridSpacing);
			//	vertex.Heat = vel4.w;
			//	vertex.HeatCurrent = gradT;
			//	
			//	// write out (intermediate) result
			//	*pVertices = *(pVertices + 1);
			//	*(pVertices + 1) = vertex;
			//	//*pVertices++ = vertex;
			//	//++lineLength;
			//	lineLength = 2;

			//	lastOutPos  = vertex.Position;
			//	lastOutTime = vertex.Time;
			//}

			// check if we left the current brick
			if (!isInBrickTime(vertex.Position, vertex.Time, brickBoxMin, brickBoxMax, brickTimeMin, brickTimeMax)) 
			{
				bool isOutOfDomain = c_volumeInfo.isOutsideOfDomain(vertex.Position);
				if (isOutOfDomain || !findBrickTime(	vertex.Position, vertex.Time,
														brickBoxMin, brickBoxMax, world2texOffset, world2texScale,
														brickTimeMin, brickTimeMax, time2texOffset, time2texScale))
				{
					// new brick isn't available (or we went out of the domain) - get outta here
					// (if we're still inside the domain, the new brick has already been requested in findBrickTime!)
					stayedInAvailableBrick = false;
					break;
				} 
				else 
				{
					// semi-HACK: update velocity from new brick (can be different to previous one because of lossy compression)
					//            this avoids excessively small time steps at some brick boundaries
					velocity = c_volumeInfo.velocityScale * sampleVolumeTime<filterMode, float4, float3>(g_texVolume1, w2t(vertex.Position), time2tex(vertex.Time), timestepInc);
				}
			}
		}
	}
	
	//c_lineInfo.pVertexCounts[index] = lineLength;
	//assert(c_lineInfo.pVertexCounts[index] < lineLengthMax);

	// update checkpoint for next integration round
	//c_lineInfo.pCheckpoints[index].Position = vertex.Position;
	//c_lineInfo.pCheckpoints[index].Time     = vertex.Time;
	//c_lineInfo.pCheckpoints[index].Normal   = vertex.Normal;
	//c_lineInfo.pCheckpoints[index].DeltaT   = deltaTime;

	//c_lineInfo.pCheckpoints[index].StepsAccepted += stepsAccepted;
	//c_lineInfo.pCheckpoints[index].StepsTotal    += step;

	vertex.DeltaT = deltaTime;

	dpParticles[index] = vertex;

	// if the line is still alive and in an available brick, request it again for next round
	if (vertex.Time < c_integrationParams.timeMax && stayedInAvailableBrick)
	{
		// find out which brick we're in now
		uint3 brickIndex = c_volumeInfo.getBrickIndex(vertex.Position);
		uint brickLinearIndex = c_volumeInfo.getBrickLinearIndex(brickIndex);
		uint timestepIndex = c_volumeInfo.getFloorTimestepIndex(vertex.Time);
		// request it to be loaded
		c_brickRequests.requestBrickTime(brickLinearIndex, timestepIndex);
	}
}


template<eAdvectMode advectMode, eTextureFilterMode filterMode>
__global__ void computeStreamFTLEKernel(SimpleParticleVertexDeltaT* dpParticles, uint particleCount, bool invertVelocity)
{
	uint index = blockIdx.x * blockDim.x + threadIdx.x;

	if (index >= particleCount)
		return;

	//uint lineLength = c_lineInfo.pVertexCounts[lineIndex];

	//if (lineLength >= c_lineInfo.lineLengthMax)
	//	return;

	SimpleParticleVertexDeltaT vertex = dpParticles[index];

	//LineVertex vertex;
	//// get initial position from checkpoints array
	//vertex.Position = c_lineInfo.pCheckpoints[lineIndex].Position;
	//vertex.Time = c_lineInfo.pCheckpoints[lineIndex].Time;
	//vertex.SeedPosition = c_lineInfo.pCheckpoints[lineIndex].SeedPosition;

	if (vertex.Time >= c_integrationParams.timeMax || c_volumeInfo.isOutsideOfDomain(vertex.Position))
		return;


	// find brick we're in
	float3 brickBoxMin;
	float3 brickBoxMax;
	float3 world2texOffset;
	float3 world2texScale;
	if (!findBrick(vertex.Position, brickBoxMin, brickBoxMax, world2texOffset, world2texScale)) {
		return;
	}

	// get velocity at initial position
	//float4 vel4 = sampleVolume<filterMode, float4, float4>(g_texVolume1, w2t(vertex.Position));
	//vertex.Velocity = c_volumeInfo.velocityScale * make_float3(vel4.x, vel4.y, vel4.z);

	float3 velocityScale = (invertVelocity ? -1.0 : 1.0) * c_volumeInfo.velocityScale;
	float3 velocity = velocityScale * sampleVolume<filterMode, float4, float3>(g_texVolume1, w2t(vertex.Position));

	//vertex.LineID = lineIndex;


	//LineVertex* pVertices = c_lineInfo.pVertices + lineIndex * c_lineInfo.vertexStride + lineLength;

	//if (lineLength == 0) {
	//	// new line - build normal: arbitrary vector perpendicular to velocity
	//	float3 tangent = normalize(vertex.Velocity);
	//	vertex.Normal = cross(tangent, make_float3(1.0f, 0.0f, 0.0f));
	//	if (length(vertex.Normal) < 0.01f) vertex.Normal = cross(tangent, make_float3(0.0f, 1.0f, 0.0f));
	//	vertex.Normal = normalize(vertex.Normal);
	//	vertex.Jacobian = getJacobian<filterMode>(g_texVolume1, w2t(vertex.Position), c_integrationParams.gridSpacing);
	//	float3 gradT = sampleScalarGradient<filterMode>(g_texVolume1, w2t(vertex.Position), c_integrationParams.gridSpacing);
	//	vertex.Heat = vel4.w;
	//	vertex.HeatCurrent = gradT;

	//	// write out initial vertex
	//	*pVertices++ = vertex;
	//	++lineLength;
	//	//printf("line=%d, index=0: pos=(%f, %f, %f), temp=%f\n", lineIndex, vertex.Position.x, vertex.Position.y, vertex.Position.z, vertex.Heat);
	//}
	//else {
	//	// existing line - get old normal
	//	vertex.Normal = c_lineInfo.pCheckpoints[lineIndex].Normal;
	//}

	//// get the last vertex that was written out
	//float3 lastOutPos = (pVertices - 1)->Position;
	//float  lastOutTime = (pVertices - 1)->Time;

	//float deltaTime = c_lineInfo.pCheckpoints[lineIndex].DeltaT;
	bool stayedInAvailableBrick = true;
	float deltaTime = vertex.DeltaT;
	uint step = 0;
	uint stepsAccepted = 0;

	while (step < c_integrationParams.stepCountMax && vertex.Time < c_integrationParams.timeMax)
	{
		float deltaTimeBak = deltaTime;
		// limit deltaTime ..
		// .. so we don't integrate past timeMax
		deltaTime = min(deltaTime, c_integrationParams.timeMax - vertex.Time);
		// .. so we don't leave the current brick's safe region
		float distMax = c_integrationParams.brickSafeMarginWorld + distanceToBrickBorder(vertex.Position, brickBoxMin, brickBoxMax);
		deltaTime = min(deltaTime, distMax / c_integrationParams.velocityMaxWorld);

		// integrate
		bool stepAccepted = advect<advectMode, filterMode>(	g_texVolume1,
															vertex.Position, vertex.Time, velocity,
															deltaTime,
															world2texOffset, world2texScale,
															velocityScale);

		++step;
		if (stepAccepted)
		{
			++stepsAccepted;

			// if we artificially limited deltaTime earlier, reset it now
			// (if we didn't, the new deltaTime is larger than the backup anyway)
			deltaTime = fmax(deltaTime, deltaTimeBak);

			// re-orthogonalize normal wrt. tangent == velocity direction
			//float3 binormal = cross(vertex.Velocity, vertex.Normal);
			//vertex.Normal = normalize(cross(binormal, vertex.Velocity));

			//float3 posDiff = vertex.Position - lastOutPos;
			//float timeDiff = vertex.Time - lastOutTime;
			//if ((dot(posDiff, posDiff) >= c_integrationParams.outputPosDiffSquared) || (timeDiff >= c_integrationParams.outputTimeDiff)) {
			//	//get jacobian and heat for measures
			//	vertex.Jacobian = getJacobian<filterMode>(g_texVolume1, w2t(vertex.Position), c_integrationParams.gridSpacing);
			//	vel4 = sampleVolume<filterMode, float4, float4>(g_texVolume1, w2t(vertex.Position));
			//	float3 gradT = sampleScalarGradient<filterMode>(g_texVolume1, w2t(vertex.Position), c_integrationParams.gridSpacing);
			//	vertex.Heat = vel4.w;
			//	vertex.HeatCurrent = gradT;

			//	// write out (intermediate) result
			//	*pVertices++ = vertex;
			//	++lineLength;
			//	//printf("line=%d, index=%d: pos=(%f, %f, %f), temp=%f\n", lineIndex, lineLength, vertex.Position.x, vertex.Position.y, vertex.Position.z, vertex.Heat);

			//	lastOutPos = vertex.Position;
			//	lastOutTime = vertex.Time;
			//}

			// check if we left the current brick
			if (!isInBrick(vertex.Position, brickBoxMin, brickBoxMax))
			{
				bool isOutOfDomain = c_volumeInfo.isOutsideOfDomain(vertex.Position);

				//if (isOutOfDomain) 
				//{
				//	// write out final position
				//	*pVertices++ = vertex;
				//	++lineLength;

				//	lastOutPos = vertex.Position;
				//	lastOutTime = vertex.Time;
				//}

				if (isOutOfDomain || !findBrick(vertex.Position, brickBoxMin, brickBoxMax, world2texOffset, world2texScale))
				{
					// new brick isn't available (or we went out of the domain) - get outta here
					// (if we're still inside the domain, the new brick has already been requested in findBrick!)
					stayedInAvailableBrick = false;
					break;
				}
				else
				{
					// semi-HACK: update velocity from new brick (can be different to previous one because of lossy compression)
					//            this avoids excessively small time steps at some brick boundaries
					velocity = velocityScale * sampleVolume<filterMode, float4, float3>(g_texVolume1, w2t(vertex.Position));
				}
			}
		}
	}


	//c_lineInfo.pVertexCounts[lineIndex] = lineLength;
	////assert(c_lineInfo.pVertexCounts[lineIndex] < lineLengthMax);

	//// update checkpoint for next integration round
	//c_lineInfo.pCheckpoints[lineIndex].Position = vertex.Position;
	//c_lineInfo.pCheckpoints[lineIndex].Time = vertex.Time;
	//c_lineInfo.pCheckpoints[lineIndex].Normal = vertex.Normal;
	//c_lineInfo.pCheckpoints[lineIndex].DeltaT = deltaTime;

	//c_lineInfo.pCheckpoints[lineIndex].StepsAccepted += stepsAccepted;
	//c_lineInfo.pCheckpoints[lineIndex].StepsTotal += step;

	vertex.DeltaT = deltaTime;

	dpParticles[index] = vertex;

	// if the line is still alive and in an available brick, request it again for next round
	if (vertex.Time < c_integrationParams.timeMax && stayedInAvailableBrick)
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


void integratorKernelComputeFTLE(SimpleParticleVertexDeltaT* dpParticles, uint particleCount, eAdvectMode advectMode, eTextureFilterMode filterMode, bool invertVelocity)
{
	uint blockSize = 128;
	uint blockCount = (particleCount + blockSize - 1) / blockSize;

#define INTEGRATE(advect, filter) computeStreamFTLEKernel <advect, filter> <<<blockCount, blockSize>>> (dpParticles, particleCount, invertVelocity)

	ADVECT_SWITCH;
	cudaCheckMsg("integratePathLinesKernel execution failed");

#undef INTEGRATE
}
