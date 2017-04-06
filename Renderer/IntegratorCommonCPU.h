#ifndef __TUM3D__INTEGRATOR_COMMON_CPU_H__
#define __TUM3D__INTEGRATOR_COMMON_CPU_H__


#include <vector_types.h>

#include "AdvectMode.h"
#include "BrickIndexGPU.h"
#include "BrickRequestsGPU.h"
#include "VolumeInfoGPU.h"

#include "cutil_math.h"


extern VolumeInfoGPU g_volumeInfo;
extern BrickIndexGPU g_brickIndex;
extern BrickRequestsGPU g_brickRequests;


inline bool isInBrick(float3 position, float3 brickPosMin, float3 brickPosMax)
{
	return position.x >= brickPosMin.x && position.x <= brickPosMax.x
		&& position.y >= brickPosMin.y && position.y <= brickPosMax.y
		&& position.z >= brickPosMin.z && position.z <= brickPosMax.z;
}


inline float distanceToBrickBorder(float3 position, float3 brickPosMin, float3 brickPosMax)
{
	float3 pos = brickPosMax - position;
	float3 neg = position - brickPosMin;
	return fmin(fmin(fmin(pos.x, pos.y), pos.z), fmin(fmin(neg.x, neg.y), neg.z));
}


inline bool findBrick(float3 worldPos, float3& brickBoxMin, float3& brickBoxMax, float3& world2texOffset, float3& world2texScale)
{
	// find out which brick we're in
	uint3 brickIndex = g_volumeInfo.getBrickIndex(worldPos);
	uint brickLinearIndex = g_volumeInfo.getBrickLinearIndex(brickIndex);

	// is our brick available on the GPU?
	uint2 slotIndex = g_brickIndex.pBrickToSlot[brickLinearIndex];
	if(slotIndex.x == g_brickIndex.INVALID) {
		// brick isn't here - request it to be loaded
		g_brickRequests.requestBrickCPU(brickLinearIndex);
		return false;
	}

	// get brick box and world2tex transform
	g_volumeInfo.getBrickBox(brickIndex, brickBoxMin, brickBoxMax);
	g_volumeInfo.computeWorld2Tex(brickBoxMin, world2texOffset, world2texScale);
	// brick slots are stacked in y direction (x direction is different time steps!)
	world2texOffset.y += slotIndex.x * g_volumeInfo.brickSizeVoxelsWithOverlap / world2texScale.x;
	world2texOffset.z += slotIndex.y * g_volumeInfo.brickSizeVoxelsWithOverlap / world2texScale.y;

	return true;
}


#endif
