#ifndef __TUM3D__INTEGRATOR_COMMON_CUH__
#define __TUM3D__INTEGRATOR_COMMON_CUH__


#include "AdvectMode.h"
#include "BrickIndexGPU.h"
#include "BrickRequestsGPU.h"
#include "VolumeInfoGPU.h"

//extern __constant__ VolumeInfoGPU c_volumeInfo;
extern __constant__ BrickIndexGPU c_brickIndex;
extern __constant__ BrickRequestsGPU c_brickRequests;


__device__ inline bool isInBrick(float3 position, float3 brickPosMin, float3 brickPosMax)
{
	return position.x >= brickPosMin.x && position.x <= brickPosMax.x
		&& position.y >= brickPosMin.y && position.y <= brickPosMax.y
		&& position.z >= brickPosMin.z && position.z <= brickPosMax.z;
}

__device__ inline bool isInBrickTime(float3 position, float time, float3 brickPosMin, float3 brickPosMax, float brickTimeMin, float brickTimeMax)
{
	return isInBrick(position, brickPosMin, brickPosMax) && time > brickTimeMin && time < brickTimeMax;
}


__device__ inline float distanceToBrickBorder(float3 position, float3 brickPosMin, float3 brickPosMax)
{
	float3 pos = brickPosMax - position;
	float3 neg = position - brickPosMin;
	return fmin(fmin(fmin(pos.x, pos.y), pos.z), fmin(fmin(neg.x, neg.y), neg.z));
}


__device__ inline bool isAdvectModeAdaptive(eAdvectMode advectMode)
{
	switch(advectMode)
	{
		case ADVECT_EULER:
			return false;
		case ADVECT_HEUN:
			return false;
		case ADVECT_RK3:
			return false;
		case ADVECT_RK4:
			return false;
		case ADVECT_BS32:
			return true;
		case ADVECT_RKF34:
			return true;
		case ADVECT_RKF45:
			return true;
		case ADVECT_RKF54:
			return true;
		case ADVECT_RK547M:
			return true;
		default:
			//assert(false);
			return false;
	}
}


__device__ inline bool findBrick(VolumeInfoGPU& c_volumeInfo, float3 worldPos, float3& brickBoxMin, float3& brickBoxMax, float3& world2texOffset, float3& world2texScale)
{
	// find out which brick we're in
	uint3 brickIndex = c_volumeInfo.getBrickIndex(worldPos);
	uint brickLinearIndex = c_volumeInfo.getBrickLinearIndex(brickIndex);

	// is our brick available on the GPU?
	uint2 slotIndex = c_brickIndex.dpBrickToSlot[brickLinearIndex];
	if(slotIndex.x == c_brickIndex.INVALID) {
		// brick isn't here - request it to be loaded
		c_brickRequests.requestBrick(brickLinearIndex);
		return false;
	}

	// get brick box and world2tex transform
	c_volumeInfo.getBrickBox(brickIndex, brickBoxMin, brickBoxMax);
	c_volumeInfo.computeWorld2Tex(brickBoxMin, world2texOffset, world2texScale);
	// brick slots are stacked in y direction (x direction is different time steps!)
	world2texOffset.y += slotIndex.x * c_volumeInfo.brickSizeVoxelsWithOverlap / world2texScale.y;
	world2texOffset.z += slotIndex.y * c_volumeInfo.brickSizeVoxelsWithOverlap / world2texScale.z;

	return true;
}


__device__ inline bool findBrickTime(VolumeInfoGPU& c_volumeInfo,
	float3 worldPos, float time,
	float3& brickBoxMin, float3& brickBoxMax, float3& world2texOffset, float3& world2texScale,
	float& brickTimeMin, float& brickTimeMax, float& time2texOffset, float& time2texScale)
{
	// find out which brick we're in
	uint3 brickIndex = c_volumeInfo.getBrickIndex(worldPos);
	uint brickLinearIndex = c_volumeInfo.getBrickLinearIndex(brickIndex);

	uint brickTimestepIndex = c_volumeInfo.getFloorTimestepIndex(time);

	// is our brick available on the GPU?
	uint2 slotIndex = c_brickIndex.dpBrickToSlot[brickLinearIndex];
	if(slotIndex.x == c_brickIndex.INVALID) {
		// brick isn't here - request it to be loaded
		c_brickRequests.requestBrickTime(brickLinearIndex, brickTimestepIndex);
		return false;
	}
	uint slotIndexLinear = slotIndex.x + c_brickIndex.slotCount.x * slotIndex.y;
	uint timestepMin = c_brickIndex.dpSlotTimestepMin[slotIndexLinear];
	uint timestepMax = c_brickIndex.dpSlotTimestepMax[slotIndexLinear];
	if(brickTimestepIndex < timestepMin || brickTimestepIndex + 1 > timestepMax) { // need two timesteps for interpolation!
		// required time steps aren't here
		c_brickRequests.requestBrickTime(brickLinearIndex, brickTimestepIndex);
		return false;
	}

	// get brick box and world2tex transform
	c_volumeInfo.getBrickBox(brickIndex, brickBoxMin, brickBoxMax);
	c_volumeInfo.computeWorld2Tex(brickBoxMin, world2texOffset, world2texScale);
	// brick slots are stacked in y direction (x direction is different time steps!)
	world2texOffset.y += slotIndex.x * c_volumeInfo.brickSizeVoxelsWithOverlap / world2texScale.x;
	world2texOffset.z += slotIndex.y * c_volumeInfo.brickSizeVoxelsWithOverlap / world2texScale.y;
	// get brick time interval and time2tex transform
	c_volumeInfo.getBrickTime(timestepMin, timestepMax, brickTimeMin, brickTimeMax);
	c_volumeInfo.computeTime2Tex(brickTimeMin, time2texOffset, time2texScale);

	return true;
}


#endif
