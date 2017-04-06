#ifndef __TUM3D__VOLUME_INFO_GPU_H__
#define __TUM3D__VOLUME_INFO_GPU_H__


#include <global.h>

#include <vector_types.h>

#include "cutil_math.h"


class TimeVolumeInfo;


struct VolumeInfoGPU
{
	// fill all members from the given TimeVolumeInfo
	void Fill(const TimeVolumeInfo& info);
	// upload this instance to the GPU's global instance in constant memory
	// assumes that this instance is in pinned memory
	// does not sync on the upload, so don't overwrite any members without syncing first!
	void Upload(bool cpuTracing) const;


	uint3  volumeSizeVoxels;
	float3 volumeHalfSizeWorld;

	float3 gridSpacing;
	float timeSpacing;

	uint  brickSizeVoxelsWithOverlap;
	float3 brickSizeWorld;
	float3 brickOverlapWorld;
	uint3 brickCount;

	float3 velocityScale;


	__host__ __device__ inline bool isInsideOfDomain(float3 posWorld) const
	{
		return posWorld.x >= -volumeHalfSizeWorld.x && posWorld.y >= -volumeHalfSizeWorld.y && posWorld.z >= -volumeHalfSizeWorld.z
			&& posWorld.x <=  volumeHalfSizeWorld.x && posWorld.y <=  volumeHalfSizeWorld.y && posWorld.z <=  volumeHalfSizeWorld.z;
	}

	__host__ __device__ inline bool isOutsideOfDomain(float3 posWorld) const
	{
		return !isInsideOfDomain(posWorld);
	}

	__host__ __device__ inline uint3 getBrickIndex(float3 posWorld) const
	{
		return make_uint3((posWorld + volumeHalfSizeWorld) / brickSizeWorld);
	}

	__host__ __device__ inline uint getBrickLinearIndex(uint3 brickIndex) const
	{
		return brickIndex.x + brickCount.x * (brickIndex.y + brickCount.y * brickIndex.z);
	}

	__host__ __device__ inline void getBrickBox(uint3 brickIndex, float3& boxMin, float3& boxMax) const
	{
		boxMin = -volumeHalfSizeWorld + make_float3(brickIndex) * brickSizeWorld;
		boxMax = boxMin + brickSizeWorld;
		//TODO add epsilon = brickSizeWorld * 0.0001f ?
		boxMax = fmin(boxMax, volumeHalfSizeWorld); // clamp against global volume box
	}

	__host__ __device__ inline void computeWorld2Tex(float3 brickBoxMin, float3& world2texOffset, float3& world2texScale) const
	{
		world2texOffset = -brickBoxMin + brickOverlapWorld;
		world2texScale  = brickSizeVoxelsWithOverlap / (brickSizeWorld + 2.0f * brickOverlapWorld);
	}

	__host__ __device__ inline uint getFloorTimestepIndex(float time) const
	{
		return uint(time / timeSpacing);
	}

	__host__ __device__ inline void getBrickTime(uint timestepMin, uint timestepMax, float& brickTimeMin, float& brickTimeMax) const
	{
		brickTimeMin = float(timestepMin) * timeSpacing;
		brickTimeMax = float(timestepMax) * timeSpacing;
	}

	__host__ __device__ inline void computeTime2Tex(float timeMin, float& time2texOffset, float& time2texScale) const
	{
		time2texOffset = -timeMin;
		time2texScale  = 1.0f / timeSpacing;
	}
};


#endif
