#ifndef __TUM3D__BRICK_REQUESTS_GPU_H__
#define __TUM3D__BRICK_REQUESTS_GPU_H__


#include <global.h>

#ifndef __CUDACC__
#include <Windows.h>
#endif


struct BrickRequestsGPU
{
	// Can't have constructors on the device...
	void Init() { dpBrickRequestCount = nullptr; dpBrickTimestepMin = nullptr; pBrickRequestCount = nullptr; };

	// Alloc/dealloc arrays in GPU global memory.
	void Allocate(bool cpuTracing, uint brickCount);
	void Deallocate();
	// Upload this instance to the GPU's global instance in constant memory.
	// Assumes that this instance is in pinned memory.
	// Does not sync on the upload, so don't overwrite any members without syncing first!
	void Upload(bool cpuTracing) const;
	// Clear (part of) the GPU arrays.
	void Clear(bool cpuTracing, uint brickCount, uint offset = 0);
	// Download the arrays from GPU memory (asynchronously).
	// Output pointers may be null.
	// Output memory must be page-locked.
	// Caller is responsible for syncing as necessary!
	// CPU version: pBrickTimestepMin not supported atm!
	void Download(bool cpuTracing, uint* pBrickRequestCount, uint* pBrickTimestepMin, uint brickCount);


	// number of requests per brick (indexed by linear brick index)
	uint* dpBrickRequestCount;
	// minimum requested timestep per brick (indexed by linear brick index)
	uint* dpBrickTimestepMin;


	// for CPU tracing
	uint* pBrickRequestCount;


#ifdef __CUDACC__
	__device__ inline void requestBrick(uint linearIndex)
	{
		atomicAdd(dpBrickRequestCount + linearIndex, 1);
	}

	__device__ inline void requestBrickTime(uint linearIndex, uint timestep)
	{
		atomicAdd(dpBrickRequestCount + linearIndex, 1);
		atomicMin(dpBrickTimestepMin + linearIndex, timestep);
	}
#else
	inline void requestBrickCPU(uint linearIndex)
	{
		//pBrickRequestCount[linearIndex]++;
		static_assert(sizeof(LONG) == sizeof(uint), "InterlockedIncrement only works on LONG, not uint :(");
		InterlockedIncrement(reinterpret_cast<LONG*>(pBrickRequestCount + linearIndex));
	}
#endif
};


#endif
