#ifndef __TUM3D__BRICK_INDEX_GPU_H__
#define __TUM3D__BRICK_INDEX_GPU_H__


#include <global.h>

#include <vector_functions.h>
#include <vector_types.h>


struct BrickIndexGPU
{
	// can't have constructors on the device...
	void Init() { brickCount = 0; slotCount = make_uint2(0, 0); dpBrickToSlot = nullptr; dpSlotTimestepMin = nullptr; dpSlotTimestepMax = nullptr; pBrickToSlot = nullptr; };

	// alloc/dealloc arrays in GPU global memory
	void Allocate(bool cpuTracing, uint brickCountNew, uint2 slotCountNew);
	void Deallocate();
	// update the arrays in GPU memory (asynchronously)
	// input memory must be page-locked
	// caller is responsible for syncing as necessary!
	void Update(bool cpuTracing, const uint2* pBrickToSlot, const uint* pSlotTimestepMin, const uint* pSlotTimestepMax);
	// upload this instance to the GPU's global instance in constant memory
	// assumes that this instance is in pinned memory
	// does not sync on the upload, so don't overwrite any members without syncing first!
	void Upload(bool cpuTracing) const;


	static const uint INVALID = ~0u;

	uint brickCount;
	uint2 slotCount;

	// map from linear brick index to the index of the slot the brick is stored in,
	// or INVALID_SLOT if the brick isn't on the GPU
	uint2* dpBrickToSlot;
	// map from slot index to min/max timestep stored there
	uint* dpSlotTimestepMin;
	uint* dpSlotTimestepMax;


	// for CPU tracing
	uint2* pBrickToSlot;
};


#endif
