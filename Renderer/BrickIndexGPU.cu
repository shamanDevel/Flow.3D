#include "BrickIndexGPU.h"

#include <cuda_runtime.h>

#include "cudaUtil.h"

#include <iostream>

//__constant__ BrickIndexGPU c_brickIndex;
BrickIndexGPU g_brickIndex;


void BrickIndexGPU::Allocate(bool cpuTracing, uint brickCountNew, uint2 slotCountNew)
{
	brickCount = brickCountNew;
	slotCount = slotCountNew;

	if(cpuTracing)
	{
		pBrickToSlot = new uint2[brickCount];
	}
	else
	{
		cudaSafeCall(cudaMalloc2(&dpBrickToSlot, brickCount * sizeof(uint2)));
		cudaSafeCall(cudaMalloc2(&dpSlotTimestepMin, slotCount.x * slotCount.y * sizeof(uint)));
		cudaSafeCall(cudaMalloc2(&dpSlotTimestepMax, slotCount.x * slotCount.y * sizeof(uint)));
	}
}

void BrickIndexGPU::Deallocate()
{
	if(dpSlotTimestepMax)
	{
		cudaSafeCall(cudaFree(dpSlotTimestepMax));
		dpSlotTimestepMax = nullptr;
	}
	if(dpSlotTimestepMin)
	{
		cudaSafeCall(cudaFree(dpSlotTimestepMin));
		dpSlotTimestepMin = nullptr;
	}
	if(dpBrickToSlot)
	{
		cudaSafeCall(cudaFree(dpBrickToSlot));
		dpBrickToSlot = nullptr;
	}

	delete[] pBrickToSlot;
	pBrickToSlot = nullptr;
}

void BrickIndexGPU::Update(bool cpuTracing, const uint2* pBrickToSlot, const uint* pSlotTimestepMin, const uint* pSlotTimestepMax)
{
	if(cpuTracing)
	{
		memcpy(this->pBrickToSlot, pBrickToSlot, brickCount * sizeof(uint2));
	}
	else
	{
		cudaSafeCall(cudaMemcpyAsync(dpBrickToSlot, pBrickToSlot, brickCount * sizeof(uint2), cudaMemcpyHostToDevice));
		if(pSlotTimestepMin != nullptr)
			cudaSafeCall(cudaMemcpyAsync(dpSlotTimestepMin, pSlotTimestepMin, slotCount.x * slotCount.y * sizeof(uint), cudaMemcpyHostToDevice));
		if(pSlotTimestepMax != nullptr)
			cudaSafeCall(cudaMemcpyAsync(dpSlotTimestepMax, pSlotTimestepMax, slotCount.x * slotCount.y * sizeof(uint), cudaMemcpyHostToDevice));
	}
}

void BrickIndexGPU::Upload(bool cpuTracing) const
{
	if(cpuTracing)
		memcpy(&g_brickIndex, this, sizeof(g_brickIndex));
	//else
	//	cudaSafeCall(cudaMemcpyToSymbolAsync(c_brickIndex, this, sizeof(*this), 0, cudaMemcpyHostToDevice));
}
