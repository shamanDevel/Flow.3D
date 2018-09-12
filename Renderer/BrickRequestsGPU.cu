#include "BrickRequestsGPU.h"

#include <cassert>

#include <cuda_runtime.h>

#include "cudaUtil.h"


__constant__ BrickRequestsGPU c_brickRequests;
BrickRequestsGPU g_brickRequests;


void BrickRequestsGPU::Allocate(bool cpuTracing, uint brickCount)
{
	assert(dpBrickRequestCount == nullptr);
	assert(dpBrickTimestepMin == nullptr);
	cudaSafeCall(cudaMalloc2(&dpBrickRequestCount, brickCount * sizeof(uint)));
	cudaSafeCall(cudaMalloc2(&dpBrickTimestepMin,  brickCount * sizeof(uint)));

	pBrickRequestCount = new uint[brickCount];
}

void BrickRequestsGPU::Deallocate()
{
	if(dpBrickRequestCount)
	{
		cudaSafeCall(cudaFree(dpBrickRequestCount));
		dpBrickRequestCount = nullptr;
	}
	if(dpBrickTimestepMin)
	{
		cudaSafeCall(cudaFree(dpBrickTimestepMin));
		dpBrickTimestepMin = nullptr;
	}

	delete[] pBrickRequestCount;
	pBrickRequestCount = nullptr;
}

void BrickRequestsGPU::Upload(bool cpuTracing) const
{
	if(cpuTracing)
	{
		memcpy(&g_brickRequests, this, sizeof(g_brickRequests));
	}
	else
	{
		cudaSafeCall(cudaMemcpyToSymbolAsync(c_brickRequests, this, sizeof(*this), 0, cudaMemcpyHostToDevice));
	}
}

void BrickRequestsGPU::Clear(bool cpuTracing, uint brickCount, uint offset)
{
	if(cpuTracing)
	{
		memset(pBrickRequestCount + offset, 0, brickCount * sizeof(uint));
	}
	else
	{
		cudaSafeCall(cudaMemsetAsync(dpBrickRequestCount + offset,  0u, brickCount * sizeof(uint)));
		cudaSafeCall(cudaMemsetAsync(dpBrickTimestepMin  + offset, ~0u, brickCount * sizeof(uint)));
	}
}

void BrickRequestsGPU::Download(bool cpuTracing, uint* pBrickRequestCount, uint* pBrickTimestepMin, uint brickCount)
{
	if(cpuTracing)
	{
		if(pBrickRequestCount != nullptr)
			memcpy(pBrickRequestCount, this->pBrickRequestCount, brickCount * sizeof(uint));
		if(pBrickTimestepMin != nullptr)
			printf("BrickRequestsGPU::CopyFromCPU warning: pBrickTimestepMin not supported\n");
	}
	else
	{
		if(pBrickRequestCount != nullptr)
			cudaSafeCall(cudaMemcpyAsync(pBrickRequestCount, dpBrickRequestCount, brickCount * sizeof(uint), cudaMemcpyDeviceToHost));
		if(pBrickTimestepMin != nullptr)
			cudaSafeCall(cudaMemcpyAsync(pBrickTimestepMin,  dpBrickTimestepMin,  brickCount * sizeof(uint), cudaMemcpyDeviceToHost));
	}
}
