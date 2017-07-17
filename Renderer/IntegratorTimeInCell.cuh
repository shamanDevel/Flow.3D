#pragma once

#include <cuda_runtime.h>
#include <global.h>
#include "TracingCommon.h"
#include "cudaUtil.h"

// host code
struct CellTextureGPU
{
	size_t width;
	size_t height;
	cudaArray_t textureArray;
	CellTextureGPU() : width(0), height(0), textureArray(NULL) {}
};

class IntegratorTimeInCell
{
public:
	static void Upload(CellTextureGPU& info, uint32* textureMemCPU, size_t width, size_t height);

	static void Free(CellTextureGPU& info);

	// device code
	static __device__ void processParticle(LineVertex* vertex);
};


