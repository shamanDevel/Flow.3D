#include "LineInfoGPU.h"

#include <cuda_runtime.h>

#include "cudaUtil.h"


__constant__ LineInfoGPU c_lineInfo;
LineInfoGPU g_lineInfo;


void LineInfoGPU::Upload(bool cpuTracing) const
{
	if(cpuTracing)
	{
		memcpy(&g_lineInfo, this, sizeof(g_lineInfo));
	}
	else
	{
		cudaSafeCall(cudaMemcpyToSymbolAsync(c_lineInfo, this, sizeof(*this), 0, cudaMemcpyHostToDevice));
	}
}
