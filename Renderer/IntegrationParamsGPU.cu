#include "IntegrationParamsGPU.h"

#include <cuda_runtime.h>

#include "cudaUtil.h"


__constant__ IntegrationParamsGPU c_integrationParams;
IntegrationParamsGPU g_integrationParams;


void IntegrationParamsGPU::Upload(bool cpuTracing) const
{
	if(cpuTracing)
	{
		memcpy(&g_integrationParams, this, sizeof(g_integrationParams));
	}
	else
	{
		cudaSafeCall(cudaMemcpyToSymbolAsync(c_integrationParams, this, sizeof(*this), 0, cudaMemcpyHostToDevice));
	}
}
