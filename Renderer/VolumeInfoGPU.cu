#include "VolumeInfoGPU.h"

#include <cuda_runtime.h>

#include <TimeVolumeInfo.h>

#include "cudaTum3D.h"
#include "cudaUtil.h"

using namespace tum3D;


//__constant__ VolumeInfoGPU c_volumeInfo;
VolumeInfoGPU g_volumeInfo;


void VolumeInfoGPU::Fill(const TimeVolumeInfo& info)
{
	volumeSizeVoxels    = make_uint3(make_int3(info.GetVolumeSize()));
	volumeHalfSizeWorld = make_float3(info.GetVolumeHalfSizeWorld());

	gridSpacing = make_float3(info.GetGridSpacing());
	timeSpacing = info.GetTimeSpacing();

	brickSizeVoxelsWithOverlap = info.GetBrickSizeWithOverlap();
	brickSizeWorld             = make_float3(info.GetBrickSizeWorld());
	brickOverlapWorld          = make_float3(info.GetBrickOverlapWorld());
	brickCount                 = make_uint3(make_int3(info.GetBrickCount()));

	velocityScale = make_float3(info.GetPhysicalToWorldFactor());
}

void VolumeInfoGPU::Upload(bool cpuTracing) const
{
	if(cpuTracing)
		memcpy(&g_volumeInfo, this, sizeof(g_volumeInfo));
	//else
	//	cudaSafeCall(cudaMemcpyToSymbolAsync(c_volumeInfo, this, sizeof(*this), 0, cudaMemcpyHostToDevice));
}
