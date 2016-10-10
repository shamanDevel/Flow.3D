#ifndef __TUM3D__TEXTURE_FILTER_TIME_CUH__
#define __TUM3D__TEXTURE_FILTER_TIME_CUH__


#include "TextureFilter.cuh"


template <eTextureFilterMode F, typename TexType, typename ResultType>
__device__ inline ResultType sampleVolumeTime(texture<TexType, cudaTextureType3D, cudaReadModeElementType> tex, float x, float y, float z, float t, float timestepInc)
{
	float t0 = floor(t);
	float t1 = t0 + 1.0f;
	ResultType val0 = sampleVolume<F, TexType, ResultType>(tex, t0 * timestepInc + x, y, z);
	ResultType val1 = sampleVolume<F, TexType, ResultType>(tex, t1 * timestepInc + x, y, z);
	return (t1 - t) * val0 + (t - t0) * val1;
}

template <eTextureFilterMode F, typename TexType, typename ResultType>
__device__ inline ResultType sampleVolumeTime(texture<TexType, cudaTextureType3D, cudaReadModeElementType> tex, float3 coord, float t, float timestepInc)
{
	return sampleVolumeTime<F, TexType, ResultType>(tex, coord.x, coord.y, coord.z, t, timestepInc);
}


#endif
