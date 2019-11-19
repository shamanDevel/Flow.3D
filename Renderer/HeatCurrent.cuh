#ifndef __TUM3D__HEAT_CURRENT_CUH__
#define __TUM3D__HEAT_CURRENT_CUH__


#include "TextureFilterMode.h"

#include "MatrixMath.cuh"
#include "TextureFilter.cuh"


/******************************************************************************
** Heat current and derived quantities
******************************************************************************/


// get heat current from velocity/temperature texture
template <eTextureFilterMode F>
__device__ inline float3 getHeatCurrent(cudaTextureObject_t tex, const float3& texCoord, const float3& h)
{
	float4 velT = sampleVolume<F, float4, float4>(tex, texCoord);
	float3 gradT = sampleScalarGradient<F>(tex, texCoord, h);

	float3 j;

	float ra = 7e5;
	float pr = 0.7;
	float kappa = 1 / sqrt(ra*pr);

	j.x = velT.x * velT.w - kappa * gradT.x;
	j.y = velT.y * velT.w - kappa * gradT.y;
	j.z = velT.z * velT.w - kappa * gradT.z;

	return j;
}


template <eTextureFilterMode F>
__device__ inline float getHeatCurrentAlignment(cudaTextureObject_t tex, const float3& texCoord, const float& h)
{
	float4 velT = sampleVolume<F, float4, float4>(tex, texCoord);
	float3 gradT = sampleScalarGradient<F>(tex, texCoord, h);

	float3 j;

	float ra = 7e5;
	float pr = 0.7;
	float kappa = 1 / sqrt(ra*pr);

	j.x = velT.x * velT.w - kappa * gradT.x;
	j.y = velT.y * velT.w - kappa * gradT.y;
	j.z = velT.z * velT.w - kappa * gradT.z;

	j = normalize(j);
	float3 vel = normalize(make_float3(velT));

	return dot(j, vel);
}



#endif
