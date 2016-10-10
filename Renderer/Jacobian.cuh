#ifndef __TUM3D__JACOBIAN_CUH__
#define __TUM3D__JACOBIAN_CUH__


#include "TextureFilterMode.h"

#include "MatrixMath.cuh"
#include "TextureFilter.cuh"


/******************************************************************************
** Jacobian and derived tensors
******************************************************************************/


__device__ inline float3x3 getStrainRateTensor(const float3x3 &J)
{
	// S_ij = 1/2 (J_ij + J_ji)
	float3x3 S;

	S.m[0] = make_float3( (J.m[0].x+J.m[0].x)/2.0f, (J.m[0].y+J.m[1].x)/2.0f, (J.m[0].z+J.m[2].x)/2.0f );
	S.m[1] = make_float3( (J.m[1].x+J.m[0].y)/2.0f, (J.m[1].y+J.m[1].y)/2.0f, (J.m[1].z+J.m[2].y)/2.0f );
	S.m[2] = make_float3( (J.m[2].x+J.m[0].z)/2.0f, (J.m[2].y+J.m[1].z)/2.0f, (J.m[2].z+J.m[2].z)/2.0f );

	return S;
}



__device__ inline float3x3 getSpinTensor(const float3x3 &J)
{
	// O_ij = 1/2 (J_ij - J_ji)
	float3x3 O;

	O.m[0] = make_float3( (J.m[0].x-J.m[0].x)/2.0f, (J.m[0].y-J.m[1].x)/2.0f, (J.m[0].z-J.m[2].x)/2.0f );
	O.m[1] = make_float3( (J.m[1].x-J.m[0].y)/2.0f, (J.m[1].y-J.m[1].y)/2.0f, (J.m[1].z-J.m[2].y)/2.0f );
	O.m[2] = make_float3( (J.m[2].x-J.m[0].z)/2.0f, (J.m[2].y-J.m[1].z)/2.0f, (J.m[2].z-J.m[2].z)/2.0f );

	return O;
}


// get jacobian from velocity texture
template <eTextureFilterMode F>
__device__ inline float3x3 getJacobian(texture<float4, cudaTextureType3D, cudaReadModeElementType> tex, const float3& texCoord, float h)
{
	return sampleVolumeGradient<F>(tex, texCoord, h);
}



#endif
