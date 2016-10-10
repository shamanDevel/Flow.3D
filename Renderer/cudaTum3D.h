#ifndef __TUM3D__CUDA_TUM3D_H__
#define __TUM3D__CUDA_TUM3D_H__


#include <global.h>

#include <Vec.h>

#include "cutil_math.h"


inline int2 make_int2(const tum3D::Vec2i& vec)
{
	return make_int2(vec.x(), vec.y());
}

inline int3 make_int3(const tum3D::Vec3i& vec)
{
	return make_int3(vec.x(), vec.y(), vec.z());
}

inline int4 make_int4(const tum3D::Vec4i& vec)
{
	return make_int4(vec.x(), vec.y(), vec.z(), vec.w());
}

inline uint2 make_uint2(const tum3D::Vec2ui& vec)
{
	return make_uint2(vec.x(), vec.y());
}

inline uint3 make_uint3(const tum3D::Vec3ui& vec)
{
	return make_uint3(vec.x(), vec.y(), vec.z());
}

inline uint4 make_uint4(const tum3D::Vec4ui& vec)
{
	return make_uint4(vec.x(), vec.y(), vec.z(), vec.w());
}

inline float2 make_float2(const tum3D::Vec2f& vec)
{
	return make_float2(vec.x(), vec.y());
}

inline float3 make_float3(const tum3D::Vec3f& vec)
{
	return make_float3(vec.x(), vec.y(), vec.z());
}

inline float4 make_float4(const tum3D::Vec4f& vec)
{
	return make_float4(vec.x(), vec.y(), vec.z(), vec.w());
}

inline float3x4 make_float3x4(const tum3D::Mat4f& mat)
{
	float3x4 result = { make_float4(mat.getRow(0)), make_float4(mat.getRow(1)), make_float4(mat.getRow(2)) };
	return result;
}


#endif
