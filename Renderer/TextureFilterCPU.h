#ifndef __TUM3D__TEXTURE_FILTER_CPU_H__
#define __TUM3D__TEXTURE_FILTER_CPU_H__


#include <vector>

#include <vector_functions.h>
#include <vector_types.h>

#include "TextureCPU.h"
#include "TextureFilterMode.h"


///////////////////////////////////////////////////////////////////////////////
// Linear and cubic texture filter
///////////////////////////////////////////////////////////////////////////////

// helper to create all-zeros float<n>
template <typename TOut>
struct make_zero_floatn_impl { static inline TOut exec(); };
// no implementation - only specializations allowed:
template <>
struct make_zero_floatn_impl<float1> { static inline float1 exec() { return make_float1(0.0f); } };
template <>
struct make_zero_floatn_impl<float2> { static inline float2 exec() { return make_float2(0.0f, 0.0f); } };
template <>
struct make_zero_floatn_impl<float3> { static inline float3 exec() { return make_float3(0.0f, 0.0f, 0.0f); } };
template <>
struct make_zero_floatn_impl<float4> { static inline float4 exec() { return make_float4(0.0f, 0.0f, 0.0f, 0.0f); } };

template <typename TOut>
inline TOut make_zero_floatn()
{
	return make_zero_floatn_impl<TOut>::exec();
}

// helper to convert float<n> to float<m> (or just float)
template <typename TIn, typename TOut>
struct make_floatn_impl { static inline TOut exec(TIn val); };
// no implementation - only specializations allowed:
template <typename TIn>
struct make_floatn_impl<TIn, float>  { static inline float  exec(TIn val) { return val.x; } };
template <typename TIn>
struct make_floatn_impl<TIn, float1> { static inline float1 exec(TIn val) { return make_float1(val.x); } };
template <typename TIn>
struct make_floatn_impl<TIn, float2> { static inline float2 exec(TIn val) { return make_float2(val.x, val.y); } };
template <typename TIn>
struct make_floatn_impl<TIn, float3> { static inline float3 exec(TIn val) { return make_float3(val.x, val.y, val.z); } };
template <typename TIn>
struct make_floatn_impl<TIn, float4> { static inline float4 exec(TIn val) { return val; } };

template <typename TIn, typename TOut>
inline TOut make_floatn(TIn val)
{
	return make_floatn_impl<TIn, TOut>::exec(val);
}


// no interpolation here! (TEXTURE_FILTER_LINEAR does it manually, and the others don't use it anyway)
template<typename TexType>
inline TexType tex3D(const TextureCPU<TexType>& tex, float x, float y, float z)
{
	int3 coord = make_int3((int)floorf(x), (int)floorf(y), (int)floorf(z));
	coord = clamp(coord, make_int3(0, 0, 0), make_int3(tex.size) - 1);
	uint index = coord.x + tex.size.x * (coord.y + tex.size.y * coord.z);
	return tex.data[index];
}


namespace // sampleVolume implementations
{

template <eTextureFilterMode F, typename TexType, typename ResultType>
struct sampleVolume_impl
{
	//static ResultType exec(const TextureCPU<TexType>& tex, float x, float y, float z);
	// no exec() implementation - only specializations allowed
};


template <typename TexType, typename ResultType>
struct sampleVolume_impl<TEXTURE_FILTER_LINEAR, TexType, ResultType>
{
	static inline ResultType exec(const TextureCPU<TexType>& tex, float x, float y, float z)
	{
		float3 t = make_float3(x - floorf(x), y - floorf(y), z - floorf(z));
		ResultType v000 = make_floatn<TexType, ResultType>(tex3D(tex, x, y, z));
		ResultType v001 = make_floatn<TexType, ResultType>(tex3D(tex, x, y, z + 1.0f));
		ResultType v00 = (1.0f - t.z) * v000 + t.z * v001;
		ResultType v010 = make_floatn<TexType, ResultType>(tex3D(tex, x, y + 1.0f, z));
		ResultType v011 = make_floatn<TexType, ResultType>(tex3D(tex, x, y + 1.0f, z + 1.0f));
		ResultType v01 = (1.0f - t.z) * v010 + t.z * v011;
		ResultType v0 = (1.0f - t.y) * v00 + t.z * v01;
		ResultType v100 = make_floatn<TexType, ResultType>(tex3D(tex, x + 1.0f, y, z));
		ResultType v101 = make_floatn<TexType, ResultType>(tex3D(tex, x + 1.0f, y, z + 1.0f));
		ResultType v10 = (1.0f - t.z) * v000 + t.z * v001;
		ResultType v110 = make_floatn<TexType, ResultType>(tex3D(tex, x + 1.0f, y + 1.0f, z));
		ResultType v111 = make_floatn<TexType, ResultType>(tex3D(tex, x + 1.0f, y + 1.0f, z + 1.0f));
		ResultType v11 = (1.0f - t.z) * v010 + t.z * v011;
		ResultType v1 = (1.0f - t.y) * v00 + t.z * v01;
		return (1.0f - t.x) * v0 + t.x * v1;
	}
};


template <typename TexType, typename ResultType>
struct sampleVolume_impl<TEXTURE_FILTER_CATROM, TexType, ResultType>
{
	static inline ResultType exec(const TextureCPU<TexType>& tex, float x, float y, float z)
	{
		// shift the coordinate from [0,extent] to [-0.5, extent-0.5] (so that grid points are at integers)
		const float3 coord_grid = make_float3(x, y, z) - 0.5f;
		const float3 index = floor(coord_grid);
		const float3 t = coord_grid - index;

		// compute weights for grid points (cubic catmull-rom spline)
		const float3 t2 = t * t;
		const float3 t3 = t2 * t;

		const float3 w0 = 0.5f * (-t3 + 2.0f * t2 - t);
		const float3 w1 = 0.5f * (3.0f * t3 - 5.0f * t2 + 2.0f);
		const float3 w2 = 0.5f * (-3.0f * t3 + 4.0f * t2 + t);
		const float3 w3 = 0.5f * (t3 - t2);
		// note: can't use hardware linear interpolations for catmull-rom because the weights can be negative

		float3 texCoord;
		ResultType K[4];
		for(int k = -1; k <= 2; k++)
		{
			texCoord.z = index.z + k + 0.5f;
			ResultType J[4];
			for(int j = -1; j <= 2; j++)
			{
				texCoord.y = index.y + j + 0.5f;
				ResultType I[4];
				for(int i = -1; i <= 2; i++)
				{
					texCoord.x = index.x + i + 0.5f;
					I[i+1] = make_floatn<TexType, ResultType>(tex3D(tex, texCoord.x, texCoord.y, texCoord.z));
				}
				J[j+1] = w0.x * I[0] + w1.x * I[1] + w2.x * I[2] + w3.x * I[3];
			}
			K[k+1] = w0.y * J[0] + w1.y * J[1] + w2.y * J[2] + w3.y * J[3];
		}
		return w0.z * K[0] + w1.z * K[1] + w2.z * K[2] + w3.z * K[3];
	}
};


// 1D Lagrange4 interpolation
template<typename Type>
inline Type interpolateLagrange4(float t, Type v0, Type v1, Type v2, Type v3)
{
	// d_i = product for j from 0 to 3 and i!=j: (i-j)
	// d = [ -6, 2, -2, 6 ]

	// f_i = product for j from 0 to 3 and i!=j: (t+1-j) / d_i  [t specifies the position between v1 and v2 here!]
	float f0 =         (t) * (t-1) * (t-2) / -6.0f;
	float f1 = (t+1)       * (t-1) * (t-2) /  2.0f;
	float f2 = (t+1) * (t)         * (t-2) / -2.0f;
	float f3 = (t+1) * (t) * (t-1)         /  6.0f;

	return v0 * f0 + v1 * f1 + v2 * f2 + v3 * f3;
}

template <typename TexType, typename ResultType>
struct sampleVolume_impl<TEXTURE_FILTER_LAGRANGE4, TexType, ResultType>
{
	static inline ResultType exec(const TextureCPU<TexType>& tex, float x, float y, float z)
	{
		// shift the coordinate from [0,extent] to [-0.5, extent-0.5] (so that grid points are at integers)
		const float3 coord_grid = make_float3(x, y, z) - 0.5f;
		const float3 index = floor(coord_grid);
		const float3 t = coord_grid - index;

		float3 texCoord;
		ResultType K[4];
		for(int k = -1; k <= 2; k++)
		{
			texCoord.z = index.z + k + 0.5f;
			ResultType J[4];
			for(int j = -1; j <= 2; j++)
			{
				texCoord.y = index.y + j + 0.5f;
				ResultType I[4];
				for(int i = -1; i <= 2; i++)
				{
					texCoord.x = index.x + i + 0.5f;
					I[i+1] = make_floatn<TexType, ResultType>(tex3D(tex, texCoord.x, texCoord.y, texCoord.z));
				}
				J[j+1] = interpolateLagrange4(t.x, I[0], I[1], I[2], I[3]);
			}
			K[k+1] = interpolateLagrange4(t.y, J[0], J[1], J[2], J[3]);
		}
		return interpolateLagrange4(t.z, K[0], K[1], K[2], K[3]);
	}
};


// 1D Lagrange6 interpolation
template<typename Type>
inline Type interpolateLagrange6(float t, Type v0, Type v1, Type v2, Type v3, Type v4, Type v5)
{
	// d_i = product for j from 0 to 5 and i!=j: (i-j)
	// d = [ -120, 24, -12, 12, -24, 120 ]

	// f_i = product for j from 0 to 5 and i!=j: (t+2-j) / d_i  [t specifies the position between v2 and v3 here!]
	float f0 =         (t+1) * (t) * (t-1) * (t-2) * (t-3) / -120.0f;
	float f1 = (t+2)         * (t) * (t-1) * (t-2) * (t-3) /   24.0f;
	float f2 = (t+2) * (t+1)       * (t-1) * (t-2) * (t-3) /  -12.0f;
	float f3 = (t+2) * (t+1) * (t)         * (t-2) * (t-3) /   12.0f;
	float f4 = (t+2) * (t+1) * (t) * (t-1)         * (t-3) /  -24.0f;
	float f5 = (t+2) * (t+1) * (t) * (t-1) * (t-2)         /  120.0f;

	return v0 * f0 + v1 * f1 + v2 * f2 + v3 * f3 + v4 * f4 + v5 * f5;
}

template <typename TexType, typename ResultType>
struct sampleVolume_impl<TEXTURE_FILTER_LAGRANGE6, TexType, ResultType>
{
	static inline ResultType exec(const TextureCPU<TexType>& tex, float x, float y, float z)
	{
		// shift the coordinate from [0,extent] to [-0.5, extent-0.5] (so that grid points are at integers)
		const float3 coord_grid = make_float3(x, y, z) - 0.5f;
		const float3 index = floor(coord_grid);
		const float3 t = coord_grid - index;

		float3 texCoord;
		ResultType K[6];
		for(int k = -2; k <= 3; k++)
		{
			texCoord.z = index.z + k + 0.5f;
			ResultType J[6];
			for(int j = -2; j <= 3; j++)
			{
				texCoord.y = index.y + j + 0.5f;
				ResultType I[6];
				for(int i = -2; i <= 3; i++)
				{
					texCoord.x = index.x + i + 0.5f;
					I[i+2] = make_floatn<TexType, ResultType>(tex3D(tex, texCoord.x, texCoord.y, texCoord.z));
				}
				J[j+2] = interpolateLagrange6(t.x, I[0], I[1], I[2], I[3], I[4], I[5]);
			}
			K[k+2] = interpolateLagrange6(t.y, J[0], J[1], J[2], J[3], J[4], J[5]);
		}
		return interpolateLagrange6(t.z, K[0], K[1], K[2], K[3], K[4], K[5]);
	}
};


// 1D Lagrange8 interpolation
template<typename Type>
inline Type interpolateLagrange8(float t, Type v0, Type v1, Type v2, Type v3, Type v4, Type v5, Type v6, Type v7)
{
	// d_i = product for j from 0 to 7 and i!=j: (i-j)
	// d = [ -5040, 720, -240, 144, -144, 240, -720, 5040 ]

	// f_i = product for j from 0 to 7 and i!=j: (t+3-j) / d_i  [t specifies the position between v3 and v4 here!]
	float f0 =         (t+2) * (t+1) * (t) * (t-1) * (t-2) * (t-3) * (t-4) / -5040.0f;
	float f1 = (t+3)         * (t+1) * (t) * (t-1) * (t-2) * (t-3) * (t-4) /   720.0f;
	float f2 = (t+3) * (t+2)         * (t) * (t-1) * (t-2) * (t-3) * (t-4) /  -240.0f;
	float f3 = (t+3) * (t+2) * (t+1)       * (t-1) * (t-2) * (t-3) * (t-4) /   144.0f;
	float f4 = (t+3) * (t+2) * (t+1) * (t)         * (t-2) * (t-3) * (t-4) /  -144.0f;
	float f5 = (t+3) * (t+2) * (t+1) * (t) * (t-1)         * (t-3) * (t-4) /   240.0f;
	float f6 = (t+3) * (t+2) * (t+1) * (t) * (t-1) * (t-2)         * (t-4) /  -720.0f;
	float f7 = (t+3) * (t+2) * (t+1) * (t) * (t-1) * (t-2) * (t-3)         /  5040.0f;

	return v0 * f0 + v1 * f1 + v2 * f2 + v3 * f3 + v4 * f4 + v5 * f5 + v6 * f6 + v7 * f7;
}

template <typename TexType, typename ResultType>
struct sampleVolume_impl<TEXTURE_FILTER_LAGRANGE8, TexType, ResultType>
{
	static inline ResultType exec(const TextureCPU<TexType>& tex, float x, float y, float z)
	{
		// shift the coordinate from [0,extent] to [-0.5, extent-0.5] (so that grid points are at integers)
		const float3 coord_grid = make_float3(x, y, z) - 0.5f;
		const float3 index = floor(coord_grid);
		const float3 t = coord_grid - index;

		float3 texCoord;
		ResultType K[8];
		for(int k = -3; k <= 4; k++)
		{
			texCoord.z = index.z + k + 0.5f;
			ResultType J[8];
			for(int j = -3; j <= 4; j++)
			{
				texCoord.y = index.y + j + 0.5f;
				ResultType I[8];
				for(int i = -3; i <= 4; i++)
				{
					texCoord.x = index.x + i + 0.5f;
					I[i+3] = make_floatn<TexType, ResultType>(tex3D(tex, texCoord.x, texCoord.y, texCoord.z));
				}
				J[j+3] = interpolateLagrange8(t.x, I[0], I[1], I[2], I[3], I[4], I[5], I[6], I[7]);
			}
			K[k+3] = interpolateLagrange8(t.y, J[0], J[1], J[2], J[3], J[4], J[5], J[6], J[7]);
		}
		return interpolateLagrange8(t.z, K[0], K[1], K[2], K[3], K[4], K[5], K[6], K[7]);
	}
};

} // namespace (sampleVolume implementations)

// "public" sampleVolume functions:

template <eTextureFilterMode F, typename TexType, typename ResultType>
inline ResultType sampleVolume(const TextureCPU<TexType>& tex, float x, float y, float z)
{
	return sampleVolume_impl<F, TexType, ResultType>::exec(tex, x, y, z);
}

template <eTextureFilterMode F, typename TexType, typename ResultType>
inline ResultType sampleVolume(const TextureCPU<TexType>& tex, float3 coord)
{
	return sampleVolume_impl<F, TexType, ResultType>::exec(tex, coord.x, coord.y, coord.z);
}



#endif
