#ifndef __TUM3D__TEXTURE_FILTER_CUH__
#define __TUM3D__TEXTURE_FILTER_CUH__


#include <math_constants.h>
#include <math_functions.h>

#include "TextureFilterMode.h"


///////////////////////////////////////////////////////////////////////////////
// Linear and cubic texture filter
///////////////////////////////////////////////////////////////////////////////

// helper to create all-zeros float<n>
template <typename TOut>
struct make_zero_floatn_impl { __device__ static __inline TOut exec(); };
// no implementation - only specializations allowed:
template <>
struct make_zero_floatn_impl<float1> { __device__ static __inline float1 exec() { return make_float1(0.0f); } };
template <>
struct make_zero_floatn_impl<float2> { __device__ static __inline float2 exec() { return make_float2(0.0f, 0.0f); } };
template <>
struct make_zero_floatn_impl<float3> { __device__ static __inline float3 exec() { return make_float3(0.0f, 0.0f, 0.0f); } };
template <>
struct make_zero_floatn_impl<float4> { __device__ static __inline float4 exec() { return make_float4(0.0f, 0.0f, 0.0f, 0.0f); } };

template <typename TOut>
__device__ inline TOut make_zero_floatn()
{
	return make_zero_floatn_impl<TOut>::exec();
}

// helper to convert float<n> to float<m> (or just float)
template <typename TIn, typename TOut>
struct make_floatn_impl { __device__ static __inline TOut exec(TIn val); };
// no implementation - only specializations allowed:
template <typename TIn>
struct make_floatn_impl<TIn, float>  { __device__ static __inline float  exec(TIn val) { return val.x; } };
template <typename TIn>
struct make_floatn_impl<TIn, float1> { __device__ static __inline float1 exec(TIn val) { return make_float1(val.x); } };
template <typename TIn>
struct make_floatn_impl<TIn, float2> { __device__ static __inline float2 exec(TIn val) { return make_float2(val.x, val.y); } };
template <typename TIn>
struct make_floatn_impl<TIn, float3> { __device__ static __inline float3 exec(TIn val) { return make_float3(val.x, val.y, val.z); } };
template <typename TIn>
struct make_floatn_impl<TIn, float4> { __device__ static __inline float4 exec(TIn val) { return val; } };

template <typename TIn, typename TOut>
__device__ inline TOut make_floatn(TIn val)
{
	return make_floatn_impl<TIn, TOut>::exec(val);
}


namespace
{

template <typename TexType, typename ResultType>
__device__ inline ResultType getInterpolatedCubicBSpline(texture<TexType, cudaTextureType3D, cudaReadModeElementType> tex, float3 g0, float3 g1, float3 h0, float3 h1)
{
	// fetch the eight linear interpolations
	// weighting and fetching is interleaved for performance and stability reasons
	ResultType tex000 = make_floatn<TexType, ResultType>(tex3D(tex, h0.x, h0.y, h0.z));
	ResultType tex100 = make_floatn<TexType, ResultType>(tex3D(tex, h1.x, h0.y, h0.z));
	tex000 = g0.x * tex000 + g1.x * tex100;  // weight along the x-direction
	ResultType tex010 = make_floatn<TexType, ResultType>(tex3D(tex, h0.x, h1.y, h0.z));
	ResultType tex110 = make_floatn<TexType, ResultType>(tex3D(tex, h1.x, h1.y, h0.z));
	tex010 = g0.x * tex010 + g1.x * tex110;  // weight along the x-direction
	tex000 = g0.y * tex000 + g1.y * tex010;  // weight along the y-direction
	ResultType tex001 = make_floatn<TexType, ResultType>(tex3D(tex, h0.x, h0.y, h1.z));
	ResultType tex101 = make_floatn<TexType, ResultType>(tex3D(tex, h1.x, h0.y, h1.z));
	tex001 = g0.x * tex001 + g1.x * tex101;  // weight along the x-direction
	ResultType tex011 = make_floatn<TexType, ResultType>(tex3D(tex, h0.x, h1.y, h1.z));
	ResultType tex111 = make_floatn<TexType, ResultType>(tex3D(tex, h1.x, h1.y, h1.z));
	tex011 = g0.x * tex011 + g1.x * tex111;  // weight along the x-direction
	tex001 = g0.y * tex001 + g1.y * tex011;  // weight along the y-direction

	return (g0.z * tex000 + g1.z * tex001);  // weight along the z-direction
}

}


namespace // sampleVolume implementations
{

template <eTextureFilterMode F, typename TexType, typename ResultType>
struct sampleVolume_impl
{
	//__device__ static ResultType exec(texture<TexType, cudaTextureType3D, cudaReadModeElementType> tex, float x, float y, float z);
	// no exec() implementation - only specializations allowed
};


template <typename TexType, typename ResultType>
struct sampleVolume_impl<TEXTURE_FILTER_LINEAR, TexType, ResultType>
{
	__device__ static inline ResultType exec(texture<TexType, cudaTextureType3D, cudaReadModeElementType> tex, float x, float y, float z)
	{
		return make_floatn<TexType, ResultType>(tex3D(tex, x, y, z));
	}
};


template <typename TexType, typename ResultType>
struct sampleVolume_impl<TEXTURE_FILTER_CUBIC, TexType, ResultType>
{
	__device__ static inline ResultType exec(texture<TexType, cudaTextureType3D, cudaReadModeElementType> tex, float x, float y, float z)
	{
		// shift the coordinate from [0,extent] to [-0.5, extent-0.5] (so that grid points are at integers)
		const float3 coord_grid = make_float3(x - 0.5f, y - 0.5f, z - 0.5f);
		const float3 index = floor(coord_grid);
		const float3 t = coord_grid - index;

		// compute weights for grid points (cubic b-spline)
		const float3 inv_t = 1.0f - t;
		const float3 t_sqr = t * t;
		const float3 inv_t_sqr = inv_t * inv_t;

		const float3 w0 = 1.0f/6.0f * inv_t_sqr * inv_t;
		const float3 w1 = 2.0f/3.0f - 0.5f * t_sqr * (2.0f - t);
		const float3 w2 = 2.0f/3.0f - 0.5f * inv_t_sqr * (2.0f - inv_t);
		const float3 w3 = 1.0f/6.0f * t_sqr * t;

		// weights and offsets for linear sampling from texture
		const float3 g0 = w0 + w1;
		const float3 g1 = w2 + w3;
		const float3 h0 = (w1 / g0) - 0.5f + index;
		const float3 h1 = (w3 / g1) + 1.5f + index;

		return getInterpolatedCubicBSpline<TexType, ResultType>(tex, g0, g1, h0, h1);
	}
};


template <typename TexType, typename ResultType>
struct sampleVolume_impl<TEXTURE_FILTER_CATROM, TexType, ResultType>
{
	__device__ static inline ResultType exec(texture<TexType, cudaTextureType3D, cudaReadModeElementType> tex, float x, float y, float z)
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
				#pragma unroll
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


template <typename TexType, typename ResultType>
struct sampleVolume_impl<TEXTURE_FILTER_CATROM_STAGGERED, TexType, ResultType>
{
	__device__ static inline ResultType exec(texture<TexType, cudaTextureType3D, cudaReadModeElementType> tex, float x, float y, float z)
	{
		ResultType resX = sampleVolume_impl<TEXTURE_FILTER_CATROM, TexType, ResultType>::exec(tex, x - 0.5f, y, z);
		ResultType resY = sampleVolume_impl<TEXTURE_FILTER_CATROM, TexType, ResultType>::exec(tex, x, y - 0.5f, z);
		ResultType resZ = sampleVolume_impl<TEXTURE_FILTER_CATROM, TexType, ResultType>::exec(tex, x, y, z - 0.5f);
		return make_float3(resX.x, resY.y, resZ.z);
	}
};


// 1D Lagrange4 interpolation
template<typename Type>
__device__ inline Type interpolateLagrange4(float t, const Type v[4])
{
	// d_i = product for j from 0 to 3 and i!=j: (i-j)
	// d = [ -6, 2, -2, 6 ]

	// f_i = product for j from 0 to 3 and i!=j: (t+1-j) / d_i  [t specifies the position between v1 and v2 here!]
	float f0 =         (t) * (t-1) * (t-2) / -6.0f;
	float f1 = (t+1)       * (t-1) * (t-2) /  2.0f;
	float f2 = (t+1) * (t)         * (t-2) / -2.0f;
	float f3 = (t+1) * (t) * (t-1)         /  6.0f;

	return v[0] * f0 + v[1] * f1 + v[2] * f2 + v[3] * f3;
}

template <typename TexType, typename ResultType>
struct sampleVolume_impl<TEXTURE_FILTER_LAGRANGE4, TexType, ResultType>
{
	__device__ static inline ResultType exec(texture<TexType, cudaTextureType3D, cudaReadModeElementType> tex, float x, float y, float z)
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
				#pragma unroll
				for(int i = -1; i <= 2; i++)
				{
					texCoord.x = index.x + i + 0.5f;
					I[i+1] = make_floatn<TexType, ResultType>(tex3D(tex, texCoord.x, texCoord.y, texCoord.z));
				}
				J[j+1] = interpolateLagrange4(t.x, I);
			}
			K[k+1] = interpolateLagrange4(t.y, J);
		}
		return interpolateLagrange4(t.z, K);
	}
};


__device__ static inline float lag6tprod(float t, int exclude)
{
	float f = 1.0f;
	for(int i = -3; i <= 2; i++) {
		if(i != exclude) {
			f *= t + i;
		}
	}
	return f;
}

__constant__ float c_lag6factors[8] = {
	1.0f / -120.0f,
	1.0f /   24.0f,
	1.0f /  -12.0f,
	1.0f /   12.0f,
	1.0f /  -24.0f,
	1.0f /  120.0f
};

// 1D Lagrange6 interpolation
template<typename Type>
__device__ inline Type interpolateLagrange6(float t, const Type v[6])
{
	// straightforward loop implementation:
	Type result = make_zero_floatn<Type>();
	for(int i = 0; i < 6; i++) {
		float f = lag6tprod(t, 2 - i) * c_lag6factors[i];
		result += f * v[i];
	}
	return result;

	// manually unrolled implementation: (slightly slower)
	//// d_i = product for j from 0 to 5 and i!=j: (i-j)
	//// d = [ -120, 24, -12, 12, -24, 120 ]

	//// f_i = product for j from 0 to 5 and i!=j: (t+2-j) / d_i  [t specifies the position between v2 and v3 here!]
	//float f0 =         (t+1) * (t) * (t-1) * (t-2) * (t-3) / -120.0f;
	//float f1 = (t+2)         * (t) * (t-1) * (t-2) * (t-3) /   24.0f;
	//float f2 = (t+2) * (t+1)       * (t-1) * (t-2) * (t-3) /  -12.0f;
	//float f3 = (t+2) * (t+1) * (t)         * (t-2) * (t-3) /   12.0f;
	//float f4 = (t+2) * (t+1) * (t) * (t-1)         * (t-3) /  -24.0f;
	//float f5 = (t+2) * (t+1) * (t) * (t-1) * (t-2)         /  120.0f;

	//return v[0] * f0 + v[1] * f1 + v[2] * f2 + v[3] * f3 + v[4] * f4 + v[5] * f5;
}

template <typename TexType, typename ResultType>
struct sampleVolume_impl<TEXTURE_FILTER_LAGRANGE6, TexType, ResultType>
{
	__device__ static inline ResultType exec(texture<TexType, cudaTextureType3D, cudaReadModeElementType> tex, float x, float y, float z)
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
				#pragma unroll
				for(int i = -2; i <= 3; i++)
				{
					texCoord.x = index.x + i + 0.5f;
					I[i+2] = make_floatn<TexType, ResultType>(tex3D(tex, texCoord.x, texCoord.y, texCoord.z));
				}
				J[j+2] = interpolateLagrange6(t.x, I);
			}
			K[k+2] = interpolateLagrange6(t.y, J);
		}
		return interpolateLagrange6(t.z, K);
	}
};


__device__ static inline float lag8tprod(float t, int exclude)
{
	float f = 1.0f;
	for(int i = -4; i <= 3; i++) {
		if(i != exclude) {
			f *= t + i;
		}
	}
	return f;
}

__constant__ float c_lag8factors[8] = {
	1.0f / -5040.0f,
	1.0f /   720.0f,
	1.0f /  -240.0f,
	1.0f /   144.0f,
	1.0f /  -144.0f,
	1.0f /   240.0f,
	1.0f /  -720.0f,
	1.0f /  5040.0f
};

// 1D Lagrange8 interpolation
template<typename Type>
__device__ inline Type interpolateLagrange8(float t, const Type v[8])
{
	// straightforward loop implementation:
	Type result = make_zero_floatn<Type>();
	for(int i = 0; i < 8; i++) {
		float f = lag8tprod(t, 3 - i) * c_lag8factors[i];
		result += f * v[i];
	}
	return result;

	// manually unrolled implementation: (slightly slower)
	//// d_i = product for j from 0 to 7 and i!=j: (i-j)
	//// d = [ -5040, 720, -240, 144, -144, 240, -720, 5040 ]

	//// f_i = product for j from 0 to 7 and i!=j: (t+3-j) / d_i  [t specifies the position between v3 and v4 here!]
	//float f0 =         (t+2) * (t+1) * (t) * (t-1) * (t-2) * (t-3) * (t-4) / -5040.0f;
	//float f1 = (t+3)         * (t+1) * (t) * (t-1) * (t-2) * (t-3) * (t-4) /   720.0f;
	//float f2 = (t+3) * (t+2)         * (t) * (t-1) * (t-2) * (t-3) * (t-4) /  -240.0f;
	//float f3 = (t+3) * (t+2) * (t+1)       * (t-1) * (t-2) * (t-3) * (t-4) /   144.0f;
	//float f4 = (t+3) * (t+2) * (t+1) * (t)         * (t-2) * (t-3) * (t-4) /  -144.0f;
	//float f5 = (t+3) * (t+2) * (t+1) * (t) * (t-1)         * (t-3) * (t-4) /   240.0f;
	//float f6 = (t+3) * (t+2) * (t+1) * (t) * (t-1) * (t-2)         * (t-4) /  -720.0f;
	//float f7 = (t+3) * (t+2) * (t+1) * (t) * (t-1) * (t-2) * (t-3)         /  5040.0f;

	//return v[0] * f0 + v[1] * f1 + v[2] * f2 + v[3] * f3 + v[4] * f4 + v[5] * f5 + v[6] * f6 + v[7] * f7;
}


template <typename TexType, typename ResultType>
struct sampleVolume_impl<TEXTURE_FILTER_LAGRANGE8, TexType, ResultType>
{
	__device__ static inline ResultType exec(texture<TexType, cudaTextureType3D, cudaReadModeElementType> tex, float x, float y, float z)
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
				#pragma unroll
				for(int i = -3; i <= 4; i++)
				{
					texCoord.x = index.x + i + 0.5f;
					I[i+3] = make_floatn<TexType, ResultType>(tex3D(tex, texCoord.x, texCoord.y, texCoord.z));
				}
				J[j+3] = interpolateLagrange8(t.x, I);
			}
			K[k+3] = interpolateLagrange8(t.y, J);
		}
		return interpolateLagrange8(t.z, K);
	}
};


// d_i = product for j from 0 to 15 and i!=j: (i-j)
__constant__ float c_lag16factors[16] = {
	1.0f / -1307674368000.0f,
	1.0f /    87178291200.0f,
	1.0f /   -12454041600.0f,
	1.0f /     2874009600.0f,
	1.0f /     -958003200.0f,
	1.0f /      435456000.0f,
	1.0f /     -261273600.0f,
	1.0f /      203212800.0f,
	1.0f /     -203212800.0f,
	1.0f /      261273600.0f,
	1.0f /     -435456000.0f,
	1.0f /      958003200.0f,
	1.0f /    -2874009600.0f,
	1.0f /    12454041600.0f,
	1.0f /   -87178291200.0f,
	1.0f /  1307674368000.0f
};

__device__ static inline float lag16tprod(float t, int exclude)
{
	// f_i = product for j from 0 to 15 and i!=j: (t+7-j) / d_i  [t specifies the position between v5 and v6 here!]
	float f = 1.0f;
	for(int i = -8; i <= 7; i++) {
		if(i != exclude) {
			f *= t + i;
		}
	}
	return f;
}

// 1D Lagrange16 interpolation
template<typename Type>
__device__ inline Type interpolateLagrange16(float t, const Type v[16])
{
	Type result = make_zero_floatn<Type>();
	for(int i = 0; i < 16; i++) {
		float f = lag16tprod(t, 7 - i) * c_lag16factors[i];
		result += f * v[i];
	}
	return result;
}

template <typename TexType, typename ResultType>
struct sampleVolume_impl<TEXTURE_FILTER_LAGRANGE16, TexType, ResultType>
{
	__device__ static float3 exec(texture<float4, cudaTextureType3D, cudaReadModeElementType> tex, float x, float y, float z)
	{
		// shift the coordinate from [0,extent] to [-0.5, extent-0.5] (so that grid points are at integers)
		const float3 coord_grid = make_float3(x, y, z) - 0.5f;
		const float3 index = floor(coord_grid);
		const float3 t = coord_grid - index;

		float3 texCoord;
		ResultType K[16];
		#pragma unroll 1
		for(int k = -7; k <= 8; k++)
		{
			texCoord.z = index.z + k + 0.5f;
			ResultType J[16];
			#pragma unroll 1
			for(int j = -7; j <= 8; j++)
			{
				texCoord.y = index.y + j + 0.5f;
				ResultType I[16];
				#pragma unroll
				for(int i = -7; i <= 8; i++)
				{
					texCoord.x = index.x + i + 0.5f;
					I[i+7] = make_floatn<TexType, ResultType>(tex3D(tex, texCoord.x, texCoord.y, texCoord.z));
				}
				J[j+7] = interpolateLagrange16(t.x, I);
			}
			K[k+7] = interpolateLagrange16(t.y, J);
		}
		return interpolateLagrange16(t.z, K);
	}
};


//// 1D WENO4 interpolation (weighted essentially non-oscillatory)
//template<typename Type>
//__device__ inline Type interpolateWENO4(float x, Type v0, Type v1, Type v2, Type v3)
//{
//	Type p1 = v1 + 0.5f * (v2-v0)*x            + 0.5f*(v2-2.0f*v1+v0)*x*x;
//	Type p2 = v1 + 0.5f * (-v3+4.0f*v2-3*v1)*x + 0.5f*(v3-2.0f*v2+v1)*x*x;
//
//	float C1 = (2.0f-x) / 3.0f;
//	float C2 = (x+1.0f) / 3.0f;
//
//	Type IS1 = (26.0f*v2*v0 - 52.0f*v1*v0 - 76.0f*v2*v1 + 25.0f*v2*v2 + 64.0f*v1*v1 + 13.0f*v0*v0)/12.0f;
//	Type IS2 = (26.0f*v3*v1 - 52.0f*v3*v2 - 76.0f*v2*v1 + 25.0f*v1*v1 + 64.0f*v2*v2 + 13.0f*v3*v3)/12.0f;
//
//	Type a1 = C1 / fmax(IS1*IS1, 10e-6f);
//	Type a2 = C2 / fmax(IS2*IS2, 10e-6f);
//
//	Type result = (a1 * p1 + a2 * p2) / (a1 + a2);
//	Type minimum = fmin(fmin(fmin(v0, v1), v2), v3);
//	Type maximum = fmax(fmax(fmax(v0, v1), v2), v3);
//	return clamp(result, minimum, maximum);
//}

//template <typename TexType, typename ResultType>
//struct sampleVolume_impl<TEXTURE_FILTER_WENO4, TexType, ResultType>
//{
//	__device__ static inline ResultType exec(texture<TexType, cudaTextureType3D, cudaReadModeElementType> tex, float x, float y, float z)
//	{
//		const float3 coord_grid = make_float3(x, y, z) - 0.5f;
//		const float3 index = floor(coord_grid);
//		const float3 w = clamp(coord_grid - index, 0.0f, 1.0f);
//		float3 texCoord;
//		ResultType K[4];
//		//#pragma unroll
//		for(int k = -1; k <= 2; k++)
//		{
//			texCoord.z = index.z + k + 0.5f;
//			ResultType J[4];
//			//#pragma unroll
//			for(int j = -1; j <= 2; j++)
//			{
//				texCoord.y = index.y + j + 0.5f;
//				ResultType I[4];
//				#pragma unroll
//				for(int i = -1; i <= 2; i++)
//				{
//					texCoord.x = index.x + i + 0.5f;
//					I[i+1] = make_floatn<TexType, ResultType>(tex3D(tex, texCoord.x, texCoord.y, texCoord.z));
//				}
//				J[j+1] = interpolateWENO4(w.x, I[0], I[1], I[2], I[3]);
//			}
//			K[k+1] = interpolateWENO4(w.y, J[0], J[1], J[2], J[3]);
//		}
//		return interpolateWENO4(w.z, K[0], K[1] ,K[2], K[3]);
//	}
//};


//namespace DoubleGyre
//{
//	__device__ const float PI = 3.1415926536f;
//
//	__device__ const float A = 0.1f;
//	__device__ const float EPS = 0.25f;
//	__device__ const float OMEGA = 0.6283185307f; //2.0f * PI / 10.0f;
//
//	__device__ const float DOMAIN_X = 2.0f;
//	__device__ const float DOMAIN_Y = 1.0f;
//	__device__ const float DOMAIN_Z = 2.0f;
//	__device__ const float DOMAIN_T = 10.0f;
//
//	__device__ inline float a(float t)
//	{
//		return EPS * sin(OMEGA * t);
//	}
//
//	__device__ inline float b(float t)
//	{
//		return 1.0f - 2.0f * EPS * sin(OMEGA * t);
//	}
//
//	__device__ inline float f(float x, float t)
//	{
//		return a(t) * x * x + b(t) * x;
//	}
//
//	__device__ inline float dfdx(float x, float t)
//	{
//		return 2.0f * a(t) * x + b(t);
//	}
//
//	__device__ inline float x(float x, float y, float z, float t)
//	{
//		return -PI * A * sin(PI * f(x, t + z / DOMAIN_Z * DOMAIN_T)) * cos(PI * y);
//	}
//
//	__device__ inline float y(float x, float y, float z, float t)
//	{
//		return  PI * A * cos(PI * f(x, t + z / DOMAIN_Z * DOMAIN_T)) * sin(PI * y) * dfdx(x, t);
//	}
//
//	__device__ inline float z(float x, float y, float z, float t)
//	{
//		return  0.0f;
//	}
//}
//
//
//template <typename TexType>
//struct sampleVolume_impl<TEXTURE_ANALYTIC_DOUBLEGYRE, TexType, float3>
//{
//	__device__ static inline float3 exec(texture<TexType, cudaTextureType3D, cudaReadModeElementType> tex, float x, float y, float z)
//	{
//		x += 1.0f;
//		y += 0.5f;
//		z += 1.0f;
//		float t = 0.0f;
//
//		return make_float3(DoubleGyre::x(x, y, z, t), DoubleGyre::y(x, y, z, t), DoubleGyre::z(x, y, z, t));
//	}
//};

} // namespace (sampleVolume implementations)

// "public" sampleVolume functions:

template <eTextureFilterMode F, typename TexType, typename ResultType>
__device__ inline ResultType sampleVolume(texture<TexType, cudaTextureType3D, cudaReadModeElementType> tex, float x, float y, float z)
{
	return sampleVolume_impl<F, TexType, ResultType>::exec(tex, x, y, z);
}

template <eTextureFilterMode F, typename TexType, typename ResultType>
__device__ inline ResultType sampleVolume(texture<TexType, cudaTextureType3D, cudaReadModeElementType> tex, float3 coord)
{
	return sampleVolume_impl<F, TexType, ResultType>::exec(tex, coord.x, coord.y, coord.z);
}


namespace // sampleVolumeDerivativeD implementations (where D in X, Y, Z)
{

template <eTextureFilterMode F, typename TexType, typename ResultType>
struct sampleVolumeDerivativeX_impl
{
	__device__ static inline ResultType exec(texture<TexType, cudaTextureType3D, cudaReadModeElementType> tex, float x, float y, float z, float h)
	{
		// default implementation: central differences
		ResultType dp = sampleVolume<F, TexType, ResultType>(tex, x + 1.0f, y, z);
		ResultType dn = sampleVolume<F, TexType, ResultType>(tex, x - 1.0f, y, z);
		return (dp - dn) / (2.0f * h);
	}
};

template <eTextureFilterMode F, typename TexType, typename ResultType>
struct sampleVolumeDerivativeY_impl
{
	__device__ static inline ResultType exec(texture<TexType, cudaTextureType3D, cudaReadModeElementType> tex, float x, float y, float z, float h)
	{
		// default implementation: central differences
		ResultType dp = sampleVolume<F, TexType, ResultType>(tex, x, y + 1.0f, z);
		ResultType dn = sampleVolume<F, TexType, ResultType>(tex, x, y - 1.0f, z);
		return (dp - dn) / (2.0f * h);
	}
};

template <eTextureFilterMode F, typename TexType, typename ResultType>
struct sampleVolumeDerivativeZ_impl
{
	__device__ static inline ResultType exec(texture<TexType, cudaTextureType3D, cudaReadModeElementType> tex, float x, float y, float z, float h)
	{
		// default implementation: central differences
		ResultType dp = sampleVolume<F, TexType, ResultType>(tex, x, y, z + 1.0f);
		ResultType dn = sampleVolume<F, TexType, ResultType>(tex, x, y, z - 1.0f);
		return (dp - dn) / (2.0f * h);
	}
};


template <typename TexType, typename ResultType>
struct sampleVolumeDerivativeX_impl<TEXTURE_FILTER_CUBIC, TexType, ResultType>
{
	__device__ static inline ResultType exec(texture<TexType, cudaTextureType3D, cudaReadModeElementType> tex, float x, float y, float z, float h)
	{
		// shift the coordinate from [0,extent] to [-0.5, extent-0.5] (so that grid points are at integers)
		const float3 coord_grid = make_float3(x - 0.5f, y - 0.5f, z - 0.5f);
		const float3 index = floor(coord_grid);
		const float3 t = coord_grid - index;

		// compute weights for grid points (cubic b-spline)
		const float3 inv_t = 1.0f - t;
		const float3 t_sqr = t * t;
		const float3 inv_t_sqr = inv_t * inv_t;

		float3 w0 = 1.0f/6.0f * inv_t_sqr * inv_t;
		float3 w1 = 2.0f/3.0f - 0.5f * t_sqr * (2.0f - t);
		float3 w2 = 2.0f/3.0f - 0.5f * inv_t_sqr * (2.0f - inv_t);
		float3 w3 = 1.0f/6.0f * t_sqr * t;

		// derivative in x - replace wn.x
		w0.x = 0.5f * (-t_sqr.x + 2.0f * t.x - 1.0f);
		w1.x = 0.5f * (3.0f * t_sqr.x - 4.0f * t.x);
		w2.x = 0.5f * (-3.0f * t_sqr.x + 2.0f * t.x + 1.0f);
		w3.x = 0.5f * t_sqr.x;

		// weights and offsets for linear sampling from texture
		const float3 g0 = w0 + w1;
		const float3 g1 = w2 + w3;
		const float3 h0 = (w1 / g0) - 0.5f + index;
		const float3 h1 = (w3 / g1) + 1.5f + index;

		return getInterpolatedCubicBSpline<TexType, ResultType>(tex, g0, g1, h0, h1) / h;
	}
};

template <typename TexType, typename ResultType>
struct sampleVolumeDerivativeY_impl<TEXTURE_FILTER_CUBIC, TexType, ResultType>
{
	__device__ static inline ResultType exec(texture<TexType, cudaTextureType3D, cudaReadModeElementType> tex, float x, float y, float z, float h)
	{
		// shift the coordinate from [0,extent] to [-0.5, extent-0.5] (so that grid points are at integers)
		const float3 coord_grid = make_float3(x - 0.5f, y - 0.5f, z - 0.5f);
		const float3 index = floor(coord_grid);
		const float3 t = coord_grid - index;

		// compute weights for grid points (cubic b-spline)
		const float3 inv_t = 1.0f - t;
		const float3 t_sqr = t * t;
		const float3 inv_t_sqr = inv_t * inv_t;

		float3 w0 = 1.0f/6.0f * inv_t_sqr * inv_t;
		float3 w1 = 2.0f/3.0f - 0.5f * t_sqr * (2.0f - t);
		float3 w2 = 2.0f/3.0f - 0.5f * inv_t_sqr * (2.0f - inv_t);
		float3 w3 = 1.0f/6.0f * t_sqr * t;

		// derivative in y - replace wn.y
		w0.y = 0.5f * (-t_sqr.y + 2.0f * t.y - 1.0f);
		w1.y = 0.5f * (3.0f * t_sqr.y - 4.0f * t.y);
		w2.y = 0.5f * (-3.0f * t_sqr.y + 2.0f * t.y + 1.0f);
		w3.y = 0.5f * t_sqr.y;

		// weights and offsets for linear sampling from texture
		const float3 g0 = w0 + w1;
		const float3 g1 = w2 + w3;
		const float3 h0 = (w1 / g0) - 0.5f + index;
		const float3 h1 = (w3 / g1) + 1.5f + index;

		return getInterpolatedCubicBSpline<TexType, ResultType>(tex, g0, g1, h0, h1) / h;
	}
};

template <typename TexType, typename ResultType>
struct sampleVolumeDerivativeZ_impl<TEXTURE_FILTER_CUBIC, TexType, ResultType>
{
	__device__ static inline ResultType exec(texture<TexType, cudaTextureType3D, cudaReadModeElementType> tex, float x, float y, float z, float h)
	{
		// shift the coordinate from [0,extent] to [-0.5, extent-0.5] (so that grid points are at integers)
		const float3 coord_grid = make_float3(x - 0.5f, y - 0.5f, z - 0.5f);
		const float3 index = floor(coord_grid);
		const float3 t = coord_grid - index;

		// compute weights for grid points (cubic b-spline)
		const float3 inv_t = 1.0f - t;
		const float3 t_sqr = t * t;
		const float3 inv_t_sqr = inv_t * inv_t;

		float3 w0 = 1.0f/6.0f * inv_t_sqr * inv_t;
		float3 w1 = 2.0f/3.0f - 0.5f * t_sqr * (2.0f - t);
		float3 w2 = 2.0f/3.0f - 0.5f * inv_t_sqr * (2.0f - inv_t);
		float3 w3 = 1.0f/6.0f * t_sqr * t;

		// derivative in z - replace wn.z
		w0.z = 0.5f * (-t_sqr.z + 2.0f * t.z - 1.0f);
		w1.z = 0.5f * (3.0f * t_sqr.z - 4.0f * t.z);
		w2.z = 0.5f * (-3.0f * t_sqr.z + 2.0f * t.z + 1.0f);
		w3.z = 0.5f * t_sqr.z;

		// weights and offsets for linear sampling from texture
		const float3 g0 = w0 + w1;
		const float3 g1 = w2 + w3;
		const float3 h0 = (w1 / g0) - 0.5f + index;
		const float3 h1 = (w3 / g1) + 1.5f + index;

		return getInterpolatedCubicBSpline<TexType, ResultType>(tex, g0, g1, h0, h1) / h;
	}
};


template <typename TexType, typename ResultType>
struct sampleVolumeDerivativeX_impl<TEXTURE_FILTER_CATROM, TexType, ResultType>
{
	__device__ static inline ResultType exec(texture<TexType, cudaTextureType3D, cudaReadModeElementType> tex, float x, float y, float z, float h)
	{
		// shift the coordinate from [0,extent] to [-0.5, extent-0.5] (so that grid points are at integers)
		const float3 coord_grid = make_float3(x - 0.5f, y - 0.5f, z - 0.5f);
		const float3 index = floor(coord_grid);
		const float3 t = coord_grid - index;

		// compute weights for grid points (cubic catmull-rom spline)
		const float3 t2 = t * t;
		const float3 t3 = t2 * t;

		float3 w0 = 0.5f * (-t3 + 2.0f * t2 - t);
		float3 w1 = 0.5f * (3.0f * t3 - 5.0f * t2 + 2.0f);
		float3 w2 = 0.5f * (-3.0f * t3 + 4.0f * t2 + t);
		float3 w3 = 0.5f * (t3 - t2);
		// note: can't use hardware linear interpolations for catmull-rom,
		//       because the weights can be negative

		// derivative in x - replace wn.x
		w0.x = 0.5f * (-3.0f * t2.x + 4.0f * t.x - 1.0f);
		w1.x = 0.5f * (9.0f * t2.x - 10.0f * t.x);
		w2.x = 0.5f * (-9.0f * t2.x + 8.0f * t.x + 1.0f);
		w3.x = 0.5f * (3.0f * t2.x - 2.0f * t.x);

		float3 texCoord;
		ResultType K[4];
		//#pragma unroll
		for(int k = -1; k <= 2; k++)
		{
			texCoord.z = index.z + k;
			ResultType J[4];
			//#pragma unroll
			for(int j = -1; j <= 2; j++)
			{
				texCoord.y = index.y + j;
				ResultType I[4];
				#pragma unroll
				for(int i = -1; i <= 2; i++)
				{
					texCoord.x = index.x + i;
					I[i+1] = make_floatn<TexType, ResultType>(tex3D(tex, texCoord.x, texCoord.y, texCoord.z));
				}
				J[j+1] = w0.x * I[0] + w1.x * I[1] + w2.x * I[2] + w3.x * I[3];
			}
			K[k+1] = w0.y * J[0] + w1.y * J[1] + w2.y * J[2] + w3.y * J[3];
		}
		return (w0.z * K[0] + w1.z * K[1] + w2.z * K[2] + w3.z * K[3]) / h;
	}
};

template <typename TexType, typename ResultType>
struct sampleVolumeDerivativeY_impl<TEXTURE_FILTER_CATROM, TexType, ResultType>
{
	__device__ static inline ResultType exec(texture<TexType, cudaTextureType3D, cudaReadModeElementType> tex, float x, float y, float z, float h)
	{
		// shift the coordinate from [0,extent] to [-0.5, extent-0.5] (so that grid points are at integers)
		const float3 coord_grid = make_float3(x - 0.5f, y - 0.5f, z - 0.5f);
		const float3 index = floor(coord_grid);
		const float3 t = coord_grid - index;

		// compute weights for grid points (cubic catmull-rom spline)
		const float3 t2 = t * t;
		const float3 t3 = t2 * t;

		float3 w0 = 0.5f * (-t3 + 2.0f * t2 - t);
		float3 w1 = 0.5f * (3.0f * t3 - 5.0f * t2 + 2.0f);
		float3 w2 = 0.5f * (-3.0f * t3 + 4.0f * t2 + t);
		float3 w3 = 0.5f * (t3 - t2);
		// note: can't use hardware linear interpolations for catmull-rom,
		//       because the weights can be negative

		// derivative in y - replace wn.y
		w0.y = 0.5f * (-3.0f * t2.y + 4.0f * t.y - 1.0f);
		w1.y = 0.5f * (9.0f * t2.y - 10.0f * t.y);
		w2.y = 0.5f * (-9.0f * t2.y + 8.0f * t.y + 1.0f);
		w3.y = 0.5f * (3.0f * t2.y - 2.0f * t.y);

		float3 texCoord;
		ResultType K[4];
		//#pragma unroll
		for(int k = -1; k <= 2; k++)
		{
			texCoord.z = index.z + k;
			ResultType J[4];
			//#pragma unroll
			for(int j = -1; j <= 2; j++)
			{
				texCoord.y = index.y + j;
				ResultType I[4];
				#pragma unroll
				for(int i = -1; i <= 2; i++)
				{
					texCoord.x = index.x + i;
					I[i+1] = make_floatn<TexType, ResultType>(tex3D(tex, texCoord.x, texCoord.y, texCoord.z));
				}
				J[j+1] = w0.x * I[0] + w1.x * I[1] + w2.x * I[2] + w3.x * I[3];
			}
			K[k+1] = w0.y * J[0] + w1.y * J[1] + w2.y * J[2] + w3.y * J[3];
		}
		return (w0.z * K[0] + w1.z * K[1] + w2.z * K[2] + w3.z * K[3]) / h;
	}
};

template <typename TexType, typename ResultType>
struct sampleVolumeDerivativeZ_impl<TEXTURE_FILTER_CATROM, TexType, ResultType>
{
	__device__ static inline ResultType exec(texture<TexType, cudaTextureType3D, cudaReadModeElementType> tex, float x, float y, float z, float h)
	{
		// shift the coordinate from [0,extent] to [-0.5, extent-0.5] (so that grid points are at integers)
		const float3 coord_grid = make_float3(x - 0.5f, y - 0.5f, z - 0.5f);
		const float3 index = floor(coord_grid);
		const float3 t = coord_grid - index;

		// compute weights for grid points (cubic catmull-rom spline)
		const float3 t2 = t * t;
		const float3 t3 = t2 * t;

		float3 w0 = 0.5f * (-t3 + 2.0f * t2 - t);
		float3 w1 = 0.5f * (3.0f * t3 - 5.0f * t2 + 2.0f);
		float3 w2 = 0.5f * (-3.0f * t3 + 4.0f * t2 + t);
		float3 w3 = 0.5f * (t3 - t2);
		// note: can't use hardware linear interpolations for catmull-rom,
		//       because the weights can be negative

		// derivative in z - replace wn.z
		w0.z = 0.5f * (-3.0f * t2.z + 4.0f * t.z - 1.0f);
		w1.z = 0.5f * (9.0f * t2.z - 10.0f * t.z);
		w2.z = 0.5f * (-9.0f * t2.z + 8.0f * t.z + 1.0f);
		w3.z = 0.5f * (3.0f * t2.z - 2.0f * t.z);

		float3 texCoord;
		ResultType K[4];
		//#pragma unroll
		for(int k = -1; k <= 2; k++)
		{
			texCoord.z = index.z + k;
			ResultType J[4];
			//#pragma unroll
			for(int j = -1; j <= 2; j++)
			{
				texCoord.y = index.y + j;
				ResultType I[4];
				#pragma unroll
				for(int i = -1; i <= 2; i++)
				{
					texCoord.x = index.x + i;
					I[i+1] = make_floatn<TexType, ResultType>(tex3D(tex, texCoord.x, texCoord.y, texCoord.z));
				}
				J[j+1] = w0.x * I[0] + w1.x * I[1] + w2.x * I[2] + w3.x * I[3];
			}
			K[k+1] = w0.y * J[0] + w1.y * J[1] + w2.y * J[2] + w3.y * J[3];
		}
		return (w0.z * K[0] + w1.z * K[1] + w2.z * K[2] + w3.z * K[3]) / h;
	}
};

} // namespace (sampleVolumeDerivativeD implementations)

// "public" sampleVolumeDerivative functions:

template <eTextureFilterMode F, typename TexType, typename ResultType>
__device__ inline ResultType sampleVolumeDerivativeX(texture<float4, cudaTextureType3D, cudaReadModeElementType> tex, float3 coord, float h)
{
	return sampleVolumeDerivativeX_impl<F, TexType, ResultType>::exec(tex, coord.x, coord.y, coord.z, h);
}

template <eTextureFilterMode F, typename TexType, typename ResultType>
__device__ inline ResultType sampleVolumeDerivativeY(texture<float4, cudaTextureType3D, cudaReadModeElementType> tex, float3 coord, float h)
{
	return sampleVolumeDerivativeY_impl<F, TexType, ResultType>::exec(tex, coord.x, coord.y, coord.z, h);
}

template <eTextureFilterMode F, typename TexType, typename ResultType>
__device__ inline ResultType sampleVolumeDerivativeZ(texture<float4, cudaTextureType3D, cudaReadModeElementType> tex, float3 coord, float h)
{
	return sampleVolumeDerivativeZ_impl<F, TexType, ResultType>::exec(tex, coord.x, coord.y, coord.z, h);
}


namespace // sampleVolumeGradient implementations
{

template <eTextureFilterMode F>
struct sampleVolumeGradient_impl
{
	__device__ static inline float3x3 exec(texture<float4, cudaTextureType3D, cudaReadModeElementType> tex, float3 coord, float h)
	{
		// default implementation: just get derivatives in x, y, z
		// note: result is "transposed" by convention (i.e. first index is component, second index is derivative direction) - fix this some time?
		float3x3 J;

		// derivative in x direction
		float3 dx = sampleVolumeDerivativeX<F, float4, float3>(tex, coord, h);
		J.m[0].x = dx.x;
		J.m[1].x = dx.y;
		J.m[2].x = dx.z;

		// derivative in y direction
		float3 dy = sampleVolumeDerivativeY<F, float4, float3>(tex, coord, h);
		J.m[0].y = dy.x;
		J.m[1].y = dy.y;
		J.m[2].y = dy.z;

		// derivative in z direction
		float3 dz = sampleVolumeDerivativeZ<F, float4, float3>(tex, coord, h);
		J.m[0].z = dz.x;
		J.m[1].z = dz.y;
		J.m[2].z = dz.z;

		return J;
	}
};

template <>
struct sampleVolumeGradient_impl<TEXTURE_FILTER_CATROM>
{
	__device__ static inline float3x3 exec(texture<float4, cudaTextureType3D, cudaReadModeElementType> tex, float3 coord, float h)
	{
		// in CatRom case, do the 64 fetches only once and combine with different weights
		// note: result is "transposed" by convention (i.e. first index is component, second index is derivative direction) - fix this some time?

		// shift the coordinate from [0,extent] to [-0.5, extent-0.5] (so that grid points are at integers)
		const float3 coord_grid = coord - 0.5f;
		const float3 index = floor(coord_grid);
		const float3 t = coord_grid - index;

		// compute weights for grid points (cubic catmull-rom spline)
		const float3 t2 = t * t;
		const float3 t3 = t2 * t;

		const float3 w0 = 0.5f * (-t3 + 2.0f * t2 - t);
		const float3 w1 = 0.5f * (3.0f * t3 - 5.0f * t2 + 2.0f);
		const float3 w2 = 0.5f * (-3.0f * t3 + 4.0f * t2 + t);
		const float3 w3 = 0.5f * (t3 - t2);

		// weights for derivatives
		const float3 wd0 = 0.5f * (-3.0f * t2 + 4.0f * t - 1.0f);
		const float3 wd1 = 0.5f * (9.0f * t2 - 10.0f * t);
		const float3 wd2 = 0.5f * (-9.0f * t2 + 8.0f * t + 1.0f);
		const float3 wd3 = 0.5f * (3.0f * t2 - 2.0f * t);

		float3x3 result;
		float3 texCoord;
		float3 K [4];
		float3 Kx[4];
		float3 Ky[4];
		//#pragma unroll
		for(int k = -1; k <= 2; k++)
		{
			texCoord.z = index.z + k;
			float3 J [4];
			float3 Jx[4];
			//#pragma unroll
			for(int j = -1; j <= 2; j++)
			{
				texCoord.y = index.y + j;
				float3 I[4];
				#pragma unroll
				for(int i = -1; i <= 2; i++)
				{
					texCoord.x = index.x + i;
					I[i+1] = make_floatn<float4, float3>(tex3D(tex, texCoord.x, texCoord.y, texCoord.z));
				}
				J [j+1] =  w0.x * I[0] +  w1.x * I[1] +  w2.x * I[2] +  w3.x * I[3];
				Jx[j+1] = wd0.x * I[0] + wd1.x * I[1] + wd2.x * I[2] + wd3.x * I[3];
			}
			K [k+1] =  w0.y * J [0] +  w1.y * J [1] +  w2.y * J [2] +  w3.y * J [3];
			Kx[k+1] =  w0.y * Jx[0] +  w1.y * Jx[1] +  w2.y * Jx[2] +  w3.y * Jx[3];
			Ky[k+1] = wd0.y * J [0] + wd1.y * J [1] + wd2.y * J [2] + wd3.y * J [3];
		}
		result.m[0] = ( w0.z * Kx[0] +  w1.z * Kx[1] +  w2.z * Kx[2] +  w3.z * Kx[3]) / h;
		result.m[1] = ( w0.z * Ky[0] +  w1.z * Ky[1] +  w2.z * Ky[2] +  w3.z * Ky[3]) / h;
		result.m[2] = (wd0.z * K [0] + wd1.z * K [1] + wd2.z * K [2] + wd3.z * K [3]) / h;

		// transpose result
		float tmp;
		tmp = result.m[0].y; result.m[0].y = result.m[1].x; result.m[1].x = tmp;
		tmp = result.m[0].z; result.m[0].z = result.m[2].x; result.m[2].x = tmp;
		tmp = result.m[1].z; result.m[1].z = result.m[2].y; result.m[2].y = tmp;

		return result;
	}
};

template <eTextureFilterMode F>
struct sampleScalarGradient_impl
{
	__device__ static inline float3 exec(texture<float4, cudaTextureType3D, cudaReadModeElementType> tex, float3 coord, float h)
	{
		// default implementation: just get derivatives in x, y, z
		float3 grad;

		// derivative in x direction
		float4 dx = sampleVolumeDerivativeX<F, float4, float4>(tex, coord, h);
		grad.x = dx.w;

		// derivative in y direction
		float4 dy = sampleVolumeDerivativeY<F, float4, float4>(tex, coord, h);
		grad.y = dy.w;

		// derivative in z direction
		float4 dz = sampleVolumeDerivativeZ<F, float4, float4>(tex, coord, h);
		grad.z = dz.w;

		return grad;
	}
};

} // namespace (sampleVolumeGradient implementations)

// "public" sampleVolumeGradient function:

template <eTextureFilterMode F>
__device__ inline float3x3 sampleVolumeGradient(texture<float4, cudaTextureType3D, cudaReadModeElementType> tex, float3 coord, float h)
{
	return sampleVolumeGradient_impl<F>::exec(tex, coord, h);
}

template <eTextureFilterMode F>
__device__ inline float3 sampleScalarGradient(texture<float4, cudaTextureType3D, cudaReadModeElementType> tex, float3 coord, float h)
{
	return sampleScalarGradient_impl<F>::exec(tex, coord, h);
}



#endif
