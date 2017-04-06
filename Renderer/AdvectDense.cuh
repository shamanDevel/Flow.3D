#ifndef __TUM3D__ADVECT_DENSE_CUH__
#define __TUM3D__ADVECT_DENSE_CUH__


// particle advection with dense output (output is a polynomial instead of just a point)


#include "cutil_math.h"
#include "AdvectMode.h"
#include "IntegrationParamsGPU.h"
#include "TextureFilterMode.h"

#include "Coords.cuh"
#include "TextureFilter.cuh"


extern __constant__ IntegrationParamsGPU c_integrationParams;


// hacky helpers to save typing/make things more readable
#define sample sampleVolume<filterMode, float4, float3>
#define getVelocity(pos) (velocityScale * sample(tex, w2t(pos)))

// maximum adjustment of deltaTime per step
#define SCALE_MIN 0.125f
#define SCALE_MAX 4.0f
#define SCALE_SAFETY_FACTOR 0.8f


template <eAdvectMode advectMode>
struct advectDenseInfo
{
};

template<>
struct advectDenseInfo<ADVECT_RK547M>
{
	static const uint OutputCoeffCount = 5;
};


template <eAdvectMode advectMode, eTextureFilterMode filterMode>
struct advectDense_impl
{
	//__device__ static inline bool exec(texture<float4, cudaTextureType3D, cudaReadModeElementType> tex,
	//	float3& pos, float& age, float3& vel, float& deltaTime,
	//	float3* pOutputCoeffs,
	//	const float3& world2texOffset, const float world2texScale,
	//	const float velocityScale);
	// no implementation - only specializations allowed
};

template <eTextureFilterMode filterMode>
struct advectDense_impl<ADVECT_RK547M, filterMode>
{
	__device__ static inline bool exec(texture<float4, cudaTextureType3D, cudaReadModeElementType> tex,
		float3& pos, float& age, float3& vel, float& deltaTime,
		float3* pOutputCoeffs,
		const float3& world2texOffset, const float3& world2texScale,
		const float3& velocityScale)
	{
		const float b21 = 0.2f;
		const float b31 = 3.0f / 40.0f;
		const float b32 = 9.0f / 40.0f;
		const float b41 = 44.0f / 45.0f;
		const float b42 = -56.0f / 15.0f;
		const float b43 = 32.0f / 9.0f;
		const float b51 = 19372.0f / 6561.0f;
		const float b52 = -25360.0f / 2187.0f;
		const float b53 = 64448.0f / 6561.0f;
		const float b54 = -212.0f / 729.0f;
		const float b61 = 9017.0f / 3168.0f;
		const float b62 = -355.0f / 33.0f;
		const float b63 = 46732.0f / 5247.0f;
		const float b64 = 49.0f / 176.0f;
		const float b65 = -5103.0f / 18656.0f;
		const float b71 = 35.0f / 384.0f;
		const float b73 = 500.0f / 1113.0f;
		const float b74 = 125.0f / 192.0f;
		const float b75 = -2187.0f / 6784.0f;
		const float b76 = 11.0f / 84.0f;

		// FSAL: c_i == b_7i
		//const float c1 = b71;
		//const float c3 = b73;
		//const float c4 = b74;
		//const float c5 = b75;
		//const float c6 = b76;

		const float d1 = 71.0f / 57600.0f;
		const float d3 = -71.0f / 16695.0f;
		const float d4 = 71.0f / 1920.0f;
		const float d5 = -17253.0f / 339200.0f;
		const float d6 = 22.0f / 525.0f;
		const float d7 = -1.0f / 40.0f;

		const float p1 = float(-12715105075.0 / 11282082432.0 + 1.0);
		const float p3 = float(87487479700.0 / 32700410799.0);
		const float p4 = float(-10690763975.0 / 1880347072.0); 
		const float p5 = float(701980252875.0 / 199316789632.0);
		const float p6 = float(-1453857185.0 / 822651844.0);
		const float p7 = float(69997945.0 / 29380423.0 - 1.0);

		float3 vel1 = vel;
		float3 vel2 = getVelocity(pos + deltaTime * (b21 * vel1));
		float3 vel3 = getVelocity(pos + deltaTime * (b31 * vel1 + b32 * vel2));
		float3 vel4 = getVelocity(pos + deltaTime * (b41 * vel1 + b42 * vel2 + b43 * vel3));
		float3 vel5 = getVelocity(pos + deltaTime * (b51 * vel1 + b52 * vel2 + b53 * vel3 + b54 * vel4));
		float3 vel6 = getVelocity(pos + deltaTime * (b61 * vel1 + b62 * vel2 + b63 * vel3 + b64 * vel4 + b65 * vel5));
		float3 deltaVel = deltaTime * (b71 * vel1 + b73 * vel3 + b74 * vel4 + b75 * vel5 + b76 * vel6);
		float3 vel7 = getVelocity(pos + deltaVel); // == vel1 in the next step (if step is accepted)

		float3 errorVec = d1*vel1 + d3*vel3 + d4*vel4 + d5*vel5 + d6*vel6 + d7*vel7;
		float errorSquared = dot(errorVec, errorVec);
		bool result = false;
		if(errorSquared < c_integrationParams.toleranceSquared || deltaTime <= c_integrationParams.deltaTimeMin) {
			pOutputCoeffs[0] = pos;
			pOutputCoeffs[1] = pos + deltaTime * vel1 / 4.0f;
			pOutputCoeffs[2] = pos + 0.5f * deltaVel + deltaTime * (p1*vel1 + p3*vel3 + p4*vel4 + p5*vel5 + p6*vel6 + p7*vel7) / 6.0f;
			pos += deltaVel;
			pOutputCoeffs[3] = pos - deltaTime * vel7 / 4.0f;
			pOutputCoeffs[4] = pos;
			age += deltaTime;
			vel = vel7;
			result = true;
		}
		float scale = SCALE_SAFETY_FACTOR * pow(c_integrationParams.toleranceSquared / errorSquared, (1.0f / 5.0f) * 0.5f); // (tolerance/error)^(1/order)
		scale = clamp(scale, SCALE_MIN, SCALE_MAX);
		deltaTime *= scale;
		deltaTime = clamp(deltaTime, c_integrationParams.deltaTimeMin, c_integrationParams.deltaTimeMax);

		return result;
	}
};

#undef SCALE_SAFETY_FACTOR
#undef SCALE_MAX
#undef SCALE_MIN

#undef getVelocity
#undef sample


template <eAdvectMode advectMode, eTextureFilterMode filterMode>
__device__ inline bool advectDense(texture<float4, cudaTextureType3D, cudaReadModeElementType> tex,
	float3& pos, float& age, float3& vel, float& deltaTime,
	float3* pOutputCoeffs,
	const float3& world2texOffset, const float3& world2texScale,
	const float3& velocityScale)
{
	return advectDense_impl<advectMode, filterMode>::exec(
		tex,
		pos, age, vel, deltaTime,
		pOutputCoeffs,
		world2texOffset, world2texScale,
		velocityScale);
}


#endif
