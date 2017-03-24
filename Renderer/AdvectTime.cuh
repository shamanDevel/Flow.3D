#ifndef __TUM3D__ADVECT_TIME_CUH__
#define __TUM3D__ADVECT_TIME_CUH__


#include "cutil_math.h"
#include "AdvectMode.h"
#include "IntegrationParamsGPU.h"
#include "TextureFilterMode.h"

#include "Coords.cuh"
#include "TextureFilterTime.cuh"


extern __constant__ IntegrationParamsGPU c_integrationParams;


// hacky helpers to save typing/make things more readable
#define sample sampleVolumeTime<filterMode, float4, float3>
#define getVelocity(pos, time) (velocityScale * sample(tex, w2t(pos), time2tex(time), timestepInc))

// maximum adjustment of deltaTime per step
#define SCALE_MIN 0.125f
#define SCALE_MAX 4.0f
#define SCALE_SAFETY_FACTOR 0.8f


template <eAdvectMode advectMode, eTextureFilterMode filterMode>
struct advectTime_impl
{
	// no exec() implementation - only specializations allowed
};


template <eTextureFilterMode filterMode>
struct advectTime_impl<ADVECT_EULER, filterMode>
{
	__device__ static inline bool exec(texture<float4, cudaTextureType3D, cudaReadModeElementType> tex,
		float3& pos, float& time, float3& vel, float& deltaTime,
		const float3& world2texOffset, const float world2texScale,
		const float time2texOffset, const float time2texScale, const float timestepInc,
		const float velocityScale)
	{
		pos += vel * deltaTime;
		time += deltaTime;

		vel = getVelocity(pos, time);

		return true;
	}
};

template <eTextureFilterMode filterMode>
struct advectTime_impl<ADVECT_HEUN, filterMode>
{
	__device__ static inline bool exec(texture<float4, cudaTextureType3D, cudaReadModeElementType> tex,
		float3& pos, float& time, float3& vel, float& deltaTime,
		const float3& world2texOffset, const float world2texScale,
		const float time2texOffset, const float time2texScale, const float timestepInc,
		const float velocityScale)
	{
		float3 velocity0 = vel;
		float3 velocity1 = getVelocity(pos + velocity0 * deltaTime, time + deltaTime);

		float3 velocity = 0.5f * (velocity0 + velocity1);

		pos += velocity * deltaTime;
		time += deltaTime;

		vel = getVelocity(pos, time);

		return true;
	}
};

template <eTextureFilterMode filterMode>
struct advectTime_impl<ADVECT_RK3, filterMode>
{
	__device__ static inline bool exec(texture<float4, cudaTextureType3D, cudaReadModeElementType> tex,
		float3& pos, float& time, float3& vel, float& deltaTime,
		const float3& world2texOffset, const float world2texScale,
		const float time2texOffset, const float time2texScale, const float timestepInc,
		const float velocityScale)
	{
		float3 velocity0 = vel;
		float3 velocity1 = getVelocity(pos + velocity0 * 0.5f * deltaTime, time + 0.5f * deltaTime);
		float3 velocity2 = getVelocity(pos + (velocity1 + velocity1 - velocity0) * deltaTime, time + deltaTime);

		float3 velocity = (1.0f/6.0f) * (velocity0 + 4.0f * velocity1 + velocity2);

		pos += velocity * deltaTime;
		time += deltaTime;

		vel = getVelocity(pos, time);

		return true;
	}
};

template <eTextureFilterMode filterMode>
struct advectTime_impl<ADVECT_RK4, filterMode>
{
	__device__ static inline bool exec(texture<float4, cudaTextureType3D, cudaReadModeElementType> tex,
		float3& pos, float& time, float3& vel, float& deltaTime,
		const float3& world2texOffset, const float world2texScale,
		const float time2texOffset, const float time2texScale, const float timestepInc,
		const float velocityScale)
	{
		float3 velocity0 = vel;
		float3 velocity1 = getVelocity(pos + velocity0 * 0.5f * deltaTime, time + 0.5f * deltaTime);
		float3 velocity2 = getVelocity(pos + velocity1 * 0.5f * deltaTime, time + 0.5f * deltaTime);
		float3 velocity3 = getVelocity(pos + velocity2 * deltaTime, time + deltaTime);

		float3 velocity = (1.0f/6.0f) * (velocity0 + 2.0f * velocity1 + 2.0f * velocity2 + velocity3);

		pos += velocity * deltaTime;
		time += deltaTime;

		vel = getVelocity(pos, time);

		return true;
	}
};


template <eTextureFilterMode filterMode>
struct advectTime_impl<ADVECT_BS32, filterMode>
{
	__device__ static inline bool exec(texture<float4, cudaTextureType3D, cudaReadModeElementType> tex,
		float3& pos, float& time, float3& vel, float& deltaTime,
		const float3& world2texOffset, const float world2texScale,
		const float time2texOffset, const float time2texScale, const float timestepInc,
		const float velocityScale)
	{
		const float a2 = 1.0f / 2.0f;
		const float a3 = 3.0f / 4.0f;

		const float b21 = 1.0f / 2.0f;
		const float b32 = 3.0f / 4.0f;
		const float b41 = 2.0f / 9.0f;
		const float b42 = 1.0f / 3.0f;
		const float b43 = 4.0f / 9.0f;

		// FSAL: c_i == b_4i

		const float d1 = -5.0f / 72.0f;
		const float d2 = 1.0f / 12.0f;
		const float d3 = 1.0f / 9.0f;
		const float d4 = -1.0f / 8.0f;

		float3 vel1 = vel;
		float3 vel2 = getVelocity(pos + deltaTime * (b21 * vel1), time + a2 * deltaTime);
		float3 vel3 = getVelocity(pos + deltaTime * (b32 * vel2), time + a3 * deltaTime);
		float3 deltaVel = deltaTime * (b41 * vel1 + b42 * vel2 + b43 * vel3);
		float3 vel4 = getVelocity(pos + deltaVel, time + deltaTime); // == vel1 in the next step (if step is accepted)

		float3 errorVec = d1*vel1 + d2*vel2 + d3*vel3 + d4*vel4;
		float errorSquared = dot(errorVec, errorVec);
		bool result = false;
		if(errorSquared < c_integrationParams.toleranceSquared || deltaTime <= c_integrationParams.deltaTimeMin) {
			pos += deltaVel;
			time += deltaTime;
			vel = vel4;
			result = true;
		}
		float scale = SCALE_SAFETY_FACTOR * pow(c_integrationParams.toleranceSquared / errorSquared, (1.0f / 3.0f) * 0.5f); // (tolerance/error)^(1/order)
		scale = clamp(scale, SCALE_MIN, SCALE_MAX);
		deltaTime *= scale;
		deltaTime = clamp(deltaTime, c_integrationParams.deltaTimeMin, c_integrationParams.deltaTimeMax);

		return result;
	}
};

template <eTextureFilterMode filterMode>
struct advectTime_impl<ADVECT_RKF34, filterMode>
{
	__device__ static inline bool exec(texture<float4, cudaTextureType3D, cudaReadModeElementType> tex,
		float3& pos, float& time, float3& vel, float& deltaTime,
		const float3& world2texOffset, const float world2texScale,
		const float time2texOffset, const float time2texScale, const float timestepInc,
		const float velocityScale)
	{
		//const float a2 = 2.0f / 7.0f;
		//const float a3 = 7.0f / 15.0f;
		//const float a4 = 35.0f / 38.0f;

		const float b21 = 2.0f / 7.0f;
		const float b31 = 77.0f / 900.0f;
		const float b32 = 343.0f / 900.0f;
		const float b41 = 805.0f / 1444.0f;
		const float b42 = -77175.0f / 54872.0f;
		const float b43 = 97125.0f / 54872.0f;
		const float b51 = 79.0f / 490.0f;
		const float b53 = 2175.0f / 3626.0f;
		const float b54 = 2166.0f / 9065.0f;

		const float c1 = 229.0f / 1470.0f;
		const float c3 = 1125.0f / 1813.0f;
		const float c4 = 13718.0f / 81585.0f;
		const float c5 = 1.0f / 18.0f;

		const float d1 = -888.0f / 163170.0f;
		const float d3 = 3375.0f / 163170.0f;
		const float d4 = -11552.0f / 163170.0f;
		const float d5 = 9065.0f / 163170.0f;

		// h2 = a2 * h;
		float3 vel1 = getVelocity(pos, 0); //(*f)(x0, *y);
		float3 vel2 = getVelocity(pos + deltaTime * (b21 * vel1), 0); //(*f)(x0+h2, *y + h2 * k1);
		float3 vel3 = getVelocity(pos + deltaTime * (b31 * vel1 + b32 * vel2), 0); //(*f)(x0+a3*h, *y + h * ( b31*k1 + b32*k2) );
		float3 vel4 = getVelocity(pos + deltaTime * (b41 * vel1 + b42 * vel2 + b43 * vel3), 0); //(*f)(x0+a4*h, *y + h * ( b41*k1 + b42*k2 + b43*k3) );
		float3 vel5 = getVelocity(pos + deltaTime * (b51 * vel1 + b53 * vel3 + b54 * vel4), 0); //(*f)(x0+h,  *y + h * ( b51*k1 + b53*k3 + b54*k4) );

		//error = d1*k1 + d3*k3 + d4*k4 + d5*k5;
		float3 errorVec = d1*vel1 + d3*vel3 + d4*vel4 + d5*vel5;
		float errorSquared = dot(errorVec, errorVec);
		bool result = false;
		if(errorSquared < c_integrationParams.toleranceSquared || deltaTime <= c_integrationParams.deltaTimeMin) {
			//*(y+1) = *y +  h * (c1*k1 + c3*k3 + c4*k4 + c5*k5);
			pos += deltaTime * (c1*vel1 + c3*vel3 + c4*vel4 + c5*vel5);
			time += deltaTime;
			result = true;
		}
		float scale = SCALE_SAFETY_FACTOR * pow(c_integrationParams.toleranceSquared / errorSquared, (1.0f / 3.0f) * 0.5f); // (tolerance/error)^(1/order)
		scale = clamp(scale, SCALE_MIN, SCALE_MAX);
		deltaTime *= scale;
		deltaTime = clamp(deltaTime, c_integrationParams.deltaTimeMin, c_integrationParams.deltaTimeMax);

		return result;
	}
};

template <eTextureFilterMode filterMode>
struct advectTime_impl<ADVECT_RKF45, filterMode>
{
	__device__ static inline bool exec(texture<float4, cudaTextureType3D, cudaReadModeElementType> tex,
		float3& pos, float& time, float3& vel, float& deltaTime,
		const float3& world2texOffset, const float world2texScale,
		const float time2texOffset, const float time2texScale, const float timestepInc,
		const float velocityScale)
	{
		const float a2 = 0.25f;
		const float a3 = 0.375f;
		const float a4 = 12.0f / 13.0f;
		//const float a5 = 1.0f;
		const float a6 = 0.5f;

		const float b21 = 0.25f;
		const float b31 = 3.0f / 32.0f;
		const float b32 = 9.0f / 32.0f;
		const float b41 = 1932.0f / 2197.0f;
		const float b42 = -7200.0f / 2197.0f;
		const float b43 = 7296.0f / 2197.0f;
		const float b51 = 439.0f / 216.0f;
		const float b52 = -8.0f;
		const float b53 = 3680.0f / 513.0f;
		const float b54 = -845.0f / 4104.0f;
		const float b61 = -8.0f / 27.0f;
		const float b62 = 2.0f;
		const float b63 = -3544.0f / 2565.0f;
		const float b64 = 1859.0f / 4104.0f;
		const float b65 = -11.0f / 40.0f;

		const float c1 = 25.0f / 216.0f;
		const float c3 = 1408.0f / 2565.0f;
		const float c4 = 2197.0f / 4104.0f;
		const float c5 = -0.20f;

		const float d1 = 1.0f / 360.0f;
		const float d3 = -128.0f / 4275.0f;
		const float d4 = -2197.0f / 75240.0f;
		const float d5 = 0.02f;
		const float d6 = 2.0f / 55.0f;

		//h2 = a2 * h, h3 = a3 * h, h4 = a4 * h, h6 = a6 * h;
		float3 vel1 = vel; // k1 = (*f)(x0, *y);
		float3 vel2 = getVelocity(pos + deltaTime * (b21 * vel1), time + a2 * deltaTime); // k2 = (*f)(x0+h2, *y + h * b21 * k1);
		float3 vel3 = getVelocity(pos + deltaTime * (b31 * vel1 + b32 * vel2), time + a3 * deltaTime); // k3 = (*f)(x0+h3, *y + h * ( b31*k1 + b32*k2) );
		float3 vel4 = getVelocity(pos + deltaTime * (b41 * vel1 + b42 * vel2 + b43 * vel3), time + a4 * deltaTime); // k4 = (*f)(x0+h4, *y + h * ( b41*k1 + b42*k2 + b43*k3) );
		float3 vel5 = getVelocity(pos + deltaTime * (b51 * vel1 + b52 * vel2 + b53 * vel3 + b54 * vel4), time + deltaTime); // k5 = (*f)(x0+h,  *y + h * ( b51*k1 + b52*k2 + b53*k3 + b54*k4) );
		float3 vel6 = getVelocity(pos + deltaTime * (b61 * vel1 + b62 * vel2 + b63 * vel3 + b64 * vel4 + b65 * vel5), time + a6 * deltaTime); // k6 = (*f)(x0+h6, *y + h * ( b61*k1 + b62*k2 + b63*k3 + b64*k4 + b65*k5) );

		//error = d1*k1 + d3*k3 + d4*k4 + d5*k5 + d6*k6;
		float3 errorVec = d1*vel1 + d3*vel3 + d4*vel4 + d5*vel5 + d6*vel6;
		float errorSquared = dot(errorVec, errorVec);
		bool result = false;
		if(errorSquared < c_integrationParams.toleranceSquared || deltaTime <= c_integrationParams.deltaTimeMin) {
			//*(y+1) = *y +  h * (c1*k1 + c3*k3 + c4*k4 + c5*k5);
			pos += deltaTime * (c1*vel1 + c3*vel3 + c4*vel4 + c5*vel5);
			time += deltaTime;
			vel = getVelocity(pos, time);
			result = true;
		}
		float scale = SCALE_SAFETY_FACTOR * pow(c_integrationParams.toleranceSquared / errorSquared, (1.0f / 4.0f) * 0.5f); // (tolerance/error)^(1/order)
		scale = clamp(scale, SCALE_MIN, SCALE_MAX);
		deltaTime *= scale;
		deltaTime = clamp(deltaTime, c_integrationParams.deltaTimeMin, c_integrationParams.deltaTimeMax);

		return result;
	}
};

template <eTextureFilterMode filterMode>
struct advectTime_impl<ADVECT_RKF54, filterMode>
{
	__device__ static inline bool exec(texture<float4, cudaTextureType3D, cudaReadModeElementType> tex,
		float3& pos, float& time, float3& vel, float& deltaTime,
		const float3& world2texOffset, const float world2texScale,
		const float time2texOffset, const float time2texScale, const float timestepInc,
		const float velocityScale)
	{
		const float a2 = 0.25f;
		const float a3 = 0.375f;
		const float a4 = 12.0f / 13.0f;
		//const float a5 = 1.0f;
		const float a6 = 0.5f;

		const float b21 = 0.25f;
		const float b31 = 3.0f / 32.0f;
		const float b32 = 9.0f / 32.0f;
		const float b41 = 1932.0f / 2197.0f;
		const float b42 = -7200.0f / 2197.0f;
		const float b43 = 7296.0f / 2197.0f;
		const float b51 = 439.0f / 216.0f;
		const float b52 = -8.0f;
		const float b53 = 3680.0f / 513.0f;
		const float b54 = -845.0f / 4104.0f;
		const float b61 = -8.0f / 27.0f;
		const float b62 = 2.0f;
		const float b63 = -3544.0f / 2565.0f;
		const float b64 = 1859.0f / 4104.0f;
		const float b65 = -11.0f / 40.0f;

		const float c1 = 16.0f / 135.0f;
		const float c3 = 6656.0f / 12825.0f;
		const float c4 = 28561.0f / 56430.0f;
		const float c5 = -9.0f / 50.0f;
		const float c6 = 2.0f / 55.0f;

		const float d1 = 1.0f / 360.0f;
		const float d3 = -128.0f / 4275.0f;
		const float d4 = -2197.0f / 75240.0f;
		const float d5 = 0.02f;
		const float d6 = 2.0f / 55.0f;

		//h2 = a2 * h, h3 = a3 * h, h4 = a4 * h, h6 = a6 * h;
		float3 vel1 = vel; // k1 = (*f)(x0, *y);
		float3 vel2 = getVelocity(pos + deltaTime * (b21 * vel1), time + a2 * deltaTime); // k2 = (*f)(x0+h2, *y + h * b21 * k1);
		float3 vel3 = getVelocity(pos + deltaTime * (b31 * vel1 + b32 * vel2), time + a3 * deltaTime); // k3 = (*f)(x0+h3, *y + h * ( b31*k1 + b32*k2) );
		float3 vel4 = getVelocity(pos + deltaTime * (b41 * vel1 + b42 * vel2 + b43 * vel3), time + a4 * deltaTime); // k4 = (*f)(x0+h4, *y + h * ( b41*k1 + b42*k2 + b43*k3) );
		float3 vel5 = getVelocity(pos + deltaTime * (b51 * vel1 + b52 * vel2 + b53 * vel3 + b54 * vel4), time + deltaTime); // k5 = (*f)(x0+h,  *y + h * ( b51*k1 + b52*k2 + b53*k3 + b54*k4) );
		float3 vel6 = getVelocity(pos + deltaTime * (b61 * vel1 + b62 * vel2 + b63 * vel3 + b64 * vel4 + b65 * vel5), time + a6 * deltaTime); // k6 = (*f)(x0+h6, *y + h * ( b61*k1 + b62*k2 + b63*k3 + b64*k4 + b65*k5) );

		//error = d1*k1 + d3*k3 + d4*k4 + d5*k5 + d6*k6;
		float3 errorVec = d1*vel1 + d3*vel3 + d4*vel4 + d5*vel5 + d6*vel6;
		float errorSquared = dot(errorVec, errorVec);
		bool result = false;
		if(errorSquared < c_integrationParams.toleranceSquared || deltaTime <= c_integrationParams.deltaTimeMin) {
			//*(y+1) = *y +  h * (c1*k1 + c3*k3 + c4*k4 + c5*k5 + c6*k6);
			pos += deltaTime * (c1*vel1 + c3*vel3 + c4*vel4 + c5*vel5 + c6*vel6);
			time += deltaTime;
			vel = getVelocity(pos, time);
			result = true;
		}
		float scale = SCALE_SAFETY_FACTOR * pow(c_integrationParams.toleranceSquared / errorSquared, (1.0f / 5.0f) * 0.5f); // (tolerance/error)^(1/order)
		scale = clamp(scale, SCALE_MIN, SCALE_MAX);
		deltaTime *= scale;
		deltaTime = clamp(deltaTime, c_integrationParams.deltaTimeMin, c_integrationParams.deltaTimeMax);

		return result;
	}
};

template <eTextureFilterMode filterMode>
struct advectTime_impl<ADVECT_RK547M, filterMode>
{
	__device__ static inline bool exec(texture<float4, cudaTextureType3D, cudaReadModeElementType> tex,
		float3& pos, float& time, float3& vel, float& deltaTime,
		const float3& world2texOffset, const float world2texScale,
		const float time2texOffset, const float time2texScale, const float timestepInc,
		const float velocityScale)
	{
		const float a2 = 0.2f;
		const float a3 = 0.3f;
		const float a4 = 0.8f;
		const float a5 = 8.0f / 9.0f;
		//const float a6 = a7 = 1.0f;

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

		float3 vel1 = vel;
		float3 vel2 = getVelocity(pos + deltaTime * (b21 * vel1), time + a2 * deltaTime);
		float3 vel3 = getVelocity(pos + deltaTime * (b31 * vel1 + b32 * vel2), time + a3 * deltaTime);
		float3 vel4 = getVelocity(pos + deltaTime * (b41 * vel1 + b42 * vel2 + b43 * vel3), time + a4 * deltaTime);
		float3 vel5 = getVelocity(pos + deltaTime * (b51 * vel1 + b52 * vel2 + b53 * vel3 + b54 * vel4), time + a5 * deltaTime);
		float3 vel6 = getVelocity(pos + deltaTime * (b61 * vel1 + b62 * vel2 + b63 * vel3 + b64 * vel4 + b65 * vel5), time + deltaTime);
		float3 deltaVel = deltaTime * (b71 * vel1 + b73 * vel3 + b74 * vel4 + b75 * vel5 + b76 * vel6);
		float3 vel7 = getVelocity(pos + deltaVel, time + deltaTime); // == vel1 in the next step (if step is accepted)

		float3 errorVec = d1*vel1 + d3*vel3 + d4*vel4 + d5*vel5 + d6*vel6 + d7*vel7;
		float errorSquared = dot(errorVec, errorVec);
		bool result = false;
		if(errorSquared < c_integrationParams.toleranceSquared || deltaTime <= c_integrationParams.deltaTimeMin) {
			pos += deltaVel;
			time += deltaTime;
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
__device__ inline bool advectTime(texture<float4, cudaTextureType3D, cudaReadModeElementType> tex,
	float3& pos, float& time, float3& vel, float& deltaTime,
	const float3& world2texOffset, const float world2texScale,
	const float time2texOffset, const float time2texScale, const float timestepInc,
	const float velocityScale)
{
	return advectTime_impl<advectMode, filterMode>::exec(
		tex,
		pos, time, vel, deltaTime,
		world2texOffset, world2texScale,
		time2texOffset, time2texScale, timestepInc,
		velocityScale);
}


#endif
