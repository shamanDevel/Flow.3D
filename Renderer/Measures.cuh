#ifndef __TUM3D__MEASURES_CUH__
#define __TUM3D__MEASURES_CUH__


#include "Measure.h"

#include "Jacobian.cuh"
#include "HeatCurrent.cuh"
#include "MatrixMath.cuh"

/******************************************************************************
** Helper functions
******************************************************************************/

#define TUM3D_PI (3.14159265f)


__device__ inline float getSign(const float value)
{
	return signbit(value) ? -1.0f : 1.0f;
}


/******************************************************************************
** (Invariant) Measures:
******************************************************************************/

__device__ inline float3 getVorticity(const float3x3& J)
{
	return make_float3(J.m[2].y - J.m[1].z, J.m[0].z - J.m[2].x, J.m[1].x - J.m[0].y);
}

template <eTextureFilterMode F>
__device__ inline float3 getVorticity(cudaTextureObject_t tex, const float3 &pos)
{
	float3x3 jacobian = getJacobian<F>(tex, pos);
	return getVorticity(jacobian);
}

__device__ float getLambda2(const float3x3& J);
__device__ float getQHunt(const float3x3& J);
__device__ float getDeltaChong(const float3x3& J);
__device__ float getSquareRotation(const float3x3& J);
__device__ float getEnstrophyProduction(const float3x3& J);
__device__ float getStrainProduction(const float3x3& J);
__device__ float getSquareRateOfStrain(const float3x3& J);
__device__ float getPVA(const float3x3 &J);



/*************************************************************************************************************************************
** Scalar measure (of the velocity field, its gradient and derived tensors)
*************************************************************************************************************************************/

// runtime switched versions
__device__ inline float getMeasureFromRaw(eMeasure measure, const float4 vel4)
{
	switch(measure)
	{
		case MEASURE_VELOCITY:
			return length(make_float3(vel4));
		case MEASURE_VELOCITY_Z:
			return vel4.z;
		case MEASURE_TEMPERATURE:
			return vel4.w;
		default:
			//assert(false) ?
			return 0.0f;
	}
}

__device__ inline float getMeasureFromHeatCurrent(eMeasure measure, const float3 heatCurrent)
{
	switch (measure)
	{
	case MEASURE_HEAT_CURRENT:
		return length(heatCurrent);
	case MEASURE_HEAT_CURRENT_X:
		return heatCurrent.x;
	case MEASURE_HEAT_CURRENT_Y:
		return heatCurrent.y;
	case MEASURE_HEAT_CURRENT_Z:
		return heatCurrent.z;
	default:
		//assert(false) ?
		return 0.0f;
	}
}

__device__ inline float getMeasureFromJac(eMeasure measure, const float3x3& jacobian)
{
	switch(measure)
	{
		case MEASURE_VORTICITY:
			return length(getVorticity(jacobian));
		case MEASURE_LAMBDA2:
			return getLambda2(jacobian);
		case MEASURE_QHUNT:
			return getQHunt(jacobian);
		case MEASURE_DELTACHONG:
			return getDeltaChong(jacobian);
		case MEASURE_ENSTROPHY_PRODUCTION:
			return getEnstrophyProduction(jacobian);
		case MEASURE_STRAIN_PRODUCTION:
			return getStrainProduction(jacobian);
		case MEASURE_SQUARE_ROTATION:
			return getSquareRotation(jacobian);
		case MEASURE_SQUARE_RATE_OF_STRAIN:
			return getSquareRateOfStrain(jacobian);
		case MEASURE_TRACE_JJT:
			return TraceAAT(jacobian);
		case MEASURE_PVA:
			return getPVA(jacobian);
		default:
			//assert(false) ?
			return 0.0f;
	}
}

// interface function - runtime switched
template <eMeasureSource source, eTextureFilterMode F, eMeasureComputeMode C>
struct getMeasureFromVolume_Impl2
{
    __device__ static inline float exec(eMeasure measure, cudaTextureObject_t tex, float3 pos, float3 h);
	// no implementation - only specializations allowed
};

template <eMeasureSource source, eTextureFilterMode F>
struct getMeasureFromVolume_Impl2<source, F, MEASURE_COMPUTE_PRECOMP_DISCARD>
{
    __device__ static inline float exec(eMeasure measure, cudaTextureObject_t tex, float3 pos, float3 h)
	{
		// precomputed measure volume: just read value
		return sampleVolume<F, float1, float>(g_texFeatureVolume, pos);
	}
};

template <eTextureFilterMode F>
struct getMeasureFromVolume_Impl2<MEASURE_SOURCE_RAW, F, MEASURE_COMPUTE_ONTHEFLY>
{
    __device__ static inline float exec(eMeasure measure, cudaTextureObject_t tex, float3 pos, float3 h)
	{
		float4 vel4 = sampleVolume<F, float4, float4>(tex, pos);
		return getMeasureFromRaw(measure, vel4);
	}
};

template <eTextureFilterMode F>
struct getMeasureFromVolume_Impl2<MEASURE_SOURCE_HEAT_CURRENT, F, MEASURE_COMPUTE_ONTHEFLY>
{
	__device__ static inline float exec(eMeasure measure, cudaTextureObject_t tex, float3 pos, float3 h)
	{
		float3 heatCurrent = getHeatCurrent<F>(tex, pos, h);
		return getMeasureFromHeatCurrent(measure, heatCurrent);
	}
};

template <eTextureFilterMode F>
struct getMeasureFromVolume_Impl2<MEASURE_SOURCE_JACOBIAN, F, MEASURE_COMPUTE_ONTHEFLY>
{
    __device__ static inline float exec(eMeasure measure, cudaTextureObject_t tex, float3 pos, float3 h)
	{
		float3x3 jacobian = getJacobian<F>(tex, pos, h);
		return getMeasureFromJac(measure, jacobian);
	}
};


template <eMeasureSource source, eTextureFilterMode F, eMeasureComputeMode C>
__device__ inline float getMeasure(eMeasure measure, cudaTextureObject_t tex, float3 pos, float3 h, float measureScale)
{
	return (measureScale * getMeasureFromVolume_Impl2<source, F, C>::exec(measure, tex, pos, h));
}




#endif
