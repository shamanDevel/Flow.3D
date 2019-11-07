#include <vector>
#include <cutil_math.h>


float4 sampleVolume(const std::vector<std::vector<float>>& tex, int x, int y, int z) {
    // TODO
    return float4{0.0f, 0.0f, 0.0f, 0.0f};
}

inline float sampleVolumeDerivativeX(const std::vector<std::vector<float>>& tex, int x, int y, int z, float h)
{
	// default implementation: central differences
	float dp = sampleVolume(tex, x + 1, y, z);
	float dn = sampleVolume(tex, x - 1, y, z);
	return (dp - dn) / (2.0f * h);
}

inline float sampleVolumeDerivativeY(const std::vector<std::vector<float>>& tex, int x, int y, int z, float h)
{
	// default implementation: central differences
	float dp = sampleVolume(tex, x, y + 1, z);
	float dn = sampleVolume(tex, x, y - 1, z);
	return (dp - dn) / (2.0f * h);
}

inline float sampleVolumeDerivativeY(const std::vector<std::vector<float>>& tex, int x, int y, int z, float h)
{
	// default implementation: central differences
	float dp = sampleVolume(tex, x, y, z + 1);
	float dn = sampleVolume(tex, x, y, z - 1);
	return (dp - dn) / (2.0f * h);
}

float3x3 getJacobian(int idx) {
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


/******************************************************************************
** Heat current and derived quantities
******************************************************************************/


// get heat current from velocity/temperature texture
inline float3 getHeatCurrent(texture<float4, cudaTextureType3D, cudaReadModeElementType> tex, const float3& texCoord, const float3& h)
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


inline float getHeatCurrentAlignment(texture<float4, cudaTextureType3D, cudaReadModeElementType> tex, const float3& texCoord, const float& h)
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


/******************************************************************************
** Helper functions
******************************************************************************/

#define TUM3D_PI (3.14159265f)


inline float getSign(const float value)
{
	return signbit(value) ? -1.0f : 1.0f;
}


/******************************************************************************
** (Invariant) Measures:
******************************************************************************/

inline float3 getVorticity(const float3x3& J)
{
	return make_float3(J.m[2].y - J.m[1].z, J.m[0].z - J.m[2].x, J.m[1].x - J.m[0].y);
}

inline float3 getVorticity(texture<float4, cudaTextureType3D, cudaReadModeElementType> tex, const float3 &pos)
{
	float3x3 jacobian = getJacobian<F>(tex, pos);
	return getVorticity(jacobian);
}

float getLambda2(const float3x3& J);
float getQHunt(const float3x3& J);
float getDeltaChong(const float3x3& J);
float getSquareRotation(const float3x3& J);
float getEnstrophyProduction(const float3x3& J);
float getStrainProduction(const float3x3& J);
float getSquareRateOfStrain(const float3x3& J);
float getPVA(const float3x3 &J);



/*************************************************************************************************************************************
** Scalar measure (of the velocity field, its gradient and derived tensors)
*************************************************************************************************************************************/

// runtime switched versions
inline float getMeasureFromRaw(eMeasure measure, const float4 vel4)
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

inline float getMeasureFromHeatCurrent(eMeasure measure, const float3 heatCurrent)
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

inline float getMeasureFromJac(eMeasure measure, const float3x3& jacobian)
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
    static inline float exec(eMeasure measure, texture<float4, cudaTextureType3D, cudaReadModeElementType> tex, float3 pos, float3 h);
	// no implementation - only specializations allowed
};

template <eMeasureSource source, eTextureFilterMode F>
struct getMeasureFromVolume_Impl2<source, F, MEASURE_COMPUTE_PRECOMP_DISCARD>
{
    static inline float exec(eMeasure measure, texture<float4, cudaTextureType3D, cudaReadModeElementType> tex, float3 pos, float3 h)
	{
		// precomputed measure volume: just read value
		return sampleVolume<F, float1, float>(g_texFeatureVolume, pos);
	}
};

template <eTextureFilterMode F>
struct getMeasureFromVolume_Impl2<MEASURE_SOURCE_RAW, F, MEASURE_COMPUTE_ONTHEFLY>
{
    static inline float exec(eMeasure measure, texture<float4, cudaTextureType3D, cudaReadModeElementType> tex, float3 pos, float3 h)
	{
		float4 vel4 = sampleVolume<F, float4, float4>(tex, pos);
		return getMeasureFromRaw(measure, vel4);
	}
};

template <eTextureFilterMode F>
struct getMeasureFromVolume_Impl2<MEASURE_SOURCE_HEAT_CURRENT, F, MEASURE_COMPUTE_ONTHEFLY>
{
	static inline float exec(eMeasure measure, texture<float4, cudaTextureType3D, cudaReadModeElementType> tex, float3 pos, float3 h)
	{
		float3 heatCurrent = getHeatCurrent<F>(tex, pos, h);
		return getMeasureFromHeatCurrent(measure, heatCurrent);
	}
};

template <eTextureFilterMode F>
struct getMeasureFromVolume_Impl2<MEASURE_SOURCE_JACOBIAN, F, MEASURE_COMPUTE_ONTHEFLY>
{
    static inline float exec(eMeasure measure, texture<float4, cudaTextureType3D, cudaReadModeElementType> tex, float3 pos, float3 h)
	{
		float3x3 jacobian = getJacobian<F>(tex, pos, h);
		return getMeasureFromJac(measure, jacobian);
	}
};


template <eMeasureSource source, eTextureFilterMode F, eMeasureComputeMode C>
inline float getMeasure(eMeasure measure, texture<float4, cudaTextureType3D, cudaReadModeElementType> tex, float3 pos, float3 h, float measureScale)
{
	return (measureScale * getMeasureFromVolume_Impl2<source, F, C>::exec(measure, tex, pos, h));
}


/******************************************************************************
** (Invariant) Measures:
******************************************************************************/

inline float3 getCosines(float arg)
{
	float acs = acosf(arg);
	return make_float3(cosf(acs/3.0f), cosf(acs/3.0f + 2.0f/3.0f * TUM3D_PI), cosf(acs/3.0f + 4.0f/3.0f * TUM3D_PI));
}

float getLambda2(const float3x3& J)
{
	float s01 = 0.5*( (J.m[0].x+J.m[1].y)*(J.m[0].y+J.m[1].x) + J.m[0].z*J.m[2].y + J.m[1].z*J.m[2].x);
	float s02 = 0.5*( (J.m[0].x+J.m[2].z)*(J.m[0].z+J.m[2].x) + J.m[0].y*J.m[1].z + J.m[2].y*J.m[1].x);
	float s12 = 0.5*( (J.m[1].y+J.m[2].z)*(J.m[1].z+J.m[2].y) + J.m[1].x*J.m[0].z + J.m[2].x*J.m[0].y);

	float s00 = J.m[0].x*J.m[0].x + J.m[0].y*J.m[1].x + J.m[0].z*J.m[2].x;
	float s11 = J.m[1].x*J.m[0].y + J.m[1].y*J.m[1].y + J.m[1].z*J.m[2].y;
	float s22 = J.m[2].x*J.m[0].z + J.m[2].y*J.m[1].z + J.m[2].z*J.m[2].z;

	float b= +s00 +s11 +s22;
	float c= -s00*(s11+s22) -s11*s22 + s12*s12 + s01*s01 +s02*s02;
	float d=  s00*(s11*s22-s12*s12)+2.0*s01*s12*s02 -s02*s02*s11 -s01*s01*s22;

	const float onethird=1.0f/3.0f;
	float xN		= b*onethird;
	float yN		= d + xN*(c + xN*(b - xN));
	float deltasqr	= xN*xN + c*onethird;
	float delta     = -getSign(yN)*sqrt(deltasqr);
	float hsqr		= 4.0f*deltasqr*deltasqr*deltasqr;
	float h         = -2.0f*delta*deltasqr;
	float yNsqr     = yN*yN;
	float lambda2;
	
	if (yNsqr>hsqr)
	{		
		float D = sqrt(yNsqr-hsqr);
		lambda2 = xN + getSign(yN-D) * powf(0.5f*fabsf(yN-D),onethird)
		             + getSign(yN+D) * powf(0.5f*fabsf(yN+D),onethird);
	}
	else if (yNsqr<hsqr) 
	{
		float3 L= xN + 2.0*delta*getCosines(-yN/h);
		lambda2 = fminf(fmaxf(fminf(L.x,L.y),L.z),fmaxf(L.x,L.y));
	}
	else 
	{
		if (h==0.0f) lambda2 = xN;
		else		 lambda2 = xN+delta;
	}

	return lambda2;
}


float getQHunt(const float3x3& jacobian)
{
	float3x3 S = getStrainRateTensor( jacobian );
	float3x3 O = getSpinTensor( jacobian );
	float fS   = FrobeniusNorm3x3( S );
	float fO   = FrobeniusNorm3x3( O );
	return (0.5 * ( fO*fO - fS*fS ));
}


float getDeltaChong(const float3x3& J)
{
	float3x3 S		= getStrainRateTensor( J );
	float3x3 O		= getSpinTensor( J );
	float3x3 SS		= multMat3x3(S,S);
	float3x3 OO		= multMat3x3(O,O);
	float3x3 SSpOO	= addMat3x3(SS,OO);

	float Q = -0.5f * Trace3x3(SSpOO);
	//float Q = getQHunt(J);
	float R = Det3x3(J);

	Q /= 3.0f;
	R /= 2.0f;

	return (Q*Q*Q + R*R);
}


float getSquareRotation(const float3x3& J)
{
	float3x3 O	 = getSpinTensor( J );
	float3x3 Osq = multMat3x3(O,O);

	return float(-0.5 * Trace3x3(Osq));
}


float getEnstrophyProduction(const float3x3& J)
{
	float3x3 S	= getStrainRateTensor( J );
	float3 w	= getVorticity( J );

	float e = S.m[0].x * w.x * w.x +
			  S.m[0].y * w.x * w.y +
			  S.m[0].z * w.x * w.z +
			  S.m[1].x * w.y * w.x +
			  S.m[1].y * w.y * w.y +
			  S.m[1].z * w.y * w.z +
			  S.m[2].x * w.z * w.x +
			  S.m[2].y * w.z * w.y +
			  S.m[2].z * w.z * w.z;

	return e;
}


float getStrainProduction(const float3x3& J)
{
	float3x3 S = getStrainRateTensor( J );

	float e =	S.m[0].x*S.m[0].x*S.m[0].x + S.m[0].x*S.m[0].y*S.m[1].x + S.m[0].x*S.m[0].z*S.m[2].x +
				S.m[0].y*S.m[1].x*S.m[0].x + S.m[0].y*S.m[1].y*S.m[1].x + S.m[0].y*S.m[1].z*S.m[2].x +
				S.m[0].z*S.m[2].x*S.m[0].x + S.m[0].z*S.m[2].y*S.m[1].x + S.m[0].z*S.m[2].z*S.m[2].x +
				S.m[1].x*S.m[0].x*S.m[0].y + S.m[1].x*S.m[0].y*S.m[1].y + S.m[1].x*S.m[0].z*S.m[2].y +
				S.m[1].y*S.m[1].x*S.m[0].y + S.m[1].y*S.m[1].y*S.m[1].y + S.m[1].y*S.m[1].z*S.m[2].y +
				S.m[1].z*S.m[2].x*S.m[0].y + S.m[1].z*S.m[2].y*S.m[1].y + S.m[1].z*S.m[2].z*S.m[2].y +
				S.m[2].x*S.m[0].x*S.m[0].z + S.m[2].x*S.m[0].y*S.m[1].z + S.m[2].x*S.m[0].z*S.m[2].z +
				S.m[2].y*S.m[1].x*S.m[0].z + S.m[2].y*S.m[1].y*S.m[1].z + S.m[2].y*S.m[1].z*S.m[2].z +
				S.m[2].z*S.m[2].x*S.m[0].z + S.m[2].z*S.m[2].y*S.m[1].z + S.m[2].z*S.m[2].z*S.m[2].z;

	return e;
}


// SquareRateOfStrain == Q_S in the paper except for a factor -0.5
float getSquareRateOfStrain(const float3x3& J)
{
	float3x3 S	 = getStrainRateTensor(J);
	float3x3 Ssq = multMat3x3(S,S);
	return Trace3x3(Ssq);
}


/***********************************************************************************************
* Eigensolver by Hasan et al. 
* additional sorting of the eigenvalues (no positive definite tensor)
***********************************************************************************************/

inline void sort3Items( float3 &v )
{
	float t;
	if (v.y < v.x)
	{
		t = v.x;
		if (v.z < v.y) { v.x = v.z; v.z = t; }
		else
		{
			if (v.z < t) { v.x = v.y; v.y = v.z; v.z = t; }
			else		 { v.x = v.y; v.y = t; }
		}
	}
	else
	{
		if (v.z < v.y)
		{
			t = v.z; 
			v.z = v.y;

			if (t < v.x) { v.y = v.x; v.x = t; }
			else		 { v.y = t; }
		}
	}
}



inline float3 getOrthonormalEigenvector( const float &eigenValue, const float3 &vDiag, const float3 &vOffDiag)
{
	float3 vABC  = make_float3(vDiag.x - eigenValue, vDiag.y - eigenValue, vDiag.z - eigenValue);

	return normalize(make_float3( (vOffDiag.x*vOffDiag.z-vABC.y*vOffDiag.y)*(vOffDiag.y*vOffDiag.z-vABC.z*vOffDiag.x), 
								 -(vOffDiag.y*vOffDiag.z-vABC.z*vOffDiag.x)*(vOffDiag.y*vOffDiag.x-vABC.x*vOffDiag.z), 
								  (vOffDiag.x*vOffDiag.z-vABC.y*vOffDiag.y)*(vOffDiag.y*vOffDiag.x-vABC.x*vOffDiag.z)));
}



inline void eigensolveHasan(const float3x3 &J, float3 &sortedEigenvalues, float3 &eigenVector1, float3 &eigenVector2, float3 &eigenVector3)
{
		const float3 vOne	= make_float3(1,1,1);
		float3 vDiag		= make_float3(J.m[0].x, J.m[1].y, J.m[2].z);  // xx , yy , zz
		float3 vOffDiag		= make_float3(J.m[0].y, J.m[0].z, J.m[1].z);  // xy , xz , yz
		float3 offSq		= vOffDiag*vOffDiag;
		float I1			= dot(vDiag, vOne);
		float I2			= dot(make_float3(vDiag.x, vDiag.x, vDiag.y), make_float3(vDiag.y, vDiag.z, vDiag.z)) - dot(offSq, vOne);
		float I3			= vDiag.x*vDiag.y*vDiag.z + 2.0f*vOffDiag.x*vOffDiag.y*vOffDiag.z - dot(make_float3(vDiag.z, vDiag.y, vDiag.x), offSq);
		float I1_3			= I1 / 3.0f;
		float I1_3Sq		= I1_3 * I1_3;
		float v				= I1_3Sq - I2 / 3.0f;
		float vInv			= 1.0f / v;
		float s				= I1_3Sq * I1_3 - I1 * I2 / 6.0f + I3 / 2.0f;
		float phi			= acosf(s * vInv * sqrt(vInv)) / 3.0f;
		float vSqrt2		= 2.0f * sqrt(v);

		sortedEigenvalues = make_float3(I1_3 + vSqrt2 * cosf(phi), I1_3 - vSqrt2 * cosf((TUM3D_PI / 3.0f) + phi), I1_3 - vSqrt2 * cosf((TUM3D_PI / 3.0f) - phi));
		sort3Items( sortedEigenvalues );

		eigenVector1 = getOrthonormalEigenvector(sortedEigenvalues.x, vDiag, vOffDiag);
		eigenVector2 = getOrthonormalEigenvector(sortedEigenvalues.y, vDiag, vOffDiag);
		eigenVector3 = cross(eigenVector1, eigenVector2);
}


/***********************************************************************************************
* Preferential vorticity alignment (cosine of the angle between the second largest eigenvector
* of the strain rate tensor and the vorticity)
***********************************************************************************************/

float getPVA(const float3x3 &J)
{
		const float3 vOne	= make_float3(1,1,1);
		float3 vDiag		= make_float3(J.m[0].x, J.m[1].y, J.m[2].z);  // xx , yy , zz
		float3 vOffDiag		= make_float3(J.m[0].y, J.m[0].z, J.m[1].z);  // xy , xz , yz
		float3 offSq		= vOffDiag*vOffDiag;
		float I1			= dot(vDiag, vOne);
		float I2			= dot(make_float3(vDiag.x, vDiag.x, vDiag.y), make_float3(vDiag.y, vDiag.z, vDiag.z)) - dot(offSq, vOne);
		float I3			= vDiag.x*vDiag.y*vDiag.z + 2.0f*vOffDiag.x*vOffDiag.y*vOffDiag.z - dot(make_float3(vDiag.z, vDiag.y, vDiag.x), offSq);
		float I1_3			= I1 / 3.0f;
		float I1_3Sq		= I1_3 * I1_3;
		float v				= I1_3Sq - I2 / 3.0f;
		float vInv			= 1.0f / v;
		float s				= I1_3Sq * I1_3 - I1 * I2 / 6.0f + I3 / 2.0f;
		float phi			= acosf(s * vInv * sqrt(vInv)) / 3.0f;
		float vSqrt2		= 2.0f * sqrt(v);

		float3 sortedEigenvalues = make_float3(I1_3 + vSqrt2 * cosf(phi), I1_3 - vSqrt2 * cosf((TUM3D_PI / 3.0f) + phi), I1_3 - vSqrt2 * cosf((TUM3D_PI / 3.0f) - phi));
		sort3Items( sortedEigenvalues ); 

		float3 e2		 = getOrthonormalEigenvector(sortedEigenvalues.y, vDiag, vOffDiag);
		float3 vorticity = getVorticity( J );

		return abs(dot(e2,vorticity));
}
