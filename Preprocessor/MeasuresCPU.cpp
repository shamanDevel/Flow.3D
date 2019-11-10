#include <vector>
#include <cmath>
#include "MeasuresCPU.h"

/******************************************************************************
** Linear algebra implementations
******************************************************************************/

vec3 make_vec3(float x, float y, float z) {
	return vec3{x, y, z};
}

vec3 make_vec3(const vec4& v4) {
	return vec3{v4.x, v4.y, v4.z};
}

vec4 make_vec4(float x, float y, float z, float w) {
	return vec4{x, y, z, w};
}

vec4 make_vec4(const vec3& v3) {
	return vec4{v3.x, v3.y, v3.z, 1.0f};
}

float dot(const vec3& v1, const vec3& v2) {
	return v1.x*v2.x + v1.y*v2.y + v1.z*v2.z;
}

float length(const vec3& v) {
	return std::sqrt(v.x*v.x + v.y*v.y + v.z*v.z);
}

vec3 normalize(const vec3& v) {
	return v / length(v);
}

vec3 cross(vec3 a, vec3 b) { 
    return make_vec3(a.y*b.z - a.z*b.y, a.z*b.x - a.x*b.z, a.x*b.y - a.y*b.x); 
}



template<typename T>
T sampleVolume(const VolumeTextureCPU& tex, int x, int y, int z);


template<>
vec4 sampleVolume<vec4>(const VolumeTextureCPU& tex, int x, int y, int z) {
    return tex.samplevec4(x, y, z);
}

template<>
vec3 sampleVolume<vec3>(const VolumeTextureCPU& tex, int x, int y, int z) {
    return tex.samplevec3(x, y, z);
}


template<typename T>
inline T sampleVolumeDerivativeX(const VolumeTextureCPU& tex, int x, int y, int z, const vec3& h)
{
	// default implementation: central differences
	T dp = sampleVolume<T>(tex, x + 1, y, z);
	T dn = sampleVolume<T>(tex, x - 1, y, z);
	return (dp - dn) / (2.0f * h.x);
}

template<typename T>
inline T sampleVolumeDerivativeY(const VolumeTextureCPU& tex, int x, int y, int z, const vec3& h)
{
	// default implementation: central differences
	T dp = sampleVolume<T>(tex, x, y + 1, z);
	T dn = sampleVolume<T>(tex, x, y - 1, z);
	return (dp - dn) / (2.0f * h.y);
}

template<typename T>
inline T sampleVolumeDerivativeZ(const VolumeTextureCPU& tex, int x, int y, int z, const vec3& h)
{
	// default implementation: central differences
	T dp = sampleVolume<T>(tex, x, y, z + 1);
	T dn = sampleVolume<T>(tex, x, y, z - 1);
	return (dp - dn) / (2.0f * h.z);
}

mat3x3 getJacobian(const VolumeTextureCPU& tex, int x, int y, int z, const vec3& h) {
	// default implementation: just get derivatives in x, y, z
	// note: result is "transposed" by convention (i.e. first index is component, second index is derivative direction) - fix this some time?
	mat3x3 J;

	// derivative in x direction
	vec3 dx = sampleVolumeDerivativeX<vec3>(tex, x, y, z, h);
	J.m[0].x = dx.x;
	J.m[1].x = dx.y;
	J.m[2].x = dx.z;

	// derivative in y direction
	vec3 dy = sampleVolumeDerivativeY<vec3>(tex, x, y, z, h);
	J.m[0].y = dy.x;
	J.m[1].y = dy.y;
	J.m[2].y = dy.z;

	// derivative in z direction
	vec3 dz = sampleVolumeDerivativeZ<vec3>(tex, x, y, z, h);
	J.m[0].z = dz.x;
	J.m[1].z = dz.y;
	J.m[2].z = dz.z;

	return J;
}

inline vec3 sampleScalarGradient(const VolumeTextureCPU& tex, int x, int y, int z, const vec3& h)
{
	// default implementation: just get derivatives in x, y, z
	vec3 grad;

	// derivative in x direction
	vec4 dx = sampleVolumeDerivativeX<vec4>(tex, x, y, z, h);
	grad.x = dx.w;

	// derivative in y direction
	vec4 dy = sampleVolumeDerivativeY<vec4>(tex, x, y, z, h);
	grad.y = dy.w;

	// derivative in z direction
	vec4 dz = sampleVolumeDerivativeZ<vec4>(tex, x, y, z, h);
	grad.z = dz.w;

	return grad;
}


///////////////////////////////////////////////////////////////////////////////
// Matrix Math
///////////////////////////////////////////////////////////////////////////////

inline mat3x3 multMat3x3(const mat3x3 &A, const mat3x3 &B)
{
	mat3x3 erg;

	erg.m[0] = make_vec3(A.m[0].x*B.m[0].x + A.m[0].y*B.m[1].x + A.m[0].z*B.m[2].x, 
						   A.m[0].x*B.m[0].y + A.m[0].y*B.m[1].y + A.m[0].z*B.m[2].y,
						   A.m[0].x*B.m[0].z + A.m[0].y*B.m[1].z + A.m[0].z*B.m[2].z);

	erg.m[1] = make_vec3(A.m[1].x*B.m[0].x + A.m[1].y*B.m[1].x + A.m[1].z*B.m[2].x, 
						   A.m[1].x*B.m[0].y + A.m[1].y*B.m[1].y + A.m[1].z*B.m[2].y,
						   A.m[1].x*B.m[0].z + A.m[1].y*B.m[1].z + A.m[1].z*B.m[2].z);

	erg.m[2] = make_vec3(A.m[2].x*B.m[0].x + A.m[2].y*B.m[1].x + A.m[2].z*B.m[2].x, 
						   A.m[2].x*B.m[0].y + A.m[2].y*B.m[1].y + A.m[2].z*B.m[2].y,
						   A.m[2].x*B.m[0].z + A.m[2].y*B.m[1].z + A.m[2].z*B.m[2].z);

	return erg;
}



inline mat3x3 addMat3x3(const mat3x3 &A, const mat3x3 &B)
{
	mat3x3 erg;

	erg.m[0] = make_vec3(A.m[0].x+B.m[0].x, A.m[0].y+B.m[0].y, A.m[0].z+B.m[0].z);
	erg.m[1] = make_vec3(A.m[1].x+B.m[1].x, A.m[1].y+B.m[1].y, A.m[1].z+B.m[1].z);
	erg.m[2] = make_vec3(A.m[2].x+B.m[2].x, A.m[2].y+B.m[2].y, A.m[2].z+B.m[2].z);

	return erg;
}



inline float Det3x3(const mat3x3 &A)
{
	return float( A.m[0].x*A.m[1].y*A.m[2].z + 
				  A.m[0].y*A.m[1].z*A.m[2].x + 
				  A.m[0].z*A.m[1].x*A.m[2].y - 
				  A.m[0].x*A.m[2].y*A.m[1].z - 
				  A.m[1].x*A.m[0].y*A.m[2].z - 
				  A.m[2].x*A.m[1].y*A.m[0].z );
}



inline float Trace3x3(const mat3x3 &A)
{
	return float(A.m[0].x + A.m[1].y + A.m[2].z);
}



inline float TraceAAT(const mat3x3 &A)
{
	return float(A.m[0].x*A.m[0].x + A.m[0].y*A.m[0].y + A.m[0].z*A.m[0].z + 
				 A.m[1].x*A.m[1].x + A.m[1].y*A.m[1].y + A.m[1].z*A.m[1].z + 
				 A.m[2].x*A.m[2].x + A.m[2].y*A.m[2].y + A.m[2].z*A.m[2].z);
}



inline float FrobeniusNorm3x3(const mat3x3 &A)
{
	return sqrtf( TraceAAT(A) );
}



inline void TransposeInplace3x3(mat3x3 &A)
{
	float tmp;

	tmp = A.m[0].y; A.m[0].y = A.m[1].x; A.m[1].x = tmp;
	tmp = A.m[0].z; A.m[0].z = A.m[2].x; A.m[2].x = tmp;
	tmp = A.m[1].z; A.m[1].z = A.m[2].y; A.m[2].y = tmp;
}



inline mat3x3 Transpose3x3(const mat3x3 &A)
{
	mat3x3 AT;

	AT.m[0] = make_vec3(A.m[0].x, A.m[1].x, A.m[2].x);
	AT.m[1] = make_vec3(A.m[0].y, A.m[1].y, A.m[2].y);
	AT.m[2] = make_vec3(A.m[0].z, A.m[1].z, A.m[2].z);

	return AT;
}



/******************************************************************************
** Heat current and derived quantities
******************************************************************************/


// get heat current from velocity/temperature texture
inline vec3 getHeatCurrent(const VolumeTextureCPU& tex, int x, int y, int z, const vec3& h)
{
	vec4 velT = sampleVolume<vec4>(tex, x, y, z);
	vec3 gradT = sampleScalarGradient(tex, x, y, z, h);

	vec3 j;

	float ra = 7e5f;
	float pr = 0.7f;
	float kappa = 1 / sqrt(ra*pr);

	j.x = velT.x * velT.w - kappa * gradT.x;
	j.y = velT.y * velT.w - kappa * gradT.y;
	j.z = velT.z * velT.w - kappa * gradT.z;

	return j;
}


// TODO: Originally "const float& h". Why?
inline float getHeatCurrentAlignment(const VolumeTextureCPU& tex, int x, int y, int z, const vec3& h)
{
	vec4 velT = sampleVolume<vec4>(tex, x, y, z);
	vec3 gradT = sampleScalarGradient(tex, x, y, z, h);

	vec3 j;

	float ra = 7e5f;
	float pr = 0.7f;
	float kappa = 1 / sqrt(ra*pr);

	j.x = velT.x * velT.w - kappa * gradT.x;
	j.y = velT.y * velT.w - kappa * gradT.y;
	j.z = velT.z * velT.w - kappa * gradT.z;

	j = normalize(j);
	vec3 vel = normalize(make_vec3(velT));

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
** Jacobian and derived tensors
******************************************************************************/


inline mat3x3 getStrainRateTensor(const mat3x3 &J)
{
	// S_ij = 1/2 (J_ij + J_ji)
	mat3x3 S;

	S.m[0] = make_vec3( (J.m[0].x+J.m[0].x)/2.0f, (J.m[0].y+J.m[1].x)/2.0f, (J.m[0].z+J.m[2].x)/2.0f );
	S.m[1] = make_vec3( (J.m[1].x+J.m[0].y)/2.0f, (J.m[1].y+J.m[1].y)/2.0f, (J.m[1].z+J.m[2].y)/2.0f );
	S.m[2] = make_vec3( (J.m[2].x+J.m[0].z)/2.0f, (J.m[2].y+J.m[1].z)/2.0f, (J.m[2].z+J.m[2].z)/2.0f );

	return S;
}



inline mat3x3 getSpinTensor(const mat3x3 &J)
{
	// O_ij = 1/2 (J_ij - J_ji)
	mat3x3 O;

	O.m[0] = make_vec3( (J.m[0].x-J.m[0].x)/2.0f, (J.m[0].y-J.m[1].x)/2.0f, (J.m[0].z-J.m[2].x)/2.0f );
	O.m[1] = make_vec3( (J.m[1].x-J.m[0].y)/2.0f, (J.m[1].y-J.m[1].y)/2.0f, (J.m[1].z-J.m[2].y)/2.0f );
	O.m[2] = make_vec3( (J.m[2].x-J.m[0].z)/2.0f, (J.m[2].y-J.m[1].z)/2.0f, (J.m[2].z-J.m[2].z)/2.0f );

	return O;
}


/******************************************************************************
** (Invariant) Measures:
******************************************************************************/

inline vec3 getVorticity(const mat3x3& J)
{
	return make_vec3(J.m[2].y - J.m[1].z, J.m[0].z - J.m[2].x, J.m[1].x - J.m[0].y);
}

inline vec3 getVorticity(const VolumeTextureCPU& tex, int x, int y, int z, const vec3& h)
{
	mat3x3 jacobian = getJacobian(tex, x, y, z, h);
	return getVorticity(jacobian);
}

float getLambda2(const mat3x3& J);
float getQHunt(const mat3x3& J);
float getDeltaChong(const mat3x3& J);
float getSquareRotation(const mat3x3& J);
float getEnstrophyProduction(const mat3x3& J);
float getStrainProduction(const mat3x3& J);
float getSquareRateOfStrain(const mat3x3& J);
float getPVA(const mat3x3 &J);



/*************************************************************************************************************************************
** Scalar measure (of the velocity field, its gradient and derived tensors)
*************************************************************************************************************************************/

// runtime switched versions
inline float getMeasureFromRaw(eMeasure measure, const vec4 vel4)
{
	switch(measure)
	{
		case MEASURE_VELOCITY:
			return length(make_vec3(vel4));
		case MEASURE_VELOCITY_Z:
			return vel4.z;
		case MEASURE_TEMPERATURE:
			return vel4.w;
		default:
			//assert(false) ?
			return 0.0f;
	}
}

inline float getMeasureFromHeatCurrent(eMeasure measure, const vec3 heatCurrent)
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

inline float getMeasureFromJac(eMeasure measure, const mat3x3& jacobian)
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

float getMeasureFromVolume(
		const VolumeTextureCPU& tex, int x, int y, int z, const vec3& h,
		eMeasureSource measureSource, eMeasure measure) {
	if (measureSource == MEASURE_SOURCE_RAW) {
		vec4 vel4 = sampleVolume<vec4>(tex, x, y, z);
		return getMeasureFromRaw(measure, vel4);
	} else if (measureSource == MEASURE_SOURCE_HEAT_CURRENT) {
		vec3 heatCurrent = getHeatCurrent(tex, x, y, z, h);
		return getMeasureFromHeatCurrent(measure, heatCurrent);
	} else if (measureSource == MEASURE_SOURCE_JACOBIAN) {
		mat3x3 jacobian = getJacobian(tex, x, y, z, h);
		return getMeasureFromJac(measure, jacobian);
	} else {
		return 0.0f;
	}
}


/******************************************************************************
** (Invariant) Measures:
******************************************************************************/

inline vec3 getCosines(float arg)
{
	float acs = acosf(arg);
	return vec3{cosf(acs/3.0f), cosf(acs/3.0f + 2.0f/3.0f * TUM3D_PI), cosf(acs/3.0f + 4.0f/3.0f * TUM3D_PI)};
}

float getLambda2(const mat3x3& J)
{
	float s01 = 0.5f*( (J.m[0].x+J.m[1].y)*(J.m[0].y+J.m[1].x) + J.m[0].z*J.m[2].y + J.m[1].z*J.m[2].x);
	float s02 = 0.5f*( (J.m[0].x+J.m[2].z)*(J.m[0].z+J.m[2].x) + J.m[0].y*J.m[1].z + J.m[2].y*J.m[1].x);
	float s12 = 0.5f*( (J.m[1].y+J.m[2].z)*(J.m[1].z+J.m[2].y) + J.m[1].x*J.m[0].z + J.m[2].x*J.m[0].y);

	float s00 = J.m[0].x*J.m[0].x + J.m[0].y*J.m[1].x + J.m[0].z*J.m[2].x;
	float s11 = J.m[1].x*J.m[0].y + J.m[1].y*J.m[1].y + J.m[1].z*J.m[2].y;
	float s22 = J.m[2].x*J.m[0].z + J.m[2].y*J.m[1].z + J.m[2].z*J.m[2].z;

	float b= +s00 +s11 +s22;
	float c= -s00*(s11+s22) -s11*s22 + s12*s12 + s01*s01 +s02*s02;
	float d=  s00*(s11*s22-s12*s12)+2.0f*s01*s12*s02 -s02*s02*s11 -s01*s01*s22;

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
		//vec3 L = xN + 2.0f*delta*getCosines(-yN/h);
		vec3 L = 2.0f*delta*getCosines(-yN/h);
		L.x += xN;
		L.y += xN;
		L.z += xN;
		lambda2 = fminf(fmaxf(fminf(L.x,L.y),L.z),fmaxf(L.x,L.y));
	}
	else 
	{
		if (h==0.0f) lambda2 = xN;
		else		 lambda2 = xN+delta;
	}

	return lambda2;
}


float getQHunt(const mat3x3& jacobian)
{
	mat3x3 S = getStrainRateTensor( jacobian );
	mat3x3 O = getSpinTensor( jacobian );
	float fS   = FrobeniusNorm3x3( S );
	float fO   = FrobeniusNorm3x3( O );
	return (0.5f * ( fO*fO - fS*fS ));
}


float getDeltaChong(const mat3x3& J)
{
	mat3x3 S		= getStrainRateTensor( J );
	mat3x3 O		= getSpinTensor( J );
	mat3x3 SS		= multMat3x3(S,S);
	mat3x3 OO		= multMat3x3(O,O);
	mat3x3 SSpOO	= addMat3x3(SS,OO);

	float Q = -0.5f * Trace3x3(SSpOO);
	//float Q = getQHunt(J);
	float R = Det3x3(J);

	Q /= 3.0f;
	R /= 2.0f;

	return (Q*Q*Q + R*R);
}


float getSquareRotation(const mat3x3& J)
{
	mat3x3 O	 = getSpinTensor( J );
	mat3x3 Osq = multMat3x3(O,O);

	return float(-0.5 * Trace3x3(Osq));
}


float getEnstrophyProduction(const mat3x3& J)
{
	mat3x3 S	= getStrainRateTensor( J );
	vec3 w	= getVorticity( J );

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


float getStrainProduction(const mat3x3& J)
{
	mat3x3 S = getStrainRateTensor( J );

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
float getSquareRateOfStrain(const mat3x3& J)
{
	mat3x3 S	 = getStrainRateTensor(J);
	mat3x3 Ssq = multMat3x3(S,S);
	return Trace3x3(Ssq);
}


/***********************************************************************************************
* Eigensolver by Hasan et al. 
* additional sorting of the eigenvalues (no positive definite tensor)
***********************************************************************************************/

inline void sort3Items( vec3 &v )
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



inline vec3 getOrthonormalEigenvector( const float &eigenValue, const vec3 &vDiag, const vec3 &vOffDiag)
{
	vec3 vABC  = make_vec3(vDiag.x - eigenValue, vDiag.y - eigenValue, vDiag.z - eigenValue);

	return normalize(make_vec3( (vOffDiag.x*vOffDiag.z-vABC.y*vOffDiag.y)*(vOffDiag.y*vOffDiag.z-vABC.z*vOffDiag.x), 
								 -(vOffDiag.y*vOffDiag.z-vABC.z*vOffDiag.x)*(vOffDiag.y*vOffDiag.x-vABC.x*vOffDiag.z), 
								  (vOffDiag.x*vOffDiag.z-vABC.y*vOffDiag.y)*(vOffDiag.y*vOffDiag.x-vABC.x*vOffDiag.z)));
}



inline void eigensolveHasan(const mat3x3 &J, vec3 &sortedEigenvalues, vec3 &eigenVector1, vec3 &eigenVector2, vec3 &eigenVector3)
{
		const vec3 vOne	= make_vec3(1,1,1);
		vec3 vDiag		= make_vec3(J.m[0].x, J.m[1].y, J.m[2].z);  // xx , yy , zz
		vec3 vOffDiag		= make_vec3(J.m[0].y, J.m[0].z, J.m[1].z);  // xy , xz , yz
		vec3 offSq		= vOffDiag*vOffDiag;
		float I1			= dot(vDiag, vOne);
		float I2			= dot(make_vec3(vDiag.x, vDiag.x, vDiag.y), make_vec3(vDiag.y, vDiag.z, vDiag.z)) - dot(offSq, vOne);
		float I3			= vDiag.x*vDiag.y*vDiag.z + 2.0f*vOffDiag.x*vOffDiag.y*vOffDiag.z - dot(make_vec3(vDiag.z, vDiag.y, vDiag.x), offSq);
		float I1_3			= I1 / 3.0f;
		float I1_3Sq		= I1_3 * I1_3;
		float v				= I1_3Sq - I2 / 3.0f;
		float vInv			= 1.0f / v;
		float s				= I1_3Sq * I1_3 - I1 * I2 / 6.0f + I3 / 2.0f;
		float phi			= acosf(s * vInv * sqrt(vInv)) / 3.0f;
		float vSqrt2		= 2.0f * sqrt(v);

		sortedEigenvalues = make_vec3(I1_3 + vSqrt2 * cosf(phi), I1_3 - vSqrt2 * cosf((TUM3D_PI / 3.0f) + phi), I1_3 - vSqrt2 * cosf((TUM3D_PI / 3.0f) - phi));
		sort3Items( sortedEigenvalues );

		eigenVector1 = getOrthonormalEigenvector(sortedEigenvalues.x, vDiag, vOffDiag);
		eigenVector2 = getOrthonormalEigenvector(sortedEigenvalues.y, vDiag, vOffDiag);
		eigenVector3 = cross(eigenVector1, eigenVector2);
}


/***********************************************************************************************
* Preferential vorticity alignment (cosine of the angle between the second largest eigenvector
* of the strain rate tensor and the vorticity)
***********************************************************************************************/

float getPVA(const mat3x3 &J)
{
		const vec3 vOne	= make_vec3(1,1,1);
		vec3 vDiag		= make_vec3(J.m[0].x, J.m[1].y, J.m[2].z);  // xx , yy , zz
		vec3 vOffDiag		= make_vec3(J.m[0].y, J.m[0].z, J.m[1].z);  // xy , xz , yz
		vec3 offSq		= vOffDiag*vOffDiag;
		float I1			= dot(vDiag, vOne);
		float I2			= dot(make_vec3(vDiag.x, vDiag.x, vDiag.y), make_vec3(vDiag.y, vDiag.z, vDiag.z)) - dot(offSq, vOne);
		float I3			= vDiag.x*vDiag.y*vDiag.z + 2.0f*vOffDiag.x*vOffDiag.y*vOffDiag.z - dot(make_vec3(vDiag.z, vDiag.y, vDiag.x), offSq);
		float I1_3			= I1 / 3.0f;
		float I1_3Sq		= I1_3 * I1_3;
		float v				= I1_3Sq - I2 / 3.0f;
		float vInv			= 1.0f / v;
		float s				= I1_3Sq * I1_3 - I1 * I2 / 6.0f + I3 / 2.0f;
		float phi			= acosf(s * vInv * sqrt(vInv)) / 3.0f;
		float vSqrt2		= 2.0f * sqrt(v);

		vec3 sortedEigenvalues = make_vec3(I1_3 + vSqrt2 * cosf(phi), I1_3 - vSqrt2 * cosf((TUM3D_PI / 3.0f) + phi), I1_3 - vSqrt2 * cosf((TUM3D_PI / 3.0f) - phi));
		sort3Items( sortedEigenvalues ); 

		vec3 e2		 = getOrthonormalEigenvector(sortedEigenvalues.y, vDiag, vOffDiag);
		vec3 vorticity = getVorticity( J );

		return abs(dot(e2,vorticity));
}
