#include "Measures.cuh"


/******************************************************************************
** (Invariant) Measures:
******************************************************************************/

__device__ inline float3 getCosines(float arg)
{
	float acs = acosf(arg);
	return make_float3(cosf(acs/3.0f), cosf(acs/3.0f + 2.0f/3.0f * TUM3D_PI), cosf(acs/3.0f + 4.0f/3.0f * TUM3D_PI));
}

__device__ float getLambda2(const float3x3& J)
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


__device__ float getQHunt(const float3x3& jacobian)
{
	float3x3 S = getStrainRateTensor( jacobian );
	float3x3 O = getSpinTensor( jacobian );
	float fS   = FrobeniusNorm3x3( S );
	float fO   = FrobeniusNorm3x3( O );
	return (0.5 * ( fO*fO - fS*fS ));
}


__device__ float getDeltaChong(const float3x3& J)
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


__device__ float getSquareRotation(const float3x3& J)
{
	float3x3 O	 = getSpinTensor( J );
	float3x3 Osq = multMat3x3(O,O);

	return float(-0.5 * Trace3x3(Osq));
}


__device__ float getEnstrophyProduction(const float3x3& J)
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


__device__ float getStrainProduction(const float3x3& J)
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
__device__ float getSquareRateOfStrain(const float3x3& J)
{
	float3x3 S	 = getStrainRateTensor(J);
	float3x3 Ssq = multMat3x3(S,S);
	return Trace3x3(Ssq);
}


/***********************************************************************************************
* Eigensolver by Hasan et al. 
* additional sorting of the eigenvalues (no positive definite tensor)
***********************************************************************************************/

__device__ inline void sort3Items( float3 &v )
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



__device__ inline float3 getOrthonormalEigenvector( const float &eigenValue, const float3 &vDiag, const float3 &vOffDiag)
{
	float3 vABC  = make_float3(vDiag.x - eigenValue, vDiag.y - eigenValue, vDiag.z - eigenValue);

	return normalize(make_float3( (vOffDiag.x*vOffDiag.z-vABC.y*vOffDiag.y)*(vOffDiag.y*vOffDiag.z-vABC.z*vOffDiag.x), 
								 -(vOffDiag.y*vOffDiag.z-vABC.z*vOffDiag.x)*(vOffDiag.y*vOffDiag.x-vABC.x*vOffDiag.z), 
								  (vOffDiag.x*vOffDiag.z-vABC.y*vOffDiag.y)*(vOffDiag.y*vOffDiag.x-vABC.x*vOffDiag.z)));
}



__device__ inline void eigensolveHasan(const float3x3 &J, float3 &sortedEigenvalues, float3 &eigenVector1, float3 &eigenVector2, float3 &eigenVector3)
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

__device__ float getPVA(const float3x3 &J)
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
