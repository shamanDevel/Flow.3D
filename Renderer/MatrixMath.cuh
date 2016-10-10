#ifndef __TUM3D__MATRIX_MATH_CUH__
#define __TUM3D__MATRIX_MATH_CUH__


#include "cutil_math.h"


///////////////////////////////////////////////////////////////////////////////
// Matrix Math
///////////////////////////////////////////////////////////////////////////////

__device__ inline float3 transformPos(const float3x4 &M, const float3 &v)
{
    float3 r;
    r.x = dot(make_float4(v, 1.0f), M.m[0]);
    r.y = dot(make_float4(v, 1.0f), M.m[1]);
    r.z = dot(make_float4(v, 1.0f), M.m[2]);
    return r;
}

__device__ inline float3 transformDir(const float3x4 &M, const float3 &v)
{
    float3 r;
    r.x = dot(v, make_float3(M.m[0]));
    r.y = dot(v, make_float3(M.m[1]));
    r.z = dot(v, make_float3(M.m[2]));
    return r;
}


__device__ inline float3x3 multMat3x3(const float3x3 &A, const float3x3 &B)
{
	float3x3 erg;

	erg.m[0] = make_float3(A.m[0].x*B.m[0].x + A.m[0].y*B.m[1].x + A.m[0].z*B.m[2].x, 
						   A.m[0].x*B.m[0].y + A.m[0].y*B.m[1].y + A.m[0].z*B.m[2].y,
						   A.m[0].x*B.m[0].z + A.m[0].y*B.m[1].z + A.m[0].z*B.m[2].z);

	erg.m[1] = make_float3(A.m[1].x*B.m[0].x + A.m[1].y*B.m[1].x + A.m[1].z*B.m[2].x, 
						   A.m[1].x*B.m[0].y + A.m[1].y*B.m[1].y + A.m[1].z*B.m[2].y,
						   A.m[1].x*B.m[0].z + A.m[1].y*B.m[1].z + A.m[1].z*B.m[2].z);

	erg.m[2] = make_float3(A.m[2].x*B.m[0].x + A.m[2].y*B.m[1].x + A.m[2].z*B.m[2].x, 
						   A.m[2].x*B.m[0].y + A.m[2].y*B.m[1].y + A.m[2].z*B.m[2].y,
						   A.m[2].x*B.m[0].z + A.m[2].y*B.m[1].z + A.m[2].z*B.m[2].z);

	return erg;
}



__device__ inline float3x3 addMat3x3(const float3x3 &A, const float3x3 &B)
{
	float3x3 erg;

	erg.m[0] = make_float3(A.m[0].x+B.m[0].x, A.m[0].y+B.m[0].y, A.m[0].z+B.m[0].z);
	erg.m[1] = make_float3(A.m[1].x+B.m[1].x, A.m[1].y+B.m[1].y, A.m[1].z+B.m[1].z);
	erg.m[2] = make_float3(A.m[2].x+B.m[2].x, A.m[2].y+B.m[2].y, A.m[2].z+B.m[2].z);

	return erg;
}



__device__ inline float Det3x3(const float3x3 &A)
{
	return float( A.m[0].x*A.m[1].y*A.m[2].z + 
				  A.m[0].y*A.m[1].z*A.m[2].x + 
				  A.m[0].z*A.m[1].x*A.m[2].y - 
				  A.m[0].x*A.m[2].y*A.m[1].z - 
				  A.m[1].x*A.m[0].y*A.m[2].z - 
				  A.m[2].x*A.m[1].y*A.m[0].z );
}



__device__ inline float Trace3x3(const float3x3 &A)
{
	return float(A.m[0].x + A.m[1].y + A.m[2].z);
}



__device__ inline float TraceAAT(const float3x3 &A)
{
	return float(A.m[0].x*A.m[0].x + A.m[0].y*A.m[0].y + A.m[0].z*A.m[0].z + 
				 A.m[1].x*A.m[1].x + A.m[1].y*A.m[1].y + A.m[1].z*A.m[1].z + 
				 A.m[2].x*A.m[2].x + A.m[2].y*A.m[2].y + A.m[2].z*A.m[2].z);
}



__device__ inline float FrobeniusNorm3x3(const float3x3 &A)
{
	return sqrtf( TraceAAT(A) );
}



__device__ inline void TransposeInplace3x3(float3x3 &A)
{
	float tmp;

	tmp = A.m[0].y; A.m[0].y = A.m[1].x; A.m[1].x = tmp;
	tmp = A.m[0].z; A.m[0].z = A.m[2].x; A.m[2].x = tmp;
	tmp = A.m[1].z; A.m[1].z = A.m[2].y; A.m[2].y = tmp;
}



__device__ inline float3x3 Transpose3x3(const float3x3 &A)
{
	float3x3 AT;

	AT.m[0] = make_float3(A.m[0].x, A.m[1].x, A.m[2].x);
	AT.m[1] = make_float3(A.m[0].y, A.m[1].y, A.m[2].y);
	AT.m[2] = make_float3(A.m[0].z, A.m[1].z, A.m[2].z);

	return AT;
}


#endif
