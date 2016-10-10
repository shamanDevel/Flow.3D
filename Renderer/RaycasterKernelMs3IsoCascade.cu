//#define RAYCASTER_ENABLE_MS3_ISO_CASCADE

#include "RaycasterKernelParams.h"
#ifdef RAYCASTER_ENABLE_MS3_ISO_CASCADE
#include "cudaUtil.h"
#include "RaycasterKernelDefines.h"


#include "RaycasterKernelGlobals.cui"
#include "RaycasterKernelHelpers.cui"

#include "RaycasterKernelMs3IsoCascadeStep.cui"



template < eMeasure M, eTextureFilterMode F, eMeasureComputeMode C, eColorMode CM > 
__global__ void ms3IsoCascadeKernel(
	int2 brickMinScreen,
	int2 brickSizeScreen,
	int2 renderTargetOffset,
	float3 boxMin,
	float3 boxMax,
	float3 world2texOffset,
	float world2texScale
)
{
	const float opacityThreshold = 0.999f;

	uint x = blockIdx.x*blockDim.x + threadIdx.x;
	uint y = blockIdx.y*blockDim.y + threadIdx.y;

	if ((x >= brickSizeScreen.x) || (y >= brickSizeScreen.y)) return;
	x += brickMinScreen.x;
	y += brickMinScreen.y;

	// calculate eye ray in world space
	float3 rayPos = getRayPos(c_raycastParams.viewInv);
	float3 rayDir = getRayDir(c_raycastParams.viewInv, x, y);

	x += renderTargetOffset.x;
	y += renderTargetOffset.y;

	// find intersection with box
	float tnear, tfar;
	if (!intersectBox(rayPos, rayDir, boxMin, boxMax, &tnear, &tfar)) return;
	tnear = fmaxf(tnear, 0.0f); // clamp to near plane

	// get initial color from render target
	uchar4 colorStart;
	surf2Dread(&colorStart, g_surfTarget, x * 4, y);

	float4 sum = rgbaUCharToFloat(colorStart);

	// get rendering params
	float measureScale = c_raycastParams.measureScale;
	float3 isoValues = c_raycastParams.isoValues;
	float4 isoColor1 = c_raycastParams.isoColor1;
	float4 isoColor2 = c_raycastParams.isoColor2;
	float4 isoColor3 = c_raycastParams.isoColor3;

	// march along ray from front to back, accumulating color
	float3 pos = rayPos + rayDir * tnear;
	float3 step = rayDir * c_raycastParams.stepSizeWorld;

	// get value at entry point
	bool bWasInsideCoarseTube = (isoValues.x < getMeasure<M,F,C>(g_texVolume3, w2t(pos), measureScale));
	bool bWasInsideMidTube	  = (isoValues.y < getMeasure<M,F,C>(g_texVolume2, w2t(pos), measureScale));
	pos += step;

	// ray march through volume
	int numSteps = int(ceilf((tfar - tnear) / c_raycastParams.stepSizeWorld)) - 1;
	while(numSteps-- > 0 && sum.w < opacityThreshold) 
	{
		ms3IsoCascadeStep<M,F,C,CM>(sum, bWasInsideCoarseTube, bWasInsideMidTube, world2texOffset, world2texScale, pos, step, rayDir, measureScale, isoValues, isoColor1, isoColor2, isoColor3);
		pos += step;
	}

	// last step at exit point
	pos = rayPos + rayDir * tfar;
	ms3IsoCascadeStep<M,F,C,CM>(sum, bWasInsideCoarseTube, bWasInsideMidTube, world2texOffset, world2texScale, pos, step, rayDir, measureScale, isoValues, isoColor1, isoColor2, isoColor3);

	// write output color
	surf2Dwrite(rgbaFloatToUChar(sum), g_surfTarget, x * 4, y);
}
#endif


void raycasterKernelMs3IsoCascade(RaycasterKernelParams& params)
{
#ifdef RAYCASTER_ENABLE_MS3_ISO_CASCADE
	switch(params.filterMode) {
		#ifdef RAYCASTER_ENABLE_LINEAR
		case TEXTURE_FILTER_LINEAR : RAYCASTER_COLOR_SWITCH(ms3IsoCascadeKernel, TEXTURE_FILTER_LINEAR); break;
		#endif
		#ifdef RAYCASTER_ENABLE_CUBIC
		case TEXTURE_FILTER_CUBIC  : RAYCASTER_COLOR_SWITCH(ms3IsoCascadeKernel, TEXTURE_FILTER_CUBIC); break;
		#endif
	}
#endif
}
