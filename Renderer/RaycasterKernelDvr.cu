#define RAYCASTER_ENABLE_DVR

#include "RaycasterKernelParams.h"
#ifdef RAYCASTER_ENABLE_DVR
#include "cudaUtil.h"
#include "RaycasterKernelDefines.h"


#include "RaycasterKernelGlobals.cuh"
#include "RaycasterKernelHelpers.cuh"



template <eMeasureSource measureSource, eTextureFilterMode F, eMeasureComputeMode C, eColorMode CM> 
__global__ void dvrKernel(
	int2 brickMinScreen,
	int2 brickSizeScreen,
	int2 renderTargetOffset,
	float3 boxMin,
	float3 boxMax,
	float3 world2texOffset,
	float3 world2texScale
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

	// current position and step increment in world space
	float3 pos = rayPos + rayDir * tnear;
	float3 step = rayDir * c_raycastParams.stepSizeWorld;
	float depthLinear = -transformPos(c_raycastParams.view, pos).z;
	float depthStepLinear = -transformDir(c_raycastParams.view, step).z;

	// read depth buffer
	float depthMax;
	surf2Dread(&depthMax, g_surfDepth, x * sizeof(float), y);
	float depthMaxLinear = depthToLinear(depthMax);
	// restrict depthMaxLinear to exit point depth, so we can use it as stop criterion
	depthMaxLinear = min(depthMaxLinear, -transformPos(c_raycastParams.view, rayPos + rayDir * tfar).z);

	if(depthLinear >= depthMaxLinear) return;

	// get initial color from render target
	uchar4 colorStart;
	surf2Dread(&colorStart, g_surfTarget, x * 4, y);

	float4 sum = rgbaUCharToFloat(colorStart);

	float stepFactor = c_raycastParams.stepSizeWorld * c_raycastParams.density;

	// march along ray from front to back, accumulating color
	while((depthLinear + depthStepLinear) < depthMaxLinear && sum.w < opacityThreshold)
	{
		float4 color = getColor<measureSource, F, C, false>(c_raycastParams.measure1, g_texVolume1, w2t(pos), rayDir, c_raycastParams.gridSpacing,
			stepFactor, c_raycastParams.transferOffset, c_raycastParams.transferScale, c_raycastParams.tfAlphaScale, c_raycastParams.measureScale1);
		sum += (1.0f - sum.w) * color;

		pos += step;
		depthLinear += depthStepLinear;
	}

	// do last (partial) step
	float lastStepRatio = min((depthMaxLinear - depthLinear) / depthStepLinear, 1.0f);
	float4 color = getColor<measureSource, F, C, false>(c_raycastParams.measure1, g_texVolume1, w2t(pos), rayDir, c_raycastParams.gridSpacing,
		stepFactor * lastStepRatio, c_raycastParams.transferOffset, c_raycastParams.transferScale, c_raycastParams.tfAlphaScale, c_raycastParams.measureScale1);
	sum += (1.0f - sum.w) * color;

	// write output color
	surf2Dwrite(rgbaFloatToUChar(sum), g_surfTarget, x * 4, y);
}
#endif


void raycasterKernelDvr(RaycasterKernelParams& params)
{
#ifdef RAYCASTER_ENABLE_DVR
	// color mode isn't used -> directly call RAYCASTER_COMPUTE_SWITCH
	switch(params.filterMode) {
		#ifdef RAYCASTER_ENABLE_LINEAR
		case TEXTURE_FILTER_LINEAR : RAYCASTER_COMPUTE_SWITCH_RT(dvrKernel, TEXTURE_FILTER_LINEAR, COLOR_MODE_UI); break;
		#endif
		#ifdef RAYCASTER_ENABLE_CUBIC
		case TEXTURE_FILTER_CUBIC  : RAYCASTER_COMPUTE_SWITCH_RT(dvrKernel, TEXTURE_FILTER_CUBIC,  COLOR_MODE_UI); break;
		#endif
	}
#endif
}
