#define RAYCASTER_ENABLE_ISO2_SEPARATE

#include "RaycasterKernelParams.h"
#ifdef RAYCASTER_ENABLE_ISO2_SEPARATE
#include "cudaUtil.h"
#include "RaycasterKernelDefines.h"


#include "RaycasterKernelGlobals.cuh"
#include "RaycasterKernelHelpers.cuh"

#include "RaycasterKernelIso2SeparateStep.cuh"



template <eMeasureSource measureSource1, eMeasureSource measureSource2, eTextureFilterMode F, eMeasureComputeMode C, eColorMode CM>
__global__ void iso2separateKernel(
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

	// translate into viewport (-> side-by-side stereo!)
	x += renderTargetOffset.x;
	y += renderTargetOffset.y;

	// find intersections with box
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

	// early-out z test
	if(depthLinear >= depthMaxLinear) return;

	// get initial color from render target
	uchar4 colorStart;
	surf2Dread(&colorStart, g_surfTarget, x * 4, y);

	float4 sum = rgbaUCharToFloat(colorStart);


	// get value at entry point
	bool bInsideCoarse = (c_raycastParams.isoValues.x < getMeasure<measureSource1, F, C>(c_raycastParams.measure1, g_texVolume1, w2t(pos), c_raycastParams.gridSpacing, c_raycastParams.measureScale1));
	bool bInsideFine = (c_raycastParams.isoValues.y < getMeasure<measureSource2, F, C>(c_raycastParams.measure2, g_texVolume2, w2t(pos), c_raycastParams.gridSpacing, c_raycastParams.measureScale2));

	// march along ray from front to back
	while((depthLinear + depthStepLinear) < depthMaxLinear && sum.w < opacityThreshold)
	{
		// go to end of current segment
		pos += step;
		depthLinear += depthStepLinear;

		iso2separateStep<measureSource1, measureSource2, F, C, CM>(sum, bInsideCoarse, bInsideFine, world2texOffset, world2texScale, w2t(rayPos), pos, step, rayDir);
	}

	// if we didn't hit the alpha threshold, do final (smaller) step to z buffer hit (or brick exit point)
	if(sum.w < opacityThreshold)
	{
		step *= (depthMaxLinear - depthLinear) / depthStepLinear;
		pos += step;

		iso2separateStep<measureSource1, measureSource2, F, C, CM>(sum, bInsideCoarse, bInsideFine, world2texOffset, world2texScale, w2t(rayPos), pos, step, rayDir);
	}

	// write output color
	surf2Dwrite(rgbaFloatToUChar(sum), g_surfTarget, x * 4, y);
}
#endif


void raycasterKernelIso2Separate(RaycasterKernelParams& params)
{
#ifdef RAYCASTER_ENABLE_ISO2_SEPARATE
	switch(params.filterMode) {
		#ifdef RAYCASTER_ENABLE_LINEAR
		case TEXTURE_FILTER_LINEAR : RAYCASTER_COLOR_SWITCH_RT2(iso2separateKernel, TEXTURE_FILTER_LINEAR); break;
		#endif
		#ifdef RAYCASTER_ENABLE_CUBIC
		case TEXTURE_FILTER_CUBIC  : RAYCASTER_COLOR_SWITCH_RT2(iso2separateKernel, TEXTURE_FILTER_CUBIC); break;
		#endif
	}
#endif
}
