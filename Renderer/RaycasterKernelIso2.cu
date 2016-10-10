#define RAYCASTER_ENABLE_ISO2

#include "RaycasterKernelParams.h"
#ifdef RAYCASTER_ENABLE_ISO2
#include "cudaUtil.h"
#include "RaycasterKernelDefines.h"


#include "RaycasterKernelGlobals.cui"
#include "RaycasterKernelHelpers.cui"



template <bool fromJac, eTextureFilterMode F, eMeasureComputeMode C, bool shadeSI> 
__device__ inline void iso2Step(
	float4& sum,
	float& oldVal,
	const float3& world2texOffset,
	const float world2texScale,
	const float3& rayPosTx,
	const float3& pos,
	const float3& step,
	const float3& rayDir)
{
	float val = getMeasure<fromJac,F,C>(c_raycastParams.measure1, g_texVolume1, w2t(pos), c_raycastParams.gridSpacing, c_raycastParams.measureScale1);

	float fDist = 1e10;
	float4 vColors[2] = {make_float4(0,0,0,0), make_float4(0,0,0,0)};

	// passed first iso surface?
	if ((val>=c_raycastParams.isoValues.x) != (oldVal>=c_raycastParams.isoValues.x))
	{
		float factor = (val>=c_raycastParams.isoValues.x) ? 1.0f : 0.0f;
		float3 pp = binarySearch<fromJac,F,C>(c_raycastParams.measure1, g_texVolume1, w2t(pos - factor * step), w2t(pos - (1.0f-factor) * step), c_raycastParams.gridSpacing, c_raycastParams.isoValues.x, c_raycastParams.measureScale1);

		float3 grad = getGradient<fromJac,F,C>(c_raycastParams.measure1, g_texVolume1, pp, c_raycastParams.gridSpacing);
		if(shadeSI)
			vColors[0] = shadeScaleInvariant(rayDir, grad, c_raycastParams.isoColor1);
		else
			vColors[0] = shadeIsosurface(rayDir, grad, c_raycastParams.isoColor1);

		fDist = lengthSq(pp-rayPosTx);
	}

	// passed second iso surface?
	if ((val>=c_raycastParams.isoValues.y) != (oldVal>=c_raycastParams.isoValues.y))
	{
		float factor = (val>=c_raycastParams.isoValues.y) ? 1.0f : 0.0f;
		float3 pp = binarySearch<fromJac,F,C>(c_raycastParams.measure1, g_texVolume1, w2t(pos - factor * step), w2t(pos - (1.0f-factor) * step), c_raycastParams.gridSpacing, c_raycastParams.isoValues.y, c_raycastParams.measureScale1);

		// sort hits
		vColors[1] = vColors[0];
		int index = (lengthSq(pp-rayPosTx) <= fDist) ? 0 : 1;

		float3 grad = getGradient<fromJac,F,C>(c_raycastParams.measure1, g_texVolume1, pp, c_raycastParams.gridSpacing);
		if(shadeSI)
			vColors[index] = shadeScaleInvariant(rayDir, grad, c_raycastParams.isoColor2);
		else
			vColors[index] = shadeIsosurface(rayDir, grad, c_raycastParams.isoColor2);
	}

	// blend both potential hits "behind"
	sum += (1.0f - sum.w) * vColors[0];
	sum += (1.0f - sum.w) * vColors[1];

	oldVal = val;
}

template <bool fromJac, eTextureFilterMode F, eMeasureComputeMode C, bool shadeSI> 
__device__ inline void iso2Raycast(
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

	// translate from brick's 2D screen space bbox to global
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
	float oldVal = getMeasure<fromJac,F,C>(c_raycastParams.measure1, g_texVolume1, w2t(pos), c_raycastParams.gridSpacing, c_raycastParams.measureScale1);

	// march along ray from front to back
	while((depthLinear + depthStepLinear) < depthMaxLinear && sum.w < opacityThreshold)
	{
		// go to end of current segment
		pos += step;
		depthLinear += depthStepLinear;

		iso2Step<fromJac,F,C,shadeSI>(sum, oldVal, world2texOffset, world2texScale, w2t(rayPos), pos, step, rayDir);
	}

	// if we didn't hit the alpha threshold, do final (smaller) step to z buffer hit (or brick exit point)
	if(sum.w < opacityThreshold)
	{
		step *= (depthMaxLinear - depthLinear) / depthStepLinear;
		pos += step;

		iso2Step<fromJac,F,C,shadeSI>(sum, oldVal, world2texOffset, world2texScale, w2t(rayPos), pos, step, rayDir);
	}

	// write output color
	surf2Dwrite(rgbaFloatToUChar(sum), g_surfTarget, x * 4, y);
}

template <bool fromJac, eTextureFilterMode F, eMeasureComputeMode C, eColorMode CM> 
__global__ void iso2Kernel(
	int2 brickMinScreen,
	int2 brickSizeScreen,
	int2 renderTargetOffset,
	float3 boxMin,
	float3 boxMax,
	float3 world2texOffset,
	float world2texScale
)
{
	iso2Raycast<fromJac,F,C,false>(brickMinScreen, brickSizeScreen, renderTargetOffset, boxMin, boxMax, world2texOffset, world2texScale);
}

template <bool fromJac, eTextureFilterMode F, eMeasureComputeMode C, eColorMode CM> 
__global__ void iso2SiKernel(
	int2 brickMinScreen,
	int2 brickSizeScreen,
	int2 renderTargetOffset,
	float3 boxMin,
	float3 boxMax,
	float3 world2texOffset,
	float world2texScale
)
{
	iso2Raycast<fromJac,F,C,true>(brickMinScreen, brickSizeScreen, renderTargetOffset, boxMin, boxMax, world2texOffset, world2texScale);
}
#endif


void raycasterKernelIso2(RaycasterKernelParams& params)
{
#ifdef RAYCASTER_ENABLE_ISO2
	// color mode isn't used -> directly call RAYCASTER_COMPUTE_SWITCH
	switch(params.filterMode) {
		#ifdef RAYCASTER_ENABLE_LINEAR
		case TEXTURE_FILTER_LINEAR : RAYCASTER_COMPUTE_SWITCH_RT(iso2Kernel, TEXTURE_FILTER_LINEAR, COLOR_MODE_UI); break;
		#endif
		#ifdef RAYCASTER_ENABLE_CUBIC
		case TEXTURE_FILTER_CUBIC  : RAYCASTER_COMPUTE_SWITCH_RT(iso2Kernel, TEXTURE_FILTER_CUBIC,  COLOR_MODE_UI); break;
		#endif
	}
#endif
}

void raycasterKernelIso2Si(RaycasterKernelParams& params)
{
#ifdef RAYCASTER_ENABLE_ISO2
	// color mode isn't used -> directly call RAYCASTER_COMPUTE_SWITCH
	switch(params.filterMode) {
		#ifdef RAYCASTER_ENABLE_LINEAR
		case TEXTURE_FILTER_LINEAR : RAYCASTER_COMPUTE_SWITCH_RT(iso2SiKernel, TEXTURE_FILTER_LINEAR, COLOR_MODE_UI); break;
		#endif
		#ifdef RAYCASTER_ENABLE_CUBIC
		case TEXTURE_FILTER_CUBIC  : RAYCASTER_COMPUTE_SWITCH_RT(iso2SiKernel, TEXTURE_FILTER_CUBIC,  COLOR_MODE_UI); break;
		#endif
	}
#endif
}
