#define RAYCASTER_ENABLE_ISO

#include "RaycasterKernelParams.h"
#ifdef RAYCASTER_ENABLE_ISO
#include "cudaUtil.h"
#include "RaycasterKernelDefines.h"


#include "RaycasterKernelGlobals.cui"
#include "RaycasterKernelHelpers.cui"


template <eMeasureSource measureSource, eTextureFilterMode F, eMeasureComputeMode C, bool shadeSI>
__device__ inline void isoStep(
	float4& sum,
	bool& wasInside,
	const float3& world2texOffset,
	float world2texScale,
	const float3& pos,
	const float3& step,
	const float3& rayDir)
{
	bool isInside = (getMeasure<measureSource, F, C>(c_raycastParams.measure1, g_texVolume1, w2t(pos), c_raycastParams.gridSpacing, c_raycastParams.measureScale1) >= c_raycastParams.isoValues.x);

	// did we cross an iso-surface?
	if(wasInside != isInside)
	{
		float3 posOut = isInside ? pos - step : pos;
		float3 posIn  = isInside ? pos : pos - step;
		float3 posIsoTex = binarySearch<measureSource, F, C>(c_raycastParams.measure1, g_texVolume1, w2t(posOut), w2t(posIn), c_raycastParams.gridSpacing, c_raycastParams.isoValues.x, c_raycastParams.measureScale1);

		float3 grad = getGradient<measureSource, F, C>(c_raycastParams.measure1, g_texVolume1, posIsoTex, c_raycastParams.gridSpacing);
		float4 color = shadeSI ? shadeScaleInvariant(rayDir, grad, c_raycastParams.isoColor1) : shadeIsosurface(rayDir, grad, c_raycastParams.isoColor1);
		// color from vorticity direction:
		//float3 vort = fabs(normalize(getVorticity<F>(g_texVolume1, pos)));
		//float4 color = shadeIsosurface(rayDir, grad, make_float4(vort, c_raycastParams.isoColor1.w));

		sum += (1.0f - sum.w) * color;
	}

	wasInside = isInside;
}

template <eMeasureSource measureSource, eTextureFilterMode F, eMeasureComputeMode C, bool shadeSI>
__device__ inline void isoRaycast(
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

	bool wasInside = (getMeasure<measureSource, F, C>(c_raycastParams.measure1, g_texVolume1, w2t(pos), c_raycastParams.gridSpacing, c_raycastParams.measureScale1) >= c_raycastParams.isoValues.x);

	// march along ray from front to back
	while((depthLinear + depthStepLinear) < depthMaxLinear && sum.w < opacityThreshold)
	{
		// go to end of current segment
		pos += step;
		depthLinear += depthStepLinear;

		isoStep<measureSource, F, C, shadeSI>(sum, wasInside, world2texOffset, world2texScale, pos, step, rayDir);
	}

	// check at z buffer hit (or brick exit point)
	if(sum.w < opacityThreshold)
	{
		step *= (depthMaxLinear - depthLinear) / depthStepLinear;
		pos += step;

		isoStep<measureSource, F, C, shadeSI>(sum, wasInside, world2texOffset, world2texScale, pos, step, rayDir);
	}

	// write output color
	surf2Dwrite(rgbaFloatToUChar(sum), g_surfTarget, x * 4, y);
}

template <eMeasureSource measureSource, eTextureFilterMode F, eMeasureComputeMode C, eColorMode CM>
__global__ void isoKernel(
	int2 brickMinScreen,
	int2 brickSizeScreen,
	int2 renderTargetOffset,
	float3 boxMin,
	float3 boxMax,
	float3 world2texOffset,
	float world2texScale
)
{
	isoRaycast<measureSource, F, C, false>(brickMinScreen, brickSizeScreen, renderTargetOffset, boxMin, boxMax, world2texOffset, world2texScale);
}

template <eMeasureSource measureSource, eTextureFilterMode F, eMeasureComputeMode C, eColorMode CM>
__global__ void isoSiKernel(
	int2 brickMinScreen,
	int2 brickSizeScreen,
	int2 renderTargetOffset,
	float3 boxMin,
	float3 boxMax,
	float3 world2texOffset,
	float world2texScale
)
{
	isoRaycast<measureSource, F, C, true>(brickMinScreen, brickSizeScreen, renderTargetOffset, boxMin, boxMax, world2texOffset, world2texScale);
}
#endif


void raycasterKernelIso(RaycasterKernelParams& params)
{
#ifdef RAYCASTER_ENABLE_ISO
	// color mode isn't used -> directly call RAYCASTER_COMPUTE_SWITCH
	switch(params.filterMode) {
		#ifdef RAYCASTER_ENABLE_LINEAR
		case TEXTURE_FILTER_LINEAR    : RAYCASTER_COMPUTE_SWITCH_RT(isoKernel, TEXTURE_FILTER_LINEAR, COLOR_MODE_UI); break;
		#endif
		#ifdef RAYCASTER_ENABLE_CUBIC
		case TEXTURE_FILTER_CUBIC     : RAYCASTER_COMPUTE_SWITCH_RT(isoKernel, TEXTURE_FILTER_CUBIC,  COLOR_MODE_UI); break;
		#endif
		//FIXME Catmull-Rom etc don't work with precomp measures at the moment (nvcc snafu), so skip COMPUTE_SWITCH here
		#ifdef RAYCASTER_ENABLE_CATROM
		case TEXTURE_FILTER_CATROM    : if(params.measureComputeMode == MEASURE_COMPUTE_ONTHEFLY) { RAYCASTER_MEASURE_SWITCH_RT(isoKernel, TEXTURE_FILTER_CATROM, MEASURE_COMPUTE_ONTHEFLY, COLOR_MODE_UI); } break;
		#endif
		#ifdef RAYCASTER_ENABLE_LAGRANGE4
		case TEXTURE_FILTER_LAGRANGE4 : if(params.measureComputeMode == MEASURE_COMPUTE_ONTHEFLY) { RAYCASTER_MEASURE_SWITCH_RT(isoKernel, TEXTURE_FILTER_LAGRANGE4, MEASURE_COMPUTE_ONTHEFLY, COLOR_MODE_UI); } break;
		#endif
		#ifdef RAYCASTER_ENABLE_LAGRANGE6
		case TEXTURE_FILTER_LAGRANGE6 : if(params.measureComputeMode == MEASURE_COMPUTE_ONTHEFLY) { RAYCASTER_MEASURE_SWITCH_RT(isoKernel, TEXTURE_FILTER_LAGRANGE6, MEASURE_COMPUTE_ONTHEFLY, COLOR_MODE_UI); } break;
		#endif
		#ifdef RAYCASTER_ENABLE_WENO4
		case TEXTURE_FILTER_WENO4     : if(params.measureComputeMode == MEASURE_COMPUTE_ONTHEFLY) { RAYCASTER_MEASURE_SWITCH_RT(isoKernel, TEXTURE_FILTER_WENO4,  MEASURE_COMPUTE_ONTHEFLY, COLOR_MODE_UI); } break;
		#endif
	}
#endif
}

void raycasterKernelIsoSi(RaycasterKernelParams& params)
{
#ifdef RAYCASTER_ENABLE_ISO
	// color mode isn't used -> directly call RAYCASTER_COMPUTE_SWITCH
	switch(params.filterMode) {
		#ifdef RAYCASTER_ENABLE_LINEAR
		case TEXTURE_FILTER_LINEAR    : RAYCASTER_COMPUTE_SWITCH_RT(isoSiKernel, TEXTURE_FILTER_LINEAR, COLOR_MODE_UI); break;
		#endif
		#ifdef RAYCASTER_ENABLE_CUBIC
		case TEXTURE_FILTER_CUBIC     : RAYCASTER_COMPUTE_SWITCH_RT(isoSiKernel, TEXTURE_FILTER_CUBIC,  COLOR_MODE_UI); break;
		#endif
	}
#endif
}
