#ifndef __TUM3D__RAYCASTER_KERNEL_PARAMS_H__
#define __TUM3D__RAYCASTER_KERNEL_PARAMS_H__


#include "global.h"

#include <vector>

#include "cutil_math.h"

#include "Measure.h"
#include "RaycastMode.h"
#include "TextureFilterMode.h"


struct ProjectionParamsGPU
{
	float    left;
	float    right;
	float    bottom;
	float    top;
	float    eyeOffset;
	uint     imageWidth;
	uint     imageHeight;
	float4   depthParams; // near, far, far*near, far-near
};

struct RaycastParamsGPU
{
	float3 gridSpacing;

	float3x4 view;
	float3x4 viewInv;

	float stepSizeWorld;

	eMeasure measure1;
	eMeasure measure2;
	float measureScale1;
	float measureScale2;

	float density;
	float transferOffset;
	float transferScale;
	float tfAlphaScale;

	float3 isoValues;
	float4 isoColor1;
	float4 isoColor2;
	float4 isoColor3;
};


class BrickSlot;

struct RaycasterKernelParams
{
	RaycasterKernelParams(
		eMeasure measure1, eMeasure measure2, eMeasureComputeMode measureComputeMode,
		eTextureFilterMode filterMode,
		eColorMode colorMode,
		const std::vector<BrickSlot*>& brickSlots,
		dim3 blockSize,
		dim3 gridSize,
		int2 brickMinScreen,
		int2 brickSizeScreen,
		int2 renderTargetOffset,
		float3 boxMin,
		float3 boxMax,
		float3 world2texOffset,
		float3 world2texScale)
	: measure1(measure1), measure2(measure2), measureComputeMode(measureComputeMode),
	  filterMode(filterMode),
	  colorMode(colorMode),
	  brickSlots(brickSlots),
	  blockSize(blockSize), gridSize(gridSize),
	  brickMinScreen(brickMinScreen), brickSizeScreen(brickSizeScreen), renderTargetOffset(renderTargetOffset),
	  boxMin(boxMin), boxMax(boxMax),
	  world2texOffset(world2texOffset), world2texScale(world2texScale) {}

	eMeasure measure1;
	eMeasure measure2;
	eMeasureComputeMode measureComputeMode;
	eTextureFilterMode filterMode;
	eColorMode colorMode;
	const std::vector<BrickSlot*>& brickSlots;
	dim3 blockSize;
	dim3 gridSize;
	int2 brickMinScreen;
	int2 brickSizeScreen;
	int2 renderTargetOffset;
	float3 boxMin;
	float3 boxMax;
	float3 world2texOffset;
	float3 world2texScale;
};


#endif
