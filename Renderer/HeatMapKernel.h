#ifndef __TUM3D__HEATMAP_KERNEL_H__
#define __TUM3D__HEATMAP_KERNEL_H__

#include <utility>
#include <cuda_runtime.h>
#include "TracingCommon.h"

void heatmapKernelFillChannel(uint* channel,
	LineVertex* vertices, uint* indices, uint numVertices,
	int3 size, float3 worldOffset, float3 worldToGrid,
	unsigned int* seedTexture, int2 seedTextureSize, unsigned int seedPicked);

std::pair<uint, uint> heatmapKernelFindMinMax(uint* channel, int3 size);

float heatmapKernelFindMean(uint* channel, int3 size);

float heatmapKernelFindMedian(uint* channel, int3 size);

void heatmapKernelCopyChannel(uint* channel, float* texture, int3 size);

#endif