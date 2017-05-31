#include "HeatMapKernel.h"

#include <thrust/reduce.h>
#include <thrust/device_ptr.h>

__global__ void heatmapFillChannelKernel(uint* channel,
	LineVertex* vertices, uint* indices, uint numVertices,
	int3 size, float3 worldOffset, float3 worldToGrid)
{
	uint index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index > numVertices) return;

	float3 position = vertices[indices[index]].Position;
	int3 gridPosition = make_int3((position + worldOffset) * worldToGrid);
	int gridIndex = gridPosition.x + size.x * (gridPosition.y + size.y * gridPosition.z);

	atomicAdd(channel + gridIndex, 1);
}


void heatmapKernelFillChannel(uint* channel, 
	LineVertex* vertices, uint* indices, uint numVertices, 
	int3 size, float3 worldOffset, float3 worldToGrid)
{
	uint blockSize = 128;
	uint blockCount = (numVertices + blockSize - 1) / blockSize;
	heatmapFillChannelKernel<<<blockCount, blockSize>>>(
		channel, vertices, indices, numVertices,
		size, worldOffset, worldToGrid);
}

std::pair<uint, uint> heatmapKernelFindMinMax(uint * channel, int3 size)
{
	thrust::device_ptr<uint> thrustChannelPtr(channel);
	size_t count = size.x * size.y * size.z;
	uint minValue = thrust::reduce(thrustChannelPtr, thrustChannelPtr + count, 0, thrust::minimum<uint>());
	uint maxValue = thrust::reduce(thrustChannelPtr, thrustChannelPtr + count, 0, thrust::maximum<uint>());

	return std::pair<uint, uint>(minValue, maxValue);
}

__global__ void heatmapCopyChannelKernel(uint* channel, float* texture, int count)
{
	uint index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index > count) return;
	texture[index] = (float)channel[index];
}

void heatmapKernelCopyChannel(uint * channel, float * texture, int3 size)
{
	int count = size.x * size.y * size.z;
	uint blockSize = 128;
	uint blockCount = (count + blockSize - 1) / blockSize;
	heatmapCopyChannelKernel <<<blockCount, blockSize >>> (
		channel, texture, count);
}
