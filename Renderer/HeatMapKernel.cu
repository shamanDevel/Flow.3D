#include "HeatMapKernel.h"

#include <thrust/reduce.h>
#include <thrust/device_ptr.h>

__global__ void heatmapFillChannelKernel(uint* channel,
	LineVertex* vertices, uint* indices, uint numVertices,
	int3 size, float3 worldOffset, float3 worldToGrid,
	unsigned int* seedTexture, int2 seedTextureSize, unsigned int seedPicked)
{
	uint index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index > numVertices) return;

	float3 position = vertices[indices[index]].Position;
	float3 gridPositionF = (position + worldOffset) * worldToGrid;
	int3 gridPosition = make_int3(gridPositionF);
	
	if (seedTexture != NULL && seedPicked > 0) {
		//get position
		float3 seedPos = vertices[indices[index]].SeedPosition;
		seedPos = (seedPos + worldOffset) * worldToGrid / make_float3(size);
		seedPos.y = 1 - seedPos.y;
		//test against seed texture
		int seedX = min(seedTextureSize.x - 1, max(0, (int)(seedPos.x * seedTextureSize.x)));
		int seedY = min(seedTextureSize.y - 1, max(0, (int)(seedPos.y * seedTextureSize.y)));
		unsigned int seed = seedTexture[seedX + seedTextureSize.x * seedY];
		if (seed != seedPicked) {
			return;
		}
	}

	int gridIndex = gridPosition.x + size.x * (gridPosition.y + size.y * gridPosition.z);
	atomicAdd(channel + gridIndex, 1);
}


void heatmapKernelFillChannel(uint* channel, 
	LineVertex* vertices, uint* indices, uint numVertices, 
	int3 size, float3 worldOffset, float3 worldToGrid,
	unsigned int* seedTexture, int2 seedTextureSize, unsigned int seedPicked)
{
	uint blockSize = 128;
	uint blockCount = (numVertices + blockSize - 1) / blockSize;
	heatmapFillChannelKernel<<<blockCount, blockSize>>>(
		channel, vertices, indices, numVertices,
		size, worldOffset, worldToGrid,
		seedTexture, seedTextureSize, seedPicked);
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
