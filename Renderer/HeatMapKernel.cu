#include "HeatMapKernel.h"

#include <iostream>
#include <thrust/reduce.h>
#include <thrust/device_ptr.h>
#include <thrust/transform_reduce.h>
#include <thrust/functional.h>
#include <thrust/execution_policy.h>
#include <thrust/copy.h>
#include <thrust/binary_search.h>
#include <cudaUtil.h>

__global__ void heatmapFillChannelKernel(uint* channel,
	LineVertex* vertices, uint* indices, uint numVertices,
	int3 size, float3 worldOffset, float3 worldToGrid,
	unsigned int* seedTexture, int2 seedTextureSize, unsigned int seedPicked)
{
	uint index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index > numVertices) return;
	if (vertices[indices[index]].Time < 0) return;

	float3 position = vertices[indices[index]].Position;
	float3 gridPositionF = (position + worldOffset) * worldToGrid;
	int3 gridPosition = make_int3(gridPositionF);
	if (gridPosition.x < 0 || gridPosition.y < 0 || gridPosition.z < 0
		|| gridPosition.x >= size.x || gridPosition.y >= size.y || gridPosition.z >= size.z) {
		return;
	}
	
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

struct clip_at_one : public thrust::unary_function<uint, uint>
{
	__host__ __device__ uint operator()(const uint &x) const
	{
		return min(x, 1);
	}
};

struct divide_by_count : public thrust::unary_function<uint, float>
{
	const float factor;
	divide_by_count(float factor) : factor(1.0 / factor) {}
	__host__ __device__ float operator()(const uint &x) const
	{
		return x * factor;
	}
};

float heatmapKernelFindMean(uint* channel, int3 size)
{
	thrust::device_ptr<uint> thrustChannelPtr(channel);
	size_t count = size.x * size.y * size.z;
	uint countNonZeros = thrust::transform_reduce(thrustChannelPtr, thrustChannelPtr + count, 
		clip_at_one(), 0, thrust::plus<uint>());
	//divide by count inplace to avoid integer overflow
	float mean = thrust::transform_reduce(thrustChannelPtr, thrustChannelPtr + count,
		divide_by_count((float) countNonZeros), 0.0f, thrust::plus<float>());
	std::cout << "count: " << count << ", count of non-zeros: " << countNonZeros
		<< ", mean: " << mean << std::endl;
	return mean;
}

struct heatmap_non_zero_pred
{
	__host__ __device__ bool operator()(const uint x) const
	{
		return x >= 1;
	}
};

float heatmapKernelFindMedian(uint * channel, int3 size)
{
	thrust::device_ptr<uint> thrustChannelPtr(channel);
	size_t count = size.x * size.y * size.z;
	//1. collect all non-zero values
	/*
	uint countNonZeros = thrust::transform_reduce(thrustChannelPtr, thrustChannelPtr + count,
		clip_at_one(), 0, thrust::plus<uint>());
	if (countNonZeros == 0) return 1;
	uint* nonZeroEntries;
	cudaSafeCall(cudaMalloc2(&nonZeroEntries, sizeof(uint) * countNonZeros));
	thrust::device_ptr<uint> nonZeroEntriesPtr(nonZeroEntries);
	thrust::copy_if(thrust::device, thrustChannelPtr, thrustChannelPtr + count, nonZeroEntriesPtr, heatmap_non_zero_pred());
	//2. sort that array
	thrust::sort(nonZeroEntriesPtr, nonZeroEntriesPtr + countNonZeros);
	//3. store median
	uint median;
	cudaSafeCall(cudaMemcpy(&median, nonZeroEntries + countNonZeros / 2, sizeof(uint), cudaMemcpyDeviceToHost));
	//4. clean up
	cudaSafeCall(cudaFree(nonZeroEntries));
	*/

	uint* nonZeroEntries;
	cudaSafeCall(cudaMalloc2(&nonZeroEntries, sizeof(uint) * count));
	cudaSafeCall(cudaMemcpy(nonZeroEntries, channel, sizeof(uint) * count, cudaMemcpyDeviceToDevice));
	thrust::device_ptr<uint> nonZeroEntriesPtr(nonZeroEntries);
	thrust::sort(nonZeroEntriesPtr, nonZeroEntriesPtr + count);
	auto it = thrust::upper_bound(nonZeroEntriesPtr, nonZeroEntriesPtr + count, 0);
	size_t start = it - nonZeroEntriesPtr;
	size_t index_of_median = (count - start) / 2 + start;
	uint median;
	cudaSafeCall(cudaMemcpy(&median, nonZeroEntries + index_of_median, sizeof(uint), cudaMemcpyDeviceToHost));
	cudaSafeCall(cudaFree(nonZeroEntries));

	std::cout << "Median: " << median << std::endl;
	return max(1, median);
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
