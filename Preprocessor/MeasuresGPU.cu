/**
 * @author Christoph Neuhauser
 */

#include <cudaUtil.h>
#include "MeasuresGPU.h"
#include "../Renderer/Measures.cuh"

/**
 * Reference: Based on kernel 4 from https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf
 * @param input The array of input values (of size 'sizeOfInput').
 * @param output The output array (of size iceil(sizeOfInput, blockSize1D*2)).
 * @param sizeOfInput The number of input values.
 */
 __global__ void calculateMinimum(float* input, float* output, int sizeOfInput, int blockSize1D) {
    extern __shared__ float sdata[];

    unsigned int threadID = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x * 2 + threadIdx.x;

    // Copy the data to the shared memory and do the first reduction step.
    if (i + blockDim.x < sizeOfInput){
        sdata[threadID] = fminf(input[i], input[i + blockDim.x]);
    } else if (i < sizeOfInput){
        sdata[threadID] = input[i];
    } else{
        sdata[threadID] = FLT_MAX;
    }
    __syncthreads();

    // Do the reduction in the shared memory.
    for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadID < stride) {
            sdata[threadID] = fminf(sdata[threadID], sdata[threadID + stride]);
        }
        __syncthreads();
    }

    // Write the result for this block to global memory.
    if (threadID == 0) {
        output[blockIdx.x] = sdata[0];
    }
}

/**
 * Reference: Based on kernel 4 from https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf
 * @param input The array of input values (of size 'sizeOfInput').
 * @param output The output array (of size iceil(sizeOfInput, blockSize1D*2)).
 * @param sizeOfInput The number of input values.
 */
 __global__ void calculateMaximum(float* input, float* output, int sizeOfInput, int blockSize1D) {
    extern __shared__ float sdata[];

    unsigned int threadID = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x * 2 + threadIdx.x;

    // Copy the data to the shared memory and do the first reduction step.
    if (i + blockDim.x < sizeOfInput){
        sdata[threadID] = fmaxf(input[i], input[i + blockDim.x]);
    } else if (i < sizeOfInput){
        sdata[threadID] = input[i];
    } else{
        sdata[threadID] = -FLT_MAX;
    }
    __syncthreads();

    // Do the reduction in the shared memory.
    for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadID < stride) {
            sdata[threadID] = fmaxf(sdata[threadID], sdata[threadID + stride]);
        }
        __syncthreads();
    }

    // Write the result for this block to global memory.
    if (threadID == 0) {
        output[blockIdx.x] = sdata[0];
    }
}

// The volume texture
texture<float4, cudaTextureType3D, cudaReadModeElementType> texVolume;

/**
 * Computes the passed measure type at all locations in the brick.
 */
template <eMeasureSource measureSource>
__global__ void computeMeasuresKernel(
        int sizeX, int sizeY, int sizeZ, int brickOverlap,
		float3 h, eMeasure measure,
        float* minValueReductionArray, float* maxValueReductionArray)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;
	int k = blockIdx.z * blockDim.z + threadIdx.z;
	int x = i + brickOverlap;
    int y = j + brickOverlap;
    int z = k + brickOverlap;

    if (x < sizeX - brickOverlap && y < sizeY - brickOverlap && z < sizeZ - brickOverlap) {
        // Sample at the voxel centers
		float3 pos{ x + 0.5f, y + 0.5f, z + 0.5f };
		float value = getMeasure<measureSource, TEXTURE_FILTER_LINEAR, MEASURE_COMPUTE_ONTHEFLY>(
			measure, texVolume, pos, h, 1.0f);

		int reductionWriteIndex = i + (j + k * (sizeY - 2 * brickOverlap)) * (sizeX - 2 * brickOverlap);
		minValueReductionArray[reductionWriteIndex] = value;
        maxValueReductionArray[reductionWriteIndex] = value;
    }
}

inline int iceil(int x, int y)
{
	return 1 + ((x - 1) / y);
}

/**
 * Computes the minimum/maximum value in reductionArrayMin0/reductionArrayMax0.
 */
void reduceMeasureArray(
        float& minValue, float& maxValue,
        float* reductionArrayMin0, float* reductionArrayMin1,
        float* reductionArrayMax0, float* reductionArrayMax1,
        int sizeX, int sizeY, int sizeZ, int brickOverlap)
{
    const int BLOCK_SIZE = 256;
    int numBlocks = (sizeX-2*brickOverlap)*(sizeY-2*brickOverlap)*(sizeZ-2*brickOverlap);
    int inputSize;

    if (numBlocks == 0) {
        return;
    }

    float* minReductionInput;
    float* maxReductionInput;
    float* minReductionOutput;
    float* maxReductionOutput;

    int iteration = 0;
    while (numBlocks > 1) {
        inputSize = numBlocks;
        numBlocks = iceil(numBlocks, BLOCK_SIZE*2);

        if (iteration % 2 == 0) {
            minReductionInput = reductionArrayMin0;
            maxReductionInput = reductionArrayMax0;
            minReductionOutput = reductionArrayMin1;
            maxReductionOutput = reductionArrayMax1;
        } else {
            minReductionInput = reductionArrayMin1;
            maxReductionInput = reductionArrayMax1;
            minReductionOutput = reductionArrayMin0;
            maxReductionOutput = reductionArrayMax0;
        }

        int sharedMemorySize = BLOCK_SIZE * sizeof(float);
		calculateMinimum<<<numBlocks, BLOCK_SIZE, sharedMemorySize>>>(
			minReductionInput, minReductionOutput, inputSize, BLOCK_SIZE);
		calculateMaximum<<<numBlocks, BLOCK_SIZE, sharedMemorySize>>>(
			maxReductionInput, maxReductionOutput, inputSize, BLOCK_SIZE);

        iteration++;
    }

    cudaMemcpy(&minValue, minReductionOutput, sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(&maxValue, maxReductionOutput, sizeof(float), cudaMemcpyDeviceToHost);
}

MinMaxMeasureGPUHelperData::MinMaxMeasureGPUHelperData(size_t sizeX, size_t sizeY, size_t sizeZ, size_t brickOverlap)
{
	cudaChannelFormatDesc channelDesc;
	channelDesc = cudaCreateChannelDesc<float4>();
    int reductionArraySize = int((sizeX - 2*brickOverlap)*(sizeY - 2*brickOverlap)*(sizeZ - 2*brickOverlap));
    cudaSafeCall(cudaMalloc(&reductionArrayMin0, reductionArraySize*sizeof(float)));
    cudaSafeCall(cudaMalloc(&reductionArrayMin1, reductionArraySize*sizeof(float)));
    cudaSafeCall(cudaMalloc(&reductionArrayMax0, reductionArraySize*sizeof(float)));
    cudaSafeCall(cudaMalloc(&reductionArrayMax1, reductionArraySize*sizeof(float)));
    cudaSafeCall(cudaMalloc3DArray(&textureArray, &channelDesc, make_cudaExtent(sizeX, sizeY, sizeZ), cudaArraySurfaceLoadStore));
    cpuData = new float[sizeX*sizeY*sizeZ*4]; // 4 channels
}

MinMaxMeasureGPUHelperData::~MinMaxMeasureGPUHelperData()
{
    cudaSafeCall(cudaFreeArray(textureArray));
    cudaSafeCall(cudaFree(reductionArrayMin0));
    cudaSafeCall(cudaFree(reductionArrayMin1));
    cudaSafeCall(cudaFree(reductionArrayMax0));
    cudaSafeCall(cudaFree(reductionArrayMax1));
    delete[] cpuData;
}

void computeMeasureMinMaxGPU(
        VolumeTextureCPU& texCPU, const tum3D::Vec3f& h,
        eMeasureSource measureSource, eMeasure measure,
        MinMaxMeasureGPUHelperData* helperData,
        float& minVal, float& maxVal)
{
	texVolume.addressMode[0] = cudaAddressModeClamp;
	texVolume.addressMode[1] = cudaAddressModeClamp;
	texVolume.addressMode[2] = cudaAddressModeClamp;
	texVolume.normalized = false;
	texVolume.filterMode = cudaFilterModeLinear;

    cudaArray* textureArray = helperData->textureArray;
    int sizeX = int(texCPU.getSizeX());
    int sizeY = int(texCPU.getSizeY());
    int sizeZ = int(texCPU.getSizeZ());
    int brickOverlap = int(texCPU.getBrickOverlap());

	// Copy the channels to contiguous memory.
    float* cpuData = helperData->cpuData;
    const std::vector<std::vector<float>>& channelData = texCPU.getChannelData();
    size_t n = channelData.at(0).size();
	#pragma omp parallel for
    for (int i = 0; i < n; i++) {
        for (int c = 0; c < 4; c++) {
            cpuData[i*4 + c] = channelData.at(c).at(i);
        }
    }

    // Upload the channel data to the GPU.
	cudaMemcpy3DParms memcpyParams = { 0 };
	memcpyParams.srcPtr   = make_cudaPitchedPtr(cpuData, sizeX * 4 * sizeof(float), sizeX, sizeY);
	memcpyParams.dstArray = textureArray;
	memcpyParams.extent   = make_cudaExtent(sizeX, sizeY, sizeZ);
	memcpyParams.kind     = cudaMemcpyHostToDevice;

    cudaSafeCall(cudaMemcpy3D(&memcpyParams));
	cudaSafeCall(cudaBindTextureToArray(texVolume, textureArray));

    dim3 dimBlock(32, 4, 1);
    dim3 dimGrid(
        iceil(sizeX-2*brickOverlap, dimBlock.x),
        iceil(sizeY-2*brickOverlap, dimBlock.y),
        iceil(sizeZ-2*brickOverlap, dimBlock.z));

	// Compute the measure for all locations in the brick.
    float3 h_cu{h.x(), h.y(), h.z() };
	switch (measureSource) {
	case MEASURE_SOURCE_RAW:
		computeMeasuresKernel<MEASURE_SOURCE_RAW><<<dimGrid, dimBlock>>>(
			sizeX, sizeY, sizeZ, brickOverlap, h_cu, measure,
			helperData->reductionArrayMin0, helperData->reductionArrayMax0);
		break;
	case MEASURE_SOURCE_HEAT_CURRENT:
		computeMeasuresKernel<MEASURE_SOURCE_HEAT_CURRENT><<<dimGrid, dimBlock>>>(
			sizeX, sizeY, sizeZ, brickOverlap, h_cu, measure,
			helperData->reductionArrayMin0, helperData->reductionArrayMax0);
		break;
	case MEASURE_SOURCE_JACOBIAN:
		computeMeasuresKernel<MEASURE_SOURCE_JACOBIAN><<<dimGrid, dimBlock>>>(
			sizeX, sizeY, sizeZ, brickOverlap, h_cu, measure,
			helperData->reductionArrayMin0, helperData->reductionArrayMax0);
		break;
	}

	// Compute minimum and maximum measure value.
    reduceMeasureArray(
        minVal, maxVal,
		helperData->reductionArrayMin0, helperData->reductionArrayMin1,
		helperData->reductionArrayMax0, helperData->reductionArrayMax1,
        sizeX, sizeY, sizeZ, brickOverlap);

    // Clean-up.
	cudaSafeCall(cudaUnbindTexture(texVolume));
}
