#include "MeasuresGPU.h"
#include "../Renderer/Measures.cuh"
#include "../Renderer/Coords.cuh"

/**
 * Reference: Based on kernel 4 from https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf
 * @param input The array of input values (of size 'sizeOfInput').
 * @param output The output array (of size iceil(numberOfBlocksI, blockSize1D*2)).
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
        sdata[threadID] = 0;
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
 * @param output The output array (of size iceil(numberOfBlocksI, blockSize1D*2)).
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
        sdata[threadID] = 0;
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

inline int iceil(int x, int y)
{
    return 1 + ((x - 1) / y);
}


__device__ void computeMeasuresKernel(
        texture<float4, cudaTextureType3D, cudaReadModeElementType>& texVolume,
        int sizeX, int sizeY, int sizeZ, int brickOverlap,
        const float3& h, eMeasureSource measureSource, eMeasure measure,
        float* minValueReductionArray, float* maxValueReductionArray)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x + brickOverlap;
    int y = blockIdx.y * blockDim.y + threadIdx.y + brickOverlap;
    int z = blockIdx.z * blockDim.z + threadIdx.z + brickOverlap;

    int reductionWriteIndex = 0;
    if (x < sizeX - brickOverlap && y < sizeY - brickOverlap && z < sizeZ - brickOverlap) {
        // Sample at the voxel centers
        float3 pos{(x + 0.5f)/float(sizeX), (y + 0.5f)/float(sizeY), (z + 0.5f)/float(sizeZ)};
        float value = getMeasure<measureSource, TEXTURE_FILTER_LINEAR, MEASURE_COMPUTE_ONTHEFLY>(
            measure, texVolume, pos, h, 1.0f);
        minValueReductionArray[reductionWriteIndex] = value;
        maxValueReductionArray[reductionWriteIndex] = value;
        reductionWriteIndex++;
    }
}

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

    bool finished = false;
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
        calculateMaximum<<<numBlocks, BLOCK_SIZE, sharedMemorySize>>>(
                reductionInput, reductionOutput, numElements, BLOCK_SIZE);
    
        iteration++;
    }

    cudaMemcpy(&minValue, minReductionOutput, sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(&maxValue, maxReductionOutput, sizeof(float), cudaMemcpyDeviceToHost);
}

MinMaxMeasureGPUHelperData::MinMaxMeasureGPUHelperData(size_t sizeX, size_t sizeY, size_t sizeZ, size_t brickOverlap)
{
    int reductionArraySize = (sizeX - 2*brickOverlap)*(sizeY - 2*brickOverlap)*(sizeZ - 2*brickOverlap);
    cudaSafeCall(cudaMalloc(&reductionArrayMin0, reductionArraySize*sizeof(float)));
    cudaSafeCall(cudaMalloc(&reductionArrayMin1, reductionArraySize*sizeof(float)));
    cudaSafeCall(cudaMalloc(&reductionArrayMax0, reductionArraySize*sizeof(float)));
    cudaSafeCall(cudaMalloc(&reductionArrayMax1, reductionArraySize*sizeof(float)));
    cudaSafeCall(cudaMalloc3DArray(&textureArray, &channelDesc, extent, cudaArraySurfaceLoadStore));
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
        VolumeTextureCPU& texCPU, const vec3& h,
        eMeasureSource measureSource, eMeasure measure,
        MinMaxMeasureGPUHelperData& helperData,
        float& minVal, float& maxVal)
{
    texture<float4, cudaTextureType3D, cudaReadModeElementType> texVolume;
    texVolume.addressMode[0] = cudaAddressModeClamp;
    texVolume.addressMode[1] = cudaAddressModeClamp;
    texVolume.addressMode[2] = cudaAddressModeClamp;
    texVolume.normalized = false;
    texVolume.filterMode = cudaFilterModeLinear;    


    cudaArray* textureArray = helperData.textureArray;
    int sizeX = texCPU.getSizeX();
    int sizeY = texCPU.getSizeY();
    int sizeZ = texCPU.getSizeZ();
    int brickOverlap = texCPU.getBrickOverlap();

    float* cpuData = helperData.cpuData;
    std::vector<std::vector<float>>& channelData = texCPU.getChannelData();
    size_t n = channelData.at(0).size();
    for (int i = 0; i < n; i++) {
        for (int c = 0; c < 4; c++) {
            cpuData[i*4 + c] = channelData.at(c).at(i);
        }
    }

    // Upload the channel data to the GPU
    cudaChannelFormatDesc channelDesc;
	channelDesc = cudaCreateChannelDesc<float4>();
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
    
    float3 h_cu{h.x, h.y, h.z};
    computeMeasuresKernel<<<dimGrid,dimBlock>>>(
        texVolume, sizeX, sizeY, sizeZ, brickOverlap,
        h_cu, eMeasureSource measureSource, eMeasure measure,
        reductionArrayMin0, reductionArrayMax0);
    reduceMeasureArray(
        minVal, maxVal,
        reductionArrayMin0, reductionArrayMin1,
        reductionArrayMax0, reductionArrayMax1,
        sizeX, sizeY, sizeZ, brickOverlap);

    // Clean-up
    cudaSafeCall(cudaUnbindTexture(texVolume));
}
