#include "ClearCudaArray.h"

#include "cudaUtil.h"


surface<void, cudaSurfaceType2D> g_surf2D;


template<typename T>
__global__ void clearCudaArray2D(uint width, uint height)
{
	uint x = blockIdx.x * blockDim.x + threadIdx.x;
	uint y = blockIdx.y * blockDim.y + threadIdx.y;
	if(x < width && y < height)
	{
		surf2Dwrite(T(), g_surf2D, x * sizeof(T), y);
	}
}


void clearCudaArray2Duchar4(cudaArray* pArray, uint width, uint height)
{
	cudaBindSurfaceToArray(g_surf2D, pArray);

	dim3 blockSize(16, 16);
	dim3 blockCount((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);
	clearCudaArray2D<uchar4><<<blockCount, blockSize>>>(width, height);
	cudaCheckMsg("clearCudaArray2D kernel execution failed");
}
