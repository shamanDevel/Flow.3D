#include "IntegratorTimeInCell.cuh"

#include <iostream>

texture<uint32, cudaTextureType2D, cudaReadModeElementType> g_cellTexture;

void IntegratorTimeInCell::Upload(CellTextureGPU& info, uint32 * textureMemCPU, size_t width, size_t height)
{
	Free(info);

	cudaSafeCall(cudaMallocArray(&info.textureArray, &g_cellTexture.channelDesc, width, height));
	cudaSafeCall(cudaMemcpyToArray(info.textureArray, 0, 0, textureMemCPU, width * height * sizeof(uint32), cudaMemcpyHostToDevice));
	cudaChannelFormatDesc channelFormat = { 32, 0, 0, 0, cudaChannelFormatKindUnsigned };
	cudaBindTextureToArray(g_cellTexture, info.textureArray);
	g_cellTexture.normalized = true;
	g_cellTexture.addressMode[0] = cudaAddressModeClamp;

	std::cout << "IntegratorTimeInCell: CellTexture uploaded" << std::endl;
}

void IntegratorTimeInCell::Free(CellTextureGPU & info)
{
	if (info.textureArray == NULL) return;
	cudaSafeCall(cudaUnbindTexture(g_cellTexture));
	cudaSafeCall(cudaFreeArray(info.textureArray));
	info.textureArray = NULL;
	std::cout << "IntegratorTimeInCell: CellTexture freed" << std::endl;
}

__device__ void IntegratorTimeInCell::processParticle(LineVertex * vertex, float deltaTime)
{
	
}
