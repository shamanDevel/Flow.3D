#include "IntegratorTimeInCell.cuh"

#include <iostream>

#include "VolumeInfoGPU.h"
#include "IntegrationParamsGPU.h"

extern __constant__ VolumeInfoGPU c_volumeInfo;
extern __constant__ IntegrationParamsGPU c_integrationParams;
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
	static const int CURRENT_CELL_INDEX = 0;	//for some reasons, when the shader calls RecordedCellIndices[0],
	static const int THRESHOLD_CELL_INDEX = 1;	// the value at index 1 is read. So lets use index 1 for now instead of 0
	
	if (!c_integrationParams.timeInCellEnabled) {
		return; //no seed texture
	}
	
	// find out current cell
	float3 posInVolume = (vertex->Position + c_volumeInfo.volumeHalfSizeWorld) / (2 * c_volumeInfo.volumeHalfSizeWorld);
	float texX = posInVolume.x;
	float texY = 1 - posInVolume.y;
	uint32 cell = tex2D(g_cellTexture, texX, texY);
	//printf("pos=(%5.3f, %5.3f, %5.3f), cell=%x\n", vertex->Position.x, vertex->Position.y, vertex->Position.z, cell);
	//vertex->RecordedCellIndices[0] = cell;
	//vertex->RecordedCellIndices[1] = cell;
	//vertex->RecordedCellIndices[2] = cell;
	vertex->RecordedCellIndices[3] = cell;
}
