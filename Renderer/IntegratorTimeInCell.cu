#include "IntegratorTimeInCell.cuh"

#include <iostream>

#include "VolumeInfoGPU.h"
#include "IntegrationParamsGPU.h"

//extern __constant__ VolumeInfoGPU c_volumeInfo;
extern __constant__ IntegrationParamsGPU c_integrationParams;
texture<uint32, cudaTextureType2D, cudaReadModeElementType> g_cellTexture;

void IntegratorTimeInCell::Upload(CellTextureGPU& info, uint32 * textureMemCPU, size_t width, size_t height)
{
	Free(info);
	std::cout << "cudaMallocArray " << (width * height * sizeof(uint32)) / 1024.0f << "KB" << std::endl;
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

__device__ void IntegratorTimeInCell::processParticle(LineVertex * vertex, VolumeInfoGPU& c_volumeInfo, float deltaTime)
{
	if (!c_integrationParams.timeInCellEnabled) {
		return; //no seed texture
	}
	
	// find out current cell
	float3 posInVolume = (vertex->Position + c_volumeInfo.volumeHalfSizeWorld) / (2 * c_volumeInfo.volumeHalfSizeWorld);
	float texX = posInVolume.x;
	float texY = 1 - posInVolume.y;
	uint32 cell = tex2D(g_cellTexture, texX, texY);
	uint32 currentCell = vertex->RecordedCellIndices[0];
	if (currentCell == cell) {
		//increase time
		vertex->TimeInCell[0] += deltaTime;
	}
	else if (currentCell == 0) {
		//no previous cell
		vertex->RecordedCellIndices[0] = cell;
		vertex->TimeInCell[0] = deltaTime;
	}
	else {
		//outside of the current cell
		vertex->TimeInCell[1] += deltaTime;
		if (vertex->TimeInCell[1] > c_integrationParams.cellChangeThreshold) {
			//we are there long enough, switch cell
			vertex->RecordedCellIndices[0] = cell;
			vertex->TimeInCell[0] = vertex->TimeInCell[1];
			vertex->TimeInCell[1] = 0;
		} //else: stay still in the same cell to prevent noise
	}

	//check for longest cell
	if (vertex->TimeInCell[0] > vertex->TimeInCell[2]) {
		//longest cell changed
		vertex->TimeInCell[2] = vertex->TimeInCell[0];
		vertex->RecordedCellIndices[2] = vertex->RecordedCellIndices[0];
	}

	//TODO: eventually add second longest cell
	//similar to above, just with index 3
}
