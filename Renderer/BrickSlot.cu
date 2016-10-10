#include "BrickSlot.h"

#include <cassert>
#include <utility>

#include <cudaUtil.h>

using namespace tum3D;



surface<void, cudaSurfaceType3D> g_surfArray;


BrickSlot::BrickSlot()
	: m_size(0), m_sizeFilled(0, 0, 0)
	, m_channelCount(0)
	, m_slotCount(0, 0, 0)
{
}

BrickSlot::BrickSlot(BrickSlot&& other)
	: m_size(other.m_size)
	, m_channelCount(other.m_channelCount)
	, m_slotCount(other.m_slotCount)
	, m_sizeFilled(other.m_sizeFilled)
{
	m_pArray.swap(other.m_pArray);
}

BrickSlot::~BrickSlot()
{
	assert(!IsCreated());
}


BrickSlot& BrickSlot::operator=(BrickSlot&& other)
{
	if(this != &other)
	{
		Release();

		m_size = other.m_size;
		m_channelCount = other.m_channelCount;
		m_slotCount = other.m_slotCount;

		m_sizeFilled = other.m_sizeFilled;

		m_pArray.swap(other.m_pArray);
	}

	return *this;
}



bool BrickSlot::Create(uint size, uint channelCount, const Vec3ui& slotCount)
{
	Release();

	if(channelCount < 1) {
		return false;
	}
	if(slotCount.volume() < 1) {
		return false;
	}

	m_size = size;
	m_channelCount = channelCount;
	m_slotCount = slotCount;

	Vec3ui textureSize = m_size * m_slotCount;
	cudaExtent extent = make_cudaExtent(textureSize.x(), textureSize.y(), textureSize.z());

	uint textureCount = (m_channelCount + 3) / 4;
	std::vector<cudaChannelFormatDesc> channelDesc(textureCount);
	m_pArray.resize(textureCount);

	for(uint tex = 0; tex < textureCount; tex++)
	{
		uint channelCountThisTex = ((tex == textureCount - 1) ? (m_channelCount % 4) : 4);

		switch(channelCountThisTex) {
			case 1:
				channelDesc[tex] = cudaCreateChannelDesc<float1>();
				break;
			case 2:
				channelDesc[tex] = cudaCreateChannelDesc<float2>();
				break;
			case 3: // no 3-channel textures in cuda - fall-through to 4
			case 4:
			default:
				channelDesc[tex] = cudaCreateChannelDesc<float4>();
				break;
		}

		cudaError_t result = cudaMalloc3DArray(&m_pArray[tex], &channelDesc[tex], extent, cudaArraySurfaceLoadStore);

		if(result != cudaSuccess) {
			// the only error that should happen here is out-of-memory
			assert(result == cudaErrorMemoryAllocation);
			if(result != cudaErrorMemoryAllocation)
			{
				printf("BrickSlot::Create failed with cuda error %i\n", int(result));
			}
			// clear the error, or the next error check will fail...
			cudaGetLastError();

			Release();
			return false;
		}
	}

	return true;
}

void BrickSlot::Release()
{
	for(size_t i = 0; i < m_pArray.size(); i++)
	{
		cudaSafeCall(cudaFreeArray(m_pArray[i]));
	}
	m_pArray.clear();

	m_size = 0;
	m_channelCount = 0;
	m_slotCount.set(0, 0, 0);

	m_sizeFilled.set(0, 0, 0);
}

void BrickSlot::Fill(const float* pData, const Vec3ui& dataSize, bool async)
{
	assert(IsCreated());

	if(m_channelCount > 4)
	{
		printf("BrickSlot::Fill not implemented for > 4 channels\n");
		return;
	}
	if(m_channelCount == 3)
	{
		printf("BrickSlot::Fill not implemented for 3 channels (TODO...)\n");
		return;
	}

	m_sizeFilled.set(m_size, m_size, m_size);
	minimum(m_sizeFilled, dataSize);

	cudaMemcpy3DParms memcpyParams = { 0 };
	memcpyParams.srcPtr   = make_cudaPitchedPtr(const_cast<float*>(pData), dataSize.x() * m_channelCount * sizeof(float), dataSize.x(), dataSize.y());
	memcpyParams.dstArray = m_pArray[0];
	memcpyParams.extent   = make_cudaExtent(m_sizeFilled.x(), m_sizeFilled.y(), m_sizeFilled.z());
	memcpyParams.kind     = cudaMemcpyHostToDevice;

	if(async) {
		cudaSafeCall(cudaMemcpy3DAsync(&memcpyParams));
	} else {
		cudaSafeCall(cudaMemcpy3D(&memcpyParams));
	}
}

__global__ void BrickCopyFromChannelKernel(
	uint sizeX, uint sizeY, uint sizeZ,
	uint offsetX, uint offsetY, uint offsetZ,
	const float* pChannel0,
	uint rowPitch, uint slicePitch)
{
	uint x = blockIdx.x * blockDim.x + threadIdx.x;
	uint y = blockIdx.y * blockDim.y + threadIdx.y;
	uint z = blockIdx.z * blockDim.z + threadIdx.z;

	if(x >= sizeX || y >= sizeY || z >= sizeZ) return;

	uint index = x + rowPitch * y + slicePitch * z;

	float value = pChannel0[index];
	surf3Dwrite(value, g_surfArray, (offsetX + x) * sizeof(float), offsetY + y, offsetZ + z);
}

__global__ void BrickCopyFrom2ChannelsKernel(
	uint sizeX, uint sizeY, uint sizeZ,
	uint offsetX, uint offsetY, uint offsetZ,
	const float* pChannel0, const float* pChannel1,
	uint rowPitch, uint slicePitch)
{
	uint x = blockIdx.x * blockDim.x + threadIdx.x;
	uint y = blockIdx.y * blockDim.y + threadIdx.y;
	uint z = blockIdx.z * blockDim.z + threadIdx.z;

	if(x >= sizeX || y >= sizeY || z >= sizeZ) return;

	uint index = x + rowPitch * y + slicePitch * z;

	float2 value;
	value.x = pChannel0[index];
	value.y = pChannel1[index];
	surf3Dwrite(value, g_surfArray, (offsetX + x) * 2 * sizeof(float), offsetY + y, offsetZ + z);
}

__global__ void BrickCopyFrom3ChannelsKernel(
	uint sizeX, uint sizeY, uint sizeZ,
	uint offsetX, uint offsetY, uint offsetZ,
	const float* pChannel0, const float* pChannel1, const float* pChannel2,
	uint rowPitch, uint slicePitch)
{
	uint x = blockIdx.x * blockDim.x + threadIdx.x;
	uint y = blockIdx.y * blockDim.y + threadIdx.y;
	uint z = blockIdx.z * blockDim.z + threadIdx.z;

	if(x >= sizeX || y >= sizeY || z >= sizeZ) return;

	uint index = x + rowPitch * y + slicePitch * z;

	float4 value;
	value.x = pChannel0[index];
	value.y = pChannel1[index];
	value.z = pChannel2[index];
	value.w = 0.0f;
	surf3Dwrite(value, g_surfArray, (offsetX + x) * 4 * sizeof(float), offsetY + y, offsetZ + z);
}

__global__ void BrickCopyFrom4ChannelsKernel(
	uint sizeX, uint sizeY, uint sizeZ,
	uint offsetX, uint offsetY, uint offsetZ,
	const float* pChannel0, const float* pChannel1, const float* pChannel2, const float* pChannel3,
	uint rowPitch, uint slicePitch)
{
	uint x = blockIdx.x * blockDim.x + threadIdx.x;
	uint y = blockIdx.y * blockDim.y + threadIdx.y;
	uint z = blockIdx.z * blockDim.z + threadIdx.z;

	if(x >= sizeX || y >= sizeY || z >= sizeZ) return;

	uint index = x + rowPitch * y + slicePitch * z;

	float4 value;
	value.x = pChannel0[index];
	value.y = pChannel1[index];
	value.z = pChannel2[index];
	value.w = pChannel3[index];
	surf3Dwrite(value, g_surfArray, (offsetX + x) * 4 * sizeof(float), offsetY + y, offsetZ + z);
}

void BrickSlot::FillFromGPUChannels(const float** dppChannelData, const Vec3ui& dataSize, const Vec3ui& slotIndex)
{
	uint textureCount = GetTextureCount();
	for(uint textureIndex = 0; textureIndex < textureCount; textureIndex++)
	{
		FillTextureFromGPUChannels(textureIndex, dppChannelData + textureIndex * 4, dataSize, slotIndex);
	}
}

void BrickSlot::FillTextureFromGPUChannels(uint textureIndex, const float** dppChannelData, const Vec3ui& dataSize, const Vec3ui& slotIndex)
{
	assert(IsCreated());

	//FIXME ? (we're filling only one tex, but setting this globally..)
	m_sizeFilled.set(m_size, m_size, m_size);
	minimum(m_sizeFilled, dataSize);

	uint textureCount = GetTextureCount();
	assert(textureIndex < textureCount);

	Vec3ui offset = slotIndex * m_size;

	dim3 blockSize(32, 3, 1);
	dim3 blockCount(
		(m_sizeFilled.x() + blockSize.x - 1) / blockSize.x,
		(m_sizeFilled.y() + blockSize.y - 1) / blockSize.y,
		(m_sizeFilled.z() + blockSize.z - 1) / blockSize.z);

	cudaSafeCall(cudaBindSurfaceToArray(g_surfArray, m_pArray[textureIndex]));

	uint channelCountThisTex = ((textureIndex == textureCount - 1) ? (m_channelCount % 4) : 4);
	switch(channelCountThisTex) {
		case 1:
			BrickCopyFromChannelKernel<<<blockCount, blockSize>>>(
				m_sizeFilled.x(), m_sizeFilled.y(), m_sizeFilled.z(),
				offset.x(), offset.y(), offset.z(),
				dppChannelData[0],
				dataSize.x(), dataSize.x() * dataSize.y());
			cudaCheckMsg("BrickCopyFromChannelKernel execution failed");
			break;
		case 2:
			BrickCopyFrom2ChannelsKernel<<<blockCount, blockSize>>>(
				m_sizeFilled.x(), m_sizeFilled.y(), m_sizeFilled.z(),
				offset.x(), offset.y(), offset.z(),
				dppChannelData[0], dppChannelData[1],
				dataSize.x(), dataSize.x() * dataSize.y());
			cudaCheckMsg("BrickCopyFrom2ChannelsKernel execution failed");
			break;
		case 3:
			BrickCopyFrom3ChannelsKernel<<<blockCount, blockSize>>>(
				m_sizeFilled.x(), m_sizeFilled.y(), m_sizeFilled.z(),
				offset.x(), offset.y(), offset.z(),
				dppChannelData[0], dppChannelData[1], dppChannelData[2],
				dataSize.x(), dataSize.x() * dataSize.y());
			cudaCheckMsg("BrickCopyFrom3ChannelsKernel execution failed");
			break;
		case 4:
			BrickCopyFrom4ChannelsKernel<<<blockCount, blockSize>>>(
				m_sizeFilled.x(), m_sizeFilled.y(), m_sizeFilled.z(),
				offset.x(), offset.y(), offset.z(),
				dppChannelData[0], dppChannelData[1], dppChannelData[2], dppChannelData[3],
				dataSize.x(), dataSize.x() * dataSize.y());
			cudaCheckMsg("BrickCopyFrom4ChannelsKernel execution failed");
			break;
	}
}

void BrickSlot::CopySlot(const Vec3ui& fromIndex, const Vec3ui& toIndex)
{
	Vec3ui srcPos = fromIndex * m_size;
	Vec3ui dstPos = toIndex   * m_size;
	cudaMemcpy3DParms params = {};
	params.kind = cudaMemcpyDeviceToDevice;
	params.srcPos = make_cudaPos(srcPos.x(), srcPos.y(), srcPos.z());
	params.dstPos = make_cudaPos(dstPos.x(), dstPos.y(), dstPos.z());
	params.extent = make_cudaExtent(m_sizeFilled.x(), m_sizeFilled.y(), m_sizeFilled.z());
	for(size_t textureIndex = 0; textureIndex < m_pArray.size(); textureIndex++)
	{
		params.srcArray = m_pArray[textureIndex];
		params.dstArray = m_pArray[textureIndex];
		cudaSafeCall(cudaMemcpy3DAsync(&params));
	}
}
