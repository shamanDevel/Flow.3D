#ifndef __TUM3D__BRICKSLOT_H__
#define __TUM3D__BRICKSLOT_H__


#include <global.h>

#include <vector>

#include <cuda_runtime.h>

#include <Vec.h>


// a cubic brick of volume data, stored in a cuda array
// only part of the array may be filled with data
// actually, this is now a "multi-slot": can hold multiple bricks in one cuda array
class BrickSlot
{
public:
	BrickSlot();
	BrickSlot(BrickSlot&& other);
	~BrickSlot();

	BrickSlot& operator=(BrickSlot&& other);

	// create/release cuda resources
	bool Create(uint size, uint channelCount, const tum3D::Vec3ui& slotCount = tum3D::Vec3ui(1, 1, 1));
	void Release();
	bool IsCreated() const { return !m_pArray.empty(); }

	// fill with data (must be created first)

	// fill from CPU array (with interleaved channels);
	// for the async flag to work, the memory must be page-locked (cudaMallocHost/cudaHostAlloc or cudaHostRegister)
	void Fill(const float* pData, const tum3D::Vec3ui& dataSize, bool async = false);
	// fill from GPU arrays (one array per channel)
	void FillFromGPUChannels(const float** dppChannelData, const tum3D::Vec3ui& dataSize, const tum3D::Vec3ui& slotIndex = tum3D::Vec3ui(0, 0, 0));
	void FillTextureFromGPUChannels(uint textureIndex, const float** dppChannelData, const tum3D::Vec3ui& dataSize, const tum3D::Vec3ui& slotIndex = tum3D::Vec3ui(0, 0, 0));

	void CopySlot(const tum3D::Vec3ui& fromIndex, const tum3D::Vec3ui& toIndex);

	// brick is cubic, so size is a scalar
	uint GetSize() const { return m_size; }
	const tum3D::Vec3ui& GetFilledSize() const { return m_sizeFilled; }
	uint GetChannelCount() const { return m_channelCount; }
	const tum3D::Vec3ui& GetSlotCount() const { return m_slotCount; }
	uint GetTextureCount() const { return (uint)m_pArray.size(); }

	const cudaArray* GetCudaArray(uint index = 0) const { return m_pArray[index]; }

private:
	uint                    m_size;
	uint                    m_channelCount;
	tum3D::Vec3ui           m_slotCount;
	std::vector<cudaArray*> m_pArray;

	tum3D::Vec3ui           m_sizeFilled;


	// disallow copy and assignment
	BrickSlot(const BrickSlot&);
	BrickSlot& operator=(const BrickSlot&);
};


#endif
