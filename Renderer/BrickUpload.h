#ifndef __TUM3D__BRICKUPLOAD_H__
#define __TUM3D__BRICKUPLOAD_H__


#include <global.h>

#include <vector>

#include <TimeVolumeIO.h>
#include <Vec.h>

#include "BrickSlot.h"
#include "MultiTimerGPU.h"

#include "CompressVolume.h"
#include "GPUResources.h"


// note: passed in data must *not* be page-locked already!

void UploadBrick(GPUResources* pShared, CompressVolumeResources* pRes,
				 const TimeVolumeInfo& volumeInfo, const TimeVolumeIO::Brick& brick,
				 float** pdpBuffer,
				 BrickSlot* pBrickSlot = nullptr, const tum3D::Vec3ui& slotIndex = tum3D::Vec3ui(0, 0, 0),
				 MultiTimerGPU* pTimer = nullptr);
void UploadBrick(GPUResources* pShared, CompressVolumeResources* pRes,
				 const std::vector<uint*>& data, const std::vector<uint>& dataSize,
				 const tum3D::Vec3ui& brickSize,
				 eCompressionType compressionType, const std::vector<float>& quantSteps, bool lessRLE,
				 float** pdpBuffer,
				 BrickSlot* pBrickSlot = nullptr, const tum3D::Vec3ui& slotIndex = tum3D::Vec3ui(0, 0, 0), int textureIndex = -1,
				 MultiTimerGPU* pTimer = nullptr);
void UploadBrickChannel(GPUResources* pShared, CompressVolumeResources* pRes,
						const TimeVolumeInfo& volumeInfo, const TimeVolumeIO::Brick& brick,
						int channel,
						float* dpBuffer,
						MultiTimerGPU* pTimer = nullptr);
void UploadBrickChannel(GPUResources* pShared, CompressVolumeResources* pRes,
						const uint* pData, uint dataSize,
						const tum3D::Vec3ui& brickSize,
						eCompressionType compressionType, float quantStep, bool lessRLE,
						int channel,
						float* dpBuffer,
						MultiTimerGPU* pTimer = nullptr);


#endif
