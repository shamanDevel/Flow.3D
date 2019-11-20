/**
 * @author Christoph Neuhauser
 */

#ifndef __TUM3D__MEASURESGPU_H__
#define __TUM3D__MEASURESGPU_H__

#include <vector>
#include "Vec.h"
#include "../Renderer/Measure.h"

struct cudaArray;

/**
 * This struct stores arrays shared between consecutive invocations of @see computeMeasureMinMaxGPU.
 */
struct MinMaxMeasureGPUHelperData {
	MinMaxMeasureGPUHelperData(size_t sizeX, size_t sizeY, size_t sizeZ, size_t brickOverlap);
	~MinMaxMeasureGPUHelperData();

	cudaArray* textureArray;
	float* cpuData;
	float* reductionArrayMin0;
	float* reductionArrayMin1;
	float* reductionArrayMax0;
	float* reductionArrayMax1;
};

/******************************************************************************
** Class for encapsulating a volume texture on the CPU
******************************************************************************/

class VolumeTextureCPU {
public:
	VolumeTextureCPU(const std::vector<std::vector<float>>& channelData,
		size_t sizeX, size_t sizeY, size_t sizeZ, size_t brickOverlap)
		: channelData(channelData), sizeX(sizeX), sizeY(sizeY), sizeZ(sizeZ), brickOverlap(brickOverlap) {}

	inline const std::vector<std::vector<float>>& getChannelData() { return channelData; }
	inline size_t getSizeX() { return sizeX; }
	inline size_t getSizeY() { return sizeY; }
	inline size_t getSizeZ() { return sizeZ; }
	inline size_t getBrickOverlap() { return brickOverlap; }

private:
	size_t sizeX, sizeY, sizeZ;
	size_t brickOverlap;
	const std::vector<std::vector<float>>& channelData;
};

/**
 * Computes the minimum and maximum value for the passed measure type for the brick texture.
 * h is the spacing between two cells.
 */
void computeMeasureMinMaxGPU(
	VolumeTextureCPU& texCPU, const tum3D::Vec3f& h,
	eMeasureSource measureSource, eMeasure measure,
	MinMaxMeasureGPUHelperData* helperData,
	float& minVal, float& maxVal);

#endif
