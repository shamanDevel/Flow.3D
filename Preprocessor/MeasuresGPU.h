#ifndef __TUM3D__MEASURESGPU_H__
#define __TUM3D__MEASURESGPU_H__

#include "MeasuresCPU.h"

struct cudaArray;

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

void computeMeasureMinMaxGPU(
	VolumeTextureCPU& texCPU, const vec3& h,
    eMeasureSource measureSource, eMeasure measure,
    MinMaxMeasureGPUHelperData& helperData,
    float& minVal, float& maxVal);

#endif
