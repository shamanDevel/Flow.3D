#ifndef __TUM3D__TEXTURE_FILTER_MODE_H__
#define __TUM3D__TEXTURE_FILTER_MODE_H__


#include <global.h>

#include <string>

#include <cuda_runtime.h>


enum eTextureFilterMode
{
	TEXTURE_FILTER_LINEAR = 0,
	TEXTURE_FILTER_CUBIC,
	TEXTURE_FILTER_CATROM,
	TEXTURE_FILTER_CATROM_STAGGERED,
	TEXTURE_FILTER_LAGRANGE4,
	TEXTURE_FILTER_LAGRANGE6,
	TEXTURE_FILTER_LAGRANGE8,
	TEXTURE_FILTER_LAGRANGE16,
	//TEXTURE_FILTER_WENO4,
	//TEXTURE_ANALYTIC_DOUBLEGYRE,
	TEXTURE_FILTER_MODE_COUNT,
	TEXTURE_FILTER_FORCE32 = 0xFFFFFFFF
};
const char* GetTextureFilterModeName(eTextureFilterMode filterMode);
eTextureFilterMode GetTextureFilterModeFromName(const std::string& name);
cudaTextureFilterMode GetCudaTextureFilterMode(eTextureFilterMode filterMode);
uint GetTextureFilterModeCellRadius(eTextureFilterMode filterMode);


#endif
