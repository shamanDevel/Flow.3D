#include "TextureFilterMode.h"

#include <cassert>


namespace
{
	std::string g_textureFilterModeName[TEXTURE_FILTER_MODE_COUNT + 1] = {
		"Linear",
		"Cubic",
		"Catmull-Rom",
		"Catmull-Rom-Staggered",
		"Lagrange4",
		"Lagrange6",
		"Lagrange8",
		"Lagrange16",
		//"WENO4",
		//"Analytic DoubleGyre",
		"Unknown"
	};
}

std::string GetTextureFilterModeName(eTextureFilterMode filterMode)
{
	return g_textureFilterModeName[min(filterMode, TEXTURE_FILTER_MODE_COUNT)];
}

eTextureFilterMode GetTextureFilterModeFromName(const std::string& name)
{
	for(uint i = 0; i < TEXTURE_FILTER_MODE_COUNT; i++)
	{
		if(g_textureFilterModeName[i] == name)
		{
			return eTextureFilterMode(i);
		}
	}
	return TEXTURE_FILTER_MODE_COUNT;
}

cudaTextureFilterMode GetCudaTextureFilterMode(eTextureFilterMode filterMode)
{
	switch(filterMode)
	{
		case TEXTURE_FILTER_LINEAR:
		case TEXTURE_FILTER_CUBIC:
			return cudaFilterModeLinear;
		case TEXTURE_FILTER_CATROM:
		case TEXTURE_FILTER_CATROM_STAGGERED:
		case TEXTURE_FILTER_LAGRANGE4:
		case TEXTURE_FILTER_LAGRANGE6:
		case TEXTURE_FILTER_LAGRANGE8:
		case TEXTURE_FILTER_LAGRANGE16:
		//case TEXTURE_FILTER_WENO4:
			return cudaFilterModePoint;
		// for analytic fields, this is irrelevant, but avoid the assert below..
		//case TEXTURE_ANALYTIC_DOUBLEGYRE:
		//	return cudaFilterModePoint;
	}
	assert(false);
	return cudaFilterModeLinear;
}


uint GetTextureFilterModeCellRadius(eTextureFilterMode filterMode)
{
	switch(filterMode)
	{
		case TEXTURE_FILTER_LINEAR:
			return 0;
		case TEXTURE_FILTER_CUBIC:
		case TEXTURE_FILTER_CATROM:
		case TEXTURE_FILTER_LAGRANGE4:
		//case TEXTURE_FILTER_WENO4:
			return 1;
		case TEXTURE_FILTER_CATROM_STAGGERED:
		case TEXTURE_FILTER_LAGRANGE6:
			return 2;
		case TEXTURE_FILTER_LAGRANGE8:
			return 3;
		case TEXTURE_FILTER_LAGRANGE16:
			return 7;
		//case TEXTURE_ANALYTIC_DOUBLEGYRE:
		//	return 0;
	}
	assert(false);
	return 0;
}
