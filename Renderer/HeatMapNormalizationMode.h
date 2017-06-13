#ifndef __TUM3D__HEATMAP_NORMALIZATION_MODE_H
#define __TUM3D__HEATMAP_NORMALIZATION_MODE_H

#include <global.h>

#include <string>

enum eHeatMapNormalizationMode
{
	NORMALIZATION_OFF = 0,
	NORMALIZATION_MAX,
	NORMALIZATION_MEAN,
	NORMALIZATION_MEDIAN,
	HEAT_MAP_NORMALIZATION_MODE_COUNT
};
std::string GetHeatMapNormalizationModeName(eHeatMapNormalizationMode renderMode);
eHeatMapNormalizationMode GetHeatMapNormalizationModeFromName(const std::string& name);

#endif