#include "HeatMapNormalizationMode.h"

namespace
{
	std::string g_heatMapNormalizationModeName[HEAT_MAP_NORMALIZATION_MODE_COUNT + 1] = {
		"Off",
		"Max",
		"Mean",
		"Median",
		"Unknown"
	};
}

std::string GetHeatMapNormalizationModeName(eHeatMapNormalizationMode renderMode)
{
	return g_heatMapNormalizationModeName[min(renderMode, HEAT_MAP_NORMALIZATION_MODE_COUNT)];
}
eHeatMapNormalizationMode GetHeatMapNormalizationModeFromName(const std::string& name)
{
	for (uint i = 0; i < HEAT_MAP_NORMALIZATION_MODE_COUNT; i++)
	{
		if (g_heatMapNormalizationModeName[i] == name)
		{
			return eHeatMapNormalizationMode(i);
		}
	}
	return HEAT_MAP_NORMALIZATION_MODE_COUNT;
}
