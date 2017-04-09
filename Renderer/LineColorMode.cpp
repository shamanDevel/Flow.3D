#include "LineColorMode.h"


namespace
{
	std::string g_lineColorModeName[LINE_COLOR_MODE_COUNT + 1] = {
		"Line ID",
		"Age",
		"Texture",
		"Measure",
		"Unknown"
	};
}

std::string GetLineColorModeName(eLineColorMode renderMode)
{
	return g_lineColorModeName[min(renderMode, LINE_COLOR_MODE_COUNT)];
}

eLineColorMode GetLineColorModeFromName(const std::string& name)
{
	for(uint i = 0; i < LINE_COLOR_MODE_COUNT; i++)
	{
		if(g_lineColorModeName[i] == name)
		{
			return eLineColorMode(i);
		}
	}
	return LINE_COLOR_MODE_COUNT;
}
