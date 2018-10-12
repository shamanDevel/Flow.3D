#include "LineRenderMode.h"


namespace
{
	const char* g_lineRenderModeName[LINE_RENDER_MODE_COUNT + 1] = {
		"Line",
		"Ribbon",
		"Tube",
		"Particle",
		"Unknown"
	};
}

const char* GetLineRenderModeName(eLineRenderMode renderMode)
{
	return g_lineRenderModeName[min(renderMode, LINE_RENDER_MODE_COUNT)];
}

eLineRenderMode GetLineRenderModeFromName(const std::string& name)
{
	for(uint i = 0; i < LINE_RENDER_MODE_COUNT; i++)
	{
		if(g_lineRenderModeName[i] == name)
		{
			return eLineRenderMode(i);
		}
	}
	return LINE_RENDER_MODE_COUNT;
}
