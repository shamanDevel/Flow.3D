#include "LineMode.h"

#include <cassert>


namespace
{
	const char* g_lineModeName[LINE_MODE_COUNT + 1] = {
		"Stream",
		"Path",
		"Particle-Stream",
		"Particles (new seeds)",
		"Path FTLE",
		//"Streak",
		"Unknown"
	};
}

const char* GetLineModeName(eLineMode mode)
{
	return g_lineModeName[min(mode, LINE_MODE_COUNT)];
}

eLineMode GetLineModeFromName(const std::string& name)
{
	for(uint i = 0; i < LINE_MODE_COUNT; i++)
	{
		if(g_lineModeName[i] == name)
		{
			return eLineMode(i);
		}
	}
	return LINE_MODE_COUNT;
}

bool LineModeIsTimeDependent(eLineMode mode)
{
	switch(mode)
	{
		case LINE_STREAM:
			return false;

		case LINE_PATH_FTLE:
			return false;
		
		case LINE_PATH:
			return true;

		case LINE_PARTICLE_STREAM:
		case LINE_PARTICLES:
			return false;

		default:
			assert(false);
			return false;
	}
}

bool LineModeIsIterative(eLineMode mode)
{
	if (mode == LINE_PARTICLE_STREAM ||
		mode == LINE_PARTICLES)
		return true;
	return false;
}

bool LineModeGenerateAlwaysNewSeeds(eLineMode mode)
{
	return (mode == LINE_PARTICLES);
}
