#include "LineMode.h"

#include <cassert>


namespace
{
	std::string g_lineModeName[LINE_MODE_COUNT + 1] = {
		"Stream",
		"Path",
		"Particle-Stream",
		//"Streak",
		"Unknown"
	};
}

std::string GetLineModeName(eLineMode mode)
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

		case LINE_PATH:
			return true;

		case LINE_PARTICLE_STREAM:
			return false;

		default:
			assert(false);
			return false;
	}
}

bool LineModeIsIterative(eLineMode mode)
{
	if (mode == LINE_PARTICLE_STREAM) {
		return true;
	}
	return false;
}
