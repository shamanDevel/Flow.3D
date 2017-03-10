#include "ParticleRenderMode.h"

namespace
{
	std::string g_particleRenderModeName[PARTICLE_RENDER_MODE_COUNT + 1] = {
		"Additive",
		"Multiplicative",
		"Unknown"
	};
}

std::string GetParticleRenderModeName(eParticleRenderMode renderMode)
{
	return g_particleRenderModeName[min(renderMode, PARTICLE_RENDER_MODE_COUNT)];
}
eParticleRenderMode GetParticleRenderModeFromName(const std::string& name)
{
	for (uint i = 0; i < PARTICLE_RENDER_MODE_COUNT; i++)
	{
		if (g_particleRenderModeName[i] == name)
		{
			return eParticleRenderMode(i);
		}
	}
	return PARTICLE_RENDER_MODE_COUNT;
}
