#ifndef __TUM3D__PARTICLE_RENDER_MODE_H
#define __TUM3D__PARTICLE_RENDER_MODE_H

#include <global.h>

#include <string>

enum eParticleRenderMode
{
	PARTICLE_RENDER_ADDITIVE = 0,
	PARTICLE_RENDER_ORDER_INDEPENDENT,
	PARTICLE_RENDER_MODE_COUNT
};
std::string GetParticleRenderModeName(eParticleRenderMode renderMode);
eParticleRenderMode GetParticleRenderModeFromName(const std::string& name);

#endif