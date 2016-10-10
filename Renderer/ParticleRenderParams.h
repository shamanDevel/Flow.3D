#ifndef __TUM3D__PARTICLERENDERPARAMS_H__
#define __TUM3D__PARTICLERENDERPARAMS_H__


#include <global.h>

#include <ConfigFile.h>
#include <Vec.h>

#include "LineRenderMode.h"


struct ParticleRenderParams
{
	ParticleRenderParams();

	void Reset();

	void ApplyConfig(const ConfigFile& config);
	void WriteConfig(ConfigFile& config) const;

	bool m_linesEnabled;

	eLineRenderMode m_lineRenderMode;
	float m_ribbonWidth;
	float m_tubeRadius;

	bool  m_tubeRadiusFromVelocity;
	float m_referenceVelocity;

	bool         m_colorByTime;
	tum3D::Vec4f m_color0;
	tum3D::Vec4f m_color1;

	bool  m_timeStripes;
	float m_timeStripeLength;

	bool operator==(const ParticleRenderParams& rhs) const;
	bool operator!=(const ParticleRenderParams& rhs) const;
};


#endif
