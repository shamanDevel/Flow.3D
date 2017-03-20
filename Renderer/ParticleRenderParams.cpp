#include "ParticleRenderParams.h"

#include <cstring> // for memcmp
#include <string>

using namespace tum3D;


ParticleRenderParams::ParticleRenderParams()
{
	Reset();
}

void ParticleRenderParams::Reset()
{
	m_linesEnabled = true;

	m_lineRenderMode = LINE_RENDER_TUBE;
	m_ribbonWidth = 0.8f;
	m_tubeRadius = 0.4f;

	m_particleRenderMode = PARTICLE_RENDER_ADDITIVE;
	m_particleTransparency = 0.3f;
	m_particleSize = 0.5f;

	m_tubeRadiusFromVelocity = true;
	m_referenceVelocity = 1.5f;

	m_colorByTime = false;
	m_color0 = Vec4f(0.0f, 0.251f, 1.0f, 1.0f);
	m_color1 = Vec4f(1.0f, 0.0f, 0.0f, 1.0f);

	m_timeStripes = false;
	m_timeStripeLength = 0.03f;
}

void ParticleRenderParams::ApplyConfig(const ConfigFile& config)
{
	Reset();

	const std::vector<ConfigSection>& sections = config.GetSections();
	for(size_t s = 0; s < sections.size(); s++)
	{
		const ConfigSection& section = sections[s];
		if(section.GetName() == "ParticleRenderParams")
		{
			// this is our section - parse entries
			for(size_t e = 0; e < section.GetEntries().size(); e++) {
				const ConfigEntry& entry = section.GetEntries()[e];

				std::string entryName = entry.GetName();

				if(entryName == "LinesEnabled")
				{
					entry.GetAsBool(m_linesEnabled);
				}
				else if(entryName == "LineRenderMode")
				{
					std::string val;
					entry.GetAsString(val);
					m_lineRenderMode = GetLineRenderModeFromName(val);
				}
				else if(entryName == "RibbonWidth")
				{
					entry.GetAsFloat(m_ribbonWidth);
				}
				else if(entryName == "TubeRadius")
				{
					entry.GetAsFloat(m_tubeRadius);
				}
				else if(entryName == "TubeRadiusFromVelocity")
				{
					entry.GetAsBool(m_tubeRadiusFromVelocity);
				}
				else if(entryName == "ReferenceVelocity")
				{
					entry.GetAsFloat(m_referenceVelocity);
				}
				else if(entryName == "ColorByTime")
				{
					entry.GetAsBool(m_colorByTime);
				}
				else if(entryName == "Color0")
				{
					entry.GetAsVec4f(m_color0);
				}
				else if(entryName == "Color1")
				{
					entry.GetAsVec4f(m_color1);
				}
				else if(entryName == "TimeStripes")
				{
					entry.GetAsBool(m_timeStripes);
				}
				else if(entryName == "TimeStripeLength")
				{
					entry.GetAsFloat(m_timeStripeLength);
				}
				else
				{
					printf("WARNING: ParticleRenderParams::ApplyConfig: unknown entry \"%s\" ignored\n", entryName.c_str());
				}
			}
		}
	}
}

void ParticleRenderParams::WriteConfig(ConfigFile& config) const
{
	ConfigSection section("ParticleRenderParams");

	section.AddEntry(ConfigEntry("LinesEnabled", m_linesEnabled));
	section.AddEntry(ConfigEntry("LineRenderMode", GetLineRenderModeName(m_lineRenderMode)));
	section.AddEntry(ConfigEntry("RibbonWidth", m_ribbonWidth));
	section.AddEntry(ConfigEntry("TubeRadius", m_tubeRadius));
	section.AddEntry(ConfigEntry("TubeRadiusFromVelocity", m_tubeRadiusFromVelocity));
	section.AddEntry(ConfigEntry("ReferenceVelocity", m_referenceVelocity));
	section.AddEntry(ConfigEntry("ColorByTime", m_colorByTime));
	section.AddEntry(ConfigEntry("Color0", m_color0));
	section.AddEntry(ConfigEntry("Color1", m_color1));
	section.AddEntry(ConfigEntry("TimeStripes", m_timeStripes));
	section.AddEntry(ConfigEntry("TimeStripeLength", m_timeStripeLength));

	config.AddSection(section);
}

bool ParticleRenderParams::operator==(const ParticleRenderParams& rhs) const
{
	return memcmp(this, &rhs, sizeof(ParticleRenderParams)) == 0;
}

bool ParticleRenderParams::operator!=(const ParticleRenderParams& rhs) const
{
	return !(*this == rhs);
}
