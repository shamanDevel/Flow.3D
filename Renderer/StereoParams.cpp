#include "StereoParams.h"


StereoParams::StereoParams()
{
	Reset();
}

void StereoParams::Reset()
{
	m_stereoEnabled = false;
	m_eyeDistance = 0.05f;
}


void StereoParams::ApplyConfig(const ConfigFile& config)
{
	Reset();

	const std::vector<ConfigSection>& sections = config.GetSections();
	for(size_t s = 0; s < sections.size(); s++)
	{
		const ConfigSection& section = sections[s];
		if(section.GetName() == "StereoParams")
		{
			// this is our section - parse entries
			for(size_t e = 0; e < section.GetEntries().size(); e++) {
				const ConfigEntry& entry = section.GetEntries()[e];

				std::string entryName = entry.GetName();

				if(entryName == "StereoEnabled")
				{
					entry.GetAsBool(m_stereoEnabled);
				}
				else if(entryName == "EyeDistance")
				{
					entry.GetAsFloat(m_eyeDistance);
				}
				else
				{
					printf("WARNING: StereoParams::ApplyConfig: unknown entry \"%s\" ignored\n", entryName.c_str());
				}
			}
		}
	}
}

void StereoParams::WriteConfig(ConfigFile& config) const
{
	ConfigSection section("StereoParams");

	section.AddEntry(ConfigEntry("StereoEnabled", m_stereoEnabled));
	section.AddEntry(ConfigEntry("EyeDistance", m_eyeDistance));

	config.AddSection(section);
}


bool StereoParams::operator==(const StereoParams& rhs) const
{
	return memcmp(this, &rhs, sizeof(StereoParams)) == 0;
}

bool StereoParams::operator!=(const StereoParams& rhs) const
{
	return !(*this == rhs);
}
