#include "FilterParams.h"

#include <cstring>


FilterParams::FilterParams()
{
	Reset();
}

void FilterParams::Reset()
{
	m_radius.clear();
}


void FilterParams::ApplyConfig(const ConfigFile& config)
{
	Reset();

	const std::vector<ConfigSection>& sections = config.GetSections();
	for(size_t s = 0; s < sections.size(); s++)
	{
		const ConfigSection& section = sections[s];
		if(section.GetName() == "FilterParams")
		{
			// this is our section - parse entries
			for(size_t e = 0; e < section.GetEntries().size(); e++)
			{
				const ConfigEntry& entry = section.GetEntries()[e];

				std::string entryName = entry.GetName();

				if(entryName == "Radius")
				{
					int radius;
					for(size_t i = 0; i < entry.GetValueCount(); i++)
					{
						entry.GetValueAsInt(radius, i);
						m_radius.push_back(radius);
					}
				}
				else
				{
					printf("WARNING: FilterParams::ApplyConfig: unknown entry \"%s\" ignored\n", entryName.c_str());
				}
			}
		}
	}
}

void FilterParams::WriteConfig(ConfigFile& config) const
{
	ConfigSection section("FilterParams");

	ConfigEntry radius("Radius");
	for(size_t i = 0; i < m_radius.size(); i++)
	{
		radius.AddValue(int(m_radius[i]));
	}
	section.AddEntry(radius);

	config.AddSection(section);
}


bool FilterParams::HasNonZeroRadius() const
{
	for(size_t i = 0; i < m_radius.size(); i++)
	{
		if(m_radius[i] != 0) return true;
	}
	return false;
}


bool FilterParams::operator==(const FilterParams& rhs) const
{
	return m_radius == rhs.m_radius;
}

bool FilterParams::operator!=(const FilterParams& rhs) const
{
	return !(*this == rhs);
}
