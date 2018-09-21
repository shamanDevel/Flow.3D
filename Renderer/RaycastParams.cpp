#include "RaycastParams.h"

#include <cstring> // for memcmp

using namespace tum3D;


RaycastParams::RaycastParams()
{
	Reset();
}

void RaycastParams::Reset()
{
	m_raycastingEnabled			= false;

	m_measure1					= MEASURE_VELOCITY;
	m_measure2					= MEASURE_VELOCITY;
	m_measureScale1				= 1.0f;
	m_measureScale2				= 1.0f;
	m_measureComputeMode		= MEASURE_COMPUTE_ONTHEFLY;

	m_colorMode					= COLOR_MODE_VORTICITY_ALIGNMENT;
	m_raycastMode				= RAYCAST_MODE_ISO;
	m_textureFilterMode			= TEXTURE_FILTER_LINEAR;
	m_sampleRate				= 1.0f;

	m_filterOffset				= 0;

	m_density					= 10.0f;
	m_transferFunctionRangeMin	= 0.0f;
	m_transferFunctionRangeMax	= 1.0f;
	m_alphaScale				= 1.0f;

	m_isoValue1					= 1.0f;
	m_isoValue2					= 1.0f;
	m_isoValue3					= 1.0f;
	m_isoColor1					= Vec4f(0.7f, 0.7f, 0.7f, 1.0f);
	m_isoColor2					= Vec4f(1.0f, 100.0f/255.0f, 24.0f/255.0f, 1.0f);
	m_isoColor3					= Vec4f(100.0f/255.0f, 1.0f, 24.0f/255.0f, 1.0f);

	m_clipBoxMin				= Vec3f(-1.0f, -1.0f, -1.0f);
	m_clipBoxMax				= Vec3f( 1.0f,  1.0f,  1.0f);
}

void RaycastParams::ApplyConfig(const ConfigFile& config)
{
	Reset();

	const std::vector<ConfigSection>& sections = config.GetSections();
	for(size_t s = 0; s < sections.size(); s++)
	{
		const ConfigSection& section = sections[s];
		if(section.GetName() == "RaycastParams")
		{
			// this is our section - parse entries
			for(size_t e = 0; e < section.GetEntries().size(); e++) {
				const ConfigEntry& entry = section.GetEntries()[e];

				std::string entryName = entry.GetName();

				if(entryName == "RaycastingEnabled")
				{
					entry.GetAsBool(m_raycastingEnabled);
				}
				else if(entryName == "Measure1" || entryName == "Measure")
				{
					std::string val;
					entry.GetAsString(val);
					m_measure1 = GetMeasureFromName(val);
				}
				else if(entryName == "Measure2")
				{
					std::string val;
					entry.GetAsString(val);
					m_measure2 = GetMeasureFromName(val);
				}
				else if(entryName == "MeasureScale1" || entryName == "MeasureScale")
				{
					entry.GetAsFloat(m_measureScale1);
				}
				else if(entryName == "MeasureScale2")
				{
					entry.GetAsFloat(m_measureScale2);
				}
				else if(entryName == "MeasureComputeMode")
				{
					std::string val;
					entry.GetAsString(val);
					m_measureComputeMode = GetMeasureComputeModeFromName(val);
				}
				else if(entryName == "RaycastMode")
				{
					std::string val;
					entry.GetAsString(val);
					m_raycastMode = GetRaycastModeFromName(val);
				}
				else if(entryName == "ColorMode")
				{
					std::string val;
					entry.GetAsString(val);
					m_colorMode = GetColorModeFromName(val);
				}
				else if(entryName == "TextureFilterMode" || entryName == "VolumeFilterMode")
				{
					std::string val;
					entry.GetAsString(val);
					m_textureFilterMode = GetTextureFilterModeFromName(val);
				}
				else if(entryName == "SampleRate")
				{
					entry.GetAsFloat(m_sampleRate);
				}
				else if(entryName == "FilterOffset")
				{
					int val = 0;
					entry.GetAsInt(val);
					m_filterOffset = uint(max(val, 0));
				}
				else if(entryName == "Density")
				{
					entry.GetAsFloat(m_density);
				}
				else if(entryName == "AlphaScale")
				{
					entry.GetAsFloat(m_alphaScale);
				}
				else if(entryName == "TransferFunctionRangeMin")
				{
					entry.GetAsFloat(m_transferFunctionRangeMin);
				}
				else if(entryName == "TransferFunctionRangeMax")
				{
					entry.GetAsFloat(m_transferFunctionRangeMax);
				}
				else if(entryName == "IsoValue1")
				{
					entry.GetAsFloat(m_isoValue1);
				}
				else if(entryName == "IsoValue2")
				{
					entry.GetAsFloat(m_isoValue2);
				}
				else if(entryName == "IsoValue3")
				{
					entry.GetAsFloat(m_isoValue3);
				}
				else if(entryName == "IsoColor1")
				{
					entry.GetAsVec4f(m_isoColor1);
				}
				else if(entryName == "IsoColor2")
				{
					entry.GetAsVec4f(m_isoColor2);
				}
				else if(entryName == "IsoColor3")
				{
					entry.GetAsVec4f(m_isoColor3);
				}
				else if(entryName == "ClipBoxMin")
				{
					entry.GetAsVec3f(m_clipBoxMin);
				}
				else if(entryName == "ClipBoxMax")
				{
					entry.GetAsVec3f(m_clipBoxMax);
				}
				else
				{
					printf("WARNING: RaycastParams::ApplyConfig: unknown entry \"%s\" ignored\n", entryName.c_str());
				}
			}
		}
	}
}

void RaycastParams::WriteConfig(ConfigFile& config) const
{
	ConfigSection section("RaycastParams");

	section.AddEntry(ConfigEntry("RaycastingEnabled", m_raycastingEnabled));
	section.AddEntry(ConfigEntry("Measure1", GetMeasureName(m_measure1)));
	section.AddEntry(ConfigEntry("Measure2", GetMeasureName(m_measure2)));
	section.AddEntry(ConfigEntry("MeasureScale1", m_measureScale1));
	section.AddEntry(ConfigEntry("MeasureScale2", m_measureScale2));
	section.AddEntry(ConfigEntry("MeasureComputeMode", GetMeasureComputeModeName(m_measureComputeMode)));
	section.AddEntry(ConfigEntry("RaycastMode", GetRaycastModeName(m_raycastMode)));
	section.AddEntry(ConfigEntry("ColorMode", GetColorModeName(m_colorMode)));
	section.AddEntry(ConfigEntry("TextureFilterMode", GetTextureFilterModeName(m_textureFilterMode)));
	section.AddEntry(ConfigEntry("SampleRate", m_sampleRate));
	section.AddEntry(ConfigEntry("FilterOffset", int(m_filterOffset)));
	section.AddEntry(ConfigEntry("Density", m_density));
	section.AddEntry(ConfigEntry("AlphaScale", m_alphaScale));
	section.AddEntry(ConfigEntry("TransferFunctionRangeMin", m_transferFunctionRangeMin));
	section.AddEntry(ConfigEntry("TransferFunctionRangeMax", m_transferFunctionRangeMax));
	section.AddEntry(ConfigEntry("IsoValue1", m_isoValue1));
	section.AddEntry(ConfigEntry("IsoValue2", m_isoValue2));
	section.AddEntry(ConfigEntry("IsoValue3", m_isoValue3));
	section.AddEntry(ConfigEntry("IsoColor1", m_isoColor1));
	section.AddEntry(ConfigEntry("IsoColor2", m_isoColor2));
	section.AddEntry(ConfigEntry("IsoColor3", m_isoColor3));
	section.AddEntry(ConfigEntry("ClipBoxMin", m_clipBoxMin));
	section.AddEntry(ConfigEntry("ClipBoxMax", m_clipBoxMax));

	config.AddSection(section);
}

bool RaycastParams::operator==(const RaycastParams& rhs) const
{
	return memcmp(this, &rhs, sizeof(RaycastParams)) == 0;
}

bool RaycastParams::operator!=(const RaycastParams& rhs) const
{
	return !(*this == rhs);
}
