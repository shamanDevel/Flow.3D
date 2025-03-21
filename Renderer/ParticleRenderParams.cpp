#include "ParticleRenderParams.h"

#include <cstring> // for memcmp

#include <cudaUtil.h>
#include <cuda_d3d11_interop.h>

#include <Utils.h>
#include <WICTextureLoader.h>


using namespace tum3D;


ParticleRenderParams::ParticleRenderParams()
{
	Reset();
}

void ParticleRenderParams::Reset()
{
	m_linesEnabled = true;

	m_lineRenderMode = eLineRenderMode::LINE_RENDER_TUBE;
	m_ribbonWidth = 0.8f;
	m_tubeRadius = 0.1f;

	m_particleRenderMode = PARTICLE_RENDER_ALPHA;
	m_particleTransparency = 0.3f;
	m_particleSize = 0.5f;

	m_tubeRadiusFromVelocity = false;
	m_referenceVelocity = 1.5f;

	m_lineColorMode = eLineColorMode::LINE_ID;

	m_color0 = Vec4f(0.232f, 0.365f, 0.764f, 1.0f);
	m_color1 = Vec4f(0.755f, 0.226f, 0.226f, 1.0f);
	m_pColorTexture = NULL;
	m_measure = eMeasure::MEASURE_VELOCITY;
	m_measureScale = 1.0;

	m_timeStripes = false;
	m_timeStripeLength = 0.03f;

	m_pSliceTexture = NULL;
	m_showSlice = false;
	m_slicePosition = -0.041;
	m_sliceAlpha = 0.7f;

	m_sortParticles = true;

	m_ftleShowTexture = true;
	m_ftleTextureAlpha = 1.0f;
}


void ParticleRenderParams::LoadColorTexture(std::string filePath, ID3D11Device* device)
{
	// release old texture
	SAFE_RELEASE(m_pColorTexture);

	// create new texture
	std::wstring wfilename(filePath.begin(), filePath.end());
	ID3D11Resource* tmp = NULL;

	if (!FAILED(DirectX::CreateWICTextureFromFile(device, wfilename.c_str(), &tmp, &m_pColorTexture)))
		std::cout << "Color texture " << filePath << " loaded." << std::endl;
	else
		std::cerr << "Failed to load color texture." << std::endl;

	SAFE_RELEASE(tmp);
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
				else if(entryName == "ColorMode")
				{
					std::string val;
					entry.GetAsString(val);
					m_lineColorMode = GetLineColorModeFromName(val);
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
	section.AddEntry(ConfigEntry("ColorMode", GetLineColorModeName(m_lineColorMode)));
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
