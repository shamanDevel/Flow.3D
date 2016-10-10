#include "BatchTraceParams.h"


#include <cstring> // for memcmp
#include <string>


BatchTraceParams::BatchTraceParams()
{
	Reset();
}

void BatchTraceParams::Reset()
{
	m_advectModes.clear();
	m_advectModes.insert(eAdvectMode(ADVECT_RK547M));

	m_filterModes.clear();
	//m_filterModes.insert(TEXTURE_FILTER_LINEAR);
	//m_filterModes.insert(TEXTURE_FILTER_CATROM);
	//m_filterModes.insert(TEXTURE_FILTER_LAGRANGE4);
	m_filterModes.insert(TEXTURE_FILTER_LAGRANGE6);
	//m_filterModes.insert(TEXTURE_FILTER_LAGRANGE8);

	m_deltaTMin         = 0.001f;
	m_deltaTMax         = 0.01f;

	m_errorToleranceMin = 0.01f;
	m_errorToleranceMax = 0.1f;

	m_qualityStepCount = 1;

	m_heuristicFactorMin = 0.0f;
	m_heuristicFactorMax = 10.0f;

	m_heuristicStepCount = 11;
	m_heuristicBPSeparate = true;
}


uint BatchTraceParams::GetTotalStepCount() const
{
	return uint(m_advectModes.size() * m_filterModes.size() * m_qualityStepCount * GetTotalHeuristicStepCount());
}


uint BatchTraceParams::GetAdvectStep(uint step) const
{
	return (step / GetTotalHeuristicStepCount() / m_qualityStepCount / (uint)m_filterModes.size()) % (uint)m_advectModes.size();
}

uint BatchTraceParams::GetFilterStep(uint step) const
{
	return (step / GetTotalHeuristicStepCount() / m_qualityStepCount) % (uint)m_filterModes.size();
}

uint BatchTraceParams::GetQualityStep(uint step) const
{
	return (step / GetTotalHeuristicStepCount()) % m_qualityStepCount;
}

uint BatchTraceParams::GetHeuristicBonusStep(uint step) const
{
	if(m_heuristicBPSeparate)
		step /= max(1u, m_heuristicStepCount);
	return step % max(1u, m_heuristicStepCount);
}

uint BatchTraceParams::GetHeuristicPenaltyStep(uint step) const
{
	return step % max(1u, m_heuristicStepCount);
}


uint BatchTraceParams::GetTotalHeuristicStepCount() const
{
	uint result = max(1u, m_heuristicStepCount);
	if(m_heuristicBPSeparate)
		result *= result;
	return result;
}


namespace
{
	float GetInterpolationFactor(uint step, uint stepCount, float base)
	{
		if(stepCount <= 1) return 0.0f;
		float t = float(step) / float(stepCount - 1); // linear in [0,1]
		float e = pow(base, t); // exponential in [1,base]
		return (e - 1.0f) / (base - 1.0f);
	}
}

float BatchTraceParams::GetDeltaT(uint qualityStep) const
{
	float base = m_deltaTMax / m_deltaTMin;
	return m_deltaTMin + GetInterpolationFactor(qualityStep, m_qualityStepCount, base) * (m_deltaTMax - m_deltaTMin);
}

float BatchTraceParams::GetErrorTolerance(uint qualityStep) const
{
	float base = m_errorToleranceMax / m_errorToleranceMin;
	return m_errorToleranceMin + GetInterpolationFactor(qualityStep, m_qualityStepCount, base) * (m_errorToleranceMax - m_errorToleranceMin);
}

float BatchTraceParams::GetHeuristicFactor(uint heuristicStep) const
{
	return m_heuristicFactorMin + (m_heuristicFactorMax - m_heuristicFactorMin) * float(heuristicStep) / float(max(m_heuristicStepCount, 1u) - 1);
}


void BatchTraceParams::ApplyToTraceParams(ParticleTraceParams& params, uint step) const
{
	uint heuristicPenaltyStep = GetHeuristicPenaltyStep(step);
	uint heuristicBonusStep = GetHeuristicBonusStep(step);
	uint qualityStep = GetQualityStep(step);
	uint filterStep = GetFilterStep(step);
	uint advectStep = GetAdvectStep(step);


	auto advectMode = m_advectModes.begin();
	std::advance(advectMode, advectStep);
	params.m_advectMode = *advectMode;

	auto filterMode = m_filterModes.begin();
	std::advance(filterMode, filterStep);
	params.m_filterMode = *filterMode;

	params.m_advectDeltaT = GetDeltaT(qualityStep);
	params.m_advectErrorTolerance = GetErrorTolerance(qualityStep);

	if(m_heuristicStepCount > 0)
	{
		params.m_heuristicBonusFactor = GetHeuristicFactor(heuristicBonusStep);
		params.m_heuristicPenaltyFactor = GetHeuristicFactor(heuristicPenaltyStep);
	}

	//printf("Applied batch trace params. Advect: %s  Filter: %s  QualityStep: %u\n", GetAdvectModeName(*advectMode).c_str(), GetTextureFilterModeName(*filterMode), qualityStep);
}


void BatchTraceParams::ApplyConfig(const ConfigFile& config)
{
	Reset();

	const std::vector<ConfigSection>& sections = config.GetSections();
	for(size_t s = 0; s < sections.size(); s++)
	{
		const ConfigSection& section = sections[s];
		if(section.GetName() == "BatchTraceParams")
		{
			// this is our section - parse entries
			for(size_t e = 0; e < section.GetEntries().size(); e++) {
				const ConfigEntry& entry = section.GetEntries()[e];

				std::string entryName = entry.GetName();

				if(entryName == "AdvectModes")
				{
					m_advectModes.clear();
					const std::vector<std::string>& values = entry.GetValues();
					for(uint i = 0; i < values.size(); i++)
					{
						eAdvectMode mode = GetAdvectModeFromName(values[i]);
						m_advectModes.insert(mode);
					}
				}
				else if(entryName == "FilterModes")
				{
					m_filterModes.clear();
					const std::vector<std::string>& values = entry.GetValues();
					for(uint i = 0; i < values.size(); i++)
					{
						eTextureFilterMode mode = GetTextureFilterModeFromName(values[i]);
						m_filterModes.insert(mode);
					}
				}
				else if(entryName == "DeltaTMin")
				{
					entry.GetAsFloat(m_deltaTMin);
				}
				else if(entryName == "DeltaTMax")
				{
					entry.GetAsFloat(m_deltaTMax);
				}
				else if(entryName == "ErrorToleranceMin")
				{
					entry.GetAsFloat(m_errorToleranceMin);
				}
				else if(entryName == "ErrorToleranceMax")
				{
					entry.GetAsFloat(m_errorToleranceMax);
				}
				else if(entryName == "QualityStepCount")
				{
					int val;
					entry.GetAsInt(val);
					m_qualityStepCount = uint(max(val, 1));
				}
				else if(entryName == "HeuristicFactorMin")
				{
					entry.GetAsFloat(m_heuristicFactorMin);
				}
				else if(entryName == "HeuristicFactorMax")
				{
					entry.GetAsFloat(m_heuristicFactorMax);
				}
				else if(entryName == "HeuristicStepCount")
				{
					int val;
					entry.GetAsInt(val);
					m_heuristicStepCount = uint(val);
				}
				else if(entryName == "HeuristicBPSeparate")
				{
					entry.GetAsBool(m_heuristicBPSeparate);
				}
				else
				{
					printf("WARNING: BatchTraceParams::ApplyConfig: unknown entry \"%s\" ignored\n", entryName.c_str());
				}
			}
		}
	}
}

void BatchTraceParams::WriteConfig(ConfigFile& config) const
{
	ConfigSection section("BatchTraceParams");

	std::vector<std::string> advectModes;
	for(auto it = m_advectModes.cbegin(); it != m_advectModes.cend(); it++)
		advectModes.push_back(GetAdvectModeName(*it));

	std::vector<std::string> filterModes;
	for(auto it = m_filterModes.cbegin(); it != m_filterModes.cend(); it++)
		filterModes.push_back(GetTextureFilterModeName(*it));

	section.AddEntry(ConfigEntry("AdvectModes", advectModes));
	section.AddEntry(ConfigEntry("FilterModes", filterModes));
	section.AddEntry(ConfigEntry("DeltaTMin", m_deltaTMin));
	section.AddEntry(ConfigEntry("DeltaTMax", m_deltaTMax));
	section.AddEntry(ConfigEntry("ErrorToleranceMin", m_errorToleranceMin));
	section.AddEntry(ConfigEntry("ErrorToleranceMax", m_errorToleranceMax));
	section.AddEntry(ConfigEntry("QualityStepCount", int(m_qualityStepCount)));
	section.AddEntry(ConfigEntry("HeuristicFactorMin", m_heuristicFactorMin));
	section.AddEntry(ConfigEntry("HeuristicFactorMax", m_heuristicFactorMax));
	section.AddEntry(ConfigEntry("HeuristicStepCount", int(m_heuristicStepCount)));
	section.AddEntry(ConfigEntry("HeuristicBPSeparate", m_heuristicBPSeparate));

	config.AddSection(section);
}


bool BatchTraceParams::operator==(const BatchTraceParams& rhs) const
{
	return memcmp(this, &rhs, sizeof(BatchTraceParams)) == 0;
}

bool BatchTraceParams::operator!=(const BatchTraceParams& rhs) const
{
	return !(*this == rhs);
}
