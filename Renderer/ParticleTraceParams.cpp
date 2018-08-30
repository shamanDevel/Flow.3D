#include "ParticleTraceParams.h"

#include <cstring> // for memcmp
#include <string>

using namespace tum3D;

SeedTexture::SeedTexture()
	: m_width(0), m_height(0), m_picked(), m_colors(NULL)
{

}

bool SeedTexture::operator==(const SeedTexture& rhs) const
{
	return (
		(m_width == rhs.m_width)
		&& (m_height == rhs.m_height)
		&& (m_colors == rhs.m_colors)
		&& (m_picked == rhs.m_picked)
		);
}

bool SeedTexture::operator!=(const SeedTexture& rhs) const
{
	return !(*this == rhs);
}


ParticleTraceParams::ParticleTraceParams()
{
	Reset();
}

void ParticleTraceParams::Reset()
{
	m_seedBoxMin    = Vec3f(-0.2f, -0.2f, -0.2f);
	m_seedBoxSize   = Vec3f( 0.4f,  0.4f,  0.4f);

	m_advectMode        = ADVECT_RK547M;
	m_enableDenseOutput = true;

	m_filterMode    = TEXTURE_FILTER_CATROM;

	m_lineMode = LINE_STREAM;

	m_ftleEnabled = false;
	m_ftleSeparationDistance = tum3D::Vec3f(1.0e-7);
	m_ftleResolution = 1024;

	m_seedPattern = eSeedPattern::RANDOM;

	m_lineCount     = 1024;
	m_lineLengthMax = 1024;
	m_lineAgeMax    = 512;
	m_minVelocity   = 0;

	m_advectDeltaT         = 0.005f;
	m_advectErrorTolerance = 0.01f;

	m_advectDeltaTMin      = 0.00001f;
	m_advectDeltaTMax      = 0.01f; // TODO make dependent on max velocity etc? in iso, this corresponds to <= ~5 voxels

	m_advectStepsPerRound  = 16;
	m_purgeTimeoutInRounds = 4;

	m_heuristicBonusFactor   = 2.5f;
	m_heuristicPenaltyFactor = 2.5f;
	m_heuristicFlags         = HEURISTIC_SQRT_BASE_COUNT | HEURISTIC_SQRT_BP_COUNT | HEURISTIC_SQRT_BP_PROB;

	m_outputPosDiff  = 1.0f;
	m_outputTimeDiff = 20.0f * m_advectDeltaT; //TODO specify in time steps

	m_waitForDisk = true;
	m_enablePrefetching = true;

	m_upsampledVolumeHack = false;

	m_cpuTracing = false;

	m_particlesPerSecond = 5;
	m_cellChangeThreshold = 0.01f;
}


void ParticleTraceParams::ApplyConfig(const ConfigFile& config)
{
	Reset();

	const std::vector<ConfigSection>& sections = config.GetSections();
	for(size_t s = 0; s < sections.size(); s++)
	{
		const ConfigSection& section = sections[s];
		if(section.GetName() == "ParticleTraceParams")
		{
			// this is our section - parse entries
			for (size_t e = 0; e < section.GetEntries().size(); e++) {
				const ConfigEntry& entry = section.GetEntries()[e];

				std::string entryName = entry.GetName();

				if (entryName == "SeedBoxMin")
				{
					entry.GetAsVec3f(m_seedBoxMin);
				}
				else if (entryName == "SeedBoxSize")
				{
					entry.GetAsVec3f(m_seedBoxSize);
				}
				else if (entryName == "AdvectMode")
				{
					std::string val;
					entry.GetAsString(val);
					m_advectMode = GetAdvectModeFromName(val);
				}
				else if (entryName == "EnableDenseOutput")
				{
					entry.GetAsBool(m_enableDenseOutput);
				}
				else if (entryName == "FilterMode")
				{
					std::string val;
					entry.GetAsString(val);
					m_filterMode = GetTextureFilterModeFromName(val);
				}
				else if (entryName == "LineMode")
				{
					std::string val;
					entry.GetAsString(val);
					m_lineMode = GetLineModeFromName(val);
				}
				else if (entryName == "LineCount")
				{
					int val;
					entry.GetAsInt(val);
					m_lineCount = uint(max(val, 0));
				}
				else if (entryName == "LineLengthMax")
				{
					int val;
					entry.GetAsInt(val);
					m_lineLengthMax = uint(max(val, 0));
				}
				else if (entryName == "LineAgeMax")
				{
					entry.GetAsFloat(m_lineAgeMax);
				}
				else if (entryName == "MinVelocity")
				{
					entry.GetAsFloat(m_minVelocity);
				}
				else if(entryName == "AdvectDeltaT" || entryName == "LineDeltaT")
				{
					entry.GetAsFloat(m_advectDeltaT);
				}
				else if(entryName == "AdvectErrorTolerance" || entryName == "ErrorTolerance")
				{
					entry.GetAsFloat(m_advectErrorTolerance);
				}
				else if(entryName == "AdvectDeltaTMin" )
				{
					entry.GetAsFloat(m_advectDeltaTMin);
				}
				else if(entryName == "AdvectDeltaTMax" )
				{
					entry.GetAsFloat(m_advectDeltaTMax);
				}
				else if(entryName == "AdvectStepsPerRound")
				{
					int val;
					entry.GetAsInt(val);
					m_advectStepsPerRound = uint(max(val, 0));
				}
				else if(entryName == "PurgeTimeoutInRounds")
				{
					int val;
					entry.GetAsInt(val);
					m_purgeTimeoutInRounds = uint(max(val, 0));
				}
				else if(entryName == "NeighborPrioOffset")
				{
					// ignore - not supported anymore
				}
				else if(entryName == "HeuristicBonusFactor" || entryName == "NeighborPrioFactor")
				{
					entry.GetAsFloat(m_heuristicBonusFactor);
				}
				else if(entryName == "HeuristicPenaltyFactor")
				{
					entry.GetAsFloat(m_heuristicPenaltyFactor);
				}
				//else if(entryName == "HeuristicUseFlowGraph")   legacy..
				else if(entryName == "HeuristicFlags")
				{
					int val;
					entry.GetAsInt(val);
					m_heuristicFlags = uint(val);
				}
				else if(entryName == "OutputPosDiff" || entryName == "LineOutputPosDiff")
				{
					entry.GetAsFloat(m_outputPosDiff);
				}
				else if(entryName == "OutputTimeDiff" || entryName == "LineOutputTimeDiff")
				{
					entry.GetAsFloat(m_outputTimeDiff);
				}
				else if(entryName == "WaitForDisk" )
				{
					entry.GetAsBool(m_waitForDisk);
				}
				else if(entryName == "EnablePrefetching" )
				{
					entry.GetAsBool(m_enablePrefetching);
				}
				else if(entryName == "UpsampledVolumeHack" )
				{
					entry.GetAsBool(m_upsampledVolumeHack);
				}
				else
				{
					printf("WARNING: ParticleTraceParams::ApplyConfig: unknown entry \"%s\" ignored\n", entryName.c_str());
				}
			}
		}
	}
}

void ParticleTraceParams::WriteConfig(ConfigFile& config, bool skipBatchParams) const
{
	ConfigSection section("ParticleTraceParams");

	section.AddEntry(ConfigEntry("SeedBoxMin", m_seedBoxMin));
	section.AddEntry(ConfigEntry("SeedBoxSize", m_seedBoxSize));
	if(!skipBatchParams) section.AddEntry(ConfigEntry("AdvectMode", GetAdvectModeName(m_advectMode)));
	section.AddEntry(ConfigEntry("EnableDenseOutput", m_enableDenseOutput));
	if(!skipBatchParams) section.AddEntry(ConfigEntry("FilterMode", GetTextureFilterModeName(m_filterMode)));
	if(!skipBatchParams) section.AddEntry(ConfigEntry("LineMode", GetLineModeName(m_lineMode)));
	section.AddEntry(ConfigEntry("LineCount", int(m_lineCount)));
	section.AddEntry(ConfigEntry("LineLengthMax", int(m_lineLengthMax)));
	section.AddEntry(ConfigEntry("LineAgeMax", m_lineAgeMax)); 
	section.AddEntry(ConfigEntry("MinVelocity", m_minVelocity));
	if(!skipBatchParams) section.AddEntry(ConfigEntry("AdvectDeltaT", m_advectDeltaT));
	if(!skipBatchParams) section.AddEntry(ConfigEntry("AdvectErrorTolerance", m_advectErrorTolerance));
	section.AddEntry(ConfigEntry("AdvectDeltaTMin", m_advectDeltaTMin));
	section.AddEntry(ConfigEntry("AdvectDeltaTMax", m_advectDeltaTMax));
	section.AddEntry(ConfigEntry("AdvectStepsPerRound", int(m_advectStepsPerRound)));
	section.AddEntry(ConfigEntry("PurgeTimeoutInRounds", int(m_purgeTimeoutInRounds)));
	section.AddEntry(ConfigEntry("HeuristicBonusFactor", m_heuristicBonusFactor));
	section.AddEntry(ConfigEntry("HeuristicPenaltyFactor", m_heuristicPenaltyFactor));
	section.AddEntry(ConfigEntry("HeuristicFlags", int(m_heuristicFlags)));
	section.AddEntry(ConfigEntry("OutputPosDiff", m_outputPosDiff));
	section.AddEntry(ConfigEntry("OutputTimeDiff", m_outputTimeDiff));
	section.AddEntry(ConfigEntry("WaitForDisk", m_waitForDisk));
	section.AddEntry(ConfigEntry("EnablePrefetching", m_enablePrefetching));
	section.AddEntry(ConfigEntry("UpsampledVolumeHack", m_upsampledVolumeHack));

	config.AddSection(section);
}


void ParticleTraceParams::MoveSeedBox(const Vec3f& translation)
{
	m_seedBoxMin += translation;
}

void ParticleTraceParams::ScaleSeedBox(const Vec3f& scaling)
{
	Vec3f sizeOld = m_seedBoxSize;
	Vec3f sizeNew = sizeOld * scaling;
	Vec3f center = m_seedBoxMin + 0.5f * sizeOld;
	m_seedBoxSize = sizeNew;
	m_seedBoxMin = (center - 0.5f * sizeNew);
}


bool ParticleTraceParams::hasChangesForRetracing(const ParticleTraceParams & old) const
{
	if (m_lineMode != old.m_lineMode) return true;
	if (LineModeIsIterative(m_lineMode)) {
		//particle mode, compare only a subset
		//skip seed box, advection mode, filter mode
		if (m_lineCount != old.m_lineCount
			|| m_lineLengthMax != old.m_lineLengthMax
			|| m_lineAgeMax != old.m_lineAgeMax) {
			//These require different buffer sizes
			return true;
		}
		//skip timesteps
		if (m_heuristicBonusFactor != old.m_heuristicBonusFactor
			|| m_heuristicPenaltyFactor != old.m_heuristicPenaltyFactor
			|| m_heuristicFlags != old.m_heuristicFlags
			|| m_outputPosDiff != old.m_outputPosDiff
			|| m_outputTimeDiff != old.m_outputTimeDiff
			|| m_waitForDisk != old.m_waitForDisk
			|| m_enablePrefetching != old.m_enablePrefetching
			|| m_upsampledVolumeHack != old.m_upsampledVolumeHack) {
			//these influence the loading. I don't know how imporant they are, so be on the safe side
			return true;
		}
		if (m_cpuTracing != old.m_cpuTracing) {
			return true; //of course
		}
		//particles per second skipped
		if (m_seedTexture.m_colors != old.m_seedTexture.m_colors
			|| m_seedTexture.m_height != old.m_seedTexture.m_height
			|| m_seedTexture.m_width != old.m_seedTexture.m_width) {
			return true;
		}
		//if in stream mode, retrace on seed box changes
		if (m_lineMode == eLineMode::LINE_PARTICLE_STREAM) {
			if (m_seedBoxMin != old.m_seedBoxMin
				|| m_seedBoxSize != old.m_seedBoxSize
				|| m_seedTexture != old.m_seedTexture) {
				return true;
			}
		}
		//no relevant change
		return false;
	}
	else {
		//full comparison
		return (*this != old);
	}
}

bool ParticleTraceParams::operator==(const ParticleTraceParams& rhs) const
{
	//return memcmp(this, &rhs, sizeof(ParticleTraceParams)) == 0;
	if (m_seedBoxMin != rhs.m_seedBoxMin) return false;
	if (m_seedBoxSize != rhs.m_seedBoxSize) return false;
	if (m_advectMode != rhs.m_advectMode) return false;
	if (m_enableDenseOutput != rhs.m_enableDenseOutput) return false;
	if (m_filterMode != rhs.m_filterMode) return false;
	if (m_lineMode != rhs.m_lineMode) return false;
	if (m_lineCount != rhs.m_lineCount) return false;
	if (m_lineLengthMax != rhs.m_lineLengthMax) return false;
	if (m_lineAgeMax != rhs.m_lineAgeMax) return false;
	if (m_minVelocity != rhs.m_minVelocity) return false;
	if (m_advectDeltaT != rhs.m_advectDeltaT) return false;
	if (m_advectErrorTolerance != rhs.m_advectErrorTolerance) return false;
	if (m_advectDeltaTMin != rhs.m_advectDeltaTMin) return false;
	if (m_advectDeltaTMax != rhs.m_advectDeltaTMax) return false;
	if (m_advectStepsPerRound != rhs.m_advectStepsPerRound) return false;
	if (m_purgeTimeoutInRounds != rhs.m_purgeTimeoutInRounds) return false;
	if (m_heuristicBonusFactor != rhs.m_heuristicBonusFactor) return false;
	if (m_heuristicPenaltyFactor != rhs.m_heuristicPenaltyFactor) return false;
	if (m_heuristicFlags != rhs.m_heuristicFlags) return false;
	if (m_outputPosDiff != rhs.m_outputPosDiff) return false;
	if (m_outputTimeDiff != rhs.m_outputTimeDiff) return false;
	if (m_waitForDisk != rhs.m_waitForDisk) return false;
	if (m_enablePrefetching != rhs.m_enablePrefetching) return false;
	if (m_upsampledVolumeHack != rhs.m_upsampledVolumeHack) return false;
	if (m_cpuTracing != rhs.m_cpuTracing) return false;
	if (m_particlesPerSecond != rhs.m_particlesPerSecond) return false;
	if (m_seedTexture != rhs.m_seedTexture) return false;
	if (m_cellChangeThreshold != rhs.m_cellChangeThreshold) return false;
	return true;
}

bool ParticleTraceParams::operator!=(const ParticleTraceParams& rhs) const
{
	return !(*this == rhs);
}
