#ifndef __TUM3D__BATCHTRACEPARAMS_H__
#define __TUM3D__BATCHTRACEPARAMS_H__


#include <global.h>

#include <set>

#include <ConfigFile.h>

#include "AdvectMode.h"
#include "ParticleTraceParams.h"
#include "TextureFilterMode.h"


struct BatchTraceParams
{
	BatchTraceParams();

	void Reset();

	uint GetTotalStepCount() const;

	uint GetAdvectStep(uint step) const;
	uint GetFilterStep(uint step) const;
	uint GetQualityStep(uint step) const;
	uint GetHeuristicBonusStep(uint step) const;
	uint GetHeuristicPenaltyStep(uint step) const;

	uint GetTotalHeuristicStepCount() const;

	float GetDeltaT(uint qualityStep) const;
	float GetErrorTolerance(uint qualityStep) const;
	float GetHeuristicFactor(uint heuristicStep) const;

	void ApplyToTraceParams(ParticleTraceParams& params, uint step) const;

	void ApplyConfig(const ConfigFile& config);
	void WriteConfig(ConfigFile& config) const;


	std::set<eAdvectMode>        m_advectModes;
	std::set<eTextureFilterMode> m_filterModes;

	// for non-adaptive integrators
	float m_deltaTMin;
	float m_deltaTMax;
	// for adaptive integrators
	float m_errorToleranceMin;
	float m_errorToleranceMax;

	uint m_qualityStepCount;

	float m_heuristicFactorMin;
	float m_heuristicFactorMax;

	uint m_heuristicStepCount;
	bool m_heuristicBPSeparate;


	bool operator==(const BatchTraceParams& rhs) const;
	bool operator!=(const BatchTraceParams& rhs) const;
};


#endif
