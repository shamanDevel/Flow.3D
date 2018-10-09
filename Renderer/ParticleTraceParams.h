#ifndef __TUM3D__PARTICLETRACEPARAMS_H__
#define __TUM3D__PARTICLETRACEPARAMS_H__


#include <global.h>

#include <ConfigFile.h>
#include <Vec.h>
#include <vector>
#include <set>

#include "AdvectMode.h"
#include "LineMode.h"
#include "TextureFilterMode.h"

struct SeedTexture
{
	int m_width;
	int m_height;
	unsigned int* m_colors;
	//The picked color
	// 0: seed from seed box
	// >0: seed at positions where m_colors[pos] == m_picked
	std::set<unsigned int> m_picked;

	SeedTexture();

	bool operator==(const SeedTexture& rhs) const;
	bool operator!=(const SeedTexture& rhs) const;
};

struct ParticleTraceParams
{
	ParticleTraceParams();

	void Reset();

	void ApplyConfig(const ConfigFile& config);
	void WriteConfig(ConfigFile& config, bool skipBatchParams = false) const;

	tum3D::Vec3f m_seedBoxMin;
	tum3D::Vec3f m_seedBoxSize;

	void MoveSeedBox (const tum3D::Vec3f& translation);
	void ScaleSeedBox(const tum3D::Vec3f& scaling);

	eAdvectMode m_advectMode;
	bool        m_enableDenseOutput;

	eTextureFilterMode m_filterMode;

	eLineMode m_lineMode;

	// FTLE stuff
	bool m_ftleEnabled;
	tum3D::Vec3f m_ftleSeparationDistance;
	uint m_ftleResolution;
	float m_ftleSliceY;
	bool m_ftleInvertVelocity;


	static const char* GetSeedPatternName(int i);
	enum eSeedPattern
	{
		RANDOM = 0,
		REGULAR_GRID,
		FTLE,
		COUNT
	};
	eSeedPattern m_seedPattern;

	int  m_lineCount;
	int  m_lineLengthMax;
	float m_lineAgeMax;
	float m_minVelocity;

	float m_advectDeltaT; // for fixed integrators
	float m_advectErrorTolerance; // for adaptive integrators; relative to grid spacing

	float m_advectDeltaTMin; // for adaptive integrators
	float m_advectDeltaTMax;

	uint m_advectStepsPerRound;
	uint m_purgeTimeoutInRounds;

	float m_heuristicBonusFactor; // "s" in the paper
	float m_heuristicPenaltyFactor;
	enum eHeuristicFlags
	{
		HEURISTIC_USE_FLOW_GRAPH  = 1 << 0,
		HEURISTIC_SQRT_BASE_COUNT = 1 << 1,
		HEURISTIC_SQRT_BP_COUNT   = 1 << 2,
		HEURISTIC_SQRT_BP_PROB    = 1 << 3,
	};
	uint m_heuristicFlags;
	bool HeuristicUseFlowGraph()    const { return (m_heuristicFlags & HEURISTIC_USE_FLOW_GRAPH)  != 0; }
	bool HeuristicDoSqrtBaseCount() const { return (m_heuristicFlags & HEURISTIC_SQRT_BASE_COUNT) != 0; }
	bool HeuristicDoSqrtBPCount()   const { return (m_heuristicFlags & HEURISTIC_SQRT_BP_COUNT)   != 0; }
	bool HeuristicDoSqrtBPProb()    const { return (m_heuristicFlags & HEURISTIC_SQRT_BP_PROB)    != 0; }

	float m_outputPosDiff; // relative to grid spacing
	float m_outputTimeDiff;

	bool m_waitForDisk;
	bool m_enablePrefetching;

	bool m_upsampledVolumeHack;

	bool m_cpuTracing;

	//Particles per second and per seed
	float m_particlesPerSecond;

	//seed from texture
	SeedTexture m_seedTexture;

	//time that a particle must stay in a cell to change it's current cell
	float m_cellChangeThreshold;

	//Checks if the changes when comparing with 'old' are enough to completely retrace everything.
	//This is only important for particle modes. For line modes, this delegates to operator!=
	//But particles can deal with changes in e.g. the seed box without restarting
	bool hasChangesForRetracing(const ParticleTraceParams& old) const;

	bool operator==(const ParticleTraceParams& rhs) const;
	bool operator!=(const ParticleTraceParams& rhs) const;
};


#endif
