#ifndef __TUM3D__INTEGRATION_PARAMS_GPU__
#define __TUM3D__INTEGRATION_PARAMS_GPU__


#include <global.h>

#include "TracingCommon.h"


struct IntegrationParamsGPU
{
	// Upload this instance to the GPU's global instance in constant memory.
	// Assumes that this instance is in pinned memory.
	// Does not sync on the upload, so don't overwrite any members without syncing first!
	void Upload(bool cpuTracing) const;


	float timeMax; // age for stream lines, spawntime + age for path lines

	float toleranceSquared;

	float deltaTimeMin;
	float deltaTimeMax;

	float brickSafeMarginWorld;
	float velocityMaxWorld; // abs max of any velocity component

	uint stepCountMax;

	float outputPosDiffSquared;
	float outputTimeDiff;

	float3 gridSpacing;

	//time in cell measures
	bool timeInCellEnabled;
	float cellChangeThreshold;
};


#endif
