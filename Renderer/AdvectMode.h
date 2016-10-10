#ifndef __TUM3D__ADVECT_MODE_H__
#define __TUM3D__ADVECT_MODE_H__


#include <global.h>

#include <string>


enum eAdvectMode
{
	// non-adaptive
	ADVECT_EULER = 0,
	ADVECT_HEUN,
	ADVECT_RK3,
	ADVECT_RK4,
	// adaptive
	ADVECT_BS32, // Bogacki-Shampine
	ADVECT_RKF34,
	ADVECT_RKF45,
	ADVECT_RKF54,
	ADVECT_RK547M, // aka Dormand-Prince DoPri5

	ADVECT_MODE_COUNT
};
std::string GetAdvectModeName(eAdvectMode advectMode);
eAdvectMode GetAdvectModeFromName(const std::string& name);

bool IsAdvectModeAdaptive(eAdvectMode advectMode);
bool IsAdvectModeDenseOutput(eAdvectMode advectMode);

uint GetAdvectModeEvaluationsPerAcceptedStep(eAdvectMode advectMode);
uint GetAdvectModeEvaluationsPerRejectedStep(eAdvectMode advectMode);


#endif
