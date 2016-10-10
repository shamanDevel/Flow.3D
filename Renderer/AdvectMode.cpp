#include "AdvectMode.h"

#include <cassert>


namespace
{
	std::string g_advectModeName[ADVECT_MODE_COUNT + 1] = {
		"Euler",
		"Heun",
		"RK3",
		"RK4",
		"BS3(2)",
		"RKF3(4)",
		"RKF4(5)",
		"RKF5(4)",
		"RK5(4)7M",
		"Unknown"
	};
}

std::string GetAdvectModeName(eAdvectMode advectMode)
{
	return g_advectModeName[min(advectMode, ADVECT_MODE_COUNT)];
}

eAdvectMode GetAdvectModeFromName(const std::string& name)
{
	for(uint i = 0; i < ADVECT_MODE_COUNT; i++)
	{
		if(g_advectModeName[i] == name)
		{
			return eAdvectMode(i);
		}
	}
	return ADVECT_MODE_COUNT;
}


bool IsAdvectModeAdaptive(eAdvectMode advectMode)
{
	switch(advectMode)
	{
		case ADVECT_EULER:
			return false;
		case ADVECT_HEUN:
			return false;
		case ADVECT_RK3:
			return false;
		case ADVECT_RK4:
			return false;
		case ADVECT_BS32:
			return true;
		case ADVECT_RKF34:
			return true;
		case ADVECT_RKF45:
			return true;
		case ADVECT_RKF54:
			return true;
		case ADVECT_RK547M:
			return true;
		default:
			assert(false);
			return false;
	}
}

bool IsAdvectModeDenseOutput(eAdvectMode advectMode)
{
	switch(advectMode)
	{
		case ADVECT_RK547M:
			return true;
		default:
			return false;
	}
}


uint GetAdvectModeEvaluationsPerAcceptedStep(eAdvectMode advectMode)
{
	switch(advectMode)
	{
		case ADVECT_EULER:
			return 1;
		case ADVECT_HEUN:
			return 2;
		case ADVECT_RK3:
			return 3;
		case ADVECT_RK4:
			return 4;
		case ADVECT_BS32:
			return 3;
		case ADVECT_RKF34:
			return 5;
		case ADVECT_RKF45:
			return 6;
		case ADVECT_RKF54:
			return 6;
		case ADVECT_RK547M:
			return 6;
		default:
			assert(false);
			return 0;
	}
}

uint GetAdvectModeEvaluationsPerRejectedStep(eAdvectMode advectMode)
{
	switch(advectMode)
	{
		case ADVECT_EULER:
			return 1;
		case ADVECT_HEUN:
			return 2;
		case ADVECT_RK3:
			return 3;
		case ADVECT_RK4:
			return 4;
		case ADVECT_BS32:
			return 4;
		case ADVECT_RKF34:
			return 5;
		case ADVECT_RKF45:
			return 6;
		case ADVECT_RKF54:
			return 6;
		case ADVECT_RK547M:
			return 7;
		default:
			assert(false);
			return 0;
	}
}
