#include "Measure.h"


static const std::string g_measureNames[MEASURE_COUNT + 1] =
{
	"Velocity",
	"Velocity.Z",
	"Temperature",
	"Vorticity",
	"Lambda2",
	"QHunt",
	"Delta Chong",
	"Enstrophy Production",
	"Strain Production",
	"Square Rotation",
	"Square Rate Of Strain",
	"Trace JJT",
	"Preferential Vorticity Alignment",
	"Heat Current",
	"Heat Current.X",
	"Heat Current.Y",
	"Heat Current.Z",
	"Unknown"
};

std::string GetMeasureName(eMeasure mode)
{
	return g_measureNames[min(mode, MEASURE_COUNT)];
}

eMeasure GetMeasureFromName(const std::string& name)
{
	for(uint i = 0; i < MEASURE_COUNT; i++)
	{
		if(g_measureNames[i] == name)
		{
			return eMeasure(i);
		}
	}
	return MEASURE_COUNT;
}

float GetDefaultMeasureScale(eMeasure e)
{
	switch(e)
	{
		case MEASURE_VORTICITY:					return      0.02f;
		case MEASURE_LAMBDA2:					return     -0.001f;
		case MEASURE_QHUNT:						return      0.002f;
		case MEASURE_DELTACHONG:				return      0.000001f;
		case MEASURE_ENSTROPHY_PRODUCTION:		return      0.00005f;
		case MEASURE_STRAIN_PRODUCTION:			return      0.0005f;
		case MEASURE_SQUARE_ROTATION:			return      0.0005f;
		case MEASURE_SQUARE_RATE_OF_STRAIN:		return      0.0005f;
		case MEASURE_TRACE_JJT:					return      0.001f;
		case MEASURE_PVA:						return      0.05f;
		default :								return      1.0f;
	}
}

float GetDefaultMeasureQuantStep(eMeasure e)
{
	// these values are chosen for the iso turbulence data set (first time step)
	float quantStep = 1.0f / 256.0f;
	switch(e)
	{
		case MEASURE_QHUNT:						quantStep = 1.0f / 128.0f; break;
		case MEASURE_DELTACHONG:				quantStep = 1.0f /   0.01f; break; //FIXME this is way too coarse to be useful, but finer doesn't work
		case MEASURE_ENSTROPHY_PRODUCTION:		quantStep = 1.0f /  32.0f; break;
		case MEASURE_STRAIN_PRODUCTION:			quantStep = 1.0f /  20.0f; break;
		case MEASURE_SQUARE_ROTATION:			quantStep = 1.0f / 256.0f; break;
		case MEASURE_SQUARE_RATE_OF_STRAIN:		quantStep = 1.0f / 256.0f; break;
		case MEASURE_TRACE_JJT:					quantStep = 1.0f /  64.0f; break;
		case MEASURE_PVA:						quantStep = 1.0f / 384.0f; break;
	}
	// HACK: now the measure volumes are not pre-scaled anymore, so "remove" the scaling here
	return quantStep / GetDefaultMeasureScale(e);
}


eMeasureSource GetMeasureSource(eMeasure mode)
{
	switch(mode)
	{
		case MEASURE_VELOCITY:
		case MEASURE_VELOCITY_Z:
		case MEASURE_TEMPERATURE:
			return MEASURE_SOURCE_RAW;

		case MEASURE_HEAT_CURRENT:
		case MEASURE_HEAT_CURRENT_X:
		case MEASURE_HEAT_CURRENT_Y:
		case MEASURE_HEAT_CURRENT_Z:
			return MEASURE_SOURCE_HEAT_CURRENT;

		default:
			return MEASURE_SOURCE_JACOBIAN;
	}
}



static const std::string g_measureComputeModeNames[MEASURE_COMPUTE_COUNT + 1] =
{
	"On-the-fly",
	"Precomp (Discard after Rendering)",
	"Precomp (Store on GPU)",
	"Precomp (Compress)",
	"Unknown"
};

std::string GetMeasureComputeModeName(eMeasureComputeMode mode)
{
	return g_measureComputeModeNames[min(mode, MEASURE_COMPUTE_COUNT)];
}

eMeasureComputeMode GetMeasureComputeModeFromName(const std::string& name)
{
	for(uint i = 0; i < MEASURE_COMPUTE_COUNT; i++)
	{
		if(g_measureComputeModeNames[i] == name)
		{
			return eMeasureComputeMode(i);
		}
	}
	return MEASURE_COMPUTE_COUNT;
}
