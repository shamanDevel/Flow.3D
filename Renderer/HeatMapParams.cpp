#include "HeatMapParams.h"



HeatMapParams::HeatMapParams()
	: m_enableRecording(true)
	, m_enableRendering(true)
	, m_autoReset(false)
	, m_normalize(true)
	, m_stepSize(0.01)
	, m_densityScale(1)
	, m_tfAlphaScale(1)
	, m_tfRangeMin(0)
	, m_tfRangeMax(1)
	, m_pTransferFunction(NULL)
{
}


HeatMapParams::~HeatMapParams()
{
}

bool HeatMapParams::HasChangesForRetracing(const HeatMapParams& other, 
	const ParticleTraceParams & traceParams)
{
	if (m_enableRecording != other.m_enableRecording) return true;
	if ((m_autoReset != other.m_autoReset) 
		&& !LineModeIsIterative(traceParams.m_lineMode)) {
		return true;
	}

	return false;
}

bool HeatMapParams::HasChangesForRedrawing(const HeatMapParams& other)
{
	if (m_enableRendering != other.m_enableRendering) return true;
	if (m_normalize != other.m_normalize) return true;
	if (m_stepSize != other.m_stepSize) return true;
	if (m_densityScale != other.m_densityScale) return true;
	if (m_tfAlphaScale != other.m_tfAlphaScale) return true;
	if (m_tfRangeMin != other.m_tfRangeMin) return true;
	if (m_tfRangeMax != other.m_tfRangeMax) return true;
	if (m_pTransferFunction != other.m_pTransferFunction) return true;

	return false;
}
