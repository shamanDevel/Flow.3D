#pragma once

#include <TimeVolume.h>
#include <ParticleTraceParams.h>
#include <TracingManager.h>

struct FlowVisToolVolumeData
{
	TimeVolume*				m_volume;
	ParticleTraceParams		m_traceParams;
	TracingManager			m_tracingManager;

	bool m_retrace;
	bool m_isTracing;

	clock_t		m_lastTraceParamsUpdate;
	TimerCPU	m_timerTracing;

	FlowVisToolVolumeData()
	{
		m_volume = nullptr;
		m_retrace = false;
		m_isTracing = false;
		m_lastTraceParamsUpdate = 0;
	}

	void SetSeedingBoxToDomainSize()
	{
		m_traceParams.m_seedBoxMin = -m_volume->GetVolumeHalfSizeWorld();
		m_traceParams.m_seedBoxSize = 2 * m_volume->GetVolumeHalfSizeWorld();
	}

	bool CreateResources(ID3D11Device* pDevice)
	{
		if (!m_tracingManager.Create(pDevice))
			return false;
		return true;
	}

	void ReleaseResources()
	{
		m_tracingManager.ClearResult();
		m_tracingManager.Release();
	}

private:
	// disable copy and assignment
	FlowVisToolVolumeData(const FlowVisToolVolumeData&);
	FlowVisToolVolumeData& operator=(const FlowVisToolVolumeData&);
};