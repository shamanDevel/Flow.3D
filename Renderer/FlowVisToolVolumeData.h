#pragma once

#include <TimeVolume.h>
#include <TracingManager.h>
#include <ParticleTraceParams.h>
#include <ParticleRenderParams.h>

struct FlowVisToolVolumeData
{
	TimeVolume*				m_volume;
	TracingManager			m_tracingManager;
	ParticleTraceParams		m_traceParams;
	ParticleRenderParams	m_renderParams;
	
	bool m_tracingPaused;
	bool m_retrace;
	bool m_isTracing;

	bool	m_renderDomainBox;
	bool	m_renderClipBox;
	bool	m_renderSeedBox;
	bool	m_renderBrickBoxes;

	clock_t		m_lastTraceParamsUpdate;
	TimerCPU	m_timerTracing;

	FlowVisToolVolumeData()
	{
		m_volume = nullptr;
		m_tracingPaused = false;
		m_retrace = false;
		m_isTracing = false;
		m_lastTraceParamsUpdate = 0;

		m_renderDomainBox = true;
		m_renderClipBox = true;
		m_renderSeedBox = true;
		m_renderBrickBoxes = false;
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

		SAFE_RELEASE(m_renderParams.m_pSliceTexture);
		SAFE_RELEASE(m_renderParams.m_pColorTexture);
	}

private:
	// disable copy and assignment
	FlowVisToolVolumeData(const FlowVisToolVolumeData&);
	FlowVisToolVolumeData& operator=(const FlowVisToolVolumeData&);
};