#ifndef __TUM3D__HEATMAP_MANAGER_H__
#define __TUM3D__HEATMAP_MANAGER_H__

#include "global.h"

#include <cuda_runtime.h>
#include <cuda_d3d11_interop.h>
#include <D3D11.h>
#include <AntTweakBar\AntTweakBar.h>

#include "ProjectionParams.h"
#include "Range.h"
#include "LineBuffers.h"
#include "VolumeInfoGPU.h"
#include "GPUResources.h"
#include "CompressVolume.h"
#include "TimeVolume.h"

#include "HeatMap.h"

class HeatMapManager
{
public:
	HeatMapManager();
	~HeatMapManager();

	bool Create(GPUResources* pCompressShared, CompressVolumeResources* pCompressVolume, ID3D11Device* pDevice);
	void Release();
	bool IsCreated() const { return m_isCreated; }

	void InitAntTweakBar(TwBar* bar);

	void SetVolumeAndReset(const TimeVolume& volume);
	void ProcessLines(std::shared_ptr<LineBuffers> pLineBuffer);

private:
	void ClearChannels();

private:
	bool m_isCreated;

	// valid between create/release
	GPUResources*            m_pCompressShared;
	CompressVolumeResources* m_pCompressVolume;
	ID3D11Device*            m_pDevice;

	// only valid while tracing
	const TimeVolume*        m_pVolume;
	HeatMap*                 m_pHeatMap;
	int3                     m_resolution;
	float3                   m_worldToGrid;
	float3                   m_worldOffset;

	// settings
	bool m_enableRecording;
	bool m_enableRendering;
	bool m_autoReset;
};

#endif