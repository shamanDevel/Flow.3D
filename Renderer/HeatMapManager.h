#ifndef __TUM3D__HEATMAP_MANAGER_H__
#define __TUM3D__HEATMAP_MANAGER_H__

#include "global.h"

#include <cuda_runtime.h>
#include <cuda_d3d11_interop.h>
#include <D3D11.h>
#include <D3DX11Effect/d3dx11effect.h>
#include <AntTweakBar\AntTweakBar.h>

#include "ProjectionParams.h"
#include "Range.h"
#include "LineBuffers.h"
#include "VolumeInfoGPU.h"
#include "GPUResources.h"
#include "CompressVolume.h"
#include "TimeVolume.h"
#include "ViewParams.h"

#include "HeatMap.h"
#include "HeatMapParams.h"
#include "HeatMapRaytracerEffect.h"

class HeatMapManager
{
public:
	HeatMapManager();
	~HeatMapManager();

	bool Create(GPUResources* pCompressShared, CompressVolumeResources* pCompressVolume, ID3D11Device* pDevice);
	void Release();
	bool IsCreated() const { return m_isCreated; }

	void SetParams(const HeatMapParams& params);

	void SetVolumeAndReset(const TimeVolume& volume);
	void ProcessLines(std::shared_ptr<LineBuffers> pLineBuffer);

	//Renders the heat map.
	//It is assumed that the correct frame buffer is already set.
	void Render(const ViewParams& viewParams, const StereoParams& stereoParam,
		const D3D11_VIEWPORT& viewport, ID3D11ShaderResourceView* depthTexture);
	bool IsRenderingEnabled() { return m_params.m_enableRendering; }

private:
	void ClearChannels();
	void ReleaseRenderTextures();
	void CreateRenderTextures(ID3D11Device* pDevice);
	void CopyToRenderTexture(HeatMap::Channel_ptr channel, int slot);

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

	// for rendering
	struct HeatMapTexture
	{
		cudaGraphicsResource* cudaResource;
		ID3D11Texture3D* dxTexture;
		ID3D11ShaderResourceView* dxSRV;
	} m_textures[2];
	bool                    m_hasData;
	bool                    m_dataChanged;

	// settings
	HeatMapParams            m_params;

	// raytracing shader
	HeatMapRaytracerEffect*	 m_pShader;
};

#endif