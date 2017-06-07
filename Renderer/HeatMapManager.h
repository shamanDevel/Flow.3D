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
	void DebugPrintParams();

	void SetVolumeAndReset(const TimeVolume& volume);
	void ProcessLines(std::shared_ptr<LineBuffers> pLineBuffer);

	//Renders the heat map.
	//It is assumed that the correct frame buffer is already set.
	void Render(tum3D::Mat4f viewProjMat, ProjectionParams projParams,
		const Range1D& range, ID3D11ShaderResourceView* depthTexture);
	bool IsRenderingEnabled() { return m_params.m_enableRendering; }

	//Clears all channels
	void ClearChannels();

private:
	void ReleaseRenderTextures();
	void CreateRenderTextures(ID3D11Device* pDevice);
	void CopyToRenderTexture(HeatMap::Channel_ptr channel, int slot);

private:
	bool m_isCreated;

	// settings
	HeatMapParams            m_params;
	bool                     m_seedTexChanged;

	// valid between create/release
	GPUResources*            m_pCompressShared;
	CompressVolumeResources* m_pCompressVolume;
	ID3D11Device*            m_pDevice;

	// only valid while tracing
	const TimeVolume*        m_pVolume;
	tum3D::Vec4f             m_boxMin;
	tum3D::Vec4f             m_boxMax;
	HeatMap*                 m_pHeatMap;
	int3                     m_resolution;
	float3                   m_worldToGrid;
	float3                   m_worldOffset;
	unsigned int*            m_seedTexCuda;
	size_t                   m_seedTexSize;

	// for rendering
	struct HeatMapTexture
	{
		cudaGraphicsResource* cudaResource;
		ID3D11Texture3D* dxTexture;
		ID3D11ShaderResourceView* dxSRV;
	} m_textures[2];
	float*                  m_cudaCopyBuffer;
	bool                    m_hasData;
	bool                    m_dataChanged;
	int                     m_maxData;

	// raytracing shader
	HeatMapRaytracerEffect*	 m_pShader;
};

#endif