#ifndef __TUM3D__RENDERINGMANAGER_H__
#define __TUM3D__RENDERINGMANAGER_H__


#include <global.h>

#include <cstring>
#include <vector>

#include <cuda_runtime.h>
#include <cuda_d3d11_interop.h>
#include <D3D11.h>

#include <TimerCPU.h>
#include <MultiTimerGPU.h>
#include <Vec.h>

#include "TracingCommon.h"

#include "BallBuffers.h"
#include "Box.h"
#include "BrickSlot.h"
#include "FilteredVolume.h"
#include "LineBuffers.h"
#include "LineEffect.h"
#include "ParticleRenderParams.h"
#include "ParticleTraceParams.h"
#include "ProjectionParams.h"
#include "Range.h"
#include "Raycaster.h"
#include "RaycastParams.h"
#include "StereoParams.h"
#include "TimeVolume.h"
#include "ViewParams.h"
#include "HeatMapManager.h"

#include "GPUResources.h"
#include "CompressVolume.h"

#include "ScreenEffect.h"
#include "QuadEffect.h"

#include <FlowVisToolVolumeData.h>
#include <FlowVisToolUtils.h>
#include <RenderingParams.h>

//#define WriteVolumeToFileStuff

class RenderingManager
{
public:
	D3D11CudaTexture	m_ftleTexture;
	float				m_ftleScale = 0.02f;

	float	m_DomainBoxThickness = 0.0008f;
	bool	m_renderDomainBox;
	bool	m_renderClipBox;
	bool	m_renderSeedBox;
	bool	m_renderBrickBoxes;

	RenderingManager();
	~RenderingManager();

	// pDevice may be null!
	bool Create(ID3D11Device* pDevice);
	void Release();
	bool IsCreated() const { return m_isCreated; }

	void SetProjectionParams(const ProjectionParams& params, const Range1D& range);

	enum eRenderState
	{
		STATE_RENDERING,
		STATE_DONE,
		STATE_ERROR
	};

	eRenderState Render(
		bool isTracing,
		const TimeVolume& volume,
		const RenderingParameters& renderingParams,
		const ViewParams& viewParams,
		const StereoParams& stereoParams,
		const ParticleTraceParams& particleTraceParams,
		const ParticleRenderParams& particleRenderParams,
		const RaycastParams& raycastParams,
		const std::vector<LineBuffers*>& pLineBuffers,
		const std::vector<BallBuffers*>& pBallBuffers,
		float ballRadius,
		HeatMapManager* pHeatMapManager,
		cudaArray* pTransferFunction,
		SimpleParticleVertexDeltaT* dpParticles,
		int transferFunctionDevice = -1);

	eRenderState Render(
		std::vector<FlowVisToolVolumeData*>& volumes,
		const RenderingParameters& renderingParams,
		const ViewParams& viewParams,
		const StereoParams& stereoParams,
		const RaycastParams& raycastParams,
		const std::vector<LineBuffers*>& pLineBuffers,
		const std::vector<BallBuffers*>& pBallBuffers,
		float ballRadius,
		HeatMapManager* pHeatMapManager,
		cudaArray* pTransferFunction,
		int transferFunctionDevice = -1);
	
	ID3D11Texture2D*          GetOpaqueTex()    { return m_pOpaqueTex; }
	ID3D11ShaderResourceView* GetOpaqueSRV()    { return m_pOpaqueSRV; }

	void ClearResult();
	void CopyDepthTexture(ID3D11DeviceContext* deviceContext, ID3D11Texture2D* target);


#ifdef WriteVolumeToFileStuff
	bool WriteCurTimestepToRaws(TimeVolume& volume, const std::vector<std::string>& filenames);
	bool WriteCurTimestepToLA3Ds(TimeVolume& volume, const std::vector<std::string>& filenames);
#endif

private:
	bool CreateScreenDependentResources();

	void ReleaseScreenDependentResources();

	
	void RenderBoxes(const TimeVolume& vol, const RaycastParams& raycastParams, bool enableColor, bool blendBehind);

	//Renders the stream + path lines
	void RenderLines(const TimeVolume& vol, LineBuffers* pLineBuffers, bool enableColor, bool blendBehind);

	//If lineRenderMode==Particles, RenderLines delegates to this method after setting the shader parameters
	//If clipPlane != NULL, this defines a clip plane and only the points with a positive signed distance are drawn
	//If drawSlice == true, the slice texture drawing call is injected after the drawing of the particles.
	//  It assumes that the settings are already specified in RenderLines.
	//  This is the case if slice rendering is enabled.
	void RenderParticles(const LineBuffers* pLineBuffers, ID3D11DeviceContext* pContext, 
		D3D11_VIEWPORT viewport, const tum3D::Vec4f* clipPlane = NULL, bool drawSlice = false);

	//Prepares the slice renderer: computes the clip plane and sets the parameters
	//Returns the clip plane
	tum3D::Vec4f PrepareRenderSlice(ID3D11ShaderResourceView* tex, float alpha, float slicePosition, tum3D::Vec3f volumeSizeWorld, tum3D::Vec2f center);

	//Renders the slice if no lines are drawn.
	//When lines are drawn, slice rendering is done in the line/particle rendering 
	// to allow correct alpha blending
	void ExtraRenderSlice();

	void SortParticles(LineBuffers* pLineBuffers, ID3D11DeviceContext* pContext);
	
	void ComputeFTLE(const TimeVolume& vol, SimpleParticleVertexDeltaT* dpParticles);

	void RenderBalls(const TimeVolume& vol, const BallBuffers* pBallBuffers, float radius);
	

	void RenderHeatMap(HeatMapManager* pHeatMapManager);

	void CreateFTLETexture();

	bool			m_isCreated;
	Box				m_box;
	Range1D			m_range;
	ID3D11Device*	m_pDevice;

	ID3D11Texture1D*          m_pRandomColorsTex;
	ID3D11ShaderResourceView* m_pRandomColorsSRV;

	// screen-dependent resources
	// if there is a D3D device:
	ID3D11Texture2D*          m_pOpaqueTex;
	ID3D11ShaderResourceView* m_pOpaqueSRV;
	ID3D11RenderTargetView*   m_pOpaqueRTV;
	ID3D11Texture2D*          m_pTransparentTex;
	ID3D11ShaderResourceView* m_pTransparentSRV;
	ID3D11RenderTargetView*   m_pTransparentRTV;
	ID3D11Texture2D*          m_pDepthTex;
	ID3D11DepthStencilView*   m_pDepthDSV;
	ID3D11ShaderResourceView* m_pDepthSRV;
	
	LineEffect				  m_lineEffect;
	ScreenEffect*             m_pScreenEffect;
	QuadEffect*               m_pQuadEffect;
	
	RenderingParameters		m_renderingParams;
	ProjectionParams		m_projectionParams;
	ViewParams				m_viewParams;
	StereoParams			m_stereoParams;
	ParticleTraceParams		m_particleTraceParams;
	ParticleRenderParams	m_particleRenderParams;

private:
	// disallow copy and assignment
	RenderingManager(const RenderingManager&);
	RenderingManager& operator=(const RenderingManager&);
};


#endif
