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

class RenderingManager
{
public:
	RenderingManager();
	~RenderingManager();

	// pDevice may be null!
	bool Create(GPUResources* pCompressShared, CompressVolumeResources* pCompressVolume, ID3D11Device* pDevice);
	void Release();
	bool IsCreated() const { return m_isCreated; }

	void SetProjectionParams(const ProjectionParams& params, const Range1D& range);

	enum eRenderState
	{
		STATE_RENDERING,
		STATE_DONE,
		STATE_ERROR
	};

	eRenderState StartRendering( bool isTracing,
		const TimeVolume& volume, const std::vector<FilteredVolume>& filteredVolumes,
		const ViewParams& viewParams, const StereoParams& stereoParams,
		bool renderDomainBox, bool renderClipBox, bool renderSeedBox, bool renderBrickBoxes,
		const ParticleTraceParams& particleTraceParams, const ParticleRenderParams& particleRenderParams,
		const std::vector<LineBuffers*>& pLineBuffers, bool linesOnly,
		const std::vector<BallBuffers*>& pBallBuffers, float ballRadius,
		HeatMapManager* pHeatMapManager,
		const RaycastParams& raycastParams, cudaArray* pTransferFunction, int transferFunctionDevice = -1);
	eRenderState Render();
	void CancelRendering();
	bool IsRendering() const;
	float GetRenderingProgress() const;

	const std::vector<const TimeVolumeIO::Brick*>& GetBricksToLoad() const { return m_bricksToLoad; }

	// release memory-heavy resources (will be recreated on StartRendering)
	void ReleaseResources();

	// if there is a D3D device:
	ID3D11Texture2D*          GetRaycastTex()   { return m_pRaycastTex; }
	ID3D11ShaderResourceView* GetRaycastSRV()   { return m_pRaycastSRV; }
	ID3D11Texture2D*          GetOpaqueTex()    { return m_pOpaqueTex; }
	ID3D11ShaderResourceView* GetOpaqueSRV()    { return m_pOpaqueSRV; }
	// if there is no D3D device:
	cudaArray*                GetRaycastArray() { return m_pRaycastArray; }

	void ClearResult();


	struct Timings
	{
		Timings() { memset(this, 0, sizeof(*this)); }
		bool operator==(const Timings& other) { return memcmp(this, &other, sizeof(*this)) == 0; }
		bool operator!=(const Timings& other) { return !(*this == other); }

		MultiTimerGPU::Stats UploadDecompressGPU;
		MultiTimerGPU::Stats ComputeMeasureGPU;
		MultiTimerGPU::Stats RaycastGPU;
		MultiTimerGPU::Stats CompressDownloadGPU;

		float                RenderWall;
	};
	const Timings& GetTimings() const { return m_timings; }


	bool WriteCurTimestepToRaws(TimeVolume& volume, const std::vector<std::string>& filenames);
	bool WriteCurTimestepToLA3Ds(TimeVolume& volume, const std::vector<std::string>& filenames);

	D3D11CudaTexture	m_ftleTexture;

private:
	HRESULT CreateScreenDependentResources();
	void ReleaseScreenDependentResources();

	HRESULT CreateVolumeDependentResources();
	void ReleaseVolumeDependentResources();

	size_t GetRequiredBrickSlotCount() const;

	bool CreateMeasureBrickSlots(uint count, uint countCompressed);
	void ReleaseMeasureBrickSlots();
	bool ManageMeasureBrickSlots();


	void UpdateBricksToLoad();


	void RenderBoxes(bool enableColor, bool blendBehind);

	//Renders the stream + path lines
	void RenderLines(LineBuffers* pLineBuffers, bool enableColor, bool blendBehind);
	//If lineRenderMode==Particles, RenderLines delegates to this method after setting the shader parameters
	//If clipPlane != NULL, this defines a clip plane and only the points with a positive signed distance are drawn
	//If drawSlice == true, the slice texture drawing call is injected after the drawing of the particles.
	//  It assumes that the settings are already specified in RenderLines.
	//  This is the case if slice rendering is enabled.
	void RenderParticles(const LineBuffers* pLineBuffers, ID3D11DeviceContext* pContext, 
		D3D11_VIEWPORT viewport, const tum3D::Vec4f* clipPlane = NULL, bool drawSlice = false);

	//Prepares the slice renderer: computes the clip plane and sets the parameters
	//Returns the clip plane
	tum3D::Vec4f PrepareRenderSlice(ID3D11ShaderResourceView* tex, float alpha, float slicePosition);
	//Renders the slice if no lines are drawn.
	//When lines are drawn, slice rendering is done in the line/particle rendering 
	// to allow correct alpha blending
	void ExtraRenderSlice();

	void SortParticles(LineBuffers* pLineBuffers, ID3D11DeviceContext* pContext);
	
	void ComputeFTLE();

	void RenderBalls(const BallBuffers* pBallBuffers, float radius);
	void RenderBricks(bool recordEvents);

	void RenderHeatMap(HeatMapManager* pHeatMapManager);

	void UpdateTimings();

	void CreateFTLETexture();

	bool          m_isCreated;

	GPUResources*            m_pCompressShared;
	CompressVolumeResources* m_pCompressVolume;
	ID3D11Device*            m_pDevice;

	Box           m_box;

	LineEffect    m_lineEffect;
	Raycaster     m_raycaster;

	ProjectionParams m_projectionParams;
	Range1D          m_range;

	ID3D11Texture1D*          m_pRandomColorsTex;
	ID3D11ShaderResourceView* m_pRandomColorsSRV;

	// screen-dependent resources
	// if there is a D3D device:
	ID3D11Texture2D*          m_pOpaqueTex;
	ID3D11ShaderResourceView* m_pOpaqueSRV;
	ID3D11RenderTargetView*   m_pOpaqueRTV;
	ID3D11Texture2D*          m_pRaycastTex;
	ID3D11ShaderResourceView* m_pRaycastSRV;
	ID3D11RenderTargetView*   m_pRaycastRTV;
	cudaGraphicsResource*     m_pRaycastTexCuda;
	ID3D11Texture2D*          m_pTransparentTex;
	ID3D11ShaderResourceView* m_pTransparentSRV;
	ID3D11RenderTargetView*   m_pTransparentRTV;
	ID3D11Texture2D*          m_pDepthTex;
	ID3D11DepthStencilView*   m_pDepthDSV;
	ID3D11ShaderResourceView* m_pDepthSRV;
	ID3D11Texture2D*          m_pDepthTexCopy;
	cudaGraphicsResource*     m_pDepthTexCopyCuda;
	// if there is no D3D device:
	cudaArray*                m_pRaycastArray;
	//TODO m_pDepthArray?

	ScreenEffect*             m_pScreenEffect;
	QuadEffect*               m_pQuadEffect;

	// volume-dependent resources
	std::vector<float*> m_dpChannelBuffer;
	std::vector<BrickSlot> m_brickSlots;
	int m_brickSize;
	int m_channelCount;


	uint m_bricksPerFrame;

	// only valid while rendering
	const TimeVolume*                  m_pVolume;
	const std::vector<FilteredVolume>* m_pFilteredVolumes;
	cudaArray*                         m_pTfArray;

	ViewParams           m_viewParams;
	StereoParams         m_stereoParams;
	ParticleTraceParams  m_particleTraceParams;
	ParticleRenderParams m_particleRenderParams;
	RaycastParams        m_raycastParams;

	bool m_renderDomainBox;
	bool m_renderClipBox;
	bool m_renderSeedBox;
	bool m_renderBrickBoxes;

	std::vector<const TimeVolumeIO::Brick*> m_bricksToRender;
	std::vector<const TimeVolumeIO::Brick*> m_bricksClipped;
	uint m_nextBrickToRender;
	uint m_nextPass;
	size_t m_brickSlotsFilled;

	std::vector<const TimeVolumeIO::Brick*> m_bricksToLoad;


	// pre-computed measure bricks, only exist in appropriate eMeasureComputeModes
	std::vector<BrickSlot*> m_measureBrickSlots;
	std::vector<bool>       m_measureBrickSlotsFilled;
	std::vector<std::vector<uint>> m_measureBricksCompressed;


	// timing
	MultiTimerGPU m_timerUploadDecompress;
	MultiTimerGPU m_timerComputeMeasure;
	MultiTimerGPU m_timerRaycast;
	MultiTimerGPU m_timerCompressDownload;
	TimerCPU      m_timerRender;

	Timings       m_timings;


	// disallow copy and assignment
	RenderingManager(const RenderingManager&);
	RenderingManager& operator=(const RenderingManager&);
};


#endif
