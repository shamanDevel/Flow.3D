#pragma once

#include <Raycaster.h>
#include <RaycastParams.h>
#include <ViewParams.h>
#include <TimeVolume.h>
#include <GPUResources.h>
#include <CompressVolume.h>
#include <FilteredVolume.h>
#include <MultiTimerGPU.h>
#include <d3d11.h>

class RaycasterManager
{
public:
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

	enum eRenderState
	{
		STATE_RENDERING,
		STATE_DONE,
		STATE_ERROR
	};

	RaycasterManager();

	RaycasterManager::eRenderState RaycasterManager::StartRendering(
		const TimeVolume& volume,
		const ViewParams& viewParams,
		const StereoParams& stereoParams,
		const RaycastParams& raycastParams,
		const std::vector<FilteredVolume>& filteredVolumes,
		cudaArray* pTransferFunction,
		int transferFunctionDevice = -1);


	void ClearResult();

	bool Create(ID3D11Device* pDevice);
	void Release();
	void SetProjectionParams(const ProjectionParams& params, const Range1D& range);
	bool CreateScreenDependentResources();
	void ReleaseScreenDependentResources();

	const Timings& GetTimings() const { return m_timings; }
	const std::vector<const TimeVolumeIO::Brick*>& GetBricksToLoad() const { return m_bricksToLoad; }
	// if there is a D3D device:
	ID3D11Texture2D*          GetRaycastTex()   { return m_pRaycastTex; }
	ID3D11ShaderResourceView* GetRaycastSRV()   { return m_pRaycastSRV; }
	// if there is no D3D device:
	cudaArray*                GetRaycastArray() { return m_pRaycastArray; }

	void RenderBricks(bool recordEvents);
	bool IsRendering() const;
	// @Behdad
	bool waitForRendering() const;

	void CancelRendering();
	float GetRenderingProgress() const;
	void UpdateBricksToLoad();
	eRenderState Render();
	void UpdateTimings();

	bool CreateMeasureBrickSlots(uint count, uint countCompressed);
	void ReleaseMeasureBrickSlots();
	bool ManageMeasureBrickSlots();
	bool CreateVolumeDependentResources();
	void ReleaseVolumeDependentResources();
	// release memory-heavy resources (will be recreated on StartRendering)
	void ReleaseResources();
	size_t GetRequiredBrickSlotCount() const;


	ID3D11Device*	m_pDevice;

	Range1D						m_range;
	ViewParams					m_viewParams;
	StereoParams				m_stereoParams;
	ProjectionParams			m_projectionParams;
	RaycastParams				m_raycastParams;
	cudaArray*					m_pTfArray;
	Raycaster					m_raycaster;
	const TimeVolume*			m_pVolume;
	GPUResources				m_pCompressShared;
	CompressVolumeResources		m_pCompressVolume;
	// pre-computed measure bricks, only exist in appropriate eMeasureComputeModes
	std::vector<BrickSlot*>			m_measureBrickSlots;
	std::vector<bool>				m_measureBrickSlotsFilled;
	std::vector<std::vector<uint>>	m_measureBricksCompressed;
	std::vector<const TimeVolumeIO::Brick*> m_bricksToRender;
	std::vector<const TimeVolumeIO::Brick*> m_bricksClipped;
	std::vector<const TimeVolumeIO::Brick*> m_bricksToLoad;
	size_t									m_brickSlotsFilled;
	uint									m_bricksPerFrame;
	uint									m_nextBrickToRender;
	uint									m_nextPass;
	//@Behdad
	bool									m_waitForRendering = false;

	const std::vector<FilteredVolume>*		m_pFilteredVolumes;

	// volume-dependent resources
	std::vector<float*>		m_dpChannelBuffer;
	std::vector<BrickSlot>	m_brickSlots;
	int						m_brickSize;
	int						m_channelCount;

	// timing
	Timings       m_timings;
	TimerCPU      m_timerRender;
	MultiTimerGPU m_timerUploadDecompress;
	MultiTimerGPU m_timerComputeMeasure;
	MultiTimerGPU m_timerRaycast;
	MultiTimerGPU m_timerCompressDownload;

	// if there is no D3D device:
	cudaArray*	m_pRaycastArray;
	//TODO m_pDepthArray?

	// screen-dependent resources
	// if there is a D3D device:
	ID3D11Texture2D*          m_pRaycastTex;
	ID3D11ShaderResourceView* m_pRaycastSRV;
	ID3D11RenderTargetView*   m_pRaycastRTV;
	cudaGraphicsResource*     m_pRaycastTexCuda;
	cudaGraphicsResource*     m_pDepthTexCopyCuda;
	ID3D11Texture2D*          m_pDepthTexCopy;

private:
	// disable copy and assignment
	RaycasterManager(const RaycasterManager&);
	RaycasterManager& operator=(const RaycasterManager&);
};