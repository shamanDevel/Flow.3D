#include "RenderingManager.h"

#include <cassert>
#include <numeric>

#include <cudaUtil.h>
#include <thrust/device_ptr.h>
#include <thrust\sort.h>

#include <LargeArray3D.h>

#include "BoxUtils.h"
#include "BrickUpload.h"
#include "ClearCudaArray.h"
#include "TracingCommon.h"

using namespace tum3D;


RenderingManager::RenderingManager()
	: m_isCreated(false), m_pCompressShared(nullptr), m_pCompressVolume(nullptr), m_pDevice(nullptr)
	, m_pRandomColorsTex(nullptr), m_pRandomColorsSRV(nullptr)
	, m_pOpaqueTex(nullptr), m_pOpaqueSRV(nullptr), m_pOpaqueRTV(nullptr)
	, m_pRaycastTex(nullptr), m_pRaycastSRV(nullptr), m_pRaycastRTV(nullptr), m_pRaycastTexCuda(nullptr)
	, m_pRaycastArray(nullptr)
	, m_pTransparentTex(nullptr), m_pTransparentSRV(nullptr), m_pTransparentRTV(nullptr)
	, m_pDepthTex(nullptr), m_pDepthDSV(nullptr)
	, m_pDepthTexCopy(nullptr), m_pDepthTexCopyCuda(nullptr)
	, m_brickSize(0), m_channelCount(0)
	, m_bricksPerFrame(1)
	, m_pVolume(nullptr), m_pFilteredVolumes(nullptr)
	, m_renderDomainBox(false), m_renderClipBox(false), m_renderSeedBox(false), m_renderBrickBoxes(false)
	, m_pTfArray(nullptr)
	, m_nextBrickToRender(0), m_nextPass(0)
	, m_brickSlotsFilled(0)
	, m_pScreenEffect(nullptr), m_pQuadEffect(nullptr)
{
}

RenderingManager::~RenderingManager()
{
	assert(!m_isCreated);
}


bool RenderingManager::Create(GPUResources* pCompressShared, CompressVolumeResources* pCompressVolume, ID3D11Device* pDevice)
{
	m_pCompressShared = pCompressShared;
	m_pCompressVolume = pCompressVolume;
	m_pDevice = pDevice;

	HRESULT hr;

	if(FAILED(hr = CreateScreenDependentResources()))
	{
		Release();
		return false;
	}

	if(FAILED(hr = m_box.Create(m_pDevice)))
	{
		Release();
		return false;
	}

	if(FAILED(hr = m_lineEffect.Create(m_pDevice)))
	{
		Release();
		return false;
	}

	if(!m_raycaster.Create())
	{
		Release();
		return false;
	}

	m_pScreenEffect = new ScreenEffect();
	if (FAILED(hr = m_pScreenEffect->Create(m_pDevice)))
	{
		Release();
		return false;
	}

	m_pQuadEffect = new QuadEffect();
	if (FAILED(hr = m_pQuadEffect->Create(m_pDevice)))
	{
		Release();
		return false;
	}

	m_isCreated = true;

	return true;
}

void RenderingManager::Release()
{
	CancelRendering();

	ReleaseResources();
	ReleaseScreenDependentResources();

	m_box.Release();

	m_raycaster.Release();
	m_lineEffect.SafeRelease();

	m_timerUploadDecompress.ReleaseTimers();
	m_timerComputeMeasure.ReleaseTimers();
	m_timerRaycast.ReleaseTimers();
	m_timerCompressDownload.ReleaseTimers();

	m_pDevice = nullptr;
	m_pCompressVolume = nullptr;
	m_pCompressShared = nullptr;

	m_isCreated = false;

	if (m_pScreenEffect) {
		m_pScreenEffect->SafeRelease();
		m_pScreenEffect = nullptr;
	}
	if (m_pQuadEffect) {
		m_pQuadEffect->SafeRelease();
		m_pQuadEffect = nullptr;
	}
}


void RenderingManager::SetProjectionParams(const ProjectionParams& params, const Range1D& range)
{
	if(params == m_projectionParams && range == m_range) return;

	CancelRendering();

	bool recreateScreenResources =
		(params.GetImageHeight(range) != m_projectionParams.GetImageHeight(m_range) ||
		 params.GetImageWidth (range) != m_projectionParams.GetImageWidth (m_range));

	m_projectionParams = params;
	m_range = range;

	if(recreateScreenResources)
	{
		CreateScreenDependentResources();
	}
}


bool RenderingManager::WriteCurTimestepToRaws(TimeVolume& volume, const std::vector<std::string>& filenames)
{
	//TODO do this in slabs/slices of bricks, so that mem usage won't be quite so astronomical...
	int channelCount = volume.GetChannelCount();

	if(filenames.size() < channelCount)
	{
		printf("FAIL!\n");
		return false;
	}

	std::vector<FILE*> files;
	for(int c = 0; c < channelCount; c++)
	{
		FILE* file = fopen(filenames[c].c_str(), "wb");
		if(!file)
		{
			printf("WriteCurTimestepToRaw: Failed creating output file %s\n", filenames[c].c_str());
			//TODO close previous files
			return false;
		}
		files.push_back(file);
	}

	CancelRendering();

	m_pVolume = &volume;

	// save mem usage limit, and set to something reasonably small
	float memUsageLimitOld = volume.GetSystemMemoryUsage().GetSystemMemoryLimitMBytes();
	volume.GetSystemMemoryUsage().SetSystemMemoryLimitMBytes(1024.0f);
	volume.UnloadLRUBricks();

	CreateVolumeDependentResources();

	int brickSizeWith = volume.GetBrickSizeWithOverlap();
	int brickSizeWithout = volume.GetBrickSizeWithoutOverlap();
	int brickOverlap = volume.GetBrickOverlap();
	Vec3i volumeSize = volume.GetVolumeSize();
	std::vector<float> brickChannelData(brickSizeWith * brickSizeWith * brickSizeWith);
	std::vector<std::vector<float>> volumeData;
	for(int c = 0; c < channelCount; c++)
	{
		volumeData.push_back(std::vector<float>(volumeSize.x() * volumeSize.y() * brickSizeWithout));
	}

	volume.DiscardAllIO();
	auto& bricks = volume.GetNearestTimestep().bricks;
	// HACK: assume bricks are already in the correct order (z-major)
	for(size_t i = 0; i < bricks.size(); i++)
	{
		TimeVolumeIO::Brick& brick = bricks[i];

		// load from disk if required
		if(!brick.IsLoaded())
		{
			volume.EnqueueLoadBrickData(brick);
			volume.WaitForAllIO();
			volume.UnloadLRUBricks();
		}

		// decompress
		UploadBrick(m_pCompressShared, m_pCompressVolume, m_pVolume->GetInfo(), brick, m_dpChannelBuffer.data());
		for(int c = 0; c < channelCount; c++)
		{
			// download into CPU memory
			cudaSafeCall(cudaMemcpy(brickChannelData.data(), m_dpChannelBuffer[c], brickSizeWith * brickSizeWith * brickSizeWith * sizeof(float), cudaMemcpyDeviceToHost));
			// copy into whole volume array
			Vec3i offset = brick.GetSpatialIndex() * brickSizeWithout - brickOverlap;
			offset.z() = -brickOverlap;
			Vec3i thisBrickSize = Vec3i(brick.GetSize());
			for(int z = brickOverlap; z < thisBrickSize.z() - brickOverlap; z++)
			{
				for(int y = brickOverlap; y < thisBrickSize.y() - brickOverlap; y++)
				{
					for(int x = brickOverlap; x < thisBrickSize.x() - brickOverlap; x++)
					{
						int in = x + thisBrickSize.x() * (y + thisBrickSize.y() * z);
						int out = (offset.x()+x) + volumeSize.x() * ((offset.y()+y) + volumeSize.y() * (offset.z()+z));
						volumeData[c][out] = brickChannelData[in];
					}
				}
			}
		}

		printf("Brick %i / %i done\n", int(i+1), int(bricks.size()));

		// if this was the last brick of this slab, write it out!
		if(i+1 >= bricks.size() || bricks[i+1].GetSpatialIndex().z() != bricks[i].GetSpatialIndex().z())
		{
			int z0 = brick.GetSpatialIndex().z() * brickSizeWithout;
			int slices = brick.GetSize().z() - 2 * brickOverlap;
			printf("Writing slices %i-%i / %i...", z0 + 1, z0 + slices, volumeSize.z());
			for(int c = 0; c < channelCount; c++)
			{
				int elemsPerSlice = volumeSize.x() * volumeSize.y();
				for(int z = 0; z < slices; z++)
				{
					fwrite(volumeData[c].data() + z * elemsPerSlice, sizeof(float), elemsPerSlice, files[c]);
				}
			}
			printf("done\n");
		}
	}

	ReleaseVolumeDependentResources();

	m_pVolume = nullptr;

	for(int c = 0; c < channelCount; c++)
	{
		fclose(files[c]);
	}

	// restore mem usage limit
	volume.GetSystemMemoryUsage().SetSystemMemoryLimitMBytes(memUsageLimitOld);

	return true;
}


bool RenderingManager::WriteCurTimestepToLA3Ds(TimeVolume& volume, const std::vector<std::string>& filenames)
{
	int channelCount = volume.GetChannelCount();

	if(filenames.size() < channelCount)
	{
		printf("FAIL!\n");
		return false;
	}

	std::vector<LA3D::LargeArray3D<float>> files(channelCount);
	Vec3i volumeSize = volume.GetVolumeSize();
	for(int c = 0; c < channelCount; c++)
	{
		std::wstring filenamew(filenames[c].begin(), filenames[c].end());
		if(!files[c].Create(volumeSize.x(), volumeSize.y(), volumeSize.z(), 64, 64, 64, filenamew.c_str(), 1024 * 1024 * 1024))
		{
			printf("WriteCurTimestepToLA3Ds: Failed creating output file %s\n", filenames[c].c_str());
			//TODO close previous files
			return false;
		}
	}

	CancelRendering();

	m_pVolume = &volume;

	CreateVolumeDependentResources();

	int brickSizeWith = volume.GetBrickSizeWithOverlap();
	int brickSizeWithout = volume.GetBrickSizeWithoutOverlap();
	int brickOverlap = volume.GetBrickOverlap();
	std::vector<float> brickChannelData(brickSizeWith * brickSizeWith * brickSizeWith);

	volume.DiscardAllIO();
	auto& bricks = volume.GetNearestTimestep().bricks;
	for(size_t i = 0; i < bricks.size(); i++)
	{
		TimeVolumeIO::Brick& brick = bricks[i];
		// load from disk if required
		if(!brick.IsLoaded())
		{
			volume.EnqueueLoadBrickData(brick);
			volume.WaitForAllIO();
		}
		// decompress
		UploadBrick(m_pCompressShared, m_pCompressVolume, m_pVolume->GetInfo(), brick, m_dpChannelBuffer.data());
		for(int c = 0; c < channelCount; c++)
		{
			// download into CPU memory
			cudaSafeCall(cudaMemcpy(brickChannelData.data(), m_dpChannelBuffer[c], brickSizeWith * brickSizeWith * brickSizeWith * sizeof(float), cudaMemcpyDeviceToHost));
			// copy into la3d
			Vec3i target = brick.GetSpatialIndex() * brickSizeWithout;
			Vec3i thisBrickSize = Vec3i(brick.GetSize());
			Vec3i copySize = thisBrickSize - 2 * brickOverlap;
			size_t offset = brickOverlap + thisBrickSize.x() * (brickOverlap + thisBrickSize.y() * brickOverlap);
			files[c].CopyFrom(brickChannelData.data() + offset, target.x(), target.y(), target.z(), copySize.x(), copySize.y(), copySize.z(), thisBrickSize.x(), thisBrickSize.y());
		}

		printf("%i / %i\n", int(i+1), int(bricks.size()));
	}

	ReleaseVolumeDependentResources();

	m_pVolume = nullptr;

	for(int c = 0; c < channelCount; c++)
	{
		files[c].Close();
	}

	return true;
}


HRESULT RenderingManager::CreateScreenDependentResources()
{
	ReleaseScreenDependentResources();

	uint width  = m_projectionParams.GetImageWidth (m_range);
	uint height = m_projectionParams.GetImageHeight(m_range);

	if(width == 0 || height == 0)
	{
		return S_OK;
	}


	if(m_pDevice)
	{
		HRESULT hr;

		D3D11_TEXTURE1D_DESC desc1;
		desc1.ArraySize = 1;
		desc1.BindFlags = D3D11_BIND_SHADER_RESOURCE;
		desc1.CPUAccessFlags = 0;
		desc1.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
		desc1.MipLevels = 1;
		desc1.MiscFlags = 0;
		desc1.Usage = D3D11_USAGE_DEFAULT;
		desc1.Width = 1024;

		std::vector<byte> colors(desc1.Width * 4);
		srand(0);
		for(uint i = 0; i < desc1.Width; i++) {
			colors[4*i+0] = rand() % 256;
			colors[4*i+1] = rand() % 256;
			colors[4*i+2] = rand() % 256;
			colors[4*i+3] = 255;
		}

		// HACK: set first few colors to fixed "primary" colors, for loaded lines
		colors[0] = 0;
		colors[1] = 0;
		colors[2] = 255;
		colors[3] = 255;

		colors[4] = 255;
		colors[5] = 0;
		colors[6] = 0;
		colors[7] = 255;

		colors[8] = 255;
		colors[9] = 255;
		colors[10] = 0;
		colors[11] = 255;

		D3D11_SUBRESOURCE_DATA initData = {};
		initData.pSysMem = colors.data();
		hr = m_pDevice->CreateTexture1D(&desc1, &initData, &m_pRandomColorsTex);
		if(FAILED(hr)) return hr;
		hr = m_pDevice->CreateShaderResourceView(m_pRandomColorsTex, nullptr, &m_pRandomColorsSRV);
		if(FAILED(hr)) return hr;

		// create texture/rendertarget for opaque objects
		D3D11_TEXTURE2D_DESC desc;
		desc.ArraySize = 1;
		desc.BindFlags = D3D11_BIND_RENDER_TARGET | D3D11_BIND_SHADER_RESOURCE;
		desc.CPUAccessFlags = 0;
		desc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
		desc.MipLevels = 1;
		desc.MiscFlags = 0;
		desc.SampleDesc.Count = 1;
		desc.SampleDesc.Quality = 0;
		desc.Usage = D3D11_USAGE_DEFAULT;
		desc.Width = width;
		desc.Height = height;
		hr = m_pDevice->CreateTexture2D(&desc, nullptr, &m_pOpaqueTex);
		if(FAILED(hr)) return hr;
		hr = m_pDevice->CreateShaderResourceView(m_pOpaqueTex, nullptr, &m_pOpaqueSRV);
		if(FAILED(hr)) return hr;
		hr = m_pDevice->CreateRenderTargetView(m_pOpaqueTex, nullptr, &m_pOpaqueRTV);
		if(FAILED(hr)) return hr;

		// create texture/rendertarget for raycasting
		desc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
		hr = m_pDevice->CreateTexture2D(&desc, nullptr, &m_pRaycastTex);
		if(FAILED(hr)) return hr;
		hr = m_pDevice->CreateShaderResourceView(m_pRaycastTex, nullptr, &m_pRaycastSRV);
		if(FAILED(hr)) return hr;
		hr = m_pDevice->CreateRenderTargetView(m_pRaycastTex, nullptr, &m_pRaycastRTV);
		if(FAILED(hr)) return hr;
		cudaSafeCall(cudaGraphicsD3D11RegisterResource(&m_pRaycastTexCuda, m_pRaycastTex, cudaGraphicsRegisterFlagsSurfaceLoadStore));

		// create texture for transparent particle rendering
		desc.Format = DXGI_FORMAT_R16G16B16A16_FLOAT;
		hr = m_pDevice->CreateTexture2D(&desc, nullptr, &m_pTransparentTex);
		if (FAILED(hr)) return hr;
		hr = m_pDevice->CreateShaderResourceView(m_pTransparentTex, nullptr, &m_pTransparentSRV);
		if (FAILED(hr)) return hr;
		hr = m_pDevice->CreateRenderTargetView(m_pTransparentTex, nullptr, &m_pTransparentRTV);
		if (FAILED(hr)) return hr;

		// create depth buffer
		desc.BindFlags = D3D11_BIND_DEPTH_STENCIL;
		desc.Format = DXGI_FORMAT_D32_FLOAT;
		hr = m_pDevice->CreateTexture2D(&desc, nullptr, &m_pDepthTex);
		if(FAILED(hr)) return hr;
		hr = m_pDevice->CreateDepthStencilView(m_pDepthTex, nullptr, &m_pDepthDSV);
		if(FAILED(hr)) return hr;

		// CUDA can't share depth or typeless resources, so we have to allocate another tex and copy into it...
		desc.BindFlags = 0;
		desc.Format = DXGI_FORMAT_R32_FLOAT;
		hr = m_pDevice->CreateTexture2D(&desc, nullptr, &m_pDepthTexCopy);
		if(FAILED(hr)) return hr;
		cudaSafeCall(cudaGraphicsD3D11RegisterResource(&m_pDepthTexCopyCuda, m_pDepthTexCopy, cudaGraphicsRegisterFlagsSurfaceLoadStore));
	}
	else
	{
		cudaChannelFormatDesc desc = cudaCreateChannelDesc<uchar4>();
		cudaSafeCall(cudaMallocArray(&m_pRaycastArray, &desc, width, height, cudaArraySurfaceLoadStore));
	}

	return S_OK;
}

void RenderingManager::ReleaseScreenDependentResources()
{
	cudaSafeCall(cudaFreeArray(m_pRaycastArray));
	m_pRaycastArray = nullptr;

	if(m_pDepthTexCopyCuda)
	{
		cudaSafeCall(cudaGraphicsUnregisterResource(m_pDepthTexCopyCuda));
		m_pDepthTexCopyCuda = nullptr;
	}

	if(m_pDepthTexCopy)
	{
		m_pDepthTexCopy->Release();
		m_pDepthTexCopy = nullptr;
	}

	if(m_pDepthDSV)
	{
		m_pDepthDSV->Release();
		m_pDepthDSV = nullptr;
	}

	if(m_pDepthTex)
	{
		m_pDepthTex->Release();
		m_pDepthTex = nullptr;
	}

	if(m_pOpaqueRTV)
	{
		m_pOpaqueRTV->Release();
		m_pOpaqueRTV = nullptr;
	}

	if(m_pOpaqueSRV)
	{
		m_pOpaqueSRV->Release();
		m_pOpaqueSRV = nullptr;
	}

	if(m_pOpaqueTex)
	{
		m_pOpaqueTex->Release();
		m_pOpaqueTex = nullptr;
	}

	if(m_pRandomColorsSRV)
	{
		m_pRandomColorsSRV->Release();
		m_pRandomColorsSRV = nullptr;
	}

	if(m_pRandomColorsTex)
	{
		m_pRandomColorsTex->Release();
		m_pRandomColorsTex = nullptr;
	}

	if(m_pRaycastTexCuda)
	{
		cudaSafeCall(cudaGraphicsUnregisterResource(m_pRaycastTexCuda));
		m_pRaycastTexCuda = nullptr;
	}

	if(m_pRaycastRTV)
	{
		m_pRaycastRTV->Release();
		m_pRaycastRTV = nullptr;
	}

	if(m_pRaycastSRV)
	{
		m_pRaycastSRV->Release();
		m_pRaycastSRV = nullptr;
	}

	if(m_pRaycastTex)
	{
		m_pRaycastTex->Release();
		m_pRaycastTex = nullptr;
	}

	if (m_pTransparentRTV)
	{
		m_pTransparentRTV->Release();
		m_pTransparentRTV = nullptr;
	}

	if (m_pTransparentSRV)
	{
		m_pTransparentSRV->Release();
		m_pTransparentSRV = nullptr;
	}

	if (m_pTransparentTex)
	{
		m_pTransparentTex->Release();
		m_pTransparentTex = nullptr;
	}
}


HRESULT RenderingManager::CreateVolumeDependentResources()
{
	assert(m_pVolume != nullptr);

	ReleaseVolumeDependentResources();


	m_brickSize = m_pVolume->GetBrickSizeWithOverlap();
	m_channelCount = m_pVolume->GetChannelCount();

	int brickSizeBytePerChannel = m_brickSize * m_brickSize * m_brickSize * sizeof(float);
	m_dpChannelBuffer.resize(m_channelCount);
	for(size_t channel = 0; channel < m_dpChannelBuffer.size(); channel++)
	{
		cudaSafeCall(cudaMalloc(&m_dpChannelBuffer[channel], brickSizeBytePerChannel));
	}

	m_brickSlots.resize(GetRequiredBrickSlotCount());
	printf("RenderingManager::CreateVolumeDependentResources: Creating %u brick slot(s)\n", uint(m_brickSlots.size()));
	for(size_t i = 0; i < m_brickSlots.size(); ++i)
	{
		if(!m_brickSlots[i].Create(m_brickSize, m_channelCount))
		{
			printf("RenderingManager::CreateVolumeDependentResources: m_brickSlots[%u].Create failed\n", uint(i));
			ReleaseVolumeDependentResources();
			return E_FAIL;
		}
	}

	return S_OK;
}

void RenderingManager::ReleaseVolumeDependentResources()
{
	for(size_t i = 0; i < m_brickSlots.size(); ++i)
	{
		m_brickSlots[i].Release();
	}
	m_brickSlots.clear();

	for(size_t channel = 0; channel < m_dpChannelBuffer.size(); channel++)
	{
		cudaSafeCall(cudaFree(m_dpChannelBuffer[channel]));
	}
	m_dpChannelBuffer.clear();

	m_channelCount = 0;
	m_brickSize = 0;
}


bool RenderingManager::CreateMeasureBrickSlots(uint count, uint countCompressed)
{
	ReleaseMeasureBrickSlots();

	bool success = true;
	for(uint i = 0; i < count; i++)
	{
		m_measureBrickSlots.push_back(new BrickSlot());
		m_measureBrickSlotsFilled.push_back(false);
		if(!m_measureBrickSlots.back()->Create(m_pVolume->GetBrickSizeWithOverlap(), 1))
		{
			success = false;
			break;
		}
	}

	for(uint i = 0; i < countCompressed; i++)
	{
		m_measureBricksCompressed.push_back(std::vector<uint>());
	}

	if(!success)
	{
		ReleaseMeasureBrickSlots();
		return false;
	}

	return true;
}

void RenderingManager::ReleaseMeasureBrickSlots()
{
	for(uint i = 0; i < m_measureBrickSlots.size(); i++)
	{
		if(m_measureBrickSlots[i])
		{
			m_measureBrickSlots[i]->Release();
			delete m_measureBrickSlots[i];
		}
	}
	m_measureBrickSlots.clear();
	m_measureBrickSlotsFilled.clear();
	m_measureBricksCompressed.clear();
}

size_t RenderingManager::GetRequiredBrickSlotCount() const
{
	size_t brickSlotCount = RaycastModeScaleCount(m_raycastParams.m_raycastMode);
	brickSlotCount = max(brickSlotCount, size_t(1)); // HACK ... (happens when raycastMode is invalid)
	size_t filteredVolumeCount = m_pFilteredVolumes == nullptr ? 0 : m_pFilteredVolumes->size();
	brickSlotCount = min(brickSlotCount, filteredVolumeCount + 1);
	return brickSlotCount;
}

bool RenderingManager::ManageMeasureBrickSlots()
{
	assert(m_pVolume != nullptr);

	uint slotCountTarget = 0;
	uint slotCountCompressedTarget = 0;
	switch(m_raycastParams.m_measureComputeMode)
	{
		case MEASURE_COMPUTE_PRECOMP_DISCARD:
			slotCountTarget = 1;
			break;
		case MEASURE_COMPUTE_PRECOMP_STORE_GPU:
			slotCountTarget = m_pVolume->GetBrickCount().volume();
			break;
		case MEASURE_COMPUTE_PRECOMP_COMPRESS:
			slotCountTarget = 1;
			slotCountCompressedTarget = m_pVolume->GetBrickCount().volume();
			break;
	}

	bool recreate = (slotCountTarget != m_measureBrickSlots.size() || slotCountCompressedTarget != m_measureBricksCompressed.size());
	if(!recreate)
	{
		return true;
	}

	bool success = CreateMeasureBrickSlots(slotCountTarget, slotCountCompressedTarget);

	return success;
}


namespace
{
	float GetDistance(const TimeVolume& volume, Vec3i brickIndex, Vec3f camPos)
	{
		Vec3f brickIndexFloat; castVec(brickIndex, brickIndexFloat);
		Vec3f brickCenter = -volume.GetVolumeHalfSizeWorld() + (brickIndexFloat + 0.5f) * volume.GetBrickSizeWorld(); // use full brick size, even for border bricks which might be smaller!
		return distance(camPos, brickCenter);
	}

	struct BrickSortItem
	{
		BrickSortItem(const TimeVolumeIO::Brick* pBrick, float distance)
			: pBrick(pBrick), distance(distance) {}

		const TimeVolumeIO::Brick* pBrick;
		float                      distance;

		bool operator<(const BrickSortItem& rhs) { return distance < rhs.distance; }
	};
}


RenderingManager::eRenderState RenderingManager::StartRendering(const TimeVolume& volume, const std::vector<FilteredVolume>& filteredVolumes,
	const ViewParams& viewParams, const StereoParams& stereoParams,
	bool renderDomainBox, bool renderClipBox, bool renderSeedBox, bool renderBrickBoxes,
	const ParticleTraceParams& particleTraceParams, const ParticleRenderParams& particleRenderParams,
	const std::vector<LineBuffers*>& pLineBuffers, bool linesOnly,
	const std::vector<BallBuffers*>& pBallBuffers, float ballRadius,
	const RaycastParams& raycastParams, cudaArray* pTransferFunction, int transferFunctionDevice)
{
	if(IsRendering()) CancelRendering();

	if(!volume.IsOpen()) return STATE_ERROR;

	if(m_projectionParams.GetImageWidth(m_range) * m_projectionParams.GetImageHeight(m_range) == 0)
	{
		return STATE_DONE;
	}

	bool measureChanged = (m_raycastParams.m_measure1 != raycastParams.m_measure1);

	m_pVolume = &volume;
	m_pFilteredVolumes = &filteredVolumes;

	m_viewParams = viewParams;
	m_stereoParams = stereoParams;

	m_particleTraceParams = particleTraceParams;
	m_particleRenderParams = particleRenderParams;
	m_raycastParams = raycastParams;

	m_renderDomainBox = renderDomainBox;
	m_renderClipBox = renderClipBox;
	m_renderSeedBox = renderSeedBox;
	m_renderBrickBoxes = renderBrickBoxes;

	bool doRaycasting = m_raycastParams.m_raycastingEnabled && !linesOnly;

	// copy transfer function
	if(doRaycasting && pTransferFunction)
	{
		cudaChannelFormatDesc desc;
		cudaExtent extent;
		cudaSafeCall(cudaArrayGetInfo(&desc, &extent, nullptr, pTransferFunction));
		// for 1D arrays, this returns 0 in width and height...
		extent.height = (extent.height, size_t(1));
		extent.depth = (extent.depth, size_t(1));
		cudaSafeCall(cudaMallocArray(&m_pTfArray, &desc, extent.width, extent.height));
		size_t elemCount = extent.width * extent.height * extent.depth;
		size_t bytesPerElem = (desc.x + desc.y + desc.z + desc.w) / 8;

		int myDevice = -1;
		cudaSafeCall(cudaGetDevice(&myDevice));
		if(myDevice == transferFunctionDevice || transferFunctionDevice == -1)
		{
			cudaSafeCall(cudaMemcpyArrayToArray(m_pTfArray, 0, 0, pTransferFunction, 0, 0, elemCount * bytesPerElem));
		}
		else
		{
			cudaMemcpy3DPeerParms params = {};
			params.srcArray = pTransferFunction;
			params.srcDevice = transferFunctionDevice;
			params.srcPos = make_cudaPos(0, 0, 0);
			params.dstArray = m_pTfArray;
			params.dstDevice = myDevice;
			params.dstPos = make_cudaPos(0, 0, 0);
			params.extent = extent;
			cudaSafeCall(cudaMemcpy3DPeer(&params));
		}
	}

	//This would stop rendering if lines and raycasting are disabled
	//By removing that, at least the outlines are rendered and one can see the camera interactions
#if 0
	if(!doRaycasting && !m_particleRenderParams.m_linesEnabled && pBallBuffers.empty())
	{
		// nothing to do...
		CancelRendering();
		return STATE_DONE;
	}
#endif

	if(doRaycasting)
	{
		// set params and create resources
		m_raycaster.SetBrickSizeWorld(m_pVolume->GetBrickSizeWorld());
		m_raycaster.SetBrickOverlapWorld(m_pVolume->GetBrickOverlapWorld());
		m_raycaster.SetGridSpacing(m_pVolume->GetGridSpacing());
		m_raycaster.SetParams(m_projectionParams, m_stereoParams, m_range);

		if( m_pVolume->GetBrickSizeWithOverlap() != m_brickSize ||
			m_pVolume->GetChannelCount() != m_channelCount ||
			GetRequiredBrickSlotCount() != m_brickSlots.size())
		{
			if(FAILED(CreateVolumeDependentResources()))
			{
				MessageBoxA(nullptr, "RenderingManager::StartRendering: Failed creating volume-dependent resources! (probably not enough GPU memory)", "Fail", MB_OK | MB_ICONINFORMATION);
				CancelRendering();
				return STATE_ERROR;
			}
		}

		if(!ManageMeasureBrickSlots())
		{
			MessageBoxA(nullptr, "RenderingManager::StartRendering: Failed creating brick slots for precomputed measure (probably not enough GPU memory). Reverting to on-the-fly computation.", "Fail", MB_OK | MB_ICONINFORMATION);
			m_raycastParams.m_measureComputeMode = MEASURE_COMPUTE_ONTHEFLY;
		}

		// if the measure was changed, have to clear all the precomputed measure bricks
		if(measureChanged)
		{
			for(uint i = 0; i < m_measureBrickSlotsFilled.size(); i++)
			{
				m_measureBrickSlotsFilled[i] = false;
			}
			for(uint i = 0; i < m_measureBricksCompressed.size(); i++)
			{
				m_measureBricksCompressed[i].clear();
			}
		}
	}
	else
	{
		// raycasting disabled, release resources
		ReleaseMeasureBrickSlots();
		ReleaseVolumeDependentResources();
	}


	// compute frustum planes in view space
	Vec4f frustumPlanes[6];
	Vec4f frustumPlanes2[6];
	if(m_stereoParams.m_stereoEnabled)
	{
		m_projectionParams.GetFrustumPlanes(frustumPlanes,  EYE_LEFT,  m_stereoParams.m_eyeDistance, m_range);
		m_projectionParams.GetFrustumPlanes(frustumPlanes2, EYE_RIGHT, m_stereoParams.m_eyeDistance, m_range);

		// we want the inverse transpose of the inverse of view
		Mat4f viewLeft = m_viewParams.BuildViewMatrix(EYE_LEFT, m_stereoParams.m_eyeDistance);
		Mat4f viewLeftTrans;
		viewLeft.transpose(viewLeftTrans);
		Mat4f viewRight = m_viewParams.BuildViewMatrix(EYE_RIGHT, m_stereoParams.m_eyeDistance);
		Mat4f viewRightTrans;
		viewRight.transpose(viewRightTrans);

		for(uint i = 0; i < 6; i++)
		{
			frustumPlanes [i] = viewLeftTrans  * frustumPlanes[i];
			frustumPlanes2[i] = viewRightTrans * frustumPlanes[i];
		}
	}
	else
	{
		m_projectionParams.GetFrustumPlanes(frustumPlanes, EYE_CYCLOP, 0.0f, m_range);

		// we want the inverse transpose of the inverse of view
		Mat4f view = m_viewParams.BuildViewMatrix(EYE_CYCLOP, 0.0f);
		Mat4f viewTrans;
		view.transpose(viewTrans);

		for(uint i = 0; i < 6; i++)
		{
			frustumPlanes[i] = viewTrans * frustumPlanes[i];
		}
	}

	// if we're doing raycasting, build list of bricks to render
	m_bricksToRender.clear();
	m_nextBrickToRender = 0;
	m_nextPass = 0;
	if(doRaycasting)
	{
		// get bricks of current timestep, sort by distance
		std::vector<BrickSortItem> bricksCurTimestep;
		const Vec3f& camPos = m_viewParams.GetCameraPosition();
		auto& bricks = m_pVolume->GetNearestTimestep().bricks;
		for(auto it = bricks.cbegin(); it != bricks.cend(); ++it)
		{
			const TimeVolumeIO::Brick* pBrick = &(*it);
			bricksCurTimestep.push_back(BrickSortItem(pBrick, GetDistance(volume, pBrick->GetSpatialIndex(), camPos)));
		}
		std::sort(bricksCurTimestep.begin(), bricksCurTimestep.end());

		// walk brick list, collect bricks to render
		for(size_t brickIndex = 0; brickIndex < bricksCurTimestep.size(); brickIndex++)
		{
			// get brick
			const TimeVolumeIO::Brick* pBrick = bricksCurTimestep[brickIndex].pBrick;

			Vec3f boxMin, boxMax;
			m_pVolume->GetBrickBoxWorld(pBrick->GetSpatialIndex(), boxMin, boxMax);

			bool clipped = false;

			// check if brick is clipped against clip box
			clipped = clipped || IsBoxClippedAgainstBox(m_raycastParams.m_clipBoxMin, m_raycastParams.m_clipBoxMax, boxMin, boxMax);

			// check if brick is outside view frustum
			for(uint i = 0; i < 6; i++)
			{
				bool planeClipped = IsBoxClippedAgainstPlane(frustumPlanes[i], boxMin, boxMax);
				if(m_stereoParams.m_stereoEnabled)
				{
					planeClipped = planeClipped && IsBoxClippedAgainstPlane(frustumPlanes2[i], boxMin, boxMax);
				}
				clipped = clipped || planeClipped;
			}

			if(clipped)
			{
				m_bricksClipped.push_back(pBrick);
			}
			else
			{
				m_bricksToRender.push_back(pBrick);
			}
		}
	}

	// now we can update the list of bricks to load
	UpdateBricksToLoad();
	

	// clear rendertargets and depth buffer
	ClearResult();

	// render opaque stuff immediately
	RenderBoxes(true, false);
	if(m_particleRenderParams.m_linesEnabled)
	{
		for(size_t i = 0; i < pLineBuffers.size(); i++)
		{
			RenderLines(pLineBuffers[i], true, false);
		}
	}
	for(size_t i = 0; i < pBallBuffers.size(); i++)
	{
		RenderBalls(pBallBuffers[i], ballRadius);
	}

	//Don't do it here. It is now in RenderLines, so that the particles are drawn correctly including the transparency
	//RenderSliceTexture();

	// if there's nothing to raycast, we're finished now
	if(m_bricksToRender.empty())
	{
		CancelRendering();
		return STATE_DONE;
	}


	// start timing
	m_timerRender.Start();

	m_timerUploadDecompress.ResetTimers();
	m_timerComputeMeasure.ResetTimers();
	m_timerRaycast.ResetTimers();
	m_timerCompressDownload.ResetTimers();


	return STATE_RENDERING;
}

RenderingManager::eRenderState RenderingManager::Render()
{
	if(!IsRendering()) return STATE_ERROR;

	assert(m_nextBrickToRender < m_bricksToRender.size());

	RenderBricks(true);

	bool finished = (m_nextBrickToRender == m_bricksToRender.size());

	if(finished)
	{
		if(m_pTfArray)
		{
			cudaSafeCall(cudaFreeArray(m_pTfArray));
			m_pTfArray = nullptr;
		}
		m_bricksToLoad.clear();
		m_pFilteredVolumes = nullptr;
		m_pVolume = nullptr;

		// collect and print timings
		UpdateTimings();

		// print timings
		int device = -1;
		cudaSafeCall(cudaGetDevice(&device));
		printf("Device %i: Done rendering in %.2f s.\n", device, m_timings.RenderWall / 1000.0f);
		if(m_timings.UploadDecompressGPU.Count  > 0)
		{
			printf("Device %i: Upload/Decompress (GPU): %.2f ms (%.2f-%.2f-%.2f : %u)\n", device,
				m_timings.UploadDecompressGPU.Total,
				m_timings.UploadDecompressGPU.Min, m_timings.UploadDecompressGPU.Avg, m_timings.UploadDecompressGPU.Max,
				m_timings.UploadDecompressGPU.Count);
		}
		if(m_timings.ComputeMeasureGPU.Count  > 0)
		{
			printf("Device %i: ComputeMeasure (GPU): %.2f ms (%.2f-%.2f-%.2f : %u)\n", device,
				m_timings.ComputeMeasureGPU.Total,
				m_timings.ComputeMeasureGPU.Min, m_timings.ComputeMeasureGPU.Avg, m_timings.ComputeMeasureGPU.Max,
				m_timings.ComputeMeasureGPU.Count);
		}
		if(m_timings.RaycastGPU.Count  > 0)
		{
			printf("Device %i: Raycast (GPU): %.2f ms (%.2f-%.2f-%.2f : %u)\n", device,
				m_timings.RaycastGPU.Total,
				m_timings.RaycastGPU.Min, m_timings.RaycastGPU.Avg, m_timings.RaycastGPU.Max,
				m_timings.RaycastGPU.Count);
		}
		if(m_timings.CompressDownloadGPU.Count  > 0)
		{
			printf("Device %i: Compress/Download (GPU): %.2f ms (%.2f-%.2f-%.2f : %u)\n", device,
				m_timings.CompressDownloadGPU.Total,
				m_timings.CompressDownloadGPU.Min, m_timings.CompressDownloadGPU.Avg, m_timings.CompressDownloadGPU.Max,
				m_timings.CompressDownloadGPU.Count);
		}
		printf("\n");

		return STATE_DONE;
	}

	return STATE_RENDERING;
}

void RenderingManager::CancelRendering()
{
	if(m_pTfArray)
	{
		cudaSafeCall(cudaFreeArray(m_pTfArray));
		m_pTfArray = nullptr;
	}
	m_bricksToLoad.clear();
	m_pFilteredVolumes = nullptr;
	m_pVolume = nullptr;
}

bool RenderingManager::IsRendering() const
{
	return m_pVolume != nullptr;
}

float RenderingManager::GetRenderingProgress() const
{
	if(m_bricksToRender.empty())
	{
		return 1.0f;
	}

	return float(m_nextBrickToRender) / float(m_bricksToRender.size());
}

void RenderingManager::ClearResult()
{
	if(m_pDevice)
	{
		ID3D11DeviceContext* pContext = nullptr;
		m_pDevice->GetImmediateContext(&pContext);

		// clear rendertargets and depth buffer
		float black[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
		if(m_pOpaqueRTV)
		{
			pContext->ClearRenderTargetView(m_pOpaqueRTV, black);
		}
		if(m_pRaycastRTV)
		{
			pContext->ClearRenderTargetView(m_pRaycastRTV, black);
		}
		if(m_pDepthDSV)
		{
			pContext->ClearDepthStencilView(m_pDepthDSV, D3D11_CLEAR_DEPTH, 1.0f, 0);
			pContext->CopyResource(m_pDepthTexCopy, m_pDepthTex);
		}

		pContext->Release();
	}
	else
	{
		clearCudaArray2Duchar4(m_pRaycastArray, m_projectionParams.GetImageWidth(m_range), m_projectionParams.GetImageHeight(m_range));
		//TODO clear DepthArray?
	}
}

void RenderingManager::ReleaseResources()
{
	CancelRendering();

	ReleaseMeasureBrickSlots();
	ReleaseVolumeDependentResources();
}


void RenderingManager::UpdateBricksToLoad()
{
	m_bricksToLoad.clear();

	// first add bricks we still want to render
	m_bricksToLoad.insert(m_bricksToLoad.end(), m_bricksToRender.begin() + m_nextBrickToRender, m_bricksToRender.end());
	// then bricks we have already rendered
	m_bricksToLoad.insert(m_bricksToLoad.end(), m_bricksToRender.begin(), m_bricksToRender.begin() + m_nextBrickToRender);
	// then bricks that were clipped
	m_bricksToLoad.insert(m_bricksToLoad.end(), m_bricksClipped.begin(), m_bricksClipped.end());

	//TODO prefetching of next timestep
}


void RenderingManager::RenderBoxes(bool enableColor, bool blendBehind)
{
	if(!m_pDevice) return;

	ID3D11DeviceContext* pContext = nullptr;
	m_pDevice->GetImmediateContext(&pContext);

	Vec4f domainBoxColor(0.0f, 0.0f, 1.0f, 1.0f);
	Vec4f brickBoxColor (0.0f, 0.5f, 1.0f, 1.0f);
	Vec4f clipBoxColor  (1.0f, 0.0f, 0.0f, 1.0f);
	Vec4f seedBoxColor  (0.0f, 0.6f, 0.0f, 1.0f);
	Vec4f coordinateBoxColor(1.0f, 0.0f, 0.0f, 1.0f);

	Vec3f lightPos = m_viewParams.GetCameraPosition();

	Vec3f volumeHalfSizeWorld = m_pVolume->GetVolumeHalfSizeWorld();
	Vec3f seedBoxMin = m_particleTraceParams.m_seedBoxMin;
	Vec3f seedBoxMax = m_particleTraceParams.m_seedBoxMin + m_particleTraceParams.m_seedBoxSize;
	Vec3f clipBoxMin = m_raycastParams.m_clipBoxMin;
	Vec3f clipBoxMax = m_raycastParams.m_clipBoxMax;

	// save viewports and render targets
	uint oldViewportCount = D3D11_VIEWPORT_AND_SCISSORRECT_OBJECT_COUNT_PER_PIPELINE;
	D3D11_VIEWPORT oldViewports[D3D11_VIEWPORT_AND_SCISSORRECT_OBJECT_COUNT_PER_PIPELINE];
	pContext->RSGetViewports(&oldViewportCount, oldViewports);
	ID3D11RenderTargetView* ppOldRTVs[D3D11_SIMULTANEOUS_RENDER_TARGET_COUNT];
	ID3D11DepthStencilView* pOldDSV;
	pContext->OMGetRenderTargets(D3D11_SIMULTANEOUS_RENDER_TARGET_COUNT, ppOldRTVs, &pOldDSV);

	// set our render target
	// if color is disabled, don't set a render target
	pContext->OMSetRenderTargets(enableColor ? 1 : 0, &m_pOpaqueRTV, m_pDepthDSV);

	// build viewport
	D3D11_VIEWPORT viewport = {};
	viewport.TopLeftX = float(0);
	viewport.TopLeftY = float(0);
	viewport.Width    = float(m_projectionParams.GetImageWidth(m_range));
	viewport.Height   = float(m_projectionParams.m_imageHeight);
	viewport.MinDepth = 0.0f;
	viewport.MaxDepth = 1.0f;

	bool renderDomainBox = m_renderDomainBox;
	bool renderClipBox = m_renderClipBox && m_raycastParams.m_raycastingEnabled;
	bool renderSeedBox = m_renderSeedBox && m_particleRenderParams.m_linesEnabled;
	bool renderBrickBoxes = m_renderBrickBoxes;
	bool renderCoordinates = m_renderDomainBox;

	Vec3f brickSize = m_pVolume->GetBrickSizeWorld();
	float tubeRadiusLarge  = 0.004f;
	float tubeRadiusMedium = 0.003f;
	float tubeRadiusSmall  = 0.002f;
	if(m_stereoParams.m_stereoEnabled)
	{
		Mat4f viewLeft  = m_viewParams.BuildViewMatrix(EYE_LEFT,  m_stereoParams.m_eyeDistance);
		Mat4f viewRight = m_viewParams.BuildViewMatrix(EYE_RIGHT, m_stereoParams.m_eyeDistance);
		Mat4f projLeft  = m_projectionParams.BuildProjectionMatrix(EYE_LEFT,  m_stereoParams.m_eyeDistance, m_range);
		Mat4f projRight = m_projectionParams.BuildProjectionMatrix(EYE_RIGHT, m_stereoParams.m_eyeDistance, m_range);

		viewport.Height /= 2.0f;
		pContext->RSSetViewports(1, &viewport);

		if(renderDomainBox)	m_box.RenderLines(viewLeft, projLeft, lightPos, -volumeHalfSizeWorld, volumeHalfSizeWorld, domainBoxColor, tubeRadiusLarge,  blendBehind);
		if(renderClipBox)	m_box.RenderLines(viewLeft, projLeft, lightPos, clipBoxMin,           clipBoxMax,          clipBoxColor,   tubeRadiusMedium, blendBehind);
		if(renderSeedBox)	m_box.RenderLines(viewLeft, projLeft, lightPos, seedBoxMin,           seedBoxMax,          seedBoxColor,   tubeRadiusMedium, blendBehind);
		if (renderCoordinates) {
			m_box.RenderLines(viewLeft, projLeft, lightPos, -volumeHalfSizeWorld - 2 * tubeRadiusLarge, -volumeHalfSizeWorld + 2 * tubeRadiusLarge, coordinateBoxColor, tubeRadiusLarge, blendBehind);
			m_box.RenderLines(viewLeft, projLeft, lightPos, -volumeHalfSizeWorld, -volumeHalfSizeWorld + Vec3f(volumeHalfSizeWorld.x() / 4, 0.0f, 0.0f), coordinateBoxColor, tubeRadiusLarge*1.5f, blendBehind);
		}
		if(renderBrickBoxes) m_box.RenderBrickLines(viewLeft, projLeft, lightPos, -volumeHalfSizeWorld, volumeHalfSizeWorld, brickBoxColor, brickSize, tubeRadiusSmall, blendBehind);

		viewport.TopLeftY += viewport.Height;
		pContext->RSSetViewports(1, &viewport);

		if(renderDomainBox)	m_box.RenderLines(viewRight, projRight, lightPos, -volumeHalfSizeWorld, volumeHalfSizeWorld, domainBoxColor, tubeRadiusLarge,  blendBehind);
		if(renderClipBox)	m_box.RenderLines(viewRight, projRight, lightPos, clipBoxMin,           clipBoxMax,          clipBoxColor,   tubeRadiusMedium, blendBehind);
		if(renderSeedBox)	m_box.RenderLines(viewRight, projRight, lightPos, seedBoxMin,           seedBoxMax,          seedBoxColor,   tubeRadiusMedium, blendBehind);
		if (renderCoordinates) {
			m_box.RenderLines(viewRight, projRight, lightPos, -volumeHalfSizeWorld - 2 * tubeRadiusLarge, -volumeHalfSizeWorld + 2 * tubeRadiusLarge, coordinateBoxColor, tubeRadiusLarge, blendBehind);
			m_box.RenderLines(viewRight, projRight, lightPos, -volumeHalfSizeWorld, -volumeHalfSizeWorld + Vec3f(volumeHalfSizeWorld.x() / 4, 0.0f, 0.0f), coordinateBoxColor, tubeRadiusLarge*1.5f, blendBehind);
		}
		if(renderBrickBoxes) m_box.RenderBrickLines(viewRight, projRight, lightPos, -volumeHalfSizeWorld, volumeHalfSizeWorld, brickBoxColor, brickSize, tubeRadiusSmall, blendBehind);
	}
	else
	{
		Mat4f view = m_viewParams.BuildViewMatrix(EYE_CYCLOP, 0.0f);
		Mat4f proj = m_projectionParams.BuildProjectionMatrix(EYE_CYCLOP, 0.0f, m_range);

		pContext->RSSetViewports(1, &viewport);

		if(renderDomainBox)	m_box.RenderLines(view, proj, lightPos, -volumeHalfSizeWorld, volumeHalfSizeWorld, domainBoxColor, tubeRadiusLarge,  blendBehind);
		if(renderClipBox)	m_box.RenderLines(view, proj, lightPos, clipBoxMin,           clipBoxMax,          clipBoxColor,   tubeRadiusMedium, blendBehind);
		if(renderSeedBox)	m_box.RenderLines(view, proj, lightPos, seedBoxMin,           seedBoxMax,          seedBoxColor,   tubeRadiusMedium, blendBehind);
		if (renderCoordinates) {
			m_box.RenderLines(view, proj, lightPos, -volumeHalfSizeWorld - 2 * tubeRadiusLarge, -volumeHalfSizeWorld + 2 * tubeRadiusLarge, coordinateBoxColor, tubeRadiusLarge, blendBehind);
			m_box.RenderLines(view, proj, lightPos, -volumeHalfSizeWorld, -volumeHalfSizeWorld + Vec3f(volumeHalfSizeWorld.x() / 4, 0.0f, 0.0f), coordinateBoxColor, tubeRadiusLarge*1.5f, blendBehind);
		}
		if(renderBrickBoxes) m_box.RenderBrickLines(view, proj, lightPos, -volumeHalfSizeWorld, volumeHalfSizeWorld, brickBoxColor, brickSize, tubeRadiusSmall, blendBehind);
	}

	// restore viewports and render targets
	pContext->OMSetRenderTargets(D3D11_SIMULTANEOUS_RENDER_TARGET_COUNT, ppOldRTVs, pOldDSV);
	pContext->RSSetViewports(oldViewportCount, oldViewports);
	for(uint i = 0; i < D3D11_SIMULTANEOUS_RENDER_TARGET_COUNT; i++)
	{
		SAFE_RELEASE(ppOldRTVs[i]);
	}
	SAFE_RELEASE(pOldDSV);

	// copy z buffer into cuda-compatible texture
	pContext->CopyResource(m_pDepthTexCopy, m_pDepthTex);

	pContext->Release();
}


void DebugRenderLines(ID3D11Device* device, ID3D11DeviceContext* context, const LineBuffers* pLineBuffers)
{
	if (pLineBuffers->m_indexCountTotal == 0) {
		printf("no vertices to draw\n");
		return;
	}

	//Create staging buffers
	HRESULT hr;
	D3D11_BUFFER_DESC bufDesc = {};

	ID3D11Buffer* vbCopy = NULL;
	ID3D11Buffer* ibCopy = NULL;

	bufDesc.BindFlags = D3D11_BIND_UNORDERED_ACCESS; //D3D11_BIND_VERTEX_BUFFER;
	bufDesc.ByteWidth = pLineBuffers->m_lineCount * pLineBuffers->m_lineLengthMax * sizeof(LineVertex);
	bufDesc.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
	bufDesc.Usage = D3D11_USAGE_DEFAULT;
	if (FAILED(hr = device->CreateBuffer(&bufDesc, nullptr, &vbCopy)))
	{
		printf("unable to create vertex buffer for copying data to the cpu\n");
		SAFE_RELEASE(vbCopy);
		SAFE_RELEASE(ibCopy);
		return;
	}

	bufDesc.BindFlags = D3D11_BIND_UNORDERED_ACCESS; //D3D11_BIND_INDEX_BUFFER;
	bufDesc.ByteWidth = pLineBuffers->m_lineCount * (pLineBuffers->m_lineLengthMax - 1) * 2 * sizeof(uint);
	bufDesc.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
	bufDesc.Usage = D3D11_USAGE_DEFAULT;
	if (FAILED(hr = device->CreateBuffer(&bufDesc, nullptr, &ibCopy)))
	{
		printf("unable to create vertex buffer for copying data to the cpu\n");
		SAFE_RELEASE(vbCopy);
		SAFE_RELEASE(ibCopy);
		return;
	}

	context->CopyResource(vbCopy, pLineBuffers->m_pVB);
	context->CopyResource(ibCopy, pLineBuffers->m_pIB);

	//copy to cpu
	D3D11_MAPPED_SUBRESOURCE ms;
	hr = context->Map(vbCopy, 0, D3D11_MAP_READ, 0, &ms);
	std::vector<LineVertex> vertexBuffer;
	std::vector<uint> indexBuffer;
	bool mapped = true;
	if (FAILED(hr)) {
		printf("unable to map vertex buffer\n");
		mapped = false;
	}
	else {
		vertexBuffer.resize(pLineBuffers->m_lineCount * pLineBuffers->m_lineLengthMax);
		memcpy(&vertexBuffer[0], ms.pData, sizeof(LineVertex) * vertexBuffer.size());
		context->Unmap(vbCopy, 0);
	}
	hr = context->Map(ibCopy, 0, D3D11_MAP_READ, 0, &ms);
	if (FAILED(hr)) {
		printf("unable to map index buffer\n");
		mapped = false;
	}
	else {
		indexBuffer.resize(pLineBuffers->m_indexCountTotal);
		memcpy(&indexBuffer[0], ms.pData, sizeof(uint) * indexBuffer.size());
		context->Unmap(ibCopy, 0);
	}

	//print vertices
	if (mapped) {
		for (uint i = 0; i < pLineBuffers->m_indexCountTotal; ++i) {
			uint j = indexBuffer[i];
			const LineVertex& v = vertexBuffer[j];
			printf("%d->%d: pos=(%f, %f, %f), seed pos=(%f, %f, %f)\n", i, j, 
				v.Position.x, v.Position.y, v.Position.z, v.SeedPosition.x, v.SeedPosition.y, v.SeedPosition.z);
		}
	}

	//release staging resource
	SAFE_RELEASE(vbCopy);
	SAFE_RELEASE(ibCopy);
}

void RenderingManager::RenderLines(LineBuffers* pLineBuffers, bool enableColor, bool blendBehind)
{
	ID3D11DeviceContext* pContext = nullptr;
	m_pDevice->GetImmediateContext(&pContext);

	// build viewport
	D3D11_VIEWPORT viewport = {};
	viewport.TopLeftX = float(0);
	viewport.TopLeftY = float(0);
	viewport.Width    = float(m_projectionParams.GetImageWidth(m_range));
	viewport.Height   = float(m_projectionParams.m_imageHeight);
	viewport.MinDepth = 0.0f;
	viewport.MaxDepth = 1.0f;

	// Debug
	//DebugRenderLines(m_pDevice, pContext, pLineBuffers);

	// common shader vars
	m_lineEffect.m_pvLightPosVariable->SetFloatVector(m_viewParams.GetCameraPosition());
	m_lineEffect.m_pfRibbonHalfWidthVariable->SetFloat(0.01f * m_particleRenderParams.m_ribbonWidth * 0.5f);
	m_lineEffect.m_pfTubeRadiusVariable->SetFloat(0.01f * m_particleRenderParams.m_tubeRadius);
	m_lineEffect.m_pbTubeRadiusFromVelocityVariable->SetBool(m_particleRenderParams.m_tubeRadiusFromVelocity);
	m_lineEffect.m_pfReferenceVelocityVariable->SetFloat(m_particleRenderParams.m_referenceVelocity);
	m_lineEffect.m_pvHalfSizeWorldVariable->SetFloatVector(m_pVolume->GetVolumeHalfSizeWorld());

	m_lineEffect.m_piColorMode->SetInt(m_particleRenderParams.m_lineColorMode);
	m_lineEffect.m_pvColor0Variable->SetFloatVector(m_particleRenderParams.m_color0);
	m_lineEffect.m_pvColor1Variable->SetFloatVector(m_particleRenderParams.m_color1);
	float timeMin = (m_particleTraceParams.m_lineMode == LINE_STREAM) ? 0.0f : m_pVolume->GetCurTime();
	m_lineEffect.m_pfTimeMinVariable->SetFloat(timeMin);
	m_lineEffect.m_pfTimeMaxVariable->SetFloat(timeMin + m_particleTraceParams.m_lineAgeMax);

	m_lineEffect.m_pbTimeStripesVariable->SetBool(m_particleRenderParams.m_timeStripes);
	m_lineEffect.m_pfTimeStripeLengthVariable->SetFloat(m_particleRenderParams.m_timeStripeLength);

	m_lineEffect.m_pfParticleTransparencyVariable->SetFloat(m_particleRenderParams.m_particleTransparency);
	m_lineEffect.m_pfParticleSizeVariable->SetFloat(0.01f * m_particleRenderParams.m_particleSize);
	float aspectRatio = viewport.Width / viewport.Height;
	m_lineEffect.m_pfScreenAspectRatioVariable->SetFloat(aspectRatio);

	m_lineEffect.m_ptexColors->SetResource(m_pRandomColorsSRV);
	if (m_particleRenderParams.m_pColorTexture != nullptr) {
		m_lineEffect.m_pseedColors->SetResource(m_particleRenderParams.m_pColorTexture);
	}

	m_lineEffect.m_piMeasureMode->SetInt((int)m_particleRenderParams.m_measure);
	m_lineEffect.m_pfMeasureScale->SetFloat(m_particleRenderParams.m_measureScale);
	if (m_particleRenderParams.m_pTransferFunction != nullptr) {
		m_lineEffect.m_ptransferFunction->SetResource(m_particleRenderParams.m_pTransferFunction);
	}
	Vec2f tfRange(m_particleRenderParams.m_transferFunctionRangeMin, m_particleRenderParams.m_transferFunctionRangeMax);
	m_lineEffect.m_pvTfRange->SetFloatVector(tfRange);

	// common slice texture parameters
	bool renderSlice = m_particleRenderParams.m_showSlice 
		&& m_particleRenderParams.m_pSliceTexture != nullptr;
	tum3D::Vec4f clipPlane;
	if (renderSlice) {
		Vec3f volumeHalfSizeWorld = m_pVolume->GetVolumeHalfSizeWorld();
		tum3D::Vec3f normal(0, 0, 1);
		tum3D::Vec2f size(volumeHalfSizeWorld.x() * 2, volumeHalfSizeWorld.y() * 2);
		tum3D::Vec3f center(0, 0, m_particleRenderParams.m_slicePosition);
		m_pQuadEffect->SetParameters(m_particleRenderParams.m_pSliceTexture,
			center, normal, size, m_particleRenderParams.m_sliceAlpha);
		clipPlane.set(normal.x(), normal.y(), normal.z(), m_particleRenderParams.m_slicePosition);
		//test if we have to flip the clip plane if the camera is at the wrong side
		Mat4f view = m_viewParams.BuildViewMatrix(EYE_CYCLOP, 0.0f);
		Mat4f proj = m_projectionParams.BuildProjectionMatrix(EYE_CYCLOP, 0.0f, m_range);
		Mat4f viewproj = proj * view;
		Vec4f v1; v1 = viewproj.multVec(Vec4f(-1, -1, center.z(), 1), v1); v1 /= v1.w();
		Vec4f v2; v2 = viewproj.multVec(Vec4f(+1, -1, center.z(), 1), v2); v2 /= v2.w();
		Vec4f v3; v3 = viewproj.multVec(Vec4f(-1, +1, center.z(), 1), v3); v3 /= v3.w();
		Vec2f dir1 = v2.xy() - v1.xy();
		Vec2f dir2 = v3.xy() - v1.xy();
		if (dir1.x()*dir2.y() - dir1.y()*dir2.x() > 0) {
			//camera is at the wrong side, flip clip
			clipPlane = -clipPlane;
		}
	}

	//Check if particles should be rendered
	if (m_particleRenderParams.m_lineRenderMode == LINE_RENDER_PARTICLES) {
		SortParticles(pLineBuffers, pContext);
		if (!blendBehind) {
			if (renderSlice) {
				//render particles below, and then the slice
				RenderParticles(pLineBuffers, pContext, viewport, &clipPlane, true);
				//invert clip plane and render particles above
				clipPlane = -clipPlane;
				RenderParticles(pLineBuffers, pContext, viewport, &clipPlane, false);
			}
			else {
				RenderParticles(pLineBuffers, pContext, viewport);
			}
		}
		// copy z buffer into CUDA-compatible texture
		pContext->CopyResource(m_pDepthTexCopy, m_pDepthTex);
		pContext->Release();
		return;
	}

	// IA
	pContext->IASetInputLayout(m_lineEffect.m_pInputLayout);
	UINT stride = sizeof(LineVertex);
	UINT offset = 0;
	pContext->IASetVertexBuffers(0, 1, &pLineBuffers->m_pVB, &stride, &offset);
	pContext->IASetIndexBuffer(pLineBuffers->m_pIB, DXGI_FORMAT_R32_UINT, 0);
	pContext->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_LINELIST);

	// save viewports and render targets
	uint oldViewportCount = D3D11_VIEWPORT_AND_SCISSORRECT_OBJECT_COUNT_PER_PIPELINE;
	D3D11_VIEWPORT oldViewports[D3D11_VIEWPORT_AND_SCISSORRECT_OBJECT_COUNT_PER_PIPELINE];
	pContext->RSGetViewports(&oldViewportCount, oldViewports);
	ID3D11RenderTargetView* ppOldRTVs[D3D11_SIMULTANEOUS_RENDER_TARGET_COUNT];
	ID3D11DepthStencilView* pOldDSV;
	pContext->OMGetRenderTargets(D3D11_SIMULTANEOUS_RENDER_TARGET_COUNT, ppOldRTVs, &pOldDSV);

	// set our render target
	// if color is disabled, set no RTV - render depth only
	pContext->OMSetRenderTargets(enableColor ? 1 : 0, &m_pOpaqueRTV, m_pDepthDSV);

	uint pass = 2 * uint(m_particleRenderParams.m_lineRenderMode);
	if(blendBehind) pass++;

	if(m_stereoParams.m_stereoEnabled)
	{
		Mat4f viewLeft  = m_viewParams.BuildViewMatrix(EYE_LEFT,  m_stereoParams.m_eyeDistance);
		Mat4f viewRight = m_viewParams.BuildViewMatrix(EYE_RIGHT, m_stereoParams.m_eyeDistance);
		Mat4f projLeft  = m_projectionParams.BuildProjectionMatrix(EYE_LEFT,  m_stereoParams.m_eyeDistance, m_range);
		Mat4f projRight = m_projectionParams.BuildProjectionMatrix(EYE_RIGHT, m_stereoParams.m_eyeDistance, m_range);

		viewport.Height /= 2.0f;
		pContext->RSSetViewports(1, &viewport);

		m_lineEffect.m_pmWorldViewProjVariable->SetMatrix(projLeft * viewLeft);
		m_lineEffect.m_pTechnique->GetPassByIndex(pass)->Apply(0, pContext);

		pContext->DrawIndexed(pLineBuffers->m_indexCountTotal, 0, 0);
		if (renderSlice) {
			m_pQuadEffect->DrawTexture(projLeft * viewLeft, pContext, true);
		}

		viewport.TopLeftY += viewport.Height;
		pContext->RSSetViewports(1, &viewport);

		m_lineEffect.m_pmWorldViewProjVariable->SetMatrix(projRight * viewRight);
		m_lineEffect.m_pTechnique->GetPassByIndex(pass)->Apply(0, pContext);

		pContext->DrawIndexed(pLineBuffers->m_indexCountTotal, 0, 0);
		if (renderSlice) {
			m_pQuadEffect->DrawTexture(projRight * viewRight, pContext, true);
		}
	}
	else
	{
		Mat4f view = m_viewParams.BuildViewMatrix(EYE_CYCLOP, 0.0f);
		Mat4f proj = m_projectionParams.BuildProjectionMatrix(EYE_CYCLOP, 0.0f, m_range);

		pContext->RSSetViewports(1, &viewport);

		m_lineEffect.m_pmWorldViewProjVariable->SetMatrix(proj * view);
		m_lineEffect.m_pTechnique->GetPassByIndex(pass)->Apply(0, pContext);

		pContext->DrawIndexed(pLineBuffers->m_indexCountTotal, 0, 0);
		if (renderSlice) {
			m_pQuadEffect->DrawTexture(proj * view, pContext, true);
		}
	}

	// restore viewports and render targets
	pContext->OMSetRenderTargets(D3D11_SIMULTANEOUS_RENDER_TARGET_COUNT, ppOldRTVs, pOldDSV);
	pContext->RSSetViewports(oldViewportCount, oldViewports);
	for(uint i = 0; i < D3D11_SIMULTANEOUS_RENDER_TARGET_COUNT; i++)
	{
		SAFE_RELEASE(ppOldRTVs[i]);
	}
	SAFE_RELEASE(pOldDSV);

	// copy z buffer into CUDA-compatible texture
	pContext->CopyResource(m_pDepthTexCopy, m_pDepthTex);

	pContext->Release();
}


__global__ void FillVertexDepth(const LineVertex* vertices, const uint* indices, float* depthOut, float4 vec, uint maxIndex)
{
	uint index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index > maxIndex) return;

	float4 pos = make_float4(vertices[indices[index]].Position, 1);
	depthOut[index] = -dot(pos, vec);
}

void RenderingManager::SortParticles(LineBuffers* pLineBuffers, ID3D11DeviceContext* pContext)
{
	//compute matrix
	Mat4f view = m_viewParams.BuildViewMatrix(EYE_CYCLOP, 0.0f);
	Mat4f proj = m_projectionParams.BuildProjectionMatrix(EYE_CYCLOP, 0.0f, m_range);
	Mat4f projView = proj * view;
	//perform the inner product of the following vec4 with vec4(vertex.pos, 1) to compute the depth
	float4 depthMultiplier = make_float4(projView.get(2, 0), projView.get(2, 1), projView.get(2, 2), projView.get(2, 3));

	//aquire resources
	cudaSafeCall(cudaGraphicsMapResources(1, &pLineBuffers->m_pIBCuda));
	uint* dpIB = nullptr;
	cudaSafeCall(cudaGraphicsResourceGetMappedPointer((void**)&dpIB, nullptr, pLineBuffers->m_pIBCuda));
	cudaSafeCall(cudaGraphicsMapResources(1, &pLineBuffers->m_pIBCuda_sorted));
	uint* dpIB_sorted = nullptr;
	cudaSafeCall(cudaGraphicsResourceGetMappedPointer((void**)&dpIB_sorted, nullptr, pLineBuffers->m_pIBCuda_sorted));

	//copy indices
	cudaMemcpy(dpIB_sorted, dpIB, sizeof(uint) * pLineBuffers->m_indexCountTotal, cudaMemcpyDeviceToDevice);

	if (m_particleRenderParams.m_sortParticles) {
		//aquired vertex data
		cudaSafeCall(cudaGraphicsMapResources(1, &pLineBuffers->m_pVBCuda));
		LineVertex* dpVB = nullptr;
		cudaSafeCall(cudaGraphicsResourceGetMappedPointer((void**)&dpVB, nullptr, pLineBuffers->m_pVBCuda));

		//compute depth
		float* depth;
		cudaSafeCall(cudaMalloc(&depth, sizeof(float) * pLineBuffers->m_indexCountTotal)); //todo: cache this array
		uint blockSize = 128;
		uint blockCount = (pLineBuffers->m_indexCountTotal + blockSize - 1) / blockSize;
		FillVertexDepth <<< blockCount, blockSize >>> (dpVB, dpIB, depth, depthMultiplier, pLineBuffers->m_indexCountTotal);

		//sort indices 
		thrust::device_ptr<float> thrustKey(depth);
		thrust::device_ptr<uint> thrustValue(dpIB_sorted);
		thrust::sort_by_key(thrustKey, thrustKey + pLineBuffers->m_indexCountTotal, thrustValue);

		//release vertex data
		cudaSafeCall(cudaFree(depth));
		cudaSafeCall(cudaGraphicsUnmapResources(1, &pLineBuffers->m_pVBCuda));
	}
	else {
		
	}

	//release resources
	cudaSafeCall(cudaGraphicsUnmapResources(1, &pLineBuffers->m_pIBCuda));
	cudaSafeCall(cudaGraphicsUnmapResources(1, &pLineBuffers->m_pIBCuda_sorted));
}


void RenderingManager::RenderParticles(const LineBuffers* pLineBuffers, 
	ID3D11DeviceContext* pContext, D3D11_VIEWPORT viewport, 
	const tum3D::Vec4f* clipPlane, bool renderSlice)
{
	// IA
	pContext->IASetInputLayout(m_lineEffect.m_pInputLayout);
	UINT stride = sizeof(LineVertex);
	UINT offset = 0;
	pContext->IASetVertexBuffers(0, 1, &pLineBuffers->m_pVB, &stride, &offset);
	pContext->IASetIndexBuffer(pLineBuffers->m_pIB_sorted, DXGI_FORMAT_R32_UINT, 0);
	pContext->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_POINTLIST);

	//set rotation, needed to compute the correct transformation
	m_lineEffect.m_pmWorldViewRotation->SetMatrix(m_viewParams.BuildRotationMatrix());

	//set clip plane
	if (clipPlane == NULL) {
		tum3D::Vec4f clip(0, 0, 0, 0);
		m_lineEffect.m_pvParticleClipPlane->SetFloatVector(clip);
	}
	else {
		m_lineEffect.m_pvParticleClipPlane->SetFloatVector(*clipPlane);
	}

	// save viewports and render targets
	uint oldViewportCount = D3D11_VIEWPORT_AND_SCISSORRECT_OBJECT_COUNT_PER_PIPELINE;
	D3D11_VIEWPORT oldViewports[D3D11_VIEWPORT_AND_SCISSORRECT_OBJECT_COUNT_PER_PIPELINE];
	pContext->RSGetViewports(&oldViewportCount, oldViewports);
	ID3D11RenderTargetView* ppOldRTVs[D3D11_SIMULTANEOUS_RENDER_TARGET_COUNT];
	ID3D11DepthStencilView* pOldDSV;
	pContext->OMGetRenderTargets(D3D11_SIMULTANEOUS_RENDER_TARGET_COUNT, ppOldRTVs, &pOldDSV);

	//specify path
	int pass = 6;
	float clearColorSingle = 0;
	if (m_particleRenderParams.m_particleRenderMode == PARTICLE_RENDER_MULTIPLICATIVE) {
		pass = 7;
		clearColorSingle = 1;
	}
	else if (m_particleRenderParams.m_particleRenderMode == PARTICLE_RENDER_ALPHA) {
		pass = 8;
	}
	// set transparent offscreen target
	float clearColor[4] = { clearColorSingle, clearColorSingle, clearColorSingle, 0.0f };
	pContext->ClearRenderTargetView(m_pTransparentRTV, clearColor);
	pContext->OMSetRenderTargets(1, &m_pTransparentRTV, m_pDepthDSV);

	//render
	if (m_stereoParams.m_stereoEnabled)
	{
		Mat4f viewLeft = m_viewParams.BuildViewMatrix(EYE_LEFT, m_stereoParams.m_eyeDistance);
		Mat4f viewRight = m_viewParams.BuildViewMatrix(EYE_RIGHT, m_stereoParams.m_eyeDistance);
		Mat4f projLeft = m_projectionParams.BuildProjectionMatrix(EYE_LEFT, m_stereoParams.m_eyeDistance, m_range);
		Mat4f projRight = m_projectionParams.BuildProjectionMatrix(EYE_RIGHT, m_stereoParams.m_eyeDistance, m_range);

		viewport.Height /= 2.0f;
		pContext->RSSetViewports(1, &viewport);

		m_lineEffect.m_pmWorldViewProjVariable->SetMatrix(projLeft * viewLeft);
		m_lineEffect.m_pTechnique->GetPassByIndex(pass)->Apply(0, pContext);

		pContext->DrawIndexed(pLineBuffers->m_indexCountTotal, 0, 0);
		if (renderSlice) {
			m_pQuadEffect->DrawTexture(projLeft * viewLeft, pContext, false);
		}

		viewport.TopLeftY += viewport.Height;
		pContext->RSSetViewports(1, &viewport);

		m_lineEffect.m_pmWorldViewProjVariable->SetMatrix(projRight * viewRight);
		m_lineEffect.m_pTechnique->GetPassByIndex(pass)->Apply(0, pContext);

		pContext->DrawIndexed(pLineBuffers->m_indexCountTotal, 0, 0);
		if (renderSlice) {
			m_pQuadEffect->DrawTexture(projRight * viewRight, pContext, false);
		}
	}
	else
	{
		Mat4f view = m_viewParams.BuildViewMatrix(EYE_CYCLOP, 0.0f);
		Mat4f proj = m_projectionParams.BuildProjectionMatrix(EYE_CYCLOP, 0.0f, m_range);

		pContext->RSSetViewports(1, &viewport);

		m_lineEffect.m_pmWorldViewProjVariable->SetMatrix(proj * view);
		m_lineEffect.m_pTechnique->GetPassByIndex(pass)->Apply(0, pContext);

		pContext->DrawIndexed(pLineBuffers->m_indexCountTotal, 0, 0);
		if (renderSlice) {
			m_pQuadEffect->DrawTexture(proj * view, pContext, false);
		}
	}

	// set our final render target
	pContext->OMSetRenderTargets(1, &m_pOpaqueRTV, m_pDepthDSV);

	// render transparent texture to final output
	Vec2f screenMin(-1.0f, -1.0f);
	Vec2f screenMax(1.0f, 1.0f);
	m_pScreenEffect->m_pvScreenMinVariable->SetFloatVector(screenMin);
	m_pScreenEffect->m_pvScreenMaxVariable->SetFloatVector(screenMax);
	Vec2f texCoordMin(0.0f, 0.0f);
	Vec2f texCoordMax(1.0f, 1.0f);
	m_pScreenEffect->m_pvTexCoordMinVariable->SetFloatVector(texCoordMin);
	m_pScreenEffect->m_pvTexCoordMaxVariable->SetFloatVector(texCoordMax);
	pContext->IASetInputLayout(NULL);
	pContext->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLESTRIP);
	m_pScreenEffect->m_pTexVariable->SetResource(m_pTransparentSRV);
	//if (m_particleRenderParams.m_particleRenderMode == PARTICLE_RENDER_ADDITIVE) {
		m_pScreenEffect->m_pTechnique->GetPassByIndex(2)->Apply(0, pContext);
	//}
	//else if (m_particleRenderParams.m_particleRenderMode == PARTICLE_RENDER_ORDER_INDEPENDENT) {
		//m_pScreenEffect->m_pTechnique->GetPassByIndex(3)->Apply(0, pContext);
	//}
	pContext->Draw(4, 0);

	// restore viewports and render targets
	pContext->OMSetRenderTargets(D3D11_SIMULTANEOUS_RENDER_TARGET_COUNT, ppOldRTVs, pOldDSV);
	pContext->RSSetViewports(oldViewportCount, oldViewports);
	for (uint i = 0; i < D3D11_SIMULTANEOUS_RENDER_TARGET_COUNT; i++)
	{
		SAFE_RELEASE(ppOldRTVs[i]);
	}
	SAFE_RELEASE(pOldDSV);
}


void RenderingManager::RenderBalls(const BallBuffers* pBallBuffers, float radius)
{
	if(pBallBuffers == nullptr || pBallBuffers->m_ballCount == 0) return;

	ID3D11DeviceContext* pContext = nullptr;
	m_pDevice->GetImmediateContext(&pContext);

	// save viewports and render targets
	uint oldViewportCount = D3D11_VIEWPORT_AND_SCISSORRECT_OBJECT_COUNT_PER_PIPELINE;
	D3D11_VIEWPORT oldViewports[D3D11_VIEWPORT_AND_SCISSORRECT_OBJECT_COUNT_PER_PIPELINE];
	pContext->RSGetViewports(&oldViewportCount, oldViewports);
	ID3D11RenderTargetView* ppOldRTVs[D3D11_SIMULTANEOUS_RENDER_TARGET_COUNT];
	ID3D11DepthStencilView* pOldDSV;
	pContext->OMGetRenderTargets(D3D11_SIMULTANEOUS_RENDER_TARGET_COUNT, ppOldRTVs, &pOldDSV);

	// set our render target
	pContext->OMSetRenderTargets(1, &m_pOpaqueRTV, m_pDepthDSV);

	// build viewport
	D3D11_VIEWPORT viewport = {};
	viewport.TopLeftX = float(0);
	viewport.TopLeftY = float(0);
	viewport.Width    = float(m_projectionParams.GetImageWidth(m_range));
	viewport.Height   = float(m_projectionParams.m_imageHeight);
	viewport.MinDepth = 0.0f;
	viewport.MaxDepth = 1.0f;

	// IA
	pContext->IASetInputLayout(m_lineEffect.m_pInputLayoutBalls);
	UINT stride = sizeof(BallVertex);
	UINT offset = 0;
	pContext->IASetVertexBuffers(0, 1, &pBallBuffers->m_pVB, &stride, &offset);
	pContext->IASetIndexBuffer(nullptr, DXGI_FORMAT_R32_UINT, 0);
	pContext->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_POINTLIST);

	// shader vars
	m_lineEffect.m_pvLightPosVariable->SetFloatVector(m_viewParams.GetCameraPosition());

	m_lineEffect.m_pvColor0Variable->SetFloatVector(m_particleRenderParams.m_color0);
	m_lineEffect.m_pvColor1Variable->SetFloatVector(m_particleRenderParams.m_color1);

	Vec3f boxMin = -m_pVolume->GetVolumeHalfSizeWorld();
	tum3D::Vec3f sizePhys = (float)m_pVolume->GetVolumeSize().maximum() * m_pVolume->GetGridSpacing();
	m_lineEffect.m_pvBoxMinVariable->SetFloatVector(boxMin);
	m_lineEffect.m_pvBoxSizeVariable->SetFloatVector(Vec3f(2.0f / sizePhys));

	m_lineEffect.m_pfBallRadiusVariable->SetFloat(radius);

	uint pass = 0;

	if(m_stereoParams.m_stereoEnabled)
	{
		//Mat4f viewLeft  = m_viewParams.BuildViewMatrix(EYE_LEFT,  m_stereoParams.m_eyeDistance);
		//Mat4f viewRight = m_viewParams.BuildViewMatrix(EYE_RIGHT, m_stereoParams.m_eyeDistance);
		//Mat4f projLeft  = m_projectionParams.BuildProjectionMatrix(EYE_LEFT,  m_stereoParams.m_eyeDistance, m_range);
		//Mat4f projRight = m_projectionParams.BuildProjectionMatrix(EYE_RIGHT, m_stereoParams.m_eyeDistance, m_range);

		//viewport.Height /= 2.0f;
		//pContext->RSSetViewports(1, &viewport);

		//m_lineEffect.m_pmWorldViewProjVariable->SetMatrix(projLeft * viewLeft);
		//m_lineEffect.m_pTechniqueBalls->GetPassByIndex(pass)->Apply(0, pContext);

		//pContext->Draw(pBallBuffers->m_ballCount, 0);

		//viewport.TopLeftY += viewport.Height;
		//pContext->RSSetViewports(1, &viewport);

		//m_lineEffect.m_pmWorldViewProjVariable->SetMatrix(projRight * viewRight);
		//m_lineEffect.m_pTechniqueBalls->GetPassByIndex(pass)->Apply(0, pContext);

		//pContext->Draw(pBallBuffers->m_ballCount, 0);
	}
	else
	{
		Mat4f view = m_viewParams.BuildViewMatrix(EYE_CYCLOP, 0.0f);
		Mat4f proj = m_projectionParams.BuildProjectionMatrix(EYE_CYCLOP, 0.0f, m_range);

		Mat4f viewInv;
		invert4x4(view, viewInv);
		Vec3f camPos = Vec3f(viewInv.getCol(3));
		Vec3f camRight = Vec3f(viewInv.getCol(0));
		m_lineEffect.m_pvCamPosVariable->SetFloatVector(camPos);
		m_lineEffect.m_pvCamRightVariable->SetFloatVector(camRight);

		pContext->RSSetViewports(1, &viewport);

		m_lineEffect.m_pmWorldViewProjVariable->SetMatrix(proj * view);
		m_lineEffect.m_pTechniqueBalls->GetPassByIndex(pass)->Apply(0, pContext);

		pContext->Draw(pBallBuffers->m_ballCount, 0);
	}

	// restore viewports and render targets
	pContext->OMSetRenderTargets(D3D11_SIMULTANEOUS_RENDER_TARGET_COUNT, ppOldRTVs, pOldDSV);
	pContext->RSSetViewports(oldViewportCount, oldViewports);
	for(uint i = 0; i < D3D11_SIMULTANEOUS_RENDER_TARGET_COUNT; i++)
	{
		SAFE_RELEASE(ppOldRTVs[i]);
	}
	SAFE_RELEASE(pOldDSV);

	// copy z buffer into CUDA-compatible texture
	pContext->CopyResource(m_pDepthTexCopy, m_pDepthTex);

	pContext->Release();
}


void RenderingManager::RenderBricks(bool recordEvents)
{
	assert(m_pVolume != nullptr);

	// Walk brick list
	uint bricksRendered = 0;
	while (bricksRendered < m_bricksPerFrame && m_nextBrickToRender < (uint)m_bricksToRender.size())
	{
		// get brick
		const TimeVolumeIO::Brick* pBrick = m_bricksToRender[m_nextBrickToRender];

		// bail out if brick isn't loaded yet
		if (!pBrick->IsLoaded())
		{
			// sync GPU here - this (at least partially) unfucks cudaEvent timing...
			cudaSafeCall(cudaDeviceSynchronize());
			break;
		}

		++bricksRendered;

		////TEMP: write out individual brick as .raw file
		//if(pBrick->GetSpatialIndex()==Vec3i(2,2,2)) {
		//	int elemCount = pBrick->GetSize().volume();
		//	int channelCount = 3;
		//	int channelCountOut = 3;
		//	int dataSize = elemCount * channelCountOut * sizeof(float);

		//	float* pData = new float[elemCount * channelCountOut];
		//	memset(pData, 0, dataSize);
		//	for(int i = 0; i < elemCount; i++) {
		//		for(int c = 0; c < channelCount; c++) {
		//			pData[i * channelCountOut + c] = ((const float*)pBrick->GetChannelData(c))[i];
		//		}
		//	}
		//	FILE* file = fopen("brick222.raw", "wb");
		//	fwrite(pData, dataSize, 1, file);
		//	fclose(file);
		//	exit(42);
		//}


		Vec3i brickIndex = pBrick->GetSpatialIndex();
		Vec3i brickCount = m_pVolume->GetBrickCount();
		uint brickIndexLinear = brickIndex.x() + brickCount.x() * (brickIndex.y() + brickCount.y() * brickIndex.z());

		//printf("Rendering brick #%i (%i,%i,%i)\n", m_nextBrickToRender, brickIndex.x(), brickIndex.y(), brickIndex.z());

		// decide what we need to do
		bool needVelocityBrick = false;
		int brickSlotMeasureIndex = -1;
		bool fillBrickSlotMeasure = false;
		bool compressBrickSlotMeasure = false;
		switch(m_raycastParams.m_measureComputeMode)
		{
			case MEASURE_COMPUTE_ONTHEFLY:
				needVelocityBrick = true;
				break;
			case MEASURE_COMPUTE_PRECOMP_DISCARD:
				assert(!m_measureBrickSlots.empty());
				brickSlotMeasureIndex = 0;
				needVelocityBrick = true;
				fillBrickSlotMeasure = true;
				break;
			case MEASURE_COMPUTE_PRECOMP_STORE_GPU:
				assert(m_measureBrickSlots.size() >= brickIndexLinear);
				brickSlotMeasureIndex = brickIndexLinear;
				fillBrickSlotMeasure = !m_measureBrickSlotsFilled[brickSlotMeasureIndex];
				needVelocityBrick = fillBrickSlotMeasure;
				break;
			case MEASURE_COMPUTE_PRECOMP_COMPRESS:
				assert(!m_measureBrickSlots.empty());
				brickSlotMeasureIndex = 0;
				compressBrickSlotMeasure = (m_measureBricksCompressed[brickIndexLinear].empty());
				fillBrickSlotMeasure = compressBrickSlotMeasure;
				needVelocityBrick = fillBrickSlotMeasure;
				break;
		}
		BrickSlot* pBrickSlotMeasure = (brickSlotMeasureIndex >= 0 ? m_measureBrickSlots[brickSlotMeasureIndex] : nullptr);
		bool useFilteredBrick = (m_brickSlots.size() > 1);

		Vec3ui brickSize(m_brickSize, m_brickSize, m_brickSize);
		//bool brickIsFull = (m_brickSlots[0].GetFilledSize() == brickSize);
		bool recordEventsThisBrick = recordEvents; // && brickIsFull;
		//bool measureBrickIsFull = pBrickSlotMeasure != nullptr && (pBrickSlotMeasure->GetFilledSize() == brickSize);
		bool recordEventsThisMeasureBrick = recordEvents; // && measureBrickIsFull;

		if(m_nextPass == 0)
		{
			m_brickSlotsFilled = 0;

			if(needVelocityBrick)
			{
				// Upload filtered data to GPU
				for(size_t i = m_raycastParams.m_filterOffset; i < m_pFilteredVolumes->size() && m_brickSlotsFilled < m_brickSlots.size(); ++i)
				{
					const FilteredVolume& filteredVolume = (*m_pFilteredVolumes)[i];
					BrickSlot& brickSlotFiltered = m_brickSlots[m_brickSlotsFilled++];

					// for bricks with > 4 channels (i.e. more than 1 texture), fill each texture separately
					int channel = 0;
					uint tex = 0;
					while(channel < m_channelCount)
					{
						assert(tex < brickSlotFiltered.GetTextureCount());

						int channelCountThisTex = min(channel + 4, m_channelCount) - channel;

						std::vector<uint*> data(channelCountThisTex);
						std::vector<uint> dataSize(channelCountThisTex);
						std::vector<float> quantSteps(channelCountThisTex);
						for(int c = 0; c < channelCountThisTex; c++)
						{
							const FilteredVolume::ChannelData& channelData = filteredVolume.GetChannelData(pBrick->GetSpatialIndex(), c);
							data[c] = const_cast<uint*>(channelData.m_pData);
							dataSize[c] = uint(channelData.m_dataSizeInUInts * sizeof(uint));
							quantSteps[c] = channelData.m_quantStep;
						}

						MultiTimerGPU* pTimer = recordEventsThisBrick ? &m_timerUploadDecompress : nullptr;
						eCompressionType compression = m_pVolume->IsCompressed() ? COMPRESSION_FIXEDQUANT : COMPRESSION_NONE;
						UploadBrick(m_pCompressShared, m_pCompressVolume, data, dataSize, pBrick->GetSize(), compression, quantSteps, false, m_dpChannelBuffer.data(), &brickSlotFiltered, Vec3ui(0, 0, 0), tex, pTimer);

						channel += 4;
						tex++;
					}
				}

				// Upload original data to GPU
				if(m_brickSlotsFilled < m_brickSlots.size())
				{
					MultiTimerGPU* pTimer = recordEventsThisBrick ? &m_timerUploadDecompress : nullptr;
					UploadBrick(m_pCompressShared, m_pCompressVolume, m_pVolume->GetInfo(), *pBrick, m_dpChannelBuffer.data(), &m_brickSlots[m_brickSlotsFilled++], Vec3ui(0, 0, 0), pTimer);
				}


				// compute measure brick if necessary
				if(fillBrickSlotMeasure)
				{
					if(recordEventsThisBrick)
						m_timerComputeMeasure.StartNextTimer();

					m_raycaster.FillMeasureBrick(m_raycastParams, m_brickSlots[0], *pBrickSlotMeasure);

					if(recordEventsThisBrick)
						m_timerComputeMeasure.StopCurrentTimer();

					m_measureBrickSlotsFilled[brickSlotMeasureIndex] = (m_raycastParams.m_measureComputeMode != MEASURE_COMPUTE_PRECOMP_DISCARD);
				}
			}

			// Compress the measure brick if required
			if(compressBrickSlotMeasure)
			{
				Vec3ui size = pBrick->GetSize();

				if(recordEventsThisMeasureBrick)
					m_timerCompressDownload.StartNextTimer();

				// copy measure brick to linear memory
				cudaMemcpy3DParms parms = { 0 };
				parms.kind = cudaMemcpyDeviceToDevice;
				parms.extent = make_cudaExtent(size.x(), size.y(), size.z());
				parms.srcArray = const_cast<cudaArray*>(pBrickSlotMeasure->GetCudaArray());
				parms.srcPos = make_cudaPos(0, 0, 0);
				parms.dstPtr = make_cudaPitchedPtr(m_dpChannelBuffer[0], size.x() * sizeof(float), size.x(), size.y());
				cudaSafeCall(cudaMemcpy3DAsync(&parms));

				// compress
				float quantStep = GetDefaultMeasureQuantStep(m_raycastParams.m_measure1); //TODO configurable quality factor
				if(!compressVolumeFloat(*m_pCompressShared, *m_pCompressVolume, m_dpChannelBuffer[0], size.x(), size.y(), size.z(), 2, m_measureBricksCompressed[brickIndexLinear], quantStep, m_pVolume->GetUseLessRLE()))
				{
					printf("RenderingManager::RenderBricks: compressVolumeFloat failed\n");
					//TODO cancel rendering?
				}

				if(recordEventsThisMeasureBrick)
					m_timerCompressDownload.StopCurrentTimer();
			}

			// Decompress measure brick if required
			if(m_raycastParams.m_measureComputeMode == MEASURE_COMPUTE_PRECOMP_COMPRESS)
			{
				Vec3ui size = pBrick->GetSize();
				float quantStep = GetDefaultMeasureQuantStep(m_raycastParams.m_measure1); //TODO configurable quality factor

				if(recordEventsThisMeasureBrick)
					m_timerUploadDecompress.StartNextTimer();

				cudaSafeCall(cudaHostRegister(m_measureBricksCompressed[brickIndexLinear].data(), m_measureBricksCompressed[brickIndexLinear].size() * sizeof(uint), 0));
				decompressVolumeFloat(*m_pCompressShared, *m_pCompressVolume, m_dpChannelBuffer[0], size.x(), size.y(), size.z(), 2, m_measureBricksCompressed[brickIndexLinear], quantStep, m_pVolume->GetUseLessRLE());
				cudaSafeCall(cudaHostUnregister(m_measureBricksCompressed[brickIndexLinear].data()));

				pBrickSlotMeasure->FillFromGPUChannels(const_cast<const float**>(m_dpChannelBuffer.data()), size);

				if(recordEventsThisMeasureBrick)
					m_timerUploadDecompress.StopCurrentTimer();
			}
		}

		// Render
		std::vector<BrickSlot*> brickSlotsToRender;
		if(m_raycastParams.m_measureComputeMode == MEASURE_COMPUTE_ONTHEFLY)
		{
			for(size_t i = 0; i < m_brickSlotsFilled; ++i)
			{
				brickSlotsToRender.push_back(&m_brickSlots[i]);
			}
		}
		else
		{
			brickSlotsToRender.push_back(pBrickSlotMeasure);
		}

		Vec3f boxMin, boxMax;
		m_pVolume->GetBrickBoxWorld(pBrick->GetSpatialIndex(), boxMin, boxMax);

		if(recordEventsThisBrick || recordEventsThisMeasureBrick)
			m_timerRaycast.StartNextTimer();

		cudaArray* pRaycastArray = nullptr;
		cudaArray* pDepthArray = nullptr;
		if(m_pDevice)
		{
			assert(m_pRaycastTexCuda != nullptr);
			// map render target for cuda
			cudaSafeCall(cudaGraphicsMapResources(1, &m_pRaycastTexCuda));
			cudaSafeCall(cudaGraphicsSubResourceGetMappedArray(&pRaycastArray, m_pRaycastTexCuda, 0, 0));

			cudaSafeCall(cudaGraphicsMapResources(1, &m_pDepthTexCopyCuda));
			cudaSafeCall(cudaGraphicsSubResourceGetMappedArray(&pDepthArray, m_pDepthTexCopyCuda, 0, 0));
		}
		else
		{
			assert(m_pRaycastArray != nullptr);
			pRaycastArray = m_pRaycastArray;
			//TODO ?
			//assert(m_pDepthArray != nullptr);
			//pDepthArray = m_pDepthArray;
		}
		// render the brick
		//if(recordEventsThisBrick || recordEventsThisMeasureBrick)
		//	m_timerRaycast.StartNextTimer();
		bool brickDone = false;
		if(m_stereoParams.m_stereoEnabled)
		{
			brickDone = m_raycaster.RenderBrick(pRaycastArray, pDepthArray, m_raycastParams,
				m_viewParams.BuildViewMatrix(EYE_LEFT,  m_stereoParams.m_eyeDistance), EYE_LEFT,
				m_pTfArray, brickSlotsToRender, boxMin, boxMax, m_nextPass);
			m_raycaster.RenderBrick(pRaycastArray, pDepthArray, m_raycastParams,
				m_viewParams.BuildViewMatrix(EYE_RIGHT, m_stereoParams.m_eyeDistance), EYE_RIGHT,
				m_pTfArray, brickSlotsToRender, boxMin, boxMax, m_nextPass);
		}
		else
		{
			brickDone = m_raycaster.RenderBrick(pRaycastArray, pDepthArray, m_raycastParams,
				m_viewParams.BuildViewMatrix(EYE_CYCLOP, 0.0f), EYE_CYCLOP,
				m_pTfArray, brickSlotsToRender, boxMin, boxMax, m_nextPass);
		}
		//if(recordEventsThisBrick || recordEventsThisMeasureBrick)
		//	m_timerRaycast.StopCurrentTimer();
		if(m_pDevice)
		{
			// unmap resources again
			cudaSafeCall(cudaGraphicsUnmapResources(1, &m_pDepthTexCopyCuda));
			cudaSafeCall(cudaGraphicsUnmapResources(1, &m_pRaycastTexCuda));
		}

		if(recordEventsThisBrick || recordEventsThisMeasureBrick)
			m_timerRaycast.StopCurrentTimer();

		if(brickDone)
		{
			++m_nextBrickToRender;
			m_nextPass = 0;
		}
		else
		{
			++m_nextPass;
		}
	}

	UpdateBricksToLoad();
}

/*
void RenderingManager::RenderSliceTexture()
{
	if (!m_particleRenderParams.m_showSlice
		|| m_particleRenderParams.m_pSliceTexture == nullptr) {
		return;
	}

	ID3D11DeviceContext* pContext = nullptr;
	m_pDevice->GetImmediateContext(&pContext);

	//build shader params
	Vec3f volumeHalfSizeWorld = m_pVolume->GetVolumeHalfSizeWorld();
	tum3D::Vec3f normal(0, 0, 1);
	tum3D::Vec2f size(volumeHalfSizeWorld.x() * 2, volumeHalfSizeWorld.y() * 2);
	tum3D::Vec3f center(0, 0, m_particleRenderParams.m_slicePosition);

	// save viewports and render targets
	uint oldViewportCount = D3D11_VIEWPORT_AND_SCISSORRECT_OBJECT_COUNT_PER_PIPELINE;
	D3D11_VIEWPORT oldViewports[D3D11_VIEWPORT_AND_SCISSORRECT_OBJECT_COUNT_PER_PIPELINE];
	pContext->RSGetViewports(&oldViewportCount, oldViewports);
	ID3D11RenderTargetView* ppOldRTVs[D3D11_SIMULTANEOUS_RENDER_TARGET_COUNT];
	ID3D11DepthStencilView* pOldDSV;
	pContext->OMGetRenderTargets(D3D11_SIMULTANEOUS_RENDER_TARGET_COUNT, ppOldRTVs, &pOldDSV);

	// set our render target
	//pContext->OMSetRenderTargets(1, &m_pTransparentRTV, m_pDepthDSV);
	pContext->OMSetRenderTargets(1, &m_pOpaqueRTV, m_pDepthDSV);

	// build viewport
	D3D11_VIEWPORT viewport = {};
	viewport.TopLeftX = float(0);
	viewport.TopLeftY = float(0);
	viewport.Width = float(m_projectionParams.GetImageWidth(m_range));
	viewport.Height = float(m_projectionParams.m_imageHeight);
	viewport.MinDepth = 0.0f;
	viewport.MaxDepth = 1.0f;

	if (m_stereoParams.m_stereoEnabled)
	{
		Mat4f viewLeft = m_viewParams.BuildViewMatrix(EYE_LEFT, m_stereoParams.m_eyeDistance);
		Mat4f viewRight = m_viewParams.BuildViewMatrix(EYE_RIGHT, m_stereoParams.m_eyeDistance);
		Mat4f projLeft = m_projectionParams.BuildProjectionMatrix(EYE_LEFT, m_stereoParams.m_eyeDistance, m_range);
		Mat4f projRight = m_projectionParams.BuildProjectionMatrix(EYE_RIGHT, m_stereoParams.m_eyeDistance, m_range);

		viewport.Height /= 2.0f;
		pContext->RSSetViewports(1, &viewport);
		m_pQuadEffect->DrawTexture(m_particleRenderParams.m_pSliceTexture, center, normal, size, projLeft * viewLeft, pContext);
		
		viewport.TopLeftY += viewport.Height;
		pContext->RSSetViewports(1, &viewport);
		m_pQuadEffect->DrawTexture(m_particleRenderParams.m_pSliceTexture, center, normal, size, projRight * viewRight, pContext);
	}
	else
	{
		Mat4f view = m_viewParams.BuildViewMatrix(EYE_CYCLOP, 0.0f);
		Mat4f proj = m_projectionParams.BuildProjectionMatrix(EYE_CYCLOP, 0.0f, m_range);

		pContext->RSSetViewports(1, &viewport);
		m_pQuadEffect->DrawTexture(m_particleRenderParams.m_pSliceTexture, center, normal, size, proj * view, pContext);
	}

	// restore viewports and render targets
	pContext->OMSetRenderTargets(D3D11_SIMULTANEOUS_RENDER_TARGET_COUNT, ppOldRTVs, pOldDSV);
	pContext->RSSetViewports(oldViewportCount, oldViewports);
	for (uint i = 0; i < D3D11_SIMULTANEOUS_RENDER_TARGET_COUNT; i++)
	{
		SAFE_RELEASE(ppOldRTVs[i]);
	}
	SAFE_RELEASE(pOldDSV);

	// copy z buffer into cuda-compatible texture
	pContext->CopyResource(m_pDepthTexCopy, m_pDepthTex);

	pContext->Release();
}
*/

void RenderingManager::UpdateTimings()
{
	m_timings.UploadDecompressGPU = m_timerUploadDecompress.GetStats();
	m_timings.ComputeMeasureGPU   = m_timerComputeMeasure.GetStats();
	m_timings.RaycastGPU          = m_timerRaycast.GetStats();
	m_timings.CompressDownloadGPU = m_timerCompressDownload.GetStats();

	m_timerRender.Stop();
	m_timings.RenderWall = m_timerRender.GetElapsedTimeMS();
}
