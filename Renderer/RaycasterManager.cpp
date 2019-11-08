#include <RaycasterManager.h>

#include <cudaUtil.h>
#include <cuda_d3d11_interop.h>
#include <ClearCudaArray.h>
#include <BoxUtils.h>
#include <BrickSlot.h>
#include <BrickUpload.h>


using namespace tum3D;


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


RaycasterManager::RaycasterManager()
	: m_pDevice(nullptr)
	, m_brickSlotsFilled(0)
	, m_pTfArray(nullptr)
	, m_nextBrickToRender(0), m_nextPass(0)
	, m_brickSize(0), m_channelCount(0)
	, m_bricksPerFrame(4)
	, m_pVolume(nullptr), m_pFilteredVolumes(nullptr)
	, m_pRaycastTex(nullptr), m_pRaycastSRV(nullptr), m_pRaycastRTV(nullptr), m_pRaycastTexCuda(nullptr)
	, m_pDepthTexCopyCuda(nullptr)
{

}

bool RaycasterManager::Create(ID3D11Device* pDevice)
{
	//m_pCompressShared = pCompressShared;
	//m_pCompressVolume = pCompressVolume;
	m_pDevice = pDevice;

	if (!CreateScreenDependentResources())
	{
		Release();
		return false;
	}

	if (!m_raycaster.Create())
	{
		Release();
		return false;
	}

	return true;
}

void RaycasterManager::Release()
{
	CancelRendering();

	ReleaseResources();

	m_raycaster.Release();
	m_timerUploadDecompress.ReleaseTimers();
	m_timerComputeMeasure.ReleaseTimers();
	m_timerRaycast.ReleaseTimers();
	m_timerCompressDownload.ReleaseTimers();

	m_pDevice = nullptr;
	//m_pCompressVolume = nullptr;
	//m_pCompressShared = nullptr;
}

void RaycasterManager::SetProjectionParams(const ProjectionParams& params, const Range1D& range)
{
	if (params == m_projectionParams && range == m_range) return;

	CancelRendering();

	bool recreateScreenResources =
		(params.GetImageHeight(range) != m_projectionParams.GetImageHeight(m_range) ||
		params.GetImageWidth(range) != m_projectionParams.GetImageWidth(m_range));

	m_projectionParams = params;
	m_range = range;

	if (recreateScreenResources)
		CreateScreenDependentResources();
}

bool RaycasterManager::CreateScreenDependentResources()
{
	ReleaseScreenDependentResources();

	uint width = m_projectionParams.GetImageWidth(m_range);
	uint height = m_projectionParams.GetImageHeight(m_range);

	if (width == 0 || height == 0)
		return true;


	if (m_pDevice)
	{
		HRESULT hr;

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

		// create texture/rendertarget for raycasting
		desc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
		hr = m_pDevice->CreateTexture2D(&desc, nullptr, &m_pRaycastTex);
		if (FAILED(hr)) return false;
		hr = m_pDevice->CreateShaderResourceView(m_pRaycastTex, nullptr, &m_pRaycastSRV);
		if (FAILED(hr)) return false;
		hr = m_pDevice->CreateRenderTargetView(m_pRaycastTex, nullptr, &m_pRaycastRTV);
		if (FAILED(hr)) return false;
		cudaSafeCall(cudaGraphicsD3D11RegisterResource(&m_pRaycastTexCuda, m_pRaycastTex, cudaGraphicsRegisterFlagsSurfaceLoadStore));

		// CUDA can't share depth or typeless resources, so we have to allocate another tex and copy into it...
		desc.BindFlags = 0;
		desc.Format = DXGI_FORMAT_R32_FLOAT;
		hr = m_pDevice->CreateTexture2D(&desc, nullptr, &m_pDepthTexCopy);
		if (FAILED(hr)) return false;
		cudaSafeCall(cudaGraphicsD3D11RegisterResource(&m_pDepthTexCopyCuda, m_pDepthTexCopy, cudaGraphicsRegisterFlagsSurfaceLoadStore));
	}
	else
	{
		cudaChannelFormatDesc desc = cudaCreateChannelDesc<uchar4>();
		std::cout << "cudaMallocArray " << (width * height * sizeof(uchar4)) / 1024.0f << "KB" << std::endl;
		cudaSafeCall(cudaMallocArray(&m_pRaycastArray, &desc, width, height, cudaArraySurfaceLoadStore));
	}

	return true;
}

void RaycasterManager::ReleaseScreenDependentResources()
{
	cudaSafeCall(cudaFreeArray(m_pRaycastArray));
	m_pRaycastArray = nullptr;

	if (m_pDepthTexCopyCuda)
	{
		cudaSafeCall(cudaGraphicsUnregisterResource(m_pDepthTexCopyCuda));
		m_pDepthTexCopyCuda = nullptr;
	}

	if (m_pDepthTexCopy)
	{
		m_pDepthTexCopy->Release();
		m_pDepthTexCopy = nullptr;
	}

	if (m_pRaycastTexCuda)
	{
		cudaSafeCall(cudaGraphicsUnregisterResource(m_pRaycastTexCuda));
		m_pRaycastTexCuda = nullptr;
	}

	if (m_pRaycastRTV)
	{
		m_pRaycastRTV->Release();
		m_pRaycastRTV = nullptr;
	}

	if (m_pRaycastSRV)
	{
		m_pRaycastSRV->Release();
		m_pRaycastSRV = nullptr;
	}

	if (m_pRaycastTex)
	{
		m_pRaycastTex->Release();
		m_pRaycastTex = nullptr;
	}
}

RaycasterManager::eRenderState RaycasterManager::StartRendering(
	const TimeVolume& volume,
	const ViewParams& viewParams,
	const StereoParams& stereoParams,
	const RaycastParams& raycastParams,
	const std::vector<FilteredVolume>& filteredVolumes,
	cudaArray* pTransferFunction,
	int transferFunctionDevice)
{
	if (IsRendering())
		CancelRendering();
	//@Behdad
	m_waitForRendering = true;

	m_pVolume = &volume;
	m_pFilteredVolumes = &filteredVolumes;
	m_raycastParams = raycastParams;

	m_viewParams = viewParams;
	m_stereoParams = stereoParams;


	// compute frustum planes in view space
	Vec4f frustumPlanes[6];
	Vec4f frustumPlanes2[6];
	if (m_stereoParams.m_stereoEnabled)
	{
		m_projectionParams.GetFrustumPlanes(frustumPlanes, EYE_LEFT, m_stereoParams.m_eyeDistance, m_range);
		m_projectionParams.GetFrustumPlanes(frustumPlanes2, EYE_RIGHT, m_stereoParams.m_eyeDistance, m_range);

		// we want the inverse transpose of the inverse of view
		Mat4f viewLeft = m_viewParams.BuildViewMatrix(EYE_LEFT, m_stereoParams.m_eyeDistance);
		Mat4f viewLeftTrans;
		viewLeft.transpose(viewLeftTrans);
		Mat4f viewRight = m_viewParams.BuildViewMatrix(EYE_RIGHT, m_stereoParams.m_eyeDistance);
		Mat4f viewRightTrans;
		viewRight.transpose(viewRightTrans);

		for (uint i = 0; i < 6; i++)
		{
			frustumPlanes[i] = viewLeftTrans  * frustumPlanes[i];
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

		for (uint i = 0; i < 6; i++)
		{
			frustumPlanes[i] = viewTrans * frustumPlanes[i];
		}
	}

	// copy transfer function
	if (pTransferFunction)
	{
		cudaChannelFormatDesc desc;
		cudaExtent extent;
		cudaSafeCall(cudaArrayGetInfo(&desc, &extent, nullptr, pTransferFunction));
		// for 1D arrays, this returns 0 in width and height...
		extent.height = (extent.height, size_t(1));
		extent.depth = (extent.depth, size_t(1));
		std::cout << "cudaMallocArray " << (extent.width * extent.height * 16) / 1024.0f << "KB" << std::endl;
		cudaSafeCall(cudaMallocArray(&m_pTfArray, &desc, extent.width, extent.height));
		size_t elemCount = extent.width * extent.height * extent.depth;
		size_t bytesPerElem = (desc.x + desc.y + desc.z + desc.w) / 8;

		int myDevice = -1;
		cudaSafeCall(cudaGetDevice(&myDevice));
		if (myDevice == transferFunctionDevice || transferFunctionDevice == -1)
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

	bool measureChanged = (m_raycastParams.m_measure1 != raycastParams.m_measure1);


	// set params and create resources
	m_raycaster.SetBrickSizeWorld(m_pVolume->GetBrickSizeWorld());
	m_raycaster.SetBrickOverlapWorld(m_pVolume->GetBrickOverlapWorld());
	m_raycaster.SetGridSpacing(m_pVolume->GetGridSpacing());
	m_raycaster.SetParams(m_projectionParams, m_stereoParams, m_range);

	//if (m_pVolume->GetBrickSizeWithOverlap() != m_brickSize ||
	//	m_pVolume->GetChannelCount() != m_channelCount ||
	//	GetRequiredBrickSlotCount() != m_brickSlots.size())
	{
		if (FAILED(CreateVolumeDependentResources()))
		{
			MessageBoxA(nullptr, "RenderingManager::StartRendering: Failed creating volume-dependent resources! (probably not enough GPU memory)", "Fail", MB_OK | MB_ICONINFORMATION);
			CancelRendering();
			return STATE_ERROR;
		}
	}

	if (!ManageMeasureBrickSlots())
	{
		MessageBoxA(nullptr, "RenderingManager::StartRendering: Failed creating brick slots for precomputed measure (probably not enough GPU memory). Reverting to on-the-fly computation.", "Fail", MB_OK | MB_ICONINFORMATION);
		m_raycastParams.m_measureComputeMode = MEASURE_COMPUTE_ONTHEFLY;
	}

	// if the measure was changed, have to clear all the precomputed measure bricks
	if (measureChanged)
	{
		for (uint i = 0; i < m_measureBrickSlotsFilled.size(); i++)
		{
			m_measureBrickSlotsFilled[i] = false;
		}
		for (uint i = 0; i < m_measureBricksCompressed.size(); i++)
		{
			m_measureBricksCompressed[i].clear();
		}
	}

	// if we're doing raycasting, build list of bricks to render
	m_bricksToRender.clear();
	m_nextBrickToRender = 0;
	m_nextPass = 0;

	// get bricks of current timestep, sort by distance
	std::vector<BrickSortItem> bricksCurTimestep;
	const Vec3f& camPos = m_viewParams.GetCameraPosition();
	auto& bricks = m_pVolume->GetNearestTimestep().bricks;
	for (auto it = bricks.cbegin(); it != bricks.cend(); ++it)
	{
		const TimeVolumeIO::Brick* pBrick = &(*it);
		bricksCurTimestep.push_back(BrickSortItem(pBrick, GetDistance(volume, pBrick->GetSpatialIndex(), camPos)));
	}
	std::sort(bricksCurTimestep.begin(), bricksCurTimestep.end());

	// walk brick list, collect bricks to render
	for (size_t brickIndex = 0; brickIndex < bricksCurTimestep.size(); brickIndex++)
	{
		// get brick
		const TimeVolumeIO::Brick* pBrick = bricksCurTimestep[brickIndex].pBrick;

		Vec3f boxMin, boxMax;
		m_pVolume->GetBrickBoxWorld(pBrick->GetSpatialIndex(), boxMin, boxMax);

		bool clipped = false;

		// check if brick is clipped against clip box
		clipped = clipped || IsBoxClippedAgainstBox(m_raycastParams.m_clipBoxMin, m_raycastParams.m_clipBoxMax, boxMin, boxMax);

		// check if brick is outside view frustum
		for (uint i = 0; i < 6; i++)
		{
			bool planeClipped = IsBoxClippedAgainstPlane(frustumPlanes[i], boxMin, boxMax);
			if (m_stereoParams.m_stereoEnabled)
			{
				planeClipped = planeClipped && IsBoxClippedAgainstPlane(frustumPlanes2[i], boxMin, boxMax);
			}
			clipped = clipped || planeClipped;
		}


		// Clip the voxel if the iso value is not in the data range of our voxel.
		bool isoValueNotInMinMaxRange = false;
		if (m_raycastParams.m_raycastMode == RAYCAST_MODE_ISO) {
			float minValueMeasure1 = m_raycastParams.m_measureScale1 * pBrick->GetMinMeasuresInBrick()[static_cast<size_t>(m_raycastParams.m_measure1)];
			float maxValueMeasure1 = m_raycastParams.m_measureScale1 * pBrick->GetMaxMeasuresInBrick()[static_cast<size_t>(m_raycastParams.m_measure1)];
			if (m_raycastParams.m_isoValue1 < minValueMeasure1 || m_raycastParams.m_isoValue1 > maxValueMeasure1) {
				isoValueNotInMinMaxRange = true;
			}
		}

		if (clipped || isoValueNotInMinMaxRange)
		{
			m_bricksClipped.push_back(pBrick);
		}
		else
		{
			m_bricksToRender.push_back(pBrick);
		}
	}


	// now we can update the list of bricks to load
	UpdateBricksToLoad();


	// clear rendertargets and depth buffer
	ClearResult();


	// if there's nothing to raycast, we're finished now
	if (m_bricksToRender.empty())
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

void RaycasterManager::ClearResult()
{
	if (m_pDevice)
	{
		ID3D11DeviceContext* pContext = nullptr;
		m_pDevice->GetImmediateContext(&pContext);

		// clear rendertargets and depth buffer
		float black[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
		if (m_pRaycastRTV)
			pContext->ClearRenderTargetView(m_pRaycastRTV, black);

		// FIXME: we must clear our copy of the depth buffer texture.
		//if (m_pDepthDSV)
		//{
		//	pContext->ClearDepthStencilView(m_pDepthDSV, D3D11_CLEAR_DEPTH, 1.0f, 0);
		//	pContext->CopyResource(m_pDepthTexCopy, m_pDepthTex);
		//}

		pContext->Release();
	}
	else
	{
		clearCudaArray2Duchar4(m_pRaycastArray, m_projectionParams.GetImageWidth(m_range), m_projectionParams.GetImageHeight(m_range));
		//TODO clear DepthArray?
	}
}

size_t RaycasterManager::GetRequiredBrickSlotCount() const
{
	size_t brickSlotCount = RaycastModeScaleCount(m_raycastParams.m_raycastMode);
	brickSlotCount = max(brickSlotCount, size_t(1)); // HACK ... (happens when raycastMode is invalid)
	size_t filteredVolumeCount = m_pFilteredVolumes == nullptr ? 0 : m_pFilteredVolumes->size();
	brickSlotCount = min(brickSlotCount, filteredVolumeCount + 1);
	return brickSlotCount;
}

void RaycasterManager::ReleaseResources()
{
	CancelRendering();

	ReleaseMeasureBrickSlots();
	ReleaseVolumeDependentResources();

	m_pCompressShared.destroy();
	m_pCompressVolume.destroy();
}

bool RaycasterManager::CreateVolumeDependentResources()
{
	assert(m_pVolume != nullptr);

	ReleaseVolumeDependentResources();

	if (m_pVolume->IsCompressed())
	{
		uint brickSize = m_pVolume->GetBrickSizeWithOverlap();
		// do multi-channel decoding only for small bricks; for large bricks, mem usage gets too high
		uint channelCount = (brickSize <= 128) ? m_pVolume->GetChannelCount() : 1;
		uint huffmanBits = m_pVolume->GetHuffmanBitsMax();

		GPUResources::Config newconfig = CompressVolumeResources::getRequiredResources(brickSize, brickSize, brickSize, channelCount, huffmanBits);
		GPUResources::Config currentconfig = m_pCompressShared.getConfig();

		if (currentconfig.blockCountMax != newconfig.blockCountMax || currentconfig.bufferSize != newconfig.bufferSize || currentconfig.elemCountPerBlockMax != newconfig.elemCountPerBlockMax || currentconfig.log2HuffmanDistinctSymbolCountMax != newconfig.log2HuffmanDistinctSymbolCountMax || currentconfig.offsetIntervalMin != newconfig.offsetIntervalMin)
		{
			m_pCompressShared.destroy();
			m_pCompressVolume.destroy();

			m_pCompressShared.create(newconfig);
			m_pCompressVolume.create(m_pCompressShared.getConfig());
		}
	}


	m_brickSize = m_pVolume->GetBrickSizeWithOverlap();
	m_channelCount = m_pVolume->GetChannelCount();

	int brickSizeBytePerChannel = m_brickSize * m_brickSize * m_brickSize * sizeof(float);
	m_dpChannelBuffer.resize(m_channelCount);
	for (size_t channel = 0; channel < m_dpChannelBuffer.size(); channel++)
		cudaSafeCall(cudaMalloc2(&m_dpChannelBuffer[channel], brickSizeBytePerChannel));

	m_brickSlots.resize(GetRequiredBrickSlotCount());
	printf("RenderingManager::CreateVolumeDependentResources: Creating %u brick slot(s)\n", uint(m_brickSlots.size()));
	for (size_t i = 0; i < m_brickSlots.size(); ++i)
	{
		if (!m_brickSlots[i].Create(m_brickSize, m_channelCount))
		{
			printf("RenderingManager::CreateVolumeDependentResources: m_brickSlots[%u].Create failed\n", uint(i));
			ReleaseVolumeDependentResources();
			return false;
		}
	}

	return true;
}

void RaycasterManager::ReleaseVolumeDependentResources()
{
	//if (m_pCompressShared)
	//{
	//	m_pCompressShared->destroy();
	//	delete m_pCompressShared;
	//	m_pCompressShared = nullptr;
	//}

	//if (m_pCompressVolume)
	//{
	//	m_pCompressVolume->destroy();
	//	delete m_pCompressVolume;
	//	m_pCompressVolume = nullptr;
	//}

	for (size_t i = 0; i < m_brickSlots.size(); ++i)
		m_brickSlots[i].Release();
	m_brickSlots.clear();

	for (size_t channel = 0; channel < m_dpChannelBuffer.size(); channel++)
		cudaSafeCall(cudaFree(m_dpChannelBuffer[channel]));
	m_dpChannelBuffer.clear();

	m_channelCount = 0;
	m_brickSize = 0;
}

bool RaycasterManager::ManageMeasureBrickSlots()
{
	assert(m_pVolume != nullptr);

	uint slotCountTarget = 0;
	uint slotCountCompressedTarget = 0;
	switch (m_raycastParams.m_measureComputeMode)
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
	if (!recreate)
	{
		return true;
	}

	bool success = CreateMeasureBrickSlots(slotCountTarget, slotCountCompressedTarget);

	return success;
}

bool RaycasterManager::CreateMeasureBrickSlots(uint count, uint countCompressed)
{
	ReleaseMeasureBrickSlots();

	bool success = true;
	for (uint i = 0; i < count; i++)
	{
		m_measureBrickSlots.push_back(new BrickSlot());
		m_measureBrickSlotsFilled.push_back(false);
		if (!m_measureBrickSlots.back()->Create(m_pVolume->GetBrickSizeWithOverlap(), 1))
		{
			success = false;
			break;
		}
	}

	for (uint i = 0; i < countCompressed; i++)
	{
		m_measureBricksCompressed.push_back(std::vector<uint>());
	}

	if (!success)
	{
		ReleaseMeasureBrickSlots();
		return false;
	}

	return true;
}

void RaycasterManager::ReleaseMeasureBrickSlots()
{
	for (uint i = 0; i < m_measureBrickSlots.size(); i++)
	{
		if (m_measureBrickSlots[i])
		{
			m_measureBrickSlots[i]->Release();
			delete m_measureBrickSlots[i];
		}
	}
	m_measureBrickSlots.clear();
	m_measureBrickSlotsFilled.clear();
	m_measureBricksCompressed.clear();
}

bool RaycasterManager::IsRendering() const
{
	return m_pVolume != nullptr;
}

RaycasterManager::eRenderState RaycasterManager::Render()
{
	if (!IsRendering()) return STATE_ERROR;

	assert(m_nextBrickToRender < m_bricksToRender.size());

	RenderBricks(true);

	bool finished = (m_nextBrickToRender == m_bricksToRender.size());

	if (finished)
	{
		if (m_pTfArray)
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
		if (m_timings.UploadDecompressGPU.Count  > 0)
		{
			printf("Device %i: Upload/Decompress (GPU): %.2f ms (%.2f-%.2f-%.2f : %u)\n", device,
				m_timings.UploadDecompressGPU.Total,
				m_timings.UploadDecompressGPU.Min, m_timings.UploadDecompressGPU.Avg, m_timings.UploadDecompressGPU.Max,
				m_timings.UploadDecompressGPU.Count);
		}
		if (m_timings.ComputeMeasureGPU.Count  > 0)
		{
			printf("Device %i: ComputeMeasure (GPU): %.2f ms (%.2f-%.2f-%.2f : %u)\n", device,
				m_timings.ComputeMeasureGPU.Total,
				m_timings.ComputeMeasureGPU.Min, m_timings.ComputeMeasureGPU.Avg, m_timings.ComputeMeasureGPU.Max,
				m_timings.ComputeMeasureGPU.Count);
		}
		if (m_timings.RaycastGPU.Count  > 0)
		{
			printf("Device %i: Raycast (GPU): %.2f ms (%.2f-%.2f-%.2f : %u)\n", device,
				m_timings.RaycastGPU.Total,
				m_timings.RaycastGPU.Min, m_timings.RaycastGPU.Avg, m_timings.RaycastGPU.Max,
				m_timings.RaycastGPU.Count);
		}
		if (m_timings.CompressDownloadGPU.Count  > 0)
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

void RaycasterManager::CancelRendering()
{
	if (m_pTfArray)
	{
		cudaSafeCall(cudaFreeArray(m_pTfArray));
		m_pTfArray = nullptr;
	}
	m_bricksToLoad.clear();
	m_pFilteredVolumes = nullptr;
	m_pVolume = nullptr;

	ClearResult();
}

float RaycasterManager::GetRenderingProgress() const
{
	if (m_bricksToRender.empty())
	{
		return 1.0f;
	}

	return float(m_nextBrickToRender) / float(m_bricksToRender.size());
}

void RaycasterManager::UpdateBricksToLoad()
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

void RaycasterManager::RenderBricks(bool recordEvents)
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
		switch (m_raycastParams.m_measureComputeMode)
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

		if (m_nextPass == 0)
		{
			m_brickSlotsFilled = 0;

			if (needVelocityBrick)
			{
				// Upload filtered data to GPU
				for (size_t i = m_raycastParams.m_filterOffset; i < m_pFilteredVolumes->size() && m_brickSlotsFilled < m_brickSlots.size(); ++i)
				{
					const FilteredVolume& filteredVolume = (*m_pFilteredVolumes)[i];
					BrickSlot& brickSlotFiltered = m_brickSlots[m_brickSlotsFilled++];

					// for bricks with > 4 channels (i.e. more than 1 texture), fill each texture separately
					int channel = 0;
					uint tex = 0;
					while (channel < m_channelCount)
					{
						assert(tex < brickSlotFiltered.GetTextureCount());

						int channelCountThisTex = min(channel + 4, m_channelCount) - channel;

						std::vector<uint*> data(channelCountThisTex);
						std::vector<uint> dataSize(channelCountThisTex);
						std::vector<float> quantSteps(channelCountThisTex);
						for (int c = 0; c < channelCountThisTex; c++)
						{
							const FilteredVolume::ChannelData& channelData = filteredVolume.GetChannelData(pBrick->GetSpatialIndex(), c);
							data[c] = const_cast<uint*>(channelData.m_pData);
							dataSize[c] = uint(channelData.m_dataSizeInUInts * sizeof(uint));
							quantSteps[c] = channelData.m_quantStep;
						}

						MultiTimerGPU* pTimer = recordEventsThisBrick ? &m_timerUploadDecompress : nullptr;
						eCompressionType compression = m_pVolume->IsCompressed() ? COMPRESSION_FIXEDQUANT : COMPRESSION_NONE;
						UploadBrick(&m_pCompressShared, &m_pCompressVolume, data, dataSize, pBrick->GetSize(), compression, quantSteps, false, m_dpChannelBuffer.data(), &brickSlotFiltered, Vec3ui(0, 0, 0), tex, pTimer);

						channel += 4;
						tex++;
					}
				}

				// Upload original data to GPU
				if (m_brickSlotsFilled < m_brickSlots.size())
				{
					MultiTimerGPU* pTimer = recordEventsThisBrick ? &m_timerUploadDecompress : nullptr;
					UploadBrick(&m_pCompressShared, &m_pCompressVolume, m_pVolume->GetInfo(), *pBrick, m_dpChannelBuffer.data(), &m_brickSlots[m_brickSlotsFilled++], Vec3ui(0, 0, 0), pTimer);
				}


				// compute measure brick if necessary
				if (fillBrickSlotMeasure)
				{
					if (recordEventsThisBrick)
						m_timerComputeMeasure.StartNextTimer();

					m_raycaster.FillMeasureBrick(m_raycastParams, m_brickSlots[0], *pBrickSlotMeasure);

					if (recordEventsThisBrick)
						m_timerComputeMeasure.StopCurrentTimer();

					m_measureBrickSlotsFilled[brickSlotMeasureIndex] = (m_raycastParams.m_measureComputeMode != MEASURE_COMPUTE_PRECOMP_DISCARD);
				}
			}

			// Compress the measure brick if required
			if (compressBrickSlotMeasure)
			{
				Vec3ui size = pBrick->GetSize();

				if (recordEventsThisMeasureBrick)
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
				if (!compressVolumeFloat(m_pCompressShared, m_pCompressVolume, m_dpChannelBuffer[0], size.x(), size.y(), size.z(), 2, m_measureBricksCompressed[brickIndexLinear], quantStep, m_pVolume->GetUseLessRLE()))
				{
					printf("RenderingManager::RenderBricks: compressVolumeFloat failed\n");
					//TODO cancel rendering?
				}

				if (recordEventsThisMeasureBrick)
					m_timerCompressDownload.StopCurrentTimer();
			}

			// Decompress measure brick if required
			if (m_raycastParams.m_measureComputeMode == MEASURE_COMPUTE_PRECOMP_COMPRESS)
			{
				Vec3ui size = pBrick->GetSize();
				float quantStep = GetDefaultMeasureQuantStep(m_raycastParams.m_measure1); //TODO configurable quality factor

				if (recordEventsThisMeasureBrick)
					m_timerUploadDecompress.StartNextTimer();

				cudaSafeCall(cudaHostRegister(m_measureBricksCompressed[brickIndexLinear].data(), m_measureBricksCompressed[brickIndexLinear].size() * sizeof(uint), 0));
				decompressVolumeFloat(m_pCompressShared, m_pCompressVolume, m_dpChannelBuffer[0], size.x(), size.y(), size.z(), 2, m_measureBricksCompressed[brickIndexLinear], quantStep, m_pVolume->GetUseLessRLE());
				cudaSafeCall(cudaHostUnregister(m_measureBricksCompressed[brickIndexLinear].data()));

				pBrickSlotMeasure->FillFromGPUChannels(const_cast<const float**>(m_dpChannelBuffer.data()), size);

				if (recordEventsThisMeasureBrick)
					m_timerUploadDecompress.StopCurrentTimer();
			}
		}

		// Render
		std::vector<BrickSlot*> brickSlotsToRender;
		if (m_raycastParams.m_measureComputeMode == MEASURE_COMPUTE_ONTHEFLY)
		{
			for (size_t i = 0; i < m_brickSlotsFilled; ++i)
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

		if (recordEventsThisBrick || recordEventsThisMeasureBrick)
			m_timerRaycast.StartNextTimer();

		cudaArray* pRaycastArray = nullptr;
		cudaArray* pDepthArray = nullptr;
		if (m_pDevice)
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
		if (m_stereoParams.m_stereoEnabled)
		{
			brickDone = m_raycaster.RenderBrick(pRaycastArray, pDepthArray, m_raycastParams,
				m_viewParams.BuildViewMatrix(EYE_LEFT, m_stereoParams.m_eyeDistance), EYE_LEFT,
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
		if (m_pDevice)
		{
			// unmap resources again
			cudaSafeCall(cudaGraphicsUnmapResources(1, &m_pDepthTexCopyCuda));
			cudaSafeCall(cudaGraphicsUnmapResources(1, &m_pRaycastTexCuda));
		}

		if (recordEventsThisBrick || recordEventsThisMeasureBrick)
			m_timerRaycast.StopCurrentTimer();

		if (brickDone)
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

void RaycasterManager::UpdateTimings()
{
	m_timings.UploadDecompressGPU = m_timerUploadDecompress.GetStats();
	m_timings.ComputeMeasureGPU = m_timerComputeMeasure.GetStats();
	m_timings.RaycastGPU = m_timerRaycast.GetStats();
	m_timings.CompressDownloadGPU = m_timerCompressDownload.GetStats();

	m_timerRender.Stop();
	m_timings.RenderWall = m_timerRender.GetElapsedTimeMS();
}

//@Behdada
bool RaycasterManager::waitForRendering() const
{
	return this->m_waitForRendering;
}