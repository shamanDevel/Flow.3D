#include "HeatMapManager.h"

#include <algorithm>
#include <iostream>
#include <cudaUtil.h>

#include "TracingCommon.h"
#include "HeatMapKernel.h"

using namespace tum3D;


HeatMapManager::HeatMapManager()
	: m_pCompressShared(NULL)
	, m_pCompressVolume(NULL)
	, m_pDevice(NULL)
	, m_pVolume(NULL)
	, m_pHeatMap(NULL)
	, m_isCreated(false)
	, m_params()
{
}


HeatMapManager::~HeatMapManager()
{
	delete m_pHeatMap;
}

bool HeatMapManager::Create(GPUResources * pCompressShared, CompressVolumeResources * pCompressVolume, ID3D11Device * pDevice)
{
	m_pCompressShared = pCompressShared;
	m_pCompressVolume = pCompressVolume;
	m_pDevice = pDevice;

	m_isCreated = true;
	return true;
}

void HeatMapManager::Release()
{
	delete m_pHeatMap;
	m_pHeatMap = nullptr;
}

void HeatMapManager::SetParams(const HeatMapParams & params)
{
	m_params = params;
}

void HeatMapManager::SetVolumeAndReset(const TimeVolume & volume)
{
	delete m_pHeatMap;
	m_pHeatMap = nullptr;

	m_pVolume = &volume;

	//compute heatmap size
	Vec3i volumeResolution = volume.GetVolumeSize();
	Vec3f volumeHalfSizeWorld = volume.GetVolumeHalfSizeWorld();
	static const int MAX_RESOLUTION = 128;
	static const int MIN_RESOLUTION = 4;
	int scalingFactor = std::max(1, volumeResolution.maximum() / MAX_RESOLUTION);
	tum3D::Vec3i heatMapResolution = tum3D::Vec3i(
		max(MIN_RESOLUTION, volumeResolution.x() / scalingFactor),
		max(MIN_RESOLUTION, volumeResolution.y() / scalingFactor),
		max(MIN_RESOLUTION, volumeResolution.z() / scalingFactor)
	);
	std::cout << "Volume resolution: " << volumeResolution << std::endl;
	std::cout << "Heat Map resolution: " << heatMapResolution << std::endl;
	m_resolution = make_int3(heatMapResolution.x(), heatMapResolution.y(), heatMapResolution.z());
	m_worldOffset = make_float3(volumeHalfSizeWorld.x(), volumeHalfSizeWorld.y(), volumeHalfSizeWorld.z());
	m_worldToGrid = make_float3(
		heatMapResolution.x() / (2 * volumeHalfSizeWorld.x()),
		heatMapResolution.y() / (2 * volumeHalfSizeWorld.y()),
		heatMapResolution.z() / (2 * volumeHalfSizeWorld.z()));

	//create heatmap
	m_pHeatMap = new HeatMap(heatMapResolution.x(), heatMapResolution.y(), heatMapResolution.z());
	std::cout << "Heat Map created" << std::endl;
}

void HeatMapManager::ProcessLines(std::shared_ptr<LineBuffers> pLineBuffer)
{
	if (!m_params.m_enableRecording) return;
	std::cout << "Process lines" << std::endl;

	//first test, fill channel '0'
	HeatMap::Channel_ptr channel = m_pHeatMap->createChannel(0);
	ClearChannels();

	//aquire buffers
	cudaSafeCall(cudaGraphicsMapResources(1, &pLineBuffer->m_pIBCuda));
	uint* dpIB = nullptr;
	cudaSafeCall(cudaGraphicsResourceGetMappedPointer((void**)&dpIB, nullptr, pLineBuffer->m_pIBCuda));
	cudaSafeCall(cudaGraphicsMapResources(1, &pLineBuffer->m_pVBCuda));
	LineVertex* dpVB = nullptr;
	cudaSafeCall(cudaGraphicsResourceGetMappedPointer((void**)&dpVB, nullptr, pLineBuffer->m_pVBCuda));

	//fill channel
	heatmapKernelFillChannel(channel->getCudaBuffer(), dpVB, dpIB, pLineBuffer->m_indexCountTotal,
		m_resolution, m_worldOffset, m_worldToGrid);

	//release resources
	cudaSafeCall(cudaGraphicsUnmapResources(1, &pLineBuffer->m_pVBCuda));
	cudaSafeCall(cudaGraphicsUnmapResources(1, &pLineBuffer->m_pIBCuda));

	//test
	std::pair<uint, uint> minMaxValue = heatmapKernelFindMinMax(channel->getCudaBuffer(), m_resolution);
	std::cout << "Done, min value: " << minMaxValue.first << ", max value: " << minMaxValue.second << std::endl;
}

void HeatMapManager::Render(const ViewParams & viewParams, const StereoParams & stereoParam)
{
	if (!m_params.m_enableRendering) return;
}

void HeatMapManager::ClearChannels()
{
	m_pHeatMap->clearAllChannels();
}

