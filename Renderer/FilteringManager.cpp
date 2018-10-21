#include "FilteringManager.h"

#include <cassert>
#include <numeric>

#include <cuda_runtime.h>

#include <cudaCompress/util/Quantize.h>

#include <cudaUtil.h>

#include "BrickUpload.h"

using namespace tum3D;


FilteringManager::FilteringManager()
	: m_isCreated(false), m_pCompressShared(nullptr), m_pCompressVolume(nullptr)
	, m_brickSize(0)
	, m_dpBufferCenter(nullptr), m_dpBufferLeft(nullptr), m_dpBufferRight(nullptr), m_dpBufferOut(nullptr)
	, m_pVolume(nullptr)
	, m_direction(DIR_X), m_channel(0) , m_nextBrickToFilter(0)
{
}

FilteringManager::~FilteringManager()
{
	assert(!m_isCreated);
}


bool FilteringManager::Create()
{
	std::cout << "Creating FilteringManager..." << std::endl;
	//if(!m_volumeFilter.Create())
	//{
	//	Release();
	//	return false;
	//}

	//m_pCompressShared = pCompressShared;
	//m_pCompressVolume = pCompressVolume;

	m_isCreated = true;

	std::cout << "FilteringManager created." << std::endl;
	return true;
}

void FilteringManager::Release()
{
	m_timerUploadDecompress.ReleaseTimers();
	m_timerConvolution.ReleaseTimers();
	m_timerCompressDownload.ReleaseTimers();

	ReleaseVolumeDependentResources();

	//m_pCompressVolume = nullptr;
	//m_pCompressShared = nullptr;

	m_isCreated = false;
}


bool FilteringManager::StartFiltering(const TimeVolume& volume, const FilterParams& filterParams)
{
	if(!volume.IsOpen()) return false;

	printf("Starting filtering...\n");

	m_pVolume = &volume;
	m_filterParams = filterParams;


	//if(m_pVolume->GetBrickSizeWithOverlap() != m_brickSize)
	{
		if(FAILED(CreateVolumeDependentResources()))
		{
			printf("Failed creating resources for filtering!\n");
			CancelFiltering();
			return false;
		}
	}

	// get list of bricks, sort into default order
	m_bricks.clear();
	const std::vector<TimeVolumeIO::Brick>& bricks = m_pVolume->GetNearestTimestep().bricks;
	Vec3i brickCount = m_pVolume->GetBrickCount();
	m_bricks.resize(bricks.size());
	for(size_t i = 0; i < bricks.size(); i++)
	{
		const Vec3i& brickPos = bricks[i].GetSpatialIndex();
		size_t index = brickPos.x() + brickCount.x() * (brickPos.y() + brickCount.y() * brickPos.z());
		m_bricks[index] = &bricks[i];
	}

	// find last non-zero radius
	m_radiusCount = 0;
	for(size_t i = 0; i < m_filterParams.m_radius.size(); i++)
	{
		if(m_filterParams.m_radius[i] > 0) m_radiusCount = uint(i + 1);
	}
	// allocate filtered volumes
	m_filteredVolume.resize(m_radiusCount);
	for(size_t i = 0; i < m_filteredVolume.size(); i++)
	{
		FilteredVolume& filteredVolume = m_filteredVolume[i];

		filteredVolume.Resize(m_pVolume->GetBrickCount(), m_pVolume->GetChannelCount());
	}

	m_nextBrickToFilter = 0;
	m_channel = 0;
	m_direction = DIR_X;
	m_radiusIndex = 0;

	UpdateBricksToLoad();


	// start timing
	m_timerFilter.Start();

	m_timerConvolution.ResetTimers();
	m_timerCompressDownload.ResetTimers();


	return true;
}

bool FilteringManager::Filter()
{
	if(!IsFiltering()) return false;

	assert(m_nextBrickToFilter < m_bricks.size());


	bool done = false;

	Vec3i brickPos = GetBrickPos(m_nextBrickToFilter, m_direction);

	// get neighboring bricks
	Vec3i offset(0, 0, 0);
	offset[m_direction] = 1;
	const TimeVolumeIO::Brick* pBrickLeft   = GetBrick(brickPos - offset);
	const TimeVolumeIO::Brick* pBrickCenter = GetBrick(brickPos);
	const TimeVolumeIO::Brick* pBrickRight  = GetBrick(brickPos + offset);

	if(!pBrickCenter->IsLoaded() || (pBrickLeft && !pBrickLeft->IsLoaded()) || (pBrickRight && !pBrickRight->IsLoaded()))
	{
		return false;
	}

	printf("Filter %u %c channel %i brick (%i,%i,%i)\n",
		m_filterParams.m_radius[m_radiusIndex],
		'X' + char(m_direction),
		m_channel,
		pBrickCenter->GetSpatialIndex().x(),
		pBrickCenter->GetSpatialIndex().y(),
		pBrickCenter->GetSpatialIndex().z());

	bool firstPass = (m_direction == DIR_X);

	// prepare data
	if(pBrickLeft == nullptr)
	{
		// new line: decompress center and right
		if(firstPass)
		{
			UploadBrickChannel(m_pCompressShared, m_pCompressVolume, m_pVolume->GetInfo(), *pBrickCenter, m_channel, m_dpBufferCenter, &m_timerUploadDecompress);
			if(pBrickRight != nullptr) {
				UploadBrickChannel(m_pCompressShared, m_pCompressVolume, m_pVolume->GetInfo(), *pBrickRight,  m_channel, m_dpBufferRight, &m_timerUploadDecompress);
			}
		}
		else
		{
			const FilteredVolume& filteredVolume = m_filteredVolume[m_radiusIndex];
			eCompressionType compression = m_pVolume->IsCompressed() ? COMPRESSION_FIXEDQUANT : COMPRESSION_NONE;

			const FilteredVolume::ChannelData& dataCenter = filteredVolume.GetChannelData(pBrickCenter->GetSpatialIndex(), m_channel);
			uint dataCenterSize = uint(dataCenter.m_dataSizeInUInts * sizeof(uint));
			UploadBrickChannel(m_pCompressShared, m_pCompressVolume, dataCenter.m_pData, dataCenterSize, pBrickCenter->GetSize(), compression, dataCenter.m_quantStep, false, m_channel, m_dpBufferCenter, &m_timerUploadDecompress);

			if(pBrickRight != nullptr) {
				const FilteredVolume::ChannelData& dataRight = filteredVolume.GetChannelData(pBrickRight->GetSpatialIndex(), m_channel);
				uint dataRightSize = uint(dataRight.m_dataSizeInUInts * sizeof(uint));
				UploadBrickChannel(m_pCompressShared, m_pCompressVolume, dataRight.m_pData, dataRightSize, pBrickRight->GetSize(), compression, dataRight.m_quantStep, false, m_channel, m_dpBufferRight, &m_timerUploadDecompress);
			}
		}
	}
	else
	{
		// continue existing line: swap pointers and decompress right
		float* dpTemp    = m_dpBufferLeft;
		m_dpBufferLeft   = m_dpBufferCenter;
		m_dpBufferCenter = m_dpBufferRight;
		m_dpBufferRight  = dpTemp;

		if(pBrickRight)
		{
			if(firstPass)
			{
				UploadBrickChannel(m_pCompressShared, m_pCompressVolume, m_pVolume->GetInfo(), *pBrickRight, m_channel, m_dpBufferRight, &m_timerUploadDecompress);
			}
			else
			{
				eCompressionType compression = m_pVolume->IsCompressed() ? COMPRESSION_FIXEDQUANT : COMPRESSION_NONE;
				const FilteredVolume::ChannelData& dataRight = m_filteredVolume[m_radiusIndex].GetChannelData(pBrickRight->GetSpatialIndex(), m_channel);
				uint dataRightSize = uint(dataRight.m_dataSizeInUInts * sizeof(uint));
				UploadBrickChannel(m_pCompressShared, m_pCompressVolume, dataRight.m_pData, dataRightSize, pBrickRight->GetSize(), compression, dataRight.m_quantStep, false, m_channel, m_dpBufferRight, &m_timerUploadDecompress);
			}
		}
	}

	Vec3ui size = pBrickCenter->GetSize();

	// filter
	VolumeFilter::ChannelData data =
	{
		m_dpBufferCenter,
		pBrickLeft  ? m_dpBufferLeft : nullptr,
		pBrickRight ? m_dpBufferRight : nullptr,
		m_dpBufferOut
	};
	int sizeLeft  = pBrickLeft  ? pBrickLeft ->GetSize()[m_direction] : 0;
	int sizeRight = pBrickRight ? pBrickRight->GetSize()[m_direction] : 0;
	m_timerConvolution.StartNextTimer();
	m_volumeFilter.Filter(EFilterDirection(m_direction), m_filterParams.m_radius[m_radiusIndex], data, size, m_pVolume->GetBrickOverlap(), sizeLeft, sizeRight);
	m_timerConvolution.StopCurrentTimer();

	// compress / download
	m_timerCompressDownload.StartNextTimer();
	FilteredVolume::ChannelData& channelDataTarget = m_filteredVolume[m_radiusIndex].GetChannelData(brickPos, m_channel);
	CompressDownload(m_pCompressShared, m_pCompressVolume, channelDataTarget, m_dpBufferOut, size, m_pVolume->IsCompressed());
	m_timerCompressDownload.StopCurrentTimer();

	// advance brick
	m_nextBrickToFilter++;
	if(m_nextBrickToFilter >= (uint)m_pVolume->GetBrickCount().volume())
	{
		m_nextBrickToFilter = 0;
		m_channel++;
		if(m_channel >= m_filteredVolume[m_radiusIndex].GetChannelCount()) {
			m_channel = 0;
			m_direction = EFilterDirection(m_direction + 1);
			if(m_direction == DIR_COUNT)
			{
				m_direction = DIR_X;
				m_radiusIndex++;
				if(m_radiusIndex == m_radiusCount)
				{
					done = true;
				}
			}
		}

		UpdateBricksToLoad();
	}

	if(done)
	{
		m_pVolume = nullptr;
		m_bricksToLoad.clear();

		UpdateTimings();

		float sizeMB = 0.0f;
		for(size_t i = 0; i < m_filteredVolume.size(); i++)
		{
			sizeMB += float(m_filteredVolume[i].GetTotalSizeBytes()) / (1024.0f * 1024.0f);
		}
		printf("Done filtering in %.2f s; size: %.2f MB.\n", m_timings.FilterWall / 1000.0f, sizeMB);
		printf("Upload/Decomp (GPU): %.2f ms (%.2f-%.2f-%.2f : %u)\n",
			m_timings.UploadDecompressGPU.Total,
			m_timings.UploadDecompressGPU.Min, m_timings.UploadDecompressGPU.Avg, m_timings.UploadDecompressGPU.Max,
			m_timings.UploadDecompressGPU.Count);
		printf("Convolution (GPU): %.2f ms (%.2f-%.2f-%.2f : %u)\n",
			m_timings.ConvolutionGPU.Total,
			m_timings.ConvolutionGPU.Min, m_timings.ConvolutionGPU.Avg, m_timings.ConvolutionGPU.Max,
			m_timings.ConvolutionGPU.Count);
		printf("Comp/Download (GPU): %.2f ms (%.2f-%.2f-%.2f : %u)\n",
			m_timings.CompressDownloadGPU.Total,
			m_timings.CompressDownloadGPU.Min, m_timings.CompressDownloadGPU.Avg, m_timings.CompressDownloadGPU.Max,
			m_timings.CompressDownloadGPU.Count);
	}
	return done;
}

void FilteringManager::CancelFiltering()
{
	if(IsFiltering())
	{
		printf("FilteringManager::CancelFiltering.\n");
		m_bricks.clear();
		m_bricksToLoad.clear();
		m_pVolume = nullptr;
	}
}

bool FilteringManager::IsFiltering() const
{
	return m_pVolume != nullptr;
}

float FilteringManager::GetFilteringProgress() const
{
	if(m_pVolume == nullptr)
	{
		return 0.0f;
	}

	assert(!m_filteredVolume.empty());

	int channelCount = m_filteredVolume[0].GetChannelCount();

	float progress = 0.0f;
	// m_radiusIndex * DIR_COUNT * channelCount filtering passes done
	progress += float(m_radiusIndex) * float(DIR_COUNT) * float(channelCount);
	// in current radius: m_direction * channelCount filtering passes done
	progress += float(m_direction) * float(channelCount);
	// in current direction: m_channel filtering passes done
	progress += float(m_channel);
	// current filtering pass
	progress += float(m_nextBrickToFilter) / float(m_pVolume->GetBrickCount().volume());

	int numPasses = m_radiusCount * DIR_COUNT * channelCount;

	return progress / float(numPasses);
}


void FilteringManager::ReleaseResources()
{
	CancelFiltering();

	ReleaseVolumeDependentResources();
}


void FilteringManager::ClearResult()
{
	CancelFiltering();
	m_filteredVolume.clear();
}


bool FilteringManager::CreateVolumeDependentResources()
{
	assert(m_pVolume != nullptr);

	ReleaseVolumeDependentResources();

	m_pCompressShared = new GPUResources();
	m_pCompressVolume = new CompressVolumeResources();

	if (m_pVolume->IsCompressed())
	{
		uint brickSize = m_pVolume->GetBrickSizeWithOverlap();
		// do multi-channel decoding only for small bricks; for large bricks, mem usage gets too high
		uint channelCount = (brickSize <= 128) ? m_pVolume->GetChannelCount() : 1;
		uint huffmanBits = m_pVolume->GetHuffmanBitsMax();

		m_pCompressShared->create(CompressVolumeResources::getRequiredResources(brickSize, brickSize, brickSize, channelCount, huffmanBits));
		m_pCompressVolume->create(m_pCompressShared->getConfig());
	}

	m_brickSize = m_pVolume->GetBrickSizeWithOverlap();

	uint bufferSizeBytes = m_brickSize * m_brickSize * m_brickSize * sizeof(float);

	// allocate brick buffers
	cudaSafeCall(cudaMalloc2(&m_dpBufferCenter, bufferSizeBytes));
	cudaSafeCall(cudaMalloc2(&m_dpBufferLeft,   bufferSizeBytes));
	cudaSafeCall(cudaMalloc2(&m_dpBufferRight,  bufferSizeBytes));
	cudaSafeCall(cudaMalloc2(&m_dpBufferOut,    bufferSizeBytes));

	return true;
}

void FilteringManager::ReleaseVolumeDependentResources()
{
	if (m_pCompressShared)
	{
		m_pCompressShared->destroy();
		delete m_pCompressShared;
		m_pCompressShared = nullptr;
	}

	if (m_pCompressVolume)
	{
		m_pCompressVolume->destroy();
		delete m_pCompressVolume;
		m_pCompressVolume = nullptr;
	}

	cudaSafeCall(cudaFree(m_dpBufferOut));
	cudaSafeCall(cudaFree(m_dpBufferRight));
	cudaSafeCall(cudaFree(m_dpBufferLeft));
	cudaSafeCall(cudaFree(m_dpBufferCenter));

	m_dpBufferOut    = nullptr;
	m_dpBufferRight  = nullptr;
	m_dpBufferLeft   = nullptr;
	m_dpBufferCenter = nullptr;

	m_brickSize = 0;
}

namespace
{
	uint GetBrickLinearIndex(const Vec3i& brickPos, const Vec3i& brickCount, EFilterDirection dir)
	{
		uint result = 0;
		switch(dir) {
			case DIR_X:
				// XYZ
				result = brickPos.x() + brickCount.x() * (brickPos.y() + brickCount.y() * brickPos.z());
				break;
			case DIR_Y:
				// YXZ
				result = brickPos.y() + brickCount.y() * (brickPos.x() + brickCount.x() * brickPos.z());
				break;
			case DIR_Z:
				// ZXY
				result = brickPos.z() + brickCount.z() * (brickPos.x() + brickCount.x() * brickPos.y());
				break;
			default:
				assert(false);
		}
		return result;
	}

	Vec3i GetBrickPos(uint linearIndex, const Vec3i& brickCount, EFilterDirection dir)
	{
		Vec3i result;
		switch(dir) {
			case DIR_X:
				// XYZ
				result.x() = linearIndex % brickCount.x();
				result.y() = (linearIndex / brickCount.x()) % brickCount.y();
				result.z() = linearIndex / (brickCount.x() * brickCount.y());
				break;
			case DIR_Y:
				// YXZ
				result.x() = (linearIndex / brickCount.y()) % brickCount.x();
				result.y() = linearIndex % brickCount.y();
				result.z() = linearIndex / (brickCount.x() * brickCount.y());
				break;
			case DIR_Z:
				// ZXY
				result.x() = (linearIndex / brickCount.z()) % brickCount.x();
				result.y() = linearIndex / (brickCount.x() * brickCount.z());
				result.z() = linearIndex % brickCount.z();
				break;
			default:
				assert(false);
		}
		return result;
	}
}

uint FilteringManager::GetBrickLinearIndex(const Vec3i& pos, EFilterDirection dir)
{
	assert(m_pVolume != nullptr);

	return ::GetBrickLinearIndex(pos, m_pVolume->GetBrickCount(), dir);
}

Vec3i FilteringManager::GetBrickPos(uint linearIndex, EFilterDirection dir)
{
	assert(m_pVolume != nullptr);

	return ::GetBrickPos(linearIndex, m_pVolume->GetBrickCount(), dir);
}

const TimeVolumeIO::Brick* FilteringManager::GetBrick(const Vec3i& pos)
{
	Vec3i count = m_pVolume->GetBrickCount();
	if(pos.x() < 0 || pos.x() >= count.x() || pos.y() < 0 || pos.y() >= count.y() || pos.z() < 0 || pos.z() >= count.z()) {
		return nullptr;
	}

	const TimeVolumeIO::Brick* pBrick = m_bricks[pos.x() + count.x() * (pos.y() + count.y() * pos.z())];
	assert(pBrick->GetSpatialIndex() == pos);

	return pBrick;
}

bool FilteringManager::CompressDownload(GPUResources* pCompressShared, CompressVolumeResources* pCompressVolume, FilteredVolume::ChannelData& channelDataTarget, const float* dpBuffer, const Vec3ui& size, bool compressed)
{
	channelDataTarget.Clear(); //FIXME remove? should not be necessary

	if(compressed)
	{
		// find appropriate quant step
		float* dpValMax = nullptr;
		cudaSafeCall(cudaMalloc2(&dpValMax, sizeof(float)));
		cudaCompress::util::getMaxAbs(pCompressShared->m_pCuCompInstance, dpBuffer, size.volume(), dpValMax);
		float valMax;
		cudaSafeCall(cudaMemcpy(&valMax, dpValMax, sizeof(float), cudaMemcpyDeviceToHost));
		cudaSafeCall(cudaFree(dpValMax));

		const uint symbolMax = ((1 << 14) - 1) * 3 / 8; // 14 bits are allowed, but leave some room (values might get larger because of DWT)
		channelDataTarget.m_quantStep = 2.0f * valMax / float(symbolMax); // "2.0f *" because of signed/unsigned

		//printf("valmax: %.3f  quantstep: %.5f\n", valMax, channelDataTarget.m_quantStep);

		// compress
		std::vector<uint> bitStream;
		if(!compressVolumeFloat(*pCompressShared, *pCompressVolume, dpBuffer, size.x(), size.y(), size.z(), 2, bitStream, channelDataTarget.m_quantStep, false))
		{
			printf("FilteringManager::CompressDownload: compressVolumeFloat failed\n");
			return false;
		}
		else
		{
			channelDataTarget.Alloc(bitStream.size());
			memcpy(channelDataTarget.m_pData, bitStream.data(), channelDataTarget.m_dataSizeInUInts * sizeof(uint));
		}
	}
	else
	{
		// if volume is not compressed, download raw filtered data
		channelDataTarget.Alloc(size.volume()); // sizeof(uint) == sizeof(float)...
		cudaSafeCall(cudaMemcpy(channelDataTarget.m_pData, dpBuffer, size.volume() * sizeof(float), cudaMemcpyDeviceToHost));
	}

	return true;
}


namespace
{
	struct BrickSorter
	{
		BrickSorter(const Vec3i& brickCount, EFilterDirection dir) : m_brickCount(brickCount), m_dir(dir) {}

		bool operator() (const TimeVolumeIO::Brick* pBrick1, const TimeVolumeIO::Brick* pBrick2) const
		{
			assert(pBrick1);
			assert(pBrick2);

			uint index1 = GetBrickLinearIndex(pBrick1->GetSpatialIndex(), m_brickCount, m_dir);
			uint index2 = GetBrickLinearIndex(pBrick2->GetSpatialIndex(), m_brickCount, m_dir);

			return index1 < index2;
		}

	private:
		Vec3i m_brickCount;
		EFilterDirection m_dir;
	};
}

void FilteringManager::UpdateBricksToLoad()
{
	m_bricksToLoad.clear();

	std::vector<const TimeVolumeIO::Brick*> bricks = m_bricks;
	std::sort(bricks.begin(), bricks.end(), BrickSorter(m_pVolume->GetBrickCount(), m_direction));

	m_bricksToLoad.insert(m_bricksToLoad.end(), bricks.begin() + m_nextBrickToFilter, bricks.end());

	//TODO prefetching?
}


void FilteringManager::UpdateTimings()
{
	m_timings.UploadDecompressGPU = m_timerUploadDecompress.GetStats();
	m_timings.ConvolutionGPU      = m_timerConvolution.GetStats();
	m_timings.CompressDownloadGPU = m_timerCompressDownload.GetStats();

	m_timerFilter.Stop();
	m_timings.FilterWall = m_timerFilter.GetElapsedTimeMS();
}
