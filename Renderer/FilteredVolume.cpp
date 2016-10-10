#include "FilteredVolume.h"

//#include <cuda_runtime.h>
//
//#include "cudaUtil.h"


using namespace tum3D;


FilteredVolume::ChannelData::ChannelData(ChannelData&& other)
{
	// copy other's state
	m_quantStep = other.m_quantStep;
	m_pData = other.m_pData;
	m_dataSizeInUInts = other.m_dataSizeInUInts;

	// clear other
	other.m_pData = nullptr;
	other.m_dataSizeInUInts = 0;
}

FilteredVolume::ChannelData::~ChannelData()
{
	Clear();
}


FilteredVolume::ChannelData& FilteredVolume::ChannelData::operator=(ChannelData&& other)
{
	if(this == &other) return *this;

	// release own resources
	//cudaSafeCall(cudaFreeHost(m_pData));
	delete[] m_pData;

	// copy other's state
	m_quantStep = other.m_quantStep;
	m_pData = other.m_pData;
	m_dataSizeInUInts = other.m_dataSizeInUInts;

	// clear other
	other.m_pData = nullptr;
	other.m_dataSizeInUInts = 0;

	return *this;
}


void FilteredVolume::ChannelData::Alloc(size_t sizeInUInts)
{
	if(m_dataSizeInUInts > 0) Clear();

	m_dataSizeInUInts = sizeInUInts;
	//cudaSafeCall(cudaMallocHost(&m_pData, m_dataSizeInUInts * sizeof(uint), cudaHostAllocPortable));
	m_pData = new uint[m_dataSizeInUInts];
}

void FilteredVolume::ChannelData::Clear()
{
	m_quantStep = 0.0f;
	//cudaSafeCall(cudaFreeHost(m_pData));
	delete[] m_pData;
	m_pData = nullptr;
	m_dataSizeInUInts = 0;
}


FilteredVolume::FilteredVolume()
	: m_brickCount(0, 0, 0), m_channelCount(0)
{
}

FilteredVolume::FilteredVolume(const Vec3i& brickCount, int channelCount)
	: m_brickCount(0, 0, 0), m_channelCount(0)
{
	Resize(brickCount, channelCount);
}

FilteredVolume::FilteredVolume(FilteredVolume&& other)
	: m_brickCount(other.m_brickCount), m_channelCount(other.m_channelCount)
{
	m_data.swap(other.m_data);
}


FilteredVolume& FilteredVolume::operator=(FilteredVolume&& other)
{
	if(this != &other)
	{
		m_brickCount = other.m_brickCount;
		m_channelCount = other.m_channelCount;
		m_data.swap(other.m_data);
	}

	return *this;
}


void FilteredVolume::Resize(const Vec3i& brickCount, int channelCount)
{
	Clear();

	m_brickCount = brickCount;
	m_channelCount = channelCount;

	m_data.resize(m_brickCount.volume() * m_channelCount);
}

void FilteredVolume::Clear()
{
	m_data.swap(std::vector<ChannelData>());
	//m_data.clear();
	//m_data.shrink_to_fit();
}


int FilteredVolume::GetLinearBrickIndex(const Vec3i& brickIndex) const
{
	return brickIndex.x() + m_brickCount.x() * (brickIndex.y() + m_brickCount.y() * brickIndex.z());
}

FilteredVolume::ChannelData& FilteredVolume::GetChannelData(const Vec3i& brickIndex, int channel)
{
	return m_data[GetLinearBrickIndex(brickIndex) * m_channelCount + channel];
}

const FilteredVolume::ChannelData& FilteredVolume::GetChannelData(const Vec3i& brickIndex, int channel) const
{
	return m_data[GetLinearBrickIndex(brickIndex) * m_channelCount + channel];
}

void FilteredVolume::ClearAllBricks()
{
	for(auto it = m_data.begin(); it != m_data.end(); it++)
	{
		it->Clear();
	}
}

size_t FilteredVolume::GetTotalSizeBytes() const
{
	size_t result = 0;

	for(auto it = m_data.begin(); it != m_data.end(); it++)
	{
		result += it->m_dataSizeInUInts * sizeof(uint);
	}

	return result;
}
