#include "HeatMap.h"

#include "cudaUtil.h"

HeatMap::HeatMap(size_t width, size_t height, size_t depth)
	: m_width(width), m_height(height), m_depth(depth)
{
}

HeatMap::~HeatMap()
{
}

HeatMap::Channel_ptr HeatMap::createChannel(unsigned int id)
{
	auto it = m_channels.find(id);
	if (it == m_channels.end()) {
		Channel_ptr c = std::make_shared<Channel>(m_width, m_height, m_depth);
		m_channels[id] = c;
		return c;
	}
	else {
		return it->second;
	}
}

HeatMap::Channel_ptr HeatMap::getChannel(unsigned int id)
{
	auto it = m_channels.find(id);
	if (it == m_channels.end())
		return nullptr;
	else
		return it->second;
}

bool HeatMap::deleteChannel(unsigned int id)
{
	return m_channels.erase(id)==1;
}

void HeatMap::deleteAllChannels()
{
	m_channels.clear();
}

void HeatMap::clearAllChannels()
{
	for (auto& e : m_channels) {
		Channel_ptr c = e.second;
		c->clear();
	}
}

size_t HeatMap::getChannelCount()
{
	return m_channels.size();
}

std::vector<unsigned int> HeatMap::getAllChannelIDs() const
{
	std::vector<unsigned int> v;
	for (const auto& e : m_channels) {
		v.push_back(e.first);
	}
	return v;
}

HeatMap::Channel::Channel(size_t width, size_t height, size_t depth)
	: m_pCudaBuffer(nullptr)
{
	m_count = sizeof(uint) * width * height * depth;
	cudaSafeCall(cudaMalloc2(&m_pCudaBuffer, m_count));
}

HeatMap::Channel::~Channel()
{
	if (m_pCudaBuffer != nullptr) {
		cudaSafeCall(cudaFree(m_pCudaBuffer));
		m_pCudaBuffer = nullptr;
	}
}

void HeatMap::Channel::clear()
{
	cudaMemset(m_pCudaBuffer, 0, m_count);
}
