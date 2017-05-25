#include "HeatMap.h"

#include "cudaUtil.h"

HeatMap::HeatMap(size_t width, size_t height, size_t depth)
	: m_width(width), m_height(height), m_depth(depth)
{
}

HeatMap::~HeatMap()
{
}

HeatMap::Channel_ptr HeatMap::createChannel(int id)
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

HeatMap::Channel_ptr HeatMap::getChannel(int id)
{
	auto it = m_channels.find(id);
	if (it == m_channels.end())
		return nullptr;
	else
		return it->second;
}

bool HeatMap::deleteChannel(int id)
{
	return m_channels.erase(id)==1;
}

void HeatMap::deleteAllChannels()
{
	m_channels.clear();
}

void HeatMap::clearAllChannels()
{
	size_t count = sizeof(uint) * m_width * m_height * m_depth;
	for (auto& e : m_channels) {
		Channel_ptr c = e.second;
		cudaMemset(c->getCudaBuffer(), 0, count);
	}
}

HeatMap::Channel::Channel(size_t width, size_t height, size_t depth)
	: m_pCudaBuffer(nullptr)
{
	cudaSafeCall(cudaMalloc(&m_pCudaBuffer, sizeof(uint) * width * height * depth));
}

HeatMap::Channel::~Channel()
{
	if (m_pCudaBuffer != nullptr) {
		cudaSafeCall(cudaFree(m_pCudaBuffer));
		m_pCudaBuffer = nullptr;
	}
}
