#ifndef __TUM3D__HEATMAP_H__
#define __TUM3D__HEATMAP_H__

#include "global.h"

#include <cuda_runtime.h>
#include <cuda_d3d11_interop.h>
#include <D3D11.h>

#include <memory>
#include <unordered_map>

class HeatMap
{
public:
	static class Channel
	{
	private:
		//The 3d buffer, written to with atomicAdd
		uint* m_pCudaBuffer;
		size_t m_count;

	public:
		Channel(size_t width, size_t height, size_t depth);
		~Channel();

		uint* getCudaBuffer() { return m_pCudaBuffer; }
		void clear();
	};
	typedef std::shared_ptr<Channel> Channel_ptr;

private:
	//the resolution
	const size_t m_width, m_height, m_depth;
	
	std::unordered_map<int, Channel_ptr> m_channels;

public:
	HeatMap(size_t width, size_t height, size_t depth);
	~HeatMap();

	//Get or creates the channel with the specified id
	Channel_ptr createChannel(int id);
	//Get the channel with the specified id, or nullptr if it does not exist
	Channel_ptr getChannel(int id);
	//Deletes the channel with the specified id, returns true if a channel was found
	bool deleteChannel(int id);
	//As the name suggests: deletes all channels
	void deleteAllChannels();
	//Sets the data of all channels to zero.
	void clearAllChannels();
	//returns the number of channels
	size_t getChannelCount();
};

#endif