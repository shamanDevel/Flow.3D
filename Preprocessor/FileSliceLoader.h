#pragma once

#include <iostream>
#include <string>
#include <vector>
#include <queue>
#include <map>
#include <cstdio>
#include <cassert>

class FileSliceLoader {
public:
	FileSliceLoader() : m_file(nullptr), m_sizeX(0), m_sizeY(0), m_sizeZ(0), m_numChannels(0), m_sliceHeight(0), m_readGranularity(0),
		m_fileSizeBytes(0), m_lastUsedSliceIndex(-1), m_lastUsedSliceBuffer(nullptr) {}
	~FileSliceLoader();
	bool openFile(const std::string& filename, size_t bufferSizeBytes, size_t sliceHeight, int64_t sizeX, int64_t sizeY, int64_t sizeZ, int64_t numChannels);
	void closeFile();
	float getMemoryAt(int64_t x, int64_t y, int64_t z, int64_t channelIdx);

private:
	float* loadSlice(int64_t index);
	void unloadOldestSlice();

	FILE* m_file;
	size_t m_sizeX, m_sizeY, m_sizeZ;
	size_t m_numChannels;
	size_t m_sliceHeight;
	size_t m_readGranularity;
	int64_t m_fileSizeBytes;
	int64_t m_lastUsedSliceIndex;
	float* m_lastUsedSliceBuffer;
	std::vector<float*> m_sliceList;
	std::vector<float*> m_freeSliceStack;
	std::queue<std::pair<int64_t, float*>> m_pageLoadingQueue;
	std::map<int64_t, float*> m_loadedSlices;
};
