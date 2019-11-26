#include "FileSliceLoader.h"

FileSliceLoader::~FileSliceLoader() {
	closeFile();
}

void FileSliceLoader::closeFile() {
	if (m_file != nullptr) {
		fclose(m_file);
		m_file = nullptr;

		for (auto& sliceBuffer : m_sliceList) {
			delete[] sliceBuffer;
		}
		m_sliceList.clear();
		m_freeSliceStack.clear();
		m_pageLoadingQueue = std::queue<std::pair<int64_t, float*>>();
		m_loadedSlices.clear();

		m_sizeX = 0;
		m_sizeY = 0;
		m_sizeZ = 0;
		m_numChannels = 0;
		m_sliceHeight = 0;
		m_readGranularity = 0;
		m_fileSizeBytes = 0;
		m_lastUsedSliceIndex = -1;
		m_lastUsedSliceBuffer = nullptr;
	}
}

bool FileSliceLoader::openFile(const std::string& filename, size_t bufferSizeBytes, size_t sliceHeight, int64_t sizeX, int64_t sizeY, int64_t sizeZ, int64_t numChannels) {
	closeFile();

	fopen_s(&m_file, filename.c_str(), "rb");
	if (m_file == nullptr) {
		std::cerr << "Couldn't open file \"" << filename << "\"!" << std::endl;
		return false;
	}

	_fseeki64(m_file, 0L, SEEK_END);
	m_fileSizeBytes = _ftelli64(m_file);
	_fseeki64(m_file, 0L, SEEK_SET);
	assert(m_fileSizeBytes == sizeX * sizeY * sizeZ * numChannels * sizeof(float));

	m_numChannels = numChannels;
	m_readGranularity = sizeX * sliceHeight * numChannels;
	int64_t numSlicesLoadedSimultaneously = bufferSizeBytes / m_readGranularity;
	for (int64_t i = 0; i < numSlicesLoadedSimultaneously; i++) {
		float* sliceBuffer = new float[m_readGranularity];
		m_sliceList.push_back(sliceBuffer);
		m_freeSliceStack.push_back(sliceBuffer);
	}

	return true;
}

float FileSliceLoader::getMemoryAt(int64_t x, int64_t y, int64_t z, int64_t channelIdx) {
	int64_t filePosBytes = (x + (y + z * m_sizeY) * m_sizeX) * sizeof(float) * m_numChannels;
	int64_t sliceOffset = filePosBytes % m_readGranularity;
	int64_t sliceIndex = filePosBytes - sliceOffset;

	if (m_lastUsedSliceIndex == sliceIndex) {
		return m_lastUsedSliceBuffer[sliceOffset];
	}

	auto it = m_loadedSlices.find(sliceIndex);
	if (it == m_loadedSlices.end()) {
		m_lastUsedSliceBuffer = loadSlice(sliceIndex);
	}
	else {
		m_lastUsedSliceBuffer = it->second;
	}
	m_lastUsedSliceIndex = sliceIndex;

	return m_lastUsedSliceBuffer[sliceOffset];
}

float* FileSliceLoader::loadSlice(int64_t index) {
	while (m_freeSliceStack.empty()) {
		unloadOldestSlice();
	}
	float* sliceBuffer = m_freeSliceStack.back();
	m_freeSliceStack.pop_back();

	size_t readSizeElements = m_readGranularity;
	if (index + int64_t(readSizeElements) > m_fileSizeBytes) {
		readSizeElements = m_fileSizeBytes - index;
	}

	_fseeki64(m_file, index, SEEK_SET);
	fread(sliceBuffer, readSizeElements, 1, m_file);

	std::pair<int64_t, float*> insertionPair = std::make_pair(index, sliceBuffer);
	m_pageLoadingQueue.push(insertionPair);
	m_loadedSlices.insert(insertionPair);

	return sliceBuffer;
}

void FileSliceLoader::unloadOldestSlice() {
	std::pair<int64_t, float*> unloadedSlice = m_pageLoadingQueue.back();
	m_pageLoadingQueue.pop();
	m_freeSliceStack.push_back(unloadedSlice.second);
	m_loadedSlices.erase(unloadedSlice.first);
}

