/**
 * @author Christoph Neuhauser
 */

#include "FileSliceLoader.h"

FileSliceLoader::~FileSliceLoader() {
	closeFile();
}

void FileSliceLoader::closeFile() {
	if (m_file != nullptr) {
		fclose(m_file);
		m_file = nullptr;

		// Clean-up of memory.
		for (auto& sliceBuffer : m_sliceList) {
			delete[] sliceBuffer;
		}
		m_sliceList.clear();
		m_freeSliceStack.clear();
		m_pageLoadingQueue = std::queue<std::pair<int64_t, float*>>();
		m_loadedSlices.clear();

		// Reset all member variables.
		m_sizeX = 0;
		m_sizeY = 0;
		m_sizeZ = 0;
		m_virtualSizeY = 0;
		m_numChannels = 0;
		m_sliceHeight = 0;
		m_readGranularity = 0;
		m_fileSizeBytes = 0;
		m_lastUsedSliceIndex = -1;
		m_lastUsedSliceBuffer = nullptr;
	}
}

/// Integer ceiling division of x by y.
inline int64_t iceil(int64_t x, int64_t y)
{
	return 1 + ((x - 1) / y);
}

bool FileSliceLoader::openFile(const std::string& filename, size_t bufferSizeBytes, size_t sliceHeight, int64_t sizeX, int64_t sizeY, int64_t sizeZ, int64_t numChannels) {
	closeFile();

	fopen_s(&m_file, filename.c_str(), "rb");
	if (m_file == nullptr) {
		std::cerr << "Couldn't open file \"" << filename << "\"!" << std::endl;
		return false;
	}

	m_sizeX = sizeX;
	m_sizeY = sizeY;
	m_sizeZ = sizeZ;
	m_numChannels = numChannels;
	m_sliceHeight = sliceHeight;
	m_readGranularity = sizeX * sliceHeight * numChannels * sizeof(float);

	// Determine the file size.
	_fseeki64(m_file, 0L, SEEK_END);
	m_fileSizeBytes = _ftelli64(m_file);
	_fseeki64(m_file, 0L, SEEK_SET);
	assert(m_fileSizeBytes == m_sizeX * m_sizeY * m_sizeZ * m_numChannels * sizeof(float));

	// Assuming size in y is not divisible by slice height -> implicit padding.
	m_virtualSizeY = iceil(int64_t(m_sizeY), int64_t(m_sliceHeight)) * m_sliceHeight;
	uint64_t maxNumSlices = iceil(int64_t(m_sizeX * m_virtualSizeY * m_sizeZ * m_numChannels * sizeof(float)), int64_t(m_readGranularity));

	// Reserve memory for buffering slices.
	int64_t numSlicesLoadedSimultaneously = std::min(bufferSizeBytes / m_readGranularity, maxNumSlices);
	for (int64_t i = 0; i < numSlicesLoadedSimultaneously; i++) {
		float* sliceBuffer = new float[m_readGranularity];
		m_sliceList.push_back(sliceBuffer);
		m_freeSliceStack.push_back(sliceBuffer);
	}

	return true;
}

float FileSliceLoader::getMemoryAt(int64_t x, int64_t y, int64_t z, int64_t channelIdx) {
	int64_t sliceOffsetY = y % m_sliceHeight; // Position in slice
	int64_t sliceStartY = y - sliceOffsetY; // Global start position
	int64_t sliceIndex = (sliceStartY + z * m_sizeY) * m_sizeX * m_numChannels * sizeof(float); // Byte start position
	int64_t sliceOffset = (x + sliceOffsetY * m_sizeX) * m_numChannels + channelIdx; // Read offset in buffer (in floats)

	// Same page accessed again?
	if (m_lastUsedSliceIndex == sliceIndex) {
		return m_lastUsedSliceBuffer[sliceOffset];
	}

	// Page already loaded?
	auto it = m_loadedSlices.find(sliceIndex);
	if (it == m_loadedSlices.end()) {
		m_lastUsedSliceBuffer = loadSlice(sliceIndex, sliceStartY);
	} else {
		m_lastUsedSliceBuffer = it->second;
	}
	m_lastUsedSliceIndex = sliceIndex;

	return m_lastUsedSliceBuffer[sliceOffset];
}

float* FileSliceLoader::loadSlice(int64_t index, int64_t sliceStartY) {
	while (m_freeSliceStack.empty()) {
		// Get memory for new page.
		unloadOldestSlice();
	}
	float* sliceBuffer = m_freeSliceStack.back();
	m_freeSliceStack.pop_back();

	// Use padding if volume height is not divisible by the slice height.
	size_t readSizeElements = m_readGranularity;
	if (sliceStartY + m_sliceHeight > m_sizeY) {
		size_t reducedSliceHeight = m_sizeY - sliceStartY;
		readSizeElements = m_sizeX * reducedSliceHeight * m_numChannels * sizeof(float);
	}

	// Read the data.
	_fseeki64(m_file, index, SEEK_SET);
	fread(sliceBuffer, readSizeElements, 1, m_file);

	// Insert into queue & add to loaded slice map.
	std::pair<int64_t, float*> insertionPair = std::make_pair(index, sliceBuffer);
	m_pageLoadingQueue.push(insertionPair);
	m_loadedSlices.insert(insertionPair);

	return sliceBuffer;
}

void FileSliceLoader::unloadOldestSlice() {
	// Unload slice at the front of the queue (i.e., insertion time longest ago).
	std::pair<int64_t, float*> unloadedSlice = m_pageLoadingQueue.front();
	m_pageLoadingQueue.pop();
	m_freeSliceStack.push_back(unloadedSlice.second);
	m_loadedSlices.erase(unloadedSlice.first);
}

