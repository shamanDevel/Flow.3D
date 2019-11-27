/**
 * @author Christoph Neuhauser
 */

#pragma once

#include <iostream>
#include <string>
#include <vector>
#include <queue>
#include <map>
#include <cstdio>
#include <cassert>

/**
 * @class FileSliceLoader is used for bricking volume data set files where the data is stored contiguously.
 * This class abstracts memory accesses by "getMemoryAt" and memory slices are implicitly loaded in the
 * background when needed. Loaded slices are also cached (if enough memory is available), as multiple
 * bricks will access the same slice.
 * This enables the user to process bricks one after another, still read large chunks of data linearly
 * from the file and reuse loaded data for other bricks.
 * It is assumed that the files contain floating point data in the following order: x -> y -> z -> channel
 */
class FileSliceLoader {
public:
	FileSliceLoader() : m_file(nullptr), m_sizeX(0), m_sizeY(0), m_sizeZ(0), m_virtualSizeY(0), m_numChannels(0), m_sliceHeight(0), m_readGranularity(0),
		m_fileSizeBytes(0), m_lastUsedSliceIndex(-1), m_lastUsedSliceBuffer(nullptr) {}
	~FileSliceLoader();

	/**
	 * Opens a volume data set file for reading.
	 * @param bufferSizeBytes The maximum amount of memory in bytes that may be used for buffering already loaded slices.
	 * @param sliceHeight The height of a slice in x-y. The width is exactly the width of the volume.
	 * @param sizeX The number of entries in the volume data in x direction.
	 * @param sizeY The number of entries in the volume data in y direction.
	 * @param sizeZ The number of entries in the volume data in z direction.
	 * @param numChannels The number of data channels.
	 * @return true if the file could be opened. false otherwise.
	 */
	bool openFile(const std::string& filename, size_t bufferSizeBytes, size_t sliceHeight, int64_t sizeX, int64_t sizeY, int64_t sizeZ, int64_t numChannels);

	/**
	 * Closes a previously opened file.
	 */
	void closeFile();

	/**
	 * Returns the value in the volume at the specified position/channel.
	 * @param x The global x position in the volume.
	 * @param y The global y position in the volume.
	 * @param z The global z position in the volume.
	 * @param channelIdx The channel to get the data from.
	 * @return The memory at the address.
	 */
	float getMemoryAt(int64_t x, int64_t y, int64_t z, int64_t channelIdx);

private:
	/**
	 * Loads a slice at a certain byte index from the file.
	 * @param index The byte position of the slice to load.
	 * @param sliceStartY The start position of the slice in y direction.
	 * @return The buffer of the loaded slice.
	 */
	float* loadSlice(int64_t index, int64_t sliceStartY);

	/**
	 * Unloads the oldest slice.
	 */
	void unloadOldestSlice();

	FILE* m_file;
	size_t m_sizeX, m_sizeY, m_sizeZ;
	size_t m_virtualSizeY; //< Assuming size in y is not divisible by slice height -> implicit padding
	size_t m_numChannels;
	size_t m_sliceHeight;
	size_t m_readGranularity; //< How much data is read with one call of "fread" (assuming no padding).
	int64_t m_fileSizeBytes;
	int64_t m_lastUsedSliceIndex; //< For fast access to last slice.
	float* m_lastUsedSliceBuffer; //< For fast access to last slice.
	std::vector<float*> m_sliceList; //< List of all slice buffers.
	std::vector<float*> m_freeSliceStack; //< List of unused slice buffers.
	std::queue<std::pair<int64_t, float*>> m_pageLoadingQueue; //< Queue of pages in loading order (for determining oldest slice when unloading).
	std::map<int64_t, float*> m_loadedSlices; //< A map slice byte index -> buffer of all loaded slices.
};
