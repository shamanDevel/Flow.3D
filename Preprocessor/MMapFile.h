/**
 * @author Christoph Neuhauser
 */

#ifndef __MMAP_FILE_H__
#define __MMAP_FILE_H__

#include <string>
#include <windows.h>

/**
 * @class MMapFile represents a file mapped to the main memory.
 * This uses Window's file mapping functionality.
 * The pointer returned by "openFile" can be used like normal memory.
 * When memory is accessed that is not yet loaded, the operating system
 * generates an interrupt and loads memory of the size of a page.
 */
class MMapFile
{
public:
	MMapFile();
	~MMapFile();
	void* openFile(const std::string& filename);
	void closeFile();
	inline size_t getFileSizeInBytes() { return m_ulFileSize; }

private:
	HANDLE m_hFile;
	HANDLE m_hFileMapping;
	LPVOID m_lpMapAddress;
	size_t m_ulFileSize;
};

#endif
