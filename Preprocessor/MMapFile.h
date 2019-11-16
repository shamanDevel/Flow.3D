#ifndef __MMAP_FILE_H__
#define __MMAP_FILE_H__

#include <windows.h>

class MMapFile
{
public:
	MMapFile();
	~MMapFile();
	void* openFile(const std::wstring& filename);
	void closeFile();
	inline size_t getFileSizeInBytes() { return m_ulFileSize; }

private:
	HANDLE m_hFile;
	HANDLE m_hFileMapping;
	LPVOID m_lpMapAddress;
	size_t m_ulFileSize;
};

#endif
