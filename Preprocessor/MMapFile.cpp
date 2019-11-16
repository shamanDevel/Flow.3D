#include "MMapFile.h"

MMapFile::MMapFile() : m_lpMapAddress(NULL), m_ulFileSize(0ul)
{}

MMapFile::~MMapFile()
{
	if (m_lpMapAddress != NULL) {
		closeFile();
	}
}

void* MMapFile::openFile(const std::wstring& filename)
{
	const wchar_t* pwcFileName = filename.c_str();
	m_hFile = CreateFile(pwcFileName, GENERIC_READ, FILE_SHARE_READ, NULL, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
	LARGE_INTEGER liFileSize;
	GetFileSizeEx(m_hFile, &liFileSize);
	m_ulFileSize = static_cast<size_t>(liFileSize.QuadPart);
	m_hFileMapping = CreateFileMapping(m_hFile, NULL, PAGE_READONLY, 0, 0, NULL);
	m_lpMapAddress = MapViewOfFile(hFileMapping, FILE_MAP_READ, 0, 0, 0);
	return m_lpMapAddress;
}

void* MMapFile::closeFile()
{
	UnmapViewOfFile(m_lpMapAddress);
	CloseHandle(m_hFileMapping);
	CloseHandle(m_hFile);
}
