/**
 * @author Christoph Neuhauser
 */

#include <iostream>
#include "MMapFile.h"

MMapFile::MMapFile() : m_lpMapAddress(nullptr), m_ulFileSize(0ul)
{}

MMapFile::~MMapFile()
{
	if (m_lpMapAddress != nullptr) {
		closeFile();
	}
}

void* MMapFile::openFile(const std::string& filename)
{
	std::wstring wstrFileName(filename.begin(), filename.end());
	const wchar_t* pwcFileName = wstrFileName.c_str();
	m_hFile = CreateFile(pwcFileName, GENERIC_READ, FILE_SHARE_READ, NULL, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL | FILE_FLAG_SEQUENTIAL_SCAN, NULL);
	if (m_hFile == INVALID_HANDLE_VALUE) {
		std::cerr << "MMapFile::openFile: File \"" << filename << "\" could not be opened." << std::endl;
		return nullptr;
	}

	LARGE_INTEGER liFileSize;
	GetFileSizeEx(m_hFile, &liFileSize);
	m_ulFileSize = static_cast<size_t>(liFileSize.QuadPart);
	m_hFileMapping = CreateFileMapping(m_hFile, NULL, PAGE_READONLY, 0, 0, NULL);
	if (m_hFileMapping == 0) {
		std::cerr << "MMapFile::openFile: Couldn't create a file mapping for \"" << filename << "\"." << std::endl;
		CloseHandle(m_hFile);
		return nullptr;
	}

	m_lpMapAddress = MapViewOfFile(m_hFileMapping, FILE_MAP_READ, 0, 0, 0);
	return m_lpMapAddress;
}

void MMapFile::closeFile()
{
	UnmapViewOfFile(m_lpMapAddress);
	CloseHandle(m_hFileMapping);
	CloseHandle(m_hFile);
	m_lpMapAddress = nullptr;
	m_hFileMapping = nullptr;
	m_hFile = nullptr;
}
