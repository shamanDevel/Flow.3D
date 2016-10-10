#include "FilePool.h"

#include <cassert>


FilePool::FilePool(uint openFileCountLimit)
	: m_openFileCountLimit(openFileCountLimit), m_openFileCount(0)
{
}

FilePool::~FilePool()
{
	UnregisterAllFiles();
}


void FilePool::RegisterFile(const std::string& filename, uint id)
{
	FileDesc* pFileDesc = new FileDesc(id, filename);

	std::map<uint, FileDesc*>::iterator p = m_files.find(id);
	if (p != m_files.end())
	{
		delete p->second;
	}
	m_files[id] = pFileDesc;
}

void FilePool::UnregisterAllFiles()
{
	for(auto it = m_files.begin(); it != m_files.end(); it++)
	{
		FileDesc* pDesc = it->second;
		if (pDesc->m_usageCount != 0 || pDesc->m_isInOpenFileCache)
		{
			CloseHandle(pDesc->m_handle);
			delete pDesc;
		}
	}
	m_openFileCache.clear();
	m_openFileCount = 0;
	m_files.clear();
}


HANDLE FilePool::GetFileHandle(uint id)
{
	std::map<uint, FileDesc*>::iterator p = m_files.find(id);
	if (p == m_files.end())
	{
		printf("ERROR: FilePool::GetFileHandle: file with id %u not found!\n", id);
		return 0;
	}

	FileDesc* pFileDesc = p->second;

	if (pFileDesc->m_usageCount != 0)
	{
		// already open
		pFileDesc->m_usageCount++;
		return pFileDesc->m_handle;
	}

	if (pFileDesc->m_isInOpenFileCache)
	{
		// in cache - remove from there and return
		assert(pFileDesc->m_usageCount == 0);
		m_openFileCache.erase(pFileDesc->m_openFileCacheLocator);
		pFileDesc->m_isInOpenFileCache = false;
		pFileDesc->m_usageCount++;
		return pFileDesc->m_handle;
	}

	// still here -> file is not open yet

	// close files from cache if necessary
	if (m_openFileCount >= m_openFileCountLimit && !m_openFileCache.empty())
	{
		FileDesc* pFileToCloseDesc = m_openFileCache.front();
		assert(pFileToCloseDesc->m_usageCount == 0);
		m_openFileCache.pop_front();
		//printf("FilePool: Closing file %u\n", pFileToCloseDesc->m_id);
		pFileToCloseDesc->m_isInOpenFileCache = false;
		CloseHandle(pFileToCloseDesc->m_handle);
		pFileToCloseDesc->m_handle = INVALID_HANDLE_VALUE;
		m_openFileCount--;
	}

	// open file
	//printf("FilePool: Opening file %u\n", id);
	pFileDesc->m_handle = CreateFileA(pFileDesc->m_filename.c_str(), GENERIC_READ, FILE_SHARE_READ, 0,
		OPEN_EXISTING, FILE_FLAG_NO_BUFFERING | FILE_FLAG_OVERLAPPED, 0);

	assert(pFileDesc->m_handle != INVALID_HANDLE_VALUE);

	m_openFileCount++;
	pFileDesc->m_usageCount++;

	return pFileDesc->m_handle;
}


void FilePool::ReleaseFileHandle(uint id)
{
	std::map<uint, FileDesc*>::iterator p = m_files.find(id);
	if (p == m_files.end())
	{
		printf("ERROR: FilePool::ReleaseFileHandle: file with id %u not found!\n", id);
		return;
	}

	FileDesc* pFileDesc = p->second;

	assert(pFileDesc->m_usageCount > 0);

	pFileDesc->m_usageCount--;
	if (pFileDesc->m_usageCount == 0)
	{
		if (m_openFileCount <= m_openFileCountLimit)
		{
			m_openFileCache.push_back(pFileDesc);
			pFileDesc->m_openFileCacheLocator = m_openFileCache.end();
			pFileDesc->m_openFileCacheLocator--;
			pFileDesc->m_isInOpenFileCache = true;
		}
		else
		{
			assert(m_openFileCache.empty()); // otherwise we shouldn't be over the limit in the first place!
			CloseHandle(pFileDesc->m_handle);
			pFileDesc->m_handle = INVALID_HANDLE_VALUE;
			m_openFileCount--;
		}
	}
}
