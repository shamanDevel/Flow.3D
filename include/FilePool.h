#ifndef __TUM3D__FILEPOOL_H__
#define __TUM3D__FILEPOOL_H__


#include <global.h>

#include <list>
#include <map>
#include <memory>
#include <string>

#include <Windows.h>


class FilePool
{
public:
	FilePool(uint openFileCountLimit = 64);
	~FilePool();

	void RegisterFile(const std::string& filename, uint id);
	void UnregisterAllFiles(); // all files will be closed!

	HANDLE GetFileHandle(uint id);
	void ReleaseFileHandle(uint id);

private:
	struct FileDesc
	{
		FileDesc(uint id, const std::string& filename)
			: m_id(id), m_filename(filename), m_handle(INVALID_HANDLE_VALUE), m_usageCount(0), m_isInOpenFileCache(false) {}

		uint        m_id;
		std::string m_filename;
		HANDLE      m_handle;
		uint        m_usageCount;
		std::list<FileDesc*>::iterator m_openFileCacheLocator;
		bool        m_isInOpenFileCache;
	};
	// m_usageCount == 0 && !m_isInOpenFileCache => file is open
	// m_usageCount != 0 || m_isInOpenFileCache => file is closed

	uint						m_openFileCountLimit;

	uint						m_openFileCount;
	std::map<uint, FileDesc*>	m_files;
	// cache contains files with usageCount == 0 that are still kept open
	std::list<FileDesc*>		m_openFileCache;
};


#endif
