/*************************************************************************************

Author: Christian Dick, Marc Treib

Copyright (c) Christian Dick, Marc Treib

mailto:dick@in.tum.de
mailto:treib@in.tum.de


Changes Florian Reichl, 11/16/2011:
	- Fixed compilation error for data types other than unsigned char in CopyTo()
	- Added OpenHeader()

*************************************************************************************/

#ifndef __TUM3D__LARGEARRAY3D_H__
#define __TUM3D__LARGEARRAY3D_H__

#include <stdio.h>
#include <windows.h>
#include <list>
#include <vector>
#include <assert.h>



#define ___STR(x) #x
#define __STR(x) ___STR(x)
#define ___L(x) L ## x
#define __L(x) ___L(x)
#define __LOCATION__ L"(" __L(__FILE__) L":" __L(__STR(__LINE__)) L")"


typedef unsigned char    uchar;
typedef unsigned short   ushort;
typedef unsigned int     uint;
typedef unsigned __int64 uint64;



namespace LA3D
{

struct Page;

struct Frame
{
	BYTE*						m_pData;
	Page*						m_pPage; // NULL iff free
};

struct Page
{
	Frame*						m_pFrame; // NULL iff paged out
	unsigned int				m_uiPageX;
	unsigned int				m_uiPageY;
	unsigned int				m_uiPageZ;
	std::list<Page*>::iterator	m_pageCacheLocator;
	bool						m_bIsInitialized;
};

template <class T>
class LargeArray3D
{
public:
	LargeArray3D(unsigned int uiDim = 1);
	~LargeArray3D();

	// Create new array
	bool Create(unsigned int uiSizeX, unsigned int uiSizeY, unsigned int uiSizeZ, unsigned int uiPageSizeX, unsigned int uiPageSizeY, unsigned int uiPageSizeZ, const wchar_t* pwcFileName, unsigned __int64 ui64MemoryUsageLimit);
	// Open exisiting array
	bool Open(const wchar_t* pwcFileName, bool bReadOnly, unsigned __int64 ui64MemoryUsageLimit);

	// Added by Florian Reichl 11/16/2011
	// Opens and immediatly closes an LA3D file to read file properties; no memory is allocated
	bool OpenHeader(const wchar_t* pwcFileName);

	void SetMemoryUsageLimit(unsigned __int64 ui64MemoryUsageLimit);

	unsigned int GetSizeX() const { return m_uiSizeX; }
	unsigned int GetSizeY() const { return m_uiSizeY; }
	unsigned int GetSizeZ() const { return m_uiSizeZ; }
	unsigned int GetPageSizeX() const { return m_uiPageSizeX; }
	unsigned int GetPageSizeY() const { return m_uiPageSizeY; }
	unsigned int GetPageSizeZ() const { return m_uiPageSizeZ; }
	unsigned int GetNumPagesX() const { return m_uiNumPagesX; }
	unsigned int GetNumPagesY() const { return m_uiNumPagesY; }
	unsigned int GetNumPagesZ() const { return m_uiNumPagesZ; }
	bool IsOpen() const { return m_bIsOpen; }

	T* Get(unsigned int uiX, unsigned int uiY, unsigned int uiZ);
	void Initialize(const T* pDefaultValue = NULL);

	void CopyTo(   void* pDest,															// destination pointer
	               unsigned int uiX,     unsigned int uiY,      unsigned int uiZ,		// position of the source block in this large array
	               unsigned int uiWidth, unsigned int uiHeight, unsigned int uiDepth,	// size of the block to be copied
	               unsigned int uiDestWidth = 0, unsigned int uiDestHeight = 0);		// width and height of entire destination buffer (not of the block to be copied!)

	void CopyFrom( void* pSource,														// source pointer
	               unsigned int uiX,     unsigned int uiY,      unsigned int uiZ,		// position of the destination block in this large array
	               unsigned int uiWidth, unsigned int uiHeight, unsigned int uiDepth,	// size of the block to be copied
	               unsigned int uiSourceWidth = 0, unsigned int uiSourceHeight = 0);	// width and height of entire source buffer (not of the block to be copied!)

	bool ConvertToRAW(const wchar_t* pwcFileName);

	// if bDiscard==true, close without saving any changes
	void Close(bool bDiscard = false);

private:
	void WriteHeader();
	bool ReadHeader();
	void WriteInitFlags();
	void ReadInitFlags();

	static bool ToggleManageVolumePrivilege(bool bEnable);

	Frame* AllocateFrame();
	void DisposeFrame(Frame* pFrame);

	void PageIn(Page* pPage);
	void PageOut(Page* pPage, bool bDiscard = false);

	void GetPage(unsigned int uiPageX, unsigned int uiPageY, unsigned int uiPageZ);

	unsigned int		m_uiDim;

	bool				m_bIsOpen;

	bool				m_bReadOnly;
	DWORD				m_dwSystemPageSize;

	unsigned int		m_uiSizeX, m_uiSizeY, m_uiSizeZ;
	unsigned int		m_uiPageSizeX, m_uiPageSizeY, m_uiPageSizeZ;
	unsigned int		m_uiNumPagesX, m_uiNumPagesY, m_uiNumPagesZ;

	unsigned int		m_uiInitFlagsMemorySize;
	unsigned int		m_uiPageMemorySize;

	T*					m_defaultValue;

	HANDLE				m_hFile;

	std::vector<Frame*>	m_pFrames;
	std::vector<Frame*>	m_pFreeFrames;

	Page*				m_pages;
	std::list<Page*>	m_pageCache;

	unsigned int		m_uiLastPageX, m_uiLastPageY, m_uiLastPageZ;
	BYTE*				m_pLastPageData;
};


template<class T>
LargeArray3D<T>::LargeArray3D(unsigned int uiDim)
: m_bIsOpen(false), m_uiDim(uiDim)
{
	m_defaultValue = new T[m_uiDim];
}


template<class T>
bool LargeArray3D<T>::Create(unsigned int uiSizeX, unsigned int uiSizeY, unsigned int uiSizeZ, unsigned int uiPageSizeX, unsigned int uiPageSizeY, unsigned int uiPageSizeZ, const wchar_t* pwcFileName, unsigned __int64 ui64MemoryUsageLimit)
{
	assert(uiSizeX > 0);
	assert(uiSizeY > 0);
	assert(uiSizeZ > 0);
	assert(uiPageSizeX > 0);
	assert(uiPageSizeY > 0);
	assert(uiPageSizeZ > 0);
	
	if (m_bIsOpen)
	{
		Close();
	}

	m_bReadOnly = false;

	SYSTEM_INFO systemInfo;
	GetSystemInfo(&systemInfo);
	m_dwSystemPageSize = systemInfo.dwPageSize;

	m_uiSizeX = uiSizeX;
	m_uiSizeY = uiSizeY;
	m_uiSizeZ = uiSizeZ;
	m_uiPageSizeX = uiPageSizeX;
	m_uiPageSizeY = uiPageSizeY;
	m_uiPageSizeZ = uiPageSizeZ;
	m_uiNumPagesX = (m_uiSizeX + m_uiPageSizeX - 1) / m_uiPageSizeX;
	m_uiNumPagesY = (m_uiSizeY + m_uiPageSizeY - 1) / m_uiPageSizeY;
	m_uiNumPagesZ = (m_uiSizeZ + m_uiPageSizeZ - 1) / m_uiPageSizeZ;
	
	m_uiInitFlagsMemorySize = ((sizeof(T) + sizeof(bool) * m_uiNumPagesX * m_uiNumPagesY * m_uiNumPagesZ + m_dwSystemPageSize - 1) / m_dwSystemPageSize) * m_dwSystemPageSize;
	m_uiPageMemorySize = ((m_uiDim * sizeof(T) * m_uiPageSizeX * m_uiPageSizeY * m_uiPageSizeZ + m_dwSystemPageSize - 1) / m_dwSystemPageSize) * m_dwSystemPageSize;

	wprintf_s(L"Page memory size: %.1f MB\n", static_cast<double>(m_uiPageMemorySize) / (1024.0 * 1024.0));

	ZeroMemory(m_defaultValue, m_uiDim * sizeof(T));

	// Create file

	// Enable Manage Volume privilege, if present in the access token associated with the process
	// This privilege is required by the SetFileValidData function which is used to avoid filling the file with zeroes
	// The privilege has to be enabled _before_ creating the file
	
	bool bHasManageVolumePrivilege = ToggleManageVolumePrivilege(true);

	if (!bHasManageVolumePrivilege)
	{
		wprintf_s(L"WARNING: Running without Manage Volume privilege\n");
	}

	m_hFile = CreateFile(pwcFileName, GENERIC_READ | GENERIC_WRITE, FILE_SHARE_READ, NULL, CREATE_ALWAYS, FILE_FLAG_NO_BUFFERING, NULL);
	if (m_hFile == INVALID_HANDLE_VALUE)
	{
		DWORD error = GetLastError();
		wprintf_s(L"ERROR: Error %u trying to create file %s " __LOCATION__ L"\n", unsigned int(error), pwcFileName);
		return false;
	}

	// Extend file

	LARGE_INTEGER liPos, liNewPos;
	liPos.QuadPart = static_cast<__int64>(m_dwSystemPageSize) + static_cast<__int64>(m_uiInitFlagsMemorySize) + static_cast<__int64>(m_uiPageMemorySize) * static_cast<__int64>(m_uiNumPagesX * m_uiNumPagesY * m_uiNumPagesZ);
	BOOL bResult = SetFilePointerEx(m_hFile, liPos, &liNewPos, FILE_BEGIN);
	if (!bResult || liNewPos.QuadPart != liPos.QuadPart)
	{
		wprintf_s(L"ERROR: SetFilePointerEx failed " __LOCATION__ L"\n");
		exit(-1);
	}

	bResult = SetEndOfFile(m_hFile); // Set physical end of file
	if (!bResult)
	{
		wprintf_s(L"ERROR: SetEndOfFile failed " __LOCATION__ L"\n");
		exit(-1);
	}

	if (bHasManageVolumePrivilege)
	{
		bResult = SetFileValidData(m_hFile, liPos.QuadPart); // Set logical end of file to avoid filling the file with zeros
		if (!bResult) // SetFileValidData fails on network files
		{
			DWORD dwError = GetLastError();
			wprintf_s(L"WARNING: SetFileValidData failed (Error: %u) " __LOCATION__ L"\n", dwError);
		}
	}

	ToggleManageVolumePrivilege(false);

	WriteHeader();

	m_pages = new Page[m_uiNumPagesX * m_uiNumPagesY * m_uiNumPagesZ];
	ZeroMemory(m_pages, sizeof(Page) * m_uiNumPagesX * m_uiNumPagesY * m_uiNumPagesZ);
	for (unsigned int uiPageZ = 0; uiPageZ < m_uiNumPagesZ; uiPageZ++)
	{
		for (unsigned int uiPageY = 0; uiPageY < m_uiNumPagesY; uiPageY++)
		{
			for (unsigned int uiPageX = 0; uiPageX < m_uiNumPagesX; uiPageX++)
			{
				Page* pPage = &m_pages[m_uiNumPagesY * m_uiNumPagesX * uiPageZ + m_uiNumPagesX * uiPageY + uiPageX];
				pPage->m_uiPageX = uiPageX;
				pPage->m_uiPageY = uiPageY; 
				pPage->m_uiPageZ = uiPageZ; 
			}
		}
	}

	m_uiLastPageX = ~0u; // -1
	m_uiLastPageY = ~0u; // -1
	m_uiLastPageZ = ~0u; // -1
	m_pLastPageData = NULL;

	SetMemoryUsageLimit(ui64MemoryUsageLimit);

	m_bIsOpen = true;

	return true;
}


template<class T>
bool LargeArray3D<T>::Open(const wchar_t* pwcFileName, bool bReadOnly, unsigned __int64 ui64MemoryUsageLimit)
{
	if (m_bIsOpen)
	{
		Close();
	}

	m_bReadOnly = bReadOnly;

	SYSTEM_INFO systemInfo;
	GetSystemInfo(&systemInfo);
	m_dwSystemPageSize = systemInfo.dwPageSize;

	// Open file

	DWORD dwDesiredAccess = GENERIC_READ;
	if (!m_bReadOnly) { dwDesiredAccess |= GENERIC_WRITE; }
	m_hFile = CreateFile(pwcFileName, dwDesiredAccess, FILE_SHARE_READ, NULL, OPEN_EXISTING, FILE_FLAG_NO_BUFFERING, NULL);
	if (m_hFile == INVALID_HANDLE_VALUE)
	{
		//wprintf_s(L"ERROR: Cannot open file %s " __LOCATION__ L"\n", pwcFileName);
		return false;
	}

	if (!ReadHeader())
	{
		return false;
	}

	m_uiNumPagesX = (m_uiSizeX + m_uiPageSizeX - 1) / m_uiPageSizeX;
	m_uiNumPagesY = (m_uiSizeY + m_uiPageSizeY - 1) / m_uiPageSizeY;
	m_uiNumPagesZ = (m_uiSizeZ + m_uiPageSizeZ - 1) / m_uiPageSizeZ;

	m_uiInitFlagsMemorySize = ((sizeof(T) + sizeof(bool) * m_uiNumPagesX * m_uiNumPagesY * m_uiNumPagesZ + m_dwSystemPageSize - 1) / m_dwSystemPageSize) * m_dwSystemPageSize;
	m_uiPageMemorySize = ((m_uiDim * sizeof(T) * m_uiPageSizeX * m_uiPageSizeY * m_uiPageSizeZ + m_dwSystemPageSize - 1) / m_dwSystemPageSize) * m_dwSystemPageSize;
	
	wprintf_s(L"Page memory size: %.1f MB\n", static_cast<double>(m_uiPageMemorySize) / (1024.0 * 1024.0));

	m_pages = new Page[m_uiNumPagesX * m_uiNumPagesY * m_uiNumPagesZ];
	ZeroMemory(m_pages, sizeof(Page) * m_uiNumPagesX * m_uiNumPagesY * m_uiNumPagesZ);
	for (unsigned int uiPageZ = 0; uiPageZ < m_uiNumPagesZ; uiPageZ++)
	{
		for (unsigned int uiPageY = 0; uiPageY < m_uiNumPagesY; uiPageY++)
		{
			for (unsigned int uiPageX = 0; uiPageX < m_uiNumPagesX; uiPageX++)
			{
				Page* pPage = &m_pages[m_uiNumPagesY * m_uiNumPagesX * uiPageZ + m_uiNumPagesX * uiPageY + uiPageX];
				pPage->m_uiPageX = uiPageX;
				pPage->m_uiPageY = uiPageY; 
				pPage->m_uiPageZ = uiPageZ; 
			}
		}
	}

	ReadInitFlags();

	m_uiLastPageX = ~0u; // -1
	m_uiLastPageY = ~0u; // -1
	m_uiLastPageZ = ~0u; // -1
	m_pLastPageData = NULL;
	
	SetMemoryUsageLimit(ui64MemoryUsageLimit);

	m_bIsOpen = true;

	return true;
}


template<class T>
void LargeArray3D<T>::WriteHeader()
{
	LARGE_INTEGER liPos, liNewPos;
	liPos.QuadPart = 0;
	BOOL bResult = SetFilePointerEx(m_hFile, liPos, &liNewPos, FILE_BEGIN);
	if (!bResult || liNewPos.QuadPart != liPos.QuadPart)
	{
		wprintf_s(L"ERROR: SetFilePointerEx failed " __LOCATION__ L"\n");
		exit(-1);
	}

	BYTE* pHeader = (BYTE*)VirtualAlloc(NULL, m_dwSystemPageSize, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
	if (pHeader == NULL)
	{
		wprintf_s(L"ERROR: VirtualAlloc failed " __LOCATION__ L"\n");
		exit(-1);
	}

	ZeroMemory(pHeader, m_dwSystemPageSize);
	BYTE* ptr = pHeader;
	memcpy_s(ptr, m_dwSystemPageSize, "TREIB-LARGEARRAY3D", 18 * sizeof(char));
	ptr += sizeof(char) * 18;
	*reinterpret_cast<DWORD*>(ptr) = m_dwSystemPageSize;
	ptr += sizeof(DWORD);
	*reinterpret_cast<unsigned int*>(ptr) = sizeof(T);
	ptr += sizeof(unsigned int);
	*reinterpret_cast<unsigned int*>(ptr) = m_uiDim;
	ptr += sizeof(unsigned int);
	*reinterpret_cast<unsigned int*>(ptr) = m_uiSizeX;
	ptr += sizeof(unsigned int);
	*reinterpret_cast<unsigned int*>(ptr) = m_uiSizeY;
	ptr += sizeof(unsigned int);
	*reinterpret_cast<unsigned int*>(ptr) = m_uiSizeZ;
	ptr += sizeof(unsigned int);
	*reinterpret_cast<unsigned int*>(ptr) = m_uiPageSizeX;
	ptr += sizeof(unsigned int);
	*reinterpret_cast<unsigned int*>(ptr) = m_uiPageSizeY;
	ptr += sizeof(unsigned int);
	*reinterpret_cast<unsigned int*>(ptr) = m_uiPageSizeZ;
	ptr += sizeof(unsigned int);

	DWORD dwNumberOfBytesWritten;
	bResult = WriteFile(m_hFile, pHeader, m_dwSystemPageSize, &dwNumberOfBytesWritten, NULL);
	if (!bResult || dwNumberOfBytesWritten != m_dwSystemPageSize)
	{
		wprintf_s(L"ERROR: WriteFile failed " __LOCATION__ L"\n");
		exit(-1);
	}

	bResult = VirtualFree(pHeader, 0, MEM_RELEASE);
	if (!bResult)
	{
		wprintf_s(L"ERROR: VirtualFree failed " __LOCATION__ L"\n");
		exit(-1);
	}
}


template<class T>
bool LargeArray3D<T>::ReadHeader()
{
	LARGE_INTEGER liPos, liNewPos;
	liPos.QuadPart = 0;
	BOOL bResult = SetFilePointerEx(m_hFile, liPos, &liNewPos, FILE_BEGIN);
	if (!bResult || liNewPos.QuadPart != liPos.QuadPart)
	{
		wprintf_s(L"ERROR: SetFilePointerEx failed " __LOCATION__ L"\n");
		exit(-1);
	}

	BYTE* pHeader = (BYTE*)VirtualAlloc(NULL, m_dwSystemPageSize, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
	if (pHeader == NULL)
	{
		wprintf_s(L"ERROR: VirtualAlloc failed " __LOCATION__ L"\n");
		exit(-1);
	}

	DWORD dwNumberOfBytesRead;
	bResult = ReadFile(m_hFile, pHeader, m_dwSystemPageSize, &dwNumberOfBytesRead, NULL);
	if (!bResult || dwNumberOfBytesRead != m_dwSystemPageSize)
	{
		wprintf_s(L"ERROR: ReadFile failed " __LOCATION__ L"\n");
		exit(-1);
	}

	BYTE* ptr = pHeader;
	if (strncmp(reinterpret_cast<char*>(ptr), "TREIB-LARGEARRAY3D", 18) != 0)
	{
		//wprintf_s(L"ERROR: Signature check failed " __LOCATION__ L"\n");
		bResult = VirtualFree(pHeader, 0, MEM_RELEASE);
		if (!bResult)
		{
			wprintf_s(L"ERROR: VirtualFree failed " __LOCATION__ L"\n");
			exit(-1);
		}
		return false;
	}
	ptr += sizeof(char) * 18;
	if (*reinterpret_cast<DWORD*>(ptr) != m_dwSystemPageSize)
	{
		//wprintf_s(L"ERROR: System page size check failed " __LOCATION__ L"\n");
		bResult = VirtualFree(pHeader, 0, MEM_RELEASE);
		if (!bResult)
		{
			wprintf_s(L"ERROR: VirtualFree failed " __LOCATION__ L"\n");
			exit(-1);
		}
		return false;
	}
	ptr += sizeof(DWORD);
	if (*reinterpret_cast<unsigned int*>(ptr) != sizeof(T))
	{
		//wprintf_s(L"ERROR: Type size check failed " __LOCATION__ L"\n");
		bResult = VirtualFree(pHeader, 0, MEM_RELEASE);
		if (!bResult)
		{
			wprintf_s(L"ERROR: VirtualFree failed " __LOCATION__ L"\n");
			exit(-1);
		}
		return false;
	}
	ptr += sizeof(unsigned int);
	if (*reinterpret_cast<unsigned int*>(ptr) != m_uiDim)
	{
		//wprintf_s(L"ERROR: Dimension check failed " __LOCATION__ L"\n");
		bResult = VirtualFree(pHeader, 0, MEM_RELEASE);
		if (!bResult)
		{
			wprintf_s(L"ERROR: VirtualFree failed " __LOCATION__ L"\n");
			exit(-1);
		}
		return false;
	}
	ptr += sizeof(unsigned int);
	m_uiSizeX = *reinterpret_cast<unsigned int*>(ptr);
	ptr += sizeof(unsigned int);
	m_uiSizeY = *reinterpret_cast<unsigned int*>(ptr);
	ptr += sizeof(unsigned int);
	m_uiSizeZ = *reinterpret_cast<unsigned int*>(ptr);
	ptr += sizeof(unsigned int);
	m_uiPageSizeX = *reinterpret_cast<unsigned int*>(ptr);
	ptr += sizeof(unsigned int);
	m_uiPageSizeY = *reinterpret_cast<unsigned int*>(ptr);
	ptr += sizeof(unsigned int);
	m_uiPageSizeZ = *reinterpret_cast<unsigned int*>(ptr);
	ptr += sizeof(unsigned int);

	bResult = VirtualFree(pHeader, 0, MEM_RELEASE);
	if (!bResult)
	{
		wprintf_s(L"ERROR: VirtualFree failed " __LOCATION__ L"\n");
		exit(-1);
	}

	return true;
}


template<class T>
void LargeArray3D<T>::WriteInitFlags()
{
	LARGE_INTEGER liPos, liNewPos;
	liPos.QuadPart = m_dwSystemPageSize;
	BOOL bResult = SetFilePointerEx(m_hFile, liPos, &liNewPos, FILE_BEGIN);
	if (!bResult || liNewPos.QuadPart != liPos.QuadPart)
	{
		wprintf_s(L"ERROR: SetFilePointerEx failed " __LOCATION__ L"\n");
		exit(-1);
	}

	BYTE* pInitFlags = (BYTE*)VirtualAlloc(NULL, m_uiInitFlagsMemorySize, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
	if (pInitFlags == NULL)
	{
		wprintf_s(L"ERROR: VirtualAlloc failed " __LOCATION__ L"\n");
		exit(-1);
	}

	ZeroMemory(pInitFlags, m_uiInitFlagsMemorySize);
	BYTE* ptr = pInitFlags;
	for (unsigned int ui = 0; ui < m_uiDim; ui++)
	{
		*reinterpret_cast<T*>(ptr) = m_defaultValue[ui];
		ptr += sizeof(T);
	}
	for (unsigned int uiPage = 0; uiPage < m_uiNumPagesX * m_uiNumPagesY * m_uiNumPagesZ; uiPage++)
	{
		*reinterpret_cast<bool*>(ptr) = m_pages[uiPage].m_bIsInitialized;
		ptr += sizeof(bool);
	}

	DWORD dwNumberOfBytesWritten;
	bResult = WriteFile(m_hFile, pInitFlags, m_uiInitFlagsMemorySize, &dwNumberOfBytesWritten, NULL);
	if (!bResult || dwNumberOfBytesWritten != m_uiInitFlagsMemorySize)
	{
		wprintf_s(L"ERROR: WriteFile failed " __LOCATION__ L"\n");
		exit(-1);
	}

	bResult = VirtualFree(pInitFlags, 0, MEM_RELEASE);
	if (!bResult)
	{
		wprintf_s(L"ERROR: VirtualFree failed " __LOCATION__ L"\n");
		exit(-1);
	}
}


template<class T>
void LargeArray3D<T>::ReadInitFlags()
{
	LARGE_INTEGER liPos, liNewPos;
	liPos.QuadPart = m_dwSystemPageSize;
	BOOL bResult = SetFilePointerEx(m_hFile, liPos, &liNewPos, FILE_BEGIN);
	if (!bResult || liNewPos.QuadPart != liPos.QuadPart)
	{
		wprintf_s(L"ERROR: SetFilePointerEx failed " __LOCATION__ L"\n");
		exit(-1);
	}

	BYTE* pInitFlags = (BYTE*)VirtualAlloc(NULL, m_uiInitFlagsMemorySize, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
	if (pInitFlags == NULL)
	{
		wprintf_s(L"ERROR: VirtualAlloc failed " __LOCATION__ L"\n");
		exit(-1);
	}

	DWORD dwNumberOfBytesRead;
	bResult = ReadFile(m_hFile, pInitFlags, m_uiInitFlagsMemorySize, &dwNumberOfBytesRead, NULL);
	if (!bResult || dwNumberOfBytesRead != m_uiInitFlagsMemorySize)
	{
		wprintf_s(L"ERROR: ReadFile failed " __LOCATION__ L"\n");
		exit(-1);
	}

	BYTE* ptr = pInitFlags;
	for (unsigned int ui = 0; ui < m_uiDim; ui++)
	{
		m_defaultValue[ui] = *reinterpret_cast<T*>(ptr);
		ptr += sizeof(T);
	}
	for (unsigned int uiPage = 0; uiPage < m_uiNumPagesX * m_uiNumPagesY * m_uiNumPagesZ; uiPage++)
	{
		m_pages[uiPage].m_bIsInitialized = *reinterpret_cast<bool*>(ptr);
		ptr += sizeof(bool);
	}

	bResult = VirtualFree(pInitFlags, 0, MEM_RELEASE);
	if (!bResult)
	{
		wprintf_s(L"ERROR: VirtualFree failed " __LOCATION__ L"\n");
		exit(-1);
	}
}


template<class T>
bool LargeArray3D<T>::ToggleManageVolumePrivilege(bool bEnable)
{
	HANDLE hToken;
	BOOL bResult = OpenProcessToken(GetCurrentProcess(), TOKEN_ALL_ACCESS, &hToken);
	if (!bResult)
	{
		wprintf_s(L"ERROR: OpenProcessToken failed " __LOCATION__ L"\n");
		exit(-1);
	}

	LUID luid;
	bResult = LookupPrivilegeValue(NULL, SE_MANAGE_VOLUME_NAME, &luid);
	if (!bResult)
	{
		wprintf_s(L"ERROR: LookupPrivilegeValue failed " __LOCATION__ L"\n");
		exit(-1);
	}

	TOKEN_PRIVILEGES tp;
	tp.PrivilegeCount = 1;
	tp.Privileges[0].Luid = luid;
	tp.Privileges[0].Attributes = bEnable ? SE_PRIVILEGE_ENABLED : 0;
	bResult = AdjustTokenPrivileges(hToken, FALSE, &tp, sizeof(TOKEN_PRIVILEGES), NULL, NULL);
	if (!bResult)
	{
		wprintf_s(L"ERROR: AdjustTokenPrivileges failed " __LOCATION__ L"\n");
		exit(-1);
	}

	bool bHasManageVolumePrivilege = GetLastError() != ERROR_NOT_ALL_ASSIGNED;

	bResult = CloseHandle(hToken);
	if (!bResult)
	{
		wprintf_s(L"ERROR: CloseHandle failed " __LOCATION__ L"\n");
		exit(-1);
	}

	return bHasManageVolumePrivilege;
}


template<class T>
void LargeArray3D<T>::SetMemoryUsageLimit(unsigned __int64 ui64MemoryUsageLimit)
{
	unsigned int uiNumFrames = max(static_cast<unsigned int>(ui64MemoryUsageLimit / static_cast<unsigned __int64>(m_uiPageMemorySize)), 1u);
	uiNumFrames = min(uiNumFrames, m_uiNumPagesX * m_uiNumPagesY * m_uiNumPagesZ);

	if (uiNumFrames < m_pFrames.size())
	{
		unsigned int uiNumFramesToDispose = static_cast<unsigned int>(m_pFrames.size()) - uiNumFrames;
		for (unsigned int ui = 0; ui < uiNumFramesToDispose; ui++)
		{
			Frame* pFrame = m_pFrames.back();
			m_pFrames.pop_back();
			if (pFrame->m_pPage != NULL)
			{
				PageOut(pFrame->m_pPage);
			}
			for (std::vector<Frame*>::iterator p = m_pFreeFrames.begin(); p != m_pFreeFrames.end(); p++)
			{
				if (*p == pFrame)
				{
					m_pFreeFrames.erase(p);
					break;
				}
			}
			DisposeFrame(pFrame);
		}
		m_uiLastPageX = ~0u; // -1
		m_uiLastPageY = ~0u; // -1
		m_uiLastPageZ = ~0u; // -1
		m_pLastPageData = NULL;
	}
	else
	{
		unsigned int uiNumFramesToAllocate = uiNumFrames - static_cast<unsigned int>(m_pFrames.size());
		for (unsigned int ui = 0; ui < uiNumFramesToAllocate; ui++)
		{
			Frame* pFrame = AllocateFrame();
			m_pFrames.push_back(pFrame);
			m_pFreeFrames.push_back(pFrame);
		}
	}

	wprintf_s(L"Number of frames allocated: %u\n", uiNumFrames);
}


template<class T>
Frame* LargeArray3D<T>::AllocateFrame()
{
	Frame* pFrame = new Frame();
	ZeroMemory(pFrame, sizeof(Frame));

	pFrame->m_pData = (BYTE*)VirtualAlloc(NULL, m_uiPageMemorySize, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
	if (pFrame->m_pData == NULL)
	{
		wprintf_s(L"ERROR: VirtualAlloc failed " __LOCATION__ L"\n");
		exit(-1);
	}
	ZeroMemory(pFrame->m_pData, m_uiPageMemorySize);

	return pFrame;
}


template<class T>
void LargeArray3D<T>::DisposeFrame(Frame* pFrame)
{
	BOOL bResult = VirtualFree(pFrame->m_pData, 0, MEM_RELEASE);
	if (!bResult)
	{
		wprintf_s(L"ERROR: VirtualFree failed " __LOCATION__ L"\n");
		exit(-1);
	}

	delete pFrame;
}


template<class T>
void LargeArray3D<T>::PageIn(Page* pPage)
{
	assert(pPage->m_pFrame == NULL);
	assert(m_pFreeFrames.size() > 0);

	Frame* pFrame = m_pFreeFrames.back();
	m_pFreeFrames.pop_back();

	pFrame->m_pPage = pPage;
	pPage->m_pFrame = pFrame;

	if (pPage->m_bIsInitialized)
	{
		LARGE_INTEGER liPos, liNewPos;
		liPos.QuadPart = static_cast<__int64>(m_dwSystemPageSize) + static_cast<__int64>(m_uiInitFlagsMemorySize) + static_cast<__int64>(m_uiPageMemorySize) * static_cast<__int64>(m_uiNumPagesY * m_uiNumPagesX * pPage->m_uiPageZ + m_uiNumPagesX * pPage->m_uiPageY + pPage->m_uiPageX);
		BOOL bResult = SetFilePointerEx(m_hFile, liPos, &liNewPos, FILE_BEGIN);
		if (!bResult || liNewPos.QuadPart != liPos.QuadPart)
		{
			wprintf_s(L"ERROR: SetFilePointerEx failed " __LOCATION__ L"\n");
			exit(-1);
		}
		
		DWORD dwNumBytesRead;
		bResult = ReadFile(m_hFile, pPage->m_pFrame->m_pData, m_uiPageMemorySize, &dwNumBytesRead, NULL);
		if (!bResult || dwNumBytesRead != m_uiPageMemorySize)
		{
			wprintf_s(L"ERROR: ReadFile failed " __LOCATION__ L"\n");
			exit(-1);
		}

		//wprintf_s(L"Page (%u, %u, %u) read\n", pPage->m_uiPageX, pPage->m_uiPageY, pPage->m_uiPageZ);
	}
	else
	{
		BYTE* pData = pFrame->m_pData;
		for (unsigned int ui = 0; ui < m_uiPageSizeX * m_uiPageSizeY * m_uiPageSizeZ; ui++)
		{
			for (unsigned int uj = 0; uj < m_uiDim; uj++)
			{
				reinterpret_cast<T*>(pData)[m_uiDim * ui + uj] = m_defaultValue[uj];
			}
		}

		if (!m_bReadOnly)
		{
			pPage->m_bIsInitialized = true;
		}

		//wprintf_s(L"Page (%u, %u, %u) initialized\n", pPage->m_uiPageX, pPage->m_uiPageY, pPage->m_uiPageZ);
	}

}


template<class T>
void LargeArray3D<T>::PageOut(Page* pPage, bool bDiscard)
{
	assert(pPage->m_pFrame != NULL);

	if (!m_bReadOnly && !bDiscard)
	{
		LARGE_INTEGER liPos, liNewPos;
		liPos.QuadPart = static_cast<__int64>(m_dwSystemPageSize) + static_cast<__int64>(m_uiInitFlagsMemorySize) + static_cast<__int64>(m_uiPageMemorySize) * static_cast<__int64>(m_uiNumPagesY * m_uiNumPagesX * pPage->m_uiPageZ + m_uiNumPagesX * pPage->m_uiPageY + pPage->m_uiPageX);
		BOOL bResult = SetFilePointerEx(m_hFile, liPos, &liNewPos, FILE_BEGIN);
		if (!bResult || liNewPos.QuadPart != liPos.QuadPart)
		{
			wprintf_s(L"ERROR: SetFilePointerEx failed " __LOCATION__ L"\n");
			exit(-1);
		}
		
		DWORD dwNumBytesWritten;
		bResult = WriteFile(m_hFile, pPage->m_pFrame->m_pData, m_uiPageMemorySize, &dwNumBytesWritten, NULL);
		if (!bResult || dwNumBytesWritten != m_uiPageMemorySize)
		{
			wprintf_s(L"ERROR: WriteFile failed " __LOCATION__ L"\n");
			exit(-1);
		}

		//wprintf_s(L"Page (%u, %u, %u) written\n", pPage->m_uiPageX, pPage->m_uiPageY, pPage->m_uiPageZ);
	}

	Frame* pFrame = pPage->m_pFrame;
	pPage->m_pFrame = NULL;
	pFrame->m_pPage = NULL;

	m_pFreeFrames.push_back(pFrame);
}


template<class T>
inline T* LargeArray3D<T>::Get(unsigned int uiX, unsigned int uiY, unsigned int uiZ)
{
	assert(uiX < m_uiSizeX);
	assert(uiY < m_uiSizeY);
	assert(uiZ < m_uiSizeZ);

	unsigned int uiPageX = uiX / m_uiPageSizeX;
	unsigned int uiPageY = uiY / m_uiPageSizeY;
	unsigned int uiPageZ = uiZ / m_uiPageSizeZ;
	unsigned int uiRelX = uiX % m_uiPageSizeX;
	unsigned int uiRelY = uiY % m_uiPageSizeY;
	unsigned int uiRelZ = uiZ % m_uiPageSizeZ;

	if (uiPageX != m_uiLastPageX || uiPageY != m_uiLastPageY || uiPageZ != m_uiLastPageZ)
	{
		GetPage(uiPageX, uiPageY, uiPageZ);
	}

	return &reinterpret_cast<T*>(m_pLastPageData)[m_uiDim * (m_uiPageSizeY * m_uiPageSizeX * uiRelZ + m_uiPageSizeX * uiRelY + uiRelX)];
}


template<class T>
void LargeArray3D<T>::GetPage(unsigned int uiPageX, unsigned int uiPageY, unsigned int uiPageZ)
{
	Page* pPage = &m_pages[m_uiNumPagesY * m_uiNumPagesX * uiPageZ + m_uiNumPagesX * uiPageY + uiPageX];
	if (pPage->m_pFrame == NULL)
	{
		if (m_pFreeFrames.size() == 0)
		{
			Page* pPageToPageOut = m_pageCache.back();
			m_pageCache.pop_back();
			PageOut(pPageToPageOut);
		}
		PageIn(pPage);
		m_pageCache.push_front(pPage);
		pPage->m_pageCacheLocator = m_pageCache.begin();
	}

	m_uiLastPageX = uiPageX;
	m_uiLastPageY = uiPageY;
	m_uiLastPageZ = uiPageZ;
	m_pLastPageData = pPage->m_pFrame->m_pData;
}


template<class T>
void LargeArray3D<T>::Initialize(const T* pDefaultValue)
{
	if (!m_bReadOnly)
	{
		m_uiLastPageX = ~0u; // -1
		m_uiLastPageY = ~0u; // -1
		m_uiLastPageZ = ~0u; // -1
		m_pLastPageData = NULL;

		m_pFreeFrames.clear();
		for (unsigned int ui = 0; ui < m_pFrames.size(); ui++)
		{
			Frame* pFrame = m_pFrames[ui];
			if (pFrame->m_pPage != NULL)
			{
				pFrame->m_pPage->m_pFrame = NULL;
				pFrame->m_pPage = NULL;
			}
			m_pFreeFrames.push_back(pFrame);
		}

		if (pDefaultValue != NULL)
		{
			for (unsigned int ui = 0; ui < m_uiDim; ui++)
			{
				m_defaultValue[ui] = pDefaultValue[ui];
			}
		}
		else
		{
			for (unsigned int ui = 0; ui < m_uiDim; ui++)
			{
				m_defaultValue[ui] = 0;
			}
		}
		for (unsigned int uiPage = 0; uiPage < m_uiNumPagesX * m_uiNumPagesY * m_uiNumPagesZ; uiPage++)
		{
			m_pages[uiPage].m_bIsInitialized = false;
		}
	}
}

template<class T>
void LargeArray3D<T>::CopyTo(void* pDest, unsigned int uiX, unsigned int uiY, unsigned int uiZ, 
								  unsigned int uiWidth, unsigned int uiHeight, unsigned int uiDepth, 
								  unsigned int uiDestWidth, unsigned int uiDestHeight)
{
	/*assert(pDest != NULL && "LargeArray3D<T>::CopyTo()");
	assert(uiX < m_uiSizeX && "LargeArray3D<T>::CopyTo()");
	assert(uiY < m_uiSizeY && "LargeArray3D<T>::CopyTo()");
	assert(uiZ < m_uiSizeZ && "LargeArray3D<T>::CopyTo()");
	assert(uiDestWidth  == 0 || uiDestWidth  >= uiWidth  && "LargeArray3D<T>::CopyTo()");
	assert(uiDestHeight == 0 || uiDestHeight >= uiHeight && "LargeArray3D<T>::CopyTo()");*/

	assert(pDest != NULL);
	assert(uiX < m_uiSizeX);
	assert(uiY < m_uiSizeY);
	assert(uiZ < m_uiSizeZ);
	assert(uiDestWidth == 0 || uiDestWidth >= uiWidth);
	assert(uiDestHeight == 0 || uiDestHeight >= uiHeight);

	if (uiDestWidth  == 0)
		uiDestWidth  = uiWidth;
	if (uiDestHeight == 0)
		uiDestHeight = uiHeight;

	if (uiX + uiWidth  > m_uiSizeX)
		uiWidth  = m_uiSizeX - uiX;
	if (uiY + uiHeight > m_uiSizeY)
		uiHeight = m_uiSizeY - uiY;
	if (uiZ + uiDepth  > m_uiSizeZ)
		uiDepth  = m_uiSizeZ - uiZ;

	if (uiWidth == 0 || uiHeight == 0 || uiDepth == 0)
		return;


	unsigned int uiFirstColumn = uiX / m_uiPageSizeX;
	unsigned int uiLastColumn  = (uiX + uiWidth - 1) / m_uiPageSizeX;
	unsigned char* pucDestBase = reinterpret_cast<unsigned char*>(pDest);

	for (unsigned int z = uiZ; z < uiZ + uiDepth; z++) {
		for (unsigned int uiColumn = uiFirstColumn; uiColumn <= uiLastColumn; ++uiColumn) {
			unsigned int uiPageStart = uiColumn * m_uiPageSizeX;
			unsigned int uiStart = max(uiX, uiPageStart);
			unsigned int uiEnd   = min(uiX + uiWidth, uiPageStart + m_uiPageSizeX);

			for (unsigned int uiLine = 0; uiLine < uiHeight; ++uiLine) {
				unsigned char* pucDest   = &pucDestBase[((uiStart - uiX) + (uiLine + (z-uiZ) * uiDestHeight) * uiDestWidth) * sizeof(T)*m_uiDim];

				// Cast to unsigned char was missing, intention? 11/15/2011 Florian Reichl
				unsigned char* pucSource = reinterpret_cast<unsigned char*>(Get(uiStart, uiY + uiLine, z));
				memcpy(pucDest, pucSource, (uiEnd - uiStart) * sizeof(T)*m_uiDim);
			}
		}
	}
}

template<class T>
void LargeArray3D<T>::CopyFrom(void* pSource, unsigned int uiX, unsigned int uiY, unsigned int uiZ,
									unsigned int uiWidth, unsigned int uiHeight, unsigned int uiDepth,
									unsigned int uiSourceWidth, unsigned int uiSourceHeight)
{
	/*assert(pSource != NULL && "LargeArray3D<T>::CopyFrom()");
	assert(uiX < m_uiSizeX && "LargeArray3D<T>::CopyFrom()");
	assert(uiY < m_uiSizeY && "LargeArray3D<T>::CopyFrom()");
	assert(uiZ < m_uiSizeZ && "LargeArray3D<T>::CopyFrom()");
	assert(uiSourceWidth  == 0 || uiSourceWidth  >= uiWidth  && "LargeArray3D<T>::CopyFrom()");
	assert(uiSourceHeight == 0 || uiSourceHeight >= uiHeight && "LargeArray3D<T>::CopyFrom()");*/

	assert(pSource != NULL);
	assert(uiX < m_uiSizeX);
	assert(uiY < m_uiSizeY);
	assert(uiZ < m_uiSizeZ);
	assert(uiSourceWidth == 0 || uiSourceWidth >= uiWidth);
	assert(uiSourceHeight == 0 || uiSourceHeight >= uiHeight);
 
	if (uiSourceWidth  == 0)
		uiSourceWidth  = uiWidth;
	if (uiSourceHeight == 0)
		uiSourceHeight = uiHeight;

	if (uiX + uiWidth  > m_uiSizeX)
		uiWidth  = m_uiSizeX - uiX;
	if (uiY + uiHeight > m_uiSizeY)
		uiHeight = m_uiSizeY - uiY;
	if (uiZ + uiDepth  > m_uiSizeZ)
		uiDepth  = m_uiSizeZ - uiZ;


	if (uiWidth == 0 || uiHeight == 0 || uiDepth == 0)
		return;


	unsigned char* pucSource = reinterpret_cast<unsigned char*>(pSource);
	for (unsigned int z = uiZ; z < uiZ + uiDepth; z++) {
		for (unsigned int y = uiY; y < uiY + uiHeight; y++) {
			for (unsigned int x = uiX; x < uiX + uiWidth; x++) {
				memcpy(Get(x,y,z), &pucSource[m_uiDim * sizeof(T) * (x-uiX + (y-uiY + (size_t)(z-uiZ)*uiSourceHeight)*uiSourceWidth)], sizeof(T)*m_uiDim);
			}
		}
	}
}


template<class T>
bool LargeArray3D<T>::ConvertToRAW(const wchar_t* pwcFileName)
{
	FILE* hFile;
	errno_t err = _wfopen_s(&hFile, pwcFileName, L"wb");
	if (err != 0)
	{
		//wprintf_s(L"ERROR: Cannot open file %s " __LOCATION__ L"\n", pwcFileName);
		return false;
	}
	for (unsigned int uiZ = 0; uiZ < m_uiSizeZ; uiZ++)
	{
		for (unsigned int uiY = 0; uiY < m_uiSizeY; uiY++)
		{
			for (unsigned int uiX = 0; uiX < m_uiSizeX; uiX++)
			{
				fwrite(Get(uiX, uiY, uiZ), sizeof(T), m_uiDim, hFile);
			}
		}
	}
	fclose(hFile);

	return true;
}


template<class T>
void LargeArray3D<T>::Close(bool bDiscard)
{
	if (m_bIsOpen)
	{
		if (!m_bReadOnly && !bDiscard)
		{
			WriteInitFlags();
		}

		for (unsigned int ui = 0; ui < m_pFrames.size(); ui++)
		{
			Frame* pFrame = m_pFrames[ui];
			if (pFrame->m_pPage != NULL)
			{
				PageOut(pFrame->m_pPage, bDiscard);
			}
			DisposeFrame(pFrame);
		}
		
		m_pFreeFrames.clear();
		m_pFrames.clear();
		m_pageCache.clear();

		delete[] m_pages;

		BOOL bResult = CloseHandle(m_hFile);
		if (!bResult)
		{
			wprintf_s(L"ERROR: CloseHandle failed " __LOCATION__ L"\n");
			exit(-1);
		}

		m_bIsOpen = false;
	}
}


template<class T>
LargeArray3D<T>::~LargeArray3D()
{
	if (m_bIsOpen)
	{
		Close();
	}

	delete[] m_defaultValue;
}


template<class T>
bool LargeArray3D<T>::OpenHeader(const wchar_t* pwcFileName)
{
	if (m_bIsOpen)
	{
		Close();
	}


	m_bReadOnly = true;

	SYSTEM_INFO systemInfo;
	GetSystemInfo(&systemInfo);
	m_dwSystemPageSize = systemInfo.dwPageSize;
	
	// Open file
	DWORD dwDesiredAccess = GENERIC_READ;
	if (!m_bReadOnly) { dwDesiredAccess |= GENERIC_WRITE; }
	m_hFile = CreateFile(pwcFileName, dwDesiredAccess, FILE_SHARE_READ, NULL, OPEN_EXISTING, FILE_FLAG_NO_BUFFERING, NULL);
	if (m_hFile == INVALID_HANDLE_VALUE)
	{
		//wprintf_s(L"ERROR: Cannot open file %s " __LOCATION__ L"\n", pwcFileName);
		return false;
	}

	if (!ReadHeader())
	{
		return false;
	}

	m_uiNumPagesX = (m_uiSizeX + m_uiPageSizeX - 1) / m_uiPageSizeX;
	m_uiNumPagesY = (m_uiSizeY + m_uiPageSizeY - 1) / m_uiPageSizeY;
	m_uiNumPagesZ = (m_uiSizeZ + m_uiPageSizeZ - 1) / m_uiPageSizeZ;

	CloseHandle(m_hFile);
	return true;
}

}; // end namespace LA3D


#endif






/*

#include "largearray2d.h"

#include <stdlib.h>
#include <time.h>
#include <windows.h>


struct Point
{
	unsigned int	m_uiX;
	unsigned int	m_uiY;
	float			m_fValue1;
	float			m_fValue2;
};

int wmain(int argc, wchar_t* argv[])
{
	LargeArray3D<float,2> la;
	la.Create(20000, 20000, 1024, 1024, L"C:\\Temp\\data.la2d", 3 * 8 * 1024 * 1024);
	wprintf_s(L"Size X: %u, Size Y: %u\n", la.GetSizeX(), la.GetSizeY());

	float init[2] = {123.0f, 456.0f};
	la.Initialize(init);

	srand( (unsigned)time( NULL ) );

	const unsigned int uiNUM_POINTS = 10;
	
	Point* points = new Point[uiNUM_POINTS];
	ZeroMemory(points, uiNUM_POINTS * sizeof(Point));

	for (unsigned int ui = 0; ui < uiNUM_POINTS; ui++)
	{
		points[ui].m_uiX = (unsigned int)((double)rand() / (double)RAND_MAX * 19999.0);
		points[ui].m_uiY = (unsigned int)((double)rand() / (double)RAND_MAX * 19999.0);
		points[ui].m_fValue1 = (float)rand();
		points[ui].m_fValue2 = (float)rand();

		float* pf = la.Get(points[ui].m_uiX, points[ui].m_uiY);
		pf[0] = points[ui].m_fValue1;
		pf[1] = points[ui].m_fValue2;
	}

	la.Close();

	la.Open(L"C:\\Temp\\data.la2d", true, 3 * 8 * 1024 * 1024);
	wprintf_s(L"Size X: %u, Size Y: %u\n", la.GetSizeX(), la.GetSizeY());

	for (unsigned int ui = 0; ui < uiNUM_POINTS; ui++)
	{
		float* pf = la.Get(points[ui].m_uiX, points[ui].m_uiY);
		wprintf_s(L"Value1: %f Expected: %f Value2: %f Exptected: %f\n", pf[0], points[ui].m_fValue1, pf[1], points[ui].m_fValue2);
		if (pf[0] != points[ui].m_fValue1 || pf[1] != points[ui].m_fValue2)
		{
			wprintf_s(L"ERROR!\n");
			exit(-1);
		}
	}

	for (unsigned int ui = 0; ui < uiNUM_POINTS; ui++)
	{
		unsigned int uiX = (unsigned int)((double)rand() / (double)RAND_MAX * 19999.0);
		unsigned int uiY = (unsigned int)((double)rand() / (double)RAND_MAX * 19999.0);

		float* pf = la.Get(uiX, uiY);

		wprintf_s(L"Random Read: %f %f\n", pf[0], pf[1]);
	}

	la.Close();

}

*/