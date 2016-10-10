/*************************************************************************************

Terrain3D

Author: Christian Dick

Copyright (c) Christian Dick

mailto:dick@in.tum.de

This software is property of Christian Dick. All rights reserved.
Unauthorized use prohibited.

*************************************************************************************/

#ifndef __TUM3D__CSYSTOOLS_H__
#define __TUM3D__CSYSTOOLS_H__

#include "global.h"

#include <string>
#include <vector>

#include <windows.h>


#ifndef __LOCATION__
#define ___STR(x) #x
#define __STR(x) ___STR(x)
#define ___L(x) L##x
#define __L(x) ___L(x)
#define __LOCATION__ __L(__FILE__) L":" __L(__STR(__LINE__))
#endif


namespace CSysTools
{

inline std::wstring RemoveFileExtension(const std::wstring& strFileName)
{
	size_t index = strFileName.find_last_of(L'.');
	size_t indexPathSep = strFileName.find_last_of(L'\\');
	if (indexPathSep != std::wstring::npos && index < indexPathSep) { index = std::wstring::npos; } // Ignore . in path
	return strFileName.substr(0, index); // . is not included
}


inline std::wstring ChangeFileExtension(const std::wstring& strFileName, const std::wstring& strNewExtension)
{
	return RemoveFileExtension(strFileName) + L"." + strNewExtension;
}


inline std::wstring GetPath(const std::wstring& strFileName)
{
	size_t index = strFileName.find_last_of(L'\\');
	return strFileName.substr(0, index == std::wstring::npos ? 0 : index); // Last \ is not included
}


inline std::wstring GetFileName(const std::wstring& strFileName)
{
	size_t index = strFileName.find_last_of(L'\\');
	return strFileName.substr(index == std::wstring::npos ? 0 : index + 1, std::wstring::npos);
}


inline std::wstring GetFileExtension(const std::wstring& strFileName)
{
	size_t index = strFileName.find_last_of(L'.');
	size_t indexPathSep = strFileName.find_last_of(L'\\');
	if (indexPathSep != std::wstring::npos && index < indexPathSep) { index = std::wstring::npos; } // Ignore . in path
	return strFileName.substr(index == std::wstring::npos ? strFileName.length() : index + 1, std::wstring::npos);
}


inline std::wstring GetExePath()
{
	wchar_t buf[MAX_PATH];
	GetModuleFileName(NULL, buf, MAX_PATH);
	return GetPath(buf);
}


bool FileNameDialog(const wchar_t* pwcTitle, const wchar_t* pwcFilter, DWORD dwFilterIndex, const wchar_t* pwcInitialDir, const wchar_t* pwcDefaultExt, std::wstring& strFileName, bool bSave, HWND hwndOwner = NULL);


bool PathDialog();


void FindFiles(const std::wstring& strPath, const std::wstring& strMask, std::vector<std::wstring>& strFileNames, bool bRecursive = false, const std::wstring& strRelPath = L"");


inline void OutDbg(const std::wstring& str)
{
	OutputDebugString(str.c_str());
}


inline void ShowErrorMessageBox(const std::wstring& strTitle, const std::wstring& strMessage)
{
	MessageBox(NULL, strMessage.c_str(), strTitle.c_str(), 1);
}


inline std::wstring Wide(const std::string& str)
{
	return std::wstring(str.begin(), str.end());
}


};

#endif
