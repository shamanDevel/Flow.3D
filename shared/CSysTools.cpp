/*************************************************************************************

Terrain3D

Author: Christian Dick

Copyright (c) Christian Dick

mailto:dick@in.tum.de

This software is property of Christian Dick. All rights reserved.
Unauthorized use prohibited.

*************************************************************************************/

#include "CSysTools.h"

#include <CommDlg.h>


namespace CSysTools
{

bool FileNameDialog(const wchar_t* pwcTitle, const wchar_t* pwcFilter, DWORD dwFilterIndex, const wchar_t* pwcInitialDir, const wchar_t* pwcDefaultExt, std::wstring& strFileName, bool bSave, HWND hwndOwner)
{
	// Filter: (<Name>\0<Mask>(;<Mask>)*\0)+
	wchar_t wcFileName[MAX_PATH];
	OPENFILENAME openFileName;
	ZeroMemory(&openFileName, sizeof(OPENFILENAME));
	openFileName.lStructSize = sizeof(OPENFILENAME);
	openFileName.hwndOwner = hwndOwner;
	openFileName.lpstrFilter = pwcFilter;
	openFileName.nFilterIndex = dwFilterIndex;
	wcFileName[0] = 0;
	openFileName.lpstrFile = wcFileName;
	openFileName.nMaxFile = MAX_PATH;
	openFileName.lpstrInitialDir = pwcInitialDir;
	openFileName.lpstrTitle = pwcTitle;
	openFileName.lpstrDefExt = pwcDefaultExt;
	BOOL bResult;
	if (bSave)
	{
		openFileName.Flags = OFN_OVERWRITEPROMPT;
		bResult = GetSaveFileName(&openFileName);
	}
	else
	{
		openFileName.Flags = OFN_FILEMUSTEXIST;
		bResult = GetOpenFileName(&openFileName);
	}
	if (bResult != 0)
	{
		strFileName = wcFileName;
		return true;
	}
	else
	{
		return false;
	}
}


bool PathDialog()
{
	return false;
}


void FindFiles(const std::wstring& strPath, const std::wstring& strMask, std::vector<std::wstring>& strFileNames, bool bRecursive, const std::wstring& strRelPath)
{
	WIN32_FIND_DATA findData;
	HANDLE hFind;
	bool bFirst = true;
	while (true)
	{
		if (bFirst)
		{
			std::wstring strFullName = strPath + L"\\" + strRelPath + strMask;
			hFind = FindFirstFile(strFullName.c_str(), &findData);
			if (hFind == INVALID_HANDLE_VALUE)
			{
				break;
			}
			bFirst = false;
		}
		else
		{
			if (!FindNextFile(hFind, &findData))
			{
				FindClose(hFind);
				break;
			}
		}
		if ((findData.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) == 0)
		{
			strFileNames.push_back(strRelPath + findData.cFileName);
		}
	}

	if (bRecursive)
	{
		std::vector<std::wstring> strDirectories;
		bFirst = true;
		while (true)
		{
			if (bFirst)
			{
				std::wstring strFullName = strPath + L"\\" + strRelPath + L"*";
				hFind = FindFirstFile(strFullName.c_str(), &findData);
				if (hFind == INVALID_HANDLE_VALUE)
				{
					break;
				}
				bFirst = false;
			}
			else
			{
				if (!FindNextFile(hFind, &findData))
				{
					FindClose(hFind);
					break;
				}
			}
			if ((findData.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) != 0)
			{
				std::wstring strDirectory = findData.cFileName;
				if (strDirectory != L"." && strDirectory != L"..")
				{
					strDirectories.push_back(strDirectory);
				}
			}
		}

		for (std::vector<std::wstring>::iterator i = strDirectories.begin(); i != strDirectories.end(); i++)
		{
			FindFiles(strPath, strMask, strFileNames, true, strRelPath + *i + L"\\");
		}
	}

}

};
