#pragma once

#ifndef SYSTOOLS_H
#define SYSTOOLS_H

/**************************************************************
 *                                                            *
 * description: simple routines for filename handling         *
 * version    : 1.4                                          *
 * date       : 10.Jan.2007                                   *
 * modified   : 31.Jul.2008                                   *
 * author     : Jens Krueger                                  *
 * e-mail     : mail@jens-krueger.com                         *
 *                                                            *
 **************************************************************/


#include <string>
#include <vector>

#ifdef _WIN32
	#include <Windows.h>
	#include <WinBase.h>

	#ifdef max
		#undef max
	#endif

	#ifdef min
		#undef min
	#endif

#else
	#include <wchar.h>
	typedef wchar_t WCHAR;
	typedef unsigned char CHAR;
#endif

namespace tum3d 
{
		std::wstring ToLowerCase(const std::wstring& str);
		std::string ToLowerCase(const std::string& str);
		std::wstring ToUpperCase(const std::wstring& str);
		std::string ToUpperCase(const std::string& str);

		std::string GetFromResourceOnMac(const std::string& fileName);
		std::wstring GetFromResourceOnMac(const std::wstring& fileName);

		bool FileExists(const std::string& fileName);
		bool FileExists(const std::wstring& fileName);

		std::string GetExt(const std::string& fileName);
		std::wstring GetExt(const std::wstring& fileName);

		std::string GetPath(const std::string& fileName);
		std::wstring GetPath(const std::wstring& fileName);

		std::string GetFilename(const std::string& fileName);
		std::wstring GetFilename(const std::wstring& fileName);

		std::string FindPath(const std::string& fileName, const std::string& path);
		std::wstring FindPath(const std::wstring& fileName, const std::wstring& path);

		std::string  RemoveExt(const std::string& fileName);
		std::wstring RemoveExt(const std::wstring& fileName);

		std::string  ChangeExt(const std::string& fileName, const std::string& newext);
		std::wstring ChangeExt(const std::wstring& fileName, const std::wstring& newext);

		std::string  AppendFilename(const std::string& fileName, const std::string& tag);
		std::wstring AppendFilename(const std::wstring& fileName, const std::wstring& tag);

		std::string  FindNextSequenceName(const std::string& fileName, const std::string& ext, const std::string& dir="");
		std::wstring FindNextSequenceName(const std::wstring& fileName, const std::wstring& ext, const std::wstring& dir=L"");
		std::wstring FindNextSequenceNameEX(const std::wstring& fileName, const std::wstring& ext, const std::wstring& dir=L"");

		unsigned int FindNextSequenceIndex(const std::string& fileName, const std::string& ext, const std::string& dir="");
		unsigned int FindNextSequenceIndex(const std::wstring& fileName, const std::wstring& ext, const std::wstring& dir=L"");

	#ifdef _WIN32
		std::vector<std::wstring> GetDirContents(const std::wstring& dir, const std::wstring& fileName=L"*", const std::wstring& ext=L"*");
		std::vector<std::string> GetDirContents(const std::string& dir, const std::string& fileName="*", const std::string& ext="*");
	#else
		std::vector<std::wstring> GetDirContents(const std::wstring& dir, const std::wstring& fileName=L"*", const std::wstring& ext=L"");
		std::vector<std::string> GetDirContents(const std::string& dir, const std::string& fileName="*", const std::string& ext="");
	#endif

		bool GetFileStats(const std::string& strFileName, struct stat& stat_buf);
		bool GetFileStats(const std::wstring& wstrFileName, struct stat& stat_buf);

		void RemoveLeadingWhitespace(std::wstring &str);
		void RemoveLeadingWhitespace(std::string &str);
		void RemoveTailingWhitespace(std::wstring &str);
		void RemoveTailingWhitespace(std::string &str);

	#ifdef _WIN32
		bool GetFilenameDialog(const std::string& title, const /*std::string&*/char* filter, std::string &filename, const bool save, HWND owner=NULL, DWORD* nFilterIndex=NULL);
		bool GetFilenameDialog(const std::wstring& title, const /*std::string&*/wchar_t* filter, std::wstring &filename, const bool save, HWND owner=NULL, DWORD* nFilterIndex=NULL);
		bool GetMultiFilenameDialog(const std::string& title, const /*std::string&*/char* filter, std::vector<std::string>& filenames, HWND owner=NULL);
		bool GetPathDialog(const std::string& title, std::string& path, HWND owner=NULL);
	#endif

		class CmdLineParams {
			public:
				#ifdef _WIN32
					CmdLineParams();
				#endif
				CmdLineParams(int argc, char** argv);

				bool SwitchSet(const std::string& parameter);
				bool SwitchSet(const std::wstring& parameter);

				bool GetValue(const std::string& parameter, double& value);
				bool GetValue(const std::wstring& parameter, double& value);
				bool GetValue(const std::string& parameter, float& value);
				bool GetValue(const std::wstring& parameter, float& value);
				bool GetValue(const std::string& parameter, int& value);
				bool GetValue(const std::wstring& parameter, int& value);
				bool GetValue(const std::string& parameter, std::string& value);
				bool GetValue(const std::wstring& parameter, std::wstring& value);
		
			protected:
				std::vector<std::string> m_strArrayParameters;
				std::vector<std::string> m_strArrayValues;

				std::string m_strFilename;
		};
}

#endif // SYSTOOLS_H
