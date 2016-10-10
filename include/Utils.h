#pragma once

#include "global.h"
#include <Unknwn.h>
#include <string>
#include <sstream>
#include <stdio.h>
#include <iostream>


using std::string;
using std::cout;
using std::stringstream;

// **********************************
// Exceptions
// **********************************

#define THROW(stream) { stringstream s; s << stream; throw TimeVolumeException(s.str(), __FILE__, __LINE__); }
#define E_ASSERT(stream, condition) if (!(condition)) { THROW(stream); }

class TimeVolumeException
{
public:
	TimeVolumeException(const string& msg, const char* file, int32 line)
	{
		m_msg = msg;
		cout << "*** ERROR *** \n" <<
			file << ": " << line << "\n" <<
			msg << "\n";
	}

	const string& GetMsg() { return m_msg; }

private:
	string		m_msg;
};



// **********************************
// Helper macros
// **********************************


#ifndef SAFE_RELEASE
#define SAFE_RELEASE(p)      { if (p) { (p)->Release(); (p)=NULL; } }
#endif


// **********************************
// Mathematical stuff
// **********************************

template<typename T>
T RoundToNextMultiple(const T value, T multiple)
{
	return ((value - 1) / multiple + 1) * multiple;
}

template<typename T>
T RoundToPrevMultiple(const T value, T multiple)
{
	return (value / multiple) * multiple;
}

template<typename T>
T Log2(T value)
{
	T r = 0;
	while (value >>= 1) // unroll for more speed...
	{
		r++;
	}

	return r;
}


template<typename T>
T Square(T value)
{
	return value * value;
}


template<typename T>
T Cube(T value)
{
	return value * value * value;
}


inline bool IsPow2(uint32 x)
{
	return (x & (x - 1)) == 0;
}





// **********************************
// Small helper classes
// **********************************


// Helper class: Templated ringbuffer
template<typename Type, int size>
class RingBuffer
{
public:
	RingBuffer(): m_iCurrent(0) { assert(size > 0); }

	static int Size() { return size; }
	Type& Current() { return m_array[m_iCurrent]; }

	void Reset() { m_iCurrent = 0; }
	void Next() { m_iCurrent = (m_iCurrent + 1) % size; }

	Type& operator[](int index)
	{
		assert((index >= 0) && (index < size));
		return m_array[index];
	}

	template<class Functor>
	void Iterate(Functor func)
	{
		for (int i = 0; i < size; ++i)
		{
			func(m_array[i]);
		}
	}

private:
	Type		m_array[size];
	int			m_iCurrent;
};


template<class T>
class InterfaceScope
{
public:
	InterfaceScope(): m_pInterface(nullptr) {}
	~InterfaceScope() { SAFE_RELEASE(m_pInterface); }

	T*& operator*() { return m_pInterface; }
	T* operator->() { return m_pInterface; }

private:
	T*			m_pInterface;
};





// **********************************
// Memory
// **********************************

inline int64 GetAvailableMemory()
{
	// compute absolute usage limit
	MEMORYSTATUSEX memoryStatusEx;
	ZeroMemory(&memoryStatusEx, sizeof(MEMORYSTATUSEX));
	memoryStatusEx.dwLength = sizeof(MEMORYSTATUSEX);
	if (!GlobalMemoryStatusEx(&memoryStatusEx))
	{
		THROW("GlobalMemoryStatusEx failed");
	}

	return min(memoryStatusEx.ullAvailPhys,	memoryStatusEx.ullAvailVirtual);
}





// **********************************
// I/O
// **********************************

inline void SimpleProgress(int curVal, int maxVal)
{
	static const char bar[] =      "                    ";
	static const char progress[] = "********************";
	static const unsigned char barEnd[] = {219, 0};

	float fPercent = static_cast<float>(curVal) / maxVal;
	int iPercent = static_cast<int>(floor(fPercent * 100));
	int m = min(20, max(1, static_cast<int>(ceil(fPercent * 21))));

	if ((iPercent < 100) && (maxVal > 0))
	{
		printf("\r[%s%s%s] %d%%     ", progress + 21 - m, barEnd, bar + m, iPercent);
	} else {
		printf("\r[%s] 100%%   ", progress);
	}
}