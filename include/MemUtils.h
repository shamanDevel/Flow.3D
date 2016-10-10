#pragma once

#include "global.h"
#include <cassert>
#include <vector>
#include "Utils.h"

class SystemMemoryUsage
{
public:
	inline SystemMemoryUsage(float fRelativeUsageLimit);
	inline ~SystemMemoryUsage();

	inline void SetSystemMemoryLimitPercent(float value) { m_i64LimitBytes = int64(ceil(value * GetAvailableMemory())); }
	inline void SetSystemMemoryLimitBytes(int64 value) { m_i64LimitBytes = value; }
	inline void SetSystemMemoryLimitMBytes(float value) { m_i64LimitBytes = int64(ceil(value * 1024.0 * 1024.0)); }
	inline int64 GetSystemMemoryUsageBytes() const { return m_i64UsageBytes; }
	inline int64 GetSystemMemoryLimitBytes() const { return m_i64LimitBytes; }
	inline float GetSystemMemoryLimitMBytes() const { return float(m_i64LimitBytes / (1024.0 * 1024.0)); }

	inline bool IsMemoryAvailable(int64 bytes = 0) const { return (m_i64UsageBytes + bytes <= m_i64LimitBytes); }

	inline void MemoryAllocated(int64 bytes) { m_i64UsageBytes += bytes; }
	inline void MemoryDeallocated(int64 bytes) { m_i64UsageBytes -= bytes; }

	inline static int64 GetAvailableMemory();

private:
	int64 m_i64UsageBytes;
	int64 m_i64LimitBytes;
};




inline SystemMemoryUsage::SystemMemoryUsage(float fRelativeUsageLimit)
	: m_i64UsageBytes(0)
{
	SetSystemMemoryLimitPercent(fRelativeUsageLimit);
}

inline SystemMemoryUsage::~SystemMemoryUsage()
{
	assert(m_i64UsageBytes == 0);
}


inline int64 SystemMemoryUsage::GetAvailableMemory()
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