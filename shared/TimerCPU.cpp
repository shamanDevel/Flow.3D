#include "TimerCPU.h"

#include <Windows.h>


TimerCPU::TimerCPU()
	: m_timestampStart(0), m_timestampStop(0), m_isRunning(false)
{
}

TimerCPU::~TimerCPU()
{
}

void TimerCPU::Start()
{
	m_isRunning = true;
	LARGE_INTEGER time;
	QueryPerformanceCounter(&time);
	m_timestampStart = time.QuadPart;
}

void TimerCPU::Stop()
{
	LARGE_INTEGER time;
	QueryPerformanceCounter(&time);
	m_timestampStop = time.QuadPart;
	m_isRunning = false;
}

bool TimerCPU::IsRunning()
{
	return m_isRunning;
}

float TimerCPU::GetElapsedTimeMS()
{
	uint64 timestampStop;
	if(m_isRunning)
	{
		LARGE_INTEGER time;
		QueryPerformanceCounter(&time);
		timestampStop = time.QuadPart;
	}
	else
	{
		timestampStop = m_timestampStop;
	}

	LARGE_INTEGER freq;
	QueryPerformanceFrequency(&freq);
	return 1000.0f * float(timestampStop - m_timestampStart) / float(freq.QuadPart);
}
