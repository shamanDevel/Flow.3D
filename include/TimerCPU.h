#ifndef __TUM3D__TIMERCPU_H__
#define __TUM3D__TIMERCPU_H__


#include <global.h>


// Simple timer.
// Basic usage: Call Start(), do stuff, call Stop(). Then call GetElapsedTimeMS() to get the result.
// Calling Start()/Stop() again overwrites the start/stop timestamp.
// Calling GetElapsedTimeMS() between Start() and Stop() (i.e. while IsRunning()) returns the time since Start().
class TimerCPU
{
public:
	TimerCPU();
	~TimerCPU();

	void Start();
	void Stop();
	bool IsRunning();

	float GetElapsedTimeMS();

private:
	uint64 m_timestampStart;
	uint64 m_timestampStop;
	bool m_isRunning;
};


#endif
