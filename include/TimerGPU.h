#ifndef __TUM3D__TIMERGPU_H__
#define __TUM3D__TIMERGPU_H__


#include <global.h>

#include <cuda_runtime.h>


// Simple wrapper around cudaEvent* for GPU timing.
// Basic usage: Call Start(), launch CUDA kernels, call Stop(). Then later call GetElapsedTimeMS() to get the result.
// Calling Start()/Stop() again overwrites the start/stop timestamp.
// Calling GetElapsedTimeMS() between Start() and Stop() (i.e. while IsRunning()) gives undefined results.
class TimerGPU
{
public:
	TimerGPU();
	~TimerGPU();

	void Start();
	void Stop();
	bool IsRunning();

	// wait for the stop event
	void Sync();
	// this implies a sync on the stop event
	float GetElapsedTimeMS();
	// returns false if not ready
	bool TryGetElapsedTimeMS(float& time);

private:
	cudaEvent_t m_eventStart;
	cudaEvent_t m_eventStop;
	bool m_isRunning;
};


#endif
