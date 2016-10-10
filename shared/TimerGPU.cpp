#include "TimerGPU.h"

#include <cassert>

#include <cudaUtil.h>


TimerGPU::TimerGPU()
	: m_isRunning(false)
{
	cudaSafeCall(cudaEventCreate(&m_eventStart));
	cudaSafeCall(cudaEventCreate(&m_eventStop));

	// record start/stop immediately, so calling GetElapsedTime is always safe
	Start();
	Stop();
}

TimerGPU::~TimerGPU()
{
	cudaSafeCall(cudaEventDestroy(m_eventStop));
	cudaSafeCall(cudaEventDestroy(m_eventStart));
}

void TimerGPU::Start()
{
	m_isRunning = true;
	cudaSafeCall(cudaEventRecord(m_eventStart));
}

void TimerGPU::Stop()
{
	cudaSafeCall(cudaEventRecord(m_eventStop));
	m_isRunning = false;
}

bool TimerGPU::IsRunning()
{
	return m_isRunning;
}

void TimerGPU::Sync()
{
	cudaSafeCall(cudaEventSynchronize(m_eventStop));
}

float TimerGPU::GetElapsedTimeMS()
{
	Sync();
	float t = 0.0f;
	cudaSafeCall(cudaEventElapsedTime(&t, m_eventStart, m_eventStop));
	return t;
}

bool TimerGPU::TryGetElapsedTimeMS(float& time)
{
	cudaError_t result = cudaEventElapsedTime(&time, m_eventStart, m_eventStop);
	if(result == cudaSuccess)
	{
		return true;
	}
	else 
	{
		if(result != cudaErrorNotReady)
		{
			cudaSafeCall(result);
		}
		return false;
	}
}
