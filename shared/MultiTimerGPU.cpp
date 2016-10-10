#include "MultiTimerGPU.h"

#include <algorithm>
#include <numeric>


MultiTimerGPU::MultiTimerGPU()
	: m_nextTimerToStart(0)
{
}

MultiTimerGPU::~MultiTimerGPU()
{
	ReleaseTimers();
}


uint MultiTimerGPU::StartNextTimer()
{
	// implicit stop of previous timer
	StopCurrentTimer();

	// add new timer if necessary
	if(m_nextTimerToStart >= m_timers.size())
	{
		m_timers.push_back(new TimerGPU());
	}

	m_timers[m_nextTimerToStart]->Start();

	return (uint)m_nextTimerToStart++;
}

void MultiTimerGPU::StopCurrentTimer()
{
	if(m_nextTimerToStart > 0 && m_timers[m_nextTimerToStart - 1]->IsRunning())
	{
		m_timers[m_nextTimerToStart - 1]->Stop();
	}
}


std::vector<float> MultiTimerGPU::GetElapsedTimesMS()
{
	std::vector<float> times(m_nextTimerToStart);

	for(size_t i = 0; i < m_nextTimerToStart; i++)
	{
		times[i] = m_timers[i]->GetElapsedTimeMS();
	}

	return times;
}

float MultiTimerGPU::GetCumulativeElapsedTimeMS()
{
	float time = 0.0f;

	for(size_t i = 0; i < m_nextTimerToStart; i++)
	{
		time += m_timers[i]->GetElapsedTimeMS();
	}

	return time;
}

MultiTimerGPU::Stats MultiTimerGPU::GetStats()
{
	std::vector<float> times = GetElapsedTimesMS();

	Stats result;
	result.Total = times.empty() ? 0.0f : std::accumulate(times.begin(), times.end(), 0.0f);
	result.Min   = times.empty() ? 0.0f : *std::min_element(times.begin(), times.end());
	result.Max   = times.empty() ? 0.0f : *std::max_element(times.begin(), times.end());
	result.Count = uint(times.size());
	result.Avg   = result.Total / float(result.Count);

	return result;
}


void MultiTimerGPU::Sync()
{
	if(m_nextTimerToStart > 0)
	{
		m_timers[m_nextTimerToStart - 1]->Sync();
	}
}


void MultiTimerGPU::ResetTimers()
{
	StopCurrentTimer();

	m_nextTimerToStart = 0;
}


void MultiTimerGPU::ReleaseTimers()
{
	ResetTimers();

	for(size_t i = 0; i < m_timers.size(); i++)
	{
		delete m_timers[i];
	}
	m_timers.clear();
}
