#ifndef __TUM3D__MULTITIMERGPU_H__
#define __TUM3D__MULTITIMERGPU_H__


#include <global.h>

#include <vector>

#include "TimerGPU.h"


class MultiTimerGPU
{
public:
	struct Stats
	{
		Stats() { memset(this, 0, sizeof(*this)); }
		bool operator==(const Stats& other) { return memcmp(this, &other, sizeof(*this)) == 0; }
		bool operator!=(const Stats& other) { return !(*this == other); }

		float Min;
		float Avg;
		float Max;
		uint Count;
		float Total;
	};

	MultiTimerGPU();
	~MultiTimerGPU();

	// StartNextTimer() returns the index of the timer that was started.
	// If the previous timer is still running, it is stopped implicitly.
	uint StartNextTimer();
	void StopCurrentTimer();

	std::vector<float> GetElapsedTimesMS();
	float GetCumulativeElapsedTimeMS();
	Stats GetStats();

	void Sync();

	void ResetTimers();

	void ReleaseTimers();

private:
	std::vector<TimerGPU*> m_timers;
	size_t m_nextTimerToStart;
};


#endif
