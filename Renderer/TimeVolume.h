#pragma once


#include <global.h>

#include <vector>

#include <TimeVolumeIO.h>


class TimeVolume : public TimeVolumeIO
{
public:
	TimeVolume(float memoryUsageLimit = 0.7f);
	~TimeVolume();

	// current state
	float GetCurTime() const { return m_currentTime; }
	void SetCurTime(float time) { m_currentTime = time; }

	int GetCurFloorTimestepIndex()   const { return GetFloorTimestepIndex(m_currentTime); }
	int GetCurCeilTimestepIndex()    const { return GetCeilTimestepIndex(m_currentTime); }
	int GetCurNearestTimestepIndex() const { return GetNearestTimestepIndex(m_currentTime); }

	Timestep& GetNearestTimestep() { return GetTimestep(GetCurNearestTimestepIndex()); }
	const Timestep& GetNearestTimestep() const { return GetTimestep(GetCurNearestTimestepIndex()); }

	float GetLoadingProgress() const { return m_loadingProgress; }

	// Update loading queue according to provided list of wanted bricks.
	// Unless enqueueAll is set, this should be called regularly even when the list doesn't change.
	// Returns current loading progress.
	float UpdateLoadingQueue(const std::vector<const Brick*>& bricksToLoad, bool enqueueAll = false);

	void LoadNearestTimestep(bool blocking = true);

private:
	float m_currentTime;

	float m_loadingProgress;
};
