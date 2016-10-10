#include "TimeVolume.h"

using namespace tum3D;


TimeVolume::TimeVolume(float memoryUsageLimit)
	: TimeVolumeIO(memoryUsageLimit)
	, m_currentTime(0.0f)
	, m_loadingProgress(0.0f)
{
}

TimeVolume::~TimeVolume()
{
}


float TimeVolume::UpdateLoadingQueue(const std::vector<const TimeVolumeIO::Brick*>& bricksToLoad, bool enqueueAll)
{
	// first clear current loading queue - bricks will be re-queued in correct order
	DiscardQueuedIO();

	int bricksAlreadyLoaded = 0;
	for(auto it = bricksToLoad.cbegin(); it != bricksToLoad.cend(); ++it)
	{
		Brick* pBrick = const_cast<Brick*>(*it); // hrm...

		if(pBrick->IsLoaded())
		{
			++bricksAlreadyLoaded;
		}
		else
		{
			// by default, don't bother queueing many more bricks than we have loading slots
			if(enqueueAll || (GetLoadingQueueSize() < max(10, 3 * ms_loadingSlotCount)))
			{
				EnqueueLoadBrickData(*pBrick);
			}
		}
	}

	m_loadingProgress = bricksToLoad.empty() ? 1.0f : float(bricksAlreadyLoaded) / float(bricksToLoad.size());

	return m_loadingProgress;
}

void TimeVolume::LoadNearestTimestep(bool blocking)
{
	// collect all bricks from current timestep
	const Timestep& timestep = GetNearestTimestep();
	std::vector<const Brick*> bricks(timestep.bricks.size());
	for(size_t i = 0; i < bricks.size(); i++)
	{
		bricks[i] = &timestep.bricks[i];
	}

	// enqueue
	UpdateLoadingQueue(bricks, true);

	// block if specified
	if(blocking)
	{
		WaitForAllIO();
	}
}


//float TimeVolume::GetPrefetchingProgress() const
//{
//	if(!IsOpen()) return 0.0f;
//	// TODO: This only works because the bricklist of each timestep is always completely filled
//	int32 firstPrefetchTimestep = max(m_currentTimestepIndex - m_prefetchPrevTimesteps, 0);
//	int32 lastPrefetchTimestep = min(m_currentTimestepIndex + m_prefetchNextTimesteps, GetTimestepCount() - 1);
//	int32 totalPrefetchTimesteps = lastPrefetchTimestep - firstPrefetchTimestep; // +1 for inclusive range, -1 to not count current timestep
//	if(totalPrefetchTimesteps == 0) return 1.0f;
//	int32 totalBrickCount = totalPrefetchTimesteps * (int32)GetCurrentTimestep().bricks.size();
//	return float(totalBrickCount - GetLoadingQueueSize()) / totalBrickCount;
//}
