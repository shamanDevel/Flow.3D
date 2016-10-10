#ifndef __TUM3D__FILTERINGMANAGER_H__
#define __TUM3D__FILTERINGMANAGER_H__


#include <global.h>

#include <vector>

#include <TimerCPU.h>
#include <MultiTimerGPU.h>
#include <Vec.h>

#include "FilterParams.h"
#include "FilteredVolume.h"
#include "TimeVolume.h"
#include "VolumeFilter.h"

#include "GPUResources.h"
#include "CompressVolume.h"


class FilteringManager
{
public:
	FilteringManager();
	~FilteringManager();

	bool Create(GPUResources* pCompressShared, CompressVolumeResources* pCompressVolume);
	void Release();
	bool IsCreated() const { return m_isCreated; }

	bool StartFiltering(const TimeVolume& volume, const FilterParams& filterParams);
	bool Filter(); // returns true if finished TODO error code
	void CancelFiltering();
	bool IsFiltering() const;
	float GetFilteringProgress() const;

	const std::vector<const TimeVolumeIO::Brick*>& GetBricksToLoad() const { return m_bricksToLoad; }

	// release memory-heavy resources (will be recreated on StartFiltering)
	void ReleaseResources();

	size_t GetResultCount() const { return m_filteredVolume.size(); }
	const FilteredVolume& GetResult(size_t radiusIndex = 0) const { return m_filteredVolume[radiusIndex]; }
	const std::vector<FilteredVolume>& GetResults() const { return m_filteredVolume; }
	void ClearResult();


	struct Timings
	{
		Timings() : FilterWall(0.0f) {}
		bool operator==(const Timings& other) { return memcmp(this, &other, sizeof(*this)) == 0; }
		bool operator!=(const Timings& other) { return !(*this == other); }

		MultiTimerGPU::Stats UploadDecompressGPU;
		MultiTimerGPU::Stats ConvolutionGPU;
		MultiTimerGPU::Stats CompressDownloadGPU;

		float                FilterWall;
	};
	const Timings& GetTimings() const { return m_timings; }

private:
	bool CreateVolumeDependentResources();
	void ReleaseVolumeDependentResources();

	uint GetBrickLinearIndex(const tum3D::Vec3i& pos, EFilterDirection dir);
	tum3D::Vec3i GetBrickPos(uint linearIndex, EFilterDirection dir);
	const TimeVolumeIO::Brick* GetBrick(const tum3D::Vec3i& pos);

	static bool CompressDownload(GPUResources* pCompressShared, CompressVolumeResources* pCompressVolume, FilteredVolume::ChannelData& channelDataTarget, const float* dpBuffer, const tum3D::Vec3ui& size, bool compressed);


	void UpdateBricksToLoad();


	void UpdateTimings();


	bool m_isCreated;

	GPUResources*            m_pCompressShared;
	CompressVolumeResources* m_pCompressVolume;

	VolumeFilter m_volumeFilter;

	std::vector<FilteredVolume> m_filteredVolume;

	// volume-dependent resources
	int m_brickSize;
	// brick buffers
	float* m_dpBufferCenter;
	float* m_dpBufferLeft;
	float* m_dpBufferRight;
	float* m_dpBufferOut;

	// only valid while filtering
	const TimeVolume* m_pVolume;
	std::vector<const TimeVolumeIO::Brick*> m_bricks; // in xyz order
	FilterParams m_filterParams;
	uint m_radiusCount; // number of radii without trailing zeros
	uint m_radiusIndex;
	EFilterDirection m_direction;
	int m_channel;
	uint m_nextBrickToFilter;


	std::vector<const TimeVolumeIO::Brick*> m_bricksToLoad;


	// timing
	MultiTimerGPU m_timerUploadDecompress;
	MultiTimerGPU m_timerConvolution;
	MultiTimerGPU m_timerCompressDownload;
	TimerCPU      m_timerFilter;

	Timings m_timings;


	// disallow copy and assignment
	FilteringManager(const FilteringManager&);
	FilteringManager& operator=(const FilteringManager&);
};


#endif
