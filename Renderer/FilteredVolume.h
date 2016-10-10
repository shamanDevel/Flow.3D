#ifndef __TUM3D__FILTEREDVOLUME_H__
#define __TUM3D__FILTEREDVOLUME_H__


#include <global.h>

#include <vector>

#include <Vec.h>



class FilteredVolume
{
public:
	class ChannelData
	{
	public:
		ChannelData() : m_quantStep(0.0f), m_pData(nullptr), m_dataSizeInUInts(0) {}
		ChannelData(ChannelData&& other);
		~ChannelData();

		ChannelData& operator=(ChannelData&& other);

		void Alloc(size_t sizeInUInts);
		void Clear();

		float  m_quantStep;
		uint*  m_pData;
		size_t m_dataSizeInUInts;

	private:
		// disallow copy and assignment
		ChannelData(const ChannelData&);
		ChannelData& operator=(const ChannelData&);
	};


	FilteredVolume();
	FilteredVolume(const tum3D::Vec3i& brickCount, int channelCount);
	FilteredVolume(FilteredVolume&& other);
	~FilteredVolume() {}

	FilteredVolume& operator=(FilteredVolume&& other);

	void Resize(const tum3D::Vec3i& brickCount, int channelCount);
	int GetChannelCount() const { return m_channelCount; }
	void Clear();
	bool IsEmpty() const { return m_data.empty(); }

	int GetLinearBrickIndex(const tum3D::Vec3i& brickIndex) const;

	ChannelData& GetChannelData(const tum3D::Vec3i& brickIndex, int channel);
	const ChannelData& GetChannelData(const tum3D::Vec3i& brickIndex, int channel) const;

	void ClearAllBricks();

	size_t GetTotalSizeBytes() const;

private:
	tum3D::Vec3i m_brickCount;
	int m_channelCount;
	std::vector<ChannelData> m_data;

	// disallow copy and assignment
	FilteredVolume(const FilteredVolume&);
	FilteredVolume& operator=(const FilteredVolume&);
};


#endif
