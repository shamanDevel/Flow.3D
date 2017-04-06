#pragma once

#include <global.h>

#include <string>
#include <vector>

#include <Vec.h>


enum eCompressionType
{
	COMPRESSION_NONE = 0,
	COMPRESSION_FIXEDQUANT,
	COMPRESSION_FIXEDQUANT_QF,
	COMPRESSION_ADAPTIVEQUANT,
	COMPRESSION_TYPE_COUNT
};
std::string GetCompressionTypeName(eCompressionType comp);
eCompressionType GetCompressionTypeFromName(const std::string& name);


class TimeVolumeIO;

class TimeVolumeInfo
{
	//TODO initialize!
	friend class TimeVolumeIO;

public:
	const tum3D::Vec3i& GetVolumeSize() const { return m_volumeSize; }
	int32 GetChannelCount() const { return m_iChannels; }
	int32 GetTimestepCount() const { return m_iTimestepCount; }
	tum3D::Vec3f GetGridSpacing() const { return m_fGridSpacing; }
	float GetTimeSpacing() const { return m_fTimeSpacing; }
	bool IsPeriodic() const { return m_periodic; }

	int GetFloorTimestepIndex  (float time) const { return int(time / GetTimeSpacing()); }
	int GetCeilTimestepIndex   (float time) const { return GetFloorTimestepIndex(time) + 1; }
	int GetNearestTimestepIndex(float time) const { return int(time / GetTimeSpacing() + 0.5f); }

	int32 GetBrickSizeWithOverlap() const { return m_iBrickSize; }
	int32 GetBrickSizeWithoutOverlap() const { return m_iBrickSize - 2 * m_iOverlap; }
	int32 GetBrickOverlap() const { return m_iOverlap; } // width of the halo around each brick

	tum3D::Vec3i GetBrickCount() const { return (GetVolumeSize() + GetBrickSizeWithoutOverlap() - 1) / GetBrickSizeWithoutOverlap(); }

	uint GetBrickLinearIndex(const tum3D::Vec3i& spatialIndex) const;
	tum3D::Vec3i GetBrickSpatialIndex(uint linearIndex) const;

	tum3D::Vec3f GetVolumeHalfSizeWorld() const;
	tum3D::Vec3f GetBrickSizeWorld() const;
	tum3D::Vec3f GetBrickOverlapWorld() const;
	bool GetBrickBoxWorld(const tum3D::Vec3i& spatialIndex, tum3D::Vec3f& boxMin, tum3D::Vec3f& boxMax) const;

	tum3D::Vec3i GetContainingBrickSpatialIndex(const tum3D::Vec3f& posWorld) const;

	tum3D::Vec3f GetPhysicalToWorldFactor() const { return 2.0f / (float(GetVolumeSize().maximum()) * GetGridSpacing()); }
	float GetVoxelToWorldFactor()    const { return 2.0f / float(GetVolumeSize().maximum()); }

	bool IsCompressed() const { return m_compression != COMPRESSION_NONE; }
	eCompressionType GetCompressionType() const { return m_compression; }
	const std::vector<float>& GetQuantSteps() const { return m_vQuantSteps; } // only for COMPRESSION_FIXEDQUANT/_QF
	float GetQuantStep(int32 channel) const { return m_vQuantSteps[channel]; } // only for COMPRESSION_FIXEDQUANT/_QF
	bool GetUseLessRLE() const { return m_lessRLE; }
	uint GetHuffmanBitsMax() const { return m_huffmanBitsMax; }

private:
	tum3D::Vec3i	m_volumeSize;
	int32			m_iChannels;
	int32			m_iTimestepCount;
	tum3D::Vec3f	m_fGridSpacing;
	float			m_fTimeSpacing;
	bool			m_periodic;

	int32			m_iBrickSize;
	int32			m_iOverlap;

	eCompressionType	m_compression;
	std::vector<float>	m_vQuantSteps;
	bool				m_lessRLE;
	uint				m_huffmanBitsMax;
};
