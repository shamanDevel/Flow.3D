#include "TimeVolumeInfo.h"

using namespace tum3D;


static const std::string g_CompressionTypeNames[COMPRESSION_TYPE_COUNT + 1] =
{
	"None",
	"Fixed Quantization",
	"Fixed Quantization (QuantFirst)",
	"Adaptive Quantization",
	"Unknown"
};

std::string GetCompressionTypeName(eCompressionType comp)
{
	return g_CompressionTypeNames[min(comp, COMPRESSION_TYPE_COUNT)];
}

eCompressionType GetCompressionTypeFromName(const std::string& name)
{
	for(uint i = 0; i < COMPRESSION_TYPE_COUNT; i++)
	{
		if(g_CompressionTypeNames[i] == name)
		{
			return eCompressionType(i);
		}
	}
	return COMPRESSION_TYPE_COUNT;
}



uint TimeVolumeInfo::GetBrickLinearIndex(const Vec3i& spatialIndex) const
{
	Vec3i brickCount = GetBrickCount();
	return spatialIndex.x() + brickCount.x() * (spatialIndex.y() + brickCount.y() * spatialIndex.z());
}

Vec3i TimeVolumeInfo::GetBrickSpatialIndex(uint linearIndex) const
{
	Vec3i brickCount = GetBrickCount();
	return Vec3i(linearIndex % brickCount.x(), (linearIndex / brickCount.x()) % brickCount.y(), linearIndex / (brickCount.x() * brickCount.y()));
}


Vec3f TimeVolumeInfo::GetVolumeHalfSizeWorld() const
{
	Vec3f result;
	castVec(GetVolumeSize(), result);
	//result /= result.maximum();
	result *= GetGridSpacing();
	return result;
}

Vec3f TimeVolumeInfo::GetBrickSizeWorld() const
{
	Vec3f volumeSize;
	castVec(GetVolumeSize(), volumeSize);
	float result = float(GetBrickSizeWithoutOverlap());
	//result /= volumeSize.maximum() * 0.5f;
	result *= 2.0f;
	return GetGridSpacing() * result;
}

Vec3f TimeVolumeInfo::GetBrickOverlapWorld() const
{
	return GetBrickSizeWorld() * (float(GetBrickOverlap()) / float(GetBrickSizeWithoutOverlap()));
}

bool TimeVolumeInfo::GetBrickBoxWorld(const tum3D::Vec3i& spatialIndex, tum3D::Vec3f& boxMin, tum3D::Vec3f& boxMax) const
{
	Vec3f brickIndexFloat; castVec(spatialIndex, brickIndexFloat);
	Vec3f volumeHalfSizeWorld = GetVolumeHalfSizeWorld();
	Vec3f brickSizeWorld = GetBrickSizeWorld();
	boxMin = -volumeHalfSizeWorld + brickIndexFloat * brickSizeWorld;
	boxMax = boxMin + brickSizeWorld;
	minimum(boxMax, volumeHalfSizeWorld); // clamp against global volume box

	// all(boxMin < boxMax)
	return boxMin.compLess(boxMax).minimum();
}

Vec3i TimeVolumeInfo::GetContainingBrickSpatialIndex(const Vec3f& posWorld) const
{
	Vec3f brickSize = GetBrickSizeWorld();
	const Vec3f& volumeHalfSizeWorld = GetVolumeHalfSizeWorld();
	return Vec3i((posWorld + volumeHalfSizeWorld) / brickSize);
}
