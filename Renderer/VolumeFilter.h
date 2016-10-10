#ifndef __TUM3D__VOLUME_FILTER_H__
#define __TUM3D__VOLUME_FILTER_H__


#include <global.h>

#include <vector>

#include <Vec.h>


enum EFilterDirection
{
	DIR_X = 0,
	DIR_Y,
	DIR_Z,
	DIR_COUNT
};


class VolumeFilter
{
public:
	//VolumeFilter();
	//~VolumeFilter();

	//bool Create();
	//void Release();

	//void GetMaskedVelocity(const float* d_pDataX, const float* d_pDataY, const float* d_pDataZ, float* d_pOutX, float* d_pOutY, float* d_pOutZ, const tum3D::Vec3i& size, eMeasure measure, float threshold);
	//void GetMaskedJacobian(const float* d_pDataX, const float* d_pDataY, const float* d_pDataZ, float* d_pOut0, float* d_pOut1, float* d_pOut2, float* d_pOut3, float* d_pOut4, float* d_pOut5, float* d_pOut6, float* d_pOut7, float* d_pOut8, const tum3D::Vec3i& size, eMeasure measure, float threshold);

	struct ChannelData
	{
		const float* __restrict d_pData;
		const float* __restrict d_pLeft;
		const float* __restrict d_pRight;
		      float* __restrict d_pOut;
	};

	// box filter only for now!
	// sizeLeft and sizeRight are in the dimension of the filter, the other dimensions must match size
	// none of the data pointers may alias, but pLeft and pRight may be null (set sizeLeft and sizeRight to 0)
	void Filter(EFilterDirection dir, int radius, const ChannelData& data, const tum3D::Vec3ui& size, int overlap, int sizeLeft, int sizeRight);
};


#endif
