#ifndef __TUM3D__PROJECTIONPARAMS_H__
#define __TUM3D__PROJECTIONPARAMS_H__


#include <global.h>

#include <Vec.h>

#include "Range.h"
#include "StereoParams.h"


struct ProjectionParams
{
	ProjectionParams();

	void Reset();

	bool	 m_perspective;
	float    m_aspectRatio;
	float    m_fovy;
	float    m_near;
	float    m_far;

	uint     m_imageWidth;
	uint     m_imageHeight;

	int GetImageLeft (const Range1D& range) const;
	int GetImageRight(const Range1D& range) const;
	int GetImageWidth (const Range1D& range) const;
	int GetImageHeight(const Range1D& range) const;

	// frustum params/planes are in the following order: l, r, b, t, n, f
	void GetFrustumParams(float frustum[6], EStereoEye eye, float eyeDistance, const Range1D& range) const;
	void GetFrustumPlanes(tum3D::Vec4f planes[6], EStereoEye eye, float eyeDistance, const Range1D& range) const;

	tum3D::Mat4f BuildProjectionMatrix(EStereoEye eye, float eyeDistance, const Range1D& range) const;

	bool operator==(const ProjectionParams& rhs) const;
	bool operator!=(const ProjectionParams& rhs) const;
};


#endif
