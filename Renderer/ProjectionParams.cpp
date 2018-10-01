#include "ProjectionParams.h"

using namespace tum3D;


ProjectionParams::ProjectionParams()
{
	Reset();
}

void ProjectionParams::Reset()
{
	// parameters chosen to be "somewhat" correct for stereo rendering:
	// 25" 16:9 monitor, 70 cm in front of the viewer
	m_aspectRatio = 16.0f / 9.0f;
	m_fovy = 30.0f * PI / 180.0f; // this should be 24 deg, but a bit larger fov looks better...
	m_near = 0.01f; //0.7f;
	m_far = 100.0f;
	m_perspective = true;

	m_imageWidth = 0;
	m_imageHeight = 0;
}


int ProjectionParams::GetImageLeft(const Range1D& range) const
{
	return range.GetMinInt((int)m_imageWidth - 1);
}

int ProjectionParams::GetImageRight(const Range1D& range) const
{
	return range.GetMaxInt((int)m_imageWidth - 1);
}

int ProjectionParams::GetImageWidth(const Range1D& range) const
{
	return GetImageRight(range) - GetImageLeft(range) + 1;
}

int ProjectionParams::GetImageHeight(const Range1D& range) const
{
	return m_imageHeight;
}


void ProjectionParams::GetFrustumParams(float frustum[6], EStereoEye eye, float eyeDistance, const Range1D& range) const
{
	float halfHeight = tan(0.5f * m_fovy) * m_near;
	float halfWidth = halfHeight * m_aspectRatio;
	float eyeOffset = 0.0f;
	switch(eye) {
		case EYE_LEFT:  eyeOffset =  0.5f * eyeDistance; break;
		case EYE_RIGHT: eyeOffset = -0.5f * eyeDistance; break;
		// EYE_CYCLOP: nop
	}
	// adjustment factors for range
	// semi-hack: actually we should convert the range to int and back for correct snapping to pixels
	float factorLeft  = -((range.GetMin() * 2.0f) - 1.0f);
	float factorRight =   (range.GetMax() * 2.0f) - 1.0f;
	frustum[0] = -halfWidth * factorLeft  + eyeOffset;
	frustum[1] =  halfWidth * factorRight + eyeOffset;
	frustum[2] = -halfHeight;
	frustum[3] =  halfHeight;
	frustum[4] = m_near;
	frustum[5] = m_far;
}

void ProjectionParams::GetFrustumPlanes(tum3D::Vec4f planes[6], EStereoEye eye, float eyeDistance, const Range1D& range) const
{
	float frustumParams[6];
	GetFrustumParams(frustumParams, eye, eyeDistance, range);

	planes[0][0] = frustumParams[4];
	planes[0][1] = 0.0f;
	planes[0][2] = frustumParams[0];
	planes[0][3] = 0.0f;

	planes[1][0] = -frustumParams[4];
	planes[1][1] = 0.0f;
	planes[1][2] = -frustumParams[1];
	planes[1][3] = 0.0f;

	planes[2][0] = 0.0f;
	planes[2][1] = frustumParams[4];
	planes[2][2] = frustumParams[2];
	planes[2][3] = 0.0f;

	planes[3][0] = 0.0f;
	planes[3][1] = -frustumParams[4];
	planes[3][2] = -frustumParams[3];
	planes[3][3] = 0.0f;

	planes[4][0] = 0.0f;
	planes[4][1] = 0.0f;
	planes[4][2] = -1.0f;
	planes[4][3] = -frustumParams[4];

	planes[5][0] = 0.0f;
	planes[5][1] = 0.0f;
	planes[5][2] = 1.0f;
	planes[5][3] = frustumParams[5];

	// normalize planes
	// this only works because .w == 0 - we actually only want to normalize .xyz
	normalize(planes[0]);
	normalize(planes[1]);
	normalize(planes[2]);
	normalize(planes[3]);
	// 4 and 5 are already normalized
}


Mat4f ProjectionParams::BuildProjectionMatrix(EStereoEye eye, float eyeDistance, const Range1D& range) const
{
	float frustum[6];
	GetFrustumParams(frustum, eye, eyeDistance, range);

	Mat4f proj;
	if (m_perspective)
		perspectiveOffCenterProjMatD3D(frustum[0], frustum[1], frustum[2], frustum[3], frustum[4], frustum[5], proj);
	else
		orthoOffCenterProjMatD3D(frustum[0], frustum[1], frustum[2], frustum[3], frustum[4], frustum[5], proj);

	return proj;
}


bool ProjectionParams::operator==(const ProjectionParams& rhs) const
{
	return memcmp(this, &rhs, sizeof(ProjectionParams)) == 0;
}

bool ProjectionParams::operator!=(const ProjectionParams& rhs) const
{
	return !(*this == rhs);
}
