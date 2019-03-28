#ifndef __TUM3D__VIEWPARAMS_H__
#define __TUM3D__VIEWPARAMS_H__


#include <global.h>

#include <ConfigFile.h>
#include <Vec.h>

#include "StereoParams.h"


struct ViewParams
{
	ViewParams();

	void Reset();

	void ApplyConfig(const ConfigFile& config);
	void WriteConfig(ConfigFile& config) const;

	tum3D::Vec4f m_rotationQuat;
	tum3D::Vec3f m_lookAt;
	float        m_viewDistance;
	


	tum3D::Vec3f GetCameraPosition() const;
	tum3D::Vec3f GetViewDir() const;
	tum3D::Vec3f GetRightVector() const;

	tum3D::Mat4f BuildViewMatrix(EStereoEye eye, float eyeDistance) const;
	tum3D::Mat4f BuildRotationMatrix() const;

	bool operator==(const ViewParams& rhs) const;
	bool operator!=(const ViewParams& rhs) const;
};


#endif
