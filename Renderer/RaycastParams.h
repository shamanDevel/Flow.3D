#ifndef __TUM3D__RAYCASTPARAMS_H__
#define __TUM3D__RAYCASTPARAMS_H__


#include <global.h>

#include <string>

#include <ConfigFile.h>
#include <Vec.h>

#include "Measure.h"
#include "RaycastMode.h"
#include "TextureFilterMode.h"


struct RaycastParams
{
	RaycastParams();

	void Reset();

	void ApplyConfig(const ConfigFile& config);
	void WriteConfig(ConfigFile& config) const;

	bool				m_raycastingEnabled;

	eMeasure			m_measure1;
	eMeasure			m_measure2;
	float				m_measureScale1;
	float				m_measureScale2;
	eMeasureComputeMode	m_measureComputeMode;

	uint				m_filterOffset; // offset into array of filtered volumes

	eColorMode			m_colorMode;
	eRaycastMode		m_raycastMode;
	eTextureFilterMode	m_textureFilterMode;
	float				m_sampleRate;

	float				m_density;
	float				m_alphaScale;
	float				m_transferFunctionRangeMin;
	float				m_transferFunctionRangeMax;

	float				m_isoValue1;
	float				m_isoValue2;
	float				m_isoValue3;
	tum3D::Vec4f		m_isoColor1;
	tum3D::Vec4f		m_isoColor2;
	tum3D::Vec4f		m_isoColor3;

	tum3D::Vec3f		m_clipBoxMin;
	tum3D::Vec3f		m_clipBoxMax;

	bool operator==(const RaycastParams& rhs) const;
	bool operator!=(const RaycastParams& rhs) const;
};


#endif
