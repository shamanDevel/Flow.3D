#pragma once

#include <Vec.h>

struct RenderingParameters
{
	RenderingParameters()
	{
		Reset();
	}

	void Reset()
	{
		m_FixedLightDir = false;
		m_lightDir = tum3D::Vec3f(0.0);
		m_backgroundColor = tum3D::Vec4f(0.1f, 0.1f, 0.1f, 1.0f);
		m_renderBufferSizeFactor = 2.0f;
		m_windowSize = tum3D::Vec2i(0, 0);
		m_showPreview = true;
		m_showPreview = false;
	}

	bool m_showPreview;
	bool m_redraw;

	bool			m_FixedLightDir;
	tum3D::Vec3f	m_lightDir;

	tum3D::Vec4f	m_backgroundColor;
	float			m_renderBufferSizeFactor;

	tum3D::Vec2i	m_windowSize;
};