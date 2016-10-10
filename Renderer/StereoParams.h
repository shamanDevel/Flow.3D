#ifndef __TUM3D__STEREO_H__
#define __TUM3D__STEREO_H__


#include <global.h>

#include <ConfigFile.h>


enum EStereoEye
{
	EYE_CYCLOP,
	EYE_LEFT,
	EYE_RIGHT
};

struct StereoParams
{
	StereoParams();

	void Reset();

	void ApplyConfig(const ConfigFile& config);
	void WriteConfig(ConfigFile& config) const;

	bool     m_stereoEnabled;
	float    m_eyeDistance;

	bool operator==(const StereoParams& rhs) const;
	bool operator!=(const StereoParams& rhs) const;
};


#endif
