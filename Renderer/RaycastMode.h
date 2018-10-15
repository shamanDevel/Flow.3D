#ifndef __TUM3D__RAYCASTMODE_H__
#define __TUM3D__RAYCASTMODE_H__


#include <global.h>

#include <string>


enum eColorMode
{
	COLOR_MODE_UI=0,
	COLOR_MODE_VORTICITY_ALIGNMENT,
	COLOR_MODE_COUNT,
	COLOR_MODE_FORCE32 = 0xFFFFFFFF
};
const char* GetColorModeName(eColorMode mode);
eColorMode GetColorModeFromName(const std::string& name);


enum eRaycastMode
{
	RAYCAST_MODE_DVR=0,
	RAYCAST_MODE_DVR_EE,
	RAYCAST_MODE_ISO,
	RAYCAST_MODE_ISO_SI,
	RAYCAST_MODE_ISO2,
	RAYCAST_MODE_ISO2_SI,
	//RAYCAST_MODE_MS2_ISO_CASCADE,
	RAYCAST_MODE_ISO2_SEPARATE,
	//RAYCAST_MODE_MS3_ISO_CASCADE,
	//RAYCAST_MODE_MS3_ISO_SI,
	RAYCAST_MODE_COUNT,
	RAYCAST_MODE_FORCE32 = 0xFFFFFFFF
};
const char* GetRaycastModeName(eRaycastMode mode);
eRaycastMode GetRaycastModeFromName(const std::string& name);
bool RaycastModeNeedsTransferFunction(eRaycastMode mode);
bool RaycastModeIsMultiScale(eRaycastMode mode);
uint RaycastModeScaleCount(eRaycastMode mode);


#endif
