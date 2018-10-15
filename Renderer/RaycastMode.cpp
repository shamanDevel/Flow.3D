#include "RaycastMode.h"


static const char* g_ColorModeNames[COLOR_MODE_COUNT + 1] =
{
	"UI Color",
	"Vorticity Alignment",
	"Unknown"
};

const char* GetColorModeName(eColorMode mode)
{
	return g_ColorModeNames[min(mode, COLOR_MODE_COUNT)];
}

eColorMode GetColorModeFromName(const std::string& name)
{
	for(uint i = 0; i < COLOR_MODE_COUNT; i++)
	{
		if(g_ColorModeNames[i] == name)
		{
			return eColorMode(i);
		}
	}
	return COLOR_MODE_COUNT;
}



static const char* g_raycastModeNames[RAYCAST_MODE_COUNT + 1] =
{
	"DVR",
	"DVR EE",
	"Iso",
	"Iso (SI)",
	"Dual Iso",
	"Dual Iso (SI)",
	"Dual Iso Separate (SI)",
	//"2-scale Iso (SI)",
	//"3-scale Iso (Cascade)",
	//"3-scale Iso (SI)",
	"Unknown"
};

const char* GetRaycastModeName(eRaycastMode mode)
{
	return g_raycastModeNames[min(mode, RAYCAST_MODE_COUNT)];
}

eRaycastMode GetRaycastModeFromName(const std::string& name)
{
	for(uint i = 0; i < RAYCAST_MODE_COUNT; i++)
	{
		if(g_raycastModeNames[i] == name)
		{
			return eRaycastMode(i);
		}
	}
	return RAYCAST_MODE_COUNT;
}

bool RaycastModeNeedsTransferFunction(eRaycastMode mode)
{
	switch(mode)
	{
		case RAYCAST_MODE_DVR:
		case RAYCAST_MODE_DVR_EE:
			return true;

		default:
			return false;
	}
}

bool RaycastModeIsMultiScale(eRaycastMode mode)
{
	return RaycastModeScaleCount(mode) > 1;
}

uint RaycastModeScaleCount(eRaycastMode mode)
{
	switch(mode)
	{
		case RAYCAST_MODE_DVR :
		case RAYCAST_MODE_DVR_EE :
		case RAYCAST_MODE_ISO :
		case RAYCAST_MODE_ISO_SI :
		case RAYCAST_MODE_ISO2 :
		case RAYCAST_MODE_ISO2_SI :
			return 1;
		//case RAYCAST_MODE_MS2_ISO_CASCADE :
		case RAYCAST_MODE_ISO2_SEPARATE :
			return 2;
		//case RAYCAST_MODE_MS3_ISO_CASCADE :
		//case RAYCAST_MODE_MS3_ISO_SI :
		//	return 3;
		default:
			return 0;
	}
}
