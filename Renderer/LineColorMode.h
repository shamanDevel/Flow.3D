#ifndef __TUM3D__LINE_COLOR_MODE_H__
#define __TUM3D__LINE_COLOR_MODE_H__


#include <global.h>

#include <string>


enum eLineColorMode
{
	LINE_ID = 0,
	AGE,
	TEXTURE,
	MEASURE,
	LINE_COLOR_MODE_COUNT
};
const char* GetLineColorModeName(eLineColorMode renderMode);
eLineColorMode GetLineColorModeFromName(const std::string& name);


#endif
