#ifndef __TUM3D__LINE_MODE_H__
#define __TUM3D__LINE_MODE_H__


#include <global.h>

#include <string>


enum eLineMode
{
	LINE_STREAM = 0,
	LINE_PATH,
	//LINE_STREAK,
	LINE_MODE_COUNT
};
std::string GetLineModeName(eLineMode mode);
eLineMode GetLineModeFromName(const std::string& name);

bool LineModeIsTimeDependent(eLineMode mode);


#endif
