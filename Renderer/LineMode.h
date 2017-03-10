#ifndef __TUM3D__LINE_MODE_H__
#define __TUM3D__LINE_MODE_H__


#include <global.h>

#include <string>


enum eLineMode
{
	LINE_STREAM = 0,
	LINE_PATH,
	LINE_PARTICLE_STREAM,
	//LINE_STREAK,
	LINE_MODE_COUNT
};
std::string GetLineModeName(eLineMode mode);
eLineMode GetLineModeFromName(const std::string& name);

//Returns true if the advection depends on the time, i.e. timesteps in the future are needed
bool LineModeIsTimeDependent(eLineMode mode);

//Returns true if the line mode is a particle mode, i.e. the tracing and rendering should be called all the time
bool LineModeIsIterative(eLineMode mode);

#endif
