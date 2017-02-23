#ifndef __TUM3D__LINE_RENDER_MODE_H__
#define __TUM3D__LINE_RENDER_MODE_H__


#include <global.h>

#include <string>


enum eLineRenderMode
{
	LINE_RENDER_LINE = 0,
	LINE_RENDER_RIBBON,
	LINE_RENDER_TUBE,
	LINE_RENDER_PARTICLES,
	LINE_RENDER_MODE_COUNT
};
std::string GetLineRenderModeName(eLineRenderMode renderMode);
eLineRenderMode GetLineRenderModeFromName(const std::string& name);


#endif
