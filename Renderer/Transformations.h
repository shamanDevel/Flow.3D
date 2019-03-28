#ifndef __TUM3D__TRANSFORMATION_H__
#define __TUM3D__TRANSFORMATION_H__

#include <FlowVisTool.h>

void XY_translation(float x, float y, float panSens, float DeltaTime, FlowVisTool & flowVisTool);
void XY_Rotation(float x, float y, float orbitSens, float DeltaTime, FlowVisTool & flowVisTool);
void zoomInOut(float z, float zoomSens, float DeltaTime, FlowVisTool & flowVisTool);

#endif
