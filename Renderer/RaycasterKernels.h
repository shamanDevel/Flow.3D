#ifndef __TUM3D__RAYCASTER_KERNELS_H__
#define __TUM3D__RAYCASTER_KERNELS_H__


#include "RaycasterKernelParams.h"


void raycasterKernelDvr								(RaycasterKernelParams& params);
void raycasterKernelDvrEe							(RaycasterKernelParams& params);
void raycasterKernelIso								(RaycasterKernelParams& params);
void raycasterKernelIsoSi							(RaycasterKernelParams& params);
void raycasterKernelIso2							(RaycasterKernelParams& params);
void raycasterKernelIso2Si							(RaycasterKernelParams& params);
void raycasterKernelIso2Separate					(RaycasterKernelParams& params);


#endif
