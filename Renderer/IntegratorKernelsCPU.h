#ifndef __TUM3D__INTEGRATOR_KERNELS_CPU_H__
#define __TUM3D__INTEGRATOR_KERNELS_CPU_H__


#include "AdvectMode.h"
#include "TextureFilterMode.h"
#include "TracingCommon.h"


void integratorKernelStreamLinesDenseCPU(const LineInfo& lineInfo, eAdvectMode advectMode, eTextureFilterMode filterMode);


#endif
