#ifndef __TUM3D__INTEGRATOR_KERNELS_H__
#define __TUM3D__INTEGRATOR_KERNELS_H__


#include <global.h>

#include "AdvectMode.h"
#include "TextureFilterMode.h"
#include "TracingCommon.h"


void integratorKernelSimpleParticles(SimpleParticleVertex* dpParticles, uint particleCount, float deltaT, uint stepCountMax, eAdvectMode advectMode, eTextureFilterMode filterMode);

void integratorKernelStreamLines(const LineInfo& lineInfo, eAdvectMode advectMode, eTextureFilterMode filterMode);
void integratorKernelPathLines  (const LineInfo& lineInfo, eAdvectMode advectMode, eTextureFilterMode filterMode);
//void integratorKernelComputeFTLE(const LineInfo& lineInfo, eAdvectMode advectMode, eTextureFilterMode filterMode);
void integratorKernelComputeFTLE(SimpleParticleVertexDeltaT* dpParticles, uint particleCount, eAdvectMode advectMode, eTextureFilterMode filterMode);
void integratorKernelStreamLinesDense(const LineInfo& lineInfo, eAdvectMode advectMode, eTextureFilterMode filterMode);


#endif
