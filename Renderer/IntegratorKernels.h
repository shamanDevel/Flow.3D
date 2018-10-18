#ifndef __TUM3D__INTEGRATOR_KERNELS_H__
#define __TUM3D__INTEGRATOR_KERNELS_H__


#include <global.h>

#include "AdvectMode.h"
#include "TextureFilterMode.h"
#include "TracingCommon.h"


void integratorKernelSimpleParticles(SimpleParticleVertex* dpParticles, VolumeInfoGPU volumeInfo, uint particleCount, float deltaT, uint stepCountMax, eAdvectMode advectMode, eTextureFilterMode filterMode);

void integratorKernelStreamLines(LineInfoGPU lineInfo, VolumeInfoGPU volumeInfo, eAdvectMode advectMode, eTextureFilterMode filterMode);
void integratorKernelPathLines(LineInfoGPU lineInfo, VolumeInfoGPU volumeInfo, eAdvectMode advectMode, eTextureFilterMode filterMode);
//void integratorKernelComputeFTLE(const LineInfo& lineInfo, eAdvectMode advectMode, eTextureFilterMode filterMode);
void integratorKernelComputeFTLE(SimpleParticleVertexDeltaT* dpParticles, VolumeInfoGPU volumeInfo, uint particleCount, eAdvectMode advectMode, eTextureFilterMode filterMode, bool invertVelocity);
void integratorKernelStreamLinesDense(LineInfoGPU lineInfo, VolumeInfoGPU volumeInfo, eAdvectMode advectMode, eTextureFilterMode filterMode);


#endif
