#ifndef __TUM3D__INTEGRATOR_PARTICLE_KERNELS_H__
#define __TUM3D__INTEGRATOR_PARTICLE_KERNELS_H__

#include <global.h>

#include "AdvectMode.h"
#include "TextureFilterMode.h"
#include "TracingCommon.h"
#include <VolumeInfoGPU.h>

//advects all particles if they are still valid (time>=0)
void integratorKernelParticles(LineInfoGPU lineInfo, VolumeInfoGPU volumeInfo, BrickIndexGPU brickIndex, BrickRequestsGPU brickRequests, eAdvectMode advectMode, eTextureFilterMode filterMode, double tpf);

//Seeds new particles at 'particleEntry' in the lines
void integratorKernelSeedParticles(LineInfoGPU lineInfo, int particleEntry);

//Initializes the particles
void integratorKernelInitParticles(LineInfoGPU lineInfo);

//void integratorKernelParticlesDense(const LineInfo& lineInfo, eAdvectMode advectMode, eTextureFilterMode filterMode);

#endif