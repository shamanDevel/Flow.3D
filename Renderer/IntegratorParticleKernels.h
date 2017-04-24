#ifndef __TUM3D__INTEGRATOR_PARTICLE_KERNELS_H__
#define __TUM3D__INTEGRATOR_PARTICLE_KERNELS_H__

#include <global.h>

#include "AdvectMode.h"
#include "TextureFilterMode.h"
#include "TracingCommon.h"

//advects all particles if they are still valid (time>=0)
void integratorKernelParticles(const LineInfo& lineInfo, eAdvectMode advectMode, eTextureFilterMode filterMode, double tpf);

//Seeds new particles at 'particleEntry' in the lines
void integratorKernelSeedParticles(const LineInfo& lineInfo, int particleEntry);

//Initializes the particles
void integratorKernelInitParticles(const LineInfo& lineInfo);

//void integratorKernelParticlesDense(const LineInfo& lineInfo, eAdvectMode advectMode, eTextureFilterMode filterMode);

#endif