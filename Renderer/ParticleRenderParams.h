#ifndef __TUM3D__PARTICLERENDERPARAMS_H__
#define __TUM3D__PARTICLERENDERPARAMS_H__


#include <global.h>

#include <ConfigFile.h>
#include <Vec.h>
#include <D3D11.h>

#include "LineRenderMode.h"
#include "ParticleRenderMode.h"
#include "LineColorMode.h"
#include "Measure.h"


// Data structure for 2D texture shared between DX11 and CUDA.
// Source: cuda 9.2 samples.
struct D3D11CudaTexture
{
	ID3D11Texture2D				*pTexture = nullptr;
	ID3D11ShaderResourceView	*pSRView = nullptr;
	cudaGraphicsResource		*cudaResource = nullptr;
	void						*cudaLinearMemory = nullptr;
	size_t						pitch = 0;
	int							width = 0;
	int							height = 0;
#ifndef USEEFFECT
	int							offsetInShader = 0;
#endif

	bool IsTextureCreated();

	bool IsRegisteredWithCuda();

	bool CreateTexture(ID3D11Device* device, int width, int height, int miplevels, int arraysize, DXGI_FORMAT format);

	void ReleaseResources();

	void RegisterCUDAResources();

	void UnregisterCudaResources();
};


struct ParticleRenderParams
{
	ParticleRenderParams();

	void Reset();

	void ApplyConfig(const ConfigFile& config);
	void WriteConfig(ConfigFile& config) const;

	bool m_linesEnabled;

	eLineRenderMode m_lineRenderMode;
	float m_ribbonWidth;
	float m_tubeRadius;

	float m_particleSize;
	float m_particleTransparency;
	eParticleRenderMode m_particleRenderMode;

	bool  m_tubeRadiusFromVelocity;
	float m_referenceVelocity;

	eLineColorMode m_lineColorMode;

	tum3D::Vec4f m_color0;
	tum3D::Vec4f m_color1;
	ID3D11ShaderResourceView* m_pColorTexture;
	eMeasure m_measure;
	float m_measureScale;

	float m_transferFunctionRangeMin;
	float m_transferFunctionRangeMax;
	ID3D11ShaderResourceView* m_pTransferFunction;

	bool  m_timeStripes;
	float m_timeStripeLength;

	ID3D11ShaderResourceView* m_pSliceTexture;
	bool m_showSlice;
	float m_slicePosition;
	float m_sliceAlpha;

	bool m_sortParticles;

	bool operator==(const ParticleRenderParams& rhs) const;
	bool operator!=(const ParticleRenderParams& rhs) const;

	// FTLE stuff
	bool m_ftleShowTexture;
	float m_ftleTextureAlpha;
};


#endif
