#pragma once

#include <cuda_d3d11_interop.h>

// Data structure for 2D texture shared between DX11 and CUDA.
// Source: cuda 9.2 samples.
struct D3D11CudaTexture
{
	DXGI_FORMAT					format = DXGI_FORMAT::DXGI_FORMAT_UNKNOWN;
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

	bool IsFormatSupported(DXGI_FORMAT format);

	int GetNumberOfComponents(DXGI_FORMAT format);

	int GetNumberOfComponents();
};
