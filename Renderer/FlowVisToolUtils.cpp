#include <FlowVisToolUtils.h>

#include <iostream>

#include <cudaUtil.h>
#include <global.h>

bool D3D11CudaTexture::IsTextureCreated()
{
	return pTexture != nullptr;
}

bool D3D11CudaTexture::IsRegisteredWithCuda()
{
	return cudaResource != nullptr;
}

bool D3D11CudaTexture::CreateTexture(ID3D11Device* device, int width, int height, int miplevels, int arraysize, DXGI_FORMAT format)
{
	if (!IsFormatSupported(format))
	{
		std::cout << "Unsupported texture format." << std::endl;
		return false;
	}

	if (IsTextureCreated())
		ReleaseResources();

	this->format = format;

	this->width = width;
	this->height = height;

	D3D11_TEXTURE2D_DESC desc;
	ZeroMemory(&desc, sizeof(D3D11_TEXTURE2D_DESC));
	desc.Width = width;
	desc.Height = height;
	desc.MipLevels = miplevels;
	desc.ArraySize = arraysize;
	desc.Format = format;
	desc.SampleDesc.Count = 1;
	desc.Usage = D3D11_USAGE_DEFAULT;
	desc.BindFlags = D3D11_BIND_SHADER_RESOURCE;

	/*size_t size = width * height * 4;
	float* arr = new float[size];

	for (size_t i = 0; i < size; i++)
	arr[i] = 0.5;

	D3D11_SUBRESOURCE_DATA data;

	data.pSysMem = arr;
	data.SysMemPitch = width * 4 * sizeof(float);*/

	if (FAILED(device->CreateTexture2D(&desc, nullptr, &pTexture)))
	{
		return false; //return E_FAIL;
	}

	if (FAILED(device->CreateShaderResourceView(pTexture, nullptr, &pSRView)))
	{
		return false; //return E_FAIL;
	}

	return true;
}

void D3D11CudaTexture::ReleaseResources()
{
	if (IsRegisteredWithCuda())
		UnregisterCudaResources();

	if (pTexture)
	{
		uint count = pTexture->Release();
		std::cout << "Count after release: " << count << std::endl;
	}
	if (pSRView)
	{
		uint count = pSRView->Release();
		std::cout << "Count after release: " << count << std::endl;

	}

	format = DXGI_FORMAT::DXGI_FORMAT_UNKNOWN;

	pSRView = nullptr;
	pTexture = nullptr;

	width = 0;
	height = 0;
#ifndef USEEFFECT
	offsetInShader = 0;
#endif
}

void D3D11CudaTexture::RegisterCUDAResources()
{
	// register the Direct3D resources that we'll use
	// we'll read to and write from g_texture_2d, so don't set any special map flags for it
	cudaGraphicsD3D11RegisterResource(&cudaResource, pTexture, cudaGraphicsRegisterFlagsNone);
	cudaCheckMsg("---------- cudaGraphicsD3D11RegisterResource (D3D11CudaTexture) failed");
	// cuda cannot write into the texture directly : the texture is seen as a cudaArray and can only be mapped as a texture
	// Create a buffer so that cuda can write into it
	// pixel fmt is DXGI_FORMAT_R32G32B32A32_FLOAT
	cudaMallocPitch(&cudaLinearMemory, &pitch, width * sizeof(float) * GetNumberOfComponents(format), height);
	cudaCheckMsg("---------- cudaMallocPitch (D3D11CudaTexture) failed");
	cudaMemset(cudaLinearMemory, 1, pitch * height);
}

void D3D11CudaTexture::UnregisterCudaResources()
{
	if (!IsRegisteredWithCuda())
		return;

	cudaGraphicsUnregisterResource(cudaResource);
	cudaCheckMsg("cudaGraphicsUnregisterResource (D3D11CudaTexture) failed");
	cudaFree(cudaLinearMemory);
	cudaCheckMsg("cudaFree (D3D11CudaTexture) failed");

	pitch = 0;
	cudaResource = nullptr;
	cudaLinearMemory = nullptr;
}

bool D3D11CudaTexture::IsFormatSupported(DXGI_FORMAT format)
{
	switch (format)
	{
	case DXGI_FORMAT_R32G32B32A32_FLOAT:
	case DXGI_FORMAT_R32G32B32_FLOAT:
	case DXGI_FORMAT_R32G32_FLOAT:
	case DXGI_FORMAT_R32_FLOAT:
		return true;
	default:
		return false;
	}
}

int D3D11CudaTexture::GetNumberOfComponents(DXGI_FORMAT format)
{
	switch (format)
	{
	case DXGI_FORMAT_R32G32B32A32_FLOAT:
		return 4;
	case DXGI_FORMAT_R32G32B32_FLOAT:
		return 3;
	case DXGI_FORMAT_R32G32_FLOAT:
		return 2;
	case DXGI_FORMAT_R32_FLOAT:
		return 1;
	default:
		return 0;
	}
}
int D3D11CudaTexture::GetNumberOfComponents()
{
	return GetNumberOfComponents(format);
}
