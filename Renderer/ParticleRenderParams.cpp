#include "ParticleRenderParams.h"

#include <cstring> // for memcmp
#include <string>

#include <cudaUtil.h>
#include <cuda_d3d11_interop.h>

using namespace tum3D;



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

ParticleRenderParams::ParticleRenderParams()
{
	Reset();
}

void ParticleRenderParams::Reset()
{
	m_linesEnabled = true;

	m_lineRenderMode = LINE_RENDER_PARTICLES;
	m_ribbonWidth = 0.8f;
	m_tubeRadius = 0.4f;

	m_particleRenderMode = PARTICLE_RENDER_ALPHA;
	m_particleTransparency = 0.3f;
	m_particleSize = 0.5f;

	m_tubeRadiusFromVelocity = true;
	m_referenceVelocity = 1.5f;

	m_lineColorMode = eLineColorMode::LINE_ID;

	m_color0 = Vec4f(0.0f, 0.251f, 1.0f, 1.0f);
	m_color1 = Vec4f(1.0f, 0.0f, 0.0f, 1.0f);
	m_pColorTexture = NULL;
	m_measure = eMeasure::MEASURE_VELOCITY;
	m_measureScale = 1.0;

	m_transferFunctionRangeMin = 0.0f;
	m_transferFunctionRangeMax = 1.0f;
	m_pTransferFunction = NULL;

	m_timeStripes = false;
	m_timeStripeLength = 0.03f;

	m_pSliceTexture = NULL;
	m_showSlice = false;
	m_slicePosition = 0;
	m_sliceAlpha = 0.5f;

	m_sortParticles = true;

	m_ftleShowTexture = true;
	m_ftleTextureAlpha = 1.0f;
}

void ParticleRenderParams::ApplyConfig(const ConfigFile& config)
{
	Reset();

	const std::vector<ConfigSection>& sections = config.GetSections();
	for(size_t s = 0; s < sections.size(); s++)
	{
		const ConfigSection& section = sections[s];
		if(section.GetName() == "ParticleRenderParams")
		{
			// this is our section - parse entries
			for(size_t e = 0; e < section.GetEntries().size(); e++) {
				const ConfigEntry& entry = section.GetEntries()[e];

				std::string entryName = entry.GetName();

				if(entryName == "LinesEnabled")
				{
					entry.GetAsBool(m_linesEnabled);
				}
				else if(entryName == "LineRenderMode")
				{
					std::string val;
					entry.GetAsString(val);
					m_lineRenderMode = GetLineRenderModeFromName(val);
				}
				else if(entryName == "RibbonWidth")
				{
					entry.GetAsFloat(m_ribbonWidth);
				}
				else if(entryName == "TubeRadius")
				{
					entry.GetAsFloat(m_tubeRadius);
				}
				else if(entryName == "TubeRadiusFromVelocity")
				{
					entry.GetAsBool(m_tubeRadiusFromVelocity);
				}
				else if(entryName == "ReferenceVelocity")
				{
					entry.GetAsFloat(m_referenceVelocity);
				}
				else if(entryName == "ColorMode")
				{
					std::string val;
					entry.GetAsString(val);
					m_lineColorMode = GetLineColorModeFromName(val);
				}
				else if(entryName == "Color0")
				{
					entry.GetAsVec4f(m_color0);
				}
				else if(entryName == "Color1")
				{
					entry.GetAsVec4f(m_color1);
				}
				else if(entryName == "TimeStripes")
				{
					entry.GetAsBool(m_timeStripes);
				}
				else if(entryName == "TimeStripeLength")
				{
					entry.GetAsFloat(m_timeStripeLength);
				}
				else
				{
					printf("WARNING: ParticleRenderParams::ApplyConfig: unknown entry \"%s\" ignored\n", entryName.c_str());
				}
			}
		}
	}
}

void ParticleRenderParams::WriteConfig(ConfigFile& config) const
{
	ConfigSection section("ParticleRenderParams");

	section.AddEntry(ConfigEntry("LinesEnabled", m_linesEnabled));
	section.AddEntry(ConfigEntry("LineRenderMode", GetLineRenderModeName(m_lineRenderMode)));
	section.AddEntry(ConfigEntry("RibbonWidth", m_ribbonWidth));
	section.AddEntry(ConfigEntry("TubeRadius", m_tubeRadius));
	section.AddEntry(ConfigEntry("TubeRadiusFromVelocity", m_tubeRadiusFromVelocity));
	section.AddEntry(ConfigEntry("ReferenceVelocity", m_referenceVelocity));
	section.AddEntry(ConfigEntry("ColorMode", GetLineColorModeName(m_lineColorMode)));
	section.AddEntry(ConfigEntry("Color0", m_color0));
	section.AddEntry(ConfigEntry("Color1", m_color1));
	section.AddEntry(ConfigEntry("TimeStripes", m_timeStripes));
	section.AddEntry(ConfigEntry("TimeStripeLength", m_timeStripeLength));

	config.AddSection(section);
}

bool ParticleRenderParams::operator==(const ParticleRenderParams& rhs) const
{
	return memcmp(this, &rhs, sizeof(ParticleRenderParams)) == 0;
}

bool ParticleRenderParams::operator!=(const ParticleRenderParams& rhs) const
{
	return !(*this == rhs);
}
