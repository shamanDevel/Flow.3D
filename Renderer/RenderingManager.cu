#include "RenderingManager.h"

#include <cassert>
#include <numeric>
#include <algorithm>

#include <cudaUtil.h>
#include <thrust/device_ptr.h>
#include <thrust\sort.h>

#include <LargeArray3D.h>

#include "BoxUtils.h"
#include "BrickUpload.h"
#include "ClearCudaArray.h"
#include "TracingCommon.h"

#include <cmath>

#include "LineInfoGPU.h"
#include "MatrixMath.cuh"


#pragma region CudaKernels
__global__ void FillVertexDepth(const LineVertex* vertices, const uint* indices, float* depthOut, float4 vec, uint maxIndex)
{
	uint index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index > maxIndex) return;

	float4 pos = make_float4(vertices[indices[index]].Position, 1);
	depthOut[index] = -dot(pos, vec);
}

__device__ float maxRoot(float A, float B, float C)
{
	// normal form: x^3 + Ax^2 + Bx + C = 0
	// adapted from:  http://read.pudn.com/downloads21/sourcecode/graph/71499/gems/Roots3And4.c__.htm   (Jochen Schwarze, in Graphics Gems 1990)

	// substitute x = y - A/3 to eliminate quadric term: x^3 +px + q = 0  
	float sq_A = A * A;
	float p = 1.0 / 3 * (-1.0 / 3 * sq_A + B);
	float q = 1.0 / 2 * (2.0 / 27 * A * sq_A - 1.0 / 3 * A * B + C);

	// use Cardano's formula
	float cb_p = p * p * p;
	float D = q * q + cb_p;

	if (D == 0)
	{
		if (q == 0)  // one triple solution
		{
			return -1.0 / 3.0 * A;
		}
		else  // one single and one double solution
		{
			float u = cbrt(-q);
			return max(2 * u, -u) - 1.0 / 3.0 * A;
		}
	}
	else
	{
		if (D < 0)  // Casus irreducibilis: three real solutions
		{
			float phi = 1.0 / 3 * acos(-q / sqrt(-cb_p));
			float t = 2 * sqrt(-p);
			return max(max(
				t * cos(phi),
				-t * cos(phi + 3.14159265359 / 3)),
				-t * cos(phi - 3.14159265359 / 3)) - 1.0 / 3.0 * A;
		}
		else // one real solution
		{
			float sqrt_D = sqrt(D);
			float u = cbrt(sqrt_D - q);
			float v = -cbrt(sqrt_D + q);
			return u + v - 1.0 / 3.0 * A;
		}
	}
}

__device__ float LambdaMax(const float3x3 &m)
{
	// Computes the largest eigenvalue of a 3x3 matrix.

	float a = m.m[0].x;  float b = m.m[0].y;  float c = m.m[0].z;
	float d = m.m[1].x;  float e = m.m[1].y;  float f = m.m[1].z;
	float g = m.m[2].x;  float h = m.m[2].y;  float i = m.m[2].z;

	// determinant has the following polynomial x*x*x + a2*x*x + a1*x + a0 = 0 = det(P)
	float a2 = -(i + e + a);
	float a1 = -(-e*i - a*i + f*h + c*g - a*e + b*d);
	float a0 = -(a*(e*i - f*h) - b*(d*i - f*g) + c*(d*h - e*g));

	return maxRoot(a2, a1, a0);
}

__global__ void ComputeFTLEKernel(unsigned char *surface, int width, int height, size_t pitch, float3 separationDist, float spawnTime, SimpleParticleVertexDeltaT* dpParticles, uint particleCount, float scale)
{
	int x = blockIdx.x*blockDim.x + threadIdx.x;
	int y = blockIdx.y*blockDim.y + threadIdx.y;

	// in the case where, due to quantization into grids, we have
	// more threads than pixels, skip the threads which don't
	// correspond to valid pixels
	if (x >= width || y >= height) return;

	int index = (x * width + y) * 6;

	//assert(index < c_lineInfo.lineCount);

	if (index >= particleCount)
		return;

	float currTime = dpParticles[index + 0].Time;

	float3 posX0 = dpParticles[index + 0].Position;
	float3 posX1 = dpParticles[index + 1].Position;

	float3 posY0 = dpParticles[index + 2].Position;
	float3 posY1 = dpParticles[index + 3].Position;

	float3 posZ0 = dpParticles[index + 4].Position;
	float3 posZ1 = dpParticles[index + 5].Position;

	float3 dx = (posX1 - posX0) / (2.0 * separationDist.x);
	float3 dy = (posY1 - posY0) / (2.0 * separationDist.y);
	float3 dz = (posZ1 - posZ0) / (2.0 * separationDist.z);

	// setup Jacobian and Cauchy Green tensor P
	//float3x3 J = float3x3(dx.x, dy.x, dz.x, dx.y, dy.y, dz.y, dx.z, dy.z, dz.z);
	float3x3 J;
	J.m[0] = make_float3(dx.x, dy.x, dz.x);
	J.m[1] = make_float3(dx.y, dy.y, dz.y);
	J.m[2] = make_float3(dx.z, dy.z, dz.z);

	//float3x3 JT = J;	JT.transpose();
	float3x3 JT;
	JT.m[0] = make_float3(dx.x, dx.y, dx.z);
	JT.m[1] = make_float3(dy.x, dy.y, dy.z);
	JT.m[2] = make_float3(dz.x, dz.y, dz.z);

	//float3x3 P = JT*J;
	float3x3 P = multMat3x3(JT, J);

	// compute largest eigenvalue and finally the FTLE value
	float Lmax = LambdaMax(P);
	//float ftle = 1 / std::abs(t1 - t0) * log(sqrt(Lmax));
	float dt = abs(currTime - spawnTime);
	float ftle = 0;

	if (dt > 0)
		ftle = 1.0 / dt * log(sqrt(Lmax));

	ftle *= scale;

	//ftle = currTime;

	y = height - 1 - y;

	// get a pointer to the pixel at (x,y)
	const int components = 1;
	float* pixel = (float*)(surface + y*pitch) + components * x;

	pixel[0] = ftle;
	//pixel[1] = ftle;
	//pixel[2] = ftle;
	//pixel[3] = 1;

	//// populate it
	//float value_x = 0.5f + 0.5f*cos(1.0 + 10.0f*((2.0f*x) / width - 1.0f));
	//float value_y = 0.5f + 0.5f*cos(1.0 + 10.0f*((2.0f*y) / height - 1.0f));
	//pixel[0] = 0.5*pixel[0] + 0.5*pow(value_x, 3.0f); // red
	//pixel[1] = 0.5*pixel[1] + 0.5*pow(value_y, 3.0f); // green
	//pixel[2] = 0.5f + 0.5f*cos(1.0); // blue
	//pixel[3] = 1; // alpha
}
#pragma endregion


using namespace tum3D;


RenderingManager::RenderingManager()
	: m_isCreated(false), m_pDevice(nullptr)
	, m_pRandomColorsTex(nullptr), m_pRandomColorsSRV(nullptr)
	, m_pOpaqueTex(nullptr), m_pOpaqueSRV(nullptr), m_pOpaqueRTV(nullptr)
	, m_pTransparentTex(nullptr), m_pTransparentSRV(nullptr), m_pTransparentRTV(nullptr)
	, m_pDepthTex(nullptr), m_pDepthDSV(nullptr), m_pDepthSRV(nullptr)
	, m_pScreenEffect(nullptr), m_pQuadEffect(nullptr)
{
}

RenderingManager::~RenderingManager()
{
	assert(!m_isCreated);
}

bool RenderingManager::Create(ID3D11Device* pDevice)
{
	std::cout << "Creating RenderingManager..." << std::endl;

	m_pDevice = pDevice;

	if(!CreateScreenDependentResources())
	{
		Release();
		return false;
	}

	if(FAILED(m_box.Create(m_pDevice)))
	{
		Release();
		return false;
	}

	if(FAILED(m_lineEffect.Create(m_pDevice)))
	{
		Release();
		return false;
	}

	m_pScreenEffect = new ScreenEffect();
	if (FAILED(m_pScreenEffect->Create(m_pDevice)))
	{
		Release();
		return false;
	}

	m_pQuadEffect = new QuadEffect();
	if (FAILED(m_pQuadEffect->Create(m_pDevice)))
	{
		Release();
		return false;
	}

	m_isCreated = true;

	std::cout << "RenderingManager created." << std::endl;

	return true;
}

void RenderingManager::Release()
{
	ReleaseScreenDependentResources();

	m_box.Release();

	m_lineEffect.SafeRelease();

	m_pDevice = nullptr;

	m_isCreated = false;

	if (m_pScreenEffect) {
		m_pScreenEffect->SafeRelease();
		m_pScreenEffect = nullptr;
	}
	if (m_pQuadEffect) {
		m_pQuadEffect->SafeRelease();
		m_pQuadEffect = nullptr;
	}
}

void RenderingManager::SetProjectionParams(const ProjectionParams& params, const Range1D& range)
{
	if(params == m_projectionParams && range == m_range) return;

	bool recreateScreenResources =
		(params.GetImageHeight(range) != m_projectionParams.GetImageHeight(m_range) ||
		 params.GetImageWidth (range) != m_projectionParams.GetImageWidth (m_range));

	m_projectionParams = params;
	m_range = range;

	if(recreateScreenResources)
		CreateScreenDependentResources();
}

#ifdef WriteVolumeToFileStuff
bool RenderingManager::WriteCurTimestepToRaws(TimeVolume& volume, const std::vector<std::string>& filenames)
{
	//TODO do this in slabs/slices of bricks, so that mem usage won't be quite so astronomical...
	int channelCount = volume.GetChannelCount();

	if(filenames.size() < channelCount)
	{
		printf("FAIL!\n");
		return false;
	}

	std::vector<FILE*> files;
	for(int c = 0; c < channelCount; c++)
	{
		FILE* file = fopen(filenames[c].c_str(), "wb");
		if(!file)
		{
			printf("WriteCurTimestepToRaw: Failed creating output file %s\n", filenames[c].c_str());
			//TODO close previous files
			return false;
		}
		files.push_back(file);
	}

	CancelRendering();

	m_pVolume = &volume;

	// save mem usage limit, and set to something reasonably small
	float memUsageLimitOld = volume.GetSystemMemoryUsage().GetSystemMemoryLimitMBytes();
	volume.GetSystemMemoryUsage().SetSystemMemoryLimitMBytes(1024.0f);
	volume.UnloadLRUBricks();

	CreateVolumeDependentResources();

	int brickSizeWith = volume.GetBrickSizeWithOverlap();
	int brickSizeWithout = volume.GetBrickSizeWithoutOverlap();
	int brickOverlap = volume.GetBrickOverlap();
	Vec3i volumeSize = volume.GetVolumeSize();
	std::vector<float> brickChannelData(brickSizeWith * brickSizeWith * brickSizeWith);
	std::vector<std::vector<float>> volumeData;
	for(int c = 0; c < channelCount; c++)
	{
		volumeData.push_back(std::vector<float>(volumeSize.x() * volumeSize.y() * brickSizeWithout));
	}

	volume.DiscardAllIO();
	auto& bricks = volume.GetNearestTimestep().bricks;
	// HACK: assume bricks are already in the correct order (z-major)
	for(size_t i = 0; i < bricks.size(); i++)
	{
		TimeVolumeIO::Brick& brick = bricks[i];

		// load from disk if required
		if(!brick.IsLoaded())
		{
			volume.EnqueueLoadBrickData(brick);
			volume.WaitForAllIO();
			volume.UnloadLRUBricks();
		}

		// decompress
		UploadBrick(m_pCompressShared, m_pCompressVolume, m_pVolume->GetInfo(), brick, m_dpChannelBuffer.data());
		for(int c = 0; c < channelCount; c++)
		{
			// download into CPU memory
			cudaSafeCall(cudaMemcpy(brickChannelData.data(), m_dpChannelBuffer[c], brickSizeWith * brickSizeWith * brickSizeWith * sizeof(float), cudaMemcpyDeviceToHost));
			// copy into whole volume array
			Vec3i offset = brick.GetSpatialIndex() * brickSizeWithout - brickOverlap;
			offset.z() = -brickOverlap;
			Vec3i thisBrickSize = Vec3i(brick.GetSize());
			for(int z = brickOverlap; z < thisBrickSize.z() - brickOverlap; z++)
			{
				for(int y = brickOverlap; y < thisBrickSize.y() - brickOverlap; y++)
				{
					for(int x = brickOverlap; x < thisBrickSize.x() - brickOverlap; x++)
					{
						int in = x + thisBrickSize.x() * (y + thisBrickSize.y() * z);
						int out = (offset.x()+x) + volumeSize.x() * ((offset.y()+y) + volumeSize.y() * (offset.z()+z));
						volumeData[c][out] = brickChannelData[in];
					}
				}
			}
		}

		printf("Brick %i / %i done\n", int(i+1), int(bricks.size()));

		// if this was the last brick of this slab, write it out!
		if(i+1 >= bricks.size() || bricks[i+1].GetSpatialIndex().z() != bricks[i].GetSpatialIndex().z())
		{
			int z0 = brick.GetSpatialIndex().z() * brickSizeWithout;
			int slices = brick.GetSize().z() - 2 * brickOverlap;
			printf("Writing slices %i-%i / %i...", z0 + 1, z0 + slices, volumeSize.z());
			for(int c = 0; c < channelCount; c++)
			{
				int elemsPerSlice = volumeSize.x() * volumeSize.y();
				for(int z = 0; z < slices; z++)
				{
					fwrite(volumeData[c].data() + z * elemsPerSlice, sizeof(float), elemsPerSlice, files[c]);
				}
			}
			printf("done\n");
		}
	}

	ReleaseVolumeDependentResources();

	m_pVolume = nullptr;

	for(int c = 0; c < channelCount; c++)
	{
		fclose(files[c]);
	}

	// restore mem usage limit
	volume.GetSystemMemoryUsage().SetSystemMemoryLimitMBytes(memUsageLimitOld);

	return true;
}


bool RenderingManager::WriteCurTimestepToLA3Ds(TimeVolume& volume, const std::vector<std::string>& filenames)
{
	int channelCount = volume.GetChannelCount();

	if(filenames.size() < channelCount)
	{
		printf("FAIL!\n");
		return false;
	}

	std::vector<LA3D::LargeArray3D<float>> files(channelCount);
	Vec3i volumeSize = volume.GetVolumeSize();
	for(int c = 0; c < channelCount; c++)
	{
		std::wstring filenamew(filenames[c].begin(), filenames[c].end());
		if(!files[c].Create(volumeSize.x(), volumeSize.y(), volumeSize.z(), 64, 64, 64, filenamew.c_str(), 1024 * 1024 * 1024))
		{
			printf("WriteCurTimestepToLA3Ds: Failed creating output file %s\n", filenames[c].c_str());
			//TODO close previous files
			return false;
		}
	}

	CancelRendering();

	m_pVolume = &volume;

	CreateVolumeDependentResources();

	int brickSizeWith = volume.GetBrickSizeWithOverlap();
	int brickSizeWithout = volume.GetBrickSizeWithoutOverlap();
	int brickOverlap = volume.GetBrickOverlap();
	std::vector<float> brickChannelData(brickSizeWith * brickSizeWith * brickSizeWith);

	volume.DiscardAllIO();
	auto& bricks = volume.GetNearestTimestep().bricks;
	for(size_t i = 0; i < bricks.size(); i++)
	{
		TimeVolumeIO::Brick& brick = bricks[i];
		// load from disk if required
		if(!brick.IsLoaded())
		{
			volume.EnqueueLoadBrickData(brick);
			volume.WaitForAllIO();
		}
		// decompress
		UploadBrick(m_pCompressShared, m_pCompressVolume, m_pVolume->GetInfo(), brick, m_dpChannelBuffer.data());
		for(int c = 0; c < channelCount; c++)
		{
			// download into CPU memory
			cudaSafeCall(cudaMemcpy(brickChannelData.data(), m_dpChannelBuffer[c], brickSizeWith * brickSizeWith * brickSizeWith * sizeof(float), cudaMemcpyDeviceToHost));
			// copy into la3d
			Vec3i target = brick.GetSpatialIndex() * brickSizeWithout;
			Vec3i thisBrickSize = Vec3i(brick.GetSize());
			Vec3i copySize = thisBrickSize - 2 * brickOverlap;
			size_t offset = brickOverlap + thisBrickSize.x() * (brickOverlap + thisBrickSize.y() * brickOverlap);
			files[c].CopyFrom(brickChannelData.data() + offset, target.x(), target.y(), target.z(), copySize.x(), copySize.y(), copySize.z(), thisBrickSize.x(), thisBrickSize.y());
		}

		printf("%i / %i\n", int(i+1), int(bricks.size()));
	}

	ReleaseVolumeDependentResources();

	m_pVolume = nullptr;

	for(int c = 0; c < channelCount; c++)
	{
		files[c].Close();
	}

	return true;
}
#endif

bool RenderingManager::CreateScreenDependentResources()
{
	ReleaseScreenDependentResources();

	uint width  = m_projectionParams.GetImageWidth (m_range);
	uint height = m_projectionParams.GetImageHeight(m_range);

	if(width == 0 || height == 0)
		return true;

	if(m_pDevice)
	{
		HRESULT hr;

		D3D11_TEXTURE1D_DESC desc1;
		desc1.ArraySize = 1;
		desc1.BindFlags = D3D11_BIND_SHADER_RESOURCE;
		desc1.CPUAccessFlags = 0;
		desc1.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
		desc1.MipLevels = 1;
		desc1.MiscFlags = 0;
		desc1.Usage = D3D11_USAGE_DEFAULT;
		desc1.Width = 1024;

		std::vector<byte> colors(desc1.Width * 4);
		srand(0);
		for(uint i = 0; i < desc1.Width; i++) {
			colors[4*i+0] = rand() % 256;
			colors[4*i+1] = rand() % 256;
			colors[4*i+2] = rand() % 256;
			colors[4*i+3] = 255;
		}

		// HACK: set first few colors to fixed "primary" colors, for loaded lines
		colors[0] = 0;
		colors[1] = 0;
		colors[2] = 255;
		colors[3] = 255;

		colors[4] = 255;
		colors[5] = 0;
		colors[6] = 0;
		colors[7] = 255;

		colors[8] = 255;
		colors[9] = 255;
		colors[10] = 0;
		colors[11] = 255;

		D3D11_SUBRESOURCE_DATA initData = {};
		initData.pSysMem = colors.data();
		hr = m_pDevice->CreateTexture1D(&desc1, &initData, &m_pRandomColorsTex);
		if (FAILED(hr)) return false;
		hr = m_pDevice->CreateShaderResourceView(m_pRandomColorsTex, nullptr, &m_pRandomColorsSRV);
		if (FAILED(hr)) return false;

		// create texture/rendertarget for opaque objects
		D3D11_TEXTURE2D_DESC desc;
		desc.ArraySize = 1;
		desc.BindFlags = D3D11_BIND_RENDER_TARGET | D3D11_BIND_SHADER_RESOURCE;
		desc.CPUAccessFlags = 0;
		desc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
		desc.MipLevels = 1;
		desc.MiscFlags = 0;
		desc.SampleDesc.Count = 1;
		desc.SampleDesc.Quality = 0;
		desc.Usage = D3D11_USAGE_DEFAULT;
		desc.Width = width;
		desc.Height = height;
		hr = m_pDevice->CreateTexture2D(&desc, nullptr, &m_pOpaqueTex);
		if (FAILED(hr)) return false;
		hr = m_pDevice->CreateShaderResourceView(m_pOpaqueTex, nullptr, &m_pOpaqueSRV);
		if (FAILED(hr)) return false;
		hr = m_pDevice->CreateRenderTargetView(m_pOpaqueTex, nullptr, &m_pOpaqueRTV);
		if (FAILED(hr)) return false;

		// create texture for transparent particle rendering
		desc.Format = DXGI_FORMAT_R16G16B16A16_FLOAT;
		hr = m_pDevice->CreateTexture2D(&desc, nullptr, &m_pTransparentTex);
		if (FAILED(hr)) return false;
		hr = m_pDevice->CreateShaderResourceView(m_pTransparentTex, nullptr, &m_pTransparentSRV);
		if (FAILED(hr)) return false;
		hr = m_pDevice->CreateRenderTargetView(m_pTransparentTex, nullptr, &m_pTransparentRTV);
		if (FAILED(hr)) return false;

		// create depth buffer
		desc.BindFlags = D3D11_BIND_DEPTH_STENCIL | D3D11_BIND_SHADER_RESOURCE;
		desc.Format = DXGI_FORMAT_R32_TYPELESS;
		hr = m_pDevice->CreateTexture2D(&desc, nullptr, &m_pDepthTex);
		if (FAILED(hr)) return false;
		D3D11_DEPTH_STENCIL_VIEW_DESC dsvDesc = {};
		dsvDesc.Flags = 0;
		dsvDesc.Format = DXGI_FORMAT_D32_FLOAT;
		dsvDesc.ViewDimension = D3D11_DSV_DIMENSION_TEXTURE2D;
		dsvDesc.Texture2D.MipSlice = 0;
		hr = m_pDevice->CreateDepthStencilView(m_pDepthTex, &dsvDesc, &m_pDepthDSV);
		if (FAILED(hr)) return false;
		D3D11_SHADER_RESOURCE_VIEW_DESC srvDesc = {};
		srvDesc.Format = DXGI_FORMAT_R32_FLOAT;
		srvDesc.ViewDimension = D3D11_SRV_DIMENSION_TEXTURE2D;
		srvDesc.Texture2D.MipLevels = 1;
		hr = m_pDevice->CreateShaderResourceView(m_pDepthTex, &srvDesc, &m_pDepthSRV);
		if (FAILED(hr)) return false;
	}

	return true;
}

void RenderingManager::ReleaseScreenDependentResources()
{
	if(m_pDepthDSV)
	{
		m_pDepthDSV->Release();
		m_pDepthDSV = nullptr;
	}

	if (m_pDepthSRV)
	{
		m_pDepthSRV->Release();
		m_pDepthSRV = nullptr;
	}

	if(m_pDepthTex)
	{
		m_pDepthTex->Release();
		m_pDepthTex = nullptr;
	}

	if(m_pOpaqueRTV)
	{
		m_pOpaqueRTV->Release();
		m_pOpaqueRTV = nullptr;
	}

	if(m_pOpaqueSRV)
	{
		m_pOpaqueSRV->Release();
		m_pOpaqueSRV = nullptr;
	}

	if(m_pOpaqueTex)
	{
		m_pOpaqueTex->Release();
		m_pOpaqueTex = nullptr;
	}

	if(m_pRandomColorsSRV)
	{
		m_pRandomColorsSRV->Release();
		m_pRandomColorsSRV = nullptr;
	}

	if(m_pRandomColorsTex)
	{
		m_pRandomColorsTex->Release();
		m_pRandomColorsTex = nullptr;
	}
	if (m_pTransparentRTV)
	{
		m_pTransparentRTV->Release();
		m_pTransparentRTV = nullptr;
	}

	if (m_pTransparentSRV)
	{
		m_pTransparentSRV->Release();
		m_pTransparentSRV = nullptr;
	}

	if (m_pTransparentTex)
	{
		m_pTransparentTex->Release();
		m_pTransparentTex = nullptr;
	}
}

RenderingManager::eRenderState RenderingManager::Render(
	bool isTracing, 
	const TimeVolume& volume, 
	const RenderingParameters& renderingParams,
	const ViewParams& viewParams, 
	const StereoParams& stereoParams,
	const ParticleTraceParams& particleTraceParams, 
	const ParticleRenderParams& particleRenderParams,
	const RaycastParams& raycastParams,
	const std::vector<LineBuffers*>& pLineBuffers, 
	const std::vector<BallBuffers*>& pBallBuffers, 
	float ballRadius,
	HeatMapManager* pHeatMapManager,
	cudaArray* pTransferFunction, 
	SimpleParticleVertexDeltaT* dpParticles, 
	int transferFunctionDevice)
{
	if (!volume.IsOpen()) 
		return STATE_ERROR;

	if (m_projectionParams.GetImageWidth(m_range) * m_projectionParams.GetImageHeight(m_range) == 0)
		return STATE_DONE;

	m_renderingParams = renderingParams;
	m_viewParams = viewParams;
	m_stereoParams = stereoParams;

	m_particleTraceParams = particleTraceParams;
	m_particleRenderParams = particleRenderParams;

#pragma region ComputeFrustumPlanes
	// compute frustum planes in view space
	Vec4f frustumPlanes[6];
	Vec4f frustumPlanes2[6];
	if (m_stereoParams.m_stereoEnabled)
	{
		m_projectionParams.GetFrustumPlanes(frustumPlanes, EYE_LEFT, m_stereoParams.m_eyeDistance, m_range);
		m_projectionParams.GetFrustumPlanes(frustumPlanes2, EYE_RIGHT, m_stereoParams.m_eyeDistance, m_range);

		// we want the inverse transpose of the inverse of view
		Mat4f viewLeft = m_viewParams.BuildViewMatrix(EYE_LEFT, m_stereoParams.m_eyeDistance);
		Mat4f viewLeftTrans;
		viewLeft.transpose(viewLeftTrans);
		Mat4f viewRight = m_viewParams.BuildViewMatrix(EYE_RIGHT, m_stereoParams.m_eyeDistance);
		Mat4f viewRightTrans;
		viewRight.transpose(viewRightTrans);

		for (uint i = 0; i < 6; i++)
		{
			frustumPlanes[i] = viewLeftTrans  * frustumPlanes[i];
			frustumPlanes2[i] = viewRightTrans * frustumPlanes[i];
		}
	}
	else
	{
		m_projectionParams.GetFrustumPlanes(frustumPlanes, EYE_CYCLOP, 0.0f, m_range);

		// we want the inverse transpose of the inverse of view
		Mat4f view = m_viewParams.BuildViewMatrix(EYE_CYCLOP, 0.0f);
		Mat4f viewTrans;
		view.transpose(viewTrans);

		for (uint i = 0; i < 6; i++)
		{
			frustumPlanes[i] = viewTrans * frustumPlanes[i];
		}
	}
#pragma endregion
	
	// clear rendertargets and depth buffer
	ClearResult();

	// render opaque stuff immediately
	{
		RenderBoxes(volume, raycastParams, true, false);
		bool linesRendered = false;
		if (m_particleRenderParams.m_linesEnabled)
		{
			for (size_t i = 0; i < pLineBuffers.size(); i++)
			{
				RenderLines(volume, pLineBuffers[i], true, false);
				if (pLineBuffers[i]->m_indexCountTotal >= 0) linesRendered = true;
			}
		}

		if (!linesRendered && m_particleRenderParams.m_showSlice && m_particleRenderParams.m_pSliceTexture != nullptr)
		{
			PrepareRenderSlice(m_particleRenderParams.m_pSliceTexture, m_particleRenderParams.m_sliceAlpha, m_particleRenderParams.m_slicePosition, volume.GetVolumeHalfSizeWorld() * 2, tum3D::Vec2f(0, 0));
			ExtraRenderSlice();
		}

		if (m_particleTraceParams.m_ftleEnabled)
		{
			if (!m_ftleTexture.IsTextureCreated() || m_particleTraceParams.m_ftleResolution != m_ftleTexture.width)
				CreateFTLETexture();

			if (isTracing)
				ComputeFTLE(volume, dpParticles);

			if (m_particleRenderParams.m_ftleShowTexture)
			{
				tum3D::Vec2f center(m_particleTraceParams.m_seedBoxMin.x() + m_particleTraceParams.m_seedBoxSize.x() * 0.5f, m_particleTraceParams.m_seedBoxMin.y() + m_particleTraceParams.m_seedBoxSize.y() * 0.5f);
				PrepareRenderSlice(m_ftleTexture.pSRView, m_particleRenderParams.m_ftleTextureAlpha, m_particleTraceParams.m_ftleSliceY, m_particleTraceParams.m_seedBoxSize, center);
				ExtraRenderSlice();
			}
		}

		for (size_t i = 0; i < pBallBuffers.size(); i++)
			RenderBalls(volume, pBallBuffers[i], ballRadius);

		//render heat map directly
		if (pHeatMapManager != nullptr) 
			RenderHeatMap(pHeatMapManager);

		//Don't do it here. It is now in RenderLines, so that the particles are drawn correctly including the transparency
		//RenderSliceTexture();
	}

	return STATE_DONE;
}


RenderingManager::eRenderState RenderingManager::Render(
	std::vector<FlowVisToolVolumeData*> volumes,
	const RenderingParameters& renderingParams,
	const ViewParams& viewParams,
	const StereoParams& stereoParams,
	const RaycastParams& raycastParams,
	const std::vector<LineBuffers*>& pLineBuffers, // Lines loaded from disk
	const std::vector<BallBuffers*>& pBallBuffers,
	float ballRadius,
	HeatMapManager* pHeatMapManager,
	cudaArray* pTransferFunction,
	int transferFunctionDevice)
{
	if (m_projectionParams.GetImageWidth(m_range) * m_projectionParams.GetImageHeight(m_range) == 0)
		return STATE_DONE;

	m_renderingParams = renderingParams;
	m_viewParams = viewParams;
	m_stereoParams = stereoParams;

#pragma region ComputeFrustumPlanes
	// compute frustum planes in view space
	Vec4f frustumPlanes[6];
	Vec4f frustumPlanes2[6];
	if (m_stereoParams.m_stereoEnabled)
	{
		m_projectionParams.GetFrustumPlanes(frustumPlanes, EYE_LEFT, m_stereoParams.m_eyeDistance, m_range);
		m_projectionParams.GetFrustumPlanes(frustumPlanes2, EYE_RIGHT, m_stereoParams.m_eyeDistance, m_range);

		// we want the inverse transpose of the inverse of view
		Mat4f viewLeft = m_viewParams.BuildViewMatrix(EYE_LEFT, m_stereoParams.m_eyeDistance);
		Mat4f viewLeftTrans;
		viewLeft.transpose(viewLeftTrans);
		Mat4f viewRight = m_viewParams.BuildViewMatrix(EYE_RIGHT, m_stereoParams.m_eyeDistance);
		Mat4f viewRightTrans;
		viewRight.transpose(viewRightTrans);

		for (uint i = 0; i < 6; i++)
		{
			frustumPlanes[i] = viewLeftTrans  * frustumPlanes[i];
			frustumPlanes2[i] = viewRightTrans * frustumPlanes[i];
		}
	}
	else
	{
		m_projectionParams.GetFrustumPlanes(frustumPlanes, EYE_CYCLOP, 0.0f, m_range);

		// we want the inverse transpose of the inverse of view
		Mat4f view = m_viewParams.BuildViewMatrix(EYE_CYCLOP, 0.0f);
		Mat4f viewTrans;
		view.transpose(viewTrans);

		for (uint i = 0; i < 6; i++)
		{
			frustumPlanes[i] = viewTrans * frustumPlanes[i];
		}
	}
#pragma endregion

	// clear rendertargets and depth buffer
	ClearResult();

	bool linesRendered = false;

	// Render lines loaded from disk.
	// FIXME: should either expose rendering settings for these lines on the GUI or have default settings for them.
	for (size_t i = 0; i < pLineBuffers.size(); i++)
	{
		RenderLines(*volumes[i]->m_volume, pLineBuffers[i], true, false);
		if (pLineBuffers[i]->m_indexCountTotal >= 0)
			linesRendered = true;
	}

	// Sort volumes so that transparent objects (particles) are drawn last.
	std::sort(volumes.begin(), volumes.end(), [](FlowVisToolVolumeData* a, FlowVisToolVolumeData* b) 
	{
		return a->m_renderParams.m_lineRenderMode < b->m_renderParams.m_lineRenderMode;
	});

	// Render boxes for each dataset (opaque).
	for (size_t i = 0; i < volumes.size(); i++)
	{
		if (!volumes[i]->m_volume->IsOpen())
			return STATE_ERROR;

		m_renderDomainBox = volumes[i]->m_renderDomainBox;
		m_renderClipBox = volumes[i]->m_renderClipBox;
		m_renderSeedBox = volumes[i]->m_renderSeedBox;
		m_renderBrickBoxes = volumes[i]->m_renderBrickBoxes;

		m_particleTraceParams = volumes[i]->m_traceParams;
		m_particleRenderParams = volumes[i]->m_renderParams;

		RenderBoxes(*volumes[i]->m_volume, raycastParams, true, false);
	}

	// Render line for each dataset. The array should be sorted so that opaque line are drawn first.
	for (size_t i = 0; i < volumes.size(); i++)
	{
		if (!volumes[i]->m_volume->IsOpen())
			return STATE_ERROR;

		m_particleTraceParams =		volumes[i]->m_traceParams;
		m_particleRenderParams =	volumes[i]->m_renderParams;
			
		if (m_particleRenderParams.m_linesEnabled)
		{
			LineBuffers* pTracedLines = volumes[i]->m_tracingManager.GetResult().get();
			if (pTracedLines != nullptr)
			{
				RenderLines(*volumes[i]->m_volume, pTracedLines, true, false);
				if (pTracedLines->m_indexCountTotal >= 0)
					linesRendered = true;
			}
		}
	}
		
	//if (!linesRendered && m_particleRenderParams.m_showSlice && m_particleRenderParams.m_pSliceTexture != nullptr)
	//{
	//	PrepareRenderSlice(m_particleRenderParams.m_pSliceTexture, m_particleRenderParams.m_sliceAlpha, m_particleRenderParams.m_slicePosition, volumes[i].m_volume->GetVolumeHalfSizeWorld() * 2, tum3D::Vec2f(0, 0));
	//	ExtraRenderSlice();
	//}

	//if (m_particleTraceParams.m_ftleEnabled)
	//{
	//	if (!m_ftleTexture.IsTextureCreated() || m_particleTraceParams.m_ftleResolution != m_ftleTexture.width)
	//		CreateFTLETexture();

	//	if (isTracing)
	//		ComputeFTLE(volume, dpParticles); // g_tracingManager.m_dpParticles

	//	if (m_particleRenderParams.m_ftleShowTexture)
	//	{
	//		tum3D::Vec2f center(m_particleTraceParams.m_seedBoxMin.x() + m_particleTraceParams.m_seedBoxSize.x() * 0.5f, m_particleTraceParams.m_seedBoxMin.y() + m_particleTraceParams.m_seedBoxSize.y() * 0.5f);
	//		PrepareRenderSlice(m_ftleTexture.pSRView, m_particleRenderParams.m_ftleTextureAlpha, m_particleTraceParams.m_ftleSliceY, m_particleTraceParams.m_seedBoxSize, center);
	//		ExtraRenderSlice();
	//	}
	//}

	//for (size_t i = 0; i < pBallBuffers.size(); i++)
	//	RenderBalls(volume, pBallBuffers[i], ballRadius);

	////render heat map directly
	//if (pHeatMapManager != nullptr)
	//	RenderHeatMap(pHeatMapManager);

	//Don't do it here. It is now in RenderLines, so that the particles are drawn correctly including the transparency
	//RenderSliceTexture();


	return STATE_DONE;
}

void RenderingManager::ClearResult()
{
	if(m_pDevice)
	{
		ID3D11DeviceContext* pContext = nullptr;
		m_pDevice->GetImmediateContext(&pContext);

		// clear rendertargets and depth buffer
		float black[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
		if(m_pOpaqueRTV)
			pContext->ClearRenderTargetView(m_pOpaqueRTV, black);

		if(m_pDepthDSV)
			pContext->ClearDepthStencilView(m_pDepthDSV, D3D11_CLEAR_DEPTH, 1.0f, 0);

		pContext->Release();
	}
}

void RenderingManager::RenderBoxes(const TimeVolume& vol, const RaycastParams& raycastParams, bool enableColor, bool blendBehind)
{
	if(!m_pDevice) return;

	ID3D11DeviceContext* pContext = nullptr;
	m_pDevice->GetImmediateContext(&pContext);

	Vec4f domainBoxColor(1.0f, 1.0f, 1.0f, 1.0f);
	Vec4f brickBoxColor (0.0f, 0.5f, 1.0f, 1.0f);
	Vec4f clipBoxColor  (1.0f, 0.0f, 0.0f, 1.0f);
	Vec4f seedBoxColor  (0.0f, 0.6f, 0.0f, 1.0f);
	Vec4f coordinateBoxColor(1.0f, 0.0f, 0.0f, 1.0f);


	Vec3f lightPos;

	if (m_renderingParams.m_FixedLightDir)
		lightPos = normalize(m_renderingParams.m_lightDir);
	else
		lightPos = m_viewParams.GetCameraPosition();

	Vec3f volumeHalfSizeWorld = vol.GetVolumeHalfSizeWorld();
	Vec3f seedBoxMin = m_particleTraceParams.m_seedBoxMin;
	Vec3f seedBoxMax = m_particleTraceParams.m_seedBoxMin + m_particleTraceParams.m_seedBoxSize;
	Vec3f clipBoxMin = raycastParams.m_clipBoxMin;
	Vec3f clipBoxMax = raycastParams.m_clipBoxMax;

	// save viewports and render targets
	uint oldViewportCount = D3D11_VIEWPORT_AND_SCISSORRECT_OBJECT_COUNT_PER_PIPELINE;
	D3D11_VIEWPORT oldViewports[D3D11_VIEWPORT_AND_SCISSORRECT_OBJECT_COUNT_PER_PIPELINE];
	pContext->RSGetViewports(&oldViewportCount, oldViewports);
	ID3D11RenderTargetView* ppOldRTVs[D3D11_SIMULTANEOUS_RENDER_TARGET_COUNT];
	ID3D11DepthStencilView* pOldDSV;
	pContext->OMGetRenderTargets(D3D11_SIMULTANEOUS_RENDER_TARGET_COUNT, ppOldRTVs, &pOldDSV);

	// set our render target
	// if color is disabled, don't set a render target
	pContext->OMSetRenderTargets(enableColor ? 1 : 0, &m_pOpaqueRTV, m_pDepthDSV);

	// build viewport
	D3D11_VIEWPORT viewport = {};
	viewport.TopLeftX = float(0);
	viewport.TopLeftY = float(0);
	viewport.Width    = float(m_projectionParams.GetImageWidth(m_range));
	viewport.Height   = float(m_projectionParams.m_imageHeight);
	viewport.MinDepth = 0.0f;
	viewport.MaxDepth = 1.0f;

	bool renderDomainBox = m_renderDomainBox;
	bool renderClipBox = m_renderClipBox && raycastParams.m_raycastingEnabled;
	bool renderSeedBox = m_renderSeedBox && m_particleRenderParams.m_linesEnabled;
	bool renderBrickBoxes = m_renderBrickBoxes;
	//bool renderCoordinates = m_renderDomainBox;
	bool renderCoordinates = false;

	Vec3f brickSize = vol.GetBrickSizeWorld();
	float tubeRadiusLarge  = 0.004f;
	float tubeRadiusMedium = 0.003f;
	float tubeRadiusSmall  = 0.002f;
	if(m_stereoParams.m_stereoEnabled)
	{
		Mat4f viewLeft  = m_viewParams.BuildViewMatrix(EYE_LEFT,  m_stereoParams.m_eyeDistance);
		Mat4f viewRight = m_viewParams.BuildViewMatrix(EYE_RIGHT, m_stereoParams.m_eyeDistance);
		Mat4f projLeft  = m_projectionParams.BuildProjectionMatrix(EYE_LEFT,  m_stereoParams.m_eyeDistance, m_range);
		Mat4f projRight = m_projectionParams.BuildProjectionMatrix(EYE_RIGHT, m_stereoParams.m_eyeDistance, m_range);

		viewport.Height /= 2.0f;
		pContext->RSSetViewports(1, &viewport);

		if(renderDomainBox)	m_box.RenderLines(viewLeft, projLeft, lightPos, -volumeHalfSizeWorld, volumeHalfSizeWorld, domainBoxColor, m_DomainBoxThickness,  blendBehind);
		if(renderClipBox)	m_box.RenderLines(viewLeft, projLeft, lightPos, clipBoxMin,           clipBoxMax,          clipBoxColor,   tubeRadiusMedium, blendBehind);
		if(renderSeedBox)	m_box.RenderLines(viewLeft, projLeft, lightPos, seedBoxMin,           seedBoxMax,          seedBoxColor,   tubeRadiusMedium, blendBehind);
		if (renderCoordinates) {
			m_box.RenderLines(viewLeft, projLeft, lightPos, -volumeHalfSizeWorld - 2 * tubeRadiusLarge, -volumeHalfSizeWorld + 2 * tubeRadiusLarge, coordinateBoxColor, tubeRadiusLarge, blendBehind);
			m_box.RenderLines(viewLeft, projLeft, lightPos, -volumeHalfSizeWorld, -volumeHalfSizeWorld + Vec3f(volumeHalfSizeWorld.x() / 4, 0.0f, 0.0f), coordinateBoxColor, tubeRadiusLarge*1.5f, blendBehind);
		}
		if(renderBrickBoxes) m_box.RenderBrickLines(viewLeft, projLeft, lightPos, -volumeHalfSizeWorld, volumeHalfSizeWorld, brickBoxColor, brickSize, tubeRadiusSmall, blendBehind);

		viewport.TopLeftY += viewport.Height;
		pContext->RSSetViewports(1, &viewport);

		if (renderDomainBox)	m_box.RenderLines(viewRight, projRight, lightPos, -volumeHalfSizeWorld, volumeHalfSizeWorld, domainBoxColor, m_DomainBoxThickness, blendBehind);
		if(renderClipBox)	m_box.RenderLines(viewRight, projRight, lightPos, clipBoxMin,           clipBoxMax,          clipBoxColor,   tubeRadiusMedium, blendBehind);
		if(renderSeedBox)	m_box.RenderLines(viewRight, projRight, lightPos, seedBoxMin,           seedBoxMax,          seedBoxColor,   tubeRadiusMedium, blendBehind);
		if (renderCoordinates) {
			m_box.RenderLines(viewRight, projRight, lightPos, -volumeHalfSizeWorld - 2 * tubeRadiusLarge, -volumeHalfSizeWorld + 2 * tubeRadiusLarge, coordinateBoxColor, tubeRadiusLarge, blendBehind);
			m_box.RenderLines(viewRight, projRight, lightPos, -volumeHalfSizeWorld, -volumeHalfSizeWorld + Vec3f(volumeHalfSizeWorld.x() / 4, 0.0f, 0.0f), coordinateBoxColor, tubeRadiusLarge*1.5f, blendBehind);
		}
		if(renderBrickBoxes) m_box.RenderBrickLines(viewRight, projRight, lightPos, -volumeHalfSizeWorld, volumeHalfSizeWorld, brickBoxColor, brickSize, tubeRadiusSmall, blendBehind);
	}
	else
	{
		Mat4f view = m_viewParams.BuildViewMatrix(EYE_CYCLOP, 0.0f);
		Mat4f proj = m_projectionParams.BuildProjectionMatrix(EYE_CYCLOP, 0.0f, m_range);

		pContext->RSSetViewports(1, &viewport);

		if (renderDomainBox)	m_box.RenderLines(view, proj, lightPos, -volumeHalfSizeWorld, volumeHalfSizeWorld, domainBoxColor, m_DomainBoxThickness, blendBehind);
		if(renderClipBox)	m_box.RenderLines(view, proj, lightPos, clipBoxMin,           clipBoxMax,          clipBoxColor,   tubeRadiusMedium, blendBehind);
		if(renderSeedBox)	m_box.RenderLines(view, proj, lightPos, seedBoxMin,           seedBoxMax,          seedBoxColor,   tubeRadiusMedium, blendBehind);
		if (renderCoordinates) {
			m_box.RenderLines(view, proj, lightPos, -volumeHalfSizeWorld - 2 * tubeRadiusLarge, -volumeHalfSizeWorld + 2 * tubeRadiusLarge, coordinateBoxColor, tubeRadiusLarge, blendBehind);
			m_box.RenderLines(view, proj, lightPos, -volumeHalfSizeWorld, -volumeHalfSizeWorld + Vec3f(volumeHalfSizeWorld.x() / 4, 0.0f, 0.0f), coordinateBoxColor, tubeRadiusLarge*1.5f, blendBehind);
		}
		if(renderBrickBoxes) m_box.RenderBrickLines(view, proj, lightPos, -volumeHalfSizeWorld, volumeHalfSizeWorld, brickBoxColor, brickSize, tubeRadiusSmall, blendBehind);
	}

	// restore viewports and render targets
	pContext->OMSetRenderTargets(D3D11_SIMULTANEOUS_RENDER_TARGET_COUNT, ppOldRTVs, pOldDSV);
	pContext->RSSetViewports(oldViewportCount, oldViewports);
	for(uint i = 0; i < D3D11_SIMULTANEOUS_RENDER_TARGET_COUNT; i++)
	{
		SAFE_RELEASE(ppOldRTVs[i]);
	}
	SAFE_RELEASE(pOldDSV);

	pContext->Release();
}

void DebugRenderLines(ID3D11Device* device, ID3D11DeviceContext* context, const LineBuffers* pLineBuffers)
{
	if (pLineBuffers->m_indexCountTotal == 0) {
		printf("no vertices to draw\n");
		return;
	}

	//Create staging buffers
	HRESULT hr;
	D3D11_BUFFER_DESC bufDesc = {};

	ID3D11Buffer* vbCopy = NULL;
	ID3D11Buffer* ibCopy = NULL;

	bufDesc.BindFlags = D3D11_BIND_UNORDERED_ACCESS; //D3D11_BIND_VERTEX_BUFFER;
	bufDesc.ByteWidth = pLineBuffers->m_lineCount * pLineBuffers->m_lineLengthMax * sizeof(LineVertex);
	bufDesc.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
	bufDesc.Usage = D3D11_USAGE_DEFAULT;
	if (FAILED(hr = device->CreateBuffer(&bufDesc, nullptr, &vbCopy)))
	{
		printf("unable to create vertex buffer for copying data to the cpu\n");
		SAFE_RELEASE(vbCopy);
		SAFE_RELEASE(ibCopy);
		return;
	}

	bufDesc.BindFlags = D3D11_BIND_UNORDERED_ACCESS; //D3D11_BIND_INDEX_BUFFER;
	bufDesc.ByteWidth = pLineBuffers->m_lineCount * (pLineBuffers->m_lineLengthMax - 1) * 2 * sizeof(uint);
	bufDesc.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
	bufDesc.Usage = D3D11_USAGE_DEFAULT;
	if (FAILED(hr = device->CreateBuffer(&bufDesc, nullptr, &ibCopy)))
	{
		printf("unable to create vertex buffer for copying data to the cpu\n");
		SAFE_RELEASE(vbCopy);
		SAFE_RELEASE(ibCopy);
		return;
	}

	context->CopyResource(vbCopy, pLineBuffers->m_pVB);
	context->CopyResource(ibCopy, pLineBuffers->m_pIB);

	//copy to cpu
	D3D11_MAPPED_SUBRESOURCE ms;
	hr = context->Map(vbCopy, 0, D3D11_MAP_READ, 0, &ms);
	std::vector<LineVertex> vertexBuffer;
	std::vector<uint> indexBuffer;
	bool mapped = true;
	if (FAILED(hr)) {
		printf("unable to map vertex buffer\n");
		mapped = false;
	}
	else {
		vertexBuffer.resize(pLineBuffers->m_lineCount * pLineBuffers->m_lineLengthMax);
		memcpy(&vertexBuffer[0], ms.pData, sizeof(LineVertex) * vertexBuffer.size());
		context->Unmap(vbCopy, 0);
	}
	hr = context->Map(ibCopy, 0, D3D11_MAP_READ, 0, &ms);
	if (FAILED(hr)) {
		printf("unable to map index buffer\n");
		mapped = false;
	}
	else {
		indexBuffer.resize(pLineBuffers->m_indexCountTotal);
		memcpy(&indexBuffer[0], ms.pData, sizeof(uint) * indexBuffer.size());
		context->Unmap(ibCopy, 0);
	}

	//print vertices
	if (mapped) {
		for (uint i = 0; i < pLineBuffers->m_indexCountTotal; ++i) {
			uint j = indexBuffer[i];
			const LineVertex& v = vertexBuffer[j];
			printf("%d->%d: pos=(%f, %f, %f), seed pos=(%f, %f, %f)\n", i, j, 
				v.Position.x, v.Position.y, v.Position.z, v.SeedPosition.x, v.SeedPosition.y, v.SeedPosition.z);
		}
	}

	//release staging resource
	SAFE_RELEASE(vbCopy);
	SAFE_RELEASE(ibCopy);
}

void RenderingManager::RenderLines(const TimeVolume& vol, LineBuffers* pLineBuffers, bool enableColor, bool blendBehind)
{
	if (pLineBuffers->m_indexCountTotal == 0) return;

	ID3D11DeviceContext* pContext = nullptr;
	m_pDevice->GetImmediateContext(&pContext);

	// build viewport
	D3D11_VIEWPORT viewport = {};
	viewport.TopLeftX = float(0);
	viewport.TopLeftY = float(0);
	viewport.Width    = float(m_projectionParams.GetImageWidth(m_range));
	viewport.Height   = float(m_projectionParams.m_imageHeight);
	viewport.MinDepth = 0.0f;
	viewport.MaxDepth = 1.0f;

	// Debug
	//DebugRenderLines(m_pDevice, pContext, pLineBuffers);


	Vec3f lightDir;

	if (m_renderingParams.m_FixedLightDir)
		lightDir = normalize(m_renderingParams.m_lightDir);
	else
		lightDir = m_viewParams.GetViewDir();

	//std::cout << "viewDir(" << lightDir.x() << "," << lightDir.y() << "," << lightDir.z() << ")" << std::endl;

	// common shader vars
	m_lineEffect.m_pvLightPosVariable->SetFloatVector(lightDir);
	m_lineEffect.m_pfRibbonHalfWidthVariable->SetFloat(0.01f * m_particleRenderParams.m_ribbonWidth * 0.5f);
	m_lineEffect.m_pfTubeRadiusVariable->SetFloat(0.01f * m_particleRenderParams.m_tubeRadius);
	m_lineEffect.m_pbTubeRadiusFromVelocityVariable->SetBool(m_particleRenderParams.m_tubeRadiusFromVelocity);
	m_lineEffect.m_pfReferenceVelocityVariable->SetFloat(m_particleRenderParams.m_referenceVelocity);
	m_lineEffect.m_pvHalfSizeWorldVariable->SetFloatVector(vol.GetVolumeHalfSizeWorld());

	m_lineEffect.m_piColorMode->SetInt(m_particleRenderParams.m_lineColorMode);
	m_lineEffect.m_pvColor0Variable->SetFloatVector(m_particleRenderParams.m_color0);
	m_lineEffect.m_pvColor1Variable->SetFloatVector(m_particleRenderParams.m_color1);
	float timeMin = (m_particleTraceParams.m_lineMode == LINE_STREAM) ? 0.0f : vol.GetCurTime();
	m_lineEffect.m_pfTimeMinVariable->SetFloat(timeMin);
	m_lineEffect.m_pfTimeMaxVariable->SetFloat(timeMin + m_particleTraceParams.m_lineAgeMax);

	m_lineEffect.m_pbTimeStripesVariable->SetBool(m_particleRenderParams.m_timeStripes);
	m_lineEffect.m_pfTimeStripeLengthVariable->SetFloat(m_particleRenderParams.m_timeStripeLength);

	m_lineEffect.m_pfParticleTransparencyVariable->SetFloat(m_particleRenderParams.m_particleTransparency);
	m_lineEffect.m_pfParticleSizeVariable->SetFloat(0.01f * m_particleRenderParams.m_particleSize);
	float aspectRatio = viewport.Width / viewport.Height;
	m_lineEffect.m_pfScreenAspectRatioVariable->SetFloat(aspectRatio);

	m_lineEffect.m_ptexColors->SetResource(m_pRandomColorsSRV);
	if (m_particleRenderParams.m_pColorTexture != nullptr) {
		m_lineEffect.m_pseedColors->SetResource(m_particleRenderParams.m_pColorTexture);
	}

	m_lineEffect.m_piMeasureMode->SetInt((int)m_particleRenderParams.m_measure);
	m_lineEffect.m_pfMeasureScale->SetFloat(m_particleRenderParams.m_measureScale);
	if (m_particleRenderParams.m_transferFunction.m_srv != nullptr) {
		m_lineEffect.m_ptransferFunction->SetResource(m_particleRenderParams.m_transferFunction.m_srv);
	}
	Vec2f tfRange(m_particleRenderParams.m_transferFunction.m_rangeMin, m_particleRenderParams.m_transferFunction.m_rangeMax);
	m_lineEffect.m_pvTfRange->SetFloatVector(tfRange);

	// common slice texture parameters
	bool renderSlice = m_particleRenderParams.m_showSlice && m_particleRenderParams.m_pSliceTexture != nullptr;

	tum3D::Vec4f clipPlane;
	if (renderSlice)
		clipPlane = PrepareRenderSlice(m_particleRenderParams.m_pSliceTexture, m_particleRenderParams.m_sliceAlpha, m_particleRenderParams.m_slicePosition, vol.GetVolumeHalfSizeWorld() * 2, tum3D::Vec2f(0, 0));

	//Check if particles should be rendered
	if (m_particleRenderParams.m_lineRenderMode == LINE_RENDER_PARTICLES) 
	{
		SortParticles(pLineBuffers, pContext);
		if (!blendBehind) {
			if (renderSlice) {
				//render particles below, and then the slice
				RenderParticles(pLineBuffers, pContext, viewport, &clipPlane, true);
				//invert clip plane and render particles above
				clipPlane = -clipPlane;
				RenderParticles(pLineBuffers, pContext, viewport, &clipPlane, false);
			}
			else {
				RenderParticles(pLineBuffers, pContext, viewport);
			}
		}
		pContext->Release();

		return;
	}

	// IA
	pContext->IASetInputLayout(m_lineEffect.m_pInputLayout);
	UINT stride = sizeof(LineVertex);
	UINT offset = 0;
	pContext->IASetVertexBuffers(0, 1, &pLineBuffers->m_pVB, &stride, &offset);
	pContext->IASetIndexBuffer(pLineBuffers->m_pIB, DXGI_FORMAT_R32_UINT, 0);
	pContext->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_LINELIST);

	// save viewports and render targets
	uint oldViewportCount = D3D11_VIEWPORT_AND_SCISSORRECT_OBJECT_COUNT_PER_PIPELINE;
	D3D11_VIEWPORT oldViewports[D3D11_VIEWPORT_AND_SCISSORRECT_OBJECT_COUNT_PER_PIPELINE];
	pContext->RSGetViewports(&oldViewportCount, oldViewports);
	ID3D11RenderTargetView* ppOldRTVs[D3D11_SIMULTANEOUS_RENDER_TARGET_COUNT];
	ID3D11DepthStencilView* pOldDSV;
	pContext->OMGetRenderTargets(D3D11_SIMULTANEOUS_RENDER_TARGET_COUNT, ppOldRTVs, &pOldDSV);

	// set our render target
	// if color is disabled, set no RTV - render depth only
	pContext->OMSetRenderTargets(enableColor ? 1 : 0, &m_pOpaqueRTV, m_pDepthDSV);

	uint pass = 2 * uint(m_particleRenderParams.m_lineRenderMode);
	if(blendBehind) pass++;

	if(m_stereoParams.m_stereoEnabled)
	{
		Mat4f viewLeft  = m_viewParams.BuildViewMatrix(EYE_LEFT,  m_stereoParams.m_eyeDistance);
		Mat4f viewRight = m_viewParams.BuildViewMatrix(EYE_RIGHT, m_stereoParams.m_eyeDistance);
		Mat4f projLeft  = m_projectionParams.BuildProjectionMatrix(EYE_LEFT,  m_stereoParams.m_eyeDistance, m_range);
		Mat4f projRight = m_projectionParams.BuildProjectionMatrix(EYE_RIGHT, m_stereoParams.m_eyeDistance, m_range);

		viewport.Height /= 2.0f;
		pContext->RSSetViewports(1, &viewport);

		m_lineEffect.m_pmWorldViewProjVariable->SetMatrix(projLeft * viewLeft);
		m_lineEffect.m_pTechnique->GetPassByIndex(pass)->Apply(0, pContext);

		pContext->DrawIndexed(pLineBuffers->m_indexCountTotal, 0, 0);
		if (renderSlice) {
			m_pQuadEffect->DrawTexture(projLeft * viewLeft, pContext, true);
		}

		viewport.TopLeftY += viewport.Height;
		pContext->RSSetViewports(1, &viewport);

		m_lineEffect.m_pmWorldViewProjVariable->SetMatrix(projRight * viewRight);
		m_lineEffect.m_pTechnique->GetPassByIndex(pass)->Apply(0, pContext);

		pContext->DrawIndexed(pLineBuffers->m_indexCountTotal, 0, 0);
		if (renderSlice) {
			m_pQuadEffect->DrawTexture(projRight * viewRight, pContext, true);
		}
	}
	else
	{
		Mat4f view = m_viewParams.BuildViewMatrix(EYE_CYCLOP, 0.0f);
		Mat4f proj = m_projectionParams.BuildProjectionMatrix(EYE_CYCLOP, 0.0f, m_range);

		pContext->RSSetViewports(1, &viewport);

		m_lineEffect.m_pmWorldViewProjVariable->SetMatrix(proj * view);
		m_lineEffect.m_pTechnique->GetPassByIndex(pass)->Apply(0, pContext);

		pContext->DrawIndexed(pLineBuffers->m_indexCountTotal, 0, 0);
		if (renderSlice) {
			m_pQuadEffect->DrawTexture(proj * view, pContext, true);
		}
	}

	// restore viewports and render targets
	pContext->OMSetRenderTargets(D3D11_SIMULTANEOUS_RENDER_TARGET_COUNT, ppOldRTVs, pOldDSV);
	pContext->RSSetViewports(oldViewportCount, oldViewports);
	for(uint i = 0; i < D3D11_SIMULTANEOUS_RENDER_TARGET_COUNT; i++)
	{
		SAFE_RELEASE(ppOldRTVs[i]);
	}
	SAFE_RELEASE(pOldDSV);

	pContext->Release();
}

tum3D::Vec4f RenderingManager::PrepareRenderSlice(ID3D11ShaderResourceView* tex, float alpha, float slicePosition, tum3D::Vec3f volumeSizeWorld, tum3D::Vec2f centerr)
{
	tum3D::Vec4f clipPlane;
	//Vec3f volumeHalfSizeWorld = m_pVolume->GetVolumeHalfSizeWorld();
	tum3D::Vec3f normal(0, 0, 1);
	tum3D::Vec2f size(volumeSizeWorld.x(), volumeSizeWorld.y());
	tum3D::Vec3f center(centerr.x(), centerr.y(), slicePosition);

	m_pQuadEffect->SetParameters(tex, center, normal, size, alpha);

	clipPlane.set(normal.x(), normal.y(), normal.z(), -slicePosition);
	//test if we have to flip the clip plane if the camera is at the wrong side
	Mat4f view = m_viewParams.BuildViewMatrix(EYE_CYCLOP, 0.0f);
	Mat4f proj = m_projectionParams.BuildProjectionMatrix(EYE_CYCLOP, 0.0f, m_range);
	Mat4f viewproj = proj * view;
	Vec4f v1; v1 = viewproj.multVec(Vec4f(-1, -1, center.z(), 1), v1); v1 /= v1.w();
	Vec4f v2; v2 = viewproj.multVec(Vec4f(+1, -1, center.z(), 1), v2); v2 /= v2.w();
	Vec4f v3; v3 = viewproj.multVec(Vec4f(-1, +1, center.z(), 1), v3); v3 /= v3.w();
	Vec2f dir1 = v2.xy() - v1.xy();
	Vec2f dir2 = v3.xy() - v1.xy();
	if (dir1.x()*dir2.y() - dir1.y()*dir2.x() > 0) {
		//camera is at the wrong side, flip clip
		clipPlane = -clipPlane;
	}
	return clipPlane;
}

void RenderingManager::ExtraRenderSlice()
{
	ID3D11DeviceContext* pContext = nullptr;
	m_pDevice->GetImmediateContext(&pContext);

	// build viewport
	D3D11_VIEWPORT viewport = {};
	viewport.TopLeftX = float(0);
	viewport.TopLeftY = float(0);
	viewport.Width = float(m_projectionParams.GetImageWidth(m_range));
	viewport.Height = float(m_projectionParams.m_imageHeight);
	viewport.MinDepth = 0.0f;
	viewport.MaxDepth = 1.0f;

	// save viewports and render targets
	uint oldViewportCount = D3D11_VIEWPORT_AND_SCISSORRECT_OBJECT_COUNT_PER_PIPELINE;
	D3D11_VIEWPORT oldViewports[D3D11_VIEWPORT_AND_SCISSORRECT_OBJECT_COUNT_PER_PIPELINE];
	pContext->RSGetViewports(&oldViewportCount, oldViewports);
	ID3D11RenderTargetView* ppOldRTVs[D3D11_SIMULTANEOUS_RENDER_TARGET_COUNT];
	ID3D11DepthStencilView* pOldDSV;
	pContext->OMGetRenderTargets(D3D11_SIMULTANEOUS_RENDER_TARGET_COUNT, ppOldRTVs, &pOldDSV);

	// set transparent offscreen target
	float clearColor[4] = { 0, 0, 0, 0.0f };
	pContext->ClearRenderTargetView(m_pTransparentRTV, clearColor);
	pContext->OMSetRenderTargets(1, &m_pTransparentRTV, m_pDepthDSV);

	//render
	if (m_stereoParams.m_stereoEnabled)
	{
		Mat4f viewLeft = m_viewParams.BuildViewMatrix(EYE_LEFT, m_stereoParams.m_eyeDistance);
		Mat4f viewRight = m_viewParams.BuildViewMatrix(EYE_RIGHT, m_stereoParams.m_eyeDistance);
		Mat4f projLeft = m_projectionParams.BuildProjectionMatrix(EYE_LEFT, m_stereoParams.m_eyeDistance, m_range);
		Mat4f projRight = m_projectionParams.BuildProjectionMatrix(EYE_RIGHT, m_stereoParams.m_eyeDistance, m_range);

		viewport.Height /= 2.0f;
		m_pQuadEffect->DrawTexture(projLeft * viewLeft, pContext, true);
		m_pQuadEffect->DrawTexture(projRight * viewRight, pContext, true);
	}
	else
	{
		Mat4f view = m_viewParams.BuildViewMatrix(EYE_CYCLOP, 0.0f);
		Mat4f proj = m_projectionParams.BuildProjectionMatrix(EYE_CYCLOP, 0.0f, m_range);

		pContext->RSSetViewports(1, &viewport);

		m_pQuadEffect->DrawTexture(proj * view, pContext, true);
	}

	// set our final render target
	pContext->OMSetRenderTargets(1, &m_pOpaqueRTV, m_pDepthDSV);

	// render transparent texture to final output
	Vec2f screenMin(-1.0f, -1.0f);
	Vec2f screenMax(1.0f, 1.0f);
	m_pScreenEffect->m_pvScreenMinVariable->SetFloatVector(screenMin);
	m_pScreenEffect->m_pvScreenMaxVariable->SetFloatVector(screenMax);
	Vec2f texCoordMin(0.0f, 0.0f);
	Vec2f texCoordMax(1.0f, 1.0f);
	m_pScreenEffect->m_pvTexCoordMinVariable->SetFloatVector(texCoordMin);
	m_pScreenEffect->m_pvTexCoordMaxVariable->SetFloatVector(texCoordMax);
	pContext->IASetInputLayout(NULL);
	pContext->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLESTRIP);
	m_pScreenEffect->m_pTexVariable->SetResource(m_pTransparentSRV);
	//if (m_particleRenderParams.m_particleRenderMode == PARTICLE_RENDER_ADDITIVE) {
	m_pScreenEffect->m_pTechnique->GetPassByIndex(2)->Apply(0, pContext);
	//}
	//else if (m_particleRenderParams.m_particleRenderMode == PARTICLE_RENDER_ORDER_INDEPENDENT) {
	//m_pScreenEffect->m_pTechnique->GetPassByIndex(3)->Apply(0, pContext);
	//}
	pContext->Draw(4, 0);

	// restore viewports and render targets
	pContext->OMSetRenderTargets(D3D11_SIMULTANEOUS_RENDER_TARGET_COUNT, ppOldRTVs, pOldDSV);
	pContext->RSSetViewports(oldViewportCount, oldViewports);
	for (uint i = 0; i < D3D11_SIMULTANEOUS_RENDER_TARGET_COUNT; i++)
	{
		SAFE_RELEASE(ppOldRTVs[i]);
	}
	SAFE_RELEASE(pOldDSV);

	pContext->Release();
}

void RenderingManager::SortParticles(LineBuffers* pLineBuffers, ID3D11DeviceContext* pContext)
{
	//compute matrix
	Mat4f view = m_viewParams.BuildViewMatrix(EYE_CYCLOP, 0.0f);
	Mat4f proj = m_projectionParams.BuildProjectionMatrix(EYE_CYCLOP, 0.0f, m_range);
	Mat4f projView = proj * view;
	//perform the inner product of the following vec4 with vec4(vertex.pos, 1) to compute the depth
	float4 depthMultiplier = make_float4(projView.get(2, 0), projView.get(2, 1), projView.get(2, 2), projView.get(2, 3));

	//aquire resources
	cudaSafeCall(cudaGraphicsMapResources(1, &pLineBuffers->m_pIBCuda));
	uint* dpIB = nullptr;
	cudaSafeCall(cudaGraphicsResourceGetMappedPointer((void**)&dpIB, nullptr, pLineBuffers->m_pIBCuda));
	cudaSafeCall(cudaGraphicsMapResources(1, &pLineBuffers->m_pIBCuda_sorted));
	uint* dpIB_sorted = nullptr;
	cudaSafeCall(cudaGraphicsResourceGetMappedPointer((void**)&dpIB_sorted, nullptr, pLineBuffers->m_pIBCuda_sorted));

	//copy indices
	cudaMemcpy(dpIB_sorted, dpIB, sizeof(uint) * pLineBuffers->m_indexCountTotal, cudaMemcpyDeviceToDevice);

	if (m_particleRenderParams.m_sortParticles) {
		//aquired vertex data
		cudaSafeCall(cudaGraphicsMapResources(1, &pLineBuffers->m_pVBCuda));
		LineVertex* dpVB = nullptr;
		cudaSafeCall(cudaGraphicsResourceGetMappedPointer((void**)&dpVB, nullptr, pLineBuffers->m_pVBCuda));

		//compute depth
		float* depth;
		cudaSafeCall(cudaMalloc2(&depth, sizeof(float) * pLineBuffers->m_indexCountTotal)); //todo: cache this array
		uint blockSize = 128;
		uint blockCount = (pLineBuffers->m_indexCountTotal + blockSize - 1) / blockSize;
		FillVertexDepth << < blockCount, blockSize >> > (dpVB, dpIB, depth, depthMultiplier, pLineBuffers->m_indexCountTotal);

		//sort indices 
		thrust::device_ptr<float> thrustKey(depth);
		thrust::device_ptr<uint> thrustValue(dpIB_sorted);
		thrust::sort_by_key(thrustKey, thrustKey + pLineBuffers->m_indexCountTotal, thrustValue);

		//release vertex data
		cudaSafeCall(cudaFree(depth));
		cudaSafeCall(cudaGraphicsUnmapResources(1, &pLineBuffers->m_pVBCuda));
	}

	//release resources
	cudaSafeCall(cudaGraphicsUnmapResources(1, &pLineBuffers->m_pIBCuda));
	cudaSafeCall(cudaGraphicsUnmapResources(1, &pLineBuffers->m_pIBCuda_sorted));
}

void RenderingManager::CreateFTLETexture()
{
	std::cout << "RenderingManager::CreateFTLETexture: creating texture." << std::endl;
	size_t memFree = 0;
	size_t memTotal = 0;
	cudaSafeCall(cudaMemGetInfo(&memFree, &memTotal));
	std::cout << "\tAvailable: " << float(memFree) / (1024.0f * 1024.0f) << "MB" << std::endl;
	
	int res = m_particleTraceParams.m_ftleResolution;

	if (m_ftleTexture.IsTextureCreated())
		m_ftleTexture.ReleaseResources();

	//if (!m_ftleTexture.CreateTexture(m_pDevice, res, res, 1, 1, DXGI_FORMAT_R32G32B32A32_FLOAT))
	if (!m_ftleTexture.CreateTexture(m_pDevice, res, res, 1, 1, DXGI_FORMAT_R32_FLOAT))
		std::cerr << "------------ Error: could not create FTLE texture" << std::endl;

	m_ftleTexture.RegisterCUDAResources();

	cudaSafeCall(cudaDeviceSynchronize());

	cudaSafeCall(cudaMemGetInfo(&memFree, &memTotal));
	std::cout << "\tAvailable: " << float(memFree) / (1024.0f * 1024.0f) << "MB" << std::endl;

	std::cout << "RenderingManager::CreateFTLETexture: done." << std::endl;
}

void RenderingManager::ComputeFTLE(const TimeVolume& vol, SimpleParticleVertexDeltaT* dpParticles)
{
	std::cout << "ComputeFTLE";
	std::cout << " Res(" << m_ftleTexture.width << "," << m_ftleTexture.height << ")" << std::endl;

	if (!m_ftleTexture.IsTextureCreated())
	{
		std::cerr << "------------ Texture is not valid." << std::endl;
		return;
	}
	if (!m_ftleTexture.IsRegisteredWithCuda())
	{
		std::cerr << "------------ Texture is not registered with cuda" << std::endl;
		return;
	}

	int width = m_ftleTexture.width;
	int height = m_ftleTexture.height;

	// kick off the kernel and send the staging buffer cudaLinearMemory as an argument to allow the kernel to write to it
	//cuda_texture_2d(m_ftleTexture.cudaLinearMemory, width, height, m_ftleTexture.pitch, t);


	cudaError_t error = cudaSuccess;

	dim3 Db = dim3(16, 16, 1);   // block dimensions are fixed to be 256 threads
	dim3 Dg = dim3((width + Db.x - 1) / Db.x, (height + Db.y - 1) / Db.y);

	cudaSafeCall(cudaDeviceSynchronize());

	float3 separationDist = make_float3(m_particleTraceParams.m_ftleSeparationDistance.x(), m_particleTraceParams.m_ftleSeparationDistance.y(), m_particleTraceParams.m_ftleSeparationDistance.z());

	ComputeFTLEKernel << <Dg, Db >> >((unsigned char*)m_ftleTexture.cudaLinearMemory, width, height, m_ftleTexture.pitch, separationDist, vol.GetCurTime(), dpParticles, m_particleTraceParams.m_lineCount, m_ftleScale);

	error = cudaGetLastError();

	cudaSafeCall(cudaDeviceSynchronize());

	if (error != cudaSuccess)
		printf("ComputeFTLEKernel() failed to launch error = %d\n", error);

	cudaCheckMsg("ComputeFTLEKernel failed");

	// then we want to copy cudaLinearMemory to the D3D texture, via its mapped form : cudaArray
	cudaSafeCall(cudaGraphicsMapResources(1, &m_ftleTexture.cudaResource));

	cudaArray *cuArray;
	cudaSafeCall(cudaGraphicsSubResourceGetMappedArray(&cuArray, m_ftleTexture.cudaResource, 0, 0));
	cudaCheckMsg("cudaGraphicsSubResourceGetMappedArray (cuda_texture_2d) failed");
	
	cudaSafeCall(cudaMemcpy2DToArray(
		cuArray, // dst array
		0, 0,    // offset
		m_ftleTexture.cudaLinearMemory, m_ftleTexture.pitch,       // src
		width * m_ftleTexture.GetNumberOfComponents() * sizeof(float), height, // extent
		cudaMemcpyDeviceToDevice)); // kind
	cudaCheckMsg("cudaMemcpy2DToArray failed");

	cudaSafeCall(cudaGraphicsUnmapResources(1, &m_ftleTexture.cudaResource));
}

void RenderingManager::RenderParticles(const LineBuffers* pLineBuffers, ID3D11DeviceContext* pContext, D3D11_VIEWPORT viewport, const tum3D::Vec4f* clipPlane, bool renderSlice)
{
	// IA
	pContext->IASetInputLayout(m_lineEffect.m_pInputLayout);
	UINT stride = sizeof(LineVertex);
	UINT offset = 0;
	pContext->IASetVertexBuffers(0, 1, &pLineBuffers->m_pVB, &stride, &offset);
	pContext->IASetIndexBuffer(pLineBuffers->m_pIB_sorted, DXGI_FORMAT_R32_UINT, 0);
	pContext->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_POINTLIST);

	//set rotation, needed to compute the correct transformation
	m_lineEffect.m_pmWorldViewRotation->SetMatrix(m_viewParams.BuildRotationMatrix());

	//set clip plane
	if (clipPlane == NULL) {
		tum3D::Vec4f clip(0, 0, 0, 0);
		m_lineEffect.m_pvParticleClipPlane->SetFloatVector(clip);
	}
	else {
		m_lineEffect.m_pvParticleClipPlane->SetFloatVector(*clipPlane);
	}

	// save viewports and render targets
	uint oldViewportCount = D3D11_VIEWPORT_AND_SCISSORRECT_OBJECT_COUNT_PER_PIPELINE;
	D3D11_VIEWPORT oldViewports[D3D11_VIEWPORT_AND_SCISSORRECT_OBJECT_COUNT_PER_PIPELINE];
	pContext->RSGetViewports(&oldViewportCount, oldViewports);
	ID3D11RenderTargetView* ppOldRTVs[D3D11_SIMULTANEOUS_RENDER_TARGET_COUNT];
	ID3D11DepthStencilView* pOldDSV;
	pContext->OMGetRenderTargets(D3D11_SIMULTANEOUS_RENDER_TARGET_COUNT, ppOldRTVs, &pOldDSV);

	//specify path
	int pass = 6;
	float clearColorSingle = 0;
	if (m_particleRenderParams.m_particleRenderMode == PARTICLE_RENDER_MULTIPLICATIVE) {
		pass = 7;
		clearColorSingle = 1;
	}
	else if (m_particleRenderParams.m_particleRenderMode == PARTICLE_RENDER_ALPHA) {
		pass = 8;
	}
	// set transparent offscreen target
	float clearColor[4] = { clearColorSingle, clearColorSingle, clearColorSingle, 0.0f };
	pContext->ClearRenderTargetView(m_pTransparentRTV, clearColor);
	pContext->OMSetRenderTargets(1, &m_pTransparentRTV, m_pDepthDSV);

	//render
	if (m_stereoParams.m_stereoEnabled)
	{
		Mat4f viewLeft = m_viewParams.BuildViewMatrix(EYE_LEFT, m_stereoParams.m_eyeDistance);
		Mat4f viewRight = m_viewParams.BuildViewMatrix(EYE_RIGHT, m_stereoParams.m_eyeDistance);
		Mat4f projLeft = m_projectionParams.BuildProjectionMatrix(EYE_LEFT, m_stereoParams.m_eyeDistance, m_range);
		Mat4f projRight = m_projectionParams.BuildProjectionMatrix(EYE_RIGHT, m_stereoParams.m_eyeDistance, m_range);

		viewport.Height /= 2.0f;
		pContext->RSSetViewports(1, &viewport);

		m_lineEffect.m_pmWorldViewProjVariable->SetMatrix(projLeft * viewLeft);
		m_lineEffect.m_pTechnique->GetPassByIndex(pass)->Apply(0, pContext);

		pContext->DrawIndexed(pLineBuffers->m_indexCountTotal, 0, 0);
		if (renderSlice) {
			m_pQuadEffect->DrawTexture(projLeft * viewLeft, pContext, false);
		}

		viewport.TopLeftY += viewport.Height;
		pContext->RSSetViewports(1, &viewport);

		m_lineEffect.m_pmWorldViewProjVariable->SetMatrix(projRight * viewRight);
		m_lineEffect.m_pTechnique->GetPassByIndex(pass)->Apply(0, pContext);

		pContext->DrawIndexed(pLineBuffers->m_indexCountTotal, 0, 0);
		if (renderSlice) {
			m_pQuadEffect->DrawTexture(projRight * viewRight, pContext, false);
		}
	}
	else
	{
		Mat4f view = m_viewParams.BuildViewMatrix(EYE_CYCLOP, 0.0f);
		Mat4f proj = m_projectionParams.BuildProjectionMatrix(EYE_CYCLOP, 0.0f, m_range);

		pContext->RSSetViewports(1, &viewport);

		m_lineEffect.m_pmWorldViewProjVariable->SetMatrix(proj * view);
		m_lineEffect.m_pTechnique->GetPassByIndex(pass)->Apply(0, pContext);

		pContext->DrawIndexed(pLineBuffers->m_indexCountTotal, 0, 0);
		if (renderSlice) {
			m_pQuadEffect->DrawTexture(proj * view, pContext, false);
		}
	}

	// set our final render target
	pContext->OMSetRenderTargets(1, &m_pOpaqueRTV, m_pDepthDSV);

	// render transparent texture to final output
	Vec2f screenMin(-1.0f, -1.0f);
	Vec2f screenMax(1.0f, 1.0f);
	m_pScreenEffect->m_pvScreenMinVariable->SetFloatVector(screenMin);
	m_pScreenEffect->m_pvScreenMaxVariable->SetFloatVector(screenMax);
	Vec2f texCoordMin(0.0f, 0.0f);
	Vec2f texCoordMax(1.0f, 1.0f);
	m_pScreenEffect->m_pvTexCoordMinVariable->SetFloatVector(texCoordMin);
	m_pScreenEffect->m_pvTexCoordMaxVariable->SetFloatVector(texCoordMax);
	pContext->IASetInputLayout(NULL);
	pContext->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLESTRIP);
	m_pScreenEffect->m_pTexVariable->SetResource(m_pTransparentSRV);
	//if (m_particleRenderParams.m_particleRenderMode == PARTICLE_RENDER_ADDITIVE) {
		m_pScreenEffect->m_pTechnique->GetPassByIndex(2)->Apply(0, pContext);
	//}
	//else if (m_particleRenderParams.m_particleRenderMode == PARTICLE_RENDER_ORDER_INDEPENDENT) {
		//m_pScreenEffect->m_pTechnique->GetPassByIndex(3)->Apply(0, pContext);
	//}
	pContext->Draw(4, 0);

	// restore viewports and render targets
	pContext->OMSetRenderTargets(D3D11_SIMULTANEOUS_RENDER_TARGET_COUNT, ppOldRTVs, pOldDSV);
	pContext->RSSetViewports(oldViewportCount, oldViewports);
	for (uint i = 0; i < D3D11_SIMULTANEOUS_RENDER_TARGET_COUNT; i++)
	{
		SAFE_RELEASE(ppOldRTVs[i]);
	}
	SAFE_RELEASE(pOldDSV);
}

void RenderingManager::RenderBalls(const TimeVolume& vol, const BallBuffers* pBallBuffers, float radius)
{
	if(pBallBuffers == nullptr || pBallBuffers->m_ballCount == 0) return;

	ID3D11DeviceContext* pContext = nullptr;
	m_pDevice->GetImmediateContext(&pContext);

	// save viewports and render targets
	uint oldViewportCount = D3D11_VIEWPORT_AND_SCISSORRECT_OBJECT_COUNT_PER_PIPELINE;
	D3D11_VIEWPORT oldViewports[D3D11_VIEWPORT_AND_SCISSORRECT_OBJECT_COUNT_PER_PIPELINE];
	pContext->RSGetViewports(&oldViewportCount, oldViewports);
	ID3D11RenderTargetView* ppOldRTVs[D3D11_SIMULTANEOUS_RENDER_TARGET_COUNT];
	ID3D11DepthStencilView* pOldDSV;
	pContext->OMGetRenderTargets(D3D11_SIMULTANEOUS_RENDER_TARGET_COUNT, ppOldRTVs, &pOldDSV);

	// set our render target
	pContext->OMSetRenderTargets(1, &m_pOpaqueRTV, m_pDepthDSV);

	// build viewport
	D3D11_VIEWPORT viewport = {};
	viewport.TopLeftX = float(0);
	viewport.TopLeftY = float(0);
	viewport.Width    = float(m_projectionParams.GetImageWidth(m_range));
	viewport.Height   = float(m_projectionParams.m_imageHeight);
	viewport.MinDepth = 0.0f;
	viewport.MaxDepth = 1.0f;

	// IA
	pContext->IASetInputLayout(m_lineEffect.m_pInputLayoutBalls);
	UINT stride = sizeof(BallVertex);
	UINT offset = 0;
	pContext->IASetVertexBuffers(0, 1, &pBallBuffers->m_pVB, &stride, &offset);
	pContext->IASetIndexBuffer(nullptr, DXGI_FORMAT_R32_UINT, 0);
	pContext->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_POINTLIST);

	Vec3f lightDir;

	if (m_renderingParams.m_FixedLightDir)
		lightDir = normalize(m_renderingParams.m_lightDir);
	else
		lightDir = m_viewParams.GetViewDir();

	// shader vars
	m_lineEffect.m_pvLightPosVariable->SetFloatVector(lightDir);

	m_lineEffect.m_pvColor0Variable->SetFloatVector(m_particleRenderParams.m_color0);
	m_lineEffect.m_pvColor1Variable->SetFloatVector(m_particleRenderParams.m_color1);

	Vec3f boxMin = -vol.GetVolumeHalfSizeWorld();
	tum3D::Vec3f sizePhys = (float)vol.GetVolumeSize().maximum() * vol.GetGridSpacing();
	m_lineEffect.m_pvBoxMinVariable->SetFloatVector(boxMin);
	m_lineEffect.m_pvBoxSizeVariable->SetFloatVector(Vec3f(2.0f / sizePhys));

	m_lineEffect.m_pfBallRadiusVariable->SetFloat(radius);

	uint pass = 0;

	if(m_stereoParams.m_stereoEnabled)
	{
		//Mat4f viewLeft  = m_viewParams.BuildViewMatrix(EYE_LEFT,  m_stereoParams.m_eyeDistance);
		//Mat4f viewRight = m_viewParams.BuildViewMatrix(EYE_RIGHT, m_stereoParams.m_eyeDistance);
		//Mat4f projLeft  = m_projectionParams.BuildProjectionMatrix(EYE_LEFT,  m_stereoParams.m_eyeDistance, m_range);
		//Mat4f projRight = m_projectionParams.BuildProjectionMatrix(EYE_RIGHT, m_stereoParams.m_eyeDistance, m_range);

		//viewport.Height /= 2.0f;
		//pContext->RSSetViewports(1, &viewport);

		//m_lineEffect.m_pmWorldViewProjVariable->SetMatrix(projLeft * viewLeft);
		//m_lineEffect.m_pTechniqueBalls->GetPassByIndex(pass)->Apply(0, pContext);

		//pContext->Draw(pBallBuffers->m_ballCount, 0);

		//viewport.TopLeftY += viewport.Height;
		//pContext->RSSetViewports(1, &viewport);

		//m_lineEffect.m_pmWorldViewProjVariable->SetMatrix(projRight * viewRight);
		//m_lineEffect.m_pTechniqueBalls->GetPassByIndex(pass)->Apply(0, pContext);

		//pContext->Draw(pBallBuffers->m_ballCount, 0);
	}
	else
	{
		Mat4f view = m_viewParams.BuildViewMatrix(EYE_CYCLOP, 0.0f);
		Mat4f proj = m_projectionParams.BuildProjectionMatrix(EYE_CYCLOP, 0.0f, m_range);

		Mat4f viewInv;
		invert4x4(view, viewInv);
		Vec3f camPos = Vec3f(viewInv.getCol(3));
		Vec3f camRight = Vec3f(viewInv.getCol(0));
		m_lineEffect.m_pvCamPosVariable->SetFloatVector(camPos);
		m_lineEffect.m_pvCamRightVariable->SetFloatVector(camRight);

		pContext->RSSetViewports(1, &viewport);

		m_lineEffect.m_pmWorldViewProjVariable->SetMatrix(proj * view);
		m_lineEffect.m_pTechniqueBalls->GetPassByIndex(pass)->Apply(0, pContext);

		pContext->Draw(pBallBuffers->m_ballCount, 0);
	}

	// restore viewports and render targets
	pContext->OMSetRenderTargets(D3D11_SIMULTANEOUS_RENDER_TARGET_COUNT, ppOldRTVs, pOldDSV);
	pContext->RSSetViewports(oldViewportCount, oldViewports);
	for(uint i = 0; i < D3D11_SIMULTANEOUS_RENDER_TARGET_COUNT; i++)
	{
		SAFE_RELEASE(ppOldRTVs[i]);
	}
	SAFE_RELEASE(pOldDSV);

	pContext->Release();
}

void RenderingManager::CopyDepthTexture(ID3D11DeviceContext* deviceContext, ID3D11Texture2D* target)
{
	assert(target != nullptr);
	assert(deviceContext != nullptr);

	deviceContext->CopyResource(target, m_pDepthTex);
}

void RenderingManager::RenderHeatMap(HeatMapManager* pHeatMapManager)
{
	if (!pHeatMapManager->IsRenderingEnabled()) return;

	//get device context
	ID3D11DeviceContext* pContext = nullptr;
	m_pDevice->GetImmediateContext(&pContext);

	// build viewport
	D3D11_VIEWPORT viewport = {};
	viewport.TopLeftX = float(0);
	viewport.TopLeftY = float(0);
	viewport.Width = float(m_projectionParams.GetImageWidth(m_range));
	viewport.Height = float(m_projectionParams.m_imageHeight);
	viewport.MinDepth = 0.0f;
	viewport.MaxDepth = 1.0f;

	//set render target
	uint oldViewportCount = D3D11_VIEWPORT_AND_SCISSORRECT_OBJECT_COUNT_PER_PIPELINE;
	D3D11_VIEWPORT oldViewports[D3D11_VIEWPORT_AND_SCISSORRECT_OBJECT_COUNT_PER_PIPELINE];
	pContext->RSGetViewports(&oldViewportCount, oldViewports);
	ID3D11RenderTargetView* ppOldRTVs[D3D11_SIMULTANEOUS_RENDER_TARGET_COUNT];
	ID3D11DepthStencilView* pOldDSV;
	pContext->OMGetRenderTargets(D3D11_SIMULTANEOUS_RENDER_TARGET_COUNT, ppOldRTVs, &pOldDSV);
	pContext->OMSetRenderTargets(1, &m_pTransparentRTV, NULL);


	// set transparent offscreen target
	float clearColor[4] = {0.0f, 0.0f, 0.0f, 0.0f };
	pContext->ClearRenderTargetView(m_pTransparentRTV, clearColor);
	pContext->OMSetRenderTargets(1, &m_pTransparentRTV, NULL);

	// RENDER IT
	Mat4f view = m_viewParams.BuildViewMatrix(EYE_CYCLOP, 0.0f);
	Mat4f proj = m_projectionParams.BuildProjectionMatrix(EYE_CYCLOP, 0.0f, m_range);
	pContext->RSSetViewports(1, &viewport);
	//pHeatMapManager->Render(proj * view, m_projectionParams, viewport, m_pDepthSRV);
	pHeatMapManager->Render(view, m_projectionParams, m_range, m_pDepthSRV);

	// set our final render target
	pContext->OMSetRenderTargets(1, &m_pOpaqueRTV, m_pDepthDSV);

	// render transparent texture to final output
	Vec2f screenMin(-1.0f, -1.0f);
	Vec2f screenMax(1.0f, 1.0f);
	m_pScreenEffect->m_pvScreenMinVariable->SetFloatVector(screenMin);
	m_pScreenEffect->m_pvScreenMaxVariable->SetFloatVector(screenMax);
	Vec2f texCoordMin(0.0f, 0.0f);
	Vec2f texCoordMax(1.0f, 1.0f);
	m_pScreenEffect->m_pvTexCoordMinVariable->SetFloatVector(texCoordMin);
	m_pScreenEffect->m_pvTexCoordMaxVariable->SetFloatVector(texCoordMax);
	pContext->IASetInputLayout(NULL);
	pContext->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLESTRIP);
	m_pScreenEffect->m_pTexVariable->SetResource(m_pTransparentSRV);
	m_pScreenEffect->m_pTechnique->GetPassByIndex(2)->Apply(0, pContext);
	pContext->Draw(4, 0);

	// restore viewports and render targets
	pContext->OMSetRenderTargets(D3D11_SIMULTANEOUS_RENDER_TARGET_COUNT, ppOldRTVs, pOldDSV);
	pContext->RSSetViewports(oldViewportCount, oldViewports);
	for (uint i = 0; i < D3D11_SIMULTANEOUS_RENDER_TARGET_COUNT; i++)
	{
		SAFE_RELEASE(ppOldRTVs[i]);
	}
	SAFE_RELEASE(pOldDSV);

	//release device context
	pContext->Release();
}