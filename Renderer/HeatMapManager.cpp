#include "HeatMapManager.h"

#include <algorithm>
#include <iostream>
#include <cudaUtil.h>

#include <D3Dcompiler.h>
#include "CSysTools.h"

#include "TracingCommon.h"
#include "HeatMapKernel.h"

using namespace tum3D;


HeatMapManager::HeatMapManager()
	: m_pCompressShared(NULL)
	, m_pCompressVolume(NULL)
	, m_pDevice(NULL)
	, m_pVolume(NULL)
	, m_pHeatMap(NULL)
	, m_isCreated(false)
	, m_params()
	, m_pShader(NULL)
	, m_dataChanged(true)
	, m_hasData(false)
	, m_cudaCopyBuffer(NULL)
	, m_seedTexCuda(NULL)
	, m_seedTexSize(0)
	, m_seedTexChanged(false)
{
	m_textures[0] = {};
	m_textures[1] = {};
}


HeatMapManager::~HeatMapManager()
{
	delete m_pHeatMap;
	delete m_pShader;
}

bool HeatMapManager::Create(GPUResources * pCompressShared, CompressVolumeResources * pCompressVolume, ID3D11Device * pDevice)
{
	m_pCompressShared = pCompressShared;
	m_pCompressVolume = pCompressVolume;
	m_pDevice = pDevice;

	//create shader
	m_pShader = new HeatMapRaytracerEffect();
	m_pShader->Create(pDevice);

	m_isCreated = true;

	std::cout << "HeatMapManager created" << std::endl;

	return true;
}

void HeatMapManager::Release()
{
	delete m_pHeatMap;
	m_pHeatMap = nullptr;

	if (m_seedTexCuda != nullptr) {
		cudaSafeCall(cudaFree(m_seedTexCuda));
		m_seedTexCuda = nullptr;
	}

	if (m_pShader != nullptr) {
		m_pShader->SafeRelease();
		delete m_pShader;
		m_pShader = nullptr;
	}

	ReleaseRenderTextures();

	std::cout << "HeatMapMaanger released" << std::endl;
}

void HeatMapManager::SetParams(const HeatMapParams & params)
{
	if (m_params.m_recordTexture.m_colors != params.m_recordTexture.m_colors) {
		m_seedTexChanged = true;
	}
	m_params = params;
	//std::cout << "enable rendering: " << m_params.m_enableRendering << std::endl;
}

void HeatMapManager::DebugPrintParams()
{
	std::cout << "HeatMapParams {" << std::endl;
	std::cout << "  enable recording: " << m_params.m_enableRecording << std::endl;
	std::cout << "  enable rendering: " << m_params.m_enableRendering << std::endl;
	std::cout << "  auto reset: " << m_params.m_autoReset << std::endl;
	std::cout << "  normalize: " << m_params.m_normalize << std::endl;
	std::cout << "  step size: " << m_params.m_stepSize << std::endl;
	std::cout << "  density scale: " << m_params.m_densityScale << std::endl;
	std::cout << "  tf alpha scale: " << m_params.m_tfAlphaScale << std::endl;
	std::cout << "  tf range min: " << m_params.m_tfRangeMin << std::endl;
	std::cout << "  tf range max: " << m_params.m_tfRangeMax << std::endl;
	std::cout << "}" << std::endl;
}

void HeatMapManager::SetVolumeAndReset(const TimeVolume & volume)
{
	if (m_pVolume == &volume) {
		std::cout << "HeatMapManager::SetVolumeAndReset called with the same volume" << std::endl;
		return;
	}
	delete m_pHeatMap;
	m_pHeatMap = nullptr;
	m_pVolume = &volume;
	m_boxMin = -Vec4f(volume.GetVolumeHalfSizeWorld(), 1.0f);
	m_boxMax = Vec4f(volume.GetVolumeHalfSizeWorld(), 1.0f);

	//compute heatmap size
	Vec3i volumeResolution = volume.GetVolumeSize();
	Vec3f volumeHalfSizeWorld = volume.GetVolumeHalfSizeWorld();
	static const int MAX_RESOLUTION = 128;
	static const int MIN_RESOLUTION = 4;
	int scalingFactor = std::max(1, volumeResolution.maximum() / MAX_RESOLUTION);
	tum3D::Vec3i heatMapResolution = tum3D::Vec3i(
		max(MIN_RESOLUTION, volumeResolution.x() / scalingFactor),
		max(MIN_RESOLUTION, volumeResolution.y() / scalingFactor),
		max(MIN_RESOLUTION, volumeResolution.z() / scalingFactor)
	);
	std::cout << "Volume resolution: " << volumeResolution << std::endl;
	std::cout << "Heat Map resolution: " << heatMapResolution << std::endl;
	m_resolution = make_int3(heatMapResolution.x(), heatMapResolution.y(), heatMapResolution.z());
	m_worldOffset = make_float3(volumeHalfSizeWorld.x(), volumeHalfSizeWorld.y(), volumeHalfSizeWorld.z());
	m_worldToGrid = make_float3(
		heatMapResolution.x() / (2 * volumeHalfSizeWorld.x()),
		heatMapResolution.y() / (2 * volumeHalfSizeWorld.y()),
		heatMapResolution.z() / (2 * volumeHalfSizeWorld.z()));

	//create heatmap
	m_pHeatMap = new HeatMap(heatMapResolution.x(), heatMapResolution.y(), heatMapResolution.z());
	m_hasData = false;

	//create render textures
	CreateRenderTextures(m_pDevice);
	m_dataChanged = true;

	std::cout << "Heat Map created" << std::endl;
}

void HeatMapManager::ProcessLines(std::shared_ptr<LineBuffers> pLineBuffer)
{
	if (!m_params.m_enableRecording) return;
	std::cout << "Process lines" << std::endl;

	m_hasData = false;

	//check for a changed seed texture
	size_t expectedTexSize = m_params.m_recordTexture.m_width
		* m_params.m_recordTexture.m_height;
	if (expectedTexSize > 0 && expectedTexSize != m_seedTexSize) {
		m_seedTexSize = expectedTexSize;
		if (m_seedTexCuda != nullptr) {
			cudaSafeCall(cudaFree(m_seedTexCuda));
		}
		cudaSafeCall(cudaMalloc(&m_seedTexCuda, sizeof(unsigned int) * m_seedTexSize));
		m_seedTexChanged = true;
	}

	//check if seed texture contents have changed
	if (m_seedTexChanged && m_params.m_recordTexture.m_colors != nullptr) {
		cudaSafeCall(cudaMemcpy(m_seedTexCuda, m_params.m_recordTexture.m_colors, 
			sizeof(unsigned int) * m_seedTexSize, cudaMemcpyHostToDevice));
	}
	m_seedTexChanged = false;

	//allocate new or delete old channels
	std::set<unsigned int> picked = m_params.m_recordTexture.m_picked;
	if (picked.empty()) {
		picked.insert(0); //nothing selected -> use everything
	}
	for (unsigned int id : m_pHeatMap->getAllChannelIDs()) {
		if (picked.count(id) == 0/* && m_params.m_autoReset*/) {
			m_pHeatMap->deleteChannel(id); //delete old channels
			std::cout << "Channel with id " << id << " deleted" << std::endl;
		}
	}
	for (unsigned int id : picked) {
		HeatMap::Channel_ptr c = m_pHeatMap->getChannel(id);
		if (c == nullptr) {
			c = m_pHeatMap->createChannel(id); //create new channel
			c->clear();
			std::cout << "Channel with id " << id << " created" << std::endl;
		}
		else if (m_params.m_autoReset) {
			c->clear(); //clear existing channel
			std::cout << "Channel with id " << id << " cleared" << std::endl;
		}
	}
	std::cout << "key colors:";
	for (unsigned int k : picked) std::cout << " " << k;
	std::cout << std::endl;

	//aquire buffers
	cudaSafeCall(cudaGraphicsMapResources(1, &pLineBuffer->m_pIBCuda));
	uint* dpIB = nullptr;
	cudaSafeCall(cudaGraphicsResourceGetMappedPointer((void**)&dpIB, nullptr, pLineBuffer->m_pIBCuda));
	cudaSafeCall(cudaGraphicsMapResources(1, &pLineBuffer->m_pVBCuda));
	LineVertex* dpVB = nullptr;
	cudaSafeCall(cudaGraphicsResourceGetMappedPointer((void**)&dpVB, nullptr, pLineBuffer->m_pVBCuda));

	//fill channels
	for (unsigned int id : picked) {
		HeatMap::Channel_ptr channel = m_pHeatMap->getChannel(id);
		heatmapKernelFillChannel(channel->getCudaBuffer(), dpVB, dpIB, pLineBuffer->m_indexCountTotal,
			m_resolution, m_worldOffset, m_worldToGrid,
			m_seedTexCuda, make_int2(m_params.m_recordTexture.m_width, m_params.m_recordTexture.m_height), id);
	}

	//release resources
	cudaSafeCall(cudaGraphicsUnmapResources(1, &pLineBuffer->m_pVBCuda));
	cudaSafeCall(cudaGraphicsUnmapResources(1, &pLineBuffer->m_pIBCuda));

	//get min and max value
	std::pair<uint, uint> minMaxValue (std::numeric_limits<uint>::max(), std::numeric_limits<uint>::min());
	for (unsigned int id : m_pHeatMap->getAllChannelIDs()) {
		std::pair<uint, uint> temp = heatmapKernelFindMinMax(
			m_pHeatMap->getChannel(id)->getCudaBuffer(), m_resolution);
		minMaxValue.first = std::min(minMaxValue.first, temp.first);
		minMaxValue.second = std::max(minMaxValue.second, temp.second);
	}
	m_maxData = minMaxValue.second;
	std::cout << "Done, min value: " << minMaxValue.first << ", max value: " << minMaxValue.second << std::endl;

	m_dataChanged = true;
	m_hasData = true;
}

void HeatMapManager::Render(Mat4f viewProjMat, ProjectionParams projParams, 
	const Range1D& range, ID3D11ShaderResourceView* depthTextureSRV)
{
	if (!m_params.m_enableRendering) return;
	if (!m_hasData) return;
	//std::cout << "HeatMapManager: render " 
	//	<< m_params.m_renderedChannels[0] << " and "
	//	<< m_params.m_renderedChannels[1] << std::endl;

	//retrieve channels
	HeatMap::Channel_ptr channels[2];
	int validChannelCount = 0;
	for (int i = 0; i < 2; ++i) {
		channels[validChannelCount] = m_pHeatMap->getChannel(m_params.m_renderedChannels[i]);
		if (channels[validChannelCount] != nullptr)
			validChannelCount++;
	}
	//std::cout << "number of valid channels for rendering: " << validChannelCount << std::endl;
	if (validChannelCount == 0) return;

	ID3D11DeviceContext* pContext = nullptr;
	m_pDevice->GetImmediateContext(&pContext);

	//test if the selected channels have changed, if yes, copy them to the gpu
	for (int i = 0; i < 2; ++i) {
		if (m_oldPickedChannels[i] != m_params.m_renderedChannels[i]) {
			m_dataChanged = true;
			m_oldPickedChannels[i] = m_params.m_renderedChannels[i];
		}
	}

	if (m_dataChanged) {
		m_dataChanged = false;
		CopyToRenderTexture(channels[0], 0);
		if (validChannelCount>1)
			CopyToRenderTexture(channels[1], 1);
	}

	//perform the raytracing

	//textures, volume data
	pContext->IASetInputLayout(NULL);
	pContext->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLESTRIP);
	m_pShader->m_pTransferFunction->SetResource(m_params.m_pTransferFunction);
	m_pShader->m_pHeatMap1->SetResource(m_textures[0].dxSRV);
	if (validChannelCount>1)
		m_pShader->m_pHeatMap2->SetResource(m_textures[1].dxSRV);
	m_pShader->m_pDepthTexture->SetResource(depthTextureSRV);

	//projection settings
	Mat4f invViewProjMat;
	invert4x4(viewProjMat, invViewProjMat);
	m_pShader->m_pmWorldView->SetMatrix(viewProjMat);
	m_pShader->m_pmInvWorldView->SetMatrix(invViewProjMat);
	m_pShader->m_pvDepthParams->SetFloatVector(Vec4f(
		projParams.m_near, projParams.m_far, 
		projParams.m_near * projParams.m_far, projParams.m_far - projParams.m_near));
	m_pShader->m_pvScreenSize->SetIntVector(
		Vec2i(projParams.m_imageWidth, projParams.m_imageHeight));
	m_pShader->m_pvBoxMin->SetFloatVector(m_boxMin);
	m_pShader->m_pvBoxMax->SetFloatVector(m_boxMax);
	float frustum[6];
	projParams.GetFrustumParams(frustum, EYE_CYCLOP, 0.0f, range);
	float nearFrustum = frustum[4];
	Vec4f viewport(
		frustum[0] / nearFrustum, //left
		frustum[1] / nearFrustum, //right
		frustum[2] / nearFrustum, //bottom
		frustum[3] / nearFrustum); //top
	m_pShader->m_pvViewport->SetFloatVector(viewport);
	if (validChannelCount>1)
		m_pShader->m_pTechnique->GetPassByIndex(1)->Apply(0, pContext);
	else
		m_pShader->m_pTechnique->GetPassByIndex(0)->Apply(0, pContext);

	//tracing settings
	m_pShader->m_pfStepSizeWorld->SetFloat(m_params.m_stepSize);
	m_pShader->m_pfDensityScale->SetFloat(m_params.m_densityScale
		* (m_params.m_normalize ? 1.0/m_maxData : 1));
	m_pShader->m_pfAlphaScale->SetFloat(m_params.m_tfAlphaScale);

	pContext->Draw(4, 0);

	m_pShader->m_pDepthTexture->SetResource(nullptr); //clear depth buffer
	m_pShader->m_pTechnique->GetPassByIndex(0)->Apply(0, pContext);

	SAFE_RELEASE(pContext);
}

void HeatMapManager::ClearChannels()
{
	m_pHeatMap->clearAllChannels();
	m_pHeatMap->deleteAllChannels();
	m_hasData = false;
}

void HeatMapManager::ReleaseRenderTextures()
{
	for (int i = 0; i < 2; ++i) {
		if (m_textures[i].dxTexture == NULL) return;
		cudaSafeCall(cudaGraphicsUnregisterResource(m_textures[i].cudaResource));
		SAFE_RELEASE(m_textures[i].dxSRV);
		SAFE_RELEASE(m_textures[i].dxTexture);
	}

	if (m_cudaCopyBuffer != nullptr) {
		cudaFree(m_cudaCopyBuffer);
		m_cudaCopyBuffer = nullptr;
	}
}

void HeatMapManager::CreateRenderTextures(ID3D11Device* pDevice)
{
	ReleaseRenderTextures();

	cudaSafeCall(cudaMalloc(&m_cudaCopyBuffer, sizeof(float) 
		* m_resolution.x * m_resolution.y * m_resolution.z));

	D3D11_TEXTURE3D_DESC desc;
	ZeroMemory(&desc, sizeof(D3D11_TEXTURE3D_DESC));
	desc.Width = m_resolution.x;
	desc.Height = m_resolution.y;
	desc.Depth = m_resolution.z;
	desc.MipLevels = 1;
	desc.Format = DXGI_FORMAT_R32_FLOAT;
	desc.Usage = D3D11_USAGE_DEFAULT;
	desc.BindFlags = D3D11_BIND_SHADER_RESOURCE;

	for (int i = 0; i < 2; ++i) {
		if (FAILED(pDevice->CreateTexture3D(&desc, NULL, &m_textures[i].dxTexture)))
		{
			std::cerr << "Unable to create 3d volume texture for the heat map" << std::endl;
			return;
		}

		if (FAILED(pDevice->CreateShaderResourceView(m_textures[i].dxTexture, NULL, &m_textures[i].dxSRV)))
		{
			std::cerr << "Unable to create shader resource view for the heat map" << std::endl;
			return;
		}
		cudaSafeCall(cudaGraphicsD3D11RegisterResource(&m_textures[i].cudaResource, m_textures[i].dxTexture, cudaGraphicsRegisterFlagsNone));
	}
}

void HeatMapManager::CopyToRenderTexture(HeatMap::Channel_ptr channel, int slot)
{
	cudaSafeCall(cudaGraphicsMapResources(1, &m_textures[slot].cudaResource));
	cudaArray *cuArray;
	cudaSafeCall(cudaGraphicsSubResourceGetMappedArray(&cuArray, m_textures[slot].cudaResource, 0, 0));

	heatmapKernelCopyChannel(channel->getCudaBuffer(), m_cudaCopyBuffer, m_resolution);
	struct cudaMemcpy3DParms memcpyParams = { 0 };
	memcpyParams.dstArray = cuArray;
	memcpyParams.srcPtr.ptr = m_cudaCopyBuffer;
	memcpyParams.srcPtr.pitch = m_resolution.x * sizeof(int);
	memcpyParams.srcPtr.xsize = m_resolution.x;
	memcpyParams.srcPtr.ysize = m_resolution.y;
	memcpyParams.extent.width = m_resolution.x;
	memcpyParams.extent.height = m_resolution.y;
	memcpyParams.extent.depth = m_resolution.z;
	memcpyParams.kind = cudaMemcpyDeviceToDevice;
	cudaSafeCall(cudaMemcpy3D(&memcpyParams));

	cudaSafeCall(cudaGraphicsUnmapResources(1, &m_textures[slot].cudaResource));

	std::cout << "Heat map copied to DirectX" << std::endl;
}

