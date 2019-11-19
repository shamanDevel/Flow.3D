#include "Raycaster.h"

#include <cassert>

#include <cudaUtil.h>

#include "cudaTum3D.h"

#include "BoxUtils.h"
#include "BrickSlot.h"
#include "RaycastParams.h"

#include "RaycasterKernels.h"

using namespace tum3D;


cudaTextureObject_t g_texVolume1;
cudaTextureObject_t g_texVolume2;
cudaTextureObject_t g_texVolume3;
cudaTextureObject_t g_texFeatureVolume;
cudaTextureObject_t g_texTransferFunction;

surface<void, cudaSurfaceType2D> g_surfTarget;
surface<void, cudaSurfaceType2D> g_surfDepth;



__constant__ ProjectionParamsGPU c_projParams;

__constant__ RaycastParamsGPU c_raycastParams;




Raycaster::Raycaster()
	: m_isCreated(false)
	, m_brickSizeWorld(0.0f), m_brickOverlapWorld(0.0f), m_gridSpacing(1.0f)
	, md_pProjParamsCyclop(nullptr), md_pProjParamsLeft(nullptr), md_pProjParamsRight(nullptr)
{
}

Raycaster::~Raycaster()
{
	assert(!IsCreated());
}

bool Raycaster::Create()
{
	if(IsCreated()) return true;

	cudaSafeCall(cudaMalloc2(&md_pProjParamsCyclop, sizeof(ProjectionParamsGPU)));
	cudaSafeCall(cudaMalloc2(&md_pProjParamsLeft,   sizeof(ProjectionParamsGPU)));
	cudaSafeCall(cudaMalloc2(&md_pProjParamsRight,  sizeof(ProjectionParamsGPU)));

	cudaSafeCall(cudaHostRegister(&m_raycastParamsGPU, sizeof(m_raycastParamsGPU), cudaHostRegisterDefault));
	cudaSafeCall(cudaEventCreate(&m_raycastParamsUploadEvent, cudaEventDisableTiming));
	cudaSafeCall(cudaEventRecord(m_raycastParamsUploadEvent));

	m_isCreated = true;

	return true;
}

void Raycaster::Release()
{
	if(!IsCreated()) return;

	m_isCreated = false;

	cudaSafeCall(cudaEventDestroy(m_raycastParamsUploadEvent));
	m_raycastParamsUploadEvent = 0;
	cudaSafeCall(cudaHostUnregister(&m_raycastParamsGPU));

	cudaSafeCall(cudaFree(md_pProjParamsRight));
	cudaSafeCall(cudaFree(md_pProjParamsLeft));
	cudaSafeCall(cudaFree(md_pProjParamsCyclop));
	md_pProjParamsRight  = nullptr;
	md_pProjParamsLeft   = nullptr;
	md_pProjParamsCyclop = nullptr;
}


void Raycaster::SetParams(const ProjectionParams& projParams, const StereoParams& stereoParams, const Range1D& range)
{
	m_projParams = projParams;
	m_stereoParams = stereoParams;
	m_range = range;

	if(IsCreated())
	{
		// update constant params
		ProjectionParamsGPU projParamsGPU;

		//projParamsGPU.aspectRatio = m_projParams.m_aspectRatio;
		//projParamsGPU.tanHalfFovy = tan(0.5f * m_projParams.m_fovy);
		float frustum[6];
		m_projParams.GetFrustumParams(frustum, EYE_CYCLOP, 0.0f, m_range);
		float near = frustum[4];
		projParamsGPU.left   = frustum[0] / near;
		projParamsGPU.right  = frustum[1] / near;
		projParamsGPU.bottom = frustum[2] / near;
		projParamsGPU.top    = frustum[3] / near;
		projParamsGPU.eyeOffset = 0.0f;
		projParamsGPU.imageWidth  = m_projParams.GetImageWidth (m_range);
		projParamsGPU.imageHeight = m_projParams.GetImageHeight(m_range);
		projParamsGPU.depthParams = make_float4(m_projParams.m_near, m_projParams.m_far, m_projParams.m_far * m_projParams.m_near, m_projParams.m_far - m_projParams.m_near);
		cudaSafeCall(cudaMemcpy(md_pProjParamsCyclop, &projParamsGPU, sizeof(projParamsGPU), cudaMemcpyHostToDevice));

		// adjust image height for stereo
		projParamsGPU.imageHeight /= 2;

		m_projParams.GetFrustumParams(frustum, EYE_LEFT, m_stereoParams.m_eyeDistance, m_range);
		projParamsGPU.left   = frustum[0] / near;
		projParamsGPU.right  = frustum[1] / near;
		projParamsGPU.bottom = frustum[2] / near;
		projParamsGPU.top    = frustum[3] / near;
		projParamsGPU.eyeOffset = 0.5f * m_stereoParams.m_eyeDistance / m_projParams.m_near; // kernels assume near plane at 1, so "normalize" here
		cudaSafeCall(cudaMemcpy(md_pProjParamsLeft,  &projParamsGPU, sizeof(projParamsGPU), cudaMemcpyHostToDevice));

		m_projParams.GetFrustumParams(frustum, EYE_RIGHT, m_stereoParams.m_eyeDistance, m_range);
		projParamsGPU.left   = frustum[0] / near;
		projParamsGPU.right  = frustum[1] / near;
		projParamsGPU.bottom = frustum[2] / near;
		projParamsGPU.top    = frustum[3] / near;
		projParamsGPU.eyeOffset = -projParamsGPU.eyeOffset;
		cudaSafeCall(cudaMemcpy(md_pProjParamsRight, &projParamsGPU, sizeof(projParamsGPU), cudaMemcpyHostToDevice));
	}
}


cudaTextureObject_t Raycaster::CreateCudaTextureObject(const cudaArray* array, const RaycastParams& params)
{
	// Create the texture object
	cudaResourceDesc resDesc;
	memset(&resDesc, 0, sizeof(cudaResourceDesc));
	resDesc.resType = cudaResourceTypeArray;
	resDesc.res.array.array = array;
	cudaTextureDesc texDesc;
	memset(&texDesc, 0, sizeof(cudaTextureDesc));
	texDesc.normalizedCoords = false;
	texDesc.filterMode = GetCudaTextureFilterMode(params.m_textureFilterMode);
	texDesc.addressMode[0] = cudaAddressModeClamp;
	texDesc.addressMode[1] = cudaAddressModeClamp;
	texDesc.addressMode[2] = cudaAddressModeClamp;
	texDesc.readMode = cudaReadModeElementType;
	cudaTextureObject_t texture;
	cudaSafeCall(cudaCreateTextureObject(&texture, &resDesc, &texDesc, NULL));
	return texture;
}


bool Raycaster::RenderBrick(
		cudaArray* pTargetArray,
		cudaArray* pDepthArray,
		const RaycastParams& params,
		const Mat4f& view,
		EStereoEye eye,
		cudaArray* pTfArray,
		const std::vector<BrickSlot*>& brickSlots,
		const Vec3f& boxMin,
		const Vec3f& boxMax,
		uint pass)
{
	if(brickSlots.empty())
		return true;

	Mat4f viewInv; invert4x4(view, viewInv);
	Vec4f camPos4 = viewInv.getCol(3);
	Vec3f camPos = camPos4.xyz();

	// get clipped brick bbox
	Vec3f boxMinClipped = boxMin, boxMaxClipped = boxMax;
	if(!ClipBoxAgainstBox(params.m_clipBoxMin, params.m_clipBoxMax, boxMinClipped, boxMaxClipped)) {
		return true;
	}

	// update projection params in constant memory
	switch(eye) {
		case EYE_LEFT:  cudaSafeCall(cudaMemcpyToSymbolAsync(c_projParams, md_pProjParamsLeft,   sizeof(ProjectionParamsGPU), 0, cudaMemcpyDeviceToDevice)); break;
		case EYE_RIGHT: cudaSafeCall(cudaMemcpyToSymbolAsync(c_projParams, md_pProjParamsRight,  sizeof(ProjectionParamsGPU), 0, cudaMemcpyDeviceToDevice)); break;
		default:        cudaSafeCall(cudaMemcpyToSymbolAsync(c_projParams, md_pProjParamsCyclop, sizeof(ProjectionParamsGPU), 0, cudaMemcpyDeviceToDevice)); break;
	}

	// get stereo-adjusted projection params
	ProjectionParams projParams = m_projParams;
	if(eye != EYE_CYCLOP) {
		projParams.m_imageHeight /= 2;
	}

	// compute size on screen
	Vec2i screenMin, screenMax;
	if(!GetBoxScreenExtent(projParams, eye, m_stereoParams.m_eyeDistance, m_range, view, camPos, boxMinClipped, boxMaxClipped, screenMin, screenMax)) {
		return true;
	}

	// right eye renders into lower half of screen
	Vec2i renderTargetOffset(0, 0);
	if(eye == EYE_RIGHT) {
		renderTargetOffset.y() = projParams.m_imageHeight;
	}

	// adjust screen box for multi-pass
	uint passCount = GetNumPassesNeeded(params, view, camPos, boxMinClipped, boxMaxClipped);
	uint lineCount = screenMax.y() - screenMin.y() + 1;
	uint lineCountPerPass = (lineCount + passCount - 1) / passCount;

	//printf("pass %i/%i  before: %i-%i", pass, passCount, screenMin.y(), screenMax.y());

	screenMin.y() += pass * lineCountPerPass;
	screenMax.y() = min(screenMax.y(), screenMin.y() + lineCountPerPass - 1);

	//printf("  after: %i-%i\n", screenMin.y(), screenMax.y());

	// bind GPU resources
	cudaSafeCall(cudaBindSurfaceToArray(g_surfTarget, pTargetArray));
	cudaSafeCall(cudaBindSurfaceToArray(g_surfDepth, pDepthArray));

	g_texVolume1 = CreateCudaTextureObject(brickSlots[0]->GetCudaArray(), params);
	size_t tex1index = min(size_t(1), brickSlots.size() - 1);
	g_texVolume2 = CreateCudaTextureObject(brickSlots[tex1index]->GetCudaArray(), params);
	size_t tex2index = min(size_t(2), brickSlots.size() - 1);
	g_texVolume3 = CreateCudaTextureObject(brickSlots[tex2index]->GetCudaArray(), params);

	if(params.m_measureComputeMode != MEASURE_COMPUTE_ONTHEFLY)
	{
		g_texFeatureVolume = CreateCudaTextureObject(brickSlots[0]->GetCudaArray(), params);
	}

	if(RaycastModeNeedsTransferFunction(params.m_raycastMode))
	{
		cudaResourceDesc resDesc;
		memset(&resDesc, 0, sizeof(cudaResourceDesc));
		resDesc.resType = cudaResourceTypeArray;
		resDesc.res.array.array = pTfArray;
		cudaTextureDesc texDesc;
		memset(&texDesc, 0, sizeof(cudaTextureDesc));
		texDesc.normalizedCoords = true;
		texDesc.filterMode = cudaFilterModeLinear;
		texDesc.addressMode[0] = cudaAddressModeClamp;
		texDesc.readMode = cudaReadModeElementType;
		cudaSafeCall(cudaCreateTextureObject(&g_texTransferFunction, &resDesc, &texDesc, NULL));
	}

	// kernel params
	Vec2i imageSize = screenMax - screenMin + 1;
	Vec3f brickSizeWorldWithOverlap = m_brickSizeWorld + 2.0f * m_brickOverlapWorld;
	Vec3f world2texOffset = -boxMin + m_brickOverlapWorld;
	Vec3f world2texScale = float(brickSlots[0]->GetSize()) / brickSizeWorldWithOverlap;

	//// for analytic fields, clear world2texOffset/Scale so coords stay in world space
	//if(params.m_textureFilterMode >= TEXTURE_ANALYTIC_DOUBLEGYRE)
	//{
	//	world2texOffset = Vec3f(0.0f, 0.0f, 0.0f);
	//	world2texScale = 1.0f;
	//}

	Vec3f voxelSizeWorld = m_brickSizeWorld / float(brickSlots[0]->GetSize());

	float stepSize = voxelSizeWorld.maximum() / params.m_sampleRate;
	float transferOffset = -params.m_transferFunctionRangeMin;
	float transferScale = 1.0f / (params.m_transferFunctionRangeMax - params.m_transferFunctionRangeMin);

	RaycastParamsGPU raycastParamsGPU;
	raycastParamsGPU.gridSpacing = make_float3(m_gridSpacing);
	raycastParamsGPU.view = make_float3x4(view);
	raycastParamsGPU.viewInv = make_float3x4(viewInv);
	raycastParamsGPU.stepSizeWorld = stepSize;
	raycastParamsGPU.measure1 = params.m_measure1;
	raycastParamsGPU.measure2 = params.m_measure2;
	raycastParamsGPU.measureScale1 = params.m_measureScale1;
	raycastParamsGPU.measureScale2 = params.m_measureScale2;
	raycastParamsGPU.density = params.m_density;
	raycastParamsGPU.transferOffset = transferOffset;
	raycastParamsGPU.transferScale = transferScale;
	raycastParamsGPU.tfAlphaScale = params.m_alphaScale;
	raycastParamsGPU.isoValues = make_float3(params.m_isoValue1, params.m_isoValue2, params.m_isoValue3);
	raycastParamsGPU.isoColor1 = make_float4(params.m_isoColor1);
	raycastParamsGPU.isoColor2 = make_float4(params.m_isoColor2);
	raycastParamsGPU.isoColor3 = make_float4(params.m_isoColor3);
	if(memcmp(&m_raycastParamsGPU, &raycastParamsGPU, sizeof(m_raycastParamsGPU)) != 0)
	{
		cudaSafeCall(cudaEventSynchronize(m_raycastParamsUploadEvent));
		m_raycastParamsGPU = raycastParamsGPU;
		cudaSafeCall(cudaMemcpyToSymbolAsync(c_raycastParams, &m_raycastParamsGPU, sizeof(m_raycastParamsGPU), 0, cudaMemcpyHostToDevice));
		cudaSafeCall(cudaEventRecord(m_raycastParamsUploadEvent));
	}

	dim3 blockSize(16, 16);
	dim3 gridSize((imageSize.x() + blockSize.x - 1) / blockSize.x, (imageSize.y() + blockSize.y - 1) / blockSize.y);

	RaycasterKernelParams kernelParams(
		params.m_measure1, params.m_measure2, params.m_measureComputeMode,
		params.m_textureFilterMode,
		params.m_colorMode,
		brickSlots,
		blockSize, gridSize,
		make_int2(screenMin), make_int2(imageSize), make_int2(renderTargetOffset),
		make_float3(boxMinClipped), make_float3(boxMaxClipped),
		make_float3(world2texOffset), make_float3(world2texScale));
	switch(params.m_raycastMode)
	{
		case RAYCAST_MODE_DVR :											raycasterKernelDvr							(kernelParams); break;
		case RAYCAST_MODE_DVR_EE :										raycasterKernelDvrEe						(kernelParams); break;
		case RAYCAST_MODE_ISO :											raycasterKernelIso							(kernelParams); break;
		case RAYCAST_MODE_ISO_SI :										raycasterKernelIsoSi						(kernelParams); break;
		case RAYCAST_MODE_ISO2 :										raycasterKernelIso2							(kernelParams); break;
		case RAYCAST_MODE_ISO2_SI :										raycasterKernelIso2Si						(kernelParams); break;
		//case RAYCAST_MODE_MS2_ISO_CASCADE :								raycasterKernelMs2IsoCascade				(kernelParams); break;
		case RAYCAST_MODE_ISO2_SEPARATE :								raycasterKernelIso2Separate					(kernelParams); break;
		//case RAYCAST_MODE_MS3_ISO_CASCADE :								raycasterKernelMs3IsoCascade				(kernelParams); break;
		//case RAYCAST_MODE_MS3_ISO_SI :									raycasterKernelMs3Sivr						(kernelParams); break;
	}
	cudaCheckMsg("raycast kernel execution failed");

	// unbind GPU resources again
	if(params.m_measureComputeMode != MEASURE_COMPUTE_ONTHEFLY)
	{
		cudaSafeCall(cudaDestroyTextureObject(g_texFeatureVolume));
	}

	if(RaycastModeNeedsTransferFunction(params.m_raycastMode))
	{
		cudaSafeCall(cudaDestroyTextureObject(g_texTransferFunction));
	}

	cudaSafeCall(cudaDestroyTextureObject(g_texVolume3));
	cudaSafeCall(cudaDestroyTextureObject(g_texVolume2));
	cudaSafeCall(cudaDestroyTextureObject(g_texVolume1));


	// was this the last pass?
	return (pass + 1) >= passCount;
}

uint Raycaster::GetNumPassesNeeded(const RaycastParams& params, const Mat4f& view, const Vec3f& camPos, const Vec3f& boxMinClipped, const Vec3f& boxMaxClipped) const
{
	// compute size on screen
	Vec2i screenMin, screenMax;
	if(!GetBoxScreenExtent(m_projParams, EYE_CYCLOP, 0.0f, m_range, view, camPos, boxMinClipped, boxMaxClipped, screenMin, screenMax)) {
		return 0;
	}
	Vec2i screenSize = screenMax - screenMin + 1;

	// compute max number of pixels we want to do in one pass
	// baseline
	uint pixelsMax = 512 * 512;
	// adjust for measure compute mode - precomp is ~5 times faster
	if(params.m_measureComputeMode != MEASURE_COMPUTE_ONTHEFLY) pixelsMax *= 5;
	// adjust for sample rate
	pixelsMax = uint(pixelsMax / params.m_sampleRate);
	// adjust for texture filter
	//TODO we should have a special case for VELOCITY here (no gradient per sample!)
	switch(params.m_textureFilterMode)
	{
		// Cubic needs 8 linear fetches per sample, but gradients need only 3 samples instead of 6
		case TEXTURE_FILTER_CUBIC:     pixelsMax /=  4; break;
		// Catmull-Rom and WENO4 timings are very roughly guessed (and seem to vary quite a lot wrt to linear)
		case TEXTURE_FILTER_CATROM:    pixelsMax /= 12; break;
		case TEXTURE_FILTER_LAGRANGE4: pixelsMax /= 12; break;
		case TEXTURE_FILTER_LAGRANGE6: pixelsMax /= 40; break;
		case TEXTURE_FILTER_LAGRANGE8: pixelsMax /= 100; break;
		//case TEXTURE_FILTER_WENO4:     pixelsMax /= 60; break;
	}
	// TODO: adjust for raycast mode, metric, ...?

	// compute max number of scanlines we want to do in one pass
	uint scanlinesMax = max(pixelsMax / screenSize.x(), 16u);
	scanlinesMax = (scanlinesMax + 8) / 16 * 16;

	return (screenSize.y() + scanlinesMax - 1) / scanlinesMax;
}
