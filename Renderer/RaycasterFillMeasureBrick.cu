#include "Raycaster.h"

#include "cudaUtil.h"
#include "cudaTum3D.h"

#include "BrickSlot.h"
#include "RaycastParams.h"

using namespace tum3D;


extern texture<float4, cudaTextureType3D, cudaReadModeElementType> g_texVolume1;
surface<void, cudaSurfaceType3D> g_surfFeatureArray;


#include "Measures.cuh"

template <eMeasure M>
__global__ void genFeatureKernel(int3 featureBrickResolution, float gridSpacing)
{
	uint3 viTxc = make_uint3(uint(blockIdx.x * blockDim.x + threadIdx.x), uint(blockIdx.y * blockDim.y + threadIdx.y), uint(blockIdx.z * blockDim.z + threadIdx.z) );

	if( viTxc.x >= featureBrickResolution.x || viTxc.y >= featureBrickResolution.y || viTxc.z >= featureBrickResolution.z ) 
		return;

	if(	viTxc.x < 2 || viTxc.x >= (featureBrickResolution.x-2) ||
		viTxc.y < 2 || viTxc.y >= (featureBrickResolution.y-2) ||
		viTxc.z < 2 || viTxc.z >= (featureBrickResolution.z-2) )
	{
		surf3Dwrite( 0.0f, g_surfFeatureArray, viTxc.x * sizeof(float), viTxc.y, viTxc.z);
	}
	else
	{
		float3 vfTxc = make_float3( viTxc.x, viTxc.y, viTxc.z );
		// ignore measureScale
		float fMeasure = getMeasure<M,TEXTURE_FILTER_LINEAR,MEASURE_COMPUTE_ONTHEFLY>(g_texVolume1, vfTxc, gridSpacing, 1.0f);
		surf3Dwrite( fMeasure, g_surfFeatureArray, viTxc.x * sizeof(float), viTxc.y, viTxc.z);
	}
}


void Raycaster::FillMeasureBrick(const RaycastParams& params, const BrickSlot& brickSlotVelocity, BrickSlot& brickSlotMeasure)
{
	cudaTextureFilterMode eOldFilterMode = g_texVolume1.filterMode;
	g_texVolume1.filterMode = cudaFilterModePoint;

	cudaSafeCall(cudaBindTextureToArray(g_texVolume1, brickSlotVelocity.GetCudaArray()));
	cudaSafeCall(cudaBindSurfaceToArray(g_surfFeatureArray, brickSlotMeasure.GetCudaArray()));

	Vec3ui size = brickSlotVelocity.GetFilledSize();

	dim3 blockSize(8, 8, 8); 
	dim3 blockCount((size.x() + blockSize.x - 1) / blockSize.x, (size.y() + blockSize.y - 1) / blockSize.y, (size.z() + blockSize.z - 1) / blockSize.z);

#define GEN_FEATURE_KERNEL(measure)\
		genFeatureKernel<measure><<< blockCount, blockSize >>>(make_int3(make_uint3(size)), m_gridSpacing);
#define GEN_FEATURE_CASE(measure) case measure : GEN_FEATURE_KERNEL(measure); break

	switch(params.m_measure1) 
	{
		GEN_FEATURE_CASE(MEASURE_VELOCITY);
		GEN_FEATURE_CASE(MEASURE_VELOCITY_Z);
		GEN_FEATURE_CASE(MEASURE_VORTICITY);
		GEN_FEATURE_CASE(MEASURE_LAMBDA2);
		GEN_FEATURE_CASE(MEASURE_QHUNT);
		GEN_FEATURE_CASE(MEASURE_DELTACHONG);
		GEN_FEATURE_CASE(MEASURE_ENSTROPHY_PRODUCTION);
		GEN_FEATURE_CASE(MEASURE_STRAIN_PRODUCTION);
		GEN_FEATURE_CASE(MEASURE_SQUARE_ROTATION);
		GEN_FEATURE_CASE(MEASURE_SQUARE_RATE_OF_STRAIN);
		GEN_FEATURE_CASE(MEASURE_TRACE_JJT);
		GEN_FEATURE_CASE(MEASURE_PVA);
	} 
	cudaCheckMsg("genFeatureKernel execution failed");

#undef GEN_FEATURE_CASE
#undef GEN_FEATURE_KERNEL

	cudaSafeCall(cudaUnbindTexture(g_texVolume1));
	g_texVolume1.filterMode = eOldFilterMode;

	//if(size.volume()==128*128*128) {
	//	std::vector<float> data(128*128*128);
	//	cudaMemcpy3DParms params = {};
	//	params.srcArray = (cudaArray*)brickSlotMeasure.GetCudaArray();
	//	params.dstPtr = make_cudaPitchedPtr(data.data(), 128*sizeof(float), 128, 128);
	//	params.extent = make_cudaExtent(128, 128, 128);
	//	params.kind = cudaMemcpyDeviceToHost;
	//	cudaSafeCall(cudaMemcpy3D(&params));
	//	int a=0;
	//}
}
