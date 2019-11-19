#include "VolumeFilter.h"

#include <cuda_runtime.h>

#include "cudaUtil.h"


// HACK to get stuff to compile...
extern cudaTextureObject_t g_texVolume1;

#include "VolumeFilterKernels.cuh"


//void VolumeFilter::GetMaskedVelocity(const float* d_pDataX, const float* d_pDataY, const float* d_pDataZ, float* d_pOutX, float* d_pOutY, float* d_pOutZ, const tum3D::Vec3i& size, eMeasure measure, float threshold)
//{
//	dim3 blockSize(32, 4);
//	dim3 blockCount((size.x() + blockSize.x - 1) / blockSize.x, (size.y() + blockSize.y - 1) / blockSize.y, (size.z() + blockSize.z - 1) / blockSize.z);
//
//	#define MASKED_VELOCITY(M) maskedVelocityKernel<M><<<blockCount, blockSize>>>(d_pDataX, d_pDataY, d_pDataZ, d_pOutX, d_pOutY, d_pOutZ, size.x(), size.y(), size.z(), threshold)
//	switch(measure)
//	{
//		case MEASURE_VELOCITY				: MASKED_VELOCITY(MEASURE_VELOCITY); break;
//		case MEASURE_VORTICITY				: MASKED_VELOCITY(MEASURE_VORTICITY); break;
//		case MEASURE_LAMBDA2				: MASKED_VELOCITY(MEASURE_LAMBDA2); break;
//		case MEASURE_QHUNT					: MASKED_VELOCITY(MEASURE_QHUNT); break;
//		case MEASURE_DELTACHONG				: MASKED_VELOCITY(MEASURE_DELTACHONG); break;
//		case MEASURE_ENSTROPHY_PRODUCTION	: MASKED_VELOCITY(MEASURE_ENSTROPHY_PRODUCTION); break;
//		case MEASURE_STRAIN_PRODUCTION		: MASKED_VELOCITY(MEASURE_STRAIN_PRODUCTION); break;
//		case MEASURE_SQUARE_ROTATION		: MASKED_VELOCITY(MEASURE_SQUARE_ROTATION); break;
//		case MEASURE_SQUARE_RATE_OF_STRAIN	: MASKED_VELOCITY(MEASURE_SQUARE_RATE_OF_STRAIN); break;
//		case MEASURE_TRACE_JJT				: MASKED_VELOCITY(MEASURE_TRACE_JJT); break;
//		case MEASURE_PVA					: MASKED_VELOCITY(MEASURE_PVA); break;
//	}
//	#undef MASKED_VELOCITY
//	cudaCheckMsg("maskedVelocityKernel execution failed");
//}

//void VolumeFilter::GetMaskedJacobian(const float* d_pDataX, const float* d_pDataY, const float* d_pDataZ, float* d_pOut0, float* d_pOut1, float* d_pOut2, float* d_pOut3, float* d_pOut4, float* d_pOut5, float* d_pOut6, float* d_pOut7, float* d_pOut8, const tum3D::Vec3i& size, eMeasure measure, float threshold)
//{
//	dim3 blockSize(32, 4);
//	dim3 blockCount((size.x() + blockSize.x - 1) / blockSize.x, (size.y() + blockSize.y - 1) / blockSize.y, (size.z() + blockSize.z - 1) / blockSize.z);
//
//	#define MASKED_JACOBIAN(M) maskedJacobianKernel<M><<<blockCount, blockSize>>>(d_pDataX, d_pDataY, d_pDataZ, d_pOut0, d_pOut1, d_pOut2, d_pOut3, d_pOut4, d_pOut5, d_pOut6, d_pOut7, d_pOut8, size.x(), size.y(), size.z(), threshold)
//	switch(measure)
//	{
//		case MEASURE_VELOCITY				: MASKED_JACOBIAN(MEASURE_VELOCITY); break;
//		case MEASURE_VORTICITY				: MASKED_JACOBIAN(MEASURE_VORTICITY); break;
//		case MEASURE_LAMBDA2				: MASKED_JACOBIAN(MEASURE_LAMBDA2); break;
//		case MEASURE_QHUNT					: MASKED_JACOBIAN(MEASURE_QHUNT); break;
//		case MEASURE_DELTACHONG				: MASKED_JACOBIAN(MEASURE_DELTACHONG); break;
//		case MEASURE_ENSTROPHY_PRODUCTION	: MASKED_JACOBIAN(MEASURE_ENSTROPHY_PRODUCTION); break;
//		case MEASURE_STRAIN_PRODUCTION		: MASKED_JACOBIAN(MEASURE_STRAIN_PRODUCTION); break;
//		case MEASURE_SQUARE_ROTATION		: MASKED_JACOBIAN(MEASURE_SQUARE_ROTATION); break;
//		case MEASURE_SQUARE_RATE_OF_STRAIN	: MASKED_JACOBIAN(MEASURE_SQUARE_RATE_OF_STRAIN); break;
//		case MEASURE_TRACE_JJT				: MASKED_JACOBIAN(MEASURE_TRACE_JJT); break;
//		case MEASURE_PVA					: MASKED_JACOBIAN(MEASURE_PVA); break;
//	}
//	#undef MASKED_JACOBIAN
//	cudaCheckMsg("maskedVelocityKernel execution failed");
//}


void VolumeFilter::Filter(EFilterDirection dir, int radius, const ChannelData& data, const tum3D::Vec3ui& size, int overlap, int sizeLeft, int sizeRight)
{
	switch(dir)
	{
		case DIR_X:
		{
			dim3 blockSize(32, 4);
			dim3 blockCount((size.y() + blockSize.x - 1) / blockSize.x, (size.z() + blockSize.y - 1) / blockSize.y);
			filterXKernel<<<blockCount, blockSize>>>(radius, data.d_pData, data.d_pOut, size.x(), size.y(), size.z(), 2 * overlap, data.d_pLeft, sizeLeft, data.d_pRight, sizeRight);
			cudaCheckMsg("filterXKernel execution failed");
			break;
		}
		case DIR_Y:
		{
			dim3 blockSize(32, 4);
			dim3 blockCount((size.x() + blockSize.x - 1) / blockSize.x, (size.z() + blockSize.y - 1) / blockSize.y);
			filterYKernel<<<blockCount, blockSize>>>(radius, data.d_pData, data.d_pOut, size.x(), size.y(), size.z(), 2 * overlap, data.d_pLeft, sizeLeft, data.d_pRight, sizeRight);
			cudaCheckMsg("filterYKernel execution failed");
			break;
		}
		case DIR_Z:
		{
			dim3 blockSize(32, 4);
			dim3 blockCount((size.x() + blockSize.x - 1) / blockSize.x, (size.y() + blockSize.y - 1) / blockSize.y);
			filterZKernel<<<blockCount, blockSize>>>(radius, data.d_pData, data.d_pOut, size.x(), size.y(), size.z(), 2 * overlap, data.d_pLeft, sizeLeft, data.d_pRight, sizeRight);
			cudaCheckMsg("filterZKernel execution failed");
			break;
		}
	}
	//cudaError_t e = cudaDeviceSynchronize();
	//if(e != cudaSuccess) printf("ERROR: %i\n", int(e));
}
