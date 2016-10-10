#ifndef __TUM3D__CLEAR_CUDA_ARRAY_H__
#define __TUM3D__CLEAR_CUDA_ARRAY_H__


#include "global.h"

#include <cuda_runtime.h>


// cudaArray must have been created with cudaArraySurfaceLoadStore!

void clearCudaArray2Duchar4(cudaArray* pArray, uint width, uint height);


#endif
