#ifndef __TUM3D__LINE_INFO_GPU__
#define __TUM3D__LINE_INFO_GPU__


#include <global.h>

#include "TracingCommon.h"


struct LineInfoGPU
{
	// Upload this instance to the GPU's global instance in constant memory.
	// Assumes that this instance is in pinned memory.
	// Does not sync on the upload, so don't overwrite any members without syncing first!
	void Upload(bool cpuTracing) const;

	uint lineCount;

	LineCheckpoint* pCheckpoints;

	LineVertex* pVertices;
	uint* pVertexCounts;
	uint vertexStride;

	uint lineLengthMax;
};


#endif
