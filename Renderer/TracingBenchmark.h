#ifndef __TUM3D__TRACING_BENCHMARK_H__
#define __TUM3D__TRACING_BENCHMARK_H__


#include <global.h>

#include <vector>

#include <Vec.h>

#include "ParticleTraceParams.h"
#include "TracingCommon.h"


class TracingBenchmark
{
public:
	TracingBenchmark();
	~TracingBenchmark();

	bool RunBenchmark(const ParticleTraceParams& params, uint lineCountMin, uint lineCountMax, uint passCount, int cudaDevice = -1);

private:
	static std::vector<float>          GenerateFlowData(const tum3D::Vec3i& size);
	static std::vector<LineCheckpoint> GenerateSeeds(uint count, float deltaTime, const tum3D::Vec3f& boxMin, const tum3D::Vec3f& boxMax);
};


#endif
