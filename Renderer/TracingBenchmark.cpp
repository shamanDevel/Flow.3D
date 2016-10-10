#include "TracingBenchmark.h"

#include <cassert>
#include <limits>
#include <random>

#include <cuda_runtime.h>
#include <cuda_profiler_api.h>

#include <cudaUtil.h>

#include "cudaTum3D.h"

#include "BrickSlot.h"
#include "Integrator.h"

using namespace tum3D;


TracingBenchmark::TracingBenchmark()
{
}

TracingBenchmark::~TracingBenchmark()
{
}


bool TracingBenchmark::RunBenchmark(const ParticleTraceParams& params, uint lineCountMin, uint lineCountMax, uint passCount, int cudaDevice)
{
	if(cudaDevice >= 0) {
		cudaSafeCall(cudaSetDevice(cudaDevice));
	}

	cudaSafeCall(cudaGetDevice(&cudaDevice));
	cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop, cudaDevice);
	printf("Tracing Benchmark running on device %i (%s)\n", cudaDevice, prop.name);

	// brick and volume size: hardcoded for now...
	uint brickSizeVoxels = 256;
	float brickSizeWorld = 248.0f / 1024.0f;
	float brickOverlapWorld = 4.0f / 1024.0f;

	Vec3i brickSizeVoxels3(brickSizeVoxels, brickSizeVoxels, brickSizeVoxels);
	Vec3f brickBoxMin(0.0f, 0.0f, 0.0f);
	Vec3f brickBoxMax(brickSizeWorld, brickSizeWorld, brickSizeWorld);

	// generate data and seeds
	std::vector<float>          brickData   = GenerateFlowData(brickSizeVoxels3);
	std::vector<LineCheckpoint> seeds       = GenerateSeeds(lineCountMax, params.m_advectDeltaT, brickBoxMin, brickBoxMax);

	// alloc buffers and upload seeds to GPU
	LineCheckpoint* dpCheckpoints = nullptr;
	cudaSafeCall(cudaMalloc(&dpCheckpoints, seeds.size() * sizeof(LineCheckpoint)));
	cudaSafeCall(cudaMemcpy(dpCheckpoints, seeds.data(), seeds.size() * sizeof(LineCheckpoint), cudaMemcpyHostToDevice));

	uint vertexCountTotal = lineCountMax * params.m_advectStepsPerRound;

	// alloc line buffers
	LineVertex* dpVertices = nullptr;
	cudaSafeCall(cudaMalloc(&dpVertices, vertexCountTotal * sizeof(LineVertex)));
	uint* dpVertexCounts = nullptr;
	cudaSafeCall(cudaMalloc(&dpVertexCounts, lineCountMax * sizeof(uint)));

	// create and fill brick slot
	BrickSlot brickSlot;
	if(!brickSlot.Create(brickSizeVoxels, 4))
	{
		return false;
	}
	brickSlot.Fill(brickData.data(), Vec3ui(brickSizeVoxels, brickSizeVoxels, brickSizeVoxels));

	// timing events
	cudaEvent_t start, end;
	cudaSafeCall(cudaEventCreate(&start));
	cudaSafeCall(cudaEventCreate(&end));

	// integrate!
	Integrator integrator;
	integrator.Create();
	// FIXME integrator.SetVolumeInfo !!!
	//integrator.SetBrickSizeWorld(brickSizeWorld);
	//integrator.SetBrickOverlapWorld(brickOverlapWorld);
	//integrator.SetVelocityScaling(2.0f / (m_pVolume->GetVolumeSize().maximum() * m_pVolume->GetInfo().GetGridSpacing()));

	struct Stats
	{
		Stats(uint count, float total, float perPass, float perLine)
			: LineCount(count), TotalMS(total), PerPassMS(perPass), PerLineUS(perLine) {}

		uint LineCount;
		float TotalMS;
		float PerPassMS;
		float PerLineUS;
	};
	std::vector<Stats> stats;

	LineInfo lineInfo(0.0f, 0, dpCheckpoints, dpVertices, dpVertexCounts, params.m_advectStepsPerRound);
	for(uint lineCount = lineCountMin; lineCount <= lineCountMax; lineCount *= 2)
	{
		lineInfo.lineCount = lineCount;

		cudaSafeCall(cudaEventRecord(start));
		cudaSafeCall(cudaProfilerStart());

		for(uint pass = 0; pass < passCount; pass++)
		{
			cudaSafeCall(cudaMemsetAsync(dpVertexCounts, 0, lineCountMax * sizeof(uint)));
			integrator.IntegrateLines(brickSlot, lineInfo, params);
		}

		cudaSafeCall(cudaProfilerStop());
		cudaSafeCall(cudaEventRecord(end));
		cudaSafeCall(cudaEventSynchronize(end));
		float timeTotalMS = 0.0f;
		cudaSafeCall(cudaEventElapsedTime(&timeTotalMS, start, end));

		float timePerPassMS = timeTotalMS / float(passCount);
		float timePerLineUS = timePerPassMS * 1000.0f / float(lineCount);

		stats.push_back(Stats(lineCount, timeTotalMS, timePerPassMS, timePerLineUS));

	}
	printf("Results (avg of %i runs):\n", passCount);
	printf("Count    Total/ms  PerLine/us   Wasted/ms  PerLine/us  Eff/%%\n");
	float bestPerLineUS = std::numeric_limits<float>::max();
	for(size_t i = 0; i < stats.size(); i++) {
		bestPerLineUS = min(bestPerLineUS, stats[i].PerLineUS);
	}
	for(size_t i = 0; i < stats.size(); i++) {
		const Stats& s = stats[i];
		float wastedMS = float(s.LineCount) * (s.PerLineUS - bestPerLineUS) / 1000.0f;
		float wastedPerLineUS = 1000.0f * wastedMS / float(s.LineCount);
		float efficiency = 100.0f * (1.0f - wastedMS / s.PerPassMS);
		printf("%5u %11.2f %11.2f %11.2f %11.2f %6.1f\n", s.LineCount, s.PerPassMS, s.PerLineUS, wastedMS, wastedPerLineUS, efficiency);
	}
	printf("\n");

	// cleanup
	integrator.Release();

	brickSlot.Release();

	cudaSafeCall(cudaEventDestroy(end));
	cudaSafeCall(cudaEventDestroy(start));

	cudaSafeCall(cudaFree(dpVertexCounts));
	cudaSafeCall(cudaFree(dpVertices));
	cudaSafeCall(cudaFree(dpCheckpoints));

	return true;
}

std::vector<float> TracingBenchmark::GenerateFlowData(const Vec3i& size)
{
	float jitter = 0.1f;
	std::uniform_real_distribution<float> rng;
	std::mt19937 engine;
	std::vector<float> data(size.volume() * 4);
	uint index = 0;
	for(int z = 0; z < size.z(); z++)
	{
		for(int y = 0; y < size.y(); y++)
		{
			for(int x = 0; x < size.x(); x++)
			{
				Vec3f pos = Vec3f(float(x), float(y), float(z)) / Vec3f(size - 1);
				pos = pos * 2.0f - 1.0f;
				// circular flow in the xy plane + random jitter
				Vec3f vel(-pos.y(), pos.x(), 0.0f);
				tum3D::normalize(vel);
				float3 vel3 = make_float3(vel);
				vel3 += GetRandomJitterVector(jitter, rng, engine);
				data[index++] = vel3.x;
				data[index++] = vel3.y;
				data[index++] = vel3.z;
				data[index++] = 0.0f;
			}
		}
	}
	return data;
}

std::vector<LineCheckpoint> TracingBenchmark::GenerateSeeds(uint count, float deltaTime, const Vec3f& boxMin, const Vec3f& boxMax)
{
	float3 boxMin3 = make_float3(boxMin);
	float3 boxMax3 = make_float3(boxMax);

	std::vector<LineCheckpoint> seeds(count);
	std::uniform_real_distribution<float> rng;
	std::mt19937 engine;
	for(uint i = 0; i < count; i++)
	{
		seeds[i].Position = GetRandomVectorInBox(boxMin3, boxMax3, rng, engine);
		seeds[i].Time = 0.0f;
		seeds[i].Normal = make_float3(0.0f, 0.0f, 0.0f);
		seeds[i].DeltaT = deltaTime;
	}

	return seeds;
}
