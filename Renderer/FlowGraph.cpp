#include "FlowGraph.h"

#include <cassert>
#include <fstream>
#include <random>

#include <cuda_runtime.h>

#include <TimerCPU.h>

#include <cudaUtil.h>
#include "cudaTum3D.h"

#include "BrickUpload.h"
#include "Integrator.h"
#include "TracingCommon.h"

#include "BrickIndexGPU.h"
#include "BrickRequestsGPU.h"
#include "VolumeInfoGPU.h"

using namespace tum3D;


FlowGraph::BrickInfo::BrickInfo()
{
	memset(OutFreq, 0, sizeof(OutFreq));
}


FlowGraph::FlowGraph()
	: m_pVolume(nullptr)
{
}

FlowGraph::~FlowGraph()
{
	assert(m_pVolume == nullptr);
}


void FlowGraph::Init(TimeVolume& volume)
{
	Shutdown();

	m_pVolume = &volume;

	// get list of bricks, sort into default order
	Vec3i brickCount = m_pVolume->GetBrickCount();
	std::vector<TimeVolumeIO::Brick>& bricks = m_pVolume->GetNearestTimestep().bricks;
	m_bricks.resize(bricks.size());
	for(size_t i = 0; i < bricks.size(); i++)
	{
		const Vec3i& brickPos = bricks[i].GetSpatialIndex();
		size_t index = brickPos.x() + brickCount.x() * (brickPos.y() + brickCount.y() * brickPos.z());
		m_bricks[index] = &bricks[i];
	}
}

void FlowGraph::Shutdown()
{
	Clear();

	m_bricks.clear();
	m_pVolume = nullptr;
}


void FlowGraph::Build(GPUResources* pCompressShared, CompressVolumeResources* pCompressVolume, uint particleCountPerBrick, const ParticleTraceParams& params, const std::string& filenameOut)
{
	if(m_pVolume == nullptr)
	{
		printf("FlowGraph::Build: don't have volume\n");
		return;
	}

	Clear();


	int brickSize = m_pVolume->GetBrickSizeWithOverlap();

	Vec3i brickCount = m_pVolume->GetBrickCount();
	uint brickCountTotal = brickCount.volume();


	TimerCPU timer;

	if(m_pVolume->IsCompressed())
	{
		printf("FlowGraph::Build: Loading data...");
		// load all bricks from disk
		timer.Start();
		m_pVolume->LoadNearestTimestep();
		timer.Stop();
		printf(" done in %.2f s.\n\n", timer.GetElapsedTimeMS() / 1000.0f);
	}


	timer.Start();

	// init result set
	m_brickInfos.resize(brickCountTotal);


	// build array of random seeds (in [0,1]^3)
	std::vector<SimpleParticleVertex> seeds(particleCountPerBrick);
	std::mt19937 engine;
	std::uniform_real_distribution<float> random;
	for(size_t i = 0; i < seeds.size(); i++)
	{
		seeds[i].Position.x = random(engine);
		seeds[i].Position.y = random(engine);
		seeds[i].Position.z = random(engine);
		seeds[i].Time = 0.0f;
	}


	// allocate brick index
	uint2* pBrickToSlot = nullptr;
	cudaSafeCall(cudaMallocHost(&pBrickToSlot, brickCountTotal * sizeof(uint2)));
	for(uint i = 0; i < brickCountTotal; i++)
	{
		pBrickToSlot[i].x = BrickIndexGPU::INVALID;
		pBrickToSlot[i].y = BrickIndexGPU::INVALID;
	}


	// fill GPU info structs
	VolumeInfoGPU volumeInfoGPU;
	volumeInfoGPU.Fill(m_pVolume->GetInfo());
	volumeInfoGPU.Upload(false);

	BrickIndexGPU brickIndexGPU;
	brickIndexGPU.Init();
	brickIndexGPU.Allocate(false, brickCountTotal, make_uint2(1, 1));
	brickIndexGPU.Upload(false);

	BrickRequestsGPU brickRequestsGPU;
	brickRequestsGPU.Init();
	brickRequestsGPU.Allocate(false, brickCountTotal);
	brickRequestsGPU.Clear(false, brickCountTotal);
	brickRequestsGPU.Upload(false);

	cudaSafeCall(cudaDeviceSynchronize());


	// allocate GPU buffers
	BrickSlot brickSlot;
	brickSlot.Create(brickSize, m_pVolume->GetChannelCount());
	std::vector<float*> dpBrickBuffer(m_pVolume->GetChannelCount());
	int brickSizeBytePerChannel = brickSize * brickSize * brickSize * sizeof(float);
	for(size_t channel = 0; channel < dpBrickBuffer.size(); channel++)
	{
		cudaSafeCall(cudaMalloc(&dpBrickBuffer[channel], brickSizeBytePerChannel));
	}
	SimpleParticleVertex* dpParticles = nullptr;
	cudaSafeCall(cudaMalloc(&dpParticles, particleCountPerBrick * sizeof(SimpleParticleVertex)));


	// temporary: file output
	FILE* file = nullptr;
	if(!filenameOut.empty())
	{
		fopen(filenameOut.c_str(), "w");
	}

	if(file)
	{
		fprintf(file, "FlowGraph for %s, time step %i\n", m_pVolume->GetFilename().c_str(), m_pVolume->GetCurNearestTimestepIndex());
		fprintf(file, "%i particles per brick\n", particleCountPerBrick);
		fprintf(file, "%i advection steps with dt = %.5f\n", params.m_advectStepsPerRound, params.m_advectDeltaT);
		fprintf(file, "\n\n");
	}


	// trace particles in each brick
	Integrator integrator;
	integrator.Create();
	integrator.SetVolumeInfo(m_pVolume->GetInfo());
	// particle buffer: filled per brick
	std::vector<SimpleParticleVertex> particles(particleCountPerBrick);
	for(uint linearBrickIndex = 0; linearBrickIndex < brickCountTotal; linearBrickIndex++)
	{
		Vec3i brickIndex(
			linearBrickIndex % brickCount.x(),
			(linearBrickIndex / brickCount.x()) % brickCount.y(),
			linearBrickIndex / (brickCount.x() * brickCount.y()));

		printf("FlowGraph::Build: processing brick %i %i %i\n", brickIndex.x(), brickIndex.y(), brickIndex.z());

		Vec3f brickBoxMin, brickBoxMax;
		m_pVolume->GetBrickBoxWorld(brickIndex, brickBoxMin, brickBoxMax);
		float brickSize = m_pVolume->GetBrickSizeWorld();

		// get brick
		TimeVolumeIO::Brick* pBrick = m_bricks[linearBrickIndex];

		// load brick from disk if necessary
		if(!pBrick->IsLoaded())
		{
			while(!m_pVolume->EnqueueLoadBrickData(*pBrick))
			{
				printf("FlowGraph::Build: EnqueueLoadBrickData failed?! Trying again...\n");
				m_pVolume->WaitForAllIO();
			}
			m_pVolume->WaitForAllIO();
		}
		assert(pBrick->IsLoaded());

		m_pVolume->UnloadLRUBricks();

		// update brick index
		pBrickToSlot[linearBrickIndex].x = 0;
		pBrickToSlot[linearBrickIndex].y = 0;
		brickIndexGPU.Update(false, pBrickToSlot, nullptr, nullptr);

		// upload brick to GPU
		UploadBrick(pCompressShared, pCompressVolume, m_pVolume->GetInfo(), *pBrick, dpBrickBuffer.data(), &brickSlot);

		// transform seeds into world space coords within current brick
		for(uint i = 0; i < particleCountPerBrick; i++)
		{
			particles[i] = seeds[i];
			particles[i].Position = make_float3(brickBoxMin) + particles[i].Position * brickSize;
		}
		cudaSafeCall(cudaMemcpy(dpParticles, particles.data(), particles.size() * sizeof(SimpleParticleVertex), cudaMemcpyHostToDevice));


		// integrate!
		integrator.IntegrateSimpleParticles(brickSlot, dpParticles, uint(particles.size()), params);


		// download particles
		cudaSafeCall(cudaMemcpy(particles.data(), dpParticles, particles.size() * sizeof(SimpleParticleVertex), cudaMemcpyDeviceToHost));


		// count how many particles went into which neighbor brick
		std::vector<uint> outFreq(DIR_COUNT);
		Vec3f brickSizeRatio = (brickBoxMax - brickBoxMin) / brickSize;
		uint particleCountThisBrick = 0;
		for(uint i = 0; i < particleCountPerBrick; i++)
		{
			// ignore particles that were spawned outside the brick
			//TODO does this make sense..?
			if(seeds[i].Position.x > brickSizeRatio.x() ||
			   seeds[i].Position.y > brickSizeRatio.y() ||
			   seeds[i].Position.z > brickSizeRatio.z())
			{
				continue;
			}
			++particleCountThisBrick;

			Vec3f pos = Vec3f(particles[i].Position.x, particles[i].Position.y, particles[i].Position.z);

			if(pos.x() < brickBoxMin.x()) outFreq[DIR_NEG_X]++;
			if(pos.y() < brickBoxMin.y()) outFreq[DIR_NEG_Y]++;
			if(pos.z() < brickBoxMin.z()) outFreq[DIR_NEG_Z]++;
			if(pos.x() > brickBoxMax.x()) outFreq[DIR_POS_X]++;
			if(pos.y() > brickBoxMax.y()) outFreq[DIR_POS_Y]++;
			if(pos.z() > brickBoxMax.z()) outFreq[DIR_POS_Z]++;
		}

		// normalize counts and store
		if(file)
		{
			fprintf(file, "BRICK %i %i %i :", brickIndex.x(), brickIndex.y(), brickIndex.z());
		}
		BrickInfo& brickInfo = m_brickInfos[linearBrickIndex];
		float sum = 0.0f;
		for(uint i = 0; i < DIR_COUNT; i++)
		{
			float ratio = float(outFreq[i]) / float(particleCountThisBrick);
			brickInfo.OutFreq[i] = ratio;
			if(file)
			{
				fprintf(file, " %.3f", ratio);
			}
			sum += ratio;
		}
		if(file)
		{
			fprintf(file, "   %.3f\n", sum);
		}


		// clear our entry in the brick index again
		cudaSafeCall(cudaDeviceSynchronize());
		pBrickToSlot[linearBrickIndex].x = BrickIndexGPU::INVALID;
		pBrickToSlot[linearBrickIndex].y = BrickIndexGPU::INVALID;
	}

	integrator.Release();

	if(file)
	{
		fclose(file);
	}


	// release GPU resources
	for(size_t channel = 0; channel < dpBrickBuffer.size(); channel++)
	{
		cudaSafeCall(cudaFree(dpBrickBuffer[channel]));
	}
	cudaSafeCall(cudaFree(dpParticles));
	brickSlot.Release();


	brickRequestsGPU.Deallocate();
	brickIndexGPU.Deallocate();

	cudaSafeCall(cudaFreeHost(pBrickToSlot));
	pBrickToSlot = nullptr;


	timer.Stop();
	printf("FlowGraph::Build: Done in %.2f s.\n\n", timer.GetElapsedTimeMS() / 1000.0f);
}

void FlowGraph::Clear()
{
	m_brickInfos.clear();
}

bool FlowGraph::IsBuilt() const
{
	return m_pVolume != nullptr && !m_brickInfos.empty();
}


bool FlowGraph::SaveToFile(const std::string& filename) const
{
	if(!IsBuilt())
	{
		printf("FlowGraph::SaveToFile: flow graph isn't built!\n");
		return false;
	}

	std::ofstream file(filename.c_str(), std::ofstream::binary);
	if(!file.good())
	{
		printf("FlowGraph::SaveToFile: can't create output file %s!\n", filename.c_str());
		return false;
	}

	uint brickCount = uint(m_brickInfos.size());
	file.write((const char*)&brickCount, sizeof(brickCount));
	file.write((const char*)m_brickInfos.data(), m_brickInfos.size() * sizeof(BrickInfo));

	return true;
}

bool FlowGraph::LoadFromFile(const std::string& filename)
{
	std::ifstream file(filename.c_str(), std::ofstream::binary);
	if(!file.good())
	{
		printf("FlowGraph::LoadFromFile: can't open input file %s!\n", filename.c_str());
		return false;
	}

	uint brickCount = 0;
	file.read((char*)&brickCount, sizeof(brickCount));
	if(brickCount != m_bricks.size())
	{
		printf("FlowGraph::LoadFromFile: brickCount mismatch!\n", filename.c_str());
		return false;
	}

	m_brickInfos.resize(m_bricks.size());
	file.read((char*)m_brickInfos.data(), m_brickInfos.size() * sizeof(BrickInfo));

	return true;
}


const FlowGraph::BrickInfo& FlowGraph::GetBrickInfo(const Vec3i& brickIndex) const
{
	assert(IsBuilt());

	return GetBrickInfo(m_pVolume->GetBrickLinearIndex(brickIndex));
}

const FlowGraph::BrickInfo& FlowGraph::GetBrickInfo(uint brickLinearIndex) const
{
	assert(IsBuilt());

	return m_brickInfos[brickLinearIndex];
}
