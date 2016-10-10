#include "TracingTestData.h"

#include <random>

using namespace tum3D;


std::vector<float> GenerateXYCircle(const tum3D::Vec3i& size)
{
	//std::uniform_real_distribution<float> rng;
	//std::mt19937 engine;
	std::vector<float> data(size.volume() * 3);
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
				//tum3D::normalize(vel);
				//float3 vel3 = make_float3(vel);
				//vel3 += GetRandomJitterVector(jitter, rng, engine);
				data[index++] = vel.x();
				data[index++] = vel.y();
				data[index++] = vel.z();
			}
		}
	}
	return data;
}
