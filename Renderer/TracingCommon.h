#ifndef __TUM3D__TRACING_COMMON_H__
#define __TUM3D__TRACING_COMMON_H__


#include <global.h>

#include <random>

#include <Vec.h>

#include "cutil_math.h"


struct SimpleParticleVertex
{
	float3 Position;
	float  Time;
};


struct BallVertex
{
	float3   Position;
};


struct LineCheckpoint
{
	float3 Position;
	float  Time; // for stream lines: age. for path lines: actual time.
	float3 Normal; //TODO make scalar?
	float  DeltaT;

	uint StepsTotal;
	uint StepsAccepted;
};

struct LineVertex
{
	float3   Position;
	float    Time;
	float3   Normal; //TODO make scalar?
	float3   Velocity;
	uint     LineID;
	float3   SeedPosition;
	//float3x3 Jacobian;
};

struct LineInfo
{
	LineInfo(float lineSeedTime, uint lineCount, LineCheckpoint* dpCheckpoints, LineVertex* dpVertices, uint* dpVertexCounts, uint lineVertexStride)
		: lineSeedTime(lineSeedTime), lineCount(lineCount), dpCheckpoints(dpCheckpoints), dpVertices(dpVertices), dpVertexCounts(dpVertexCounts), lineVertexStride(lineVertexStride) {}

	float           lineSeedTime; // 0 for stream lines!

	uint            lineCount;

	// current state of each line, indexed by lineIndex
	// for iterative tracing (particles) this is the seed
	LineCheckpoint* dpCheckpoints;

	// output: line vertices, indexed by lineIndex * lineLength + vertexIndex
	LineVertex*     dpVertices; 
	//TODO move vertexcounts into checkpoints?
	uint*           dpVertexCounts; // current number of vertices per line, indexed by lineIndex
	uint            lineVertexStride; // stride in dpVertices (= max number of vertices per line)
};



//struct BrickTraceStats
//{
//	BrickTraceStats() { memset(this, 0, sizeof(*this)); }
//
//	uint CountTotal;
//	uint CountStaying;
//	uint CountGoingLeft;
//	uint CountGoingRight;
//	uint CountGoingDown;
//	uint CountGoingUp;
//	uint CountGoingBack;
//	uint CountGoingFront;
//
//	uint CountLived() const
//	{
//		return CountStaying +
//			CountGoingLeft + CountGoingRight +
//			CountGoingDown + CountGoingUp +
//			CountGoingBack + CountGoingFront;
//	}
//
//	float ChanceStaying()    const { return float(CountStaying)    / float(CountTotal); }
//	float ChanceGoingLeft()  const { return float(CountGoingLeft)  / float(CountTotal); }
//	float ChanceGoingRight() const { return float(CountGoingRight) / float(CountTotal); }
//	float ChanceGoingDown()  const { return float(CountGoingDown)  / float(CountTotal); }
//	float ChanceGoingUp()    const { return float(CountGoingUp)    / float(CountTotal); }
//	float ChanceGoingBack()  const { return float(CountGoingBack)  / float(CountTotal); }
//	float ChanceGoingFront() const { return float(CountGoingFront) / float(CountTotal); }
//};


// uniformly random vector within given box
inline float3 GetRandomVectorInBox(float3 boxMin, float3 boxSize, std::uniform_real_distribution<float>& rng, std::mt19937& engine)
{
	return make_float3(boxMin.x + boxSize.x * rng(engine), boxMin.y + boxSize.y * rng(engine), boxMin.z + boxSize.z * rng(engine));
}

// uniformly random vector on the unit sphere (i.e. length == 1)
inline float3 GetRandomUnitVector(std::uniform_real_distribution<float>& rng, std::mt19937& engine)
{
	float3 result;
	float length;
	// use rejection sampling to get a vector within the unit sphere, then normalize it
	do {
		result.x = rng(engine) * 2.0f - 1.0f;
		result.y = rng(engine) * 2.0f - 1.0f;
		result.z = rng(engine) * 2.0f - 1.0f;
		length = sqrtf(result.x * result.x + result.y * result.y + result.z * result.z);
	} while(length > 1.0f || length < 0.001f); // also reject very short vectors
	result.x /= length;
	result.y /= length;
	result.z /= length;
	return result;
}

// random vector of given length
inline float3 GetRandomJitterVector(float length, std::uniform_real_distribution<float>& rng, std::mt19937& engine)
{
	float3 vec = GetRandomUnitVector(rng, engine);
	return make_float3(vec.x * length, vec.y * length, vec.z * length);
}


#endif
