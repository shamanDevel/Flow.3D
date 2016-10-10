#ifndef __TUM3D__FLOW_GRAPH_H__
#define __TUM3D__FLOW_GRAPH_H__


#include <global.h>

#include <vector>

#include <Vec.h>

#include "ParticleTraceParams.h"
#include "TimeVolume.h"

#include "GPUResources.h"
#include "CompressVolume.h"


class FlowGraph
{
public:
	enum EDirection
	{
		DIR_NEG_X = 0,
		DIR_NEG_Y,
		DIR_NEG_Z,
		DIR_POS_X,
		DIR_POS_Y,
		DIR_POS_Z,
		DIR_COUNT
	};
	struct BrickInfo
	{
		BrickInfo();
		float OutFreq[DIR_COUNT];
	};


	FlowGraph();
	~FlowGraph();

	void Init(TimeVolume& volume);
	void Shutdown();

	void Build(GPUResources* pCompressShared, CompressVolumeResources* pCompressVolume, uint particleCountPerBrick, const ParticleTraceParams& params, const std::string& filenameOut = "");
	void Clear();
	bool IsBuilt() const;

	bool SaveToFile(const std::string& filename) const;
	bool LoadFromFile(const std::string& filename);

	// only valid if the flow graph is inited and built!
	const BrickInfo& GetBrickInfo(const tum3D::Vec3i& brickIndex) const;
	const BrickInfo& GetBrickInfo(uint brickLinearIndex) const;

private:
	std::vector<BrickInfo> m_brickInfos;

	TimeVolume* m_pVolume;
	std::vector<TimeVolumeIO::Brick*> m_bricks; // in xyz order
};


#endif
