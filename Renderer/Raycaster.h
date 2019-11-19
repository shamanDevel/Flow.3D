#ifndef __TUM3D__RAYCASTER_H__
#define __TUM3D__RAYCASTER_H__


#include <global.h>

#include <vector>

#include <cuda_runtime.h>

#include <Vec.h>

#include "cutil_math.h"

#include "ProjectionParams.h"
#include "StereoParams.h"
#include "Range.h"
#include "RaycasterKernelParams.h"


class BrickSlot;
struct RaycastParams;


class Raycaster
{
public:
	Raycaster();
	~Raycaster();

	bool Create();
	void Release();
	bool IsCreated() { return m_isCreated; }

	// size of one brick (without overlaps) in world space. cubic brick and isotropic volume -> this is a scalar
	void SetBrickSizeWorld(tum3D::Vec3f brickSizeWorld) { m_brickSizeWorld = brickSizeWorld; }
	// amount of brick overlap (left and right each) in world space, i.e. bricksize with overlap = bricksize + 2*overlap
	void SetBrickOverlapWorld(tum3D::Vec3f brickOverlapWorld) { m_brickOverlapWorld = brickOverlapWorld; }
	// distance between two grid points
	void SetGridSpacing(tum3D::Vec3f gridSpacing) { m_gridSpacing = gridSpacing; }

	void SetParams(const ProjectionParams& projParams, const StereoParams& stereoParams, const Range1D& range);

	void FillMeasureBrick(const RaycastParams& params, const BrickSlot& brickSlotVelocity, BrickSlot& brickSlotMeasure);

	// boxMin and boxMax must *not* already be clipped against clipbox! (unclipped coords needed for world->tex transform)
	// if params.m_measureComputeMode == MEASURE_COMPUTE_ONTHEFLY, brickSlots should contain velocity bricks (fine to coarse scale); otherwise it should be a single scalar measure brick (of the correct measure!)
	// returns true if this brick is now done, false if more passes are needed
	bool RenderBrick(
		cudaArray* pTargetArray,
		cudaArray* pDepthArray,
		const RaycastParams& params,
		const tum3D::Mat4f& view,
		EStereoEye eye,
		cudaArray* pTfArray,
		const std::vector<BrickSlot*>& brickSlots,
		const tum3D::Vec3f& boxMin,
		const tum3D::Vec3f& boxMax,
		uint pass);

private:
	uint GetNumPassesNeeded(
		const RaycastParams& params,
		const tum3D::Mat4f& view,
		const tum3D::Vec3f& camPos,
		const tum3D::Vec3f& boxMinClipped,
		const tum3D::Vec3f& boxMaxClipped) const;
	cudaTextureObject_t CreateCudaTextureObject(const cudaArray* array, const RaycastParams& params);

	bool					m_isCreated;

	tum3D::Vec3f			m_brickSizeWorld;
	tum3D::Vec3f			m_brickOverlapWorld;
	tum3D::Vec3f			m_gridSpacing;

	ProjectionParams		m_projParams;
	StereoParams			m_stereoParams;
	Range1D					m_range;

	ProjectionParamsGPU*	md_pProjParamsCyclop;
	ProjectionParamsGPU*	md_pProjParamsLeft;
	ProjectionParamsGPU*	md_pProjParamsRight;

	RaycastParamsGPU		m_raycastParamsGPU;
	cudaEvent_t				m_raycastParamsUploadEvent;


	// disallow copy and assignment
	Raycaster(const Raycaster&);
	Raycaster& operator=(const Raycaster&);
};


#endif
