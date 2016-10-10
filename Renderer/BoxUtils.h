#pragma once

#include <global.h>

#include <Vec.h>

#include "ProjectionParams.h"
#include "Range.h"


// return true if the result has non-zero area/volume
bool ClipBoxAgainstBox(const tum3D::Vec3f& clipBoxMin, const tum3D::Vec3f& clipBoxMax, tum3D::Vec3f& boxMin, tum3D::Vec3f& boxMax);
bool GetBoxScreenExtent(const ProjectionParams& projParams, EStereoEye eye, float eyeDistance, const Range1D& range, const tum3D::Mat4f& view, const tum3D::Vec3f& camPos, const tum3D::Vec3f& boxMin, const tum3D::Vec3f& boxMax, tum3D::Vec2i& screenMin, tum3D::Vec2i& screenMax);

// return true if box is clipped away completely
bool IsBoxClippedAgainstBox(const tum3D::Vec3f& clipBoxMin, const tum3D::Vec3f& clipBoxMax, const tum3D::Vec3f& boxMin, const tum3D::Vec3f& boxMax);
bool IsBoxClippedAgainstPlane(const tum3D::Vec4f& plane, const tum3D::Vec3f& boxMin, const tum3D::Vec3f& boxMax);
