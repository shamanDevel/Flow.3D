#include "BoxUtils.h"

using namespace tum3D;


bool ClipBoxAgainstBox(const Vec3f& clipBoxMin, const Vec3f& clipBoxMax, Vec3f& boxMin, Vec3f& boxMax)
{
	// clamp against clip box
	maximum(boxMin, clipBoxMin);
	minimum(boxMax, clipBoxMax);

	// all(boxMin < boxMax)
	return boxMin.compLess(boxMax).minimum();
}

bool GetBoxScreenExtent(const ProjectionParams& projParams, EStereoEye eye, float eyeDistance, const Range1D& range, const Mat4f& view, const Vec3f& camPos, const Vec3f& boxMin, const Vec3f& boxMax, Vec2i& screenMin, Vec2i& screenMax)
{
	// get bounding box corners
	Vec3f boxCorners[8] = {
		Vec3f(boxMin.x(), boxMin.y(), boxMin.z()),
		Vec3f(boxMax.x(), boxMin.y(), boxMin.z()),
		Vec3f(boxMin.x(), boxMax.y(), boxMin.z()),
		Vec3f(boxMax.x(), boxMax.y(), boxMin.z()),
		Vec3f(boxMin.x(), boxMin.y(), boxMax.z()),
		Vec3f(boxMax.x(), boxMin.y(), boxMax.z()),
		Vec3f(boxMin.x(), boxMax.y(), boxMax.z()),
		Vec3f(boxMax.x(), boxMax.y(), boxMax.z())
	};

	uint width  = projParams.GetImageWidth (range);
	uint height = projParams.GetImageHeight(range);

	// check if box is behind camera or clips near plane
	Vec4f viewDir4 = -view.getRow(2);
	Vec3f viewDir = viewDir4.xyz();
	normalize(viewDir);
	bool behindCamera = true;
	bool clipsNearPlane = false;
	for(uint i = 0; i < 8; i++) {
		Vec3f cam2box = boxCorners[i] - camPos;
		float dot = dotProd(viewDir, cam2box);
		if(dot > 0.0f) behindCamera = false;
		if(dot < projParams.m_near) clipsNearPlane = true;
	}
	if(behindCamera) {
		// box is fully behind camera, don't need to render anything
		screenMin.set(1, 1);
		screenMax.set(0, 0);
		return false;
	}
	if(clipsNearPlane) {
		// box clips the near plane, the projection isn't safe - render whole screen
		screenMin.set(0, 0);
		screenMax.set(width - 1, height - 1);
		return true;
	}
	// still here: box is completely in front of near plane

	// project into screen space and compute screen space bounding box
	Mat4f proj = projParams.BuildProjectionMatrix(eye, eyeDistance, range);
	Mat4f viewProj = proj * view;
	screenMin.set(width - 1, height - 1);
	screenMax.set(0, 0);
	for(uint i = 0; i < 8; i++) {
		// transform into clip space
		Vec4f boxCornerProj = viewProj * Vec4f(boxCorners[i], 1.0f);
		boxCornerProj /= boxCornerProj.w();
		// transform into pixel coords
		float x = float(width ) * ( boxCornerProj.x() + 1.0f) / 2.0f;
		float y = float(height) * (-boxCornerProj.y() + 1.0f) / 2.0f;

		screenMin.x() = min(screenMin.x(), int(floor(x)));
		screenMin.y() = min(screenMin.y(), int(floor(y)));
		screenMax.x() = max(screenMax.x(), int(ceil(x)));
		screenMax.y() = max(screenMax.y(), int(ceil(y)));
	}

	// clamp against viewport
	screenMin.x() = clamp(screenMin.x(), 0, int(width ) - 1);
	screenMin.y() = clamp(screenMin.y(), 0, int(height) - 1);
	screenMax.x() = clamp(screenMax.x(), 0, int(width ) - 1);
	screenMax.y() = clamp(screenMax.y(), 0, int(height) - 1);

	// all(screenMin < screenMax)
	return screenMin.compLess(screenMax).minimum();
}


bool IsBoxClippedAgainstBox(const Vec3f& clipBoxMin, const Vec3f& clipBoxMax, const Vec3f& boxMin, const Vec3f& boxMax)
{
	Vec3f boxMinClipped = boxMin, boxMaxClipped = boxMax;
	return !ClipBoxAgainstBox(clipBoxMin, clipBoxMax, boxMinClipped, boxMaxClipped);
}

bool IsBoxClippedAgainstPlane(const Vec4f& plane, const Vec3f& boxMin, const Vec3f& boxMax)
{
	Vec3f corners[8] = {
		Vec3f(boxMin.x(), boxMin.y(), boxMin.z()),
		Vec3f(boxMax.x(), boxMin.y(), boxMin.z()),
		Vec3f(boxMin.x(), boxMax.y(), boxMin.z()),
		Vec3f(boxMax.x(), boxMax.y(), boxMin.z()),
		Vec3f(boxMin.x(), boxMin.y(), boxMax.z()),
		Vec3f(boxMax.x(), boxMin.y(), boxMax.z()),
		Vec3f(boxMin.x(), boxMax.y(), boxMax.z()),
		Vec3f(boxMax.x(), boxMax.y(), boxMax.z())
	};

	for(uint i = 0; i < 8; i++)
	{
		if(dotProd(plane, Vec4f(corners[i], 1.0f)) > 0.0f)
		{
			return false;
		}
	}

	return true;
}
