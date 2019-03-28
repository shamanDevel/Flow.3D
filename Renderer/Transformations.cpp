#include "TransferFunction.h"
#include <DirectXMath.h>
#include <Vec.h>
#include <FlowVisTool.h>


void XY_translation(float x, float y, float panSens, float DeltaTime, FlowVisTool & flowVisTool)
{
	// assign and define the x-y movement of the mouse
	DirectX::XMFLOAT2 normDelta = DirectX::XMFLOAT2((double)x / (double)flowVisTool.g_renderingParams.m_windowSize.x(), \
	(double)y / (double)flowVisTool.g_renderingParams.m_windowSize.y());

	// convert it into vector
	DirectX::XMVECTOR normDelta_v = DirectX::XMLoadFloat2(&normDelta);

	// scale the movement
	normDelta_v = DirectX::XMVectorScale(normDelta_v, DeltaTime * (float)flowVisTool.g_viewParams.m_viewDistance * (float)panSens);


	// store it back into float2
	DirectX::XMStoreFloat2(&normDelta, normDelta_v);

	// store rotation matrix
	tum3D::Mat4f rotation = flowVisTool.g_viewParams.BuildRotationMatrix();

	// convert translation into homogenous coordinate
	tum3D::Vec4f h_normdelta = tum3D::Vec4f(normDelta.x, normDelta.y, 0, 1);

	// extract camera vectors
	tum3D::Vec3f lookat = flowVisTool.g_viewParams.m_lookAt;
	tum3D::Vec3f viewDir_n = tum3D::normalize(flowVisTool.g_viewParams.GetViewDir());
	tum3D::Vec3f right_n = tum3D::normalize(flowVisTool.g_viewParams.GetRightVector());
	tum3D::Vec3f up_n;
	tum3D::crossProd(viewDir_n, right_n, up_n);

	// translate along up and right vectors
	tum3D::Vec3f x_trans = -h_normdelta.x() * right_n;
	tum3D::Vec3f y_trans = h_normdelta.y() * up_n;

	// store and assign the new lookAt vector
	lookat.set(lookat.x() + x_trans.x() + y_trans.x(), lookat.y() + x_trans.y() + y_trans.y(), lookat.z() + x_trans.z() + y_trans.z());
	flowVisTool.g_viewParams.m_lookAt = lookat;
}

void XY_Rotation(float x, float y, float orbitSens, float DeltaTime, FlowVisTool & flowVisTool)
{
	tum3D::Vec2d normDelta = tum3D::Vec2d(x/ (double)flowVisTool.g_renderingParams.m_windowSize.x(), \
		y / (double)flowVisTool.g_renderingParams.m_windowSize.y());
	tum3D::Vec2d delta = normDelta * DeltaTime * (double)orbitSens;


	tum3D::Vec4f rotationX;
	tum3D::Vec4f rotation;
	tum3D::Vec4f rotationY;
	////////////////////////////// rotation around Y-axis /////////////////////////

	// calculate the rotation matrix by delta.x() and around up vector and store it in rotationX
	tum3D::rotationQuaternion((float)(delta.x()) * PI, tum3D::Vec3f(0.0f, 1.0f, 0.0f), rotationX);

	// calculate the 4X4 rotation matrix
	tum3D::multQuaternion(rotationX, flowVisTool.g_viewParams.m_rotationQuat, rotation);

	// apply the roation
	flowVisTool.g_viewParams.m_rotationQuat = rotation;

	////////////////////////////// rotation around X-axis /////////////////////////

	// calculate the 3x3 rotation matrix and save it in rotationY
	tum3D::rotationQuaternion((float)(delta.y()) * PI, tum3D::Vec3f(1.0f, 0.0f, 0.0f), rotationY);

	// calculate the 4x4 rotation matrix and save it in rotation
	tum3D::multQuaternion(rotationY, flowVisTool.g_viewParams.m_rotationQuat, rotation);

	// apply the roation matrix
	flowVisTool.g_viewParams.m_rotationQuat = rotation;
}

void zoomInOut(float z, float zoomSens, float DeltaTime, FlowVisTool & flowVisTool)
{
	flowVisTool.g_viewParams.m_viewDistance -= z * DeltaTime * zoomSens * flowVisTool.g_viewParams.m_viewDistance;
	flowVisTool.g_viewParams.m_viewDistance = std::max(0.0001f, flowVisTool.g_viewParams.m_viewDistance);
}