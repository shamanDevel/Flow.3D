#ifndef __TUM3D__BOX_H__
#define __TUM3D__BOX_H__


#include "global.h"

#include <D3D11.h>

#include <Vec.h>

#include "BoxEffect.h"


class Box
{
public:
	Box();
	~Box();

	HRESULT Create(ID3D11Device* pDevice);
	void Release();

	//void RenderSolid();
	void RenderLines(const tum3D::Mat4f& view, const tum3D::Mat4f& proj,
					 const tum3D::Vec3f& lightPos,
					 const tum3D::Vec3f& boxMin, const tum3D::Vec3f& boxMax,
					 const tum3D::Vec4f& color,
					 float tubeRadius = 0.0f,
					 bool secondPass = false);

	void RenderBrickLines(const tum3D::Mat4f& view, const tum3D::Mat4f& proj,
						  const tum3D::Vec3f& lightPos,
						  const tum3D::Vec3f& boxMin, const tum3D::Vec3f& boxMax,
						  const tum3D::Vec4f& color,
						  const tum3D::Vec3f& brickSize,
						  float tubeRadius = 0.0f,
						  bool secondPass = false);

private:
	ID3D11Device* m_pDevice;

	BoxEffect     m_effect;

	ID3D11Buffer* m_pVB;
	ID3D11Buffer* m_pSolidIB;
	ID3D11Buffer* m_pLinesIB;
};


#endif
