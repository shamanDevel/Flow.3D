#ifndef __TUM3D__QUADEFFECT_H__
#define __TUM3D__QUADEFFECT_H__


#include <global.h>
#include <Vec.h>

#include "Effect.h"


class QuadEffect : public Effect
{
public:
	QuadEffect() : Effect(L"Quad.fxo") { }
protected:

	// *** Variables ***

	//ID3DX11EffectMatrixVariable*			m_pmWorldViewVariable;
	//ID3DX11EffectMatrixVariable*			m_pmProjVariable;
	ID3DX11EffectMatrixVariable*			m_pmWorldViewProjVariable;

	ID3DX11EffectVectorVariable*			m_pvTangent;
	ID3DX11EffectVectorVariable*			m_pvBitangent;
	ID3DX11EffectVectorVariable*			m_pvCenter;
	ID3DX11EffectVectorVariable*			m_pvSize;

	ID3DX11EffectShaderResourceVariable*	m_pTexture;

	// *** Techniques ***
	ID3DX11EffectTechnique*					m_pTechnique;

public:

	void SetParameters(ID3D11ShaderResourceView* texture, const tum3D::Vec3f& center, const tum3D::Vec3f& normal, const tum3D::Vec2f& size);

	void DrawTexture(const tum3D::Mat4f& worldViewProjMatrix, ID3D11DeviceContext* pContext, bool withDepth);

protected:
	virtual HRESULT GetVariables();
};

#endif
