#ifndef __TUM3D__BOXEFFECT_H__
#define __TUM3D__BOXEFFECT_H__


#include <global.h>

#include "Effect.h"


class BoxEffect : public Effect
{
public:
	BoxEffect() : Effect(L"Box.fxo") { }

	// *** Variables ***

	ID3DX11EffectMatrixVariable*			m_pmWorldViewVariable;
	ID3DX11EffectMatrixVariable*			m_pmProjVariable;
	ID3DX11EffectMatrixVariable*			m_pmWorldViewProjVariable;
	ID3DX11EffectVectorVariable*			m_pvBoxMinVariable;
	ID3DX11EffectVectorVariable*			m_pvBoxSizeVariable;
	ID3DX11EffectVectorVariable*			m_pvColorVariable;
	ID3DX11EffectScalarVariable*			m_pfTubeRadiusVariable;
	ID3DX11EffectVectorVariable*			m_pvLightPosVariable;
	ID3DX11EffectVectorVariable*			m_pvCamPosVariable;
	ID3DX11EffectVectorVariable*			m_pvCamRightVariable;

	ID3DX11EffectVectorVariable*			m_pvBrickSizeVariable;
	ID3DX11EffectVectorVariable*			m_pvLineCountVariable;

	// *** Techniques ***

	ID3DX11EffectTechnique*					m_pTechnique;

	// *** Input Layouts ***

	ID3D11InputLayout*						m_pInputLayout;

protected:
	virtual HRESULT GetVariables();
};

#endif
