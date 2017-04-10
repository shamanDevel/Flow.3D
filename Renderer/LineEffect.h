#ifndef __TUM3D__LINEEFFECT_H__
#define __TUM3D__LINEEFFECT_H__


#include <global.h>

#include "Effect.h"


class LineEffect : public Effect
{
public:
	LineEffect() : Effect(L"Line.fxo") { }

	// *** Variables ***

	//ID3DX11EffectMatrixVariable*			m_pmWorldViewVariable;
	//ID3DX11EffectMatrixVariable*			m_pmProjVariable;
	ID3DX11EffectMatrixVariable*			m_pmWorldViewProjVariable;
	ID3DX11EffectMatrixVariable*            m_pmWorldViewRotation;

	ID3DX11EffectVectorVariable*			m_pvLightPosVariable;

	ID3DX11EffectScalarVariable*			m_pfRibbonHalfWidthVariable;
	ID3DX11EffectScalarVariable*			m_pfTubeRadiusVariable;

	ID3DX11EffectScalarVariable*			m_pfParticleSizeVariable;
	ID3DX11EffectScalarVariable*			m_pfScreenAspectRatioVariable;
	ID3DX11EffectScalarVariable*			m_pfParticleTransparencyVariable;
	ID3DX11EffectVectorVariable*			m_pvParticleClipPlane;

	ID3DX11EffectScalarVariable*			m_pbTubeRadiusFromVelocityVariable;
	ID3DX11EffectScalarVariable*			m_pfReferenceVelocityVariable;

	ID3DX11EffectScalarVariable*			m_piColorMode;
	ID3DX11EffectVectorVariable*			m_pvColor0Variable;
	ID3DX11EffectVectorVariable*			m_pvColor1Variable;
	ID3DX11EffectScalarVariable*			m_pfTimeMinVariable;
	ID3DX11EffectScalarVariable*			m_pfTimeMaxVariable;
	ID3DX11EffectVectorVariable*			m_pvHalfSizeWorldVariable;

	ID3DX11EffectScalarVariable*			m_pbTimeStripesVariable;
	ID3DX11EffectScalarVariable*			m_pfTimeStripeLengthVariable;

	ID3DX11EffectScalarVariable*			m_piMeasureMode;
	ID3DX11EffectScalarVariable*			m_pfMeasureScale;

	ID3DX11EffectShaderResourceVariable*	m_ptexColors;
	ID3DX11EffectShaderResourceVariable*    m_pseedColors;
	ID3DX11EffectShaderResourceVariable*    m_ptransferFunction;

	ID3DX11EffectVectorVariable*			m_pvBoxMinVariable;
	ID3DX11EffectVectorVariable*			m_pvBoxSizeVariable;
	ID3DX11EffectVectorVariable*			m_pvCamPosVariable;
	ID3DX11EffectVectorVariable*			m_pvCamRightVariable;
	ID3DX11EffectScalarVariable*			m_pfBallRadiusVariable;

	// *** Techniques ***

	ID3DX11EffectTechnique*					m_pTechnique;
	ID3DX11EffectTechnique*					m_pTechniqueBalls;

	// *** Input Layouts ***

	ID3D11InputLayout*						m_pInputLayout;
	ID3D11InputLayout*						m_pInputLayoutBalls;

protected:
	virtual HRESULT GetVariables();
};

#endif
