#ifndef __TUM3D__HEATMAPRAYTRACEREFFECT_H__
#define __TUM3D__HEATMAPRAYTRACEREFFECT_H__


#include <global.h>

#include "Effect.h"


class HeatMapRaytracerEffect : public Effect
{
public:
	HeatMapRaytracerEffect() : Effect(L"HeatMapRaytracer.fxo") { }

	// *** Variables ***

	ID3DX11EffectMatrixVariable*			m_pmWorldView;
	ID3DX11EffectMatrixVariable*			m_pmInvWorldView;
	ID3DX11EffectVectorVariable*			m_pvViewport;
	ID3DX11EffectVectorVariable*			m_pvScreenSize;
	ID3DX11EffectVectorVariable*			m_pvDepthParams;
	ID3DX11EffectVectorVariable*			m_pvBoxMin;
	ID3DX11EffectVectorVariable*			m_pvBoxMax;
	ID3DX11EffectScalarVariable*			m_pfStepSizeWorld;
	ID3DX11EffectScalarVariable*			m_pfDensityScale;
	ID3DX11EffectScalarVariable*			m_pfAlphaScale;
	ID3DX11EffectShaderResourceVariable*	m_pHeatMap1;
	ID3DX11EffectShaderResourceVariable*	m_pHeatMap2;
	ID3DX11EffectShaderResourceVariable*	m_pTransferFunction;
	ID3DX11EffectShaderResourceVariable*	m_pDepthTexture;
	ID3DX11EffectScalarVariable*			m_pfIsoValue;
	ID3DX11EffectVectorVariable*			m_pvTextureGridSpacing;

	// *** Techniques ***

	ID3DX11EffectTechnique*					m_pTechnique;

protected:
	virtual HRESULT GetVariables();
};

#endif
