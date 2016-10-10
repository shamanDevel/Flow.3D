#ifndef __TUM3D__PROGRESSBAREFFECT_H__
#define __TUM3D__PROGRESSBAREFFECT_H__


#include <global.h>

#include "Effect.h"


class ProgressBarEffect : public Effect
{
public:
	ProgressBarEffect() : Effect(L"ProgressBar.fxo") { }

	// *** Variables ***

	ID3DX11EffectVectorVariable*			m_pvPositionVariable;
	ID3DX11EffectVectorVariable*			m_pvSizeVariable;
	ID3DX11EffectVectorVariable*			m_pvColorVariable;
	ID3DX11EffectScalarVariable*			m_pfProgressVariable;

	// *** Techniques ***

	ID3DX11EffectTechnique*					m_pTechnique;

protected:
	virtual HRESULT GetVariables();
};

#endif
