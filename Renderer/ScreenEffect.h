#ifndef __TUM3D__SCREENEFFECT_H__
#define __TUM3D__SCREENEFFECT_H__


#include <global.h>

#include "Effect.h"


class ScreenEffect : public Effect
{
public:
	ScreenEffect() : Effect(L"Screen.fxo") { }

	enum Pass
	{
		SolidBlendBehind = 0,
		Blit = 1,
		BlitBlendOver = 2,
		BlitBlendOverSubWhite = 3
	};

	// *** Variables ***

	ID3DX11EffectVectorVariable*			m_pvScreenMinVariable;
	ID3DX11EffectVectorVariable*			m_pvScreenMaxVariable;
	ID3DX11EffectVectorVariable*			m_pvColorVariable;
	ID3DX11EffectVectorVariable*			m_pvTexCoordMinVariable;
	ID3DX11EffectVectorVariable*			m_pvTexCoordMaxVariable;
	ID3DX11EffectShaderResourceVariable*	m_pTexVariable;

	// *** Techniques ***

	ID3DX11EffectTechnique*					m_pTechnique;

protected:
	virtual HRESULT GetVariables();
};


#endif
