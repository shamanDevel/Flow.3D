#include "ScreenEffect.h"

#ifndef V
#define V(x)           { hr = (x); }
#endif
#ifndef V_RETURN
#define V_RETURN(x)    { hr = (x); if( FAILED(hr) ) { return hr; } }
#endif


HRESULT ScreenEffect::GetVariables()
{
	HRESULT hr;

	// *** Variables ***

	V_RETURN( GetVectorVariable("g_vScreenMin", m_pvScreenMinVariable) );
	V_RETURN( GetVectorVariable("g_vScreenMax", m_pvScreenMaxVariable) );
	V_RETURN( GetVectorVariable("g_vColor", m_pvColorVariable) );
	V_RETURN( GetVectorVariable("g_vTexCoordMin", m_pvTexCoordMinVariable) );
	V_RETURN( GetVectorVariable("g_vTexCoordMax", m_pvTexCoordMaxVariable) );
	V_RETURN( GetShaderResourceVariable("g_tex", m_pTexVariable) );

	// *** Techniques ***

	V_RETURN( GetTechnique("tScreen", m_pTechnique) );

	return S_OK;
}
