#include "ProgressBarEffect.h"

#ifndef V
#define V(x)           { hr = (x); }
#endif
#ifndef V_RETURN
#define V_RETURN(x)    { hr = (x); if( FAILED(hr) ) { return hr; } }
#endif


HRESULT ProgressBarEffect::GetVariables()
{
	HRESULT hr;

	// *** Variables ***

	V_RETURN( GetVectorVariable("g_vPosition", m_pvPositionVariable) );
	V_RETURN( GetVectorVariable("g_vSize", m_pvSizeVariable) );
	V_RETURN( GetVectorVariable("g_vColor", m_pvColorVariable) );
	V_RETURN( GetScalarVariable("g_fProgress", m_pfProgressVariable) );

	// *** Techniques ***

	V_RETURN( GetTechnique("tProgressBar", m_pTechnique) );

	return S_OK;
}
