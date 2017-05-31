#include "HeatMapRaytracerEffect.h"

#ifndef V
#define V(x)           { hr = (x); }
#endif
#ifndef V_RETURN
#define V_RETURN(x)    { hr = (x); if( FAILED(hr) ) { return hr; } }
#endif


HRESULT HeatMapRaytracerEffect::GetVariables()
{
	HRESULT hr;

	// *** Variables ***

	V_RETURN( GetMatrixVariable("g_mInvWorldViewProj", m_pmInvWorldViewProjVariable) );
	V_RETURN( GetVectorVariable("g_vViewport", m_pvViewport) );
	V_RETURN(GetVectorVariable("g_vScreenSize", m_pvScreenSize));
	V_RETURN(GetVectorVariable("g_vDepthParams", m_pvDepthParams));
	V_RETURN(GetVectorVariable("g_vBoxMin", m_pvBoxMin));
	V_RETURN(GetVectorVariable("g_vBoxMax", m_pvBoxMax));
	V_RETURN( GetScalarVariable("g_fStepSizeWorld", m_pfStepSizeWorld) );
	V_RETURN( GetScalarVariable("g_fDensityScale", m_pfDensityScale) );
	V_RETURN( GetShaderResourceVariable("g_heatMap", m_pHeatMap) );
	V_RETURN( GetShaderResourceVariable("g_transferFunction", m_pTransferFunction) );
	V_RETURN( GetShaderResourceVariable("g_depthTexture", m_pDepthTexture) );

	// *** Techniques ***
	V_RETURN( GetTechnique("tRaytrace", m_pTechnique) );

	return S_OK;
}
