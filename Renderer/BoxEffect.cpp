#include "BoxEffect.h"

#ifndef V
#define V(x)           { hr = (x); }
#endif
#ifndef V_RETURN
#define V_RETURN(x)    { hr = (x); if( FAILED(hr) ) { return hr; } }
#endif


HRESULT BoxEffect::GetVariables()
{
	HRESULT hr;

	// *** Variables ***

	V_RETURN( GetMatrixVariable("g_mWorldView", m_pmWorldViewVariable) );
	V_RETURN( GetMatrixVariable("g_mProj", m_pmProjVariable) );
	V_RETURN( GetMatrixVariable("g_mWorldViewProj", m_pmWorldViewProjVariable) );
	V_RETURN( GetVectorVariable("g_vBoxMin", m_pvBoxMinVariable) );
	V_RETURN( GetVectorVariable("g_vBoxSize", m_pvBoxSizeVariable) );
	V_RETURN( GetVectorVariable("g_vColor", m_pvColorVariable) );
	V_RETURN( GetScalarVariable("g_fTubeRadius", m_pfTubeRadiusVariable) );
	V_RETURN( GetVectorVariable("g_vLightPos", m_pvLightPosVariable) );
	V_RETURN( GetVectorVariable("g_vCamPos", m_pvCamPosVariable) );
	V_RETURN( GetVectorVariable("g_vCamRight", m_pvCamRightVariable) );

	V_RETURN( GetVectorVariable("g_vBrickSize", m_pvBrickSizeVariable) );
	V_RETURN( GetVectorVariable("g_vLineCount", m_pvLineCountVariable) );

	// *** Techniques ***

	V_RETURN( GetTechnique("tBox", m_pTechnique) );

	// *** Input Layouts ***

	D3D11_INPUT_ELEMENT_DESC layout[] = {
		{ "POSITION", 0, DXGI_FORMAT_R32G32B32_FLOAT, 0, 0, D3D11_INPUT_PER_VERTEX_DATA, 0 },
	};
	V_RETURN( CreateInputLayout(m_pTechnique->GetPassByIndex(0), layout, sizeof(layout)/sizeof(layout[0]), m_pInputLayout) );

	return S_OK;
}
