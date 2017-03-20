#include "LineEffect.h"

#ifndef V
#define V(x)           { hr = (x); }
#endif
#ifndef V_RETURN
#define V_RETURN(x)    { hr = (x); if( FAILED(hr) ) { return hr; } }
#endif


HRESULT LineEffect::GetVariables()
{
	HRESULT hr;

	// *** Variables ***

	//V_RETURN( GetMatrixVariable("g_mWorldView", m_pmWorldViewVariable) );
	//V_RETURN( GetMatrixVariable("g_mProj", m_pmProjVariable) );
	V_RETURN( GetMatrixVariable("g_mWorldViewProj", m_pmWorldViewProjVariable) );

	V_RETURN( GetVectorVariable("g_vLightPos", m_pvLightPosVariable) );

	V_RETURN( GetScalarVariable("g_fRibbonHalfWidth", m_pfRibbonHalfWidthVariable) );
	V_RETURN( GetScalarVariable("g_fTubeRadius", m_pfTubeRadiusVariable) );

	V_RETURN(GetScalarVariable("g_fParticleSize", m_pfParticleSizeVariable));
	V_RETURN(GetScalarVariable("g_fScreenAspectRatio", m_pfScreenAspectRatioVariable));
	V_RETURN( GetScalarVariable("g_fParticleTransparency", m_pfParticleTransparencyVariable) );

	V_RETURN( GetScalarVariable("g_bTubeRadiusFromVelocity", m_pbTubeRadiusFromVelocityVariable) );
	V_RETURN( GetScalarVariable("g_fReferenceVelocity", m_pfReferenceVelocityVariable) );

	V_RETURN( GetScalarVariable("g_bColorByTime", m_pbColorByTimeVariable) );
	V_RETURN( GetVectorVariable("g_vColor0", m_pvColor0Variable) );
	V_RETURN( GetVectorVariable("g_vColor1", m_pvColor1Variable) );
	V_RETURN( GetScalarVariable("g_fTimeMin", m_pfTimeMinVariable) );
	V_RETURN( GetScalarVariable("g_fTimeMax", m_pfTimeMaxVariable) );

	V_RETURN( GetScalarVariable("g_bTimeStripes", m_pbTimeStripesVariable) );
	V_RETURN( GetScalarVariable("g_fTimeStripeLength", m_pfTimeStripeLengthVariable) );

	V_RETURN( GetShaderResourceVariable("g_texColors", m_ptexColors) );

	V_RETURN( GetVectorVariable("g_vBoxMin", m_pvBoxMinVariable) );
	V_RETURN( GetVectorVariable("g_vBoxSize", m_pvBoxSizeVariable) );
	V_RETURN( GetVectorVariable("g_vCamPos", m_pvCamPosVariable) );
	V_RETURN( GetVectorVariable("g_vCamRight", m_pvCamRightVariable) );
	V_RETURN( GetScalarVariable("g_fBallRadius", m_pfBallRadiusVariable) );

	// *** Techniques ***

	V_RETURN( GetTechnique("tLine", m_pTechnique) );
	V_RETURN( GetTechnique("tBall", m_pTechniqueBalls) );

	// *** Input Layouts ***

	// This one matches TracingCommons::LineVertex 

	D3D11_INPUT_ELEMENT_DESC layout[] = {
		{ "POSITION", 0, DXGI_FORMAT_R32G32B32_FLOAT, 0, D3D11_APPEND_ALIGNED_ELEMENT, D3D11_INPUT_PER_VERTEX_DATA, 0 },
		{ "TIME",     0, DXGI_FORMAT_R32_FLOAT,       0, D3D11_APPEND_ALIGNED_ELEMENT, D3D11_INPUT_PER_VERTEX_DATA, 0 },
		{ "NORMAL",   0, DXGI_FORMAT_R32G32B32_FLOAT, 0, D3D11_APPEND_ALIGNED_ELEMENT, D3D11_INPUT_PER_VERTEX_DATA, 0 },
		{ "VELOCITY", 0, DXGI_FORMAT_R32G32B32_FLOAT, 0, D3D11_APPEND_ALIGNED_ELEMENT, D3D11_INPUT_PER_VERTEX_DATA, 0 },
		{ "LINE_ID",  0, DXGI_FORMAT_R32_UINT,        0, D3D11_APPEND_ALIGNED_ELEMENT, D3D11_INPUT_PER_VERTEX_DATA, 0 },
		//{ "JACOBIAN", 0, DXGI_FORMAT_R32G32B32_FLOAT, 0, D3D11_APPEND_ALIGNED_ELEMENT, D3D11_INPUT_PER_VERTEX_DATA, 0 },
		//{ "JACOBIAN", 1, DXGI_FORMAT_R32G32B32_FLOAT, 0, D3D11_APPEND_ALIGNED_ELEMENT, D3D11_INPUT_PER_VERTEX_DATA, 0 },
		//{ "JACOBIAN", 2, DXGI_FORMAT_R32G32B32_FLOAT, 0, D3D11_APPEND_ALIGNED_ELEMENT, D3D11_INPUT_PER_VERTEX_DATA, 0 },
	};
	V_RETURN( CreateInputLayout(m_pTechnique->GetPassByIndex(0), layout, sizeof(layout)/sizeof(layout[0]), m_pInputLayout) );

	D3D11_INPUT_ELEMENT_DESC layoutBalls[] = {
		{ "POSITION", 0, DXGI_FORMAT_R32G32B32_FLOAT, 0, D3D11_APPEND_ALIGNED_ELEMENT, D3D11_INPUT_PER_VERTEX_DATA, 0 },
	};
	V_RETURN( CreateInputLayout(m_pTechniqueBalls->GetPassByIndex(0), layoutBalls, sizeof(layoutBalls)/sizeof(layoutBalls[0]), m_pInputLayoutBalls) );

	return S_OK;
}
