#include "QuadEffect.h"

#ifndef V
#define V(x)           { hr = (x); }
#endif
#ifndef V_RETURN
#define V_RETURN(x)    { hr = (x); if( FAILED(hr) ) { return hr; } }
#endif


HRESULT QuadEffect::GetVariables()
{
	HRESULT hr;

	// *** Variables ***

	V_RETURN( GetMatrixVariable("g_mWorldViewProj", m_pmWorldViewProjVariable) );

	V_RETURN( GetVectorVariable("g_vCenter", m_pvCenter) );
	V_RETURN(GetVectorVariable("g_vTangent", m_pvTangent));
	V_RETURN(GetVectorVariable("g_vBitangent", m_pvBitangent));
	V_RETURN(GetVectorVariable("g_vSize", m_pvSize));

	V_RETURN( GetShaderResourceVariable("g_tex", m_pTexture) );

	V_RETURN( GetTechnique("tQuad", m_pTechnique) );

	return S_OK;
}

void QuadEffect::SetParameters(ID3D11ShaderResourceView* texture, const tum3D::Vec3f& center, const tum3D::Vec3f& normal, const tum3D::Vec2f& size)
{
	m_pvCenter->SetFloatVector(center);
	m_pvSize->SetFloatVector(size);
	m_pTexture->SetResource(texture);

	tum3D::Vec3f tangent(1, 0, 0);
	float d = tangent.dot(normal);
	if (d > 0.99) {
		//tangent was close to normal, bad for orthogonalization
		tangent = tum3D::Vec3f(0, 1, 0);
		d = tangent.dot(normal);
	}
	tangent -= d * normal; //orhtogonalization
	tangent = tum3D::normalize(tangent);
	tum3D::Vec3f bitangent;
	bitangent = tum3D::crossProd(tangent, normal, bitangent);
	m_pvTangent->SetFloatVector(tangent);
	m_pvBitangent->SetFloatVector(bitangent);
}

void QuadEffect::DrawTexture(const tum3D::Mat4f& worldViewProjMatrix, ID3D11DeviceContext* pContext)
{
	m_pmWorldViewProjVariable->SetMatrix(worldViewProjMatrix.data());

	m_pTechnique->GetPassByIndex(0)->Apply(0, pContext);
	pContext->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLESTRIP);
	pContext->Draw(4, 0);
}
