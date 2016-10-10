#ifndef __TUM3D__EFFECT_H__
#define __TUM3D__EFFECT_H__


#include <global.h>

#include <string>
#include <vector>

#include <D3D11.h>
#include <D3DX11Effect/d3dx11effect.h>


class Effect
{
public:
	Effect(const std::wstring& wstrFileName);
	~Effect();

	HRESULT Create(ID3D11Device* pDevice, bool bShowMessageBoxes = true);
	void SafeRelease();

	HRESULT GetTechnique(const std::string& strName, ID3DX11EffectTechnique*& pTechnique);
	HRESULT GetPass(ID3DX11EffectTechnique* pTechnique, const std::string& strName, ID3DX11EffectPass*& pPass);
	HRESULT GetScalarVariable(const std::string& strName, ID3DX11EffectScalarVariable*& pScalarVariable);
	HRESULT GetVectorVariable(const std::string& strName, ID3DX11EffectVectorVariable*& pVectorVariable);
	HRESULT GetMatrixVariable(const std::string& strName, ID3DX11EffectMatrixVariable*& pMatrixVariable);
	HRESULT GetShaderResourceVariable(const std::string& strName, ID3DX11EffectShaderResourceVariable*& pShaderResourceVariable);

	HRESULT CreateInputLayout(ID3DX11EffectPass* pPass, const D3D11_INPUT_ELEMENT_DESC* pInputElementDescs, uint uiNumDescs, ID3D11InputLayout*& pInputLayout);

	ID3DX11Effect* m_pEffect;

protected:
	virtual HRESULT GetVariables() = 0;

private:
	std::wstring                    m_wstrFileName;
	ID3D11Device*                   m_pDevice;
	bool                            m_bShowMessageBoxes;
	std::vector<ID3D11InputLayout*> m_pInputLayouts;

	// disallow copy and assignment
	Effect(const Effect&);
	Effect& operator=(const Effect&);
};


#endif
