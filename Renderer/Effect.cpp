#include "effect.h"

#include <cassert>
#include <sstream>

#include <D3Dcompiler.h>

#include "CSysTools.h"

using namespace std;


#ifndef SAFE_RELEASE
#define SAFE_RELEASE(p)      { if (p) { (p)->Release(); (p)=nullptr; } }
#endif


Effect::Effect(const wstring& wstrFileName)
: m_pDevice(NULL), m_pEffect(NULL)
{
	m_wstrFileName = CSysTools::GetExePath() + L"\\" + wstrFileName;
}

Effect::~Effect()
{
	assert(!m_pDevice);
}


HRESULT Effect::Create(ID3D11Device* pDevice, bool bShowMessageBoxes)
{
	m_pDevice = pDevice;
	m_bShowMessageBoxes = bShowMessageBoxes;


	// open the file
	HANDLE hFile = CreateFile(m_wstrFileName.c_str(), GENERIC_READ, FILE_SHARE_READ, NULL, OPEN_EXISTING, FILE_FLAG_SEQUENTIAL_SCAN, NULL);
	if(INVALID_HANDLE_VALUE == hFile)
		return E_FAIL;

	// get the file size
	LARGE_INTEGER FileSize;
	GetFileSizeEx(hFile, &FileSize);

	// create enough space for the file data
	BYTE* pFileData = new BYTE[FileSize.LowPart];
	if(!pFileData)
		return E_OUTOFMEMORY;

	// read the data in
	DWORD BytesRead;
	if(!ReadFile(hFile, pFileData, FileSize.LowPart, &BytesRead, NULL))
		return E_FAIL; 

	CloseHandle(hFile);


	HRESULT hr = S_OK;

	// compile or load the shader
	ID3DBlob* pEffectBlob = nullptr;
	if(CSysTools::GetFileExtension(m_wstrFileName) == L"fx") {
		// compile the shader
		D3D_SHADER_MACRO* pDefines = nullptr;
		UINT flags = 0;
		UINT flags2 = 0; // D3DCOMPILE_EFFECT_ALLOW_SLOW_OPS
		ID3DBlob* pErrorBlob = nullptr;
		std::wstring fileNameW = CSysTools::GetFileName(m_wstrFileName);
		std::string fileName(fileNameW.begin(), fileNameW.end());
		hr = D3DCompile(pFileData, FileSize.LowPart, fileName.c_str(), pDefines, nullptr, nullptr, "fx_5_0", flags, flags2, &pEffectBlob, &pErrorBlob);

		if(FAILED(hr)) {
			delete[] pFileData;
			OutputDebugStringA((char*)pErrorBlob->GetBufferPointer());
			SAFE_RELEASE(pErrorBlob);
			return hr;
		}
		SAFE_RELEASE(pErrorBlob);
	} else {
		// not an .fx file, assume it's binary already
		D3DCreateBlob(FileSize.LowPart, &pEffectBlob);
		memcpy(pEffectBlob->GetBufferPointer(), pFileData, FileSize.LowPart);
	}

	delete[] pFileData;

	// Create the effect
	hr = D3DX11CreateEffectFromMemory(pEffectBlob->GetBufferPointer(), pEffectBlob->GetBufferSize(), 0, m_pDevice, &m_pEffect);
	if(FAILED(hr)) {
		SafeRelease();
		return hr;
	}

	SAFE_RELEASE(pEffectBlob);

	hr = GetVariables();
	if(FAILED(hr)) {
		SafeRelease();
		return hr;
	}

	return S_OK;
}

void Effect::SafeRelease()
{
	for (auto it = m_pInputLayouts.begin(); it != m_pInputLayouts.end(); it++)
	{
		(*it)->Release();
	}
	m_pInputLayouts.clear();
	SAFE_RELEASE(m_pEffect);
	m_pDevice = nullptr;
}


HRESULT Effect::GetTechnique(const string& strName, ID3DX11EffectTechnique*& pTechnique)
{
	pTechnique = m_pEffect->GetTechniqueByName(strName.c_str());
	if (!pTechnique->IsValid())
	{
		wostringstream wssTitle;
		wssTitle << L"Effect Compile Error (" << CSysTools::GetFileName(m_wstrFileName) << L")";
		wostringstream wssMessage;
		wssMessage << L"Technique \"" << wstring(strName.begin(), strName.end()) << L"\" not found";
		wostringstream wssDebugMessage;
		/*wssDebugMessage << wssTitle << L"\n" << wssMessage.str() << L"\n";*/
		CSysTools::OutDbg(wssDebugMessage.str());
		if (m_bShowMessageBoxes) CSysTools::ShowErrorMessageBox(wssTitle.str(), wssMessage.str());
		return E_FAIL;
	}
	return S_OK;
}


HRESULT Effect::GetPass(ID3DX11EffectTechnique* pTechnique, const string& strName, ID3DX11EffectPass*& pPass)
{
	pPass = pTechnique->GetPassByName(strName.c_str());
	if (!pPass->IsValid())
	{
		wostringstream wssTitle;
		wssTitle << L"Effect Compile Error (" << CSysTools::GetFileName(m_wstrFileName) << L")";
		wostringstream wssMessage;
		wssMessage << L"Pass \"" << wstring(strName.begin(), strName.end()) << L"\" not found";
		wostringstream wssDebugMessage;
		wssDebugMessage << wssTitle.str() << L"\n" << wssMessage.str() << L"\n";
		CSysTools::OutDbg(wssDebugMessage.str());
		if (m_bShowMessageBoxes) CSysTools::ShowErrorMessageBox(wssTitle.str(), wssMessage.str());
		return E_FAIL;
	}
	return S_OK;
}


HRESULT Effect::GetScalarVariable(const string& strName, ID3DX11EffectScalarVariable*& pScalarVariable)
{
	pScalarVariable = m_pEffect->GetVariableByName(strName.c_str())->AsScalar();
	if (!pScalarVariable->IsValid())
	{
		wostringstream wssTitle;
		wssTitle << L"Effect Compile Error (" << CSysTools::GetFileName(m_wstrFileName) << L")";
		wostringstream wssMessage;
		wssMessage << L"Scalar variable \"" << wstring(strName.begin(), strName.end()) << L"\" not found";
		wostringstream wssDebugMessage;
		wssDebugMessage << wssTitle.str() << L"\n" << wssMessage.str() << L"\n";
		CSysTools::OutDbg(wssDebugMessage.str());
		if (m_bShowMessageBoxes) CSysTools::ShowErrorMessageBox(wssTitle.str(), wssMessage.str());
		return E_FAIL;
	}
	return S_OK;
}


HRESULT Effect::GetVectorVariable(const string& strName, ID3DX11EffectVectorVariable*& pVectorVariable)
{
	pVectorVariable = m_pEffect->GetVariableByName(strName.c_str())->AsVector();
	if (!pVectorVariable->IsValid())
	{
		wostringstream wssTitle;
		wssTitle << L"Effect Compile Error (" << CSysTools::GetFileName(m_wstrFileName) << L")";
		wostringstream wssMessage;
		wssMessage << L"Vector variable \"" << wstring(strName.begin(), strName.end()) << L"\" not found";
		wostringstream wssDebugMessage;
		wssDebugMessage << wssTitle.str() << L"\n" + wssMessage.str() << L"\n";
		CSysTools::OutDbg(wssDebugMessage.str());
		if (m_bShowMessageBoxes) CSysTools::ShowErrorMessageBox(wssTitle.str(), wssMessage.str());
		return E_FAIL;
	}
	return S_OK;
}


HRESULT Effect::GetMatrixVariable(const string& strName, ID3DX11EffectMatrixVariable*& pMatrixVariable)
{
	pMatrixVariable = m_pEffect->GetVariableByName(strName.c_str())->AsMatrix();
	if (!pMatrixVariable->IsValid())
	{
		wostringstream wssTitle;
		wssTitle << L"Effect Compile Error (" << CSysTools::GetFileName(m_wstrFileName) << L")";
		wostringstream wssMessage;
		wssMessage << L"Matrix variable \"" << wstring(strName.begin(), strName.end()) << L"\" not found";
		wostringstream wssDebugMessage;
		wssDebugMessage << wssTitle.str() << L"\n" + wssMessage.str() << L"\n";
		CSysTools::OutDbg(wssDebugMessage.str());
		if (m_bShowMessageBoxes) CSysTools::ShowErrorMessageBox(wssTitle.str(), wssMessage.str());
		return E_FAIL;
	}
	return S_OK;
}


HRESULT Effect::GetShaderResourceVariable(const string& strName, ID3DX11EffectShaderResourceVariable*& pShaderResourceVariable)
{
	pShaderResourceVariable = m_pEffect->GetVariableByName(strName.c_str())->AsShaderResource();
	if (!pShaderResourceVariable->IsValid())
	{
		wostringstream wssTitle;
		wssTitle << L"Effect Compile Error (" << CSysTools::GetFileName(m_wstrFileName) << L")";
		wostringstream wssMessage;
		wssMessage << L"Shader resource variable \"" << wstring(strName.begin(), strName.end()) << L"\" not found";
		wostringstream wssDebugMessage;
		wssDebugMessage << wssTitle.str() << L"\n" + wssMessage.str() << L"\n";
		CSysTools::OutDbg(wssDebugMessage.str());
		if (m_bShowMessageBoxes) CSysTools::ShowErrorMessageBox(wssTitle.str(), wssMessage.str());
		return E_FAIL;
	}
	return S_OK;
}


HRESULT Effect::CreateInputLayout(ID3DX11EffectPass* pPass, const D3D11_INPUT_ELEMENT_DESC* pInputElementDescs, uint uiNumDescs, ID3D11InputLayout*& pInputLayout)
{
	D3DX11_PASS_DESC pd;
	pPass->GetDesc(&pd);
	HRESULT hr = m_pDevice->CreateInputLayout(pInputElementDescs, uiNumDescs, pd.pIAInputSignature, pd.IAInputSignatureSize, &pInputLayout);
	if(FAILED(hr)) return hr;
	m_pInputLayouts.push_back(pInputLayout);
	return S_OK;
}
