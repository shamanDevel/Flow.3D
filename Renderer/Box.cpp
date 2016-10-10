#include "Box.h"

#include <cassert>

using namespace tum3D;


Box::Box()
	: m_pDevice(nullptr)
	, m_pVB(nullptr), m_pSolidIB(nullptr), m_pLinesIB(nullptr)
{
}

Box::~Box()
{
	assert(!m_pDevice);
}

HRESULT Box::Create(ID3D11Device* pDevice)
{
	m_pDevice = pDevice;

	HRESULT hr;

	hr = m_effect.Create(m_pDevice);
	if(FAILED(hr)) {
		Release();
		return hr;
	}

	D3D11_BUFFER_DESC desc;
	desc.CPUAccessFlags = 0;
	desc.MiscFlags = 0;
	desc.StructureByteStride = 0;
	desc.Usage = D3D11_USAGE_IMMUTABLE;

	D3D11_SUBRESOURCE_DATA initData;
	initData.SysMemPitch = 0;
	initData.SysMemSlicePitch = 0;

	// vertex buffer
	float vertices[] =
	{
		0.0f, 0.0f, 0.0f,
		1.0f, 0.0f, 0.0f,
		0.0f, 1.0f, 0.0f,
		1.0f, 1.0f, 0.0f,
		0.0f, 0.0f, 1.0f,
		1.0f, 0.0f, 1.0f,
		0.0f, 1.0f, 1.0f,
		1.0f, 1.0f, 1.0f,
	};
	desc.BindFlags = D3D11_BIND_VERTEX_BUFFER;
	desc.ByteWidth = sizeof(vertices);
	initData.pSysMem = vertices;
	hr = m_pDevice->CreateBuffer(&desc, &initData, &m_pVB);
	if(FAILED(hr)) {
		Release();
		return hr;
	}

	// index buffer (solid)
	ushort indicesSolid[] =
	{
		3,1,0, 0,2,3,    // front
		4,5,7, 4,7,6,    // back
		1,4,0, 4,1,5,    // bottom
		2,7,3, 7,2,6,    // top
		4,2,0, 4,6,2,    // left
		1,3,5, 7,5,3,    // right
	};
	desc.BindFlags = D3D11_BIND_INDEX_BUFFER;
	desc.ByteWidth = sizeof(indicesSolid);
	initData.pSysMem = indicesSolid;
	hr = m_pDevice->CreateBuffer(&desc, &initData, &m_pSolidIB);
	if(FAILED(hr)) {
		Release();
		return hr;
	}

	// index buffer (lines)
	ushort indicesLines[] =
	{
		5,4, 4,6, 6,7,	// front
		1,5, 5,7, 7,3,	// right
		0,1, 1,3, 3,2,	// back
		4,0, 0,2, 2,6	// left
	};
	desc.BindFlags = D3D11_BIND_INDEX_BUFFER;
	desc.ByteWidth = sizeof(indicesLines);
	initData.pSysMem = indicesLines;
	hr = m_pDevice->CreateBuffer(&desc, &initData, &m_pLinesIB);
	if(FAILED(hr)) {
		Release();
		return hr;
	}

	return S_OK;
}

void Box::Release()
{
	if(m_pLinesIB) m_pLinesIB->Release();
	m_pLinesIB = nullptr;
	if(m_pSolidIB) m_pSolidIB->Release();
	m_pSolidIB = nullptr;
	if(m_pVB) m_pVB->Release();
	m_pVB = nullptr;

	m_effect.SafeRelease();

	m_pDevice = nullptr;
}

void Box::RenderLines(const Mat4f& view, const Mat4f& proj,
					  const Vec3f& lightPos,
					  const Vec3f& boxMin, const Vec3f& boxMax,
					  const Vec4f& color,
					  float tubeRadius,
					  bool secondPass)
{
	ID3D11DeviceContext* pContext;
	m_pDevice->GetImmediateContext(&pContext);

	UINT stride = 3 * sizeof(float);
	UINT offset = 0;
	pContext->IASetVertexBuffers(0, 1, &m_pVB, &stride, &offset);
	pContext->IASetIndexBuffer(m_pLinesIB, DXGI_FORMAT_R16_UINT, 0);
	pContext->IASetInputLayout(m_effect.m_pInputLayout);
	pContext->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_LINELIST);

	Mat4f viewInv;
	invert4x4(view, viewInv);
	Vec3f camPos = Vec3f(viewInv.getCol(3));
	Vec3f camRight = Vec3f(viewInv.getCol(0));

	m_effect.m_pmWorldViewVariable->SetMatrix(view);
	m_effect.m_pmProjVariable->SetMatrix(proj);
	m_effect.m_pmWorldViewProjVariable->SetMatrix(proj * view);
	m_effect.m_pvBoxMinVariable->SetFloatVector(boxMin);
	m_effect.m_pvBoxSizeVariable->SetFloatVector(boxMax - boxMin);
	m_effect.m_pvColorVariable->SetFloatVector(color);
	m_effect.m_pfTubeRadiusVariable->SetFloat(tubeRadius);
	m_effect.m_pvLightPosVariable->SetFloatVector(lightPos);
	m_effect.m_pvCamPosVariable->SetFloatVector(camPos);
	m_effect.m_pvCamRightVariable->SetFloatVector(camRight);

	// draw tubes
	uint pass = (tubeRadius == 0.0f) ? 0 : 2;
	if(secondPass) pass++;
	m_effect.m_pTechnique->GetPassByIndex(pass)->Apply(0, pContext);

	pContext->DrawIndexed(24, 0, 0);

	// draw corner spheres
	if(tubeRadius != 0.0f)
	{
		pContext->IASetIndexBuffer(nullptr, DXGI_FORMAT_R16_UINT, 0);
		pContext->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_POINTLIST);

		uint pass = 4;
		if(secondPass) pass++;
		m_effect.m_pTechnique->GetPassByIndex(pass)->Apply(0, pContext);

		pContext->Draw(8, 0);
	}

	pContext->Release();
}


void Box::RenderBrickLines(const Mat4f& view, const Mat4f& proj,
						   const Vec3f& lightPos,
						   const Vec3f& boxMin, const Vec3f& boxMax,
						   const Vec4f& color,
						   const Vec3f& brickSize,
						   float tubeRadius,
						   bool secondPass)
{
	ID3D11DeviceContext* pContext;
	m_pDevice->GetImmediateContext(&pContext);

	UINT stride = 0;
	UINT offset = 0;
	ID3D11Buffer* pNullBuffer = nullptr;
	pContext->IASetVertexBuffers(0, 1, &pNullBuffer, &stride, &offset);
	pContext->IASetIndexBuffer(nullptr, DXGI_FORMAT_R16_UINT, 0);
	pContext->IASetInputLayout(nullptr);
	pContext->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_LINELIST);

	Vec3f brickCount = (boxMax - boxMin) / brickSize;
	Vec3i lineCount((int)ceil(brickCount.x()) + 1, (int)ceil(brickCount.y()) + 1, (int)ceil(brickCount.z()) + 1);

	m_effect.m_pmWorldViewVariable->SetMatrix(view);
	m_effect.m_pmProjVariable->SetMatrix(proj);
	m_effect.m_pmWorldViewProjVariable->SetMatrix(proj * view);
	m_effect.m_pvBoxMinVariable->SetFloatVector(boxMin);
	m_effect.m_pvBoxSizeVariable->SetFloatVector(boxMax - boxMin);
	m_effect.m_pvColorVariable->SetFloatVector(color);
	m_effect.m_pfTubeRadiusVariable->SetFloat(tubeRadius);
	m_effect.m_pvLightPosVariable->SetFloatVector(lightPos);
	m_effect.m_pvBrickSizeVariable->SetFloatVector(brickSize);
	m_effect.m_pvLineCountVariable->SetIntVector(lineCount);

	uint pass = (tubeRadius == 0.0f) ? 6 : 8;
	if(secondPass) pass++;
	m_effect.m_pTechnique->GetPassByIndex(pass)->Apply(0, pContext);

	uint lineCountTotal = lineCount.y() * lineCount.z() + lineCount.x() * lineCount.z() + lineCount.x() * lineCount.y();
	pContext->Draw(lineCountTotal * 2, 0);

	pContext->Release();
}
