#include <TransferFunction.h>

#include <iostream>
#include <cassert>

#include <Utils.h>

tum3D::Vec4f Lerp(tum3D::Vec4f c0, tum3D::Vec4f c1, float t)
{
	return (1.0f - t) * c0 + t * c1;
}


TransferFunction::TransferFunction()
	: m_device(nullptr), m_tex(nullptr), m_srv(nullptr), m_data(nullptr)
{
	Reset();
}

void TransferFunction::Reset()
{
	m_color0 = tum3D::Vec4f(0.232f, 0.365f, 0.764f, 1.0f);
	m_color1 = tum3D::Vec4f(0.869f, 0.372f, 0.099f, 1.0f);

	m_rangeMin = 0.0f;
	m_rangeMax = 1.0f;

	if (m_device)
		UpdateTexture();
}

bool TransferFunction::CreateResources(ID3D11Device* device)
{
	ReleaseResources();

	m_device = device;

	m_data = new float[TRANSFER_FUNCTION_SIZE * 4];

	D3D11_TEXTURE1D_DESC texDesc;
	texDesc.MipLevels = 1;
	texDesc.ArraySize = 1;
	texDesc.Format = DXGI_FORMAT_R32G32B32A32_FLOAT;
	texDesc.Usage = D3D11_USAGE_DEFAULT;
	texDesc.BindFlags = D3D11_BIND_SHADER_RESOURCE;
	texDesc.CPUAccessFlags = 0;
	texDesc.MiscFlags = 0;
	texDesc.Width = TRANSFER_FUNCTION_SIZE;

	if (FAILED(m_device->CreateTexture1D(&texDesc, NULL, &m_tex)))
		return false;

	if (FAILED(m_device->CreateShaderResourceView(m_tex, NULL, &m_srv)))
		return false;

	return true;
}

void TransferFunction::ReleaseResources()
{
	delete m_data;

	SAFE_RELEASE(m_srv);
	SAFE_RELEASE(m_tex);

	m_data = nullptr;
	m_srv = nullptr;
	m_tex = nullptr;
	m_device = nullptr;
}

void TransferFunction::UpdateTexture()
{
	FillArray();

	UINT srcRowPitch = TRANSFER_FUNCTION_SIZE * 4 * sizeof(float);
	D3D11_BOX box = { 0, 0, 0, TRANSFER_FUNCTION_SIZE, 1, 1 };

	ID3D11DeviceContext* pImmediateContext = NULL;
	m_device->GetImmediateContext(&pImmediateContext);
	assert(pImmediateContext != NULL);

	pImmediateContext->UpdateSubresource(m_tex, 0, &box, m_data, srcRowPitch, 0);

	SAFE_RELEASE(pImmediateContext);
}

void TransferFunction::FillArray()
{
	for (int i = 0; i < TRANSFER_FUNCTION_SIZE; i++)
	{
		float t = i / (float)(TRANSFER_FUNCTION_SIZE - 1);
		tum3D::Vec4f c = Lerp(m_color0, m_color1, t);

		m_data[i * 4 + 0] = c.x();
		m_data[i * 4 + 1] = c.y();
		m_data[i * 4 + 2] = c.z();
		m_data[i * 4 + 3] = c.w();
	}
}