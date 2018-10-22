#pragma once

#include <d3d11.h>

#include <Vec.h>

#define TRANSFER_FUNCTION_SIZE 500

class TransferFunction
{
public:
	ID3D11Device*				m_device;
	ID3D11Texture1D*			m_tex;
	ID3D11ShaderResourceView*	m_srv;

	tum3D::Vec4f				m_color0;
	tum3D::Vec4f				m_color1;

	float*						m_data;

	float						m_rangeMin;
	float						m_rangeMax;

	TransferFunction();

	bool CreateResources(ID3D11Device* device);

	void Reset();

	void ReleaseResources();

	void UpdateTexture();

	void FillArray();
};