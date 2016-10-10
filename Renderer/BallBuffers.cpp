#include "BallBuffers.h"

#include <cassert>
#include <fstream>
#include <numeric>

#include "cudaUtil.h"

#include "TracingCommon.h"


BallBuffers::BallBuffers(ID3D11Device* pDevice)
	: m_ballCount(0)
	, m_pVB(nullptr), m_pVBCuda(nullptr)
	, m_pDevice(pDevice)
{
}

BallBuffers::BallBuffers(ID3D11Device* pDevice, uint ballCount)
	: m_ballCount(ballCount)
	, m_pVB(nullptr), m_pVBCuda(nullptr)
	, m_pDevice(pDevice)
{
	if(!CreateResources())
	{
		printf("BallBuffers::BallBuffers: failed creating resources\n");
	}
}

BallBuffers::~BallBuffers()
{
	ReleaseResources();
}

//namespace
//{
//	const uint magicLength = 11;
//	const char magic[magicLength] = "BALLBUFFER";
//	const uint version = 1;
//}

bool BallBuffers::Read(const std::string& filename)
{
	std::ifstream file(filename, std::ifstream::binary);
	return Read(file);
}

bool BallBuffers::Read(std::istream& file)
{
	ReleaseResources();

	if(!file.good()) return false;

	//// read and check magic/version
	//char filemagic[magicLength];
	//file.read(filemagic, magicLength);
	//if(filemagic[magicLength - 1] != '\0') return false;
	//if(strcmp(filemagic, magic) != 0) return false;
	//uint fileversion;
	//file.read(reinterpret_cast<char*>(&fileversion), sizeof(fileversion));
	//if(fileversion != version) return false;

	// get file size
	size_t posOld = file.tellg();
	file.seekg(0, std::istream::end);
	size_t sizeBytes = file.tellg();
	file.seekg(posOld, std::istream::beg);

	m_ballCount = uint(sizeBytes / sizeof(BallVertex));

	// read data
	std::vector<float> posX(m_ballCount);
	std::vector<float> posY(m_ballCount);
	std::vector<float> posZ(m_ballCount);
	file.read(reinterpret_cast<char*>(posX.data()), m_ballCount * sizeof(float));
	file.read(reinterpret_cast<char*>(posY.data()), m_ballCount * sizeof(float));
	file.read(reinterpret_cast<char*>(posZ.data()), m_ballCount * sizeof(float));


	// build (interleaved) vertex buffer
	std::vector<BallVertex> vb(m_ballCount);
	for(uint i = 0; i < m_ballCount; i++)
	{
		vb[i].Position.x = posX[i];
		vb[i].Position.y = posY[i];
		vb[i].Position.z = posZ[i];
	}

	if(!CreateResources()) return false;

	// upload to GPU
	cudaSafeCall(cudaGraphicsMapResources(1, &const_cast<cudaGraphicsResource*>(m_pVBCuda)));

	BallVertex* dpVB = nullptr;
	cudaSafeCall(cudaGraphicsResourceGetMappedPointer((void**)&dpVB, nullptr, m_pVBCuda));
	cudaSafeCall(cudaMemcpy(dpVB, vb.data(), vb.size() * sizeof(BallVertex), cudaMemcpyHostToDevice));
	cudaSafeCall(cudaGraphicsUnmapResources(1, &const_cast<cudaGraphicsResource*>(m_pVBCuda)));

	return true;
}


bool BallBuffers::CreateResources()
{
	HRESULT hr;

	D3D11_BUFFER_DESC bufDesc = {};

	bufDesc.BindFlags = D3D11_BIND_VERTEX_BUFFER;
	bufDesc.ByteWidth = m_ballCount * sizeof(BallVertex);
	bufDesc.Usage = D3D11_USAGE_DEFAULT;
	if(FAILED(hr = m_pDevice->CreateBuffer(&bufDesc, nullptr, &m_pVB)))
	{
		ReleaseResources();
		return false;
	}
	cudaSafeCall(cudaGraphicsD3D11RegisterResource(&m_pVBCuda, m_pVB, cudaGraphicsRegisterFlagsNone));

	return true;
}

void BallBuffers::ReleaseResources()
{
	m_ballCount = 0;

	if(m_pVBCuda)
	{
		cudaGraphicsUnregisterResource(m_pVBCuda);
		m_pVBCuda = nullptr;
	}
	if(m_pVB)
	{
		m_pVB->Release();
		m_pVB = nullptr;
	}
}
