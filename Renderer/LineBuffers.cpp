#include "LineBuffers.h"

#include <cassert>
#include <fstream>
#include <numeric>

#include "cudaUtil.h"

#include "TracingCommon.h"


LineBuffers::LineBuffers(ID3D11Device* pDevice)
	: m_lineCount(0), m_lineLengthMax(0)
	, m_pVB(nullptr), m_pVBCuda(nullptr)
	, m_pIB(nullptr), m_pIBCuda(nullptr)
	, m_pIB_sorted(nullptr), m_pIBCuda_sorted(nullptr)
	, m_indexCountTotal(0)
	, m_pDevice(pDevice)
{
}

LineBuffers::LineBuffers(ID3D11Device* pDevice, uint lineCount, uint lineLengthMax)
	: m_lineCount(lineCount), m_lineLengthMax(lineLengthMax)
	, m_pVB(nullptr), m_pVBCuda(nullptr)
	, m_pIB(nullptr), m_pIBCuda(nullptr)
	, m_pIB_sorted(nullptr), m_pIBCuda_sorted(nullptr)
	, m_indexCountTotal(0)
	, m_pDevice(pDevice)
{
	if(!CreateResources())
	{
		printf("LineBuffers::LineBuffers: failed creating resources\n");
	}
}

LineBuffers::~LineBuffers()
{
	ReleaseResources();
}

namespace
{
	const uint magicLength = 11;
	const char magic[magicLength] = "LINEBUFFER";
	const uint version = 2;
}

bool LineBuffers::Write(const std::string& filename, float posOffset) const
{
	std::ofstream file(filename, std::ofstream::binary);
	return Write(file, posOffset);
}

bool LineBuffers::Write(std::ostream& file, float posOffset) const
{
	if(!file.good()) return false;

	// download buffers from GPU
	std::vector<LineVertex> vb(m_lineCount * m_lineLengthMax);
	std::vector<uint> ib(m_lineCount * (m_lineLengthMax - 1) * 2);

	cudaSafeCall(cudaGraphicsMapResources(1, &const_cast<cudaGraphicsResource*>(m_pVBCuda)));
	cudaSafeCall(cudaGraphicsMapResources(1, &const_cast<cudaGraphicsResource*>(m_pIBCuda)));

	LineVertex* dpVB = nullptr;
	cudaSafeCall(cudaGraphicsResourceGetMappedPointer((void**)&dpVB, nullptr, m_pVBCuda));
	cudaSafeCall(cudaMemcpy(vb.data(), dpVB, vb.size() * sizeof(LineVertex), cudaMemcpyDeviceToHost));

	uint* dpIB = nullptr;
	cudaSafeCall(cudaGraphicsResourceGetMappedPointer((void**)&dpIB, nullptr, m_pIBCuda));
	cudaSafeCall(cudaMemcpy(ib.data(), dpIB, ib.size() * sizeof(uint), cudaMemcpyDeviceToHost));

	cudaSafeCall(cudaGraphicsUnmapResources(1, &const_cast<cudaGraphicsResource*>(m_pIBCuda)));
	cudaSafeCall(cudaGraphicsUnmapResources(1, &const_cast<cudaGraphicsResource*>(m_pVBCuda)));

	if(posOffset != 0.0f)
	{
		for(size_t i = 0; i < vb.size(); i++)
		{
			vb[i].Position += posOffset;
		}
	}

	// reconstruct lines from IB
	uint index = 0;
	std::vector<LineVertex> vertices;
	std::vector<uint> lineLengths(m_lineCount);
	for(uint line = 0; line < m_lineCount; line++)
	{
		vertices.push_back(vb[ib[index]]);
		vertices.push_back(vb[ib[index + 1]]);
		lineLengths[line] = 2;
		index += 2;
		while(lineLengths[line] < m_lineLengthMax && ib[index - 1] == ib[index])
		{
			vertices.push_back(vb[ib[index + 1]]);
			lineLengths[line]++;
			index += 2;
		}
	}

	// write magic/version
	file.write(magic, magicLength);
	file.write(reinterpret_cast<const char*>(&version), sizeof(version));

	// write data
	file.write(reinterpret_cast<const char*>(&m_lineCount), sizeof(m_lineCount));
	file.write(reinterpret_cast<const char*>(lineLengths.data()), m_lineCount * sizeof(uint));
	file.write(reinterpret_cast<const char*>(vertices.data()), vertices.size() * sizeof(LineVertex));

	return true;
}

bool LineBuffers::Read(const std::string& filename, int lineIDOverwrite)
{
	std::ifstream file(filename, std::ifstream::binary);
	return Read(file, lineIDOverwrite);
}

bool LineBuffers::Read(std::istream& file, int lineIDOverwrite)
{
	ReleaseResources();

	if(!file.good()) return false;

	// read and check magic/version
	char filemagic[magicLength];
	file.read(filemagic, magicLength);
	if(filemagic[magicLength - 1] != '\0') return false;
	if(strcmp(filemagic, magic) != 0) return false;
	uint fileversion;
	file.read(reinterpret_cast<char*>(&fileversion), sizeof(fileversion));
	if(fileversion != version) return false;

	// read data
	file.read(reinterpret_cast<char*>(&m_lineCount), sizeof(m_lineCount));
	std::vector<uint> lineLengths(m_lineCount);
	file.read(reinterpret_cast<char*>(lineLengths.data()), m_lineCount * sizeof(uint));
	uint vertexCount = std::accumulate(lineLengths.begin(), lineLengths.end(), 0);
	std::vector<LineVertex> vertices(vertexCount);
	file.read(reinterpret_cast<char*>(vertices.data()), vertexCount * sizeof(LineVertex));

	if(lineIDOverwrite >= 0)
	{
		for(size_t i = 0; i < vertices.size(); i++)
		{
			vertices[i].LineID = lineIDOverwrite;
		}
	}

	m_lineLengthMax = *std::max_element(lineLengths.begin(), lineLengths.end());

	// reconstruct vertex and index buffer
	std::vector<LineVertex> vb(m_lineCount * m_lineLengthMax);
	std::vector<uint> ib(m_lineCount * (m_lineLengthMax - 1) * 2);
	uint ibIndex = 0;
	uint verticesIndex = 0;
	for(uint line = 0; line < m_lineCount; line++)
	{
		assert(lineLengths[line] >= 2);

		uint vbIndex = line * m_lineLengthMax;

		vb[vbIndex++] = vertices[verticesIndex++];
		ib[ibIndex++] = vbIndex - 1;

		vb[vbIndex++] = vertices[verticesIndex++];
		ib[ibIndex++] = vbIndex - 1;

		for(uint vertex = 2; vertex < lineLengths[line]; vertex++)
		{
			vb[vbIndex++] = vertices[verticesIndex++];
			ib[ibIndex++] = vbIndex - 2;
			ib[ibIndex++] = vbIndex - 1;
		}
	}
	m_indexCountTotal = ibIndex;

	if(!CreateResources()) return false;

	// upload to GPU
	cudaSafeCall(cudaGraphicsMapResources(1, &const_cast<cudaGraphicsResource*>(m_pVBCuda)));
	cudaSafeCall(cudaGraphicsMapResources(1, &const_cast<cudaGraphicsResource*>(m_pIBCuda)));

	LineVertex* dpVB = nullptr;
	cudaSafeCall(cudaGraphicsResourceGetMappedPointer((void**)&dpVB, nullptr, m_pVBCuda));
	cudaSafeCall(cudaMemcpy(dpVB, vb.data(), vb.size() * sizeof(LineVertex), cudaMemcpyHostToDevice));

	uint* dpIB = nullptr;
	cudaSafeCall(cudaGraphicsResourceGetMappedPointer((void**)&dpIB, nullptr, m_pIBCuda));
	cudaSafeCall(cudaMemcpy(dpIB, ib.data(), ib.size() * sizeof(uint), cudaMemcpyHostToDevice));

	cudaSafeCall(cudaGraphicsUnmapResources(1, &const_cast<cudaGraphicsResource*>(m_pIBCuda)));
	cudaSafeCall(cudaGraphicsUnmapResources(1, &const_cast<cudaGraphicsResource*>(m_pVBCuda)));

	return true;
}


bool LineBuffers::CreateResources()
{
	HRESULT hr;

	D3D11_BUFFER_DESC bufDesc = {};

	bufDesc.BindFlags = D3D11_BIND_VERTEX_BUFFER;
	bufDesc.ByteWidth = m_lineCount * m_lineLengthMax * sizeof(LineVertex);
	bufDesc.Usage = D3D11_USAGE_DEFAULT;
	if(FAILED(hr = m_pDevice->CreateBuffer(&bufDesc, nullptr, &m_pVB)))
	{
		ReleaseResources();
		return false;
	}
	cudaSafeCall(cudaGraphicsD3D11RegisterResource(&m_pVBCuda, m_pVB, cudaGraphicsRegisterFlagsNone));

	bufDesc.BindFlags = D3D11_BIND_INDEX_BUFFER;
	bufDesc.ByteWidth = m_lineCount * (m_lineLengthMax - 1) * 2 * sizeof(uint);
	bufDesc.Usage = D3D11_USAGE_DEFAULT;
	if(FAILED(hr = m_pDevice->CreateBuffer(&bufDesc, nullptr, &m_pIB)))
	{
		ReleaseResources();
		return false;
	}
	cudaSafeCall(cudaGraphicsD3D11RegisterResource(&m_pIBCuda, m_pIB, cudaGraphicsRegisterFlagsNone));
	if (FAILED(hr = m_pDevice->CreateBuffer(&bufDesc, nullptr, &m_pIB_sorted)))
	{
		ReleaseResources();
		return false;
	}
	cudaSafeCall(cudaGraphicsD3D11RegisterResource(&m_pIBCuda_sorted, m_pIB_sorted, cudaGraphicsRegisterFlagsNone));

	return true;
}

void LineBuffers::ReleaseResources()
{
	m_lineCount = 0;
	m_lineLengthMax = 0;
	m_indexCountTotal = 0;

	if (m_pIBCuda_sorted)
	{
		cudaSafeCall(cudaGraphicsUnregisterResource(m_pIBCuda_sorted));
		m_pIBCuda_sorted = nullptr;
	}
	if (m_pIB_sorted)
	{
		m_pIB_sorted->Release();
		m_pIB_sorted = nullptr;
	}

	if(m_pIBCuda)
	{
		cudaSafeCall(cudaGraphicsUnregisterResource(m_pIBCuda));
		m_pIBCuda = nullptr;
	}
	if(m_pIB)
	{
		m_pIB->Release();
		m_pIB = nullptr;
	}

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
