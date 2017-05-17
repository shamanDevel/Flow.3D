#ifndef __TUM3D__LINEBUFFERS_H__
#define __TUM3D__LINEBUFFERS_H__


#include "global.h"

#include <iostream>
#include <string>

#include <cuda_runtime.h>
#include <cuda_d3d11_interop.h>
#include <D3D11.h>


struct LineBuffers
{
	LineBuffers(ID3D11Device* pDevice);
	LineBuffers(ID3D11Device* pDevice, uint lineCount, uint lineLengthMax);
	~LineBuffers();

	uint m_lineCount;
	uint m_lineLengthMax;

	ID3D11Buffer*         m_pVB;
	cudaGraphicsResource* m_pVBCuda;
	ID3D11Buffer*         m_pIB;
	cudaGraphicsResource* m_pIBCuda;
	ID3D11Buffer*         m_pIB_sorted;
	cudaGraphicsResource* m_pIBCuda_sorted;
	uint                  m_indexCountTotal;

	bool Write(const std::string& filename, float posOffset = 0.0f) const;
	bool Write(std::ostream& file, float posOffset = 0.0f) const;
	bool Read(const std::string& filename, int lineIDOverwrite = -1);
	bool Read(std::istream& file, int lineIDOverwrite = -1);

private:
	bool CreateResources();
	void ReleaseResources();

	ID3D11Device* m_pDevice;

	// non-copyable
	LineBuffers(const LineBuffers&);
	LineBuffers& operator=(const LineBuffers&);
};


#endif
