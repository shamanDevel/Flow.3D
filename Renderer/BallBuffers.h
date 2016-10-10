#ifndef __TUM3D__BALLBUFFERS_H__
#define __TUM3D__BALLBUFFERS_H__


#include "global.h"

#include <iostream>
#include <string>

#include <cuda_runtime.h>
#include <cuda_d3d11_interop.h>
#include <D3D11.h>


struct BallBuffers
{
	BallBuffers(ID3D11Device* pDevice);
	BallBuffers(ID3D11Device* pDevice, uint ballCount);
	~BallBuffers();

	uint m_ballCount;

	ID3D11Buffer*         m_pVB;
	cudaGraphicsResource* m_pVBCuda;

	//bool Write(const std::string& filename, float posOffset = 0.0f) const;
	//bool Write(std::ostream& file, float posOffset = 0.0f) const;
	bool Read(const std::string& filename);
	bool Read(std::istream& file);

private:
	bool CreateResources();
	void ReleaseResources();

	ID3D11Device* m_pDevice;

	// non-copyable
	BallBuffers(const BallBuffers&);
	BallBuffers& operator=(const BallBuffers&);
};


#endif
