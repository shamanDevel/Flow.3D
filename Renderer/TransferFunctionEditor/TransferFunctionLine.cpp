#include "TransferFunctionLine.h"

#include <algorithm>

using namespace tum3D;

#ifndef SAFE_DELETE
#define SAFE_DELETE(p)       { if (p) { delete (p);     (p)=NULL; } }
#endif
#ifndef SAFE_DELETE_ARRAY
#define SAFE_DELETE_ARRAY(p) { if (p) { delete[] (p);   (p)=NULL; } }
#endif
#ifndef SAFE_RELEASE
#define SAFE_RELEASE(p)      { if (p) { (p)->Release(); (p)=NULL; } }
#endif


TransferFunctionLine::TransferFunctionLine(Vec4f v4fLineColor, UINT uiMaxX, UINT uiMaxY) 
	: pVb_(NULL)
	, pVl_(NULL)
	, pFirstPoint_(NULL)
	, uiNumPoints_(0)
	, v2iMaxXY_(uiMaxX, uiMaxY)
	, pData_(NULL)
	, pd3dDevice_(NULL)
	, pFxTechRenderTfEdLines_(NULL)
	, pFxTechRenderTfEdPoints_(NULL)
	, pFxv4fLineColor_(NULL)
	, v4fLineColor_(v4fLineColor)
{
}

TransferFunctionLine::~TransferFunctionLine()
{
	sControlPoint* pCurr_point = pFirstPoint_;

	while (pCurr_point != NULL) 
	{
		sControlPoint *pTmp = pCurr_point;
		pCurr_point			= pCurr_point->pNext_;
		delete(pTmp);
	}

	if ( pData_ != NULL )
		delete pData_;

	SAFE_RELEASE( pVb_ );
	SAFE_RELEASE( pVl_ );
}



HRESULT TransferFunctionLine::onCreateDevice( ID3D11Device* pd3dDevice, ID3DX11Effect* pEffect )
{
	HRESULT hr = S_OK;

	pd3dDevice_ = pd3dDevice;

	hr = createVertexBuffer();

	if(FAILED(hr))
		return hr;

	reset();

	pFxTechRenderTfEdLines_ = pEffect->GetTechniqueByName("tRender_TFLines_");
	if(pFxTechRenderTfEdLines_ == NULL) return E_FAIL;
	pFxTechRenderTfEdPoints_ = pEffect->GetTechniqueByName("tRender_TFPoints_");
	if(pFxTechRenderTfEdPoints_ == NULL) return E_FAIL;
	pFxv4fLineColor_ = pEffect->GetVariableByName("v4fLineColor_")->AsVector();
	if(pFxv4fLineColor_ == NULL) return E_FAIL;


	D3D11_INPUT_ELEMENT_DESC layout_lines_2D[] = { { "POSITION", 0, DXGI_FORMAT_R32G32_FLOAT, 0,  0, D3D11_INPUT_PER_VERTEX_DATA, 0 } };
	D3DX11_PASS_DESC PassDesc;
	pFxTechRenderTfEdLines_->GetPassByIndex( 0 )->GetDesc( &PassDesc );
	return pd3dDevice_->CreateInputLayout( layout_lines_2D, sizeof(layout_lines_2D)/sizeof(D3D11_INPUT_ELEMENT_DESC), PassDesc.pIAInputSignature, PassDesc.IAInputSignatureSize, &pVl_ );
}



void TransferFunctionLine::updateVertex(int index, sControlPoint *pCp)
{
    pData_[index] = Vec2f(float(pCp->v2iPt_.x())/float(v2iMaxXY_.x()), float(pCp->v2iPt_.y())/float(v2iMaxXY_.y()) );
}



void TransferFunctionLine::reset()
{
	sControlPoint *pCurr_point = pFirstPoint_;

	while (pCurr_point != NULL) 
	{
		sControlPoint *pTmp = pCurr_point;
		pCurr_point			= pCurr_point->pNext_;
		delete(pTmp);
	}

	//create first and last point
	Vec2i pStart(0,0);
	pFirstPoint_ = new sControlPoint(pStart, NULL, NULL);
	sControlPoint*	pLastPoint = new sControlPoint(v2iMaxXY_, pFirstPoint_, NULL);
	pFirstPoint_->pNext_ = pLastPoint;
	uiNumPoints_ = 2;

	updateVertex(0, pFirstPoint_);
	updateVertex(1, pLastPoint);

	updateVertexBuffer();
}



int TransferFunctionLine::getControlPointId(Vec2i v2iPos)
{
	int i = -1;
	sControlPoint* pCurr_point = pFirstPoint_;

	// look for point near pos
	while(pCurr_point != NULL) 
	{
		i++;
        Vec2f v2fDist( static_cast<float>(pCurr_point->v2iPt_.x() - v2iPos.x()), static_cast<float>(pCurr_point->v2iPt_.y() - v2iPos.y()) );
		float fLen = ceil(v2fDist.norm());

		if (fLen <= 10)
			return i;

		pCurr_point = pCurr_point->pNext_;
	}

	// no point found near pos:
	return -1;
}



Vec2i TransferFunctionLine::getControlPoint(int iCpId)
{
	sControlPoint* pCurr_point = pFirstPoint_;

	for( ; iCpId > 0 && pCurr_point != NULL; iCpId-- )
		pCurr_point = pCurr_point->pNext_;

	return pCurr_point->v2iPt_;	
}



bool TransferFunctionLine::liesOnTheLine(Vec2i v2iPos)
{
	sControlPoint *pCurr_point = pFirstPoint_;

	while(pCurr_point != NULL)
	{
		if( v2iPos.x() < pCurr_point->v2iPt_.x() )
		{
			float gradient = static_cast<float>( pCurr_point->v2iPt_.y() - pCurr_point->pPrev_->v2iPt_.y()) / static_cast<float>( pCurr_point->v2iPt_.x() - pCurr_point->pPrev_->v2iPt_.x() );

			float y = static_cast<float>(pCurr_point->pPrev_->v2iPt_.y()) + gradient * static_cast<float>( v2iPos.x() - pCurr_point->pPrev_->v2iPt_.x() );
			
			if( abs( static_cast<float>(v2iPos.y()) - y ) < 4 ) 
				return true;

			return false;
		}
		
		pCurr_point = pCurr_point->pNext_;
	}
	return false;
}



void TransferFunctionLine::addControlPoint(Vec2i v2iPt)
{
	assert(v2iPt.x() > 0);

	sControlPoint* pCurr_point = pFirstPoint_;
	int index = 0;

	while(true) 
	{
		if (pCurr_point == NULL)
			return;
		if (pCurr_point->v2iPt_.x() > v2iPt.x())
			break;
		pCurr_point = pCurr_point->pNext_;
		index++;
	}

	sControlPoint *pPrev = pCurr_point->pPrev_;

	
	if (v2iPt.x() > pPrev->v2iPt_.x()) //allow just one point per x-coordinate
	{
		sControlPoint *pNewCp = new sControlPoint(v2iPt, pPrev, pCurr_point);
		pPrev->pNext_		  = pNewCp;
		pCurr_point->pPrev_	  = pNewCp;

		for ( UINT i = uiNumPoints_-1; i >= static_cast<unsigned int>(index); i-- )
			pData_[i+1] = pData_[i];
		updateVertex(index, pNewCp);

		uiNumPoints_++;
	}

	updateVertexBuffer();
}



float TransferFunctionLine::getMinValue()
{
	float fMinVal		= 0;
	sControlPoint *pCp	= pFirstPoint_;

	while ( pCp != NULL && pCp->v2iPt_.y() == 0) 
	{
		fMinVal  = static_cast<float>(pCp->v2iPt_.x());
		pCp		= pCp->pNext_;
	}

	return fMinVal / static_cast<float>(v2iMaxXY_.x());
}



void TransferFunctionLine::fillArray( float* pData, int size, int channel )
{
	sControlPoint *pPrev_point = pFirstPoint_;
	sControlPoint *pNext_point = pPrev_point->pNext_;

	float fValue = static_cast<float>(pPrev_point->v2iPt_.y());
	float fGrad  = (static_cast<float>(pNext_point->v2iPt_.y())-static_cast<float>(pPrev_point->v2iPt_.y())) / (static_cast<float>(pNext_point->v2iPt_.x())-static_cast<float>(pPrev_point->v2iPt_.x()) );

	fValue  /= static_cast<float>(v2iMaxXY_.y());
	fGrad	/= static_cast<float>(v2iMaxXY_.y());

	int x = 0;

	while (true)
	{
		pData[channel + x*4] = fValue;
		fValue += fGrad;
		x++; 

		if (x >= size - 1)
		{
			pData[channel + x*4] = static_cast<float>(pNext_point->v2iPt_.y()) / static_cast<float>(v2iMaxXY_.y());
			return;
		} 
		
		if (x >= pNext_point->v2iPt_.x())
		{
			pPrev_point = pNext_point;
			pNext_point = pPrev_point->pNext_;

			fValue  = static_cast<float>(pPrev_point->v2iPt_.y());
			fGrad	= static_cast<float>(pNext_point->v2iPt_.y() - pPrev_point->v2iPt_.y())/ static_cast<float>(pNext_point->v2iPt_.x() - pPrev_point->v2iPt_.x());
			fValue /= static_cast<float>(v2iMaxXY_.y());
			fGrad  /= static_cast<float>(v2iMaxXY_.y());
		}
	}
}



void TransferFunctionLine::moveControlPoint(int iCpId, Vec2i v2iPt)
{
	assert (iCpId >= 0);
	assert (static_cast<unsigned int>(iCpId) < uiNumPoints_);

	sControlPoint *pCurr_point = pFirstPoint_;

	for (int i = 0; i < iCpId; i++)
		pCurr_point = pCurr_point->pNext_;

	pCurr_point->v2iPt_.y() = std::min( v2iMaxXY_.y(), std::max(v2iPt.y(), 0));

	if (iCpId > 0 && iCpId < static_cast<int>(uiNumPoints_) - 1) 
	{
		if (v2iPt.x() <= pCurr_point->pPrev_->v2iPt_.x())
		{
			pCurr_point->v2iPt_.x() = pCurr_point->pPrev_->v2iPt_.x() + 1;
		}
		else if (v2iPt.x() >= pCurr_point->pNext_->v2iPt_.x())
		{
			pCurr_point->v2iPt_.x() = pCurr_point->pNext_->v2iPt_.x() - 1;
		}
		else
		{
			pCurr_point->v2iPt_.x() = v2iPt.x();
		}
	}

	updateVertex(iCpId, pCurr_point);

	updateVertexBuffer();
}



void TransferFunctionLine::deleteControlPoint(int iCpId)
{
	if (iCpId <= 0 || iCpId >= static_cast<int>(uiNumPoints_)-1 )
		return;

	sControlPoint *pCurr_point = pFirstPoint_;

	for (int i = 0; i < iCpId; i++)
		pCurr_point = pCurr_point->pNext_;

	pCurr_point->pPrev_->pNext_ = pCurr_point->pNext_;
	pCurr_point->pNext_->pPrev_ = pCurr_point->pPrev_;

	// update vertex buffer
	for ( int i = iCpId; i < static_cast<int>(uiNumPoints_); i++ )
		pData_[i] = pData_[i+1];

	uiNumPoints_--;

	updateVertexBuffer();
}



HRESULT TransferFunctionLine::createVertexBuffer()
{
	int iNumElements = v2iMaxXY_.x() + 1;

    pData_ = new Vec2f[ iNumElements ];

	UINT uiSize = sizeof(Vec2f) * iNumElements;

	memset(pData_, 0, uiSize);

	D3D11_BUFFER_DESC bufDesc;
	bufDesc.ByteWidth			= uiSize;
	bufDesc.StructureByteStride	= 0;
	bufDesc.BindFlags			= D3D11_BIND_VERTEX_BUFFER;
	bufDesc.Usage				= D3D11_USAGE_DEFAULT;
	bufDesc.CPUAccessFlags		= 0;
	bufDesc.MiscFlags			= 0;

	D3D11_SUBRESOURCE_DATA srd = {};
	srd.pSysMem = pData_;

	HRESULT hr;

	if(FAILED(hr = pd3dDevice_->CreateBuffer(&bufDesc, &srd, &pVb_)))
		return hr;
	
	return S_OK;
}



void TransferFunctionLine::updateVertexBuffer()
{
	UINT  uiSize = sizeof(Vec2f) * uiNumPoints_;
	D3D11_BOX box = {0, 0, 0, uiSize, 1, 1 };
	ID3D11DeviceContext* pContext = NULL;
	pd3dDevice_->GetImmediateContext( &pContext );
	assert( pContext != NULL );
	pContext->UpdateSubresource( pVb_, 0, &box, pData_, uiSize, 0 );
	SAFE_RELEASE( pContext );
}



void TransferFunctionLine::draw( )
{
	ID3D11DeviceContext* pImmediateContext = NULL;
	pd3dDevice_->GetImmediateContext( &pImmediateContext );

	UINT uiStride = sizeof(Vec2f);
	UINT uiOffset = 0;
	pImmediateContext->IASetVertexBuffers(0, 1, &pVb_, &uiStride, &uiOffset);

	pImmediateContext->IASetInputLayout( pVl_ );

	pFxv4fLineColor_->SetFloatVector( (float*)v4fLineColor_ );

	pImmediateContext->IASetPrimitiveTopology( D3D11_PRIMITIVE_TOPOLOGY_LINESTRIP );
	pFxTechRenderTfEdLines_->GetPassByIndex(0)->Apply(0, pImmediateContext);
	pImmediateContext->Draw( uiNumPoints_, 0 );
		
	pImmediateContext->IASetPrimitiveTopology( D3D11_PRIMITIVE_TOPOLOGY_POINTLIST );
	pFxTechRenderTfEdPoints_->GetPassByIndex(0)->Apply(0, pImmediateContext);
	pImmediateContext->Draw( uiNumPoints_, 0 );

	SAFE_RELEASE( pImmediateContext );
}


