#ifndef __TRANSFERFUNCTIONLINE_H__
#define __TRANSFERFUNCTIONLINE_H__


#include <cmath>
#include <cassert>

#include <D3D11.h>
#include <D3DX11Effect/d3dx11effect.h>

#include <Vec.h>


class TransferFunctionLine
{
public:
	struct sControlPoint 
	{
		sControlPoint(tum3D::Vec2i v2iPt, sControlPoint* pPrev, sControlPoint* pNext) : v2iPt_(v2iPt), pPrev_(pPrev), pNext_(pNext) {};

		tum3D::Vec2i	v2iPt_;
		sControlPoint*	pPrev_;
		sControlPoint*	pNext_;
	};


	TransferFunctionLine(tum3D::Vec4f v4fLineColor, UINT uiMaxX, UINT uiMaxY);
	~TransferFunctionLine();

	HRESULT							onCreateDevice( ID3D11Device* pd3dDevice, ID3DX11Effect* pEffect );

	void							reset();

	sControlPoint*					getFirstPoint() { return pFirstPoint_; }

	int								getControlPointId(tum3D::Vec2i v2iPos);		//closest point to pos (-1 otherwise)
	tum3D::Vec2i					getControlPoint(int iCpId);

	bool							liesOnTheLine(tum3D::Vec2i v2iPos);

	void							addControlPoint(tum3D::Vec2i v2iPt);
	void							moveControlPoint(int iCpId, tum3D::Vec2i v2iPt);
	void							deleteControlPoint(int iCpId);

	void							draw( void );

	void							fillArray( float* pData, int iSize, int iComponent );

	float							getMinValue();

protected:
	HRESULT							createVertexBuffer();
	void							updateVertexBuffer();
	void							updateVertex(int index, sControlPoint *pCp);



	ID3D11Device*					pd3dDevice_;

	ID3DX11EffectTechnique*			pFxTechRenderTfEdLines_;
	ID3DX11EffectTechnique*			pFxTechRenderTfEdPoints_;

	ID3DX11EffectVectorVariable*	pFxv4fLineColor_;


	ID3D11Buffer*					pVb_;
	ID3D11InputLayout*				pVl_;


	sControlPoint*					pFirstPoint_;
	UINT							uiNumPoints_;


	tum3D::Vec2i					v2iMaxXY_;

	tum3D::Vec2f*					pData_;

	tum3D::Vec4f 					v4fLineColor_;
};


#endif /* __TRANSFERFUNCTIONLINE_H__ */
