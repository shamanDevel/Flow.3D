#ifndef __TRANSFERFUNCTIONEDITOR_H__
#define __TRANSFERFUNCTIONEDITOR_H__


#include <iostream>
#include <map>
#include <vector>

#include <D3D11.h>
#include <D3DX11Effect/d3dx11effect.h>

#include <AntTweakBar/AntTweakBar.h>

#include <Vec.h>

#include "TransferFunctionLine.h"

class TransferFunctionEditor;
typedef std::pair<TransferFunctionEditor*, UINT> TransferFunctionEditorCallbackClientData;
typedef std::map< ID3D11Device*, ID3DX11Effect* > TransferFunctionEditorEffectMap;

// A transfer function editor based on AntTweakBar for volume rendering
class TransferFunctionEditor
{
public:
    // Create a TransferFunctionEditor of size uiWidth x uiHeight
	TransferFunctionEditor(UINT uiWidth, UINT uiHeight);
	~TransferFunctionEditor();

    // Callbacks
	HRESULT						onCreateDevice( ID3D11Device* pd3dDevice );
	void						onDestroyDevice();
	void						onResizeSwapChain( UINT uiBBWidth, UINT uiBBHeight );
	void						onReleasingSwapChain() {};

	bool						msgProc( HWND hWnd, UINT uMsg, WPARAM wParam, LPARAM lParam );

	void						onFrameRender( float fTime, float fElapsedTime );

    // Optional: Set volume data histogram for display in background of editor.
    //           pSRV has to be a SRV on a 1D texture representing the histogram.
	void						setHistogramSRV(ID3D11ShaderResourceView* pSRV) {  pHistogramSRV_ = pSRV;  }

    // Access to the transfer function texture/SRV
	ID3D11Texture1D*			getTexture() const { return pTfTex_; }
	ID3D11ShaderResourceView*	getSRV() const { return pTfSRV_; }

    // Loading and saving of transfer functions to files
	void						saveTransferFunction();
	void						loadTransferFunction();
	void						saveTransferFunction(std::ofstream* binary_file);
	void						loadTransferFunction(std::ifstream* binary_file);

	HRESULT						saveTfLegend();

    // More advanced functionality...
	void						setVisible( bool b );
	bool						isVisible() { return bShowTfEditor_; }


	bool						transferFuncTexChanged() { return bTransferFunctionTexChanged_; }


	void						bringChannelToTop( int ch );

	void						reset( );

	void						moveSelectedControlPoint( float fPos, int iDim );
	float						getSelectedControlPointCoord( int iDim );

	int							getSelectedChannel() {return iaCurrChannelsOrder_[0];}

	bool						isPointSelected();

	void						setShowHistogram(bool b)		{bShowHistogram_=b;}
	bool						getShowHistogram()				{return bShowHistogram_;}

	void						setHistogramExpScale(float f)	{fExpHistoScale_=f;}
	float						getHistogramExpScale()			{return fExpHistoScale_;}

	int							getTimestamp() const { return iTimestamp_; }

	float						getTfRangeMin() const	{ return v2fTfRangeMinMax_.x(); }
	float						getTfRangeMax() const	{ return v2fTfRangeMinMax_.y(); }

	void						setTfRangeMin(float val)	{ v2fTfRangeMinMax_.x() = val; }
	void						setTfRangeMax(float val)	{ v2fTfRangeMinMax_.y() = val; }

	float						getAlphaScale() {return fAlphaScale_;}
	void						setAlphaScale(float f) {fAlphaScale_ = f; /*updateTexture();*/}

protected:
	static UINT								s_uiConstructorCallCount_;
	static TransferFunctionEditorEffectMap	s_mapEffect_;

	UINT									uiUniqueID_;

	bool 									bMouseOutside_;
    tum3D::Vec2i 							v2iCursorPos_;

	bool									bTransferFunctionTexChanged_;

	TransferFunctionLine*					pTfLineR_;
	TransferFunctionLine*					pTfLineG_;
	TransferFunctionLine*					pTfLineB_;
	TransferFunctionLine*					pTfLineAlpha_;
	TransferFunctionLine*					paChannels_[4]; 
	int										iaCurrChannelsOrder_[4]; 

	int										iPickedControlPoint_;	
	int										iSelectedControlPoint_;	

	tum3D::Vec2i							v2iSizeUI_;
	tum3D::Vec2i							v2iSizeTfEdt_;
	tum3D::Vec2i							v2iTopLeftCornerUI_;
	tum3D::Vec2i							v2iTopLeftCorner_;

	float*									pTexData_;

	ID3D11Texture1D*						pTfTex_;
	ID3D11ShaderResourceView*				pTfSRV_;

	bool									bShowTfEditor_;

	ID3D11Device*							pd3dDevice_;

	ID3DX11EffectTechnique*					pFxTechRenderTFEWindow_;
	ID3DX11EffectVectorVariable*			pFxv2iEditorTopLeft_;
	ID3DX11EffectVectorVariable*			pFxv2iEditorRes_;
	ID3DX11EffectScalarVariable*			pFxiSelectedPoint_;
	ID3DX11EffectShaderResourceVariable*	pFxSrTfTex_;

	TwBar*									pTfEdtUi_;

	std::vector<TransferFunctionEditorCallbackClientData >	vCallbackClientData_;

	// Histogram
	ID3DX11EffectShaderResourceVariable*	pFxSrHistogram_;
	ID3D11ShaderResourceView*				pHistogramSRV_;
	ID3DX11EffectScalarVariable*			pFxfExpHistoScale_;
	bool									bShowHistogram_;
	float									fExpHistoScale_;


	int										iTimestamp_;

    tum3D::Vec2f 							v2fTfRangeMinMax_;
	float									fAlphaScale_;



	void						drawTransferFunction( );
	bool						handleMessages( HWND hWnd, UINT uMsg, WPARAM wParam, LPARAM lParam );
	void						updateTexture();
	void						initUI();
	void						updateUiPosition( UINT uiBBWidth, UINT uiBBHeight );
};


#endif /* __TRANSFERFUNCTIONEDITOR_H__ */
