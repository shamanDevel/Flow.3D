//#ifndef __TRANSFERFUNCTIONEDITOR_H__
//#define __TRANSFERFUNCTIONEDITOR_H__
//
//
//#include <iostream>
//#include <map>
//#include <vector>
//#include <string>
//
//#include <D3D11.h>
//#include <D3DX11Effect/d3dx11effect.h>
//
////#include <AntTweakBar/AntTweakBar.h>
//
//#include <Vec.h>
//
//#include "TransferFunctionLine.h"
//
//class TransferFunctionEditor;
//typedef std::pair<TransferFunctionEditor*, UINT> TransferFunctionEditorCallbackClientData;
//typedef std::map< ID3D11Device*, ID3DX11Effect* > TransferFunctionEditorEffectMap;
//
//// A transfer function editor based on AntTweakBar for volume rendering
//class TransferFunctionEditor
//{
//public:
//    // Create a TransferFunctionEditor of size uiWidth x uiHeight with multiple instances.
//	// Each instance is completely independent of each other, in terms of scaling an the transfer function line.
//	//  The names of these instances as passed, these are displayed in the ui. 
//	//  In the API, you access them by the zero-based index.
//	TransferFunctionEditor(UINT uiWidth, UINT uiHeight, std::vector<std::string> instanceNames);
//	~TransferFunctionEditor();
//
//    // Callbacks
//	HRESULT						onCreateDevice( ID3D11Device* pd3dDevice );
//	void						onDestroyDevice();
//	void						onResizeSwapChain( UINT uiBBWidth, UINT uiBBHeight );
//	void						onReleasingSwapChain() {};
//
//	bool						msgProc( HWND hWnd, UINT uMsg, WPARAM wParam, LPARAM lParam );
//
//	void						onFrameRender( float fTime, float fElapsedTime );
//
//    // Optional: Set volume data histogram for display in background of editor.
//    //           pSRV has to be a SRV on a 1D texture representing the histogram.
//	void						setHistogramSRV(ID3D11ShaderResourceView* pSRV) {  pHistogramSRV_ = pSRV;  }
//
//    // Access to the transfer function texture/SRV
//	ID3D11Texture1D*			getTexture(int instance) const { return pTfTex_[instance]; }
//	ID3D11ShaderResourceView*	getSRV(int instance) const { return pTfSRV_[instance]; }
//
//    // Loading and saving of transfer functions to files
//	void						saveTransferFunction(int instance);
//	void						loadTransferFunction(int instance);
//	void						saveTransferFunction(int instance, std::ofstream* binary_file);
//	void						loadTransferFunction(int instance, std::ifstream* binary_file);
//
//	HRESULT						saveTfLegend();
//
//    // More advanced functionality...
//	void						setVisible( bool b );
//	bool						isVisible() { return bShowTfEditor_; }
//
//	int							getCurrentInstance() { return iCurrentInstance_; }
//	void						setCurrentInstance(int instance);
//
//	bool						transferFuncTexChanged(int instance) { return bTransferFunctionTexChanged_[instance]; }
//
//
//	void						bringChannelToTop(int instance, int ch );
//
//	void						reset(int instance);
//
//	void						moveSelectedControlPoint(int instance, float fPos, int iDim );
//	float						getSelectedControlPointCoord(int instance, int iDim );
//
//	int							getSelectedChannel(int instance) {return iaCurrChannelsOrder_[instance][0];}
//
//	bool						isPointSelected();
//
//	void						setShowHistogram(bool b)		{bShowHistogram_=b;}
//	bool						getShowHistogram()				{return bShowHistogram_;}
//
//	void						setHistogramExpScale(float f)	{fExpHistoScale_=f;}
//	float						getHistogramExpScale()			{return fExpHistoScale_;}
//
//	int							getTimestamp() const { return iTimestamp_; }
//
//	float						getTfRangeMin(int instance) const	{ return v2fTfRangeMinMax_[instance].x(); }
//	float						getTfRangeMax(int instance) const	{ return v2fTfRangeMinMax_[instance].y(); }
//
//	void						setTfRangeMin(int instance, float val)	{ v2fTfRangeMinMax_[instance].x() = val; }
//	void						setTfRangeMax(int instance, float val)	{ v2fTfRangeMinMax_[instance].y() = val; }
//
//	float						getAlphaScale(int instance) {return fAlphaScale_[instance];}
//	void						setAlphaScale(int instance, float f) {fAlphaScale_[instance] = f; /*updateTexture();*/}
//
//protected:
//	static UINT								s_uiConstructorCallCount_;
//	static TransferFunctionEditorEffectMap	s_mapEffect_;
//
//	UINT									uiUniqueID_;
//
//	bool 									bMouseOutside_;
//    tum3D::Vec2i 							v2iCursorPos_;
//
//	size_t									iInstanceCount_;
//	std::vector<std::string>				vsInstanceNames_;
//	int                                     iCurrentInstance_;
//
//	std::vector<bool>						bTransferFunctionTexChanged_;
//
//	std::vector<TransferFunctionLine*>		pTfLineR_;
//	std::vector<TransferFunctionLine*>		pTfLineG_;
//	std::vector<TransferFunctionLine*>		pTfLineB_;
//	std::vector<TransferFunctionLine*>		pTfLineAlpha_;
//	std::vector<TransferFunctionLine**>		paChannels_;
//	std::vector<int*>						iaCurrChannelsOrder_;
//
//	int										iPickedControlPoint_;	
//	int										iSelectedControlPoint_;	
//
//	tum3D::Vec2i							v2iSizeUI_;
//	tum3D::Vec2i							v2iSizeTfEdt_;
//	tum3D::Vec2i							v2iTopLeftCornerUI_;
//	tum3D::Vec2i							v2iTopLeftCorner_;
//
//	std::vector<float*>						pTexData_;
//
//	std::vector<ID3D11Texture1D*>			pTfTex_;
//	std::vector<ID3D11ShaderResourceView*>	pTfSRV_;
//
//	bool									bShowTfEditor_;
//
//	ID3D11Device*							pd3dDevice_;
//
//	ID3DX11EffectTechnique*					pFxTechRenderTFEWindow_;
//	ID3DX11EffectVectorVariable*			pFxv2iEditorTopLeft_;
//	ID3DX11EffectVectorVariable*			pFxv2iEditorRes_;
//	ID3DX11EffectScalarVariable*			pFxiSelectedPoint_;
//	ID3DX11EffectShaderResourceVariable*	pFxSrTfTex_;
//
//	TwBar*									pTfEdtUi_;
//
//	std::vector<TransferFunctionEditorCallbackClientData >	vCallbackClientData_;
//
//	// Histogram
//	ID3DX11EffectShaderResourceVariable*	pFxSrHistogram_;
//	ID3D11ShaderResourceView*				pHistogramSRV_;
//	ID3DX11EffectScalarVariable*			pFxfExpHistoScale_;
//	bool									bShowHistogram_;
//	float									fExpHistoScale_;
//
//
//	int										iTimestamp_;
//
//	std::vector<tum3D::Vec2f>				v2fTfRangeMinMax_;
//	std::vector<float>						fAlphaScale_;
//
//
//
//	void						drawTransferFunction( );
//	bool						handleMessages( HWND hWnd, UINT uMsg, WPARAM wParam, LPARAM lParam );
//	void						updateTexture(int instance);
//	void						initUI();
//	void						updateUiPosition( UINT uiBBWidth, UINT uiBBHeight );
//};
//
//
//#endif /* __TRANSFERFUNCTIONEDITOR_H__ */
