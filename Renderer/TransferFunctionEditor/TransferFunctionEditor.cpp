#include "TransferFunctionEditor.h"

#include <strsafe.h>
#include <iostream>
#include <fstream>
#include <sstream>

#include <commdlg.h>

//FIXME TODO all alphaScale stuff

using namespace std; //FIXME
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
#ifndef SAFE_RELEASE_REF
#define SAFE_RELEASE_REF(x)	{	if (x != NULL) {UINT uRef = x->Release(); if (uRef == 0) {x = NULL;} } }
#endif 


enum eUIControls_Dx11TfEditor
{
	UI_DX11TFEDT_COMBO_SELECTED_INSTANCE=0,
	UI_DX11TFEDT_COMBO_SELECTED_CHANNEL,
	UI_DX11TFEDT_BTN_RESET,
	UI_DX11TFEDT_FLOAT_POS_X,
	UI_DX11TFEDT_FLOAT_POS_Y,
	UI_DX11TFEDT_FLOAT_SCALE_ALPHA,
	UI_DX11TFEDT_FLOAT_TFRANGE_MIN,
	UI_DX11TFEDT_FLOAT_TFRANGE_MAX,	
	UI_DX11TFEDT_BOOL_SHOW_HISTOGRAM,
	UI_DX11TFEDT_FLOAT_SCALE_HISTOGRAM,
	UI_DX11TFEDT_BTN_LOAD_TF,
	UI_DX11TFEDT_BTN_SAVE_TF,
	UI_DX11TFEDT_NUM_CONTROLS,
	UI_DX11TFEDT_FORCE32 = 0xFFFFFFFF
};


#define NONE_PICKED			-1

#define RED_CHANNEL 		 0
#define GREEN_CHANNEL 		 1
#define BLUE_CHANNEL 		 2
#define ALPHA_CHANNEL 		 3


#define DX11TFEDITOR_CELL_SPACING 6


UINT TransferFunctionEditor::s_uiConstructorCallCount_ = 0;
TransferFunctionEditorEffectMap TransferFunctionEditor::s_mapEffect_;


void TW_CALL Dx11TfEditorGetVarCallback( void *value, void *clientData )
{
	TransferFunctionEditorCallbackClientData* cd	= reinterpret_cast<TransferFunctionEditorCallbackClientData*>( clientData );
	TransferFunctionEditor* pTfEdt				= cd->first;
	eUIControls_Dx11TfEditor eUiId		= static_cast<eUIControls_Dx11TfEditor>(cd->second);

	int inst = pTfEdt->getCurrentInstance();

	switch( eUiId )
    {
		case UI_DX11TFEDT_COMBO_SELECTED_INSTANCE :
		{
			*((int*)value) = inst;
			break;
		}

		case UI_DX11TFEDT_COMBO_SELECTED_CHANNEL :
		{
			int iChannelId = pTfEdt->getSelectedChannel(inst);
			*((int*)value) = iChannelId;
			break;
		}

		case UI_DX11TFEDT_FLOAT_POS_X :
		case UI_DX11TFEDT_FLOAT_POS_Y :
		{
			*((float*)value) = pTfEdt->getSelectedControlPointCoord(inst, eUiId == UI_DX11TFEDT_FLOAT_POS_X ? 0 : 1 );
			break;
		}

		case UI_DX11TFEDT_FLOAT_SCALE_ALPHA : 
		{
			*((float*)value) = pTfEdt->getAlphaScale(inst);
			break;
		}

		case UI_DX11TFEDT_FLOAT_TFRANGE_MIN : 
		{
			*((float*)value) = pTfEdt->getTfRangeMin(inst);
			break;
		}
	
		case UI_DX11TFEDT_FLOAT_TFRANGE_MAX : 
		{
			*((float*)value) = pTfEdt->getTfRangeMax(inst);
			break;
		}

		case UI_DX11TFEDT_BOOL_SHOW_HISTOGRAM : 
		{
			*((bool*)value) = pTfEdt->getShowHistogram();
			break;
		}
		
		case UI_DX11TFEDT_FLOAT_SCALE_HISTOGRAM : 
		{
			*((float*)value) = pTfEdt->getHistogramExpScale();
			break;
		}

		default: return;	
	}
}



void TW_CALL Dx11TfEditorSetVarCallback( const void *value, void *clientData )
{
	TransferFunctionEditorCallbackClientData* cd	= reinterpret_cast<TransferFunctionEditorCallbackClientData*>( clientData );
	TransferFunctionEditor* pTfEdt				= cd->first;
	eUIControls_Dx11TfEditor eUiId		= static_cast<eUIControls_Dx11TfEditor>(cd->second);

	int inst = pTfEdt->getCurrentInstance();

	switch( eUiId )
    {
		case UI_DX11TFEDT_COMBO_SELECTED_INSTANCE :
		{
			pTfEdt->setCurrentInstance(*((int*)value));
			break;
		}

		case UI_DX11TFEDT_COMBO_SELECTED_CHANNEL :
		{
			pTfEdt->bringChannelToTop(inst, *((int*)value) );
			break;
		}

		case UI_DX11TFEDT_FLOAT_POS_X :
		case UI_DX11TFEDT_FLOAT_POS_Y :
		{
			pTfEdt->moveSelectedControlPoint(inst, *((float*)value), eUiId == UI_DX11TFEDT_FLOAT_POS_X ? 0 : 1 );
			break;
		}

		case UI_DX11TFEDT_FLOAT_SCALE_ALPHA : 
		{
			pTfEdt->setAlphaScale(inst, *((float*)value) );
			break;
		}

		case UI_DX11TFEDT_FLOAT_TFRANGE_MIN : 
		{
			pTfEdt->setTfRangeMin(inst, *((float*)value) );
			break;
		}
	
		case UI_DX11TFEDT_FLOAT_TFRANGE_MAX : 
		{
			pTfEdt->setTfRangeMax(inst, *((float*)value) );
			break;
		}
		
		case UI_DX11TFEDT_BOOL_SHOW_HISTOGRAM : 
		{
			pTfEdt->setShowHistogram(*((bool*)value) );
			break;
		}
		
		case UI_DX11TFEDT_FLOAT_SCALE_HISTOGRAM : 
		{
			pTfEdt->setHistogramExpScale( *((float*)value) );
			break;
		}

		default: return;	
	}
}



void TW_CALL Dx11TfEditorButtonCallback( void *clientData )
{
	TransferFunctionEditorCallbackClientData* cd	= reinterpret_cast<TransferFunctionEditorCallbackClientData*>( clientData );
	TransferFunctionEditor* pTfEdt				= cd->first;
	eUIControls_Dx11TfEditor eUiId		= static_cast<eUIControls_Dx11TfEditor>(cd->second);

	int inst = pTfEdt->getCurrentInstance();

	switch( eUiId )
    {
		case UI_DX11TFEDT_BTN_RESET :
		{
			pTfEdt->reset(inst);
			break;
		}

		case UI_DX11TFEDT_BTN_SAVE_TF :
		{
			pTfEdt->saveTransferFunction(inst);
			break;
		}

		case UI_DX11TFEDT_BTN_LOAD_TF :
		{
			pTfEdt->loadTransferFunction(inst);
			break;
		}

		default: return;	
	}
}

//-----------------------------------------------------------------------------
// Constructor transfer function editor
//-----------------------------------------------------------------------------
TransferFunctionEditor::TransferFunctionEditor(UINT uiWidth, UINT uiHeight, std::vector<std::string> instanceNames)
	: bMouseOutside_(true)
	, v2iCursorPos_(0, 0)
	, iInstanceCount_(instanceNames.size())
	, vsInstanceNames_(instanceNames)
	, iCurrentInstance_(0)
	, bTransferFunctionTexChanged_(instanceNames.size())
	, iPickedControlPoint_(NONE_PICKED)
	, iSelectedControlPoint_(NONE_PICKED)
	, v2iSizeUI_(uiWidth, uiHeight )
	, v2iTopLeftCornerUI_(0,0)
	, v2iTopLeftCorner_(0,0)
	, fAlphaScale_(instanceNames.size())
	, iaCurrChannelsOrder_(instanceNames.size())
	, pTexData_(instanceNames.size())
	, pTfTex_(instanceNames.size())
	, pTfSRV_(instanceNames.size())
	, bShowTfEditor_(true)
	, pd3dDevice_(NULL)
	, pFxTechRenderTFEWindow_(NULL)
	, pFxv2iEditorTopLeft_(NULL)
	, pFxv2iEditorRes_(NULL)
	, pFxiSelectedPoint_(NULL)
	, pFxSrTfTex_(NULL)
	, pTfEdtUi_(NULL)
	, pFxSrHistogram_(NULL)
	, pHistogramSRV_(NULL)
	, pFxfExpHistoScale_(NULL)
	, fExpHistoScale_(0.2f)
	, bShowHistogram_(true)
	, iTimestamp_(0)
	, v2fTfRangeMinMax_(instanceNames.size())
    , v2iSizeTfEdt_(instanceNames.size())
{	
	uiUniqueID_ = ++s_uiConstructorCallCount_;
	assert(instanceNames.size() > 0);

    v2iSizeTfEdt_ = Vec2i(v2iSizeUI_.x()-(2*DX11TFEDITOR_CELL_SPACING), v2iSizeUI_.y()-(2*DX11TFEDITOR_CELL_SPACING));

	for (int inst = 0; inst < iInstanceCount_; ++inst) {
		bTransferFunctionTexChanged_[inst] = false;
		fAlphaScale_[inst] = 1.0f;
		v2fTfRangeMinMax_[inst] = Vec2f(0.0f, 1.0f);

		iaCurrChannelsOrder_[inst] = new int[4];
		for (int k = 0; k<4; k++)
			iaCurrChannelsOrder_[inst][k] = k;

		pTexData_[inst] = new float[v2iSizeTfEdt_.x() * 4];
		memset(pTexData_[inst], 0, v2iSizeTfEdt_.x() * 4 * sizeof(float));
	}

	vCallbackClientData_.resize(0);
	for(UINT i=0; i<UI_DX11TFEDT_NUM_CONTROLS; i++)
		vCallbackClientData_.push_back( TransferFunctionEditorCallbackClientData(this, i) );
}


//-----------------------------------------------------------------------------
// Destructor for transfer function editor
//-----------------------------------------------------------------------------
TransferFunctionEditor::~TransferFunctionEditor()
{
	for (int inst = 0; inst < iInstanceCount_; ++inst) {
		if (pTexData_[inst] != NULL)
			delete[] pTexData_[inst];
	}

	assert(pd3dDevice_ == NULL);
}



void TransferFunctionEditor::updateUiPosition( UINT uiBBWidth, UINT uiBBHeight )
{
	INT32 dim[2] = {220, v2iSizeUI_.y()};
	INT32 iSizeX = dim[0]+v2iSizeUI_.x(); //width of antTweakBar + TFEDT
	
	INT32 iBorder = 10;

	INT32 pos[2] = { std::max(0, INT32(uiBBWidth - iSizeX - iBorder) ), std::max(0, INT32(uiBBHeight - dim[1] - iBorder)) };

	v2iTopLeftCornerUI_.x()	= (pos[0]+(dim[0]));
	v2iTopLeftCornerUI_.y()	= max(0,pos[1]-1);

	// update editor top left corner 
	v2iTopLeftCorner_.x() = v2iTopLeftCornerUI_.x() + DX11TFEDITOR_CELL_SPACING;
	v2iTopLeftCorner_.y() = v2iTopLeftCornerUI_.y() + DX11TFEDITOR_CELL_SPACING;

	if(pTfEdtUi_==NULL)
		return;

	TwSetParam(pTfEdtUi_, NULL, "size",		TW_PARAM_INT32, 2, dim);
	TwSetParam(pTfEdtUi_, NULL, "position", TW_PARAM_INT32, 2, pos);
}



void TransferFunctionEditor::onResizeSwapChain( UINT uiBBWidth, UINT uiBBHeight )
{
	updateUiPosition( uiBBWidth, uiBBHeight );
}



HRESULT TransferFunctionEditor::onCreateDevice(ID3D11Device* pd3dDevice)
{
	HRESULT hr;

	assert(pd3dDevice != NULL);

	pd3dDevice_ = pd3dDevice;

	paChannels_.resize(iInstanceCount_);
	pTfLineAlpha_.resize(iInstanceCount_);
	pTfLineR_.resize(iInstanceCount_);
	pTfLineG_.resize(iInstanceCount_);
	pTfLineB_.resize(iInstanceCount_);

	for (int inst = 0; inst < iInstanceCount_; ++inst) {

		pTfLineAlpha_[inst] = new TransferFunctionLine(Vec4f(0, 1, 1, 1), v2iSizeTfEdt_.x() - 1, v2iSizeTfEdt_.y() - 1);
		pTfLineR_[inst] = new TransferFunctionLine(Vec4f(1, 0, 0, 1), v2iSizeTfEdt_.x() - 1, v2iSizeTfEdt_.y() - 1);
		pTfLineG_[inst] = new TransferFunctionLine(Vec4f(0, 1, 0, 1), v2iSizeTfEdt_.x() - 1, v2iSizeTfEdt_.y() - 1);
		pTfLineB_[inst] = new TransferFunctionLine(Vec4f(0, 0, 1, 1), v2iSizeTfEdt_.x() - 1, v2iSizeTfEdt_.y() - 1);

		paChannels_[inst] = new TransferFunctionLine*[4];
		paChannels_[inst][0] = pTfLineR_[inst];
		paChannels_[inst][1] = pTfLineG_[inst];
		paChannels_[inst][2] = pTfLineB_[inst];
		paChannels_[inst][3] = pTfLineAlpha_[inst];

	}

	ID3DX11Effect* pEffect = NULL;
	TransferFunctionEditorEffectMap::const_iterator it = s_mapEffect_.find( pd3dDevice_ );

	if( it == s_mapEffect_.end() || (*it).second == NULL )
	{
		// read compiled effect file
		std::string filename("TransferFunctionEditor.fxo");
		// try current working directory first
		std::ifstream file(filename.c_str(), std::ios::in | std::ios::binary);

		if (file.fail()) {
			// if that didn't work, try exe path
			char buffer[MAX_PATH + 1];
			GetModuleFileNameA(NULL, buffer, MAX_PATH + 1);
			std::string path(buffer);
			size_t lastSlash = path.find_last_of("/\\");
			path = path.substr(0, lastSlash);

			file.clear();
			file.open((path + "\\" + filename).c_str(), std::ios::in | std::ios::binary);

			if(file.fail()) {
				// if that still didn't work, give up
				return E_FAIL;
			}
		}

		file.seekg(0, std::ios::end);
		size_t bytesize = file.tellg();
		file.seekg(0, std::ios::beg);

		std::vector<char> effectData(bytesize);
		file.read(effectData.data(), effectData.size());

		if (file.fail()) {
			return E_FAIL;
		}

		file.close();

		if(FAILED(hr = D3DX11CreateEffectFromMemory(effectData.data(), effectData.size(), 0, pd3dDevice_, &pEffect)))
			return hr;

		if( pEffect != NULL && pEffect->IsValid() )
		{
			s_mapEffect_[ pd3dDevice_ ] = pEffect;
		}
	}
	else
	{
		pEffect = (*it).second;
		(*it).second->AddRef();
	}

	pFxTechRenderTFEWindow_ = pEffect->GetTechniqueByName("tRender_TFWindow_");
	pFxv2iEditorTopLeft_ = pEffect->GetVariableByName("v2iTfEditorTopLeft_")->AsVector();
	pFxv2iEditorRes_ = pEffect->GetVariableByName("v2iTfEditorRes_")->AsVector();
	pFxiSelectedPoint_ = pEffect->GetVariableByName("iSelectedPoint_")->AsScalar();
	pFxSrTfTex_ = pEffect->GetVariableByName("txTf_")->AsShaderResource();
	pFxSrHistogram_ = pEffect->GetVariableByName("txHistogram_")->AsShaderResource();
	pFxfExpHistoScale_ = pEffect->GetVariableByName("fExpScaleHisto_")->AsScalar();

	D3D11_TEXTURE1D_DESC texDesc;
	texDesc.MipLevels = 1;
	texDesc.ArraySize = 1;
	texDesc.Format = DXGI_FORMAT_R32G32B32A32_FLOAT;
	texDesc.Usage = D3D11_USAGE_DEFAULT;
	texDesc.BindFlags = D3D11_BIND_SHADER_RESOURCE;
	texDesc.CPUAccessFlags = 0;
	texDesc.MiscFlags = 0;
	texDesc.Width = v2iSizeTfEdt_.x();

	pTfTex_.resize(iInstanceCount_);
	pTfSRV_.resize(iInstanceCount_);
	for (int inst = 0; inst < iInstanceCount_; ++inst) {

		if (FAILED(hr = pTfLineAlpha_[inst]->onCreateDevice(pd3dDevice_, pEffect)))
			return hr;

		if (FAILED(hr = pTfLineR_[inst]->onCreateDevice(pd3dDevice_, pEffect)))
			return hr;

		if (FAILED(hr = pTfLineG_[inst]->onCreateDevice(pd3dDevice_, pEffect)))
			return hr;

		if (FAILED(hr = pTfLineB_[inst]->onCreateDevice(pd3dDevice_, pEffect)))
			return hr;

		if (FAILED(hr = pd3dDevice_->CreateTexture1D(&texDesc, NULL, &pTfTex_[inst])))
			return hr;

		if (FAILED(hr = pd3dDevice_->CreateShaderResourceView(pTfTex_[inst], NULL, &pTfSRV_[inst])))
			return hr;

		updateTexture(inst);

	}

	initUI();

	return S_OK;
}

void TransferFunctionEditor::onDestroyDevice()
{
	for (int inst = 0; inst < iInstanceCount_; ++inst) {
		SAFE_RELEASE(pTfSRV_[inst]);
		SAFE_RELEASE(pTfTex_[inst]);
	}

	if(pTfEdtUi_!=NULL) {
		TwDeleteBar( pTfEdtUi_ );
		pTfEdtUi_ = NULL;
	}

	TransferFunctionEditorEffectMap::iterator it = s_mapEffect_.find( pd3dDevice_ );
	if( it != s_mapEffect_.end() )
		SAFE_RELEASE_REF( (*it).second );

	for (int inst = 0; inst < iInstanceCount_; ++inst) {
		SAFE_DELETE(pTfLineAlpha_[inst]);
		SAFE_DELETE(pTfLineR_[inst]);
		SAFE_DELETE(pTfLineG_[inst]);
		SAFE_DELETE(pTfLineB_[inst]);
	}

	pd3dDevice_ = NULL;
}


void TransferFunctionEditor::initUI()
{
	if(pTfEdtUi_!= NULL)
	{
		TwDeleteBar( pTfEdtUi_ );
		pTfEdtUi_ = NULL;
	}

	stringstream ssName; 
	ssName << "TFEDT_" << uiUniqueID_;

	pTfEdtUi_ = TwNewBar( ssName.str().c_str() );
	stringstream ssDefine;
	ssDefine << " '" << ssName.str().c_str() << "' color='32 32 128' alpha=196 valueswidth=70 text=light resizable=false iconifiable=false fontresizable=false movable=false refresh=1 label='TransferFunction Editor ' ";
	TwDefine( ssDefine.str().c_str()  );

	// Instances
	vector<TwEnumVal>	vInstanceList;
	TwEnumVal entry;
	for (unsigned int i = 0; i < iInstanceCount_; i++)
	{
		entry.Value = i;
		entry.Label = vsInstanceNames_[i].c_str();
		vInstanceList.push_back(entry);
	}
	TwType instanceListType = TwDefineEnum("INSTANCELIST", &(vInstanceList[0]), (unsigned int)vInstanceList.size());
	TwAddVarCB(pTfEdtUi_, "UI_DX11TFEDT_COMBO_SELECTED_INSTANCE", instanceListType, Dx11TfEditorSetVarCallback, Dx11TfEditorGetVarCallback, &(vCallbackClientData_[UI_DX11TFEDT_COMBO_SELECTED_INSTANCE]), " label='Active Instance' ");
	TwAddSeparator(pTfEdtUi_, "", "");

	// channel data

	vector<TwEnumVal>	vChannelList;
	string straChannelnames[] = { "Red", "Green", "Blue", "Alpha" };

	for (unsigned int i = 0; i < 4; i++) 
	{
		entry.Value = i;
		entry.Label = straChannelnames[i].c_str();
		vChannelList.push_back(entry);
	}
	TwType channelListType = TwDefineEnum("CHANNELLIST", &(vChannelList[0]), (unsigned int) vChannelList.size());
	TwAddVarCB(pTfEdtUi_, "UI_DX11TFEDT_COMBO_SELECTED_CHANNEL", channelListType, Dx11TfEditorSetVarCallback, Dx11TfEditorGetVarCallback, &(vCallbackClientData_[UI_DX11TFEDT_COMBO_SELECTED_CHANNEL]), " label='Active Channel' ");
	
	TwAddVarCB( pTfEdtUi_, "UI_DX11TFEDT_FLOAT_POS_X",		TW_TYPE_FLOAT, Dx11TfEditorSetVarCallback, Dx11TfEditorGetVarCallback, &(vCallbackClientData_[UI_DX11TFEDT_FLOAT_POS_X]),		" label='Position X'  precision=6" );
	TwAddVarCB( pTfEdtUi_, "UI_DX11TFEDT_FLOAT_POS_Y",		TW_TYPE_FLOAT, Dx11TfEditorSetVarCallback, Dx11TfEditorGetVarCallback, &(vCallbackClientData_[UI_DX11TFEDT_FLOAT_POS_Y]),		" label='Position Y'  precision=6" );
	TwAddVarCB( pTfEdtUi_, "UI_DX11TFEDT_FLOAT_SCALE_ALPHA",TW_TYPE_FLOAT, Dx11TfEditorSetVarCallback, Dx11TfEditorGetVarCallback, &(vCallbackClientData_[UI_DX11TFEDT_FLOAT_SCALE_ALPHA]),	" label='Scale Alpha' precision=6 min=0.0001 step=0.0001" );
	TwAddVarCB( pTfEdtUi_, "UI_DX11TFEDT_FLOAT_TFRANGE_MIN",TW_TYPE_FLOAT, Dx11TfEditorSetVarCallback, Dx11TfEditorGetVarCallback, &(vCallbackClientData_[UI_DX11TFEDT_FLOAT_TFRANGE_MIN]),	" label='Range Min' precision=4 step=0.0001" );
	TwAddVarCB( pTfEdtUi_, "UI_DX11TFEDT_FLOAT_TFRANGE_MAX",TW_TYPE_FLOAT, Dx11TfEditorSetVarCallback, Dx11TfEditorGetVarCallback, &(vCallbackClientData_[UI_DX11TFEDT_FLOAT_TFRANGE_MAX]),	" label='Range Max' precision=4 step=0.0001" );

	// Loading / Saving

	TwAddSeparator(pTfEdtUi_, "", "");
	TwAddButton( pTfEdtUi_, "UI_DX11TFEDT_BTN_LOAD_TF", Dx11TfEditorButtonCallback, &(vCallbackClientData_[UI_DX11TFEDT_BTN_LOAD_TF]), " label='Load' ");
	TwAddButton( pTfEdtUi_, "UI_DX11TFEDT_BTN_SAVE_TF", Dx11TfEditorButtonCallback, &(vCallbackClientData_[UI_DX11TFEDT_BTN_SAVE_TF]), " label='Save' ");

	TwAddSeparator(pTfEdtUi_, "", "");
	TwAddButton( pTfEdtUi_, "UI_DX11TFEDT_BTN_RESET", Dx11TfEditorButtonCallback, &(vCallbackClientData_[UI_DX11TFEDT_BTN_RESET]), " label='Reset' ");
	//if(pFxSrHistogram_ != NULL)
	//{
	//	TwAddVarCB( pTfEdtUi_, "UI_DX11TFEDT_BOOL_SHOW_HISTOGRAM",	 TW_TYPE_BOOL8, Dx11TfEditorSetVarCallback, Dx11TfEditorGetVarCallback, &(vCallbackClientData_[UI_DX11TFEDT_BOOL_SHOW_HISTOGRAM]),		" group='Histogram' label='Show' " );
	//	TwAddVarCB( pTfEdtUi_, "UI_DX11TFEDT_FLOAT_SCALE_HISTOGRAM", TW_TYPE_FLOAT, Dx11TfEditorSetVarCallback, Dx11TfEditorGetVarCallback, &(vCallbackClientData_[UI_DX11TFEDT_FLOAT_SCALE_HISTOGRAM]),	" group='Histogram' label='ExpScale' min=0.02 max=1.0 step=0.02 " );
	//}

	setVisible( bShowTfEditor_ );
}


void TransferFunctionEditor::onFrameRender( float fTime, float fElapsedTime )
{
	UNREFERENCED_PARAMETER( fTime );
	UNREFERENCED_PARAMETER( fElapsedTime );

	if( !bShowTfEditor_ )
		return;
	
	drawTransferFunction( );
}

void TransferFunctionEditor::setCurrentInstance(int instance)
{
	iCurrentInstance_ = instance;
	iSelectedControlPoint_ = NONE_PICKED;
}

//-----------------------------------------------------------------------------
//  Draw function for the transfer function
//-----------------------------------------------------------------------------
void TransferFunctionEditor::drawTransferFunction( )
{
	D3D11_VIEWPORT OldVP;
	UINT cRT = 1;

	ID3D11DeviceContext* pImmediateContext = NULL;
	pd3dDevice_->GetImmediateContext( &pImmediateContext );
	assert(pImmediateContext != NULL);
	
	pImmediateContext->IASetInputLayout( NULL );		
	pImmediateContext->IASetPrimitiveTopology( D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST );
	pImmediateContext->RSGetViewports( &cRT, &OldVP );

    Vec2i v2iTopLeftCorner(max(v2iTopLeftCorner_.x()-DX11TFEDITOR_CELL_SPACING,0), max(v2iTopLeftCorner_.y()-DX11TFEDITOR_CELL_SPACING,0));
	D3D11_VIEWPORT SMVP;
	SMVP.Height = static_cast<float>(v2iSizeUI_.y());
	SMVP.Width  = static_cast<float>(v2iSizeUI_.x());
	SMVP.MinDepth = 0;
	SMVP.MaxDepth = 1;
	SMVP.TopLeftX = static_cast<float>(v2iTopLeftCorner.x());
	SMVP.TopLeftY = static_cast<float>(v2iTopLeftCorner.y());
	pImmediateContext->RSSetViewports( 1, &SMVP );

	pFxv2iEditorTopLeft_->SetIntVector( (int *)&v2iTopLeftCorner );
	pFxv2iEditorRes_->SetIntVector( (int*)&v2iSizeUI_ );

	pFxTechRenderTFEWindow_->GetPassByIndex(0)->Apply(0, pImmediateContext );
	pImmediateContext->Draw( 3, 0 );
	
	SMVP.Height   = static_cast<float>(v2iSizeTfEdt_.y());
	SMVP.Width    = static_cast<float>(v2iSizeTfEdt_.x());
	SMVP.TopLeftX = static_cast<float>(v2iTopLeftCorner_.x());
	SMVP.TopLeftY = static_cast<float>(v2iTopLeftCorner_.y());
	pImmediateContext->RSSetViewports( 1, &SMVP );

	pFxv2iEditorTopLeft_->SetIntVector( (int *)&v2iTopLeftCorner_ );
	pFxv2iEditorRes_->SetIntVector( (int*)&v2iSizeTfEdt_ );

	if( bShowHistogram_ && pHistogramSRV_ != NULL)
	{
		pFxfExpHistoScale_->SetFloat( fExpHistoScale_ );
		pFxSrHistogram_->SetResource( pHistogramSRV_ );
		pFxTechRenderTFEWindow_->GetPassByIndex(2)->Apply(0, pImmediateContext );
	}
	else
	{
		pFxTechRenderTFEWindow_->GetPassByIndex(1)->Apply(0, pImmediateContext );
	}
	pImmediateContext->Draw( 3, 0 );

	pFxiSelectedPoint_->SetInt( NONE_PICKED );

	paChannels_[iCurrentInstance_][ iaCurrChannelsOrder_[iCurrentInstance_][3] ]->draw( );
	paChannels_[iCurrentInstance_][ iaCurrChannelsOrder_[iCurrentInstance_][2] ]->draw( );
	paChannels_[iCurrentInstance_][ iaCurrChannelsOrder_[iCurrentInstance_][1] ]->draw( );

	pFxiSelectedPoint_->SetInt( iSelectedControlPoint_ );
	paChannels_[iCurrentInstance_][ iaCurrChannelsOrder_[iCurrentInstance_][0] ]->draw( );
	
	pImmediateContext->RSSetViewports( cRT, &OldVP );
	SAFE_RELEASE( pImmediateContext );

}



bool TransferFunctionEditor::msgProc( HWND hWnd, UINT uMsg, WPARAM wParam, LPARAM lParam )
{
	if( !bShowTfEditor_ )
		return false;
	
	if( handleMessages( hWnd, uMsg, wParam, lParam ) )
		return true;

	return false;
}



bool TransferFunctionEditor::handleMessages(HWND hWnd, UINT uMsg, WPARAM wParam, LPARAM lParam)
{	
	UNREFERENCED_PARAMETER( hWnd );
	UNREFERENCED_PARAMETER( wParam );

	if( !bShowTfEditor_ )
		return false;

	if( uMsg != WM_MOUSEMOVE && uMsg != WM_LBUTTONDBLCLK && uMsg != WM_LBUTTONDOWN && uMsg != WM_LBUTTONUP ) 
		return false;

	int iMouseX = static_cast<int>(static_cast<short>(LOWORD(lParam)));
	int iMouseY = static_cast<int>(static_cast<short>(HIWORD(lParam)));

	int iTopLeftX = v2iTopLeftCorner_.x();
	int iTopLeftY = v2iTopLeftCorner_.y();
	int iBottomRightX = v2iTopLeftCorner_.x() + v2iSizeTfEdt_.x();
	int iBottomRightY = v2iTopLeftCorner_.y() + v2iSizeTfEdt_.y();

	bMouseOutside_ = 	iMouseX < iTopLeftX || iMouseX > iBottomRightX || 
						iMouseY < iTopLeftY || iMouseY > iBottomRightY;
	
	if( bMouseOutside_ && iPickedControlPoint_ == NONE_PICKED ) 
		return false;

	v2iCursorPos_.x() = iMouseX - iTopLeftX;
	v2iCursorPos_.y() = v2iSizeTfEdt_.y() - (iMouseY - iTopLeftY);
	
	switch (uMsg)
	{
		case WM_LBUTTONDBLCLK:
		{	
			int p = paChannels_[iCurrentInstance_][ iaCurrChannelsOrder_[iCurrentInstance_][0] ]->getControlPointId( v2iCursorPos_ );
			if ( p == -1 )
			{
				paChannels_[iCurrentInstance_][ iaCurrChannelsOrder_[iCurrentInstance_][0] ]->addControlPoint( v2iCursorPos_ );
				iSelectedControlPoint_ = iPickedControlPoint_ = paChannels_[iCurrentInstance_][ iaCurrChannelsOrder_[iCurrentInstance_][0] ]->getControlPointId( v2iCursorPos_ );
			}
			else
			{
				paChannels_[iCurrentInstance_][ iaCurrChannelsOrder_[iCurrentInstance_][0] ]->deleteControlPoint( p );
				iSelectedControlPoint_ = iPickedControlPoint_ = NONE_PICKED;
			}
		
			updateTexture(iCurrentInstance_);
			break;
		}

		case WM_LBUTTONDOWN:
		{				
			iPickedControlPoint_ = paChannels_[iCurrentInstance_][ iaCurrChannelsOrder_[iCurrentInstance_][0] ]->getControlPointId( v2iCursorPos_ );

			if( iPickedControlPoint_ == NONE_PICKED )
			{
				if( paChannels_[iCurrentInstance_][ iaCurrChannelsOrder_[iCurrentInstance_][1] ]->liesOnTheLine( v2iCursorPos_ ) )
				{
					int ind = iaCurrChannelsOrder_[iCurrentInstance_][1];
					iaCurrChannelsOrder_[iCurrentInstance_][1] = iaCurrChannelsOrder_[iCurrentInstance_][0];
					iaCurrChannelsOrder_[iCurrentInstance_][0] = ind;

					iPickedControlPoint_ = paChannels_[iCurrentInstance_][ ind ]->getControlPointId( v2iCursorPos_ );
				}
				else 
				if( paChannels_[iCurrentInstance_][ iaCurrChannelsOrder_[iCurrentInstance_][2] ]->liesOnTheLine( v2iCursorPos_ ) )
				{								
					int ind = iaCurrChannelsOrder_[iCurrentInstance_][2];
					iaCurrChannelsOrder_[iCurrentInstance_][2] = iaCurrChannelsOrder_[iCurrentInstance_][1];
					iaCurrChannelsOrder_[iCurrentInstance_][1] = iaCurrChannelsOrder_[iCurrentInstance_][0];
					iaCurrChannelsOrder_[iCurrentInstance_][0] = ind;

					iPickedControlPoint_ = paChannels_[iCurrentInstance_][ ind ]->getControlPointId( v2iCursorPos_ );
				}
				else 
				if( paChannels_[iCurrentInstance_][ iaCurrChannelsOrder_[iCurrentInstance_][3] ]->liesOnTheLine( v2iCursorPos_ ) )
				{				
					int ind = iaCurrChannelsOrder_[iCurrentInstance_][3];
					iaCurrChannelsOrder_[iCurrentInstance_][3] = iaCurrChannelsOrder_[iCurrentInstance_][2];
					iaCurrChannelsOrder_[iCurrentInstance_][2] = iaCurrChannelsOrder_[iCurrentInstance_][1];
					iaCurrChannelsOrder_[iCurrentInstance_][1] = iaCurrChannelsOrder_[iCurrentInstance_][0];
					iaCurrChannelsOrder_[iCurrentInstance_][0] = ind;

					iPickedControlPoint_ = paChannels_[iCurrentInstance_][ ind ]->getControlPointId( v2iCursorPos_ );
				}
			}

			iSelectedControlPoint_ = iPickedControlPoint_;
			break;
		}

		case WM_LBUTTONUP:	
		{
			iPickedControlPoint_ = NONE_PICKED;		
			break;
		}

		case WM_MOUSEMOVE:
		if ( iPickedControlPoint_ != NONE_PICKED )
		{
			paChannels_[iCurrentInstance_][ iaCurrChannelsOrder_[iCurrentInstance_][0] ]->moveControlPoint( iPickedControlPoint_, v2iCursorPos_ );
			updateTexture(iCurrentInstance_);
		}
		break;
	}

	return true;
}


//-----------------------------------------------------------------------------
//  Update the transfer function texture on the GPU
//-----------------------------------------------------------------------------
void
TransferFunctionEditor::updateTexture(int inst)
{		
	iTimestamp_++;

	pTfLineR_[inst]	->fillArray( pTexData_[inst], v2iSizeTfEdt_.x(), 0 );
	pTfLineG_[inst]	->fillArray( pTexData_[inst], v2iSizeTfEdt_.x(), 1 );
	pTfLineB_[inst] ->fillArray( pTexData_[inst], v2iSizeTfEdt_.x(), 2 );
	pTfLineAlpha_[inst]->fillArray( pTexData_[inst], v2iSizeTfEdt_.x(), 3 );

	// update Texture
	UINT srcRowPitch = v2iSizeTfEdt_.x()*4*sizeof(float);
	D3D11_BOX box = { 0, 0, 0, v2iSizeTfEdt_.x(), 1, 1 };

	ID3D11DeviceContext* pImmediateContext = NULL;
	pd3dDevice_->GetImmediateContext( &pImmediateContext );
	assert( pImmediateContext != NULL );

	pImmediateContext->UpdateSubresource( pTfTex_[inst], 0, &box, pTexData_[inst], srcRowPitch, 0 );

	bTransferFunctionTexChanged_[inst] = true;

	SAFE_RELEASE( pImmediateContext );
}


static bool GetFilenameDialog(const wstring& lpstrTitle, const WCHAR* lpstrFilter, wstring &filename, const bool save, HWND owner=NULL, DWORD* nFilterIndex=NULL)
{
	BOOL result;
	OPENFILENAMEW ofn;
	ZeroMemory(&ofn,sizeof(OPENFILENAMEW));

	static WCHAR szFile[MAX_PATH];
	szFile[0] = 0;


	WCHAR szDir[MAX_PATH];
	if (filename.length()>0) {
		errno_t err=wcscpy_s(szDir,MAX_PATH,filename.c_str());
		filename.clear();
		if (err) return false;
	} else szDir[0]=0;
	ofn.lpstrInitialDir = szDir;

	//====== Dialog parameters
	ofn.lStructSize   = sizeof(OPENFILENAMEW);
	ofn.lpstrFilter   = lpstrFilter;
	ofn.nFilterIndex  = 1;
	ofn.lpstrFile     = szFile;
	ofn.nMaxFile      = sizeof(szFile);
	ofn.lpstrTitle    = lpstrTitle.c_str();
	ofn.nMaxFileTitle = sizeof (lpstrTitle.c_str());
	ofn.hwndOwner     = owner;

	if (save) {
		ofn.Flags = OFN_NOCHANGEDIR | OFN_HIDEREADONLY | OFN_EXPLORER | OFN_OVERWRITEPROMPT;
		result = GetSaveFileNameW(&ofn);
	} else {
		ofn.Flags = OFN_NOCHANGEDIR | OFN_HIDEREADONLY | OFN_EXPLORER | OFN_FILEMUSTEXIST;
		result = GetOpenFileNameW(&ofn);
	}
	if (result)	{
		filename = szFile;
		if (nFilterIndex != NULL) *nFilterIndex = ofn.nFilterIndex;
		return true;
	} else {
		filename = L"";
		return false;
	}
}


void TransferFunctionEditor::saveTransferFunction(int inst)
{
	wstring wstrPathName;
	if(GetFilenameDialog( L"Load TF", L"*.tf\0*.tf", wstrPathName, true ))
	{
		// add extension if necessary
		wstring ext(L".tf");
		if(wstrPathName.substr(wstrPathName.length() - ext.length()) != ext)
		{
			wstrPathName += ext;
		}
		std::ofstream file(wstrPathName, std::ios_base::binary);
		saveTransferFunction(inst, &file);
	}
}

void TransferFunctionEditor::loadTransferFunction(int inst)
{
	wstring wstrPathName;
	if(GetFilenameDialog( L"Save TF", L"*.tf\0*.tf", wstrPathName, false ))
	{
		std::ifstream file(wstrPathName, std::ios_base::binary);
		loadTransferFunction(inst, &file);
	}
}

void TransferFunctionEditor::saveTransferFunction(int inst, std::ofstream* binary_file)
{	
	if( ! binary_file->is_open() )	
		return;

	TransferFunctionLine::sControlPoint* pCp;

	for( int ch = 0; ch < 4; ch++)
	{
		pCp = paChannels_[inst][ch]->getFirstPoint();

		int count = 1;
		for (TransferFunctionLine::sControlPoint *pTmp = pCp; pTmp->pNext_ != NULL; pTmp = pTmp->pNext_)
			count++;

		binary_file->write(reinterpret_cast<char*>(&count), sizeof(count));

		for (int i = 0; i < count; i++)  
		{
            Vec2f v(static_cast<float>(pCp->v2iPt_.x())/static_cast<float>(v2iSizeTfEdt_.x()-1), static_cast<float>(pCp->v2iPt_.y())/static_cast<float>(v2iSizeTfEdt_.y()-1));
			binary_file->write(reinterpret_cast<char *>(&(v)), sizeof(Vec2f));
			pCp = pCp->pNext_;
		}
	}

	// write the order out
	binary_file->write(reinterpret_cast<char *>(&(iaCurrChannelsOrder_[inst][0])), sizeof(int));
	binary_file->write(reinterpret_cast<char *>(&(iaCurrChannelsOrder_[inst][1])), sizeof(int));
	binary_file->write(reinterpret_cast<char *>(&(iaCurrChannelsOrder_[inst][2])), sizeof(int));
	binary_file->write(reinterpret_cast<char *>(&(iaCurrChannelsOrder_[inst][3])), sizeof(int));

	// write other params: alpha scale and range
	binary_file->write(reinterpret_cast<const char*>(&fAlphaScale_[inst]), sizeof(float));
	binary_file->write(reinterpret_cast<const char*>(&v2fTfRangeMinMax_[inst]), sizeof(Vec2f));
}

void TransferFunctionEditor::loadTransferFunction(int inst, std::ifstream* binary_file)
{
	if ( ! binary_file->is_open() )
		return;

	for( int ch = 0; ch < 4; ch++ )
	{
		paChannels_[inst][ch]->reset();

		int iNumPoints;
		binary_file->read(reinterpret_cast<char *>(&iNumPoints), sizeof(iNumPoints));

        Vec2i v2iPoint(0,0);
        Vec2f v;

		binary_file->read(reinterpret_cast<char *>(&v),sizeof(Vec2f));
		v2iPoint.x() = static_cast<int>(v.x() * (v2iSizeTfEdt_.x()-1));
		v2iPoint.y() = static_cast<int>(v.y() * (v2iSizeTfEdt_.y()-1));
		
		paChannels_[inst][ch]->moveControlPoint( 0, v2iPoint );

		for (int i = 1; i < iNumPoints-1; i++)
		{
			binary_file->read(reinterpret_cast<char *>(&v),sizeof(Vec2f));
			v2iPoint.x() = static_cast<int>(v.x() * (v2iSizeTfEdt_.x()-1));
			v2iPoint.y() = static_cast<int>(v.y() * (v2iSizeTfEdt_.y()-1));
			paChannels_[inst][ch]->addControlPoint( v2iPoint );
		}

		binary_file->read(reinterpret_cast<char *>(&v),sizeof(Vec2f));
		v2iPoint.x() = static_cast<int>(v.x() * (v2iSizeTfEdt_.x()-1));
		v2iPoint.y() = static_cast<int>(v.y() * (v2iSizeTfEdt_.y()-1));
		
		paChannels_[inst][ch]->moveControlPoint( iNumPoints-1, v2iPoint );
	}

	binary_file->read(reinterpret_cast<char *>(&(iaCurrChannelsOrder_[inst][0])), sizeof(int));
	binary_file->read(reinterpret_cast<char *>(&(iaCurrChannelsOrder_[inst][1])), sizeof(int));
	binary_file->read(reinterpret_cast<char *>(&(iaCurrChannelsOrder_[inst][2])), sizeof(int));
	binary_file->read(reinterpret_cast<char *>(&(iaCurrChannelsOrder_[inst][3])), sizeof(int));

	// read other params: alpha scale and range
	binary_file->read(reinterpret_cast<char*>(&fAlphaScale_[inst]), sizeof(float));
	binary_file->read(reinterpret_cast<char*>(&v2fTfRangeMinMax_[inst]), sizeof(Vec2f));

	this->updateTexture(inst);
}



void TransferFunctionEditor::bringChannelToTop(int inst, int ch )
{
	if( iaCurrChannelsOrder_[inst][0] == ch ) 
		return;

	if( iaCurrChannelsOrder_[inst][1] == ch )
	{
		iaCurrChannelsOrder_[inst][1] = iaCurrChannelsOrder_[inst][0];
	}
	else
	if( iaCurrChannelsOrder_[inst][2] == ch )
	{
		iaCurrChannelsOrder_[inst][2] = iaCurrChannelsOrder_[inst][1];
		iaCurrChannelsOrder_[inst][1] = iaCurrChannelsOrder_[inst][0];
	}
	else
	if( iaCurrChannelsOrder_[inst][3] == ch )
	{
		iaCurrChannelsOrder_[inst][3] = iaCurrChannelsOrder_[inst][2];
		iaCurrChannelsOrder_[inst][2] = iaCurrChannelsOrder_[inst][1];
		iaCurrChannelsOrder_[inst][1] = iaCurrChannelsOrder_[inst][0];
	}

	iaCurrChannelsOrder_[inst][0] = ch;

	iSelectedControlPoint_ = NONE_PICKED;
}



void TransferFunctionEditor::reset(int inst )
{
	paChannels_[inst][ iaCurrChannelsOrder_[inst][0] ]->reset();
	paChannels_[inst][ iaCurrChannelsOrder_[inst][1] ]->reset();
	paChannels_[inst][ iaCurrChannelsOrder_[inst][2] ]->reset();
	paChannels_[inst][ iaCurrChannelsOrder_[inst][3] ]->reset();

	iSelectedControlPoint_ = NONE_PICKED;

	updateTexture(inst);
}



void TransferFunctionEditor::moveSelectedControlPoint(int inst, float fPos, int iDim )
{
	if (iSelectedControlPoint_ == NONE_PICKED) return;

	fPos = max(0.0f, min(fPos, 1.0f));

    Vec2i v2iPosition = paChannels_[inst][ iaCurrChannelsOrder_[inst][0] ]->getControlPoint( iSelectedControlPoint_ );

	if(iDim == 0)
	{
		v2iPosition.x() = static_cast<int>( fPos * v2iSizeTfEdt_.x() );
		paChannels_[inst][ iaCurrChannelsOrder_[inst][0] ]->moveControlPoint( iSelectedControlPoint_, v2iPosition );
	}
	else if(iDim == 1)
	{
		v2iPosition.y() = static_cast<int>( fPos * v2iSizeTfEdt_.y() );
		paChannels_[inst][ iaCurrChannelsOrder_[inst][0] ]->moveControlPoint( iSelectedControlPoint_, v2iPosition );
	}

	updateTexture(inst);

	return;
}



float TransferFunctionEditor::getSelectedControlPointCoord(int inst, int iDim )
{
	if(!isPointSelected())
		return 0.0f;

    Vec2i v2iPosition = paChannels_[inst][ iaCurrChannelsOrder_[inst][0] ]->getControlPoint( iSelectedControlPoint_ );
	float fResult = 0.0f;

	if(iDim == 0)
	{
		fResult = static_cast<float>(v2iPosition.x())/static_cast<float>(v2iSizeTfEdt_.x());
	}
	else if(iDim == 1)
	{
		fResult = static_cast<float>(v2iPosition.y())/static_cast<float>(v2iSizeTfEdt_.y());
	}

	return fResult;
}


bool TransferFunctionEditor::isPointSelected()
{
	return (iSelectedControlPoint_ != NONE_PICKED );
}



void TransferFunctionEditor::setVisible( bool b ) 
{ 
	bShowTfEditor_ = b; 

	if (pTfEdtUi_) 
	{
		INT32 iShowTfEditor = (INT32)bShowTfEditor_;
		TwSetParam(pTfEdtUi_, NULL, "visible", TW_PARAM_INT32, 1, &iShowTfEditor);

		float fRefresh = (bShowTfEditor_ == true)? 1.0f: 60.0f;
		TwSetParam(pTfEdtUi_, NULL, "refresh", TW_PARAM_FLOAT, 1, &fRefresh);
	}
}

HRESULT TransferFunctionEditor::saveTfLegend()
{
	MessageBoxA(NULL, "Not implemented atm (no D3DX and Kai Dx11Utils...)", "Fail", MB_OK | MB_ICONERROR);
	return S_OK;

	//unsigned int uiCheckerSize = 10;
	//Vec2ui v2uiTargetResolution(2*uiCheckerSize, 30*uiCheckerSize);


	//ID3D11Texture2D*		ppTargetTex[2] = { NULL, NULL };
	//ID3D11RenderTargetView*	ppTargetRTV[2] = { NULL, NULL };

	//D3D11_TEXTURE2D_DESC txDesc;

	//txDesc.ArraySize			= 1;
	//txDesc.BindFlags			= D3D11_BIND_RENDER_TARGET;
	//txDesc.CPUAccessFlags		= 0;
	//txDesc.Format				= DXGI_FORMAT_R8G8B8A8_UNORM;
	//txDesc.Height				= v2uiTargetResolution.y();
	//txDesc.MipLevels			= 1;
	//txDesc.MiscFlags			= 0;
	//txDesc.SampleDesc.Count		= 1;
	//txDesc.SampleDesc.Quality	= 0;
	//txDesc.Usage				= D3D11_USAGE_DEFAULT;
	//txDesc.Width				= v2uiTargetResolution.x();

	//HRESULT hr = S_OK;

	//bool bCreationError = false;

	//for(int i=0; i<2; i++)
	//{
	//	hr = pd3dDevice_->CreateTexture2D( &txDesc, NULL, &(ppTargetTex[i]) );

	//	if( FAILED(hr) )
	//	{
	//		bCreationError = true;
	//		break;
	//	}
	//	
	//	hr = pd3dDevice_->CreateRenderTargetView( ppTargetTex[i], NULL, &(ppTargetRTV[i]) );

	//	if( FAILED(hr) )
	//	{
	//		bCreationError = true;
	//		break;
	//	}
	//}

	//if(bCreationError)
	//{
	//	for(int i=0; i<2; i++)
	//	{
	//		SAFE_RELEASE( ppTargetTex[i] );
	//		SAFE_RELEASE( ppTargetRTV[i] );
	//	}

	//	return E_FAIL;
	//}



	//ID3DX11Effect* pEffect = NULL;
	//Dx11TfEditorEffectMap::const_iterator it = s_mapEffect_.find( pd3dDevice_ );

	//if( it == s_mapEffect_.end() || (*it).second == NULL )
	//{
	//	for(int i=0; i<2; i++)
	//	{
	//		SAFE_RELEASE( ppTargetTex[i] );
	//		SAFE_RELEASE( ppTargetRTV[i] );

	//	}
	//	return E_FAIL;
	//}
	//else
	//{
	//	pEffect = (*it).second;
	//}

	//ID3D11DeviceContext* pContext = NULL;
	//pd3dDevice_->GetImmediateContext( &pContext );
	//dx11utils::core::RtStateBackup stateBk( pContext );
	//stateBk.backup();

	//D3D11_VIEWPORT vps[2];
	//for(int i=0; i<2; i++)
	//{
	//	vps[i].TopLeftX = 0;
	//	vps[i].TopLeftY = 0;
	//	vps[i].MinDepth = 0;
	//	vps[i].MaxDepth = 1;
	//	vps[i].Width	= (FLOAT)v2uiTargetResolution.x();
	//	vps[i].Height	= (FLOAT)v2uiTargetResolution.y();
	//}
	//pContext->RSSetViewports( 2, vps );
	//pContext->OMSetRenderTargets( 2, ppTargetRTV, NULL );


	//ID3D11Buffer* pNull = NULL;
	//UINT uiStride = 0;
	//UINT uiOffset = 0;
	//pContext->IASetVertexBuffers(0, 1, &pNull, &uiStride, &uiOffset);
	//pContext->IASetInputLayout( NULL );
	//pContext->IASetPrimitiveTopology( D3D11_PRIMITIVE_TOPOLOGY_TRIANGLESTRIP );

	//
	//ID3DX11EffectVectorVariable* pFXv2fRes	= NULL;
	//dx11utils::effects::safeGetEffectVectorVariable( pEffect, "v2fTargetResolution", &pFXv2fRes );
	//Vec2f v2fTargetResolution(float(v2uiTargetResolution.x()), float(v2uiTargetResolution.y()));
	//pFXv2fRes->SetFloatVector( v2fTargetResolution );

	//ID3DX11EffectVectorVariable* pFXv2uiRes = NULL;
	//dx11utils::effects::safeGetEffectVectorVariable( pEffect, "v2uiTargetResolution", &pFXv2uiRes );
	//pFXv2uiRes->SetIntVector( (int*)&(v2uiTargetResolution.x()) );

	//ID3DX11EffectScalarVariable* pFXuiCheckerSize = NULL;
	//dx11utils::effects::safeGetEffectScalarVariable(pEffect, "uiCheckerSize", &pFXuiCheckerSize );
	//pFXuiCheckerSize->SetInt( uiCheckerSize );

	//ID3DX11EffectScalarVariable* pFXfTfScale = NULL;
	//dx11utils::effects::safeGetEffectScalarVariable(pEffect, "fTfAlphaScale", &pFXfTfScale );
	//pFXfTfScale->SetFloat( fAlphaScale_ );

	//ID3DX11EffectShaderResourceVariable* pFXtxTf = NULL;
	//dx11utils::effects::safeGetEffectShaderResourceVariable( pEffect, "txTf_", &pFXtxTf );
	//pFXtxTf->SetResource( getTfSrv() );

	//pEffect->GetTechniqueByName( "tRender_TFToTex_" )->GetPassByIndex( 0 )->Apply( 0, pContext );
	//pContext->Draw( 3, 0 );

	//stateBk.restore();
	//
	//wstring wstrPathName;
	//if(tum3d::GetFilenameDialog( L"Save TF Legend", L"*.png\0*.png", wstrPathName, true ))
	//{
	//	wstrPathName = tum3d::RemoveExt( wstrPathName );
	//	for(int i=0; i<2; i++)
	//	{
	//		wstringstream wssTarget;
	//		wssTarget << wstrPathName.c_str() << L"_" << i << L".png";
	//		D3DX11SaveTextureToFile( pContext, ppTargetTex[i], D3DX11_IFF_PNG, wssTarget.str().c_str() );
	//	}
	//}


	//
	//SAFE_RELEASE( pContext );

	//for(int i=0; i<2; i++)
	//{
	//	SAFE_RELEASE( ppTargetTex[i] );
	//	SAFE_RELEASE( ppTargetRTV[i] );
	//}

	//return S_OK;
}
