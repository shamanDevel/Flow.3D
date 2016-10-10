cbuffer cb1
{
    int2 	v2iTfEditorTopLeft_;
	int2 	v2iTfEditorRes_;	
	int		iSelectedPoint_;
	float   fExpScaleHisto_;
};

cbuffer colors
{
	float4 v4fLineColor_;
	float4 vBGColor[2] 		= { float4(32.0/255.0, 32.0/255.0, 128.0/255.0, 196.0/255.0), float4(0.2,0.2,0.2,0.8)};
	float4 vBorderColor[2]	= { float4(0.6,0.6,0.6,0.8), float4(0.6,0.6,0.6,0.8)};
}

Texture1D txTf_;
Texture1D txHistogramm_;

DepthStencilState DisableDepth
{
    DepthEnable		= FALSE;
    DepthWriteMask	= ZERO;
    StencilEnable	= FALSE;
};

BlendState NoBlending
{
    AlphaToCoverageEnable = FALSE;
    BlendEnable[0] = FALSE;
    BlendEnable[1] = FALSE;
    BlendEnable[2] = FALSE;
    BlendEnable[3] = FALSE;
    BlendEnable[4] = FALSE;
    BlendEnable[5] = FALSE;
    BlendEnable[6] = FALSE;
    BlendEnable[7] = FALSE;
};

BlendState AlphaBlendingAdd
{
	BlendEnable[0]  = TRUE;
	SrcBlend		= SRC_ALPHA;
	DestBlend		= ONE;
	BlendOp			= ADD;
	SrcBlendAlpha	= ZERO;
	DestBlendAlpha	= ZERO;
	BlendOpAlpha	= ADD;
	RenderTargetWriteMask[0] = 0x0F;
};

BlendState bsOver
{
	AlphaToCoverageEnable	 = FALSE;
	BlendEnable[0]			 = TRUE;
	SrcBlend				 = SRC_ALPHA;
	DestBlend				 = INV_SRC_ALPHA;
	BlendOp					 = ADD;
	SrcBlendAlpha			 = ONE;
	DestBlendAlpha			 = ZERO;
	BlendOpAlpha			 = ADD;
	RenderTargetWriteMask[0] = 0x0F;
};

SamplerState sLinear 
{
	Filter = MIN_MAG_LINEAR_MIP_POINT;
	AddressU = Clamp;
	AddressV = Clamp;
	AddressW = Clamp;
};

RasterizerState rsNoCull
{
    CullMode = None;
};

//######################################################################################
// Functions to render the transparent background of the transfer function 
//######################################################################################


float4 VS_FullScreenTri( uint id : SV_VertexID ) : SV_POSITION
{
	return float4( 
		((id << 1) & 2) *  2.0f - 1.0f,  // x (-1, 3,-1)
		( id       & 2) * -2.0f + 1.0f,  // y ( 1, 1,-3)
		0.0f, 
		1.0f ); 
}

//--------------------------------------------------------------------------------------
// Transfer Function Pixel Shader
//--------------------------------------------------------------------------------------
float4 PS_FillColor( float4 pos : SV_POSITION, uniform uint id) : SV_Target
{
	int2 texPos;
	texPos.xy = int2( pos.xy - v2iTfEditorTopLeft_.xy );

   	if ( texPos.x > 0 && texPos.y > 0 && texPos.x < v2iTfEditorRes_.x-1 && texPos.y < v2iTfEditorRes_.y-1 )
	{
		return vBGColor[id];
	}
   	else
	{
		return vBorderColor[id];
	}
}


float4 PS_FillHistogramm( float4 pos : SV_POSITION) : SV_Target
{
	int2 v2iTexPos = int2( pos.xy - v2iTfEditorTopLeft_.xy );

   	if ( v2iTexPos.x > 0 && v2iTexPos.y > 0 && v2iTexPos.x < v2iTfEditorRes_.x-1 && v2iTexPos.y < v2iTfEditorRes_.y-1 )
	{
		float2 v2fTexPos = (float2)v2iTexPos.xy / (float2)v2iTfEditorRes_.xy;
		v2fTexPos.y = 1-v2fTexPos.y;

		float fHistoVal = pow(abs(txHistogramm_.SampleLevel( sLinear, v2fTexPos.x, 0 ).x), fExpScaleHisto_);

		if( v2fTexPos.y >= fHistoVal ) 
		{
			return float4(0.5,0.5,0.5,0.6);
		}
		else
		{
			return float4(0.1,0.1,0.1,0.8);			
		}
	}
   	else
	{
		return vBorderColor[1];
	}
}

//--------------------------------------------------------------------------------------
technique10 tRender_TFWindow_
{
	pass P0
	{
		SetVertexShader( CompileShader( vs_4_0, VS_FullScreenTri() ) );
		SetGeometryShader( NULL );
		SetPixelShader( CompileShader( ps_4_0, PS_FillColor(0) ) );
              
		SetBlendState( bsOver, float4( 0.0f, 0.0f, 0.0f, 0.0f ), 0xFFFFFFFF );
        SetRasterizerState( rsNoCull );
		SetDepthStencilState( DisableDepth, 0 );
	}

	pass P1
	{
		SetVertexShader( CompileShader( vs_4_0, VS_FullScreenTri() ) );
		SetGeometryShader( NULL );
		SetPixelShader( CompileShader( ps_4_0, PS_FillColor(1) ) );
              
		SetBlendState( AlphaBlendingAdd, float4( 0.0f, 0.0f, 0.0f, 0.0f ), 0xFFFFFFFF );
        SetRasterizerState( rsNoCull );
		SetDepthStencilState( DisableDepth, 0 );
	}

	pass P2
	{
		SetVertexShader( CompileShader( vs_4_0, VS_FullScreenTri() ) );
		SetGeometryShader( NULL );
		SetPixelShader( CompileShader( ps_4_0, PS_FillHistogramm() ) );
              
		SetBlendState( AlphaBlendingAdd, float4( 0.0f, 0.0f, 0.0f, 0.0f ), 0xFFFFFFFF );
        SetRasterizerState( rsNoCull );
		SetDepthStencilState( DisableDepth, 0 );
	}

}


//######################################################################################
// Functions to render a line of the transfer function 
//######################################################################################

//--------------------------------------------------------------------------------------
// Vertex Shader
//--------------------------------------------------------------------------------------
float4 VS_TFLines( float2 pos : POSITION ) : SV_POSITION
{
       float4 newPos = float4(0,0,0,1);
       newPos.xy = pos * float2(2,2) - float2(1,1);

       return newPos;
}

//--------------------------------------------------------------------------------------
// Pixel Shaders
//--------------------------------------------------------------------------------------
float4 PS_TFLines( float4 pos : SV_POSITION ) : SV_Target
{
    return v4fLineColor_;
}

//--------------------------------------------------------------------------------------
technique10 tRender_TFLines_
{
	pass P0
	{
		SetVertexShader( CompileShader( vs_4_0, VS_TFLines() ) );
		SetGeometryShader( NULL );
		SetPixelShader( CompileShader( ps_4_0, PS_TFLines() ) );

		SetBlendState( NoBlending, float4( 0.0f, 0.0f, 0.0f, 0.0f ), 0xFFFFFFFF );
        SetRasterizerState( rsNoCull );
		SetDepthStencilState( DisableDepth, 0 );
	}   
}


//######################################################################################
// Functions to render the control points of the transfer function 
//######################################################################################

struct IN_POINT
{
    float2 	Pos : POSITION;
	uint	VertID : SV_VertexID;
};

struct OUT_POINT
{
    float4 	Pos : SV_POSITION;
	uint	VertID : VID;
	float2  Txc	   : TEXCOORDS;
};

//--------------------------------------------------------------------------------------
// Vertex Shader
//--------------------------------------------------------------------------------------
OUT_POINT VS_TF_POINT( IN_POINT p )
{
	OUT_POINT res = (OUT_POINT) 0;
	
	res.Pos = float4(0,0,0,1);
	res.Pos.xy = p.Pos * float2(2,2) - float2(1,1);

	res.VertID = p.VertID;

	return res;
}

//--------------------------------------------------------------------------------------
// Geometry Shader
//--------------------------------------------------------------------------------------
[maxvertexcount(4)]
void GS_TF_POINT( point OUT_POINT input[1], inout TriangleStream<OUT_POINT> SpriteStream )
{
	const OUT_POINT p = input[0];
	OUT_POINT output = (OUT_POINT) 0;

	output.VertID = p.VertID;

	float2 offset = 10.0f / v2iTfEditorRes_;

	output.Pos  = p.Pos + float4( offset.x, offset.y, 0, 0 );
	output.Txc	= float2(1,1);
	SpriteStream.Append(output);

	output.Pos  = p.Pos + float4( offset.x,-offset.y, 0, 0 );
	output.Txc	= float2(1,-1);
	SpriteStream.Append(output);

	output.Pos  = p.Pos + float4(-offset.x, offset.y, 0, 0 );
	output.Txc	= float2(-1,1);
	SpriteStream.Append(output);

	output.Pos  = p.Pos + float4(-offset.x,-offset.y, 0, 0 );
	output.Txc	= float2(-1,-1);
	SpriteStream.Append(output);

	SpriteStream.RestartStrip();

	return;
}

//--------------------------------------------------------------------------------------
// Pixel Shader
//--------------------------------------------------------------------------------------
float4 PS_TF_POINT( OUT_POINT p ) : SV_Target
{
	if(length(p.Txc) > 1) 
		discard;

	if ( iSelectedPoint_ == (int) p.VertID )
		return float4(1,1,1,1);
	else
		return v4fLineColor_;
}

//--------------------------------------------------------------------------------------
technique10 tRender_TFPoints_
{
	pass P0
	{
		SetVertexShader( CompileShader( vs_4_0, VS_TF_POINT() ) );
		SetGeometryShader( CompileShader( gs_4_0, GS_TF_POINT() ) );
		SetPixelShader( CompileShader( ps_4_0, PS_TF_POINT() ) );

		SetBlendState( NoBlending, float4( 0.0f, 0.0f, 0.0f, 0.0f ), 0xFFFFFFFF );
        SetRasterizerState( rsNoCull );
		SetDepthStencilState( DisableDepth, 0 );
	}
}
