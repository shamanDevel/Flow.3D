cbuffer Params
{
	float2   g_vPosition;
	float2   g_vSize;
	float4   g_vColor;
	float    g_fProgress;
}


RasterizerState CullNone
{
	CullMode = None;
};

DepthStencilState DepthDefault
{
};

DepthStencilState DepthDisable
{
	DepthEnable = false;
	DepthWriteMask = 0;
};

BlendState BlendDisable
{
};

BlendState BlendOver
{
	BlendEnable[0] = true;
	SrcBlend[0] = SRC_ALPHA;
	DestBlend[0] = INV_SRC_ALPHA;
	BlendOp[0] = ADD;
	SrcBlendAlpha[0] = ONE;
	DestBlendAlpha[0] = INV_SRC_ALPHA;
	BlendOpAlpha[0] = ADD;
};


void vsQuad(uint index : SV_VertexID, out float4 outPos : SV_Position, out float2 barPos : BARPOS)
{
	barPos = float2(float(index % 2), float(index / 2));
	outPos = float4(g_vPosition + barPos * g_vSize, 0.5f, 1.0f);
}

void vsOutline(uint index : SV_VertexID, out float4 outPos : SV_Position, out float2 barPos : BARPOS)
{
	barPos = float2((index == 1 || index == 2) ? 1.0f : 0.0f, (index == 2 || index == 3) ? 1.0f : 0.0f);
	outPos = float4(g_vPosition + barPos * g_vSize, 0.5f, 1.0f);
}

float4 psProgress(float4 pos : SV_Position, float barPos : BARPOS) : SV_Target
{
	if(barPos.x > g_fProgress) discard;
	return g_vColor;
}

float4 psColor() : SV_Target
{
	return g_vColor;
}


technique11 tProgressBar
{
	pass P0_Bar
	{
		SetVertexShader( CompileShader( vs_5_0, vsQuad() ) );
		SetGeometryShader( NULL );
		SetPixelShader( CompileShader( ps_5_0, psProgress() ) );
		SetRasterizerState( CullNone );
		SetDepthStencilState( DepthDisable, 0 );
		SetBlendState( BlendOver, float4( 0.0f, 0.0f, 0.0f, 0.0f ), 0xFFFFFFFF );
	}
	pass P1_BarOutline
	{
		SetVertexShader( CompileShader( vs_5_0, vsOutline() ) );
		SetGeometryShader( NULL );
		SetPixelShader( CompileShader( ps_5_0, psColor() ) );
		SetRasterizerState( CullNone );
		SetDepthStencilState( DepthDisable, 0 );
		SetBlendState( BlendDisable, float4( 0.0f, 0.0f, 0.0f, 0.0f ), 0xFFFFFFFF );
	}
}
