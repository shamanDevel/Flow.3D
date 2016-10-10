cbuffer PerFrame
{
	float2 g_vScreenMin;
	float2 g_vScreenMax;
	float4 g_vColor;
	float2 g_vTexCoordMin;
	float2 g_vTexCoordMax;
}

Texture2D g_tex;


RasterizerState CullNone
{
	CullMode = None;
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
	SrcBlend       = ONE;
	DestBlend      = INV_SRC_ALPHA;
	BlendOp        = ADD;
	SrcBlendAlpha  = ONE;
	DestBlendAlpha = INV_SRC_ALPHA;
	BlendOpAlpha   = ADD;
	BlendEnable[0] = TRUE;
};

BlendState BlendBehind
{
	SrcBlend       = INV_DEST_ALPHA;
	DestBlend      = ONE;
	BlendOp        = ADD;
	SrcBlendAlpha  = INV_DEST_ALPHA;
	DestBlendAlpha = ONE;
	BlendOpAlpha   = ADD;
	BlendEnable[0] = TRUE;
};

SamplerState SamplerLinear
{
	Filter         = MIN_MAG_MIP_LINEAR;
	AddressU       = Clamp;
	AddressV       = Clamp;
};



void vsScreen(uint index : SV_VertexID, out float4 outPos : SV_Position, out float2 outTexCoord : TEXCOORD)
{
	switch(index) {
		case 0:
			outPos = float4(g_vScreenMin.x, g_vScreenMin.y, 0.5f, 1.0f);
			outTexCoord = float2(g_vTexCoordMin.x, g_vTexCoordMax.y);
			break;
		case 1:
			outPos = float4(g_vScreenMax.x, g_vScreenMin.y, 0.5f, 1.0f);
			outTexCoord = float2(g_vTexCoordMax.x, g_vTexCoordMax.y);
			break;
		case 2:
			outPos = float4(g_vScreenMin.x, g_vScreenMax.y, 0.5f, 1.0f);
			outTexCoord = float2(g_vTexCoordMin.x, g_vTexCoordMin.y);
			break;
		case 3:
		default:
			outPos = float4(g_vScreenMax.x, g_vScreenMax.y, 0.5f, 1.0f);
			outTexCoord = float2(g_vTexCoordMax.x, g_vTexCoordMin.y);
			break;
	}
}


float4 psSolidColor() : SV_Target
{
	return g_vColor;
}

float4 psTex(float4 pos : SV_Position, float2 texCoord : TEXCOORD) : SV_Target
{
	return g_tex.Sample(SamplerLinear, texCoord);
}


technique11 tScreen
{
	pass P0_SolidBlendBehind
	{
		SetVertexShader( CompileShader( vs_5_0, vsScreen() ) );
		SetGeometryShader( NULL );
		SetPixelShader( CompileShader( ps_5_0, psSolidColor() ) );
		SetRasterizerState( CullNone );
		SetDepthStencilState( DepthDisable, 0 );
		SetBlendState( BlendBehind, float4( 0.0f, 0.0f, 0.0f, 0.0f ), 0xFFFFFFFF );
	}

	pass P1_Blit
	{
		SetVertexShader( CompileShader( vs_5_0, vsScreen() ) );
		SetGeometryShader( NULL );
		SetPixelShader( CompileShader( ps_5_0, psTex() ) );
		SetRasterizerState( CullNone );
		SetDepthStencilState( DepthDisable, 0 );
		SetBlendState( BlendDisable, float4( 0.0f, 0.0f, 0.0f, 0.0f ), 0xFFFFFFFF );
	}

	pass P2_BlitBlendOver
	{
		SetVertexShader( CompileShader( vs_5_0, vsScreen() ) );
		SetGeometryShader( NULL );
		SetPixelShader( CompileShader( ps_5_0, psTex() ) );
		SetRasterizerState( CullNone );
		SetDepthStencilState( DepthDisable, 0 );
		SetBlendState( BlendOver, float4( 0.0f, 0.0f, 0.0f, 0.0f ), 0xFFFFFFFF );
	}
}
