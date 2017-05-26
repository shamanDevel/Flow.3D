cbuffer PerFrame
{
	//world-view-projection matrix
	float4x4 g_mWorldViewProj;
	//viewport: left, right, bottom, top
	float4 g_vViewport;
	//screen width, height
	uint2 g_vScreenSize;
	//near, far, far*near, far-near
	float4 g_vDepthParams;

	float g_fStepSizeWorld;
	float g_fDensityScale;
}

texture3D<float> g_heatMap;
texture1D<float4> g_transferFunction;
texture2D<float> g_depthTexture;


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

SamplerState SamplerLinear
{
	Filter = MIN_MAG_MIP_LINEAR;
	AddressU = Clamp;
	AddressV = Clamp;
};


void vsScreen(uint index : SV_VertexID, out float4 outPos : SV_Position, out float2 outTexCoord : TEXCOORD)
{
	switch (index) {
	case 0:
		outPos = float4(-1.0f, -1.0f, 0.5f, 1.0f);
		outTexCoord = float2(0, 1);
		break;
	case 1:
		outPos = float4(+1.0f, -1.0f, 0.5f, 1.0f);
		outTexCoord = float2(1, 1);
		break;
	case 2:
		outPos = float4(-1.0f, +1.0f, 0.5f, 1.0f);
		outTexCoord = float2(0, 0);
		break;
	case 3:
	default:
		outPos = float4(+1.0f, +1.0f, 0.5f, 1.0f);
		outTexCoord = float2(1, 0);
		break;
	}
}

float4 psRaytrace(float4 pos : SV_Position, float2 texCoord : TEXCOORD) : SV_Target
{
	float depth = g_depthTexture.Sample(SamplerLinear, texCoord);
	
	return float4(0, 0, 0, 0);
}


technique11 tRaytrace
{
	pass P0_Blit
	{
		SetVertexShader(CompileShader(vs_5_0, vsScreen()));
		SetGeometryShader(NULL);
		SetPixelShader(CompileShader(ps_5_0, psRaytrace()));
		SetRasterizerState(CullNone);
		SetDepthStencilState(DepthDisable, 0);
		SetBlendState(BlendDisable, float4(0.0f, 0.0f, 0.0f, 0.0f), 0xFFFFFFFF);
	}
}