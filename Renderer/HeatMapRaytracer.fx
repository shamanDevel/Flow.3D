cbuffer PerFrame
{
	float4x4 g_mWorldView;
	//inverse world-view matrix
	float4x4 g_mInvWorldView;
	//viewport: left, right, bottom, top
	float4 g_vViewport;
	//screen width, height
	uint2 g_vScreenSize;
	//near, far, far*near, far-near
	float4 g_vDepthParams;
	//bounding box sizes
	float4 g_vBoxMin;
	float4 g_vBoxMax;

	float g_fStepSizeWorld;
	float g_fDensityScale;
	float g_fAlphaScale;
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

inline float depthToLinear(float depth)
{
	return g_vDepthParams.z / (g_vDepthParams.y - depth * g_vDepthParams.w);
}

float3 transformDir(float4x4 M, float3 v)
{
	float3 r;
	r.x = dot(v, M[0].xyz);
	r.y = dot(v, M[1].xyz);
	r.z = dot(v, M[2].xyz);
	return r;
}

float3 transformPos(float4x4 M, float3 v)
{
	float3 r;
	r.x = dot(float4(v, 1.0f), M[0]);
	r.y = dot(float4(v, 1.0f), M[1]);
	r.z = dot(float4(v, 1.0f), M[2]);
	return r;
}

bool intersectBox(float3 rayPos, float3 rayDir, float3 boxMin, float3 boxMax, out float tnear, out float tfar)
{
	float3 rayDirInv = float3(1.0f, 1.0f, 1.0f) / rayDir;
	if (rayDirInv.x >= 0.0f) {
		tnear = (boxMin.x - rayPos.x) * rayDirInv.x;
		tfar = (boxMax.x - rayPos.x) * rayDirInv.x;
	}
	else {
		tnear = (boxMax.x - rayPos.x) * rayDirInv.x;
		tfar = (boxMin.x - rayPos.x) * rayDirInv.x;
	}
	if (rayDirInv.y >= 0.0f) {
		tnear = max(tnear, (boxMin.y - rayPos.y) * rayDirInv.y);
		tfar = min(tfar, (boxMax.y - rayPos.y) * rayDirInv.y);
	}
	else {
		tnear = max(tnear, (boxMax.y - rayPos.y) * rayDirInv.y);
		tfar = min(tfar, (boxMin.y - rayPos.y) * rayDirInv.y);
	}
	if (rayDirInv.z >= 0.0f) {
		tnear = max(tnear, (boxMin.z - rayPos.z) * rayDirInv.z);
		tfar = min(tfar, (boxMax.z - rayPos.z) * rayDirInv.z);
	}
	else {
		tnear = max(tnear, (boxMax.z - rayPos.z) * rayDirInv.z);
		tfar = min(tfar, (boxMin.z - rayPos.z) * rayDirInv.z);
	}
	return tnear < tfar;

	//// compute intersection of ray with all six bbox planes
	//float3 invR = make_float3(1.0f) / rayDir;
	//float3 tbot = invR * (boxMin - rayPos);
	//float3 ttop = invR * (boxMax - rayPos);

	//// re-order intersections to find smallest and largest on each axis
	//float3 tmin = fminf(ttop, tbot);
	//float3 tmax = fmaxf(ttop, tbot);

	//// find the largest tmin and the smallest tmax
	//*tnear = fmaxf(fmaxf(tmin.x, tmin.y), tmin.z);
	//*tfar  = fminf(fminf(tmax.x, tmax.y), tmax.z);

	//return *tnear < *tfar;
}

float4 psRaytrace(float4 screenPos : SV_Position, float2 texCoord : TEXCOORD) : SV_Target
{
	float x = g_vViewport.x + (g_vViewport.y - g_vViewport.x) * texCoord.x;
	float y = g_vViewport.w - (g_vViewport.w - g_vViewport.z) * texCoord.y;
	
	// calculate eye ray in world space
	float3 rayPos = float3(
		g_mInvWorldView[0].w,
		g_mInvWorldView[1].w,
		g_mInvWorldView[2].w);
	float3 rayDir = normalize(transformDir(g_mInvWorldView, float3(x, y, -1.0f)));

	float tnear, tfar;
	float3 boxSize = g_vBoxMax.xyz - g_vBoxMin.xyz;
	if (!intersectBox(rayPos, rayDir, g_vBoxMin.xyz, g_vBoxMax.xyz, tnear, tfar)) {
		return float4(0, 0, 0, 0);
	}
	tnear = max(tnear, 0.0f);

	// current position and step increment in world space
	float3 pos = rayPos + rayDir * tnear;
	float3 step = rayDir * g_fStepSizeWorld;
	float depthLinear = 0;//-transformPos(g_mWorldView, pos).z;
	float depthStepLinear = g_fStepSizeWorld;//-transformDir(g_mWorldView, step).z;

	// read depth buffer
	float depthMax = g_depthTexture.Sample(SamplerLinear, texCoord);
	float depthMaxLinear = depthToLinear(depthMax);
	// restrict depthMaxLinear to exit point depth, so we can use it as stop criterion
	//depthMaxLinear = min(depthMaxLinear, -transformPos(g_mWorldView, rayPos + rayDir * tfar).z);
	depthMaxLinear = tfar - tnear;

	if (depthLinear >= depthMaxLinear) return float4(0,0,0,0);

	// march along ray from front to back, accumulating color
	float4 sum = float4(0, 0, 0, 0);
	int numSteps = 0;
	while (depthLinear < depthMaxLinear && sum.w < 0.99)
	{
		float3 posTex = (pos - g_vBoxMin.xyz) / (boxSize);
		float density = g_heatMap.SampleLevel(SamplerLinear, posTex, 0) * g_fDensityScale;
		float4 color = g_transferFunction.SampleLevel(SamplerLinear, density, 0);
		//float4 color = float4(density, density, density, 1);
		color.a *= g_fStepSizeWorld * g_fAlphaScale;
		sum.rgb += (1 - sum.a) * color.a * color.rgb;
		sum.a += (1 - sum.a) * color.a;

		pos += step;
		depthLinear += depthStepLinear;

		numSteps++;
		if (numSteps > 1000) break;
	}

	return saturate(sum);
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