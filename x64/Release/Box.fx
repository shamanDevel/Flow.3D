cbuffer PerFrame
{
	float4x4 g_mWorldView;
	float4x4 g_mProj;
	float4x4 g_mWorldViewProj;
	float3   g_vBoxMin;
	float3   g_vBoxSize;
	float4   g_vColor;
	float    g_fTubeRadius;

	float3   g_vCamPos;
	float3   g_vCamRight;
	float3   g_vLightPos;

	float3   g_vBrickSize;
	uint3    g_vLineCount;
}


RasterizerState CullFront
{
	CullMode = Front;
};

RasterizerState CullBack
{
	CullMode = Back;
};

RasterizerState CullNone
{
	CullMode = None;
};

DepthStencilState DepthDefault
{
};

DepthStencilState DepthEqual
{
	DepthEnable    = true;
	DepthWriteMask = 0;
	DepthFunc      = EQUAL;
};

BlendState BlendDisable
{
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


void vsSimpleTransform(float3 pos : POSITION, out float4 outPos : SV_Position)
{
	outPos = mul(g_mWorldViewProj, float4(g_vBoxMin + pos * g_vBoxSize, 1.0));
}


struct TubeGSIn
{
	float3 pos : POSITION;
};

void vsTube(float3 pos : POSITION, out TubeGSIn output)
{
	output.pos = g_vBoxMin + pos * g_vBoxSize;
}

void vsBrickLineTube(uint index : SV_VertexID, out TubeGSIn output)
{
	uint lineID = index / 2;
	uint lineVertex = index % 2;

	uint lineCountX = g_vLineCount.y * g_vLineCount.z;
	uint lineCountY = g_vLineCount.x * g_vLineCount.z;

	float3 dir;
	uint3 linePos;
	if(lineID < lineCountX) {
		dir = float3(1, 0, 0);
		linePos.x = 0;
		linePos.y = lineID % g_vLineCount.y;
		linePos.z = lineID / g_vLineCount.y;
	} else if(lineID < lineCountX + lineCountY) {
		lineID -= lineCountX;
		dir = float3(0, 1, 0);
		linePos.x = lineID % g_vLineCount.x;
		linePos.y = 0;
		linePos.z = lineID / g_vLineCount.x;
	} else {
		lineID -= lineCountX + lineCountY;
		dir = float3(0, 0, 1);
		linePos.x = lineID % g_vLineCount.x;
		linePos.y = lineID / g_vLineCount.x;
		linePos.z = 0;
	}

	output.pos = g_vBoxMin + linePos * g_vBrickSize;
	if(lineVertex > 0) output.pos += dir * g_vBoxSize;
	output.pos = min(output.pos, g_vBoxMin + g_vBoxSize);
}

struct TubePSIn
{
	float4 pos        : SV_Position;
	float3 posWorld   : POS_WORLD;
	float3 tubeCenter : TUBE_CENTER;
	float3 normal     : NORMAL;
};

#define TUBE_SEGMENT_COUNT 32
// maxvertexcount = 2 * (TUBE_SEGMENT_COUNT + 1)
[maxvertexcount(66)]
void gsExtrudeTube(in line TubeGSIn input[2], inout TriangleStream<TubePSIn> stream)
{
	float3 tangent = normalize(input[1].pos - input[0].pos);

	float3 normal = cross(float3(1, 0, 0), tangent);
	if(length(normal) < 0.01) normal = cross(float3(0, 1, 0), tangent);
	normal = normalize(normal);

	float3 binormal = cross(tangent, normal);


	TubePSIn output;
	
	output.normal = normal;

	output.tubeCenter = input[1].pos;
	output.posWorld = output.tubeCenter + g_fTubeRadius * output.normal;
	output.pos = mul(g_mWorldViewProj, float4(output.posWorld, 1.0));
	stream.Append(output);

	output.tubeCenter = input[0].pos;
	output.posWorld = output.tubeCenter + g_fTubeRadius * output.normal;
	output.pos = mul(g_mWorldViewProj, float4(output.posWorld, 1.0));
	stream.Append(output);

	for(int i = 1; i < TUBE_SEGMENT_COUNT; i++) {
		float t = float(i) / float(TUBE_SEGMENT_COUNT);
		float angle = radians(t * 360.0);
		float s,c;
		sincos(angle, s, c);

		output.normal = c * normal + s * binormal;

		output.tubeCenter = input[1].pos;
		output.posWorld = output.tubeCenter + g_fTubeRadius * output.normal;
		output.pos = mul(g_mWorldViewProj, float4(output.posWorld, 1.0));
		stream.Append(output);

		output.tubeCenter = input[0].pos;
		output.posWorld = output.tubeCenter + g_fTubeRadius * output.normal;
		output.pos = mul(g_mWorldViewProj, float4(output.posWorld, 1.0));
		stream.Append(output);
	}

	output.normal = normal;

	output.tubeCenter = input[1].pos;
	output.posWorld = output.tubeCenter + g_fTubeRadius * output.normal;
	output.pos = mul(g_mWorldViewProj, float4(output.posWorld, 1.0));
	stream.Append(output);

	output.tubeCenter = input[0].pos;
	output.posWorld = output.tubeCenter + g_fTubeRadius * output.normal;
	output.pos = mul(g_mWorldViewProj, float4(output.posWorld, 1.0));
	stream.Append(output);
}

struct SpherePSIn
{
	float4 pos      : SV_Position;
	float3 posWorld : POS_WORLD;
	float3 center   : CENTER;
};

[maxvertexcount(4)]
void gsExtrudeSphereQuad(in point TubeGSIn input[1], inout TriangleStream<SpherePSIn> stream)
{
	float3 view = normalize(input[0].pos - g_vCamPos);
	float3 up = normalize(cross(view, g_vCamRight));
	float3 right = cross(up, view);

	SpherePSIn output;

	output.center = input[0].pos;

	float3 centerFront = output.center - g_fTubeRadius * view;
	float3 vec0 = g_fTubeRadius * up;
	float3 vec1 = g_fTubeRadius * right;

	output.posWorld = centerFront + vec0 + vec1;
	output.pos = mul(g_mWorldViewProj, float4(output.posWorld, 1.0));
	stream.Append(output);

	output.posWorld = centerFront + vec0 - vec1;
	output.pos = mul(g_mWorldViewProj, float4(output.posWorld, 1.0));
	stream.Append(output);

	output.posWorld = centerFront - vec0 + vec1;
	output.pos = mul(g_mWorldViewProj, float4(output.posWorld, 1.0));
	stream.Append(output);

	output.posWorld = centerFront - vec0 - vec1;
	output.pos = mul(g_mWorldViewProj, float4(output.posWorld, 1.0));
	stream.Append(output);
}


float4 psSolidColor() : SV_Target
{
	return g_vColor;
}

float4 psTube(TubePSIn input) : SV_Target
{
	float3 lightDir = normalize(g_vLightPos - input.posWorld);
	float diffuse = saturate(dot(lightDir, normalize(input.normal)));
	return float4((0.3 + 0.7 * diffuse) * g_vColor.rgb, g_vColor.a);
}

float4 psSphere(SpherePSIn input, out float depth : SV_Depth) : SV_Target
{
	float3 dir = normalize(input.posWorld - g_vCamPos);
	float3 c2p = input.posWorld - input.center;
	float a = dot(dir, dir);
	float b = 2.0 * dot(dir, c2p);
	float c = dot(c2p, c2p) - g_fTubeRadius * g_fTubeRadius;
	float d = b * b - 4.0 * a * c;
	if(d <= 0.0) discard;

	float t = (-b - sqrt(d)) / (2.0 * a);
	float3 intersection = input.posWorld + dir * t;

	float4 projected = mul(g_mWorldViewProj, float4(intersection, 1.0));
	depth = projected.z / projected.w;

	float3 normal = (intersection - input.center) / g_fTubeRadius;
	float3 lightDir = normalize(g_vLightPos - intersection);
	float diffuse = saturate(dot(lightDir, normal));
	return float4((0.3 + 0.7 * diffuse) * g_vColor.rgb, g_vColor.a);
}


technique11 tBox
{
	pass P0_Lines
	{
		SetVertexShader( CompileShader( vs_5_0, vsSimpleTransform() ) );
		SetGeometryShader( NULL );
		SetPixelShader( CompileShader( ps_5_0, psSolidColor() ) );
		SetRasterizerState( CullNone );
		SetDepthStencilState( DepthDefault, 0 );
		SetBlendState( BlendDisable, float4( 0.0f, 0.0f, 0.0f, 0.0f ), 0xFFFFFFFF );
	}

	pass P1_LinesBehind
	{
		SetVertexShader( CompileShader( vs_5_0, vsSimpleTransform() ) );
		SetGeometryShader( NULL );
		SetPixelShader( CompileShader( ps_5_0, psSolidColor() ) );
		SetRasterizerState( CullNone );
		SetDepthStencilState( DepthEqual, 0 );
		SetBlendState( BlendBehind, float4( 0.0f, 0.0f, 0.0f, 0.0f ), 0xFFFFFFFF );
	}

	pass P2_Tubes
	{
		SetVertexShader( CompileShader( vs_5_0, vsTube() ) );
		SetGeometryShader( CompileShader( gs_5_0, gsExtrudeTube() ) );
		SetPixelShader( CompileShader( ps_5_0, psTube() ) );
		SetRasterizerState( CullNone );
		SetDepthStencilState( DepthDefault, 0 );
		SetBlendState( BlendDisable, float4( 0.0f, 0.0f, 0.0f, 0.0f ), 0xFFFFFFFF );
	}

	pass P3_TubesBehind
	{
		SetVertexShader( CompileShader( vs_5_0, vsTube() ) );
		SetGeometryShader( CompileShader( gs_5_0, gsExtrudeTube() ) );
		SetPixelShader( CompileShader( ps_5_0, psTube() ) );
		SetRasterizerState( CullNone );
		SetDepthStencilState( DepthEqual, 0 );
		SetBlendState( BlendBehind, float4( 0.0f, 0.0f, 0.0f, 0.0f ), 0xFFFFFFFF );
	}

	pass P4_CornerSpheres
	{
		SetVertexShader( CompileShader( vs_5_0, vsTube() ) );
		SetGeometryShader( CompileShader( gs_5_0, gsExtrudeSphereQuad() ) );
		SetPixelShader( CompileShader( ps_5_0, psSphere() ) );
		SetRasterizerState( CullNone );
		SetDepthStencilState( DepthDefault, 0 );
		SetBlendState( BlendDisable, float4( 0.0f, 0.0f, 0.0f, 0.0f ), 0xFFFFFFFF );
	}

	pass P5_CornerSpheresBehind
	{
		SetVertexShader( CompileShader( vs_5_0, vsTube() ) );
		SetGeometryShader( CompileShader( gs_5_0, gsExtrudeSphereQuad() ) );
		SetPixelShader( CompileShader( ps_5_0, psSphere() ) );
		SetRasterizerState( CullNone );
		SetDepthStencilState( DepthEqual, 0 );
		SetBlendState( BlendBehind, float4( 0.0f, 0.0f, 0.0f, 0.0f ), 0xFFFFFFFF );
	}

	pass P6_BrickLines
	{
		SetVertexShader( CompileShader( vs_5_0, vsBrickLineTube() ) );
		SetGeometryShader( NULL );
		SetPixelShader( CompileShader( ps_5_0, psSolidColor() ) );
		SetRasterizerState( CullNone );
		SetDepthStencilState( DepthDefault, 0 );
		SetBlendState( BlendDisable, float4( 0.0f, 0.0f, 0.0f, 0.0f ), 0xFFFFFFFF );
	}

	pass P7_BrickLinesBehind
	{
		SetVertexShader( CompileShader( vs_5_0, vsBrickLineTube() ) );
		SetGeometryShader( NULL );
		SetPixelShader( CompileShader( ps_5_0, psSolidColor() ) );
		SetRasterizerState( CullNone );
		SetDepthStencilState( DepthEqual, 0 );
		SetBlendState( BlendBehind, float4( 0.0f, 0.0f, 0.0f, 0.0f ), 0xFFFFFFFF );
	}

	pass P8_BrickTubes
	{
		SetVertexShader( CompileShader( vs_5_0, vsBrickLineTube() ) );
		SetGeometryShader( CompileShader( gs_5_0, gsExtrudeTube() ) );
		SetPixelShader( CompileShader( ps_5_0, psTube() ) );
		SetRasterizerState( CullNone );
		SetDepthStencilState( DepthDefault, 0 );
		SetBlendState( BlendDisable, float4( 0.0f, 0.0f, 0.0f, 0.0f ), 0xFFFFFFFF );
	}

	pass P9_BrickTubesBehind
	{
		SetVertexShader( CompileShader( vs_5_0, vsBrickLineTube() ) );
		SetGeometryShader( CompileShader( gs_5_0, gsExtrudeTube() ) );
		SetPixelShader( CompileShader( ps_5_0, psTube() ) );
		SetRasterizerState( CullNone );
		SetDepthStencilState( DepthEqual, 0 );
		SetBlendState( BlendBehind, float4( 0.0f, 0.0f, 0.0f, 0.0f ), 0xFFFFFFFF );
	}
}
