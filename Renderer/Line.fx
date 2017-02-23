cbuffer PerFrame
{
	//float4x4 g_mWorldView;
	//float4x4 g_mProj;
	float4x4 g_mWorldViewProj;

	float3   g_vLightPos;

	float    g_fRibbonHalfWidth;
	float    g_fTubeRadius;

	bool     g_bTubeRadiusFromVelocity;
	float    g_fReferenceVelocity;

	bool     g_bColorByTime;
	float4   g_vColor0;
	float4   g_vColor1;
	float    g_fTimeMin;
	float    g_fTimeMax;

	bool     g_bTimeStripes;
	float    g_fTimeStripeLength;
}

cbuffer Balls
{
	float3   g_vBoxMin;
	float3   g_vBoxSize;
	float3   g_vCamPos;
	float3   g_vCamRight;
	float    g_fBallRadius = 0.011718750051;
}

texture1D<float4> g_texColors;


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

DepthStencilState DepthDisable
{
	DepthEnable    = false;
	DepthWriteMask = 0;
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
	BlendEnable[0]    = true;
	SrcBlend[0]       = INV_DEST_ALPHA;
	SrcBlendAlpha[0]  = INV_DEST_ALPHA;
	DestBlend[0]      = ONE;
	DestBlendAlpha[0] = ONE;
};


// for Particles
// [
DepthStencilState dsNoDepth
{
	DepthEnable = FALSE;
};

RasterizerState SpriteRS
{
	CullMode = None;
};

BlendState AlphaBlending
{
	AlphaToCoverageEnable = FALSE;
	BlendEnable[0] = TRUE;
	SrcBlend = SRC_ALPHA;
	DestBlend = INV_SRC_ALPHA;
	BlendOp = ADD;
	SrcBlendAlpha = ONE;
	DestBlendAlpha = ONE;
	BlendOpAlpha = ADD;
	RenderTargetWriteMask[0] = 0x0F;
};

BlendState AdditionBlending
{
	AlphaToCoverageEnable = FALSE;
	BlendEnable[0] = TRUE;
	SrcBlend = SRC_COLOR;
	DestBlend = DEST_COLOR;
	BlendOp = ADD;
	SrcBlendAlpha = ONE;
	DestBlendAlpha = ONE;
	BlendOpAlpha = ADD;
	RenderTargetWriteMask[0] = 0x0F;
};
// ]


struct LineVertex
{
	float3   pos     : POSITION;
	float    time    : TIME;
	float3   normal  : NORMAL;
	float3   vel     : VELOCITY;
	uint     lineID  : LINE_ID;
	//float3x3 jac     : JACOBIAN;
};

struct BallVertex
{
	float3   pos     : POSITION;
	//float    radius  : RADIUS;
};


float4 getColor(uint lineID, float time)
{
	float4 color;
	if(g_bColorByTime) {
		float timeFactor = g_bColorByTime ? saturate((time - g_fTimeMin) / (g_fTimeMax - g_fTimeMin)) : 0.0;
		color = (1.0 - timeFactor) * g_vColor0 + timeFactor * g_vColor1;
	} else {
		color = g_texColors.Load(int2(lineID % 1024, 0));
	}

	if(g_bTimeStripes) {
		float segment = floor(time / g_fTimeStripeLength);
		float evenOdd = (segment % 2.0) * 2.0 - 1.0;
		color.rgb *= 1.0 + 0.1 * evenOdd;
		color.rgb += 0.25 * evenOdd;
	}

	return color;
}

float3 getVorticity(float3x3 jacobian)
{
	return float3(jacobian[2][1] - jacobian[1][2], jacobian[0][2] - jacobian[2][0], jacobian[1][0] - jacobian[0][1]);
}



void vsTransform(LineVertex input, out float4 outPos : SV_Position, out float outTime : TIME)
{
	outPos = mul(g_mWorldViewProj, float4(input.pos, 1.0));
	outTime = input.time;
}

float4 psSolidColor(float4 pos : SV_Position, float time : TIME) : SV_Target
{
	return getColor(0, time);
}



struct RibbonGSIn
{
	float3 pos  : POSITION;
	float  time : TIME;
	float3 vel  : VELOCITY;
	float3 vort : VORTICITY;
};

void vsRibbon(LineVertex input, out RibbonGSIn output)
{
	output.pos  = input.pos;
	output.time = input.time;
	output.vel  = input.vel;
	output.vort = float3(1.0, 0.0, 0.0); //getVorticity(input.jac);
}

struct RibbonPSIn
{
	float4 pos    : SV_Position;
	float3 posWorld   : POS_WORLD;
	float3 normal : NORMAL;
	float  time   : TIME;
};

[maxvertexcount(4)]
void gsExtrudeRibbon(in line RibbonGSIn input[2], inout TriangleStream<RibbonPSIn> stream)
{
	// displace in direction of vorticity, but orthogonalize wrt velocity
	float3 front0 = normalize(input[0].vel);
	float3 front1 = normalize(input[1].vel);

	float3 right0 = normalize(input[0].vort);
	float3 right1 = normalize(input[1].vort);

	float3 normal0 = cross(front0, right0);
	float3 normal1 = cross(front1, right1);

	float3 displace0 = g_fRibbonHalfWidth * cross(normal0, front0);
	float3 displace1 = g_fRibbonHalfWidth * cross(normal1, front1);


	RibbonPSIn output;

	output.normal = normalize(cross(input[0].vel, displace0));
	output.time = input[0].time;
	output.posWorld = input[0].pos - displace0;
	output.pos = mul(g_mWorldViewProj, float4(output.posWorld, 1.0));
	stream.Append(output);
	output.posWorld = input[0].pos + displace0;
	output.pos = mul(g_mWorldViewProj, float4(output.posWorld, 1.0));
	stream.Append(output);

	output.normal = normalize(cross(input[1].vel, displace1));
	output.time = input[1].time;
	output.posWorld = input[1].pos - displace1;
	output.pos = mul(g_mWorldViewProj, float4(output.posWorld, 1.0));
	stream.Append(output);
	output.posWorld = input[1].pos + displace1;
	output.pos = mul(g_mWorldViewProj, float4(output.posWorld, 1.0));
	stream.Append(output);
}

float4 psRibbon(float4 pos : SV_Position, RibbonPSIn input) : SV_Target
{
	float3 lightDir = normalize(g_vLightPos - input.posWorld);
	float4 color = getColor(0, input.time);
	float diffuse = abs(dot(lightDir, normalize(input.normal)));
	return float4(diffuse * color.rgb, color.a);
}



struct TubeGSIn
{
	float3 pos    : POSITION;
	float  time   : TIME;
	float3 normal : NORMAL;
	float3 vel    : VELOCITY;
	float3 vort   : VORTICITY;
	nointerpolation uint lineID : LINE_ID;
};

void vsTube(LineVertex input, out TubeGSIn output)
{
	output.pos    = input.pos;
	output.time   = input.time;
	output.normal = input.normal;
	output.vel    = input.vel;
	output.vort   = float3(1.0, 0.0, 0.0); //getVorticity(input.jac);
	output.lineID = input.lineID;
}

struct TubePSIn
{
	float4 pos        : SV_Position;
	float3 posWorld   : POS_WORLD;
	float3 tubeCenter : TUBE_CENTER;
	float  time       : TIME;
	float3 normal     : NORMAL;
	nointerpolation uint lineID : LINE_ID;
};

#define TUBE_SEGMENT_COUNT 16
//[maxvertexcount(2*(TUBE_SEGMENT_COUNT+1)]
[maxvertexcount(34)]
void gsExtrudeTube(in line TubeGSIn input[2], inout TriangleStream<TubePSIn> stream)
{
	float3 tangent0 = normalize(input[0].vel);
	float3 tangent1 = normalize(input[1].vel);

	float3 normal0 = input[0].normal;
	float3 normal1 = input[1].normal;

	float3 binormal0 = cross(tangent0, normal0);
	float3 binormal1 = cross(tangent1, normal1);


	float radius0 = g_fTubeRadius;
	float radius1 = g_fTubeRadius;
	if(g_bTubeRadiusFromVelocity) {
		radius0 *= clamp(sqrt(g_fReferenceVelocity / length(input[0].vel)), 0.2, 3.0);
		radius1 *= clamp(sqrt(g_fReferenceVelocity / length(input[1].vel)), 0.2, 3.0);
	}

	TubePSIn output;
	output.lineID = input[0].lineID;

	output.tubeCenter = input[1].pos;
	output.normal = normal1;
	output.posWorld = output.tubeCenter + radius1 * output.normal;
	output.pos = mul(g_mWorldViewProj, float4(output.posWorld, 1.0));
	output.time = input[1].time;
	stream.Append(output);

	output.tubeCenter = input[0].pos;
	output.normal = normal0;
	output.posWorld = output.tubeCenter + radius0 * output.normal;
	output.pos = mul(g_mWorldViewProj, float4(output.posWorld, 1.0));
	output.time = input[0].time;
	stream.Append(output);

	for(int i = 1; i < TUBE_SEGMENT_COUNT; i++) {
		//TODO sin/cos lookup table?
		float t = float(i) / float(TUBE_SEGMENT_COUNT);
		float angle = radians(t * 360.0);
		float s,c;
		sincos(angle, s, c);

		output.tubeCenter = input[1].pos;
		output.normal = c * normal1 + s * binormal1;
		output.posWorld = output.tubeCenter + radius1 * output.normal;
		output.pos = mul(g_mWorldViewProj, float4(output.posWorld, 1.0));
		output.time = input[1].time;
		stream.Append(output);

		output.tubeCenter = input[0].pos;
		output.normal = c * normal0 + s * binormal0;
		output.posWorld = output.tubeCenter + radius0 * output.normal;
		output.pos = mul(g_mWorldViewProj, float4(output.posWorld, 1.0));
		output.time = input[0].time;
		stream.Append(output);
	}

	output.tubeCenter = input[1].pos;
	output.normal = normal1;
	output.posWorld = output.tubeCenter + radius1 * output.normal;
	output.pos = mul(g_mWorldViewProj, float4(output.posWorld, 1.0));
	output.time = input[1].time;
	stream.Append(output);

	output.tubeCenter = input[0].pos;
	output.normal = normal0;
	output.posWorld = output.tubeCenter + radius0 * output.normal;
	output.pos = mul(g_mWorldViewProj, float4(output.posWorld, 1.0));
	output.time = input[0].time;
	stream.Append(output);
}

float4 psTube(TubePSIn input) : SV_Target
{
	float3 lightDir = normalize(g_vLightPos - input.posWorld);
	float4 color = getColor(input.lineID, input.time);
	float diffuse = saturate(dot(lightDir, normalize(input.normal)));
	return float4(diffuse * color.rgb, color.a);
}


void vsSphere(BallVertex input, out float3 outPos : POSITION)
{
	outPos = g_vBoxMin + input.pos * g_vBoxSize;
}

struct SphereGSIn
{
	float3 pos      : POSITION;
};

struct SpherePSIn
{
	float4 pos      : SV_Position;
	float3 posWorld : POS_WORLD;
	float3 center   : CENTER;
};

[maxvertexcount(4)]
void gsExtrudeSphereQuad(in point SphereGSIn input[1], inout TriangleStream<SpherePSIn> stream)
{
	float3 view = normalize(input[0].pos - g_vCamPos);
	float3 up = normalize(cross(view, g_vCamRight));
	float3 right = cross(up, view);

	SpherePSIn output;

	output.center = input[0].pos;

	float3 centerFront = output.center - g_fBallRadius * view;
	float3 vec0 = g_fBallRadius * up;
	float3 vec1 = g_fBallRadius * right;

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

float4 psSphere(SpherePSIn input, out float depth : SV_Depth) : SV_Target
{
	float3 dir = normalize(input.posWorld - g_vCamPos);
	float3 c2p = input.posWorld - input.center;
	float a = dot(dir, dir);
	float b = 2.0 * dot(dir, c2p);
	float c = dot(c2p, c2p) - g_fBallRadius * g_fBallRadius;
	float d = b * b - 4.0 * a * c;
	if(d <= 0.0) discard;

	float t = (-b - sqrt(d)) / (2.0 * a);
	float3 intersection = input.posWorld + dir * t;

	float4 projected = mul(g_mWorldViewProj, float4(intersection, 1.0));
	depth = projected.z / projected.w;

	float3 normal = (intersection - input.center) / g_fBallRadius;
	float3 lightDir = normalize(g_vLightPos - intersection);
	float diffuse = saturate(dot(lightDir, normal));
	return float4((0.3 + 0.7 * diffuse) * g_vColor0.rgb, g_vColor0.a);
}



struct ParticleGSIn
{
	float3 pos    : POSITION;
	float  time : TIME;
	float3 normal : NORMAL;
	float3 vel    : VELOCITY;
	float3 vort   : VORTICITY;
	nointerpolation uint lineID : LINE_ID;
};

struct ParticlePSIn
{
	float4 pos        : SV_Position;
	float2 tex    : TEXTURE;
	float3 posWorld   : POS_WORLD;
	float3 tubeCenter : TUBE_CENTER;
	float  time : TIME;
	float3 normal     : NORMAL;
	nointerpolation uint lineID : LINE_ID;
};

void vsParticle(LineVertex input, out ParticleGSIn output)
{
	output.pos = input.pos;
	output.time = input.time;
	output.normal = input.normal;
	output.vel = input.vel;
	output.vort = float3(1.0, 0.0, 0.0); //getVorticity(input.jac);
	output.lineID = input.lineID;
}

[maxvertexcount(6)]
void gsParticle(in point ParticleGSIn input[1], inout TriangleStream<ParticlePSIn> outStream)
{
	ParticlePSIn o;
	o.time = input[0].time;
	o.tubeCenter = input[0].pos;
	o.posWorld = o.tubeCenter;
	o.normal = input[0].normal;
	o.lineID = input[0].lineID;
	float4 pos = mul(g_mWorldViewProj, float4(input[0].pos, 1.0));

	float size = sqrt(1 - saturate((o.time - g_fTimeMin) / (g_fTimeMax - g_fTimeMin)))
		* g_fTubeRadius * 5;
	float spriteSizeW = size;
	float spriteSizeH = size; // * (gScreenResolution.x / gScreenResolution.y);

	o.pos = float4(pos.x - spriteSizeW, pos.y + spriteSizeH, pos.z, pos.w);
	o.tex = float2(0, 1);
	outStream.Append(o);
	o.pos = float4(pos.x - spriteSizeW, pos.y - spriteSizeH, pos.z, pos.w);
	o.tex = float2(0, 0);
	outStream.Append(o);
	o.pos = float4(pos.x + spriteSizeW, pos.y + spriteSizeH, pos.z, pos.w);
	o.tex = float2(1, 1);
	outStream.Append(o);
	outStream.RestartStrip();

	o.pos = float4(pos.x + spriteSizeW, pos.y + spriteSizeH, pos.z, pos.w);
	o.tex = float2(1, 1);
	outStream.Append(o);
	o.pos = float4(pos.x - spriteSizeW, pos.y - spriteSizeH, pos.z, pos.w);
	o.tex = float2(0, 0);
	outStream.Append(o);
	o.pos = float4(pos.x + spriteSizeW, pos.y - spriteSizeH, pos.z, pos.w);
	o.tex = float2(1, 0);
	outStream.Append(o);
	outStream.RestartStrip();
}

float4 psParticle(ParticlePSIn input) : SV_Target
{
	float4 color = getColor(input.lineID, input.time);

	float dist = length(input.tex.xy - float2 (0.5f, 0.5f)) * 2;
	float alpha = 1; //smoothstep(0, 0.3, 1 - dist);
	if (dist > 0.5) discard;

	return float4(color.rgb, color.a * alpha);
}

technique11 tLine
{
	pass P0_Line
	{
		SetVertexShader( CompileShader( vs_5_0, vsTransform() ) );
		SetGeometryShader( NULL );
		SetPixelShader( CompileShader( ps_5_0, psSolidColor() ) );
		SetRasterizerState( CullNone );
		SetDepthStencilState( DepthDefault, 0 );
		SetBlendState( BlendDisable, float4( 0.0f, 0.0f, 0.0f, 0.0f ), 0xFFFFFFFF );
	}

	pass P1_LineBehind
	{
		SetVertexShader( CompileShader( vs_5_0, vsTransform() ) );
		SetGeometryShader( NULL );
		SetPixelShader( CompileShader( ps_5_0, psSolidColor() ) );
		SetRasterizerState( CullNone );
		SetDepthStencilState( DepthEqual, 0 );
		SetBlendState( BlendBehind, float4( 0.0f, 0.0f, 0.0f, 0.0f ), 0xFFFFFFFF );
	}

	pass P2_Ribbon
	{
		SetVertexShader( CompileShader( vs_5_0, vsRibbon() ) );
		SetGeometryShader( CompileShader( gs_5_0, gsExtrudeRibbon() ) );
		SetPixelShader( CompileShader( ps_5_0, psRibbon() ) );
		SetRasterizerState( CullNone );
		SetDepthStencilState( DepthDefault, 0 );
		SetBlendState( BlendDisable, float4( 0.0f, 0.0f, 0.0f, 0.0f ), 0xFFFFFFFF );
	}

	pass P3_RibbonBehind
	{
		SetVertexShader( CompileShader( vs_5_0, vsRibbon() ) );
		SetGeometryShader( CompileShader( gs_5_0, gsExtrudeRibbon() ) );
		SetPixelShader( CompileShader( ps_5_0, psRibbon() ) );
		SetRasterizerState( CullNone );
		SetDepthStencilState( DepthEqual, 0 );
		SetBlendState( BlendBehind, float4( 0.0f, 0.0f, 0.0f, 0.0f ), 0xFFFFFFFF );
	}

	pass P4_Tube
	{
		SetVertexShader( CompileShader( vs_5_0, vsTube() ) );
		SetGeometryShader( CompileShader( gs_5_0, gsExtrudeTube() ) );
		SetPixelShader( CompileShader( ps_5_0, psTube() ) );
		SetRasterizerState( CullNone );
		SetDepthStencilState( DepthDefault, 0 );
		SetBlendState( BlendDisable, float4( 0.0f, 0.0f, 0.0f, 0.0f ), 0xFFFFFFFF );
	}

	pass P5_TubeBehind
	{
		SetVertexShader( CompileShader( vs_5_0, vsTube() ) );
		SetGeometryShader( CompileShader( gs_5_0, gsExtrudeTube() ) );
		SetPixelShader( CompileShader( ps_5_0, psTube() ) );
		SetRasterizerState( CullNone );
		SetDepthStencilState( DepthEqual, 0 );
		SetBlendState( BlendBehind, float4( 0.0f, 0.0f, 0.0f, 0.0f ), 0xFFFFFFFF );
	}

	pass P6_Particle
	{
		SetVertexShader(CompileShader(vs_5_0, vsParticle()));
		SetGeometryShader(CompileShader(gs_5_0, gsParticle()));
		SetPixelShader(CompileShader(ps_5_0, psParticle()));
		SetRasterizerState(SpriteRS);
		SetDepthStencilState(dsNoDepth, 0);
		SetBlendState(AdditionBlending, float4(0.0f, 0.0f, 0.0f, 0.0f), 0xFFFFFFFF);
	}
}


technique11 tBall
{
	pass P0_Ball
	{
		SetVertexShader( CompileShader( vs_5_0, vsSphere() ) );
		SetGeometryShader( CompileShader( gs_5_0, gsExtrudeSphereQuad() ) );
		SetPixelShader( CompileShader( ps_5_0, psSphere() ) );
		SetRasterizerState( CullNone );
		SetDepthStencilState( DepthDefault, 0 );
		SetBlendState( BlendDisable, float4( 0.0f, 0.0f, 0.0f, 0.0f ), 0xFFFFFFFF );
	}
}
