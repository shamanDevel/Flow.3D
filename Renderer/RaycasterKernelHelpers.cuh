#include "Coords.cuh"
#include "MatrixMath.cuh"
#include "Measures.cuh"
#include "TextureFilter.cuh"



/*************************************************************************************************************************************
** Iso Rendering Helper: Binary Search
*************************************************************************************************************************************/

#define BIN_SEARCH_STEPS (10)

template <eMeasure M, eTextureFilterMode F, eMeasureComputeMode C>
__device__ inline float3 binarySearch(texture<float4, cudaTextureType3D, cudaReadModeElementType> tex, float3 texCoordOutside, float3 texCoordInside, float3 h, float iso, float measureScale)
{
	float3 texCoordMiddle = 0.5f * (texCoordOutside + texCoordInside);
	for(uint k = 0; k < BIN_SEARCH_STEPS; k++) 
	{
		float v = getMeasure<M, F, C>(tex, texCoordMiddle, h, measureScale);
		if (v >= iso)
			texCoordInside = texCoordMiddle;
		else
			texCoordOutside = texCoordMiddle;
		texCoordMiddle = 0.5f * (texCoordOutside + texCoordInside);
	}
	return texCoordMiddle;
}

template <eMeasureSource source, eTextureFilterMode F, eMeasureComputeMode C>
__device__ inline float3 binarySearch(eMeasure measure, texture<float4, cudaTextureType3D, cudaReadModeElementType> tex, float3 texCoordOutside, float3 texCoordInside, float3 h, float iso, float measureScale)
{
	float3 texCoordMiddle = 0.5f * (texCoordOutside + texCoordInside);
	for(uint k = 0; k < BIN_SEARCH_STEPS; k++) 
	{
		float v = getMeasure<source,F,C>(measure, tex, texCoordMiddle, h, measureScale);
		if (v >= iso)
			texCoordInside = texCoordMiddle;
		else
			texCoordOutside = texCoordMiddle;
		texCoordMiddle = 0.5f * (texCoordOutside + texCoordInside);
	}
	return texCoordMiddle;
}




/*************************************************************************************************************************************
** Iso Rendering Helper: Gradient Computation
*************************************************************************************************************************************/

template <eMeasure M, eTextureFilterMode F, eMeasureComputeMode C>
__device__ inline float3 getGradient(texture<float4, cudaTextureType3D, cudaReadModeElementType> tex, const float3& pos, const float3& h)
{
	const float off = 1.0f;
	float3 grad;
	grad.x = getMeasure<M,F,C>(tex, make_float3(pos.x+off, pos.y, pos.z), h, 1.0f) - getMeasure<M,F,C>(tex, make_float3(pos.x-off, pos.y, pos.z), h, 1.0f);
	grad.y = getMeasure<M,F,C>(tex, make_float3(pos.x, pos.y+off, pos.z), h, 1.0f) - getMeasure<M,F,C>(tex, make_float3(pos.x, pos.y-off, pos.z), h, 1.0f);
	grad.z = getMeasure<M,F,C>(tex, make_float3(pos.x, pos.y, pos.z+off), h, 1.0f) - getMeasure<M,F,C>(tex, make_float3(pos.x, pos.y, pos.z-off), h, 1.0f);
	return normalize(grad);
}

template <eMeasureSource source, eTextureFilterMode F, eMeasureComputeMode C>
__device__ inline float3 getGradient(eMeasure measure, texture<float4, cudaTextureType3D, cudaReadModeElementType> tex, const float3& pos, const float3& h)
{
	const float off = 1.0f;
	float3 grad;
	grad.x = getMeasure<source,F,C>(measure, tex, make_float3(pos.x+off, pos.y, pos.z), h, 1.0f) - getMeasure<source,F,C>(measure, tex, make_float3(pos.x-off, pos.y, pos.z), h, 1.0f);
	grad.y = getMeasure<source,F,C>(measure, tex, make_float3(pos.x, pos.y+off, pos.z), h, 1.0f) - getMeasure<source,F,C>(measure, tex, make_float3(pos.x, pos.y-off, pos.z), h, 1.0f);
	grad.z = getMeasure<source,F,C>(measure, tex, make_float3(pos.x, pos.y, pos.z+off), h, 1.0f) - getMeasure<source,F,C>(measure, tex, make_float3(pos.x, pos.y, pos.z-off), h, 1.0f);
	return normalize(grad);
}





/*************************************************************************************************************************************
** Iso/SIVR Rendering Helper: Shading
*************************************************************************************************************************************/

__device__ inline float4 shadeIsosurface(const float3& rayDir, const float3& gradient, const float4& color) 
{
	float diffFactor = saturate(abs(dot(rayDir, gradient)));
	float specFactor = powf(diffFactor, 32.0f); //headlight
	float3 diffColor = color.w * (0.2f + 0.6f * diffFactor) * make_float3(color);
	float3 specColor = color.w * (0.3f * specFactor) * make_float3(1.0f);
	return make_float4(diffColor + specColor, color.w);
}


__device__ inline float heavy(const float &x)
{
	return (x<=0.0f ? 0.0f : 1.0f);
}

__device__ inline float4 shadeScaleInvariant(const float3 &rayDir, const float3& gradient, const float4 &color) 
{
	float4 Cd = color;
    float4 Cs = make_float4(saturate(Cd.x+0.3f), saturate(Cd.y+0.3f), saturate(Cd.z+0.3f), Cd.w);
	float3 V = rayDir;
	float VN = dot(V,gradient);

	float Tp = heavy( VN*VN); // actually VN * LN, but headlight so V == L
	float Tm = heavy(-VN*VN); // -> so this is always 0. optimize?
	float Id = saturate(VN);
	float Is = pow(Id, 32.0f);
	const float Ka=0.1f; //0.3f;
	const float Kd=0.7f; //0.5f;
	const float Ks=0.8f-color.w/2;

	float alpha;
	if (abs(VN)==0.0f)	alpha = 0.0f;
	else				alpha = saturate((Cd.w<1.0f ? 1.0f - exp(log(1.0f-Cd.w) /abs(VN)) : 1.0f));

	float3 Ca = Ka*alpha*make_float3(Cd.x, Cd.y, Cd.z);
	Cd *= Id*Kd;
	Cs *= Is*Ks;
	Tp *= alpha;
	Tm *= alpha*(1.0f-alpha);

	float4 res = make_float4(Ca + (Tp+Tm)*make_float3(Cd.x+Cs.x, Cd.y+Cs.y, Cd.z+Cs.z), alpha);
	res.x = saturate(res.x);
	res.y = saturate(res.y);
	res.z = saturate(res.z);

	return res;
}



/*************************************************************************************************************************************
** DVR Rendering Helper: TF and Lighting
*************************************************************************************************************************************/

__device__ inline void computeLighting(float4 &color, float nDotV)
{
	float diffuse   = abs(nDotV);
	float specular  = powf(diffuse, 32.0f) * 0.5f; //headlight
	diffuse = (0.2f + 0.6f * diffuse);

	color.x = saturate(diffuse*color.x + specular);
	color.y = saturate(diffuse*color.y + specular);
	color.z = saturate(diffuse*color.z + specular);
}


__device__ inline float4 evaluateTf(float value, float transferOffset, float transferScale, float tfAlphaScale) 
{
	float x = (value+transferOffset)*transferScale;

	float4 color = tex1D(g_texTransferFunction, x);
	color.w *= tfAlphaScale;

	return color;
}


__device__ inline void convertEmissionAbsorptionToColorOpacity(float4& val, float stepFactor)
{
	// emission scaling including self-absorption within this segment:
	//   tau * integrate(exp(-tau*(l-s))) for s=0 to l
	//   where l = segment length (stepFactor) and tau = absorption coefficient (color.w)
	// approximate by evaluating only at s = 0.5 * l:
	//   tau * l * exp(-tau * l * 0.5)
	float emissionFactor = val.w * stepFactor;
	emissionFactor *= exp(-val.w * stepFactor * 0.5f); // for speed, comment this line - makes little difference for reasonably small step sizes
	val.x *= emissionFactor;
	val.y *= emissionFactor;
	val.z *= emissionFactor;
	// opacity is exponential in absorption coeff and length
	val.w = 1.0f - exp(-val.w * stepFactor);
}



/*************************************************************************************************************************************
** DVR Coloring functions
*************************************************************************************************************************************/

template <eMeasure M, eTextureFilterMode F, eMeasureComputeMode C, bool lighting>
struct getColor_Impl
{
	__device__ static inline float4 exec(texture<float4, cudaTextureType3D, cudaReadModeElementType> tex, const float3& pos, const float3& dir, float h, float stepFactor, float transferOffset, float transferScale, float tfAlphaScale, float measureScale)
	{
		// same default code for all "normal" scalar measures
		// specializations for vector-valued measures and lambda2 (special coloring/alpha) are below
		float measure = getMeasure<M,F,C>(tex, pos, h, measureScale);
		float4 color = evaluateTf(measure, transferOffset, transferScale, tfAlphaScale);
		if(lighting)
		{
			float3 gradient = getGradient<M,F,C>(tex, pos, h);
			float nDotV = dot(dir, gradient);
			computeLighting(color, nDotV);
		}
		convertEmissionAbsorptionToColorOpacity(color, stepFactor);
		return color;
	}
};

template <eTextureFilterMode F, eMeasureComputeMode C, bool lighting>
struct getColor_Impl<MEASURE_LAMBDA2, F, C, lighting>
{
	__device__ static inline float4 exec(texture<float4, cudaTextureType3D, cudaReadModeElementType> tex, const float3& pos, const float3& dir, const float3& h, float stepFactor, float transferOffset, float transferScale, float tfAlphaScale, float measureScale)
	{
		float lambda2 = getMeasure<MEASURE_LAMBDA2,F,C>(tex, pos, h, measureScale);
		float sgnL2	  = getSign(lambda2);
		lambda2 = sgnL2 * fmaxf(transferOffset + sgnL2 * lambda2, 0.0f) * transferScale;
		float asqrtL2 = sqrtf(fabsf(lambda2));
		float4 color = make_float4(-sgnL2*asqrtL2, sgnL2*asqrtL2, 0.0f, fabsf(lambda2));
		//float4 color = make_float4(-sgnL2*asqrtL2, sgnL2*asqrtL2, 0.0f, fmaxf(-lambda2, 0.0f)); // Lambda 2 vortex core (l2 <= 0)
		if(lighting)
		{
			float3 gradient = getGradient<MEASURE_LAMBDA2,F,C>(tex, pos, h);
			float nDotV = dot(dir, gradient);
			computeLighting(color, nDotV);
		}
		convertEmissionAbsorptionToColorOpacity(color, stepFactor);
		return color;
	}
};

template <eMeasure M, eTextureFilterMode F, eMeasureComputeMode C, bool lighting>
__device__ inline float4 getColor(texture<float4, cudaTextureType3D, cudaReadModeElementType> tex, const float3& pos, const float3& dir, const float3& h, float stepFactor, float transferOffset, float transferScale, float tfAlphaScale, float measureScale)
{
    return getColor_Impl<M,F,C,lighting>::exec(tex, pos, dir, h, stepFactor, transferOffset, transferScale, tfAlphaScale, measureScale);
}


template <eMeasureSource source, eTextureFilterMode F, eMeasureComputeMode C, bool lighting>
struct getColor_Impl2
{
	__device__ static inline float4 exec(eMeasure measure, texture<float4, cudaTextureType3D, cudaReadModeElementType> tex, const float3& pos, const float3& dir, const float3& h, float stepFactor, float transferOffset, float transferScale, float tfAlphaScale, float measureScale)
	{
		// same default code for all "normal" scalar measures
		float val = getMeasure<source,F,C>(measure, tex, pos, h, measureScale);
		float4 color = evaluateTf(val, transferOffset, transferScale, tfAlphaScale);
		if(lighting)
		{
			float3 gradient = getGradient<source,F,C>(measure, tex, pos, h);
			float nDotV = dot(dir, gradient);
			computeLighting(color, nDotV);
		}
		convertEmissionAbsorptionToColorOpacity(color, stepFactor);
		return color;
	}
};

template <eMeasureSource measureSource, eTextureFilterMode F, eMeasureComputeMode C, bool lighting>
__device__ inline float4 getColor(eMeasure measure, texture<float4, cudaTextureType3D, cudaReadModeElementType> tex, const float3& pos, const float3& dir, const float3& h, float stepFactor, float transferOffset, float transferScale, float tfAlphaScale, float measureScale)
{
    return getColor_Impl2<measureSource,F,C,lighting>::exec(measure, tex, pos, dir, h, stepFactor, transferOffset, transferScale, tfAlphaScale, measureScale);
}


/*************************************************************************************************************************************
** Multi-scale iso surface coloring
*************************************************************************************************************************************/

//default is constant iso color
template <eTextureFilterMode F, eMeasureComputeMode C, eColorMode CM>
struct getIsoColor_Impl
{
    __device__ static inline float4 exec(texture<float4, cudaTextureType3D, cudaReadModeElementType> tex1, texture<float4, cudaTextureType3D, cudaReadModeElementType> tex2, float3 pos, const float4 &color)
	{
		return color;
	}
};

//colorize vorticity alignment, i.e. by the angle between the vorticity in fields of different scale
template <eTextureFilterMode F>
struct getIsoColor_Impl<F,MEASURE_COMPUTE_ONTHEFLY,COLOR_MODE_VORTICITY_ALIGNMENT>
{
    __device__ static inline float4 exec(texture<float4, cudaTextureType3D, cudaReadModeElementType> tex1, texture<float4, cudaTextureType3D, cudaReadModeElementType> tex2, float3 pos, const float4 &color)
	{
		float3 w1 = normalize(getVorticity<F>(tex1, pos, 1.0f)); //FIXME actually need grid spacing here...
		float3 w2 = normalize(getVorticity<F>(tex2, pos, 1.0f));
		float  dp = dot(w1, w2);

		//parallel		= green
		//perpendicular = white 
		//antiparallel	= red
		if( dp >= 0 )
		{
			return lerp( make_float4(1, 1, 0, color.w), make_float4(0, 1, 0, color.w),  dp ); 
		}
		else
		{
			return lerp( make_float4(1, 1, 0, color.w), make_float4(1, 0, 0, color.w), -dp ); 
		}
	}
};

template <eTextureFilterMode F, eMeasureComputeMode C, eColorMode CM>
__device__ inline float4 getIsoColor(texture<float4, cudaTextureType3D, cudaReadModeElementType> tex1, texture<float4, cudaTextureType3D, cudaReadModeElementType> tex2, float3 pos, const float4 &color)
{
    return getIsoColor_Impl<F,C,CM>::exec(tex1, tex2, pos, color);
}



/*************************************************************************************************************************************
** Ray-Box intersection
*************************************************************************************************************************************/

__device__ inline bool intersectBox(float3 rayPos, float3 rayDir, float3 boxMin, float3 boxMax, float *tnear, float *tfar)
{
	float3 rayDirInv = make_float3(1.0f) / rayDir;
	if(rayDirInv.x >= 0.0f) {
		*tnear = (boxMin.x - rayPos.x) * rayDirInv.x;
		*tfar  = (boxMax.x - rayPos.x) * rayDirInv.x;
	} else {
		*tnear = (boxMax.x - rayPos.x) * rayDirInv.x;
		*tfar  = (boxMin.x - rayPos.x) * rayDirInv.x;
	}
	if(rayDirInv.y >= 0.0f) {
		*tnear = fmaxf(*tnear, (boxMin.y - rayPos.y) * rayDirInv.y);
		*tfar  = fminf(*tfar,  (boxMax.y - rayPos.y) * rayDirInv.y);
	} else {
		*tnear = fmaxf(*tnear, (boxMax.y - rayPos.y) * rayDirInv.y);
		*tfar  = fminf(*tfar,  (boxMin.y - rayPos.y) * rayDirInv.y);
	}
	if(rayDirInv.z >= 0.0f) {
		*tnear = fmaxf(*tnear, (boxMin.z - rayPos.z) * rayDirInv.z);
		*tfar  = fminf(*tfar,  (boxMax.z - rayPos.z) * rayDirInv.z);
	} else {
		*tnear = fmaxf(*tnear, (boxMax.z - rayPos.z) * rayDirInv.z);
		*tfar  = fminf(*tfar,  (boxMin.z - rayPos.z) * rayDirInv.z);
	}
	return *tnear < *tfar;

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


/*************************************************************************************************************************************
** Get ray pos/dir from view matrix and pixel coords
*************************************************************************************************************************************/

__device__ inline float3 getRayPos(const float3x4& viewInv)
{
	return make_float3(viewInv.m[0].w, viewInv.m[1].w, viewInv.m[2].w);
}

__device__ inline float3 getRayDir(const float3x4& viewInv, uint x, uint y)
{
	//float u = (2.0f * ((float(x) + 0.5f) / float(c_projParams.imageWidth)) - 1.0f) * c_projParams.tanHalfFovy * c_projParams.aspectRatio + c_projParams.eyeOffset;
	//float v = (1.0f - 2.0f * ((float(y) + 0.5f) / float(c_projParams.imageHeight))) * c_projParams.tanHalfFovy;
	float u = c_projParams.left   + (c_projParams.right - c_projParams.left)   * ((float(x) + 0.5f) / float(c_projParams.imageWidth)) + c_projParams.eyeOffset;
	float v = c_projParams.top    - (c_projParams.top   - c_projParams.bottom) * ((float(y) + 0.5f) / float(c_projParams.imageHeight));
	return normalize(transformDir(viewInv, make_float3(u, v, -1.0f)));
}



/*************************************************************************************************************************************
** Depth conversion
*************************************************************************************************************************************/

__device__ inline float depthToLinear(float depth)
{
	return c_projParams.depthParams.z / (c_projParams.depthParams.y - depth * c_projParams.depthParams.w);
}



/*************************************************************************************************************************************
** float4 <-> byte4 conversion
*************************************************************************************************************************************/

__device__ inline uchar4 rgbaFloatToUChar(float4 rgba)
{
    rgba.x = __saturatef(rgba.x);
    rgba.y = __saturatef(rgba.y);
    rgba.z = __saturatef(rgba.z);
    rgba.w = __saturatef(rgba.w);
    return make_uchar4(uchar(rgba.x * 255.0f), uchar(rgba.y * 255.0f), uchar(rgba.z * 255.0f), uchar(rgba.w * 255.0f));
}

__device__ inline float4 rgbaUCharToFloat(uchar4 rgba)
{
    return make_float4(float(rgba.x) / 255.0f, float(rgba.y) / 255.0f, float(rgba.z) / 255.0f, float(rgba.w) / 255.0f);
}
