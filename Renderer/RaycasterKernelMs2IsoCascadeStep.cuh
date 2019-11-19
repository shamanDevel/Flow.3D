
template < eMeasure M, eTextureFilterMode F, eMeasureComputeMode C, eColorMode CM > 
__device__ inline void ms2IsoCascadeStep(
	//cudaTextureObject_t texFine,		//g_texVolume1
	//cudaTextureObject_t texCoarse,	//g_texVolume2
	float4& sum,
	bool& bWasInsideCoarseTube,
	const float3& world2texOffset,
	float world2texScale,
	const float3& pos,
	const float3& step,
	const float3& rayDir,
	float gridSpacing,
	float measureScale,
	const float   isoFine,
	const float   isoCoarse,
	const float4& isoColorFine,
	const float4& isoColorCoarse
)
{
	float3 pos2t_e = w2t(pos);
	float3 pos2t_s = w2t(pos-step);

	bool bInsideCoarseTube = (isoCoarse < getMeasure<M,F,C>(g_texVolume2, pos2t_e, gridSpacing, measureScale));
	bool bInsideFineTube   = false;
	
	// entered coarse tube?
	if (bInsideCoarseTube && !bWasInsideCoarseTube) 
	{
		pos2t_s		= binarySearch<M,F,C>(g_texVolume2, pos2t_s, pos2t_e, gridSpacing, isoCoarse, measureScale);
		float3 grad = getGradient<M,F,C>(g_texVolume2, pos2t_s, gridSpacing);
		float4 c	= shadeScaleInvariant(rayDir, grad, isoColorCoarse);
		sum		   += (1.0f - sum.w) * c;
	}

	// inside the coarse tube, check fine tube
	if(bInsideCoarseTube)
	{
		bInsideFineTube = (isoFine < getMeasure<M,F,C>(g_texVolume1, pos2t_e, gridSpacing, measureScale));
		// fine tube is opaque -> only need to check if we're inside
		if (bInsideFineTube) 
		{
			pos2t_s          = binarySearch<M,F,C>(g_texVolume1, pos2t_s, pos2t_e, gridSpacing, isoFine, measureScale);
			float3 grad      = getGradient<M,F,C>(g_texVolume1, pos2t_s, gridSpacing);
			float4 surfColor = getIsoColor<F,C,CM>(g_texVolume1, g_texVolume2, pos2t_s, isoColorFine);
			surfColor.w      = 1.0f; // enforce opaqueness
			float4 c         = shadeIsosurface(rayDir, grad, surfColor);
			sum             += (1.0f - sum.w) * c;
			return;
		}
	}

	// exited coarse tube?
	if (!bInsideFineTube && !bInsideCoarseTube && bWasInsideCoarseTube) 
	{
		pos2t_s		= binarySearch<M,F,C>(g_texVolume2, pos2t_e, pos2t_s, gridSpacing, isoCoarse, measureScale);
		float3 grad = getGradient<M,F,C>(g_texVolume2, pos2t_s, gridSpacing);
		float4 c	= shadeScaleInvariant(rayDir, grad, isoColorCoarse);
		sum		   += (1.0f - sum.w) * c;
	}

	bWasInsideCoarseTube = bInsideCoarseTube;
}