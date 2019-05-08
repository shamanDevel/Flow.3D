
template < eMeasure M, eTextureFilterMode F, eMeasureComputeMode C, eColorMode CM > 
__device__ inline void ms3IsoCascadeStep(
	float4& sum,
	bool& bWasInsideCoarseTube,
	bool& bWasInsideMidTube,
	const float3& world2texOffset,
	float world2texScale,
	const float3& pos,
	const float3& step,
	const float3& rayDir,
	float measureScale,
	const float3& isoValues,
	const float4& isoColor1,
	const float4& isoColor2,
	const float4& isoColor3
)
{
	float3 pos2t_e = w2t(pos);
	float3 pos2t_s = w2t(pos-step);

	bool bInsideCoarseTube = (isoValues.x < getMeasure<M,F,C>(g_texVolume3, pos2t_e, measureScale));
	bool bInsideFineTube   = false;

	// entered coarse tube?
	if (bInsideCoarseTube && !bWasInsideCoarseTube) 
	{
		pos2t_s = binarySearch<M,F,C>(g_texVolume3, pos2t_s, pos2t_e, isoValues.x, measureScale);
		float3 grad = getGradient<M,F,C>(g_texVolume3, pos2t_s);
		float4 c = shadeScaleInvariant(rayDir, grad, isoColor1);
		sum += (1.0f - sum.w) * c;
	}

	if(bInsideCoarseTube)
	{
		bool bInsideMidTube = (isoValues.y < getMeasure<M,F,C>(g_texVolume2, pos2t_e, measureScale));

		// entered mid tube?
		if (bInsideMidTube && !bWasInsideMidTube) 
		{
			pos2t_s		 = binarySearch<M,F,C>(g_texVolume2, pos2t_s, pos2t_e, isoValues.y, measureScale);
			float3 grad  = getGradient<M,F,C>(g_texVolume2, pos2t_s);
			float4 c	 = shadeScaleInvariant(rayDir, grad, isoColor2);
			sum			+= (1.0f - sum.w) * c;
		}

		// inside the mid tube, check fine tube
		if(bInsideMidTube)
		{
			bInsideFineTube = (isoValues.z < getMeasure<M,F,C>(g_texVolume1, pos2t_e, measureScale));

			// fine tube is opaque -> only need to check if we're inside
			if (bInsideFineTube) 
			{
				pos2t_s			 = binarySearch<M,F,C>(g_texVolume1, pos2t_s, pos2t_e, isoValues.z, measureScale);
				float3 grad		 = getGradient<M,F,C>(g_texVolume1, pos2t_s);
				float4 surfColor = getIsoColor<F,C,CM>(g_texVolume1, g_texVolume2, pos2t_s, isoColor3);
				surfColor.w		 = 1.0f; // enforce opaqueness
				float4 c		 = shadeIsosurface(rayDir, grad, surfColor);
				sum				+= (1.0f - sum.w) * c;
				return;
			}
		}

		// exited mid tube?
		if (!bInsideFineTube && !bInsideCoarseTube && bWasInsideCoarseTube) 
		{
			pos2t_s		= binarySearch<M,F,C>(g_texVolume2, pos2t_e, pos2t_s, isoValues.y, measureScale);
			float3 grad = getGradient<M,F,C>(g_texVolume2, pos2t_s);
			float4 c	= shadeScaleInvariant(rayDir, grad, isoColor2);
			sum		   += (1.0f - sum.w) * c;
		}

		bWasInsideMidTube = bInsideMidTube;
	}

	// exited coarse tube?
	if (!bInsideFineTube && !bInsideCoarseTube && bWasInsideCoarseTube) 
	{
		pos2t_s		= binarySearch<M,F,C>(g_texVolume3, pos2t_e, pos2t_s, isoValues.x, measureScale);
		float3 grad = getGradient<M,F,C>(g_texVolume3, pos2t_s);
		float4 c	= shadeScaleInvariant(rayDir, grad, isoColor1);
		sum		   += (1.0f - sum.w) * c;
	}

	bWasInsideCoarseTube = bInsideCoarseTube;
}
